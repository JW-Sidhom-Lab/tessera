"""Apply a TCGA-trained TESSERA CNA NoLOH model to MSK-CHORD panel-sequenced data.

Loads the panel-filtered MSK-CHORD CNA segments from
``scripts/data/msk_chord/`` (or the unfiltered raw segments), optionally
remaps each segment's ``Segment_Mean`` onto the TCGA training-set
distribution to reconcile the WES-vs-panel scale shift (manuscript
default: quantile remap), runs the trained model in inference mode, and
saves predicted vs actual segment-mean values plus summary metrics
(MSE, MAE, R^2, Pearson) to ``cna_loss_panel_filtered/``.

The cross-platform Pearson correlation reported in Figure 2 d is read
back from these pkl files by ``plot_cna_reconstruction.py``.

Inputs
------
``../data/msk_chord/cna.csv`` or ``cna_panel_filtered.csv``
    MSK-CHORD CNA segments (from scripts/data/msk_chord/create_cna.py).
``../tcga_pancan_cna_noloh/models/<MODEL>/``
    Trained TESSERA CNA NoLOH model (from scripts/tcga_pancan_cna_noloh/fit_model.py).
``../data/tcga/{train,valid}_data_cna.csv``
    Used to compute (and cache) the TCGA Segment_Mean reference
    distribution for normalisation. Cache files
    ``../data/tcga/cna_stats.json`` and ``../data/tcga/cna_sorted.npy``
    are written on first use.

Output
------
``cna_loss_panel_filtered/<MODEL>_msk_chord_loss_cna.pkl``

Usage
-----
    python get_cna_loss_acc.py                                          # defaults
    CNA_ATTN_BLOCKS=2 python get_cna_loss_acc.py                        # specific attention config
    NORMALIZATION_MODE=linear python get_cna_loss_acc.py                # linear rescale instead of quantile
    DATA_SOURCE=raw python get_cna_loss_acc.py                          # full raw segments (no panel filter)
"""

import json
import os
import pickle

import numpy as np
import pandas as pd

from tessera.model import TESSERA

import model_config

# ============================================================================
# Configuration (edit or set env vars to override)
# ============================================================================
CNA_ATTN_BLOCKS = int(os.environ.get("CNA_ATTN_BLOCKS", "2"))
MODEL_NAME = os.environ.get(
    "MODEL_NAME", f"{model_config.model_name}_attn_{CNA_ATTN_BLOCKS}"
)
MODELS_DIR = os.environ.get("MODELS_DIR", "../tcga_pancan_cna_noloh/models")

# Which MSK-CHORD CNA file to score.
# 'panel_filtered' = trimmed to MSK-IMPACT panel gene regions (manuscript default)
# 'raw'            = full segments
DATA_SOURCE = os.environ.get("DATA_SOURCE", "panel_filtered")
DATA_PATHS = {
    "raw": "../data/msk_chord/cna.csv",
    "panel_filtered": "../data/msk_chord/cna_panel_filtered.csv",
}

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "cna_loss_panel_filtered")

# Segment-mean normalisation mode applied to MSK-CHORD before inference:
#   'none'     — no transform (raw panel values)
#   'linear'   — match TCGA global mean/std
#   'quantile' — rank-map MSK onto the TCGA Segment_Mean distribution (matches
#                full shape; manuscript default)
NORMALIZATION_MODE = os.environ.get("NORMALIZATION_MODE", "quantile")
TCGA_CNA_STATS_PATH = os.environ.get(
    "TCGA_CNA_STATS_PATH", "../data/tcga/cna_stats.json"
)
TCGA_CNA_SORTED_PATH = os.environ.get(
    "TCGA_CNA_SORTED_PATH", "../data/tcga/cna_sorted.npy"
)
TCGA_CNA_DATA_PATHS = [
    os.environ.get("TCGA_TRAIN_CNA", "../data/tcga/train_data_cna.csv"),
    os.environ.get("TCGA_VALID_CNA", "../data/tcga/valid_data_cna.csv"),
]

# Optional: subsample N samples (None = use all)
N_SAMPLES = int(os.environ["N_SAMPLES"]) if os.environ.get("N_SAMPLES") else None


# ============================================================================
# TCGA Segment_Mean reference (lazy cache)
# ============================================================================

def _load_tcga_segment_mean(data_paths):
    """Concatenate Segment_Mean across the given TCGA CNA CSVs."""
    vals = []
    for p in data_paths:
        df = pd.read_csv(p, usecols=["Segment_Mean"])
        vals.append(df["Segment_Mean"].astype(float).values)
    return np.concatenate(vals)


def get_tcga_cna_stats(stats_path, data_paths):
    """Return (mean, std) of the TCGA training Segment_Mean distribution.

    Loads from cache if present; otherwise computes and writes the cache.
    """
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            s = json.load(f)
        return float(s["mean"]), float(s["std"])
    print(f"Computing TCGA CNA stats (cache miss: {stats_path})")
    arr = _load_tcga_segment_mean(data_paths)
    mean, std = float(arr.mean()), float(arr.std())
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump({"mean": mean, "std": std, "n_segments": int(arr.size)},
                  f, indent=2)
    print(f"  Cached to {stats_path}: mean={mean:.4f} std={std:.4f} (n={arr.size:,})")
    return mean, std


def get_tcga_cna_sorted(sorted_path, data_paths):
    """Return sorted TCGA Segment_Mean array used as the quantile reference.

    Loads from .npy cache if present; otherwise computes and writes the cache.
    """
    if os.path.exists(sorted_path):
        return np.load(sorted_path)
    print(f"Computing sorted TCGA CNA array (cache miss: {sorted_path})")
    arr = np.sort(_load_tcga_segment_mean(data_paths)).astype(np.float32)
    os.makedirs(os.path.dirname(sorted_path), exist_ok=True)
    np.save(sorted_path, arr)
    print(f"  Cached to {sorted_path}: n={arr.size:,}")
    return arr


def quantile_normalize_to_tcga(vals, tcga_sorted):
    """Rank-map ``vals`` onto the TCGA Segment_Mean distribution.

    Ties are averaged (not broken to consecutive ranks) so duplicate MSK
    values (e.g. many zeros) collapse onto a single TCGA quantile instead
    of spreading over a range.
    """
    from scipy.stats import rankdata
    n = len(vals)
    ranks = rankdata(vals, method="average")
    q = (ranks - 0.5) / n
    tcga_q = np.linspace(0.0, 1.0, len(tcga_sorted))
    return np.interp(q, tcga_q, tcga_sorted)


# ============================================================================
# Load MSK-CHORD CNA
# ============================================================================

if NORMALIZATION_MODE not in ("none", "linear", "quantile"):
    raise ValueError(f"NORMALIZATION_MODE must be 'none' | 'linear' | 'quantile'; "
                     f"got {NORMALIZATION_MODE!r}")
if DATA_SOURCE not in DATA_PATHS:
    raise ValueError(f"DATA_SOURCE must be 'raw' | 'panel_filtered'; got {DATA_SOURCE!r}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
data_path = DATA_PATHS[DATA_SOURCE]
print(f"Loading MSK-CHORD CNA from {data_path}")
msk_data = pd.read_csv(data_path)
print(f"  {len(msk_data):,} segments, {msk_data['Tumor_Sample_Barcode'].nunique():,} samples")

if N_SAMPLES is not None:
    np.random.seed(42)
    samples = msk_data["Tumor_Sample_Barcode"].unique()
    if N_SAMPLES < len(samples):
        keep = np.random.choice(samples, size=N_SAMPLES, replace=False)
        msk_data = msk_data[msk_data["Tumor_Sample_Barcode"].isin(keep)]
        print(f"  Subsampled to {N_SAMPLES} samples: {len(msk_data):,} segments")

sample_ids = msk_data["Tumor_Sample_Barcode"].values
chr_vals = msk_data["Chromosome"].values
start_vals = msk_data["Start"].astype(int).values
end_vals = msk_data["End"].astype(int).values
segment_mean_vals = msk_data["Segment_Mean"].astype(float).values


# ============================================================================
# Normalise MSK Segment_Mean onto the TCGA distribution
# ============================================================================

if NORMALIZATION_MODE != "none":
    msk_mean_before = float(segment_mean_vals.mean())
    msk_std_before = float(segment_mean_vals.std())

    if NORMALIZATION_MODE == "linear":
        tcga_mean, tcga_std = get_tcga_cna_stats(
            TCGA_CNA_STATS_PATH, TCGA_CNA_DATA_PATHS,
        )
        print(f"Linear rescale to TCGA mean/std: target mean={tcga_mean:.4f}, std={tcga_std:.4f}")
        if msk_std_before == 0:
            raise ValueError("MSK Segment_Mean std is 0; cannot rescale.")
        segment_mean_vals = (
            (segment_mean_vals - msk_mean_before) / msk_std_before * tcga_std
            + tcga_mean
        )
    elif NORMALIZATION_MODE == "quantile":
        tcga_sorted = get_tcga_cna_sorted(
            TCGA_CNA_SORTED_PATH, TCGA_CNA_DATA_PATHS,
        )
        print(f"Quantile remap to TCGA distribution (n={tcga_sorted.size:,}, "
              f"range=[{tcga_sorted[0]:.3f}, {tcga_sorted[-1]:.3f}])")
        segment_mean_vals = quantile_normalize_to_tcga(
            segment_mean_vals, tcga_sorted,
        ).astype(np.float32)

    print(f"  MSK before: mean={msk_mean_before:.4f} std={msk_std_before:.4f}")
    print(f"  MSK after:  mean={segment_mean_vals.mean():.4f} "
          f"std={segment_mean_vals.std():.4f}")


# ============================================================================
# Build model and run inference
# ============================================================================

model_dir = os.path.join(MODELS_DIR, MODEL_NAME)
print(f"Loading model: {model_dir}")
model = TESSERA(
    model_dir=model_dir,
    use_distributed=model_config.use_distributed,
    jit_compile=False,
    mixed_precision=False,
)

model.create_sample_dataset(
    cna_sample_ids=sample_ids,
    cna_chromosomes=chr_vals,
    cna_starts=start_vals,
    cna_ends=end_vals,
    cna_segment_means=segment_mean_vals,
    cna_lohs=None,                        # NoLOH model
    cna_subsample=model_config.cna_subsample,
    batch_size=model_config.batch_size,
    name="msk_chord_dataset",
    is_training=False,
    fixed_bag_size=True,
    z_score_cna=model_config.z_score_cna,
    z_score_clip=model_config.z_score_clip,
)

preds, actual = model.get_cna_predictions(
    "msk_chord_dataset", return_true_values=True, return_loh=False,
)


# ============================================================================
# Metrics + save
# ============================================================================

mse = float(np.mean((preds - actual) ** 2))
mae = float(np.mean(np.abs(preds - actual)))
ss_res = float(np.sum((actual - preds) ** 2))
ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
r2 = 1 - (ss_res / ss_tot)
correlation = float(np.corrcoef(actual, preds)[0, 1])

print(f"MSE={mse:.4f}  MAE={mae:.4f}  R2={r2:.4f}  Pearson={correlation:.4f}")

out_path = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_msk_chord_loss_cna.pkl")
with open(out_path, "wb") as f:
    pickle.dump({
        "predictions": preds,
        "actual": actual,
        "metrics": {"mse": mse, "mae": mae, "r2": r2, "correlation": correlation},
        "msk_data": msk_data,
        "model_name": MODEL_NAME,
        "data_source": DATA_SOURCE,
        "normalization_mode": NORMALIZATION_MODE,
        "data_info": {
            "n_segments": len(msk_data),
            "n_samples": int(msk_data["Tumor_Sample_Barcode"].nunique()),
        },
    }, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"Wrote {out_path}")
