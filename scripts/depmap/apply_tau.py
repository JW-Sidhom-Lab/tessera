"""Apply the MSK-CHORD CRC predictive biomarker tau_hat to DepMap cell lines.

Pipeline:
  1. Load DepMap TESSERA features (depmap_latent_features_panel.pkl, the
     MSK-IMPACT505 panel-restricted inference output from
     get_latent_features.py).
  2. Load MSK-CHORD RobustScalers + feature column order from
     ../predictive_bm/cache/msk_chord_scalers.pkl.
  3. Apply scalers to DepMap embeddings (per-token), then mean+max pool per
     cell line and concat with log-count features into a per-cell-line
     genomic fingerprint in the same column order MSK-CHORD's beta_eff
     expects.
  4. Load beta_eff, const_eff, tau0 from
     ../predictive_bm/attribution_analysis/crc_signatures/attribution_matrix.pkl.
  5. Compute tau_hat = beta_eff @ x + const_eff per cell line.
  6. Join with cell-line metadata (lineage, drug-response AUC) and write
     results/tau/depmap_tau.tsv.

Usage:
    cd scripts/depmap
    python3 apply_tau.py
"""
from __future__ import annotations
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PBM  = ROOT / ".." / "predictive_bm"
OUT  = ROOT / "results" / "tau"
OUT.mkdir(parents=True, exist_ok=True)

DEPMAP_FEATURES = ROOT / "depmap_latent_features_panel.pkl"
SCALERS_BUNDLE  = PBM  / "cache" / "msk_chord_scalers.pkl"
ATTRIB_BUNDLE   = PBM  / "attribution_analysis" / "crc_signatures" / "attribution_matrix.pkl"
METADATA        = ROOT / ".." / "data" / "depmap" / "metadata.csv"


def build_fingerprint(snv_emb_z: np.ndarray, snv_pids: np.ndarray,
                       cna_emb_z: np.ndarray, cna_pids: np.ndarray,
                       feature_order: list[str]) -> tuple[np.ndarray, list[str]]:
    """Replicate build_patient_features but on pre-scaled embeddings."""
    snv_df = pd.DataFrame(snv_emb_z, index=snv_pids)
    cna_df = pd.DataFrame(cna_emb_z, index=cna_pids)

    snv_mean = snv_df.groupby(level=0).mean()
    snv_max  = snv_df.groupby(level=0).max()
    cna_mean = cna_df.groupby(level=0).mean()
    cna_max  = cna_df.groupby(level=0).max()
    snv_n    = snv_df.groupby(level=0).size()
    cna_n    = cna_df.groupby(level=0).size()

    pids = sorted(set(snv_mean.index) & set(cna_mean.index))
    print(f"    cell lines with both modalities: {len(pids)}")

    snv_mean = snv_mean.loc[pids].add_prefix("snv_mean_")
    snv_max  = snv_max .loc[pids].add_prefix("snv_max_")
    cna_mean = cna_mean.loc[pids].add_prefix("cna_mean_")
    cna_max  = cna_max .loc[pids].add_prefix("cna_max_")

    out = pd.concat([snv_mean, snv_max, cna_mean, cna_max], axis=1)
    out["tmb_log1p"] = np.log1p(snv_n.reindex(pids).fillna(0).values)
    out["cna_log1p"] = np.log1p(cna_n.reindex(pids).fillna(0).values)

    # Re-order to MSK-CHORD's feature_order; verify all expected cols present
    missing = set(feature_order) - set(out.columns)
    if missing:
        raise ValueError(f"missing {len(missing)} expected feature cols, "
                          f"e.g. {sorted(missing)[:5]}")
    out = out[feature_order]
    return out.values.astype(float), pids


def main():
    print("=" * 70)
    print("Apply MSK-CHORD CRC tau_hat to DepMap cell lines")
    print("=" * 70)

    print("\n[1/5] Loading DepMap TESSERA features...")
    with open(DEPMAP_FEATURES, "rb") as fh:
        bundle = pickle.load(fh)
    snv_emb = bundle["variant_features"]
    cna_emb = bundle["cna_features"]
    snv_meta = bundle["data_snv"]
    cna_meta = bundle["data_cna"]
    print(f"  SNV embeddings: {snv_emb.shape}")
    print(f"  CNA embeddings: {cna_emb.shape}")

    snv_pids = snv_meta["Tumor_Sample_Barcode"].values
    cna_pids = cna_meta["Tumor_Sample_Barcode"].values

    print("\n[2/5] Loading MSK-CHORD scalers...")
    with open(SCALERS_BUNDLE, "rb") as fh:
        sc = pickle.load(fh)
    if snv_emb.shape[1] != sc["snv_emb_dim"]:
        raise ValueError(f"SNV emb dim mismatch: depmap {snv_emb.shape[1]} "
                          f"vs MSK-CHORD {sc['snv_emb_dim']}: re-run feature "
                          f"extraction with the same model")
    if cna_emb.shape[1] != sc["cna_emb_dim"]:
        raise ValueError(f"CNA emb dim mismatch: depmap {cna_emb.shape[1]} "
                          f"vs MSK-CHORD {sc['cna_emb_dim']}")
    snv_z = sc["snv_scaler"].transform(snv_emb)
    cna_z = sc["cna_scaler"].transform(cna_emb)
    print(f"  applied SNV RobustScaler (dim={sc['snv_emb_dim']})")
    print(f"  applied CNA RobustScaler (dim={sc['cna_emb_dim']})")

    print("\n[3/5] Building per-cell-line genomic fingerprint...")
    X, pids = build_fingerprint(snv_z, snv_pids, cna_z, cna_pids,
                                  sc["feature_order"])
    print(f"  fingerprint shape: {X.shape}  (expected n x {sc['n_features']})")
    if X.shape[1] != sc["n_features"]:
        raise ValueError(f"fingerprint dim {X.shape[1]} != {sc['n_features']}")

    print("\n[4/5] Loading beta_eff, const_eff, tau0 from attribution bundle...")
    with open(ATTRIB_BUNDLE, "rb") as fh:
        attrib = pickle.load(fh)
    beta_eff  = attrib["beta_eff"]
    const_eff = float(attrib["const_eff"])
    tau0      = float(attrib["tau0"])
    print(f"  beta_eff shape: {beta_eff.shape}; const_eff={const_eff:.3f}; tau0={tau0:.3f}")
    if beta_eff.shape[0] != X.shape[1]:
        raise ValueError(f"beta_eff dim {beta_eff.shape[0]} != fingerprint dim {X.shape[1]}")

    tau = X @ beta_eff + const_eff
    above = (tau > tau0).astype(int)
    print(f"  tau_hat summary: median {np.median(tau):+.2f}, "
          f"range [{tau.min():+.2f}, {tau.max():+.2f}]")
    print(f"  above tau0={tau0:.2f}: {int(above.sum())} / {len(tau)} cell lines")

    print("\n[5/5] Joining with metadata + drug response and writing TSV...")
    out_df = pd.DataFrame({
        "ModelID":  pids,
        "tau_hat":  tau,
        "above_tau0": above,
    })
    if METADATA.exists():
        meta = pd.read_csv(METADATA)
        out_df = out_df.merge(meta, on="ModelID", how="left")
    out_path = OUT / "depmap_tau.tsv"
    out_df.to_csv(out_path, sep="\t", index=False)
    print(f"  -> {out_path}  ({len(out_df)} rows, {out_df.shape[1]} columns)")


if __name__ == "__main__":
    main()
