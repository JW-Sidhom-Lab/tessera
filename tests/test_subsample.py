"""Tests for the per-sample subsamplers
(tessera.data.preprocessing.subsample_snv / subsample_cna).

Covers the public behaviour (caps, recurrence preservation, reproducibility,
optional LoH) and pins the package functions to the manuscript data-prep helpers
in ``scripts/data/tcga/utils.py`` so the two cannot silently drift.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tessera.data.preprocessing import subsample_snv, subsample_cna

# Import the manuscript helpers (not an installed package) for the equivalence pin.
_TCGA_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "data" / "tcga"
sys.path.insert(0, str(_TCGA_SCRIPTS))
from utils import subsample_variants_with_common, subsample_cna_priority  # noqa: E402

MUT_ID_COLS = ["Chromosome", "Start_Position", "Reference_Allele", "Tumor_Seq_Allele2"]


def _make_snv_cohort(n_samples=12, seed=0):
    """Synthetic SNV cohort with a few recurrent loci and variable burdens."""
    rng = np.random.RandomState(seed)
    recurrent = [("7", 140453136, "A", "T"), ("12", 25398284, "C", "T"),
                 ("17", 7577120, "G", "A")]  # BRAF/KRAS/TP53-like hotspots
    rows = []
    for s in range(n_samples):
        sid = f"S{s:03d}"
        n = int(rng.randint(3, 60)) if s % 4 else int(rng.randint(1500, 2600))  # some hypermutators
        for locus in recurrent:
            if rng.rand() < 0.5:
                rows.append((sid, *locus))
        for _ in range(n):
            rows.append((sid, str(rng.randint(1, 22)), int(rng.randint(1, 2_000_000)),
                         rng.choice(list("ACGT")), rng.choice(list("ACGT"))))
    df = pd.DataFrame(rows, columns=["Tumor_Sample_Barcode", *MUT_ID_COLS])
    df["vaf"] = rng.rand(len(df)).round(3)
    return df


def _make_cna_cohort(n_samples=12, seed=1):
    rng = np.random.RandomState(seed)
    rows = []
    for s in range(n_samples):
        sid = f"S{s:03d}"
        n = int(rng.randint(50, 1500))
        for _ in range(n):
            rows.append((sid, str(rng.randint(1, 22)), rng.randn() * 1.2,
                         int(rng.rand() < 0.3)))
    return pd.DataFrame(rows, columns=["Tumor_Sample_Barcode", "Chromosome",
                                       "Segment_Mean", "LOH"])


def _canon(df):
    return df.sort_values(list(df.columns)).reset_index(drop=True)


# --- behaviour --------------------------------------------------------------
def test_snv_caps_per_sample():
    df = _make_snv_cohort()
    out = subsample_snv(df, max_variants=1000, min_recurrence=2)
    per = out.groupby("Tumor_Sample_Barcode").size()
    # No sample exceeds the cap (no sample here has >1000 recurrent variants).
    assert per.max() <= 1000
    # Unchanged samples keep all their variants.
    small = df.groupby("Tumor_Sample_Barcode").filter(lambda g: len(g) <= 1000)
    assert len(subsample_snv(small, max_variants=1000)) == len(small)


def test_snv_preserves_recurrent():
    df = _make_snv_cohort()
    key = df[MUT_ID_COLS].astype(str).agg(":".join, axis=1)
    counts = key.value_counts()
    common = set(counts[counts >= 3].index)
    out = subsample_snv(df, max_variants=50, min_recurrence=3)
    out_key = out[MUT_ID_COLS].astype(str).agg(":".join, axis=1)
    # Every (sample, recurrent-locus) pair in the input survives the cap.
    in_pairs = set(zip(df["Tumor_Sample_Barcode"], key))
    out_pairs = set(zip(out["Tumor_Sample_Barcode"], out_key))
    for sid, k in in_pairs:
        if k in common:
            assert (sid, k) in out_pairs


def test_snv_reproducible():
    df = _make_snv_cohort()
    a = subsample_snv(df, max_variants=100, random_state=42)
    b = subsample_snv(df, max_variants=100, random_state=42)
    pd.testing.assert_frame_equal(a, b)


def test_snv_autobuilds_mut_id_without_locus_or_id_raises():
    df = _make_snv_cohort()[["Tumor_Sample_Barcode", "vaf"]]
    with pytest.raises(KeyError):
        subsample_snv(df, max_variants=10)


def test_cna_caps_and_keeps_largest():
    df = _make_cna_cohort()
    out = subsample_cna(df, max_segments=200, loh_weight=0.0)
    assert out.groupby("Tumor_Sample_Barcode").size().max() <= 200
    # For a capped sample, the kept |Segment_Mean| min >= the dropped max.
    big = df.groupby("Tumor_Sample_Barcode").size().idxmax()
    g_in = df[df["Tumor_Sample_Barcode"] == big]
    g_out = out[out["Tumor_Sample_Barcode"] == big]
    kept = set(map(tuple, g_out[["Chromosome", "Segment_Mean", "LOH"]].values))
    dropped = g_in[~g_in[["Chromosome", "Segment_Mean", "LOH"]]
                   .apply(tuple, axis=1).isin(kept)]
    assert g_out["Segment_Mean"].abs().min() >= dropped["Segment_Mean"].abs().max()


def test_cna_optional_loh():
    df = _make_cna_cohort().drop(columns=["LOH"])
    out = subsample_cna(df, max_segments=100)
    assert out.groupby("Tumor_Sample_Barcode").size().max() <= 100


# --- equivalence to the manuscript helpers (drift guard) --------------------
def test_snv_matches_manuscript():
    df = _make_snv_cohort()
    df["mut_id"] = df[["Chromosome", "Start_Position", "Reference_Allele",
                       "Tumor_Seq_Allele2"]].astype(str).agg("_".join, axis=1)
    pkg = subsample_snv(df, max_variants=200, min_recurrence=3, random_state=42)
    man = subsample_variants_with_common(df, subsample_size=200, min_shared=3,
                                         random_state=42)
    pd.testing.assert_frame_equal(_canon(pkg), _canon(man))


def test_cna_matches_manuscript():
    df = _make_cna_cohort()
    pkg = subsample_cna(df, max_segments=300, loh_weight=0.5)
    man = subsample_cna_priority(df, subsample_size=300, loh_weight=0.5)
    pd.testing.assert_frame_equal(_canon(pkg), _canon(man))


def test_snv_matches_manuscript_derived():
    # Pins the auto-derived mut_id path (no pre-built mut_id) to the manuscript,
    # which requires mut_id — so we hand it a copy keyed exactly as the package
    # derives internally (locus cols, ':'-joined). This is the path real callers hit.
    df = _make_snv_cohort()
    manual = df.copy()
    manual["mut_id"] = manual[MUT_ID_COLS].astype(str).agg(":".join, axis=1)
    pkg = subsample_snv(df, max_variants=200, min_recurrence=3, random_state=42)
    man = subsample_variants_with_common(manual, subsample_size=200, min_shared=3,
                                         random_state=42).drop(columns="mut_id")
    pd.testing.assert_frame_equal(_canon(pkg), _canon(man))


def test_snv_recurrent_overflow_exceeds_cap():
    # Documented soft cap: recurrent variants alone can exceed max_variants.
    rows = [(s, "1", 1000 + i, "A", "T") for s in ("A", "B") for i in range(30)]
    df = pd.DataFrame(rows, columns=["Tumor_Sample_Barcode", *MUT_ID_COLS])
    out = subsample_snv(df, max_variants=10, min_recurrence=2)
    assert (out.groupby("Tumor_Sample_Barcode").size() == 30).all()


def test_snv_empty_returns_empty():
    df = _make_snv_cohort().iloc[:0]
    out = subsample_snv(df, max_variants=100)
    assert len(out) == 0 and list(out.columns) == list(df.columns)


def test_snv_error_paths():
    df = _make_snv_cohort()
    with pytest.raises(KeyError):
        subsample_snv(df.drop(columns="Tumor_Sample_Barcode"), max_variants=10)
    with pytest.raises(ValueError):
        subsample_snv(df, max_variants=0)
    with pytest.raises(ValueError):
        subsample_snv(df, min_recurrence=0)


def test_cna_loh_absent_matches_loh_zero():
    df = _make_cna_cohort()
    a = subsample_cna(df.drop(columns="LOH"), max_segments=300, loh_weight=0.5)
    b = subsample_cna(df.assign(LOH=0), max_segments=300, loh_weight=0.5).drop(columns="LOH")
    pd.testing.assert_frame_equal(_canon(a), _canon(b))


def test_cna_preserves_caller_score_column():
    df = _make_cna_cohort()
    df["__score__"] = 999.0  # the old implementation would clobber this
    out = subsample_cna(df, max_segments=200)
    assert "__score__" in out.columns and (out["__score__"] == 999.0).all()


def test_cna_empty_returns_empty():
    df = _make_cna_cohort().iloc[:0]
    out = subsample_cna(df, max_segments=100)
    assert len(out) == 0 and list(out.columns) == list(df.columns)


def test_cna_error_paths():
    df = _make_cna_cohort()
    with pytest.raises(KeyError):
        subsample_cna(df.drop(columns="Tumor_Sample_Barcode"), max_segments=10)
    with pytest.raises(KeyError):
        subsample_cna(df.drop(columns="Segment_Mean"), max_segments=10)
    with pytest.raises(ValueError):
        subsample_cna(df, max_segments=0)


def test_accessible_via_preprocessing():
    from tessera.data import preprocessing
    assert preprocessing.subsample_snv is subsample_snv
    assert preprocessing.subsample_cna is subsample_cna
