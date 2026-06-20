"""Hermetic tests for the reconstruct() helpers — no model weights, no genome.

All of these exercise the pure numpy/pandas helpers behind
:meth:`tessera.model.TESSERA.reconstruct` plus the two input-validation guards
that fail fast before any model is touched. The integration behaviour (the joint
graph itself) is covered separately; here we pin the formulas, the score-table
shapes, the token decoding, and the guards.
"""
import math

import numpy as np
import pandas as pd
import pytest

from tessera.model import (
    TESSERA,
    _segment_mean_metrics,
    _loh_metrics,
    _snv_recon_metrics,
    _snv_score_table,
    _cna_score_table,
    _dummy_snv_df,
    _dummy_cna_df,
)

REVERSE_TOKEN_MAP = {0: "[PAD]", 1: "A", 2: "C", 3: "G", 4: "T", 5: "-", 6: "X"}


# --------------------------------------------------------------------------- #
# _segment_mean_metrics
# --------------------------------------------------------------------------- #
def test_segment_mean_metrics_perfect_prediction():
    pred = np.array([1.0, 2.0, 3.0])
    true = np.array([1.0, 2.0, 3.0])
    m = _segment_mean_metrics(pred, true)
    assert m["cna_mse"] < 1e-12
    assert m["cna_mae"] < 1e-12
    assert abs(m["cna_r2"] - 1.0) < 1e-9
    assert abs(m["cna_correlation"] - 1.0) < 1e-9


def test_segment_mean_metrics_r2_nan_when_true_constant():
    # ss_tot == 0 → r2 guarded to nan, other metrics still finite.
    m = _segment_mean_metrics(np.array([0.1, 0.2, 0.3]), np.array([0.0, 0.0, 0.0]))
    assert math.isnan(m["cna_r2"])
    assert math.isfinite(m["cna_mse"])
    assert math.isfinite(m["cna_mae"])


def test_segment_mean_metrics_correlation_nan_single_element():
    m = _segment_mean_metrics(np.array([0.5]), np.array([1.0]))
    assert math.isnan(m["cna_correlation"])
    assert math.isfinite(m["cna_mse"])


# --------------------------------------------------------------------------- #
# _loh_metrics
# --------------------------------------------------------------------------- #
def test_loh_metrics_single_class_omits_auroc():
    # roc_auc_score is undefined for a single class; the guard must drop the key
    # rather than raise.
    m = _loh_metrics(np.array([0.1, 0.2, 0.4]), np.array([0, 0, 0]))
    assert "cna_loh_auc_roc" not in m
    assert "cna_loh_accuracy" in m


def test_loh_metrics_two_class_includes_auroc():
    m = _loh_metrics(np.array([0.9, 0.1, 0.8, 0.2]), np.array([1, 0, 1, 0]))
    assert "cna_loh_auc_roc" in m
    assert 0.0 <= m["cna_loh_auc_roc"] <= 1.0
    assert m["cna_loh_accuracy"] == 1.0


# --------------------------------------------------------------------------- #
# _snv_recon_metrics
# --------------------------------------------------------------------------- #
def test_snv_recon_metrics_accuracy_and_observed_prob():
    # Two variants, both true token == 3. Sample 0 perfect, sample 1 prob 0.4.
    probs = np.zeros((2, 1, 7))
    probs[0, 0, 3] = 1.0
    probs[1, 0, 3] = 0.4
    probs[1, 0, 1] = 0.6  # argmax is wrong for sample 1
    true_tokens = np.full((2, 1), 3, dtype=int)
    loss = np.array([0.0, 0.916])
    m = _snv_recon_metrics(probs, true_tokens, loss)
    assert abs(m["snv_mean_observed_prob"] - 0.7) < 1e-9  # mean(1.0, 0.4)
    assert abs(m["snv_accuracy"] - 0.5) < 1e-9  # only sample 0 argmax-correct
    assert abs(m["snv_mean_loss"] - np.mean(loss)) < 1e-9


# --------------------------------------------------------------------------- #
# _snv_score_table
# --------------------------------------------------------------------------- #
def _snv_df(n):
    return pd.DataFrame({"Chromosome": ["1"] * n, "Start_Position": list(range(n))})


def test_snv_score_table_observed_prob_is_true_class_not_predicted():
    # Predicted class (1 / 'A') differs from true class (2 / 'C').
    p = np.zeros((3, 1, 7))
    p[:, 0, 1] = 0.7  # predicted
    p[:, 0, 2] = 0.3  # true
    tt = np.full((3, 1), 2, dtype=int)
    s = _snv_score_table(_snv_df(3), p, tt, np.zeros(3), None, None, REVERSE_TOKEN_MAP)
    assert (s["pred_base"] == "A").all()
    assert (s["true_base"] == "C").all()
    assert (s["observed_prob"].round(6) == 0.3).all()
    assert (s["pred_prob"].round(6) == 0.7).all()
    assert (~s["correct"]).all()
    assert "ref_pred_base" not in s.columns  # no ref tail supplied


def test_snv_score_table_ref_tail_adds_ref_columns():
    p = np.zeros((2, 1, 7))
    p[:, 0, 3] = 1.0
    tt = np.full((2, 1), 3, dtype=int)
    ref_probs = np.zeros((2, 1, 7))
    ref_probs[:, 0, 1] = 1.0
    ref_true = np.full((2, 1), 1, dtype=int)
    s = _snv_score_table(_snv_df(2), p, tt, np.zeros(2), ref_probs, ref_true,
                         REVERSE_TOKEN_MAP)
    assert list(s["ref_pred_base"]) == ["A", "A"]
    assert (s["ref_observed_prob"].round(6) == 1.0).all()
    assert s["ref_correct"].all()


# --------------------------------------------------------------------------- #
# _cna_score_table
# --------------------------------------------------------------------------- #
def _cna_df(n):
    return pd.DataFrame({
        "Chromosome": ["1"] * n,
        "Start": list(range(n)),
        "End": list(range(1, n + 1)),
        "Segment_Mean": np.zeros(n),
    })


def test_cna_score_table_without_loh():
    s = _cna_score_table(_cna_df(2), np.array([0.1, 0.9]), np.array([0.0, 1.0]),
                         None, None)
    assert "loh_pred" not in s.columns
    assert "loh_true" not in s.columns
    assert list(s["abs_error"].round(6)) == [0.1, 0.1]
    assert list(s["signed_error"].round(6)) == [0.1, -0.1]


def test_cna_score_table_with_loh():
    s = _cna_score_table(_cna_df(2), np.array([0.1, 0.9]), np.array([0.0, 1.0]),
                         np.array([0.3, 0.7]), np.array([0, 1]))
    assert "loh_pred" in s.columns
    assert "loh_true" in s.columns
    assert list(s["loh_true"]) == [0, 1]


# --------------------------------------------------------------------------- #
# _dummy_snv_df / _dummy_cna_df
# --------------------------------------------------------------------------- #
def test_dummy_snv_df_deduplicates_samples():
    df = _dummy_snv_df(np.array(["S1", "S1", "S2", "S2", "S2"]))
    assert list(df["Tumor_Sample_Barcode"]) == ["S1", "S2"]
    assert "vaf" in df.columns
    assert {"Chromosome", "Start_Position", "Reference_Allele",
            "Tumor_Seq_Allele2"}.issubset(df.columns)


def test_dummy_cna_df_deduplicates_samples():
    df = _dummy_cna_df(np.array(["S1", "S1", "S3"]))
    assert list(df["Tumor_Sample_Barcode"]) == ["S1", "S3"]
    assert {"Chromosome", "Start", "End", "Segment_Mean"}.issubset(df.columns)


# --------------------------------------------------------------------------- #
# reconstruct() input-validation guards (fail before any model work)
# --------------------------------------------------------------------------- #
def test_reconstruct_requires_at_least_one_modality():
    model = TESSERA.__new__(TESSERA)  # bypass __init__ / model loading
    with pytest.raises(ValueError, match="at least one"):
        model.reconstruct(snv_df=None, cna_df=None)


def test_reconstruct_loh_variant_requires_loh_column(monkeypatch):
    # A variant whose config has a LoH head (use_cna_loh=True) must reject a
    # cna_df with no 'LOH' column with a clear message, not a cryptic Keras
    # "Missing data for input cna_loh" deep in predict().
    model = TESSERA.__new__(TESSERA)
    model.use_cna_loh = True
    monkeypatch.setattr(model, "_load_model_config_if_needed", lambda: None)
    cna = pd.DataFrame({
        "Tumor_Sample_Barcode": ["S1"],
        "Chromosome": ["1"],
        "Start": [100],
        "End": [200],
        "Segment_Mean": [0.0],
    })
    with pytest.raises(ValueError, match="LoH reconstruction head"):
        model.reconstruct(cna_df=cna, from_assembly="GRCh37")
