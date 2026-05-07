"""Export MSK-CHORD per-token RobustScalers + feature ordering.

The CRC predictive-biomarker pipeline (build_patient_features →
fit_full_data_model) fits two RobustScalers on the fly: one on the SNV
per-token embeddings and one on the CNA per-token embeddings (both fitted
on the MSK-CHORD cohort). After scaling, embeddings are mean+max pooled
per patient, concatenated with log-count features, and passed through the
DR-learner pipeline whose composed coefficient vector (`beta_eff`, length
3716) is already saved in attribution_analysis/crc_signatures/attribution_matrix.pkl.

The RobustScaler state is NOT saved by build_patient_features itself, so to
apply β_eff to an external cohort (DepMap) we need to re-export it.

This script:
  1. Loads MSK-CHORD raw embeddings (msk_chord_latent_features_raw.pkl)
  2. Fits SNV and CNA RobustScalers exactly as build_patient_features does
  3. Saves the scaler state + per-modality embedding dimensions + feature
     column order to a single pickle.

Output: cache/msk_chord_scalers.pkl

Usage in downstream cohorts (DepMap, etc.):
    bundle = pickle.load(open("cache/msk_chord_scalers.pkl","rb"))
    snv_z  = bundle["snv_scaler"].transform(depmap_snv_emb)
    cna_z  = bundle["cna_scaler"].transform(depmap_cna_emb)
    # ... mean+max pool, concat in bundle["feature_order"], add log counts
    tau    = X_per_sample @ beta_eff + const_eff
"""
from __future__ import annotations
import pickle
from pathlib import Path
import numpy as np
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parent
RAW  = ROOT / "msk_chord_latent_features_raw.pkl"
OUT  = ROOT / "cache" / "msk_chord_scalers.pkl"


def main():
    print("Loading MSK-CHORD raw embeddings...")
    with open(RAW, "rb") as fh:
        bundle = pickle.load(fh)
    snv_emb = bundle["variant_features"]
    cna_emb = bundle["cna_features"]
    print(f"  SNV embeddings: {snv_emb.shape}")
    print(f"  CNA embeddings: {cna_emb.shape}")

    print("Fitting RobustScalers (matches build_patient_features) ...")
    snv_scaler = RobustScaler().fit(snv_emb)
    cna_scaler = RobustScaler().fit(cna_emb)

    # Feature column order produced by build_patient_features:
    #   snv_mean_0..d-1, snv_max_0..d-1, cna_mean_0..d-1, cna_max_0..d-1,
    #   tmb_log1p, cna_log1p
    snv_d = snv_emb.shape[1]
    cna_d = cna_emb.shape[1]
    feature_order = (
        [f"snv_mean_{i}" for i in range(snv_d)]
        + [f"snv_max_{i}"  for i in range(snv_d)]
        + [f"cna_mean_{i}" for i in range(cna_d)]
        + [f"cna_max_{i}"  for i in range(cna_d)]
        + ["tmb_log1p", "cna_log1p"]
    )
    expected_total = 2 * snv_d + 2 * cna_d + 2
    assert len(feature_order) == expected_total, (
        f"feature order length {len(feature_order)} != expected {expected_total}"
    )
    print(f"  feature order: {len(feature_order)} columns "
          f"(2 × SNV {snv_d} + 2 × CNA {cna_d} + 2 log-counts)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "wb") as fh:
        pickle.dump({
            "snv_scaler": snv_scaler,
            "cna_scaler": cna_scaler,
            "snv_emb_dim": snv_d,
            "cna_emb_dim": cna_d,
            "feature_order": feature_order,
            "n_features": expected_total,
        }, fh)
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
