"""Per-patient genomic feature matrix from raw foundation-model embeddings.

Reads the raw embedding pickle (variant-level and CNA-level embeddings,
plus the corresponding mutation/CNA tables), applies RobustScaler to each
embedding matrix, then aggregates per-patient as mean and max of the
scaled rows. Adds log1p mutation count (TMB proxy) and log1p CNA count
as additional columns.

The patient identifier is the first 9 characters of the Tumor_Sample_Barcode
(MSK CHORD's per-patient prefix). Patients without both SNV and CNA records
are dropped.

The result is cached as a pickle so subsequent calls skip the heavy
RobustScaler+groupby work.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


def build_patient_features(
    raw_pkl: str | Path,
    cache: str | Path | None = None,
) -> pd.DataFrame:
    raw_pkl = Path(raw_pkl)
    cache_path = Path(cache) if cache is not None else None

    if cache_path is not None and cache_path.exists():
        return pd.read_pickle(cache_path)

    with open(raw_pkl, "rb") as fh:
        bundle = pickle.load(fh)

    snv_emb = bundle["variant_features"]
    cna_emb = bundle["cna_features"]
    snv_meta = bundle["data_snv"].copy()
    cna_meta = bundle["data_cna"].copy()

    snv_meta["pid"] = snv_meta["Tumor_Sample_Barcode"].str[:9]
    cna_meta["pid"] = cna_meta["Tumor_Sample_Barcode"].str[:9]

    snv_z = RobustScaler().fit_transform(snv_emb)
    cna_z = RobustScaler().fit_transform(cna_emb)

    snv_df = pd.DataFrame(snv_z, index=snv_meta["pid"].values)
    cna_df = pd.DataFrame(cna_z, index=cna_meta["pid"].values)

    snv_mean = snv_df.groupby(level=0).mean().add_prefix("snv_mean_")
    snv_max  = snv_df.groupby(level=0).max().add_prefix("snv_max_")
    cna_mean = cna_df.groupby(level=0).mean().add_prefix("cna_mean_")
    cna_max  = cna_df.groupby(level=0).max().add_prefix("cna_max_")

    snv_n = snv_df.groupby(level=0).size()
    cna_n = cna_df.groupby(level=0).size()

    pids = sorted(set(snv_mean.index) & set(cna_mean.index))
    out = pd.concat([
        snv_mean.loc[pids],
        snv_max.loc[pids],
        cna_mean.loc[pids],
        cna_max.loc[pids],
    ], axis=1)
    out["tmb_log1p"] = np.log1p(snv_n.reindex(pids).fillna(0).values)
    out["cna_log1p"] = np.log1p(cna_n.reindex(pids).fillna(0).values)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_pickle(cache_path)

    return out
