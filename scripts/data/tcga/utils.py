"""Subsampling helpers shared across the TCGA model-ready dataset scripts.

``subsample_variants_with_common``
    Cap variants per sample while preserving alterations recurrent across
    samples (used by the SNV and all-variant pipelines).

``subsample_cna_priority``
    Cap segments per sample by alteration magnitude with an LOH bonus (used
    by the CNA pipeline).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def subsample_variants_with_common(data: pd.DataFrame,
                                   subsample_size: int = 100,
                                   min_shared: int = 50,
                                   sample_col: str = "Tumor_Sample_Barcode",
                                   random_state: int | None = 42) -> pd.DataFrame:
    """Subsample variants per sample while preserving recurrent variants.

    For each sample, the function first includes every variant whose
    ``mut_id`` appears in at least ``min_shared`` samples cohort-wide
    (a low-effort proxy for recurrent or driver alterations), then
    fills the remaining budget with a uniform random sample from the
    rest, capped at ``subsample_size`` total variants per sample.

    Args:
        data: DataFrame with at least ``mut_id`` and ``sample_col`` columns.
        subsample_size: Maximum variants kept per sample.
        min_shared: Minimum number of samples a variant must appear in to be
            preserved as a recurrent variant.
        sample_col: Column identifying the sample.
        random_state: Seed for the random non-recurrent sample.

    Returns:
        Subsampled frame, reset-indexed.
    """
    variant_counts = data["mut_id"].value_counts()
    common_variants = set(variant_counts[variant_counts >= min_shared].index)

    def _subsample_group(group: pd.DataFrame) -> pd.DataFrame:
        common_in_sample = group[group["mut_id"].isin(common_variants)]
        remaining_slots = subsample_size - len(common_in_sample)
        if remaining_slots <= 0:
            return common_in_sample
        other_variants = group[~group["mut_id"].isin(common_variants)]
        if other_variants.empty:
            return common_in_sample
        n_to_sample = min(len(other_variants), remaining_slots)
        sampled_others = other_variants.sample(
            n=n_to_sample, replace=False, random_state=random_state,
        )
        return pd.concat([common_in_sample, sampled_others])

    return (data.groupby(sample_col, group_keys=False)
                .apply(_subsample_group)
                .reset_index(drop=True))


def subsample_cna_priority(data: pd.DataFrame,
                           subsample_size: int = 1000,
                           sample_col: str = "Tumor_Sample_Barcode",
                           loh_weight: float = 0.5) -> pd.DataFrame:
    """Subsample CNA segments per sample by alteration magnitude.

    Segments are scored by ``|Segment_Mean| + loh_weight * LOH``, and the
    top ``subsample_size`` per sample are kept. Samples with fewer segments
    than ``subsample_size`` are returned unchanged.

    Args:
        data: DataFrame with ``Segment_Mean``, ``LOH``, and ``sample_col``.
        subsample_size: Maximum segments kept per sample.
        sample_col: Column identifying the sample.
        loh_weight: Bonus added to the score for LOH segments.

    Returns:
        Subsampled frame, reset-indexed, without the temporary score column.
    """
    def _subsample_group(group: pd.DataFrame) -> pd.DataFrame:
        if len(group) <= subsample_size:
            return group
        scored = group.copy()
        scored["alteration_score"] = (
            np.abs(scored["Segment_Mean"])
            + scored["LOH"].astype(float) * loh_weight
        )
        return (scored.nlargest(subsample_size, "alteration_score")
                      .drop(columns=["alteration_score"]))

    return (data.groupby(sample_col, group_keys=False)
                .apply(_subsample_group)
                .reset_index(drop=True))
