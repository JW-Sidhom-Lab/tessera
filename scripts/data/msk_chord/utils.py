"""Subsampling helper for the MSK-CHORD model-ready SNV pipeline."""

from __future__ import annotations

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
