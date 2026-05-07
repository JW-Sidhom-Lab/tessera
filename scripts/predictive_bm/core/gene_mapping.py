"""SNV → gene and CNA-segment → gene mapping for attribution analysis.

SNVs already carry a Hugo_Symbol; mapping is a direct lookup.

For CNAs, each segment spans a chromosomal region; we map it to ALL panel
genes whose loci overlap the segment. Per-segment attribution is duplicated
to each overlapping gene (so a 5-Mb amplification covering 12 panel genes
contributes to all 12). This is the standard CNA → gene attribution.

Gene coordinates are derived self-containedly from the variant table itself:
for each gene G, we compute the chromosomal range [min(Start_Position),
max(End_Position)] across all observed SNVs in G. The IMPACT panel has many
variants per gene (TP53: 11K, KRAS: 7K, …) so these empirical boundaries
closely approximate the true gene exons. For genes with few observations,
boundaries may be conservative; pad by `pad_bp` (default 5kb) to cover this.

This is fully self-contained - no internet or external GTF required.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


def derive_gene_coords_from_snv(
    snv_meta: pd.DataFrame, *, pad_bp: int = 5_000,
) -> pd.DataFrame:
    """Compute approximate gene coordinates from observed SNVs.

    snv_meta must have columns: Hugo_Symbol, Chromosome, Start_Position, End_Position
    Returns: DataFrame with columns gene, chr, start, end, n_variants_observed
    """
    needed = {"Hugo_Symbol", "Chromosome", "Start_Position", "End_Position"}
    missing = needed - set(snv_meta.columns)
    if missing:
        raise ValueError(f"snv_meta missing required columns: {missing}")

    g = snv_meta.dropna(subset=["Hugo_Symbol", "Chromosome",
                                  "Start_Position", "End_Position"]).copy()
    g["Chromosome"] = g["Chromosome"].astype(str)
    coords = (g.groupby("Hugo_Symbol")
                .agg(chr=("Chromosome", lambda s: s.mode().iloc[0]),
                       start=("Start_Position", "min"),
                       end=("End_Position", "max"),
                       n_variants_observed=("Hugo_Symbol", "size"))
                .reset_index()
                .rename(columns={"Hugo_Symbol": "gene"}))
    # Apply padding (genes might have introns extending beyond observed variants)
    coords["start"] = (coords["start"] - pad_bp).clip(lower=0)
    coords["end"]   = coords["end"]   + pad_bp
    return coords


def overlap_segments_with_genes(
    cna_meta: pd.DataFrame, gene_coords: pd.DataFrame,
) -> pd.DataFrame:
    """Map each CNA segment to overlapping panel genes.

    cna_meta must have columns: row_id (the integer index in the source table),
                                Chromosome, Start, End
    gene_coords must have columns: gene, chr, start, end

    Returns long-format DataFrame: row_id | gene | chr | seg_start | seg_end
    Each (row_id, gene) pair is one row.

    Implementation: per-chromosome sweep using sort + binary-search-like merge.
    O((n_segments + n_genes) log n_genes) per chromosome; runs in a few seconds
    on the full ~1.3M CNA segments × 500 genes.
    """
    cna = cna_meta[["row_id", "Chromosome", "Start", "End"]].copy()
    cna["Chromosome"] = cna["Chromosome"].astype(str)

    out_rows = []
    by_chrom_genes = {c: g.sort_values("start").reset_index(drop=True)
                       for c, g in gene_coords.groupby("chr")}
    for chrom, segs in cna.groupby("Chromosome"):
        if chrom not in by_chrom_genes:
            continue
        gdf = by_chrom_genes[chrom]
        g_starts = gdf["start"].values
        g_ends   = gdf["end"].values
        g_names  = gdf["gene"].values

        for ridx, seg_start, seg_end in zip(segs["row_id"].values,
                                                segs["Start"].values,
                                                segs["End"].values):
            # First gene whose end >= seg_start (left bound)
            lo = np.searchsorted(g_ends, seg_start, side="left")
            # Walk right while gene start <= seg_end
            i = lo
            while i < len(g_starts) and g_starts[i] <= seg_end:
                # Overlap exists (since g_ends[i] >= seg_start and g_starts[i] <= seg_end)
                out_rows.append((int(ridx), g_names[i], chrom,
                                  int(seg_start), int(seg_end)))
                i += 1

    return pd.DataFrame(out_rows,
                          columns=["row_id", "gene", "chr",
                                    "seg_start", "seg_end"])


def attach_genes_to_attributions(
    var_attr: pd.DataFrame,
    snv_meta: pd.DataFrame,
    cna_meta: pd.DataFrame,
    cna_to_gene: pd.DataFrame,
) -> pd.DataFrame:
    """Add `gene` column to per-variant attributions.

    For SNV rows: gene = snv_meta.loc[row_id, 'Hugo_Symbol'] (direct).
    For CNA rows: each row_id may map to multiple genes via cna_to_gene; we
                  duplicate the row, divided equally across overlapping genes
                  (so a CNA segment overlapping 4 genes gives each gene 1/4 of
                  the segment's attribution).

    Returns a long-format DataFrame: pid | source | row_id | gene | attribution
    """
    # SNV: direct
    snv_rows = var_attr[var_attr["source"] == "snv"].copy()
    snv_rows["gene"] = snv_rows["row_id"].map(snv_meta["Hugo_Symbol"]).values

    # CNA: expand via cna_to_gene
    cna_rows = var_attr[var_attr["source"] == "cna"].copy()
    if len(cna_rows) > 0 and len(cna_to_gene) > 0:
        # Count genes per segment for credit-splitting
        n_genes_per_seg = (cna_to_gene.groupby("row_id").size()
                            .rename("n_genes").reset_index())
        # Merge attribution → gene mapping (long expansion)
        cna_long = cna_rows.merge(cna_to_gene[["row_id", "gene"]], on="row_id",
                                       how="inner")
        cna_long = cna_long.merge(n_genes_per_seg, on="row_id", how="left")
        cna_long["attribution"] = cna_long["attribution"] / cna_long["n_genes"]
        cna_long = cna_long.drop(columns=["n_genes"])
    else:
        cna_long = pd.DataFrame(columns=["pid", "source", "row_id",
                                            "attribution", "gene"])

    out = pd.concat([snv_rows[["pid", "source", "row_id", "gene", "attribution"]],
                      cna_long[["pid", "source", "row_id", "gene", "attribution"]]],
                      ignore_index=True)
    return out


def gene_attribution_per_patient(attr_with_gene: pd.DataFrame) -> pd.DataFrame:
    """Collapse to one row per (patient, gene): summed attribution + n alterations."""
    g = attr_with_gene.dropna(subset=["gene"]).copy()
    g["is_snv"] = (g["source"] == "snv").astype(int)
    g["is_cna"] = (g["source"] == "cna").astype(int)
    out = (g.groupby(["pid", "gene"])
              .agg(attribution=("attribution", "sum"),
                     n_alterations=("attribution", "size"),
                     n_snv=("is_snv", "sum"),
                     n_cna=("is_cna", "sum"))
              .reset_index())
    return out


def gene_signature_per_subgroup(
    gene_attr_per_patient: pd.DataFrame,
    cohort_patients: list[str],
    *, name: str = "subgroup",
) -> pd.DataFrame:
    """Aggregate per-patient gene attributions across a subgroup.

    Returns: gene | mean_attribution | mean_attribution_when_altered |
             n_patients_with_alteration | frac_patients_with_alteration |
             cohort_size | subgroup_name
    """
    sub = gene_attr_per_patient[gene_attr_per_patient["pid"].isin(cohort_patients)].copy()
    n_total = len(set(cohort_patients))

    # mean over all subgroup patients (denominator = n_total, even if patient has no alteration in gene)
    sums = sub.groupby("gene")["attribution"].sum()
    mean_attr_all = sums / n_total

    # mean over patients who have the alteration
    counts = sub.groupby("gene").size()
    mean_attr_when_altered = sums / counts

    out = pd.DataFrame({
        "gene": sums.index,
        "mean_attribution": mean_attr_all.values,
        "mean_attribution_when_altered": mean_attr_when_altered.values,
        "n_patients_with_alteration": counts.values,
        "frac_patients_with_alteration": counts.values / n_total,
        "cohort_size": n_total,
        "subgroup_name": name,
    })
    out = out.sort_values("mean_attribution", ascending=True).reset_index(drop=True)
    return out
