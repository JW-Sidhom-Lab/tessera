"""Compute the Abraham 2021 67-gene panel intersection with MSK-IMPACT.

The FOLFOXai signature was originally trained on a 67-gene panel reported
in Abraham et al. 2021 (Clin Cancer Res; see ``abraham67_genes.txt``).
MSK-CHORD profiles tumours with the MSK-IMPACT panel, which covers a
different gene set. This script quantifies the intersection.

Inputs
------
``abraham67_genes.txt``
    Plain-text list of the 67 Abraham panel gene symbols.
``../data/msk_chord/snv.csv``
    MSK-CHORD somatic mutation table (``Hugo_Symbol`` column lists every
    gene where any MSK-IMPACT panel call was made).
``../data/msk_chord/cna_panel_filtered.csv``
    MSK-CHORD panel-filtered copy-number segment table (``Hugo_Symbol``).

Outputs
-------
``outputs/abraham67_msk_overlap.csv``
    Per-gene table with columns ``gene``, ``in_msk_snv``, ``in_msk_cna``,
    ``in_msk_impact`` (union). Sorted alphabetically.
``outputs/abraham67_msk_missing.csv``
    The subset of the above with ``in_msk_impact == 0`` (39 rows),
    column ``gene`` only.

Path overrides via env vars: ``ABRAHAM_GENES_TXT``, ``MSK_SNV_CSV``,
``MSK_CNA_CSV``, ``OUTPUT_DIR``.

Usage:
    python3 00_gene_overlap.py
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


HERE = Path(__file__).parent.resolve()

ABRAHAM_GENES_TXT = Path(os.environ.get(
    'ABRAHAM_GENES_TXT', HERE / 'abraham67_genes.txt'))
MSK_SNV_CSV  = Path(os.environ.get(
    'MSK_SNV_CSV', HERE / '..' / 'data' / 'msk_chord' / 'snv.csv'))
MSK_CNA_CSV  = Path(os.environ.get(
    'MSK_CNA_CSV', HERE / '..' / 'data' / 'msk_chord' / 'cna_panel_filtered.csv'))
OUTPUT_DIR   = Path(os.environ.get('OUTPUT_DIR', HERE / 'outputs'))


def read_gene_list(path: Path) -> list[str]:
    """Read one-gene-per-line text file, ignoring blank lines and comments."""
    with open(path) as f:
        rows = [line.rstrip('\n').strip() for line in f]
    return [r for r in rows if r and not r.startswith('#')]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    abraham = read_gene_list(ABRAHAM_GENES_TXT)
    print(f'Abraham 2021 gene panel: {len(abraham)} genes')

    snv_genes = set(pd.read_csv(MSK_SNV_CSV,
                                 usecols=['Hugo_Symbol'])['Hugo_Symbol']
                    .dropna().unique())
    cna_genes = set(pd.read_csv(MSK_CNA_CSV,
                                 usecols=['Hugo_Symbol'])['Hugo_Symbol']
                    .dropna().unique())
    msk_genes = snv_genes | cna_genes
    print(f'MSK-IMPACT panel (union of SNV+CNA tables): {len(msk_genes):,} genes')

    overlap = pd.DataFrame({
        'gene':          sorted(abraham),
    })
    overlap['in_msk_snv']    = overlap['gene'].isin(snv_genes).astype(int)
    overlap['in_msk_cna']    = overlap['gene'].isin(cna_genes).astype(int)
    overlap['in_msk_impact'] = overlap['gene'].isin(msk_genes).astype(int)

    overlap_path = OUTPUT_DIR / 'abraham67_msk_overlap.csv'
    overlap.to_csv(overlap_path, index=False)

    missing = (overlap[overlap['in_msk_impact'] == 0]
               [['gene']]
               .reset_index(drop=True))
    missing_path = OUTPUT_DIR / 'abraham67_msk_missing.csv'
    missing.to_csv(missing_path, index=False)

    n_in   = int(overlap['in_msk_impact'].sum())
    pct_in = 100.0 * n_in / len(overlap)
    print()
    print(f'Abraham 67 in MSK-IMPACT:  {n_in}/{len(overlap)}  ({pct_in:.1f}%)')
    print(f'Abraham 67 not covered:    {len(missing)}/{len(overlap)}')
    if not missing.empty:
        print(f'Genes not in MSK-IMPACT:   {", ".join(missing["gene"].tolist())}')
    print()
    print(f'wrote {overlap_path}')
    print(f'wrote {missing_path}')


if __name__ == '__main__':
    main()
