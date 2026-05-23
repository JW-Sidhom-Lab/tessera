"""Build the Abraham 2021 FOLFOXai feature matrix and IB/DB label on the
MSK-CHORD 1L stage IV CRC cohort. Faithful paper replication: binary
IB/DB labels over time-to-next-treatment discontinuation (TTNTD) at 270
days.

Inputs
------
``abraham67_genes.txt``
    Plain-text list of the 67 Abraham panel gene symbols.
``../../data/msk_chord_2024/GROUND_TRUTH_CRC_FOLFOX_FOLFIRI_STAGE4_TTNTD.csv``
    MSK-CHORD 1L stage IV CRC FOLFOX / FOLFIRI cohort ground truth.
``../data/msk_chord/snv.csv``
    MSK-CHORD somatic mutation table (MSK-IMPACT calls).
``../data/msk_chord/cna_panel_filtered.csv``
    MSK-CHORD panel-filtered copy-number segment table.

Output
------
``causal_inference_results/features/patient_features.csv``
    Per-patient feature matrix with columns ``PATIENT_ID``, ``REGIMEN``,
    ``OS_MONTHS``, ``OS_EVENT``, ``TTNTD_DAYS``, ``TTNTD_CENSORED``,
    ``ib_db`` (1=IB, 0=DB, NaN=dropped), plus ``<GENE>_copies`` and
    ``<GENE>_mutations`` columns for each of the 67 Abraham genes.

IB / DB label (Abraham 2021):
    IB  (1) — TTNTD_DAYS >= 270
    DB  (0) — TTNTD_DAYS  < 270 and event observed (not censored)
    NaN     — TTNTD_DAYS  < 270 and censored (short-follow-up, drop from training)

Path overrides via env vars: ``ABRAHAM_GENES_TXT``, ``GT_CSV``,
``SNV_CSV``, ``CNA_CSV``, ``OUTPUT_CSV``.

Usage:
    python3 1_build_features.py
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).parent.resolve()

ABRAHAM_GENES_TXT = Path(os.environ.get(
    'ABRAHAM_GENES_TXT', HERE / 'abraham67_genes.txt'))
GT_CSV    = Path(os.environ.get('GT_CSV',
    HERE / '..' / '..' / 'data' / 'msk_chord_2024'
         / 'GROUND_TRUTH_CRC_FOLFOX_FOLFIRI_STAGE4_TTNTD.csv'))
SNV_CSV   = Path(os.environ.get('SNV_CSV',
    HERE / '..' / 'data' / 'msk_chord' / 'snv.csv'))
CNA_CSV   = Path(os.environ.get('CNA_CSV',
    HERE / '..' / 'data' / 'msk_chord' / 'cna_panel_filtered.csv'))
OUTPUT_CSV = Path(os.environ.get('OUTPUT_CSV',
    HERE / 'causal_inference_results' / 'features' / 'patient_features.csv'))

IB_CUTOFF_DAYS = 270

CODING_IMPACT = {
    'Missense_Mutation', 'Nonsense_Mutation', 'Splice_Site',
    'Frame_Shift_Del', 'Frame_Shift_Ins', 'In_Frame_Del', 'In_Frame_Ins',
    'Translation_Start_Site', 'Nonstop_Mutation',
}


def read_gene_list(path: Path) -> list[str]:
    with open(path) as f:
        rows = [line.rstrip('\n').strip() for line in f]
    return [r for r in rows if r and not r.startswith('#')]


def load_cohort():
    gt = pd.read_csv(GT_CSV, low_memory=False)
    gt = gt[gt['LINE_OF_THERAPY'] == 1].copy()
    gt = gt[gt['REGIMEN'].isin(['FOLFOX', 'FOLFIRI'])].copy()

    gt['OS_EVENT']       = gt['OS_STATUS'].astype(str).str.split(':').str[0].astype(int)
    gt['OS_MONTHS']      = pd.to_numeric(gt['OS_MONTHS_FROM_TREATMENT_START'],
                                          errors='coerce')
    gt['TTNTD_DAYS']     = pd.to_numeric(gt['TTNTD_DAYS'], errors='coerce')
    gt['TTNTD_CENSORED'] = pd.to_numeric(gt['TTNTD_IS_CENSORED'],
                                          errors='coerce').astype('Int64')

    gt = gt.dropna(subset=['OS_MONTHS', 'TTNTD_DAYS', 'TTNTD_CENSORED'])
    gt = gt[gt['OS_MONTHS'] > 0].copy()
    gt = gt[['PATIENT_ID', 'REGIMEN', 'OS_MONTHS', 'OS_EVENT',
             'TTNTD_DAYS', 'TTNTD_CENSORED']]
    gt = gt.sort_values('OS_MONTHS').drop_duplicates('PATIENT_ID', keep='first')

    ib_db = pd.Series(np.nan, index=gt.index)
    ib_db.loc[gt['TTNTD_DAYS'] >= IB_CUTOFF_DAYS] = 1
    db_mask = (gt['TTNTD_DAYS'] < IB_CUTOFF_DAYS) & (gt['TTNTD_CENSORED'] == 0)
    ib_db.loc[db_mask] = 0
    gt['ib_db'] = ib_db
    return gt.reset_index(drop=True)


def build_mutation_matrix(patient_ids, genes):
    print(f'  reading SNV table ...')
    snv = pd.read_csv(SNV_CSV,
                       usecols=['Tumor_Sample_Barcode', 'Hugo_Symbol',
                                'Variant_Classification'],
                       low_memory=False)
    snv['PATIENT_ID'] = snv['Tumor_Sample_Barcode'].str.slice(0, 9)
    snv = snv[snv['PATIENT_ID'].isin(patient_ids)]
    snv = snv[snv['Hugo_Symbol'].isin(genes)]
    snv = snv[snv['Variant_Classification'].isin(CODING_IMPACT)]
    print(f'    SNV rows after filter: {len(snv):,}')

    mx = (snv.assign(hit=1)
              .groupby(['PATIENT_ID', 'Hugo_Symbol'])['hit'].max()
              .unstack(fill_value=0)
              .reindex(index=patient_ids, columns=genes, fill_value=0)
              .astype(int))
    mx.columns = [f'{g}_mutations' for g in mx.columns]
    return mx


def build_cna_matrix(patient_ids, genes):
    print(f'  reading CNA (panel-filtered) table ...')
    cna = pd.read_csv(CNA_CSV,
                       usecols=['Tumor_Sample_Barcode', 'Hugo_Symbol',
                                'Segment_Mean'],
                       low_memory=False)
    cna['PATIENT_ID'] = cna['Tumor_Sample_Barcode'].str.slice(0, 9)
    cna = cna[cna['PATIENT_ID'].isin(patient_ids)]
    cna = cna[cna['Hugo_Symbol'].isin(genes)]
    cna['Segment_Mean'] = pd.to_numeric(cna['Segment_Mean'], errors='coerce')
    cna = cna.dropna(subset=['Segment_Mean'])
    cna['CN'] = 2.0 * np.power(2.0, cna['Segment_Mean'].clip(-3, 3))
    print(f'    CNA rows after filter: {len(cna):,}')

    mx = (cna.groupby(['PATIENT_ID', 'Hugo_Symbol'])['CN'].mean()
              .unstack()
              .reindex(index=patient_ids, columns=genes))
    mx = mx.fillna(2.0)
    mx.columns = [f'{g}_copies' for g in mx.columns]
    return mx


def main():
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    genes = read_gene_list(ABRAHAM_GENES_TXT)
    print(f'Abraham gene list: {len(genes)} genes.')

    cohort = load_cohort()
    n_fox  = int((cohort.REGIMEN == 'FOLFOX').sum())
    n_fir  = int((cohort.REGIMEN == 'FOLFIRI').sum())
    print(f'Eligible 1L cohort: n={len(cohort)}  ({n_fox} FOLFOX / {n_fir} FOLFIRI)')

    fox = cohort[cohort['REGIMEN'] == 'FOLFOX']
    n_ib = int((fox['ib_db'] == 1).sum())
    n_db = int((fox['ib_db'] == 0).sum())
    n_drop = int(fox['ib_db'].isna().sum())
    print(f'FOLFOX IB/DB labelling: IB={n_ib}  DB={n_db}  dropped(short-censored)={n_drop}')

    patient_ids = cohort['PATIENT_ID'].tolist()
    mut_mx = build_mutation_matrix(patient_ids, genes)
    cna_mx = build_cna_matrix(patient_ids, genes)

    has_snv = mut_mx.sum(axis=1) > 0
    has_cna = (cna_mx != 2.0).any(axis=1)
    keep    = has_snv | has_cna
    n_drop_geno = int((~keep).sum())
    if n_drop_geno:
        print(f'  dropping {n_drop_geno} patients with no SNV + all-diploid CNA')
    mut_mx = mut_mx.loc[keep]
    cna_mx = cna_mx.loc[keep]
    cohort = cohort.set_index('PATIENT_ID').loc[mut_mx.index].reset_index()

    copies_cols = [f'{g}_copies'    for g in genes]
    muts_cols   = [f'{g}_mutations' for g in genes]
    feat = pd.concat([cna_mx[copies_cols], mut_mx[muts_cols]], axis=1)
    out = cohort.merge(feat, left_on='PATIENT_ID', right_index=True)
    assert len(out) == len(cohort)

    out.to_csv(OUTPUT_CSV, index=False)
    print(f'wrote {OUTPUT_CSV}')
    print(f'  final n={len(out)}  '
          f'({(out.REGIMEN == "FOLFOX").sum()} FOLFOX / '
          f'{(out.REGIMEN == "FOLFIRI").sum()} FOLFIRI)')
    fox = out[out['REGIMEN'] == 'FOLFOX']
    n_ib = int((fox['ib_db'] == 1).sum())
    n_db = int((fox['ib_db'] == 0).sum())
    n_drop = int(fox['ib_db'].isna().sum())
    print(f'  FOLFOX IB/DB after genotype filter: IB={n_ib} '
          f'({100 * n_ib / (n_ib + n_db):.1f}%)  '
          f'DB={n_db}  short-censored drops={n_drop}')


if __name__ == '__main__':
    main()
