"""
Shared sample-level feature aggregation pipeline for prognostic_bm scripts.

Canonical pipeline (used by comparison and cluster scripts):
  1. Optional scaler on token-level features before pooling
  2. Mean + max pool per Tumor_Sample_Barcode
  3. Append log1p(TMB) to SNV vector; append log1p(CNA segment count) to CNA vector
  4. Concatenate SNV and CNA aggregated vectors
  5. Return (X [n_samples × dims], sample_ids [n_samples])
"""

import gc
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler


def _build_scaler(choice):
    """
    Return a fitted-ready scaler from a string/bool choice.

    Parameters
    ----------
    choice : None | False | 'robust' | 'standard'
        None / False  -- no scaling
        True / 'robust'   -- RobustScaler
        'standard'    -- StandardScaler
    """
    if choice is None or choice is False:
        return None
    if choice is True or choice == 'robust':
        return RobustScaler()
    if choice == 'standard':
        return StandardScaler()
    raise ValueError(
        f"Unknown scaler {choice!r}. Use None, 'robust', or 'standard'."
    )


def aggregate_per_sample(features, data_df, append_tmb=False,
                          scale_tokens=None, scale_aggregated=None):
    """
    Optionally scale token features, then mean+max pool per sample,
    then optionally scale the aggregated sample vectors.

    Parameters
    ----------
    features : np.ndarray, shape (n_tokens, d)
    data_df  : pd.DataFrame with a 'Tumor_Sample_Barcode' column
    append_tmb : bool
        If True, append log1p(token count) as an extra scalar.
    scale_tokens : None | 'robust' | 'standard'
        Scaler applied to token-level features before pooling.
        None (default) = no scaling. Also accepts bool True → 'robust'.
    scale_aggregated : None | 'robust' | 'standard'
        Scaler applied to aggregated sample-level vectors after pooling.
        None (default) = no scaling. Also accepts bool True → 'robust'.

    Returns
    -------
    agg_matrix : np.ndarray, shape (n_samples, 2*d [+ 1])
    sample_ids : np.ndarray of str, shape (n_samples,)
    """
    tok_scaler = _build_scaler(scale_tokens)
    if tok_scaler is not None:
        features = tok_scaler.fit_transform(features)

    df = data_df[['Tumor_Sample_Barcode']].copy()
    df['__idx'] = np.arange(len(df))
    groups = df.groupby('Tumor_Sample_Barcode')['__idx'].apply(list)

    sample_ids, agg_rows = [], []
    for sid, idxs in groups.items():
        f = features[idxs]
        row = np.concatenate([f.mean(0), f.max(0)])
        if append_tmb:
            row = np.append(row, np.log1p(len(idxs)))
        agg_rows.append(row)
        sample_ids.append(sid)

    agg_matrix = np.array(agg_rows, dtype=np.float32)
    agg_scaler = _build_scaler(scale_aggregated)
    if agg_scaler is not None:
        agg_matrix = agg_scaler.fit_transform(agg_matrix)
    return agg_matrix, np.array(sample_ids)


def load_and_aggregate_pkl(pkl_path, label, sample_filter=None, verbose=True, scale_tokens=None, scale_aggregated=None):
    """
    Load a features PKL, optionally filter samples, and aggregate per sample.

    Parameters
    ----------
    pkl_path : str
        Path to PKL with keys: train/valid_variant_features, train/valid_cna_features,
        train/valid_data_snv, train/valid_data_cna.
    label : str
        Display name printed during loading.
    sample_filter : callable or None
        If provided, called as sample_filter(barcode) -> bool.
        Only tokens whose barcode returns True are kept.
    verbose : bool

    Returns
    -------
    X          : np.ndarray, shape (n_samples, dims)  -- SNV+CNA concatenated
    sample_ids : np.ndarray of str
    """
    if verbose:
        print(f"  Loading {label}...")
    with open(pkl_path, 'rb') as f:
        d = pickle.load(f)

    snv_feat = np.vstack([d['train_variant_features'], d['valid_variant_features']])
    cna_feat = np.vstack([d['train_cna_features'],     d['valid_cna_features']])
    snv_data = pd.concat([d['train_data_snv'], d['valid_data_snv']], axis=0).reset_index(drop=True)
    cna_data = pd.concat([d['train_data_cna'], d['valid_data_cna']], axis=0).reset_index(drop=True)
    del d; gc.collect()

    if sample_filter is not None:
        snv_mask = np.array([sample_filter(s) for s in snv_data['Tumor_Sample_Barcode']])
        cna_mask = np.array([sample_filter(s) for s in cna_data['Tumor_Sample_Barcode']])
        snv_feat = snv_feat[snv_mask]; snv_data = snv_data[snv_mask].reset_index(drop=True)
        cna_feat = cna_feat[cna_mask]; cna_data = cna_data[cna_mask].reset_index(drop=True)

    snv_agg, snv_ids = aggregate_per_sample(snv_feat, snv_data, append_tmb=True, scale_tokens=scale_tokens, scale_aggregated=scale_aggregated)
    cna_agg, cna_ids = aggregate_per_sample(cna_feat, cna_data, append_tmb=True, scale_tokens=scale_tokens, scale_aggregated=scale_aggregated)
    del snv_feat, cna_feat; gc.collect()

    snv_map = {s: i for i, s in enumerate(snv_ids)}
    cna_map = {s: i for i, s in enumerate(cna_ids)}
    common  = sorted(set(snv_ids) & set(cna_map.keys()))

    X = np.concatenate([snv_agg[[snv_map[s] for s in common]],
                        cna_agg[[cna_map[s] for s in common]]], axis=1)
    if verbose:
        print(f"    {len(common)} samples × {X.shape[1]} dims")
    return X, np.array(common)


def load_type_map(meta_csv):
    """
    Build a dict mapping Tumor_Sample_Barcode -> cancer type from the SNV CSVs.
    Reads both train and valid splits.
    """
    rows = []
    for split in ['train', 'valid']:
        p = meta_csv.replace('train_data_snv', f'{split}_data_snv')
        try:
            rows.append(pd.read_csv(p, usecols=['Tumor_Sample_Barcode', 'type']))
        except FileNotFoundError:
            pass
    df = pd.concat(rows).drop_duplicates('Tumor_Sample_Barcode')
    return dict(zip(df['Tumor_Sample_Barcode'], df['type']))


def load_subtype_map(meta_csv, clinical_csv, cancer_type, subtype_col, categories):
    """
    Build a dict mapping Tumor_Sample_Barcode -> subtype for a specific cancer type.

    Parameters
    ----------
    meta_csv     : path to train_data_snv.csv (used to get all barcodes)
    clinical_csv : path to clinical CSV with Patient_ID and subtype_col columns
    cancer_type  : str, e.g. 'STAD' -- used to filter barcodes by type
    subtype_col  : str, column name in clinical_csv containing subtype labels
    categories   : list of str -- only keep samples whose subtype is in this list

    Returns
    -------
    dict: Tumor_Sample_Barcode -> subtype string
    """
    type_map = load_type_map(meta_csv)
    barcodes = [bc for bc, t in type_map.items() if t == cancer_type]

    clin = pd.read_csv(clinical_csv, usecols=['Patient_ID', subtype_col])
    clin = clin[clin[subtype_col].isin(categories)]
    patient_to_subtype = dict(zip(clin['Patient_ID'], clin[subtype_col]))

    subtype_map = {bc: patient_to_subtype[bc[:12]]
                   for bc in barcodes if bc[:12] in patient_to_subtype}
    counts = pd.Series(subtype_map.values()).value_counts().to_dict()
    print(f"  {cancer_type} subtype map: {counts}")
    return subtype_map
