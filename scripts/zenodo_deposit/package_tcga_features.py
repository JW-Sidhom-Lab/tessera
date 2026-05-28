"""Package TCGA Pan-Cancer Atlas TESSERA features for the Zenodo deposit.

Reads the cached joint SNV+CNA InfoNCE multimodal features pkl (produced by
the joint pretraining pipeline in ``scripts/tcga_pancan_snv_cna/``) and writes
three HDF5 files for public release on Zenodo:

  snv_per_variant.h5         per-variant SNV embeddings + metadata
  cna_per_segment.h5         per-segment CNA embeddings + metadata
  per_sample_aggregated.h5   mean+max pools per modality + token counts

The deposit ships the canonical joint SNV+CNA InfoNCE-aligned model used in
Figs. 4-6 of the manuscript. Train and validation rows of the model split
are merged into a single deposit (split membership is recorded as a
``split`` metadata column so downstream users can replicate the original
partition if they need to).

Memory design
-------------
The job is split into two phases that never hold more than one large array
at a time, so peak RSS stays bounded even though the source pkl is large:

  Phase 1 (per-token): load the pkl, stream each modality's train+valid
    arrays directly into a resizable HDF5 dataset via two slice writes (no
    in-memory concatenated copy), then free the pkl.
  Phase 2 (per-sample): read each modality's per-token features back from
    its HDF5 file ONE AT A TIME, RobustScale in place in column blocks
    (no full scaled copy), mean+max pool per sample, free, repeat.

The script is also resumable: if a per-token HDF5 file already exists and is
valid, Phase 1 skips re-writing it (and skips the pkl load entirely if both
per-token files are already present).

Inputs (env-var override or default):
  PKL_PATH    path to the multimodal features pkl
  OUTPUT_DIR  directory to write the 3 HDF5 files + the deposit README

Usage:
    python3 package_tcga_features.py
"""
from __future__ import annotations

import os
import resource
import shutil
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


HERE = Path(__file__).parent.resolve()
DEFAULT_PKL = (HERE / '..' / 'tcga_pancan_snv_cna' / 'multimodal_features'
               / 'TCGA_SNV_CNA_InfoNCE_per_sample_loss_multimodal_features.pkl')
PKL_PATH   = Path(os.environ.get('PKL_PATH', DEFAULT_PKL))
OUTPUT_DIR = Path(os.environ.get('OUTPUT_DIR', HERE / 'zenodo_release'))
DEPOSIT_README_SRC = HERE / 'DEPOSIT_README.md'

MODEL_TAG = ('TESSERA joint SNV+CNA InfoNCE-aligned '
             '(per_sample_loss variant); manuscript Figs. 4-6')

# RobustScaler percentiles are computed in blocks of this many feature
# columns to bound the np.percentile temporary (the full-matrix version
# allocates a complete copy, which is what caused an earlier OOM).
SCALE_COL_CHUNK = 256

# Columns are written in this order; track_order=True on the HDF5 group
# preserves it for human-readable inspection (identifiers -> gene -> locus
# -> alleles -> annotation -> split).
SNV_METADATA_COLS = [
    'Tumor_Sample_Barcode', 'bcr_patient_barcode', 'type',   # identifiers
    'Hugo_Symbol',                                           # gene
    'Chromosome', 'Start_Position',                          # genomic locus
    'Reference_Allele', 'Tumor_Seq_Allele2',                 # alleles
    'VARIANT_CLASS', 'HGVSp_Short', 'vaf',                   # annotation
]
# identifiers -> locus -> log2 ratio -> integer copy-number states -> LOH
# -> breakpoint/delta context -> split.
CNA_METADATA_COLS = [
    'Tumor_Sample_Barcode',                                  # identifier
    'Chromosome', 'Start', 'End',                            # genomic locus
    'Segment_Mean',                                          # log2 ratio
    'Modal_HSCN_1', 'Modal_HSCN_2', 'Modal_Total_CN',        # integer copy number
    'LOH',                                                   # loss of heterozygosity
    'Breakpoint_Density', 'Delta_CN_prev', 'Delta_CN', 'Delta_CN_next',  # context
]


def _peak_gb() -> float:
    """Peak resident set size so far, in GB (macOS reports bytes, Linux KB)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS ru_maxrss is bytes; Linux is kilobytes.
    if sys.platform == 'darwin':
        return rss / 1e9
    return rss / 1e6


def _log(msg: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}  peak {_peak_gb():5.1f} GB]  {msg}',
          flush=True)


def _to_h5_strings(arr) -> np.ndarray:
    """Object array (possibly with NaN) -> object-dtype array of Python str,
    which is what h5py's variable-length string dtype requires (a fixed-width
    ``U<n>`` numpy array is rejected)."""
    return np.array(['' if pd.isna(x) else str(x) for x in arr], dtype=object)


def _decode(arr) -> np.ndarray:
    """Bytes/str array from an HDF5 string dataset -> numpy array of str."""
    return np.array([x.decode('utf-8') if isinstance(x, bytes) else str(x)
                     for x in arr])


def _h5_valid(path: Path) -> bool:
    """True if a per-token HDF5 exists with a non-empty /features dataset."""
    if not path.exists():
        return False
    try:
        with h5py.File(path, 'r') as f:
            return 'features' in f and f['features'].shape[0] > 0
    except Exception:
        return False


def _load_pkl() -> dict:
    import pickle
    _log(f'Loading {PKL_PATH} ...')
    with open(PKL_PATH, 'rb') as f:
        d = pickle.load(f)
    for k, v in d.items():
        shape = getattr(v, 'shape', None)
        _log(f'  {k:32s} {("shape=" + str(shape)) if shape is not None else type(v).__name__}')
    return d


def _prep_snv_metadata(df: pd.DataFrame, split: str) -> pd.DataFrame:
    out = df.copy()
    out['bcr_patient_barcode'] = (
        out['bcr_patient_barcode_x'].fillna(out['bcr_patient_barcode_y']))
    out = out[SNV_METADATA_COLS].copy()
    out['split'] = split
    return out


def _prep_cna_metadata(df: pd.DataFrame, split: str) -> pd.DataFrame:
    out = df[CNA_METADATA_COLS].copy()
    out['split'] = split
    return out


def _write_metadata_group(f: h5py.File, metadata: pd.DataFrame) -> None:
    # track_order=True preserves the column insertion order (the logical
    # human-readable order in SNV/CNA_METADATA_COLS) instead of h5py's
    # default alphabetical member ordering.
    meta_group = f.create_group('metadata', track_order=True)
    for col in metadata.columns:
        data = metadata[col].values
        if data.dtype == object:
            meta_group.create_dataset(
                col, data=_to_h5_strings(data),
                dtype=h5py.string_dtype(encoding='utf-8'),
                compression='gzip', compression_opts=4)
        elif data.dtype == bool:
            meta_group.create_dataset(
                col, data=data.astype(np.uint8),
                compression='gzip', compression_opts=4)
        else:
            meta_group.create_dataset(
                col, data=data, compression='gzip', compression_opts=4)


def _write_per_token_streaming(out_path: Path,
                               train_feat: np.ndarray, valid_feat: np.ndarray,
                               metadata: pd.DataFrame,
                               modality: str) -> None:
    """Write per-token features + metadata without an in-memory concatenated
    copy: train and valid are streamed into one resizable dataset via two
    slice writes. Peak footprint is just the (already-loaded) train + valid
    arrays plus small h5py compression buffers."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_train, n_valid = len(train_feat), len(valid_feat)
    n_total, dim = n_train + n_valid, train_feat.shape[1]
    _log(f'Writing {out_path.name} ({n_total:,} rows x {dim} features, '
         f'float32, gzip-4, streamed train+valid) ...')
    with h5py.File(out_path, 'w', track_order=True) as f:
        dset = f.create_dataset(
            'features', shape=(n_total, dim), dtype=np.float32,
            compression='gzip', compression_opts=4, chunks=True, shuffle=True)
        dset[:n_train] = train_feat.astype(np.float32, copy=False)
        dset[n_train:] = valid_feat.astype(np.float32, copy=False)
        _write_metadata_group(f, metadata)
        f.attrs['model'] = MODEL_TAG
        f.attrs['modality'] = modality
        f.attrs['source'] = 'TCGA Pan-Cancer Atlas (MC3 SNV + ABSOLUTE CNA)'
        f.attrs['n_rows'] = int(n_total)
        f.attrs['n_features'] = int(dim)
        f.attrs['feature_dtype'] = 'float32'
        f.attrs['creation_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ',
                                                  time.gmtime())
        f.attrs['format_version'] = '1'


def _robust_scale_inplace(features: np.ndarray,
                          col_chunk: int = SCALE_COL_CHUNK) -> tuple:
    """Per-column RobustScaler applied in place, with the percentile computed
    in column blocks so the temporary is (n, col_chunk) rather than (n, d).

    Matches sklearn.preprocessing.RobustScaler defaults: center = median,
    scale = IQR (75th - 25th, linear interpolation), and scale set to 1.0
    where the IQR is zero. Returns (center, scale) as float64 arrays.
    """
    n, d = features.shape
    center = np.empty(d, dtype=np.float64)
    scale = np.empty(d, dtype=np.float64)
    for j0 in range(0, d, col_chunk):
        j1 = min(j0 + col_chunk, d)
        q1, med, q3 = np.percentile(features[:, j0:j1], [25, 50, 75], axis=0)
        iqr = q3 - q1
        iqr[iqr == 0.0] = 1.0
        center[j0:j1] = med
        scale[j0:j1] = iqr
    # In-place transform; cast scaler params to float32 to keep the op in place.
    features -= center.astype(np.float32)
    features /= scale.astype(np.float32)
    return center, scale


def _aggregate_per_sample(features: np.ndarray,
                          sample_ids: np.ndarray) -> dict:
    """Mean + max pool per sample (features assumed already RobustScaled).
    Sorts once and walks contiguous spans."""
    order = np.argsort(sample_ids, kind='stable')
    sorted_ids = sample_ids[order]
    sorted_feat = features[order]
    unique_ids, first_idx, counts = np.unique(
        sorted_ids, return_index=True, return_counts=True)
    n_samples, n_features = len(unique_ids), features.shape[1]
    means = np.zeros((n_samples, n_features), dtype=np.float32)
    maxes = np.zeros((n_samples, n_features), dtype=np.float32)
    for i in range(n_samples):
        s, c = first_idx[i], counts[i]
        block = sorted_feat[s:s + c]
        means[i] = block.mean(axis=0)
        maxes[i] = block.max(axis=0)
    return {'mean': means, 'max': maxes,
            'n_tokens': counts.astype(np.int64),
            'sample_ids': unique_ids.astype(str)}


def _aggregate_modality_from_h5(h5_path: Path) -> tuple:
    """Read one modality's per-token features back from its HDF5 (one modality
    at a time), RobustScale in place, mean+max pool per sample. Peak memory is
    one modality's feature matrix plus a small percentile temporary."""
    _log(f'Loading {h5_path.name} features for per-sample aggregation ...')
    with h5py.File(h5_path, 'r') as f:
        features = f['features'][:]
        sample_ids = _decode(f['metadata/Tumor_Sample_Barcode'][:])
    _log(f'  RobustScaling {h5_path.name} per-token (in place, '
         f'{SCALE_COL_CHUNK}-col blocks) ...')
    center, scale = _robust_scale_inplace(features)
    _log(f'  Mean+max pooling per sample ...')
    agg = _aggregate_per_sample(features, sample_ids)
    del features
    return agg, (center, scale)


def _write_per_sample(out_path: Path, snv_agg: dict, cna_agg: dict,
                      snv_scaler: tuple, cna_scaler: tuple) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_snv = len(snv_agg['sample_ids'])
    n_cna = len(cna_agg['sample_ids'])
    n_joint = len(set(snv_agg['sample_ids']) & set(cna_agg['sample_ids']))
    _log(f'Writing {out_path.name}: SNV n={n_snv:,}, CNA n={n_cna:,}, '
         f'joint n={n_joint:,}')
    snv_center, snv_scale = snv_scaler
    cna_center, cna_scale = cna_scaler
    with h5py.File(out_path, 'w', track_order=True) as f:
        for tag, agg, count_name, center, scale in [
            ('snv', snv_agg, 'n_variants', snv_center, snv_scale),
            ('cna', cna_agg, 'n_segments', cna_center, cna_scale),
        ]:
            g = f.create_group(tag, track_order=True)
            g.create_dataset('mean', data=agg['mean'], compression='gzip',
                             compression_opts=4, chunks=True, shuffle=True)
            g.create_dataset('max', data=agg['max'], compression='gzip',
                             compression_opts=4, chunks=True, shuffle=True)
            g.create_dataset(count_name, data=agg['n_tokens'],
                             compression='gzip', compression_opts=4)
            g.create_dataset('sample_id', data=_to_h5_strings(agg['sample_ids']),
                             dtype=h5py.string_dtype(encoding='utf-8'),
                             compression='gzip', compression_opts=4)
            sc = g.create_group('scaler', track_order=True)
            sc.create_dataset('center', data=center, compression='gzip',
                              compression_opts=4)
            sc.create_dataset('scale', data=scale, compression='gzip',
                              compression_opts=4)
            sc.attrs['type'] = 'sklearn.preprocessing.RobustScaler'
            sc.attrs['note'] = ('Median (center) and IQR (scale) per per-token '
                                'feature, fit on the full per-token matrix '
                                'before mean+max pooling.')
        f.attrs['model'] = MODEL_TAG
        f.attrs['source'] = 'TCGA Pan-Cancer Atlas (MC3 SNV + ABSOLUTE CNA)'
        f.attrs['n_snv_samples'] = int(n_snv)
        f.attrs['n_cna_samples'] = int(n_cna)
        f.attrs['n_joint_samples'] = int(n_joint)
        f.attrs['feature_dtype'] = 'float32'
        f.attrs['snv_feature_dim'] = int(snv_agg['mean'].shape[1])
        f.attrs['cna_feature_dim'] = int(cna_agg['mean'].shape[1])
        f.attrs['aggregation'] = (
            'Per-modality RobustScaler fit on the full per-token matrix, then '
            'mean + max pool within each sample on the scaled values; matches '
            'scripts/predictive_bm/core/features.py. Concatenate [snv/mean, '
            'snv/max, cna/mean, cna/max] on the joint-modality sample '
            'intersection to reconstruct the per-sample input.')
        f.attrs['creation_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ',
                                                  time.gmtime())
        f.attrs['format_version'] = '1'


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    snv_h5 = OUTPUT_DIR / 'snv_per_variant.h5'
    cna_h5 = OUTPUT_DIR / 'cna_per_segment.h5'

    # --- Phase 1: per-token HDF5 (resumable; only load pkl if needed) ---
    need_snv = not _h5_valid(snv_h5)
    need_cna = not _h5_valid(cna_h5)
    if need_snv or need_cna:
        if not PKL_PATH.exists():
            sys.exit(f'PKL_PATH does not exist: {PKL_PATH}')
        bundle = _load_pkl()
        if need_snv:
            _write_per_token_streaming(
                snv_h5,
                bundle.pop('train_variant_features'),
                bundle.pop('valid_variant_features'),
                pd.concat([_prep_snv_metadata(bundle['train_data_snv'], 'train'),
                           _prep_snv_metadata(bundle['valid_data_snv'], 'valid')],
                          ignore_index=True),
                modality='SNV')
        else:
            _log(f'{snv_h5.name} already valid - skipping per-token write')
        if need_cna:
            _write_per_token_streaming(
                cna_h5,
                bundle.pop('train_cna_features'),
                bundle.pop('valid_cna_features'),
                pd.concat([_prep_cna_metadata(bundle['train_data_cna'], 'train'),
                           _prep_cna_metadata(bundle['valid_data_cna'], 'valid')],
                          ignore_index=True),
                modality='CNA')
        else:
            _log(f'{cna_h5.name} already valid - skipping per-token write')
        del bundle
    else:
        _log('Both per-token HDF5 files already valid - skipping pkl load.')

    # --- Phase 2: per-sample aggregates (read back from HDF5, one at a time) ---
    snv_agg, snv_scaler = _aggregate_modality_from_h5(snv_h5)
    cna_agg, cna_scaler = _aggregate_modality_from_h5(cna_h5)
    _write_per_sample(OUTPUT_DIR / 'per_sample_aggregated.h5',
                      snv_agg, cna_agg, snv_scaler, cna_scaler)

    # --- Deposit README ---
    if DEPOSIT_README_SRC.exists():
        shutil.copy2(DEPOSIT_README_SRC, OUTPUT_DIR / 'README.md')
        _log(f'Copied deposit README to {OUTPUT_DIR / "README.md"}')

    # --- Size report ---
    _log('Deposit summary:')
    total = 0
    for p in sorted(OUTPUT_DIR.iterdir()):
        sz = p.stat().st_size
        total += sz
        _log(f'  {p.name:32s} {sz / 1e9:7.2f} GB')
    _log(f'  {"TOTAL":32s} {total / 1e9:7.2f} GB')


if __name__ == '__main__':
    main()
