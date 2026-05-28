# TESSERA features for the TCGA Pan-Cancer Atlas

**Per-variant, per-segment, and per-sample embeddings produced by the
TESSERA joint SNV+CNA foundation model, computed on the full TCGA
Pan-Cancer Atlas cohort.**

This deposit accompanies the manuscript *"A Foundation Model for the
Cancer Genome"* (Sidhom et al.) and ships the foundation-model outputs
that back Figs. 4-6 and the associated downstream analyses. The features
are released so that downstream analyses can build on the TESSERA
representation of TCGA without re-running pretraining or inference.

- Source code:           https://github.com/JW-Sidhom-Lab/tessera
- Pretrained weights:    https://huggingface.co/JW-Sidhom-Lab/tessera-foundation

## What is in this deposit

| File | Description |
|---|---|
| `snv_per_variant.h5` | Per-variant SNV embeddings (1{,}921{,}403 rows × 1{,}169 dims) with full variant metadata. |
| `cna_per_segment.h5` | Per-segment CNA embeddings (1{,}823{,}050 rows × 688 dims) with full segment metadata. |
| `per_sample_aggregated.h5` | Per-sample mean and max pools of the per-token features, by modality, plus per-sample token counts. This is the standard input form that the downstream manuscript analyses consume. |
| `README.md` | This file. |

(Row counts are approximate; exact counts are encoded in each file's
top-level attributes.)

## Quick start (Python)

```python
import h5py
import numpy as np
import pandas as pd

# Per-variant SNV features
with h5py.File('snv_per_variant.h5', 'r') as f:
    snv_features = f['features'][:]                                 # (N, 1169) float32
    snv_meta = pd.DataFrame({
        col: f[f'metadata/{col}'][:] for col in f['metadata'].keys()
    })
    # Decode any byte-string columns to Python strings
    for col in snv_meta.select_dtypes(include='object').columns:
        snv_meta[col] = snv_meta[col].astype(str)

# Per-sample aggregated features (RobustScaler-then-pool, manuscript form)
with h5py.File('per_sample_aggregated.h5', 'r') as f:
    snv_mean = f['snv/mean'][:]            # (n_snv_samples, 1169) float32
    snv_max  = f['snv/max'][:]             # (n_snv_samples, 1169) float32
    cna_mean = f['cna/mean'][:]            # (n_cna_samples,  688) float32
    cna_max  = f['cna/max'][:]             # (n_cna_samples,  688) float32
    snv_sample_id = f['snv/sample_id'][:].astype(str)
    cna_sample_id = f['cna/sample_id'][:].astype(str)
    n_variants = f['snv/n_variants'][:]
    n_segments = f['cna/n_segments'][:]
    snv_center, snv_scale = f['snv/scaler/center'][:], f['snv/scaler/scale'][:]
    cna_center, cna_scale = f['cna/scaler/center'][:], f['cna/scaler/scale'][:]

# Concatenate to the standard 3714-dim per-sample feature used by the
# manuscript downstream analyses (joint-modality intersection)
import numpy as np
joint = np.intersect1d(snv_sample_id, cna_sample_id)
idx_snv = np.searchsorted(snv_sample_id, joint)
idx_cna = np.searchsorted(cna_sample_id, joint)
per_sample = np.concatenate([
    snv_mean[idx_snv], snv_max[idx_snv],
    cna_mean[idx_cna], cna_max[idx_cna],
], axis=1)                                  # (n_joint_samples, 2*(1169+688))
```

## File format details

All three files are HDF5 v1.10+, gzip-compressed at level 4 with the
shuffle filter, float32 features.

### `snv_per_variant.h5`

```
/features            (N_variants, 1169)  float32
/metadata/
    Tumor_Sample_Barcode    UTF-8 string  TCGA sample barcode
    Hugo_Symbol             UTF-8 string  gene symbol
    Chromosome              UTF-8 string  chromosome label
    Start_Position          int64         1-based position
    Reference_Allele        UTF-8 string  reference allele
    Tumor_Seq_Allele2       UTF-8 string  variant allele
    vaf                     float64       variant allele frequency
    VARIANT_CLASS           UTF-8 string  SNV / DNP / insertion etc.
    HGVSp_Short             UTF-8 string  protein change in HGVS short form
    bcr_patient_barcode     UTF-8 string  TCGA patient barcode (first 12 chars)
    type                    UTF-8 string  TCGA cohort code (BRCA, GBM, ...)
    split                   UTF-8 string  'train' or 'valid' (original model split)
```

Top-level HDF5 attributes record `model`, `source`, `n_rows`,
`n_features`, `feature_dtype`, `creation_utc`, `format_version`.

### `cna_per_segment.h5`

```
/features            (N_segments, 688)  float32
/metadata/
    Tumor_Sample_Barcode    UTF-8 string
    Chromosome              int64
    Start                   int64         segment start (hg19, 1-based)
    End                     int64         segment end
    Modal_HSCN_1            int64         major allele integer copy number
    Modal_HSCN_2            int64         minor allele integer copy number
    Modal_Total_CN          int64         total integer copy number
    Segment_Mean            float64       log2 ratio
    Breakpoint_Density      float64
    Delta_CN_prev           float64       integer CN change vs preceding segment
    Delta_CN_next           float64       integer CN change vs next segment
    Delta_CN                float64
    LOH                     bool          loss of heterozygosity
    split                   UTF-8 string  'train' or 'valid'
```

### `per_sample_aggregated.h5`

```
/snv/
    mean             (n_snv_samples, 1169)  float32  mean pool of SCALED per-token features
    max              (n_snv_samples, 1169)  float32  max  pool of SCALED per-token features
    n_variants       (n_snv_samples,)       int64    variants contributing per sample
    sample_id        (n_snv_samples,)       UTF-8 string
    scaler/
        center       (1169,)               float64  per-feature median (RobustScaler)
        scale        (1169,)               float64  per-feature IQR    (RobustScaler)

/cna/
    mean             (n_cna_samples, 688)   float32
    max              (n_cna_samples, 688)   float32
    n_segments       (n_cna_samples,)       int64
    sample_id        (n_cna_samples,)       UTF-8 string
    scaler/
        center       (688,)                float64
        scale        (688,)                float64
```

**Aggregation pipeline (matches `scripts/predictive_bm/core/features.py:47-66`):**

1. Take the raw per-token features from `snv_per_variant.h5` / `cna_per_segment.h5`.
2. Fit `sklearn.preprocessing.RobustScaler` per modality on the full
   per-token matrix (median-centered, IQR-scaled, per feature dimension).
3. Mean and max pool the scaled per-token features within each sample.

The `scaler/center` and `scaler/scale` datasets store the parameters of
this **pre-aggregation** RobustScaler so downstream users can (a) reverse
the scaling, or (b) apply the same TCGA-fit scaler to per-token features
from a new cohort (for example MSK-IMPACT-derived features for clinical
application work). Concretely:

```python
# Apply the same scaler to features from a new cohort
new_features_scaled = (new_features - snv_center) / snv_scale
```

Sample-ID lists for SNV and CNA differ slightly: not every TCGA sample has
both modalities profiled. To reproduce the joint-modality cohort used in
Figs. 4-6 of the manuscript, take the intersection of `snv/sample_id` and
`cna/sample_id`, then concatenate `[snv/mean, snv/max, cna/mean, cna/max]`
along the feature axis to obtain the 3,714-dim per-sample feature that
every manuscript downstream analysis consumes as **input**.

### Post-aggregation processing

The per-sample features in this deposit stop after the mean+max pool
step. Before feeding them into any downstream ML or visualization
pipeline, apply a post-aggregation scaler of your choice
(`sklearn.preprocessing.StandardScaler` or `RobustScaler` are both
appropriate; pick whichever matches your downstream estimator's
assumptions). For reference, the manuscript's tumour-type classifier
(Fig. 4) and prognostic analyses (Fig. 5) applied `RobustScaler` at this
step.

## How these features were produced

The features come from the joint SNV+CNA TESSERA foundation model,
pretrained on the TCGA Pan-Cancer Atlas through (i) masked-token
reconstruction within each modality and (ii) a bidirectional InfoNCE
contrastive objective that aligns per-sample SNV and CNA embeddings.
Concrete model architecture, training schedule, hyperparameters, and
inference protocol are detailed in the manuscript Methods and in the
public source code at https://github.com/JW-Sidhom-Lab/tessera (analysis
pipeline at `scripts/tcga_pancan_snv_cna/`).

The model checkpoint that produced these features is the variant
designated `InfoNCE_per_sample_loss` in the source code, corresponding
to the canonical joint model whose embeddings appear throughout the
manuscript's downstream analyses.

## Citation

If you use these features in your work, please cite this Zenodo deposit:

> Sidhom JW et al. *TESSERA features for the TCGA Pan-Cancer Atlas*
> [data set]. Zenodo, 2026. DOI: 10.5281/zenodo.20419467.

The accompanying manuscript, *A Foundation Model for the Cancer Genome*
(Sidhom et al.), is linked from this deposit's related-identifiers
metadata once the preprint is posted; please also cite it when available.

## License

These features are released under the **Creative Commons
Attribution-NonCommercial 4.0 International (CC-BY-NC-4.0)** license,
matching the licence of the pretrained model weights on Hugging Face Hub.
Use is permitted for academic research, education, and non-commercial
projects with attribution; commercial use requires a separate licence.

## Contact

For questions, bug reports, or data-use enquiries, contact the
corresponding author of the manuscript or open an issue on the source
repository at https://github.com/JW-Sidhom-Lab/tessera .
