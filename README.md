<p align="center">
  <img src="logo.png" alt="TESSERA logo" width="220">
</p>

<p align="center">
  <em>Tumour Embeddings via Self-Supervised Encoding and Reconstruction of Alterations</em><br>
  A foundation model for the cancer genome.
</p>

<p align="center">
  📄 <a href="https://www.biorxiv.org/content/10.64898/2026.05.27.728319v1">Preprint · bioRxiv, 2026</a>
</p>

---

TESSERA is a self-supervised foundation model jointly pretrained on somatic single-nucleotide variants (SNVs) and copy-number alterations (CNAs) from the TCGA Pan-Cancer Atlas. A single learned representation, produced once and reused without retraining, supports variant pathogenicity prediction, pan-cancer tumour-type classification, unsupervised molecular subtyping, prognostic stratification, and counterfactual treatment-effect estimation.

This repository contains the reference implementation, the pretrained-weights pointer, and the end-to-end analysis pipelines that accompany the TESSERA manuscript.

## Quick start

```bash
pip install tessera-foundation
```

*Typical install time on a modern desktop: ~2–5 minutes for the package itself. The pretrained weights (~185 MB, cached under `~/.cache/huggingface`) and the GRCh37 reference genome (~3 GB, cached under `~/.cache/tessera/ref_genomes`) are downloaded lazily on first call and reused thereafter.*

```python
import tessera, pandas as pd

# Bundled demo CSVs (~441 SNVs + ~344 CNAs of real anonymised TCGA-format rows).
# Replace these URLs with paths to your own data in the same schema:
#   SNV cols: Tumor_Sample_Barcode, Chromosome, Start_Position,
#             Reference_Allele, Tumor_Seq_Allele2, vaf
#   CNA cols: Tumor_Sample_Barcode, Chromosome, Start, End, Segment_Mean
DEMO = "https://raw.githubusercontent.com/JW-Sidhom-Lab/tessera/main/inference_api"
snv_df = pd.read_csv(f"{DEMO}/example_snv.csv")
cna_df = pd.read_csv(f"{DEMO}/example_cna.csv")

result = tessera.featurize(
    snv_df=snv_df, cna_df=cna_df,
    variant="joint_snv_cna_noloh",        # or "joint_snv_cna" (with-LoH)
    from_assembly="GRCh37",               # "GRCh38" triggers UCSC liftover
    quantile_normalize_to_tcga=False,     # set True for panel/cell-line data
)

result.snv_features      # (n_variants, 1169)  per-variant embeddings
result.cna_features      # (n_segments, 688)   per-segment embeddings
result.snv_table         # the post-liftover SNV rows that produced snv_features
result.cna_table         # the post-liftover CNA rows that produced cna_features
```

Row order in `snv_features` matches `snv_table`, and likewise for CNA. Variants that fail tokenization (non-SBS, missing reference base, off-genome coordinates after liftover) and segments rejected by basic validation are dropped before inference, so `snv_table` / `cna_table` are the authoritative pairing back to the input rows actually embedded.

First call downloads the requested model variant from Hugging Face Hub (~185 MB, cached under `~/.cache/huggingface`) and, on first featurisation, the GRCh37 reference genome (~3 GB, cached under `~/.cache/tessera/ref_genomes`). Both are reused on later calls. To pre-provision the genome (or point at an existing copy on an offline/shared machine), set `TESSERA_REF_GENOME_DIR` to a directory containing the FASTA, or run `tessera/ref_genomes/download_ref_genomes.sh`. On a 2024 MacBook Pro (Apple M3), first-call featurisation of the demo CSVs above (genome already cached) completes in ~30–60 seconds; subsequent calls within the same session run in under 5 seconds.

**CSV column conventions:**

- **SNV**: `Tumor_Sample_Barcode`, `Chromosome` (no `chr` prefix), `Start_Position`, `Reference_Allele`, `Tumor_Seq_Allele2`, plus either `vaf` or both `t_alt_count` + `t_ref_count`. Single-base substitutions only.
- **CNA**: `Tumor_Sample_Barcode`, `Chromosome`, `Start`, `End`, `Segment_Mean` (log2 ratio); optional `LOH` column triggers the with-LoH variant.

### When to set `quantile_normalize_to_tcga=True`

TESSERA was pretrained on TCGA whole-exome ABSOLUTE Segment_Means (median 0.000, IQR [0, +0.51]). Inputs whose log2-ratio distribution differs should be rank-mapped onto the TCGA reference before inference.

| Input type | Setting | Why |
|---|---|---|
| TCGA-like whole-exome ABSOLUTE | `False` (default) | Same distribution the model was pretrained on. |
| Panel sequencing (MSK-IMPACT, MSK-CHORD, GENIE) | **`True`** | Panel coverage compresses log2-ratios toward zero (KS = 0.38 vs TCGA). |
| Cell-line data (DepMap, CCLE) | **`True`** | Raw log2-ratios are right-shifted; DepMap median ≈ +1.0 vs TCGA's 0.0 (KS = 0.72). |

The bundled reference (`tessera/data/cna_sorted.npy`, 7 MB, 1.8 M segments) is loaded automatically when `True`. The helper `tessera.data.preprocessing.quantile_normalize_to_tcga` is also exposed if you'd rather pre-normalize.

### Lower-level building blocks

```python
from tessera import load_pretrained, lift_snv, lift_cna

model = load_pretrained("joint_snv_cna_noloh")          # download + instantiate
snv_df, _ = lift_snv(snv_df, from_assembly="GRCh38")    # identity for GRCh37
result = model.featurize(snv_df=snv_df, cna_df=cna_df)  # reuse without re-downloading
```

UCSC chain files for liftover are downloaded on first use to `~/.cache/pyliftover/`; offline environments can supply a local file via `chain_file=` or the `TESSERA_LIFTOVER_CHAIN` env var.

## Pre-computed TCGA Pan-Cancer features

TESSERA features for the full TCGA Pan-Cancer Atlas (from the joint SNV+CNA InfoNCE-aligned model used in Figs. 4-6 of the manuscript) are deposited on Zenodo and can be downloaded directly. This avoids installing the package or running inference if you only need features for TCGA samples.

**DOI:** [10.5281/zenodo.20419467](https://doi.org/10.5281/zenodo.20419467)

The deposit contains three HDF5 files plus a README documenting the schema:

| File | Contents |
|---|---|
| `snv_per_variant.h5` | Per-variant SNV embeddings (1{,}921{,}403 rows × 1{,}169 dims) with full variant metadata. |
| `cna_per_segment.h5` | Per-segment CNA embeddings (1{,}823{,}050 rows × 688 dims) with full segment metadata. |
| `per_sample_aggregated.h5` | Per-sample mean and max pools of per-token features, by modality, plus per-sample token counts. This is the standard input form the downstream manuscript analyses consume. |

Download:

```bash
mkdir -p tessera_features && cd tessera_features
for f in snv_per_variant.h5 cna_per_segment.h5 per_sample_aggregated.h5 README.md; do
  curl -L -o "$f" "https://zenodo.org/records/20419467/files/$f"
done
```

Load the per-sample features (joint-modality, the manuscript form):

```python
import h5py, numpy as np

with h5py.File('per_sample_aggregated.h5', 'r') as f:
    snv_mean = f['snv/mean'][:]                   # (n_snv_samples, 1169) float32
    snv_max  = f['snv/max'][:]                    # (n_snv_samples, 1169)
    cna_mean = f['cna/mean'][:]                   # (n_cna_samples,  688)
    cna_max  = f['cna/max'][:]                    # (n_cna_samples,  688)
    snv_sample_id = f['snv/sample_id'][:].astype(str)
    cna_sample_id = f['cna/sample_id'][:].astype(str)

joint = np.intersect1d(snv_sample_id, cna_sample_id)
idx_snv = np.searchsorted(snv_sample_id, joint)
idx_cna = np.searchsorted(cna_sample_id, joint)
features = np.concatenate([
    snv_mean[idx_snv], snv_max[idx_snv],
    cna_mean[idx_cna], cna_max[idx_cna],
], axis=1)                                        # (n_joint_samples, 3714)
```

The README inside the archive documents the full HDF5 schema for all three files (per-variant SNV metadata, per-segment CNA metadata, per-sample token counts, per-modality RobustScaler parameters fitted on the same TCGA Pan-Cancer split).

## Reproducing the manuscript

For training, downstream analyses, and figure generation, clone the repo:

```bash
git clone https://github.com/JW-Sidhom-Lab/tessera.git
cd tessera
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash tessera/ref_genomes/download_ref_genomes.sh
```

*Typical install time on a modern desktop: ~10–15 minutes for the Python dependencies (TensorFlow is the largest), plus a one-time ~10 minutes for the GRCh37 reference-genome download (~3 GB).*

The pipeline runs in three stages:

1. **Data preparation** ([`data/`](data/README.md)): per-cohort download instructions, source-table provenance, and the builders that turn raw releases into the analysis-ready CSVs.
2. **Foundation-model pretraining** ([`scripts/tcga_pancan_*/`](scripts/README.md)): trains the SNV models, the CNA models, and the joint SNV+CNA InfoNCE-aligned foundation model on the TCGA Pan-Cancer Atlas.
3. **Downstream analyses** ([`scripts/`](scripts/README.md)): variant-pathogenicity calibration, cross-platform validation, tumour-type classification, prognostic stratification, doubly-robust counterfactual treatment-effect estimation, and cell-line transfer.

[`scripts/README.md`](scripts/README.md) and [`data/README.md`](data/README.md) hold the per-directory tables linking each script and cohort to the relevant manuscript section.

## Repository layout

```
tessera/
├── tessera/                        # foundation-model package
│   ├── base.py                     # BaseModel: shared data + training infrastructure
│   ├── input_keys.py               # input-key helpers
│   ├── model.py                    # TESSERA: foundation-model class
│   ├── data/
│   │   └── preprocessing.py        # SNV/CNA tokenization, FASTA lookup, sample bagging
│   ├── layers/                     # custom Keras layers (attention, masking, MIL, ...)
│   ├── training/                   # training utilities (callbacks, losses, schedules)
│   └── ref_genomes/                # reference-genome download script + indices
├── data/                           # per-cohort data preparation pipelines (data/README.md)
├── scripts/                        # analysis pipelines backing the manuscript figures (scripts/README.md)
└── README.md
```

## Citing TESSERA

If you use TESSERA in your work, please cite:

> Sidhom J.W., Baras A.S., Elemento O., Shah M.A. *A foundation model for the cancer genome.* bioRxiv, 2026. https://www.biorxiv.org/content/10.64898/2026.05.27.728319v1

```bibtex
@article{sidhom2026tessera,
  title   = {A foundation model for the cancer genome},
  author  = {Sidhom, John-William and Baras, Alex S. and Elemento, Olivier and Shah, Manish A.},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.64898/2026.05.27.728319},
  url     = {https://www.biorxiv.org/content/10.64898/2026.05.27.728319v1}
}
```

## License

This repository is distributed under the **PolyForm Noncommercial License 1.0.0** (see [`LICENSE`](LICENSE)). Use is permitted for academic research, education, public-research-organization use, and personal experimentation; commercial use is not permitted without a separate license. Pretrained foundation-model weights are released on the Hugging Face Hub under **CC-BY-NC-4.0** (non-commercial, attribution required). Pretrained weights for downstream clinical task heads (CRC and PDAC treatment-effect models) remain available on request under a Data Use Agreement. Patents covering clinical applications of TESSERA are assigned to NewYork-Presbyterian; commercial licensing inquiries should be directed to NYP's technology transfer office.

## Lab

TESSERA is developed in the [JW Sidhom Lab](https://github.com/JW-Sidhom-Lab) at Weill Cornell Medicine.

For questions, collaborations, or commercial-licensing enquiries, contact the corresponding author.
