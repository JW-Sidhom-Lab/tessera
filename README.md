<p align="center">
  <img src="logo.png" alt="TESSERA logo" width="220">
</p>

<p align="center">
  <em>Tumour Embeddings via Self-Supervised Encoding and Reconstruction of Alterations</em><br>
  A foundation model for the cancer genome.
</p>

---

TESSERA is a self-supervised foundation model jointly pretrained on somatic single-nucleotide variants (SNVs) and copy-number alterations (CNAs) from the TCGA Pan-Cancer Atlas. A single learned representation, produced once and reused without retraining, supports variant pathogenicity prediction, pan-cancer tumour-type classification, unsupervised molecular subtyping, prognostic stratification, and counterfactual treatment-effect estimation.

This repository contains the reference implementation, the pretrained-weights pointer, and the end-to-end analysis pipelines that accompany the TESSERA manuscript.

## Quick start

```bash
pip install tessera-foundation
```

```python
import tessera, pandas as pd

snv_df = pd.read_csv("snv.csv")    # cols: Tumor_Sample_Barcode, Chromosome,
                                   # Start_Position, Reference_Allele,
                                   # Tumor_Seq_Allele2, vaf
cna_df = pd.read_csv("cna.csv")    # cols: Tumor_Sample_Barcode, Chromosome,
                                   # Start, End, Segment_Mean

result = tessera.featurize(
    snv_df=snv_df, cna_df=cna_df,
    variant="joint_snv_cna_noloh",   # or "joint_snv_cna" (with-LoH)
    from_assembly="GRCh37",          # "GRCh38" triggers UCSC liftover
)

result.snv_features      # (n_variants, 1169)  per-variant embeddings
result.cna_features      # (n_segments, 688)   per-segment embeddings
```

CSV column conventions:

- **SNV**: `Tumor_Sample_Barcode`, `Chromosome` (no `chr` prefix), `Start_Position`, `Reference_Allele`, `Tumor_Seq_Allele2`, plus either `vaf` or both `t_alt_count` + `t_ref_count`. Single-base substitutions only.
- **CNA**: `Tumor_Sample_Barcode`, `Chromosome`, `Start`, `End`, `Segment_Mean` (log2 ratio); optional `LOH` column triggers the with-LoH model variant.

## Local installation

For users who want to run inference offline or integrate TESSERA into a custom pipeline:

```bash
pip install tessera-foundation
```

The first call to `tessera.featurize` (below) downloads the reference genome (~3 GB) and the requested model weights from Hugging Face Hub on demand and caches both, so you don't need a separate setup step.

To reproduce the manuscript or retrain from scratch, clone the repo for the analysis scripts and the FASTA bootstrap helper:

```bash
git clone https://github.com/JW-Sidhom-Lab/tessera.git
cd tessera
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash tessera/ref_genomes/download_ref_genomes.sh
```

`requirements.txt` covers the foundation-model package, all manuscript-reproduction scripts (pretraining, classifiers, prognostic / predictive-biomarker analyses), and the Gradio inference API. A trimmer subset for deploying only the inference API is at [`inference_api/requirements.txt`](inference_api/requirements.txt).

Weights are hosted on Hugging Face Hub at [huggingface.co/JW-Sidhom-Lab/tessera-foundation](https://huggingface.co/JW-Sidhom-Lab/tessera-foundation) under CC-BY-NC-4.0. The shortest path from raw dataframes to feature tensors is the `featurize` one-liner, which downloads weights on first call (cached afterwards), lifts non-hg19 coordinates, builds the dataset, and runs both per-modality feature heads:

```python
import tessera

result = tessera.featurize(
    snv_df=snv_df,                      # columns: Tumor_Sample_Barcode, Chromosome, Start_Position,
                                        #          Reference_Allele, Tumor_Seq_Allele2, vaf
    cna_df=cna_df,                      # columns: Tumor_Sample_Barcode, Chromosome, Start, End, Segment_Mean
    variant="joint_snv_cna_noloh",      # or "joint_snv_cna" for the with-LoH variant
    from_assembly="GRCh38",             # "GRCh37" / "hg19" is a no-op; otherwise UCSC liftover runs
)

result.snv_features      # (n_variants, 1169)  per-variant embeddings, row-aligned with result.snv_table
result.cna_features      # (n_segments, 688)   per-segment embeddings, row-aligned with result.cna_table
result.liftover_stats    # {"snv": {"n_in", "n_out", "n_dropped"}, "cna": {...}}
```

For finer-grained control there are still building blocks:

```python
from tessera import load_pretrained, lift_snv, lift_cna

model = load_pretrained("joint_snv_cna_noloh")    # download + instantiate, ~3 s cold
snv_df, _ = lift_snv(snv_df, from_assembly="GRCh38")    # identity if from_assembly=="GRCh37"
cna_df, _ = lift_cna(cna_df, from_assembly="GRCh38")
result = model.featurize(snv_df=snv_df, cna_df=cna_df)  # repeat without re-downloading
```

UCSC chain files are downloaded on first use and cached at `~/.cache/pyliftover/`; offline environments can point the loader at a bundled chain file via the `chain_file=` argument or the `TESSERA_LIFTOVER_CHAIN` environment variable.

## Reproducing the manuscript

The pipeline runs in three stages:

1. **Data preparation** ([`data/`](data/README.md)): per-cohort
   download instructions, source-table provenance, and the
   `create_training_data*.py` / `build_<cohort>_metadata.py` builders
   that turn raw releases into the analysis-ready CSVs.
2. **Foundation-model pretraining**
   ([`scripts/tcga_pancan_*/`](scripts/README.md)): trains the SNV
   models, the CNA models, and the joint SNV+CNA InfoNCE-aligned
   foundation model on the TCGA Pan-Cancer Atlas.
3. **Downstream analyses** ([`scripts/`](scripts/README.md)):
   variant-pathogenicity calibration, cross-platform validation,
   tumour-type classification, prognostic stratification, doubly-robust
   counterfactual treatment-effect estimation, and cell-line transfer.

[`scripts/README.md`](scripts/README.md) and
[`data/README.md`](data/README.md) hold the per-directory tables
linking each script and cohort to the relevant manuscript section.

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

> *citation pending publication*

A BibTeX entry will be added on acceptance.

## License

This repository is distributed under the **PolyForm Noncommercial License 1.0.0** (see [`LICENSE`](LICENSE)). Use is permitted for academic research, education, public-research-organization use, and personal experimentation; commercial use is not permitted without a separate license. Pretrained foundation-model weights are released on the Hugging Face Hub under **CC-BY-NC-4.0** (non-commercial, attribution required). Pretrained weights for downstream clinical task heads (CRC and PDAC treatment-effect models) remain available on request under a Data Use Agreement. Patents covering clinical applications of TESSERA are assigned to NewYork-Presbyterian; commercial licensing inquiries should be directed to NYP's technology transfer office.

## Lab

TESSERA is developed in the [JW Sidhom Lab](https://github.com/JW-Sidhom-Lab) at Weill Cornell Medicine.

For questions, collaborations, or commercial-licensing enquiries, contact the corresponding author.
