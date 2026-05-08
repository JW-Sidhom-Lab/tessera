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
    variant="joint_snv_cna_noloh",        # or "joint_snv_cna" (with-LoH)
    from_assembly="GRCh37",               # "GRCh38" triggers UCSC liftover
    quantile_normalize_to_tcga=False,     # set True for panel/cell-line data
)

result.snv_features      # (n_variants, 1169)  per-variant embeddings
result.cna_features      # (n_segments, 688)   per-segment embeddings
```

First call downloads the requested model variant from Hugging Face Hub (~185 MB) and, on first SNV call, the GRCh37 reference genome (~3 GB); both are cached locally.

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

## Reproducing the manuscript

For training, downstream analyses, and figure generation, clone the repo:

```bash
git clone https://github.com/JW-Sidhom-Lab/tessera.git
cd tessera
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash tessera/ref_genomes/download_ref_genomes.sh
```

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

> *citation pending publication*

A BibTeX entry will be added on acceptance.

## License

This repository is distributed under the **PolyForm Noncommercial License 1.0.0** (see [`LICENSE`](LICENSE)). Use is permitted for academic research, education, public-research-organization use, and personal experimentation; commercial use is not permitted without a separate license. Pretrained foundation-model weights are released on the Hugging Face Hub under **CC-BY-NC-4.0** (non-commercial, attribution required). Pretrained weights for downstream clinical task heads (CRC and PDAC treatment-effect models) remain available on request under a Data Use Agreement. Patents covering clinical applications of TESSERA are assigned to NewYork-Presbyterian; commercial licensing inquiries should be directed to NYP's technology transfer office.

## Lab

TESSERA is developed in the [JW Sidhom Lab](https://github.com/JW-Sidhom-Lab) at Weill Cornell Medicine.

For questions, collaborations, or commercial-licensing enquiries, contact the corresponding author.
