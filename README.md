<p align="center">
  <img src="logo.png" alt="TESSERA logo" width="220">
</p>

<p align="center">
  <em>Tumour Embeddings via Self-Supervised Encoding and Reconstruction of Alterations</em><br>
  A foundation model for the cancer genome.
</p>

---

TESSERA is a self-supervised foundation model jointly pretrained on somatic single-nucleotide variants (SNVs) and copy-number alterations (CNAs) from the TCGA Pan-Cancer Atlas. A single learned representation, produced once and reused without retraining, supports variant pathogenicity prediction, pan-cancer tumour-type classification, unsupervised molecular subtyping, prognostic stratification, and counterfactual treatment-effect estimation.

This repository contains the reference implementation, pretrained weights pointer, and inference utilities described in the accompanying paper.

## Quick start

The fastest way to use TESSERA is via the public inference API on Hugging Face — no local installation required. Upload SNV and/or CNA data, get back per-variant predictions and embeddings:

🔗 **Inference API**: [huggingface.co/spaces/JW-Sidhom-Lab/tessera](https://huggingface.co/spaces/JW-Sidhom-Lab/tessera) *(coming soon)*

From Python:

```python
from gradio_client import Client

client = Client("JW-Sidhom-Lab/tessera")
result = client.predict(
    maf_file="path/to/sample.maf",
    seg_file="path/to/sample_segments.tsv",
    api_name="/predict",
)
# result includes per-variant embeddings, pathogenicity scores,
# per-segment predictions, and a sample-level joint embedding.
```

The API serves the foundation-model outputs only (embeddings + per-variant / per-segment predictions). Downstream task heads (tumour-type classifier, treatment-effect score) are available on request under a Data Use Agreement.

## Local installation

For users who want to run inference offline, integrate TESSERA into a custom pipeline, or retrain on their own data:

```bash
# Clone
git clone https://github.com/JW-Sidhom-Lab/tessera.git
cd tessera

# Install
pip install -e .

# Download reference genome (default: GRCh37)
bash tessera/ref_genomes/download_ref_genomes.sh
```

Weights are hosted on Hugging Face Hub at [huggingface.co/JW-Sidhom-Lab/tessera-foundation](https://huggingface.co/JW-Sidhom-Lab/tessera-foundation) *(coming soon)*. Loading from Python:

```python
from tessera.model import TESSERA
from huggingface_hub import snapshot_download

weights_dir = snapshot_download(repo_id="JW-Sidhom-Lab/tessera-foundation")
model = TESSERA(name="tessera_v1", model_dir=weights_dir)
```

## Repository layout

```
tessera/
├── tessera/                        # the package
│   ├── base.py                     # BaseModel — shared data + training infrastructure
│   ├── input_keys.py               # input-key helpers
│   ├── model.py                    # TESSERA — current foundation-model class
│   ├── data/
│   │   └── preprocessing.py        # SNV/CNA tokenization, FASTA lookup, sample bagging
│   ├── layers/                     # custom Keras layers (attention, masking, MIL, ...)
│   ├── training/                   # training utilities (callbacks, losses, schedules)
│   └── ref_genomes/                # reference-genome download script + indices
└── README.md
```

## Citing TESSERA

If you use TESSERA in your work, please cite:

> *citation pending publication*

A BibTeX entry will be added on acceptance.

## License

This repository is currently distributed under the **Apache License 2.0** (see [`LICENSE`](LICENSE)). License terms are subject to confirmation by Cornell's Center for Technology Licensing in light of pending patent applications covering specific clinical applications. Pretrained weights for downstream task heads (CRC and PDAC treatment-effect models) are available on request under a Data Use Agreement.

## Lab

TESSERA is developed in the [JW Sidhom Lab](https://github.com/JW-Sidhom-Lab) at Weill Cornell Medicine.

For questions, collaborations, or commercial-licensing enquiries, contact the corresponding author.
