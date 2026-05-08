# AACR Project GENIE v18.0 data preparation

This directory holds the data-preparation pipeline for the AACR Project GENIE
v18.0 public release, used as the **independent SNV cross-platform validation
cohort** in the manuscript (Figure 1 f-g). CNA cross-platform validation uses
MSK-CHORD instead, so the GENIE CNA segment file is not consumed by this
pipeline.

## Required raw inputs

GENIE is hosted on Synapse. A free Synapse account and a one-time data access
agreement are required.

**Project page:** https://www.synapse.org/genie
**Direct accession:** `syn7222066` (browse the v18.0 public release).

Files needed:

| File | Size | Notes |
|---|---|---|
| `data_mutations_extended.txt` | 837 MB | Mutations MAF (cBioPortal/GENIE format) |
| `data_clinical_sample.txt` | 36 MB | Sample-level clinical (cBioPortal four-row metadata header) |
| `genomic_information.txt` | 55 MB | Per-panel gene/exon coordinates. Not used by this directory's `create_training_data.py`, but read by `scripts/data/msk_chord/create_cna.py` for IMPACT panel filtering. |

`data_clinical_patient.txt` (19 MB) is also typically downloaded with the
release; it is not consumed by this pipeline directly but kept alongside for
context.

Place these files in this directory.

## Running the pipeline

```bash
# Single stage: mutations + sample-level clinical merge (~3 minutes; 1.2 GB output)
python create_training_data.py \
    --maf data_mutations_extended.txt \
    --clinical data_clinical_sample.txt \
    --output GENIE.csv
```

## Output

| File | Columns | Used by |
|---|---|---|
| `GENIE.csv` | All MAF columns + `vaf` + sample-level clinical (`SAMPLE_ID` -> `Tumor_Sample_Barcode`, `ONCOTREE_CODE`, `SEQ_ASSAY_ID`, `CANCER_TYPE`, etc.) | Downstream SNV-validation scripts |

## Next step

After producing `GENIE.csv`, run `scripts/data/genie/` to build the SNV
validation table consumed by `scripts/genie_snv/`.

## Citations

Cite the AACR Project GENIE Consortium when using this data:

- AACR Project GENIE Consortium. *AACR Project GENIE: Powering Precision Medicine
  through an International Consortium.* Cancer Discovery, 2017.
- Pugh et al. *AACR Project GENIE: 100,000 cases and beyond.* Cancer Discovery, 2022.
