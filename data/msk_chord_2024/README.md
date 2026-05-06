# MSK-CHORD 2024 data preparation

This directory holds the data-preparation pipeline for the MSK-CHORD 2024
release, used in the manuscript for **(a)** CNA cross-platform validation
(Figure 2 d) and **(b)** the predictive-biomarker clinical analyses
(Figure 6 b-i: CRC FOLFOX/FOLFIRI, PDAC FOLFIRINOX/Gem+nab-paclitaxel).

## Required raw inputs

Download the MSK-CHORD 2024 release. Two access paths:

- **cBioPortal:** https://www.cbioportal.org/study/summary?id=msk_chord_2024
- **Synapse:** see Jee et al., *Nature* 2024 supplementary materials for the accession.

A data-access agreement and free Synapse / cBioPortal account are required.
Solid-tumor samples sequenced with the IMPACT-HEME-400 panel are excluded
downstream by the CNA pipeline.

Files consumed by this pipeline:

| File | Size | Used by |
|---|---|---|
| `data_clinical_patient.txt` | 3.9 MB | `prepare_data.py`, cohort scripts |
| `data_clinical_sample.txt` | 6.2 MB | `prepare_data.py` |
| `data_mutations.txt` | 68 MB | `prepare_data.py` |
| `data_timeline_treatment.txt` | 6.8 MB | `prepare_data.py`, cohort scripts |
| `data_timeline_progression.txt` | 24 MB | cohort scripts (`FIRST_PROGRESSION_DATE`) |
| `data_cna_hg19.seg` | 63 MB | downstream CNA pipeline (in `methods/data/msk_chord/`) |
| `data_gene_panel_matrix.txt` | 1.1 MB | downstream CNA pipeline (sample-to-panel mapping) |

Place all files in this directory. The MSK-CHORD `LICENSE` is shipped with the
repo for redistribution context but **does not** authorise re-uploading the
data; users must download it themselves.

## Pipeline overview

```
data_mutations.txt ──┐
data_clinical_*.txt ─┼──→ prepare_data.py ──→ msk_chord_2024.csv (variants × clinical)
data_timeline_*      │                        msk_chord_2024_tx.csv (per-regimen timeline)
                     │
                     ├──→ create_ground_truth_crc_folfox_folfiri_stage4_ttntd.py
                     │       → GROUND_TRUTH_CRC_FOLFOX_FOLFIRI_STAGE4_TTNTD.csv  (Fig. 6 b–e)
                     │
                     └──→ create_ground_truth_pancreatic_gemabra_folfirinox_stage4_first_line_ttntd.py
                             → GROUND_TRUTH_PANCREATIC_GEMABRA_FOLFIRINOX_STAGE4_FIRST_LINE_TTNTD.csv  (Fig. 6 f–i)
```

`cohort_utils.py` is a shared module used by the two cohort scripts (TTNTD
computation, line-of-therapy grouping, regimen matching). Not directly
executed.

## Running the pipeline

```bash
# Stage 1: variant + clinical and treatment-regimen merges (~5 minutes)
python prepare_data.py \
    --clinical-patient data_clinical_patient.txt \
    --clinical-sample data_clinical_sample.txt \
    --mutations data_mutations.txt \
    --treatment-timeline data_timeline_treatment.txt \
    --variant-output msk_chord_2024.csv \
    --treatment-output msk_chord_2024_tx.csv

# Stage 2: CRC FOLFOX/FOLFIRI cohort (Figure 6 b–e; ~1 minute)
python create_ground_truth_crc_folfox_folfiri_stage4_ttntd.py \
    --base-dir . \
    --output GROUND_TRUTH_CRC_FOLFOX_FOLFIRI_STAGE4_TTNTD.csv

# Stage 2: PDAC FOLFIRINOX/Gem+nab cohort (Figure 6 f–i; ~1 minute)
python create_ground_truth_pancreatic_gemabra_folfirinox_stage4_first_line_ttntd.py \
    --base-dir . \
    --output GROUND_TRUTH_PANCREATIC_GEMABRA_FOLFIRINOX_STAGE4_FIRST_LINE_TTNTD.csv
```

The two cohort scripts can run in parallel; both depend only on the raw
release files, not on each other or on `prepare_data.py`'s outputs.

## Outputs

| File | Rows (manuscript) | Used by |
|---|---|---|
| `msk_chord_2024.csv` | ~1.5 M variants | downstream SNV pipeline |
| `msk_chord_2024_tx.csv` | ~30 K regimen rows | predictive-biomarker analyses |
| `GROUND_TRUTH_CRC_FOLFOX_FOLFIRI_STAGE4_TTNTD.csv` | 1,699 regimen instances (1,452 after PFS filter applied downstream) | Fig. 6 b–e |
| `GROUND_TRUTH_PANCREATIC_GEMABRA_FOLFIRINOX_STAGE4_FIRST_LINE_TTNTD.csv` | 889 (771 after PFS filter) | Fig. 6 f–i |

## Cohort definitions

The two ground-truth scripts use the TTNTD framework from Jee et al. (2024):

- **Cohort filter.** Stage IV patients of the relevant cancer type
  (Colorectal Cancer or Pancreatic Cancer), defined by `STAGE_HIGHEST_RECORDED`
  in `data_clinical_patient.txt` and `CANCER_TYPE` in `data_clinical_sample.txt`.
- **Line of therapy.** Treatments in `data_timeline_treatment.txt` are sorted
  per patient by `START_DATE`; consecutive treatments within 30 days are
  collapsed into one line.
- **Regimen matching.** A line matches a regimen when its set of unique agents
  is a superset of the regimen's required-agent set. Regimens are evaluated in
  list order; first match wins.
  - **CRC**: any line of therapy.
  - **PDAC**: first line only.
- **TTNTD.** `min(next_treatment_start, death_date) − tx_start`, where
  `death_date = OS_MONTHS × 30.44` when `OS_STATUS == "1:DECEASED"`.
- **Response label.** `TTNTD_RESPONSE_LABEL = 1` if `TTNTD > 180 days`, else 0.
  Censored patients with `treatment_duration ≥ 180 days` who are alive get a 1
  (sustained-on-therapy responder).

## Citation

Cite the MSK-CHORD release when using this data:

Jee et al. *Real-world molecular outcomes of cancer.* *Nature*, 2024.
