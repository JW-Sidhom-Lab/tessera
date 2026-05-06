# DepMap 24Q2 + CTRPv2 raw data

Raw release files for the cell-line orthogonal validation in the manuscript
(Figure 6 n: MSK-CHORD-trained $\hat{\tau}$ applied to 44 CRC cell lines from
DepMap, evaluated against the FOLFOX-vs-FOLFIRI sensitivity preference in
CTRPv2).

Unlike the other cohorts, this directory has no Stage-1 `create_training_data.py`:
the file conversion (hg38->hg19 liftover, indel re-encoding, VEP->TCGA
classification mapping), per-cell-line subsampling, 17p-loss flag computation,
and CTRPv2 AUC merge all happen in the Stage-2 script at
[`scripts/data/depmap/`](../../scripts/data/depmap/).

## Required raw inputs

### DepMap 24Q2

Download from the DepMap 24Q2 release page. Free account required.

**https://depmap.org/portal/data_page/?tab=allData** (filter to "DepMap Public 24Q2").

| File | Size | Used for |
|---|---|---|
| `Model.csv` | 678 KB | Cell-line metadata (lineage, primary disease, subtype) |
| `OmicsSomaticMutations_TCGA_format_hg19.csv` | 338 MB | TCGA-format MAF, already lifted to hg19 |
| `OmicsCNSegmentsProfile.csv` | 30 MB | Log2-ratio CNA segments (hg38) |
| `OmicsAbsoluteCNSegmentsProfile.csv` | 28 MB | Absolute-CN segments with LoH annotations; used for the per-cell-line 17p-loss flag |
| `OmicsDefaultModelProfiles.csv` | 81 KB | ProfileID -> ModelID mapping |

### CTRPv2.0

Download from the Cancer Therapeutics Response Portal v2 release. Free
account required. Place the four files inside a subdirectory named
`CTRPv2.0_2015_ctd2_ExpandedDataset/`.

**https://portals.broadinstitute.org/ctrp.v2/**

| File | Size | Used for |
|---|---|---|
| `v20.meta.per_cell_line.txt` | 70 KB | CCL identifier <-> cell-line name |
| `v20.meta.per_compound.txt` | 114 KB | Compound -> master_cpd_id |
| `v20.meta.per_experiment.txt` | 94 KB | Experiment -> master_ccl_id |
| `v20.data.curves_post_qc.txt` | 57 MB | Per-experiment dose-response area-under-curve values |

## Drugs of interest

The manuscript validates the FOLFOX-vs-FOLFIRI predictive direction using
**oxaliplatin** (FOLFOX backbone) and **SN-38** (FOLFIRI active metabolite).
The Stage-2 script will subset CTRPv2 to those two drugs (it also processes
fluorouracil for completeness, though that drug is not used in the
manuscript figures).

## Citations

- DepMap, Broad Institute (2024). DepMap 24Q2 release.
- Ghandi et al. *Next-generation characterization of the Cancer Cell Line Encyclopedia.* Nature, 2019.
- Seashore-Ludlow et al. *Harnessing connectivity in a large-scale small-molecule sensitivity dataset.* Cancer Discovery, 2015.
