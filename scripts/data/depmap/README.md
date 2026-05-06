# DepMap model-ready cell-line tables

Single-script Stage-2 data prep for the CRC cell-line orthogonal validation in
Figure 6 n. Reads the raw DepMap 24Q2 + CTRPv2 release from
[`data/DepMap/`](../../../data/DepMap/README.md), lifts CNA segments
hg38 -> hg19, restricts both modalities to MSK-IMPACT505 panel coverage (so
the model sees the same assay scope it was fit to in MSK-CHORD), and writes
the three tables the inference scripts consume.

## Pipeline

```
data/DepMap/
  ├── Model.csv
  ├── OmicsSomaticMutations_TCGA_format_hg19.csv
  ├── OmicsCNSegmentsProfile.csv         (hg38)
  ├── OmicsAbsoluteCNSegmentsProfile.csv (hg38, with LoH)
  ├── OmicsDefaultModelProfiles.csv
  └── CTRPv2.0_2015_ctd2_ExpandedDataset/
        ├── v20.meta.per_cell_line.txt
        ├── v20.meta.per_compound.txt
        ├── v20.meta.per_experiment.txt
        └── v20.data.curves_post_qc.txt

data/genie_18_0/genomic_information.txt   (MSK-IMPACT505 panel coordinates)

                              │
                              ▼
                  prepare_depmap_data.py
                              │
                              ▼
        snv_panel.csv     cna_panel.csv     metadata.csv
        (~3.5 MB)         (~7.5 MB)         (~255 KB)
```

## Running

```bash
python prepare_depmap_data.py
```

Defaults assume the layout above. Override with `--src`, `--panel-info`, or
`--output-dir`. Requires `pyliftover` (`pip install pyliftover`). Runtime is
~10 minutes, dominated by the per-segment liftover loop. Memory peaks around
3-4 GB.

## Outputs

All three are tracked in git so the manuscript Figure 6 n results are
reproducible without re-running the pipeline. Re-running on the same raw
inputs reproduces them deterministically (random subsamples are seeded).

| File | Description |
|---|---|
| `snv_panel.csv` | Per-variant TCGA-MAF rows on hg19, restricted to MSK-IMPACT505 panel genes. Capped at 1,000 SNVs per cell line; variants seen in >= 5 cell lines cohort-wide are preserved as recurrent. |
| `cna_panel.csv` | Per-segment rows on hg19, segments overlapping any IMPACT505 gene region. Columns: `Tumor_Sample_Barcode, Chromosome, Start, End, Segment_Mean, NumProbes, LOH`. Capped at 1,000 segments per cell line, ranked by `\|Segment_Mean\| + 0.5 * LOH`. |
| `metadata.csv` | Per-cell-line lineage, primary disease, subtype, TP53 / KRAS coding-mutation flags, 17p-loss flag, ploidy, CTRPv2 oxaliplatin / SN-38 / fluorouracil AUCs, and modality-availability flags (`has_snv`, `has_cna`, `has_both`). |

## Reference genome

Outputs are on **GRCh37 (hg19)** to match TESSERA's TCGA-trained coordinate
system; no additional liftover is required at inference time.

- DepMap ships SNVs pre-lifted to hg19 in `OmicsSomaticMutations_TCGA_format_hg19.csv`.
- DepMap CNA segments are released on hg38 (`OmicsCNSegmentsProfile.csv`);
  the script lifts each segment's start and end with `pyliftover`, dropping
  segments where either endpoint fails to map or strands invert (~16% loss,
  mostly centromeric / unstable regions). The matching LoH-status column from
  `OmicsAbsoluteCNSegmentsProfile.csv` is merged in **before** the liftover
  while both files are still on hg38 with identical segment boundaries.

## Cohort scope

- Solid-tumour lineages only (hematologic excluded via `OncotreeLineage`).
- 1,475 cell lines have both SNV and CNA modalities after the panel filter.
- 83 are colorectal adenocarcinoma; 44 of those carry both oxaliplatin and
  SN-38 AUCs in CTRPv2 — the n=44 cohort plotted in Figure 6 n.

## Why MSK-IMPACT505

The MSK-CHORD-fit predictive biomarker $\hat{\tau}$ in Figure 6 was trained
on samples sequenced with MSK-IMPACT505. Applying it to full WES DepMap
inputs would let TESSERA's attention layers see a different bag of variants
and segments than the model was fit to, producing per-token embeddings that
aren't directly comparable. Restricting to IMPACT505 coverage matches the
assay scope used at fit time.

## Citations

- DepMap, Broad Institute (2024). DepMap 24Q2 release.
- Ghandi et al. *Next-generation characterization of the Cancer Cell Line Encyclopedia.* Nature, 2019.
- Seashore-Ludlow et al. *Harnessing connectivity in a large-scale small-molecule sensitivity dataset.* Cancer Discovery, 2015.
