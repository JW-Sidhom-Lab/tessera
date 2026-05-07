# Prognostic biomarker analysis (Figure 5)

Per-cohort UMAP + polynomial Cox + risk-group survival pipeline that
produces the manuscript Figure 5 panels. Reads joint InfoNCE-aligned
sample features from
[`scripts/tcga_pancan_snv_cna/`](../tcga_pancan_snv_cna/README.md), the
per-cohort clinical metadata from
[`data/prognostic/`](../../data/prognostic/README.md), and writes the
per-sample cache that drives the published Figure 5 panels.

## Cohorts

| Cohort | Categorical comparator | Primary survival endpoint | Manuscript figures |
|---|---|---|---|
| `glioma` | WHO 2021 primary class | DSS, 5y | Fig 5 a |
| `brca`   | OncotypeDX RS tertile  | DSS, 5y | Fig 5 c-j |
| `prad`   | Decipher tertile       | PFI.1, 5y | Fig 5 k-r |

UCEC (Fig 5 b) is not yet wired into this directory: its clinical
metadata builder hasn't been migrated into `data/prognostic/`. Adding
`configs/ucec.py` + a `data/prognostic/ucec/` build is the only thing
needed.

## Pipeline

```
data/prognostic/<cohort>/<cohort>_clinical_metadata.csv  (subtype + survival)
scripts/tcga_pancan_snv_cna/multimodal_features/         (joint InfoNCE features)
                       │
                       ▼
              ./run_cohort.sh                            (deterministic launcher)
                       │
                       ▼
              cohort_analysis.py                         (UMAP + polynomial Cox + KM)
                       │
       ┌───────────────┼───────────────┐
       ▼               ▼               ▼
analysis_results_<cohort>_infonce_per_sample_loss/
   ├── data/fig5_cache.csv           ← per-sample UMAP + Risk_Score + survival
   ├── data/sample_assignments.csv
   └── plots/...                      ← per-cohort survival panels
                       │
                       ▼
              joint_cox.py
              (continuous z-scored joint Cox: TESSERA vs comparator)
```

## Reproducibility

**Always invoke via `./run_cohort.sh`.** The launcher pre-sets a
handful of environment variables (`NUMBA_THREADING_LAYER=workqueue`,
`BLIS_NUM_THREADS=1`, single-thread BLAS / OMP / MKL / NumExpr,
`PYTHONHASHSEED=0`) that the underlying numba JIT only honours when
they are present at process launch. Setting them from inside Python
after numba has loaded is too late: UMAP coordinates drift run-to-run,
the polynomial-Cox fit on those coordinates drifts with them, and the
joint-Cox HR shifts noticeably across reruns.

With `./run_cohort.sh` the per-sample `fig5_cache.csv` is byte-
identical across reruns on the same machine + Python environment.

## Running

```bash
cd scripts/prognostic_bm

# Run each cohort. Each invocation rewrites cohort_analysis.py's COHORT
# constant so subsequent argument-free runs default to the most recent
# cohort.
./run_cohort.sh                    # whatever COHORT is set in cohort_analysis.py (default: brca)
COHORT=glioma ./run_cohort.sh
COHORT=brca   ./run_cohort.sh
COHORT=prad   ./run_cohort.sh

# After all three cohorts have run, compute the manuscript continuous
# joint Cox numbers (BRCA Figure 5 c-j and PRAD Figure 5 k-r).
python joint_cox.py
```

## Recognised env vars

| Variable | Used in | Notes |
|---|---|---|
| `COHORT` | `run_cohort.sh` | Override the cohort to run (e.g. `glioma`); rewrites the `COHORT` constant in `cohort_analysis.py` before launching. |
| `OUTPUT_TXT` | `joint_cox.py` | Override the destination of the joint-Cox text artefact. |

## Outputs

| Path | Description |
|---|---|
| `analysis_results_<cohort>_infonce_per_sample_loss/data/fig5_cache.csv` | Per-sample UMAP coords, Risk_Score, Risk_Group, Subtype, survival. Source data for Fig. 5 panels. |
| `analysis_results_<cohort>_infonce_per_sample_loss/plots/` | Per-cohort UMAP overlays + KM panels. |
| `analysis_results_brca_infonce_per_sample_loss/joint_cox_continuous_brca_prad.txt` | The manuscript Figure 5 c-j and Figure 5 k-r joint-Cox numbers (HR per +1 SD, 95% CI, P, joint C, Spearman ρ). |

All output directories are gitignored.

## Compute requirements

Runs on CPU; the dominant cost is the consensus UMAP (10 seeds,
Procrustes-aligned, ~2-5 minutes per cohort on a 2024 laptop). No GPU
required.
