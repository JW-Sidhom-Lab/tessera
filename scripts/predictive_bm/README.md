# Predictive biomarker analysis (Figure 6)

Doubly-robust counterfactual treatment-effect pipeline applied to two
MSK-CHORD first-line metastatic cohorts: CRC FOLFOX vs FOLFIRI and PDAC
FOLFIRINOX vs gemcitabine plus nab-paclitaxel. Backs all panels of
manuscript Figure 6 (a-n) and Supplementary Figures 10, 11, 12.

## Inputs

| Path | Producer | Used by |
|---|---|---|
| `../../data/msk_chord_2024/<cohort>_*.csv` | curated MSK-CHORD 2024 release | every `crc_*.py` / `pdac_*.py` |
| `cache/patient_features.pkl`               | `core/features.build_patient_features` (auto-built on first call, cached for reuse) | every `crc_*.py` / `pdac_*.py` |
| `msk_chord_latent_features_raw.pkl` (~4.7 GB) | [`get_latent_features.py`](get_latent_features.py) using the trained TCGA-pretrained TESSERA SNV+CNA InfoNCE-noLOH checkpoint | `core.features.build_patient_features` |
| `cache/msk_chord_scalers.pkl`              | [`export_scalers.py`](export_scalers.py)                                                                                | DepMap cell-line validation (`scripts/depmap/`, separate module producing Fig 6n) |

Directory references in the scripts are repo-relative
(`../../data/msk_chord_2024/`, `../tcga_pancan_snv_cna/models/`), so as
long as the layout above holds the paths resolve unchanged from the
research workspace.

## Pipeline

```
TCGA-pretrained TESSERA SNV+CNA InfoNCE-noLOH checkpoint
                       │
                       ▼ get_latent_features.py
       msk_chord_latent_features_raw.pkl
                       │
                       ▼ core/features.build_patient_features (cached)
                cache/patient_features.pkl
                       │
       ┌───────────────┴───────────────┐
       ▼                               ▼
   crc_discovery.py              pdac_discovery.py
       │                               │           (Fig 6 b-i, Sup 10)
       ├────► crc_signatures.py        ├────► pdac_signatures.py        (PMD attribution + decomposition; Fig 6j, Sup 12a)
       │           │                   │           │
       │           ▼                   │           ▼
       │   crc_signatures_figure.py    │   pdac_signatures_figure.py
       │   crc_signature_validation.py │   pdac_signature_validation.py (Sup 12b)
       │                               │
       ├────► crc_tp53_kras_simple_rule.py            (Fig 6 k-m)
       │                               ├────► pdac_triplet_rule.py     (Sup 12 c-e)
       │                               │
       ├────► crc_ablation.py          ├────► pdac_ablation.py          (Sup 11)
       │                               │
       └────► quantify_reassignment_benefit.py        (SEER-Medicare projection in main text)
                                       │
                                       ▼
                           build_figure6_panels.py
                                       │
                                       ▼
                       results/figures/<cohort>/figure6/
                       (npz + meta.json consumed by manuscript/build/figure6*.py)
```

## Files

### Top-level scripts

| Script | Backs |
|---|---|
| [`crc_discovery.py`](crc_discovery.py) | CRC DR-learner fit; produces τ̂, τ̂_0, per-arm KM data for Fig 6 b-e + Sup 10 a-b. |
| [`pdac_discovery.py`](pdac_discovery.py) | PDAC DR-learner fit; Fig 6 f-i + Sup 10 c-d. |
| [`crc_signatures.py`](crc_signatures.py) | Per-(patient, gene-or-arm) attribution matrix + PMD K-sweep; Fig 6 j input. |
| [`pdac_signatures.py`](pdac_signatures.py) | PDAC analogue at K=10; Sup 12 a input. |
| [`crc_signatures_figure.py`](crc_signatures_figure.py) | Renders Fig 6 j heatmap + writes panel data. |
| [`pdac_signatures_figure.py`](pdac_signatures_figure.py) | Renders Sup 12 a heatmap + writes panel data. |
| [`crc_signature_validation.py`](crc_signature_validation.py) | Per-signature interaction Cox; supplementary table feeding the validation columns. |
| [`pdac_signature_validation.py`](pdac_signature_validation.py) | PDAC analogue; Sup 12 b forest. |
| [`crc_tp53_kras_simple_rule.py`](crc_tp53_kras_simple_rule.py) | TP53 / KRAS / 17p genotype subgroups; Fig 6 k-m. |
| [`pdac_triplet_rule.py`](pdac_triplet_rule.py) | TP53 + 17p-intact + 20q+ triplet; Sup 12 c-e. |
| [`crc_ablation.py`](crc_ablation.py), [`pdac_ablation.py`](pdac_ablation.py), [`_ablation_lib.py`](_ablation_lib.py) | Three-way Local / +Global / +InfoNCE feature-slice ablation; Sup 11. |
| [`quantify_reassignment_benefit.py`](quantify_reassignment_benefit.py) | SEER-Medicare 1L stage IV CRC reassignment projection. |
| [`build_figure6_panels.py`](build_figure6_panels.py) | Aggregates everything into the npz + meta.json bundle that `manuscript/build/figure6*.py` reads. |
| [`get_latent_features.py`](get_latent_features.py) | Frozen TESSERA inference on MSK-CHORD; produces the 4.7 GB latents pkl. |
| [`export_scalers.py`](export_scalers.py) | Saves the SNV / CNA RobustScaler state + feature ordering so the DepMap module can apply β_eff to external embeddings. |

### Library

| Module | Purpose |
|---|---|
| [`core/dr.py`](core/dr.py) | DR-learner: nested CV, sparse-PLS nuisances, AIPW pseudo-outcome, indifference-threshold + per-stratum HR helpers. |
| [`core/spls.py`](core/spls.py) | Sparse-PLS regressor (the second-stage and nuisance learner). |
| [`core/cohorts.py`](core/cohorts.py) | `build_crc_met` / `build_pdac_met`: 1L stage IV cohort assembly with arm + endpoint columns. |
| [`core/features.py`](core/features.py) | `build_patient_features`: SNV + CNA mean / max pools + log-counts on the frozen TESSERA latents. |
| [`core/attribution.py`](core/attribution.py) | β_eff composition + per-(patient, variant / segment) attribution computation. |
| [`core/decomposition.py`](core/decomposition.py) | Witten et al. PMD with L1 patient + feature budgets. |
| [`core/arm_mapping.py`](core/arm_mapping.py) | hg19 chromosome-arm interval-overlap aggregation. |
| [`core/gene_mapping.py`](core/gene_mapping.py) | SNV → Hugo_Symbol aggregation. |
| [`core/plots.py`](core/plots.py), [`core/plotutils.py`](core/plotutils.py) | KM, forest, and HR-curve renderers used by the per-cohort figure scripts. |

### Tests

| Script | Purpose |
|---|---|
| [`tests/verify_attribution.py`](tests/verify_attribution.py) | Reconstructs τ̂_p from the per-(patient, variant) attribution + burden contributions, asserting `|Δτ| / max(|τ|, 1) < 1e-5` for every patient (Methods §"Per-(patient, variant) attributions"). |

## Running it

```bash
cd scripts/predictive_bm

# 1. Frozen-model inference on MSK-CHORD (one-time, ~4.7 GB output, RunPod)
python3 get_latent_features.py

# 2. CRC track
python3 crc_discovery.py
python3 crc_signatures.py
python3 crc_signatures_figure.py
python3 crc_signature_validation.py
python3 crc_tp53_kras_simple_rule.py
python3 crc_ablation.py
python3 quantify_reassignment_benefit.py

# 3. PDAC track
python3 pdac_discovery.py
python3 pdac_signatures.py
python3 pdac_signatures_figure.py
python3 pdac_signature_validation.py
python3 pdac_triplet_rule.py
python3 pdac_ablation.py

# 4. Figure-6 panel bundle for the manuscript build chain
python3 build_figure6_panels.py

# 5. Optional: scaler export for the DepMap cell-line module
python3 export_scalers.py

# 6. Verify Methods β_eff reconstruction is faithful
python3 tests/verify_attribution.py
```

`cache/patient_features.pkl` is built on the first `crc_discovery.py` /
`pdac_discovery.py` run and reused by every downstream script. The
nested 10x5 CV in `*_discovery.py` is the long pole (~15-30 min per
cohort on a single workstation).

## Reproducibility

- `core.dr.fit_dr_learner` seeds `numpy` and `random` per-fold; the
  outer 10-fold and inner 5-fold splits are stratified on the joint
  (arm, event) label and deterministic.
- The PMD optimisation (Witten 2009) is iterative; the
  `core.decomposition.pmd` wrapper uses a fixed initialisation.
