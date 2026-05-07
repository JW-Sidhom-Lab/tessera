"""Build ``prad_clinical_metadata.csv`` for the TESSERA PRAD cohort.

Joins two sources into a single per-patient table:

1. **TCGA Pan-Cancer Atlas curated survival** (Liu et al., 2018,
   ``DSS_cr``, ``PFI``, ``DFI.cr``, etc.) for the PRAD subset of
   ``../../TCGA_PanCan/clinical.csv``.

2. **Published curatedPCaData TCGA scores** (Laajala et al., 2023,
   *Sci. Data*; Bioconductor ExperimentHub resource ``EH8024``) --
   per-patient Decipher (22-gene), OncotypeDX GPS, Prolaris
   cell-cycle-progression, and AR-signaling scores computed under a
   single uniform pipeline on TCGA-PRAD RNA-seq. Source:
   ``curatedPCaData_tcga_scores_20230215.Rds``. Requires ``pyreadr``.

Each of the four scores is tertile-binned within the PRAD cohort into
Low / Intermediate / High strata, in parallel to the BRCA pipeline. We
use the curatedPCaData published Decipher score as the headline
transcriptomic comparator (``Subtype`` column); the Veracyte clinical
thresholds (<0.45 / 0.45-0.6 / >0.6) are calibrated to the proprietary
score's distribution and do not transfer.

The output is the TCGA-side ground truth used by Figure 5 k-r of the
manuscript (UMAP overlay + joint Cox of the TESSERA score against the
Decipher comparator).

Inputs
------
``curatedPCaData_tcga_scores_20230215.Rds`` (this directory)
    EH8024 release, 2023-02-15.
``../../TCGA_PanCan/clinical.csv``
    TCGA Pan-Cancer Atlas curated clinical resource (Liu 2018).

Output
------
``prad_clinical_metadata.csv`` (this directory)

Citations
---------
Laajala, T. D. et al. (2023). curatedPCaData: a curated repository of
prostate cancer cohorts. *Scientific Data*. ExperimentHub: EH8024.

Erho, N. et al. (2013). Discovery and validation of a prostate cancer
genomic classifier that predicts early metastasis following radical
prostatectomy. *PLOS ONE* 8(6): e66855.

Karnes, R. J. et al. (2013). Validation of a genomic classifier that
predicts metastasis following radical prostatectomy in an at risk
patient population. *J. Urology*.

Liu, J. et al. (2018). An integrated TCGA pan-cancer clinical data
resource to drive high-quality survival outcome analytics. *Cell*.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pyreadr

HERE = Path(__file__).parent.resolve()
SCORES_RDS = Path(os.environ.get("SCORES_RDS", HERE / "curatedPCaData_tcga_scores_20230215.Rds"))
CLINICAL_CSV = Path(os.environ.get(
    "CLINICAL_CSV", HERE / ".." / ".." / "TCGA_PanCan" / "clinical.csv"))
OUTPUT_CSV = Path(os.environ.get("OUTPUT_CSV", HERE / "prad_clinical_metadata.csv"))

PATIENT_BARCODE_LEN = 12

SCORE_TO_SUBTYPE = (
    # (raw_score_col, subtype_col, low_label, high_label)
    ("decipher", "Subtype_Decipher", "Low",    "High"),
    ("oncotype", "Subtype_Oncotype", "Low",    "High"),
    ("prolaris", "Subtype_Prolaris", "Low",    "High"),
    ("ar_score", "Subtype_AR",       "AR-low", "AR-high"),
)


def _patient_id_dotted(aliquot: str) -> str:
    return "-".join(aliquot.split(".")[:3])


# ---------------------------------------------------------------------------
# 1. TCGA PRAD curated clinical (Liu 2018-style endpoints).
# ---------------------------------------------------------------------------
print(f"Loading TCGA clinical: {CLINICAL_CSV}")
clin = pd.read_csv(CLINICAL_CSV)
clin.columns = [c.lstrip("﻿") for c in clin.columns]
clin = clin[clin["type"] == "PRAD"].copy()
clin = clin.rename(columns={"bcr_patient_barcode": "Patient_ID"})
print(f"  {len(clin):,} PRAD patients with clinical rows")

# ---------------------------------------------------------------------------
# 2. curatedPCaData scores (one row per primary-tumor aliquot;
#    aggregate replicate aliquots to per-patient mean).
# ---------------------------------------------------------------------------
print(f"\nLoading curatedPCaData scores: {SCORES_RDS}")
sc_obj = pyreadr.read_r(str(SCORES_RDS))
sc_df = list(sc_obj.values())[0].T
sc_df = sc_df.reset_index().rename(columns={"index": "aliquot"})
sc_df["Patient_ID"] = sc_df["aliquot"].apply(_patient_id_dotted)
sc_df["sample_type"] = sc_df["aliquot"].apply(lambda x: x.split(".")[3])
prim = sc_df[sc_df["sample_type"] == "01"]

score_cols = ["decipher", "oncotype", "prolaris", "ar_score"]
pat_scores = prim.groupby("Patient_ID")[score_cols].mean().reset_index()
print(f"  {len(pat_scores):,} primary-tumor PRAD patients")
for c in score_cols:
    print(f"  {c:10s} mean={pat_scores[c].mean():.3f}  "
          f"sd={pat_scores[c].std():.3f}  "
          f"range=[{pat_scores[c].min():.3f}, {pat_scores[c].max():.3f}]")

# ---------------------------------------------------------------------------
# 3. Tertile-bin each score within the PRAD cohort.
# ---------------------------------------------------------------------------
for raw_col, subtype_col, low_label, high_label in SCORE_TO_SUBTYPE:
    mid_label = "Intermediate" if low_label == "Low" else "AR-intermediate"
    pat_scores[subtype_col] = pd.qcut(
        pat_scores[raw_col].rank(method="first"),
        q=3, labels=[low_label, mid_label, high_label])
pat_scores["Subtype"] = pat_scores["Subtype_Decipher"]

# Rename raw score columns with explicit `_curatedpcadata` suffix so
# their provenance is unambiguous after the merge.
pat_scores = pat_scores.rename(columns={
    "decipher": "decipher_curatedpcadata",
    "oncotype": "oncotype_curatedpcadata",
    "prolaris": "prolaris_curatedpcadata",
    "ar_score": "ar_score_curatedpcadata",
})

# ---------------------------------------------------------------------------
# 4. Merge and write output.
# ---------------------------------------------------------------------------
out = clin.merge(pat_scores, on="Patient_ID", how="left")
out["curatedpcadata_resource"] = "EH8024 / tcga_scores_20230215.Rds"

# Order columns: identifiers, headline Subtype, per-classifier categoricals,
# raw scores, original clinical columns, provenance.
front = ["Patient_ID", "Subtype",
         "Subtype_Decipher", "Subtype_Oncotype",
         "Subtype_Prolaris", "Subtype_AR",
         "decipher_curatedpcadata", "oncotype_curatedpcadata",
         "prolaris_curatedpcadata", "ar_score_curatedpcadata"]
provenance = ["curatedpcadata_resource"]
other_clin = [c for c in out.columns
              if c not in front + provenance + ["type"]]
out = out[front + other_clin + provenance]

out.to_csv(OUTPUT_CSV, index=False)
print(f"\nMerged: {len(out):,} PRAD patients; "
      f"{out['decipher_curatedpcadata'].notna().sum():,} with curatedPCaData scores")
print(f"Wrote {OUTPUT_CSV}")
