"""Build ``ucec_clinical_metadata.csv`` for the TESSERA UCEC cohort.

Joins UCEC subtype labels (Kandoth et al. 2013, with the PanCancer Atlas
reannotation from Hoadley et al. 2018) to the TCGA Pan-Cancer Clinical
Data Resource survival endpoints (Liu et al. 2018).

Inputs
------
``ucec_clinical.tsv``                       UCEC subtype + grade from cBioPortal PanCancer Atlas study
``../../TCGA_PanCan/clinical.csv``          TCGA CDR (Liu 2018) survival table

Output
------
``ucec_clinical_metadata.csv``              Per-patient table consumed by
                                             ``scripts/prognostic_bm/cohort_analysis.py``

The output schema matches the BRCA / glioma cohort metadata files:
``Patient_ID``, ``Subtype``, ``Grade``, ``DSS``, ``DSS.time``,
``DFI``, ``DFI.time``, ``PFI``, ``PFI.time``. Ambiguous DSS/DFI/PFI
events (==2) in the CDR are recoded as censored (==0) before merge,
mirroring the BRCA / PRAD migration in ``data/prognostic/{brca,prad}/``.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent.resolve()
SUBTYPE_TSV  = Path(os.environ.get("SUBTYPE_TSV",  HERE / "ucec_clinical.tsv"))
CLINICAL_CSV = Path(os.environ.get(
    "CLINICAL_CSV", HERE / ".." / ".." / "TCGA_PanCan" / "clinical.csv"))
OUTPUT_CSV   = Path(os.environ.get("OUTPUT_CSV", HERE / "ucec_clinical_metadata.csv"))

PATIENT_BARCODE_LEN = 12
ENDPOINT_RENAMES = (
    ("DSS_cr",  "DSS.time.cr",  "DSS",  "DSS.time"),
    ("DFI.cr",  "DFI.time.cr",  "DFI",  "DFI.time"),
    ("PFI.cr",  "PFI.time.cr",  "PFI",  "PFI.time"),
)
SUBTYPE_COLS_KEEP = [
    "Patient ID", "Sample ID", "Cancer Type", "Cancer Type Detailed",
    "Grade.1", "Subtype",
]

# ---------------------------------------------------------------------------
# 1. UCEC subtype + grade (Kandoth 2013 with PanCancer reannotation).
# ---------------------------------------------------------------------------
print(f"Loading UCEC subtype TSV: {SUBTYPE_TSV}")
subtype = pd.read_csv(SUBTYPE_TSV, sep="\t").dropna(axis=1, how="all")
subtype = subtype[SUBTYPE_COLS_KEEP].copy()
subtype = subtype.rename(columns={"Patient ID": "Patient_ID", "Grade.1": "Grade"})
subtype = subtype.drop_duplicates(subset=["Patient_ID"], keep="first")
print(f"  {len(subtype):,} unique UCEC patients with subtype labels")
print("  Subtype distribution:")
for label, count in subtype["Subtype"].value_counts(dropna=False).items():
    print(f"    {label}: {count}")

# ---------------------------------------------------------------------------
# 2. TCGA Pan-Cancer curated survival endpoints (UCEC subset).
# ---------------------------------------------------------------------------
print(f"\nLoading TCGA Pan-Cancer curated clinical: {CLINICAL_CSV}")
clin = pd.read_csv(CLINICAL_CSV)
clin.columns = [c.lstrip("﻿") for c in clin.columns]
clin = clin[clin["type"] == "UCEC"].copy()
clin["Patient_ID"] = clin["bcr_patient_barcode"].astype(str).str[:PATIENT_BARCODE_LEN]
clin = clin.drop_duplicates(subset=["Patient_ID"], keep="first")
surv_cols_keep = (
    ["Patient_ID"]
    + [src for src, _, _, _ in ENDPOINT_RENAMES]
    + [src for _, src, _, _ in ENDPOINT_RENAMES]
)
clin = clin[surv_cols_keep].copy()
# Ambiguous event (==2) -> censored (==0). Binary Int64 events, NaN preserved.
for cur_evt, _, _, _ in ENDPOINT_RENAMES:
    clin[cur_evt] = clin[cur_evt].replace(2, 0).astype("Int64")
rename_map = {}
for cur_evt, cur_t, leg_evt, leg_t in ENDPOINT_RENAMES:
    rename_map[cur_evt] = leg_evt
    rename_map[cur_t] = leg_t
clin = clin.rename(columns=rename_map)
print(f"  {len(clin):,} UCEC patient survival rows")

# ---------------------------------------------------------------------------
# 3. Merge.
# ---------------------------------------------------------------------------
metadata = subtype.merge(clin, on="Patient_ID", how="inner")
print(f"\nMerged: {len(metadata):,} patients")
print("  Survival-endpoint coverage:")
for col in ("DSS", "DFI", "PFI"):
    n = (metadata[col] == 1).sum()
    total = metadata[col].notna().sum()
    print(f"    {col}: {n}/{total} events")

# ---------------------------------------------------------------------------
# 4. Save.
# ---------------------------------------------------------------------------
metadata.to_csv(OUTPUT_CSV, index=False)
print(f"\nWrote {OUTPUT_CSV}: {len(metadata):,} rows, {len(metadata.columns)} columns")
