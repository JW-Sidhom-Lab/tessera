"""Build ``brca_clinical_metadata.csv`` for the TESSERA BRCA cohort.

Joins two sources into a single per-patient table:

1. **Liu 2018 curated TCGA Pan-Cancer Atlas survival endpoints** -- DSS,
   DFI, PFI from ``../../TCGA_PanCan/clinical.csv``, restricted to the
   BRCA subset.

2. **Research-reconstructed OncotypeDX Recurrence Score and MammaPrint
   risk score** from the DLRS (Deep Learning for Recurrence Score)
   release that accompanies Howard et al. 2023 (*npj Breast Cancer*).
   DLRS computed Oncotype RS via the Paik 2004 21-gene formula on
   TCGA-BRCA RNA-seq -- a research reconstruction, not Genomic Health's
   clinical score -- and binarized at the 85th percentile (``H`` = top
   15%). For consistency with the PRAD pipeline we additionally
   tertile-bin the continuous score into Low / Intermediate / High.
   Source: ``DLRS_tcga_brca_complete.csv``.

The output is the TCGA-side ground truth used by Figure 5 c-j of the
manuscript (UMAP overlay + joint Cox of the TESSERA score against the
OncotypeDX comparator).

Inputs
------
``DLRS_tcga_brca_complete.csv`` (this directory)
    DLRS Oncotype / MammaPrint reconstruction from Howard et al. 2023.
``../../TCGA_PanCan/clinical.csv``
    TCGA Pan-Cancer Atlas curated clinical resource (Liu 2018-style).

Output
------
``brca_clinical_metadata.csv`` (this directory)

Liu 2018 ambiguous-event handling
---------------------------------
The curated ``DSS_cr`` / ``DFI.cr`` / ``PFI.cr`` columns use a tri-state
encoding: ``0`` = censored, ``1`` = event, ``2`` = ambiguous cause-of-
death (uncodable). This build remaps ``2 -> 0`` before writing so the
output event columns stay strictly binary, matching the convention
``cph.fit`` and ``KaplanMeierFitter.fit`` expect.

Citations
---------
Howard, F. M. et al. (2023). Multimodal prediction of breast cancer
recurrence assays and risk of recurrence. *npj Breast Cancer*. DLRS
release: github.com/fmhoward/DLRS.

Liu, J. et al. (2018). An integrated TCGA pan-cancer clinical data
resource. *Cell* 173, 400-416.e11.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent.resolve()
DLRS_CSV = Path(os.environ.get("DLRS_CSV", HERE / "DLRS_tcga_brca_complete.csv"))
CLINICAL_CSV = Path(os.environ.get(
    "CLINICAL_CSV", HERE / ".." / ".." / "TCGA_PanCan" / "clinical.csv"))
OUTPUT_CSV = Path(os.environ.get("OUTPUT_CSV", HERE / "brca_clinical_metadata.csv"))

PATIENT_BARCODE_LEN = 12
ENDPOINT_RENAMES = (
    ("DSS_cr", "DSS.time.cr", "DSS", "DSS.time"),
    ("DFI.cr", "DFI.time.cr", "DFI", "DFI.time"),
    ("PFI.cr", "PFI.time.cr", "PFI", "PFI.time"),
)
DLRS_COLUMN_RENAMES = {
    "patient":   "Patient_ID",
    "odx_train": "oncotype_rs_dlrs",
    "odx85":     "oncotype_high_85th",
    "mp_train":  "mammaprint_score_dlrs",
    "mphr":      "mammaprint_high",
    "mpulr":     "mammaprint_ultra_low",
    "mpuhr":     "mammaprint_ultra_high",
}

# ---------------------------------------------------------------------------
# 1. TCGA Pan-Cancer curated survival endpoints (BRCA subset).
# ---------------------------------------------------------------------------
print(f"Loading TCGA Pan-Cancer curated clinical: {CLINICAL_CSV}")
clin = pd.read_csv(CLINICAL_CSV)
clin.columns = [c.lstrip("﻿") for c in clin.columns]
clin = clin[clin["type"] == "BRCA"].copy()
clin["Patient_ID"] = clin["bcr_patient_barcode"].astype(str).str[:PATIENT_BARCODE_LEN]
clin = clin.drop_duplicates(subset=["Patient_ID"], keep="first")
surv_cols_keep = ["Patient_ID"] + [src for src, _, _, _ in ENDPOINT_RENAMES] \
                                + [src for _, src, _, _ in ENDPOINT_RENAMES]
clin = clin[surv_cols_keep].copy()
# Ambiguous event (==2) -> censored (==0). Binary Int64 events, NaN preserved.
for cur_evt, _, _, _ in ENDPOINT_RENAMES:
    clin[cur_evt] = clin[cur_evt].replace(2, 0).astype("Int64")
rename_map = {}
for cur_evt, cur_t, leg_evt, leg_t in ENDPOINT_RENAMES:
    rename_map[cur_evt] = leg_evt
    rename_map[cur_t] = leg_t
clin = clin.rename(columns=rename_map)
print(f"  {len(clin):,} BRCA patient survival rows")

# ---------------------------------------------------------------------------
# 2. DLRS OncotypeDX + MammaPrint per-patient values.
# ---------------------------------------------------------------------------
print(f"\nLoading DLRS Oncotype/MammaPrint: {DLRS_CSV}")
dlrs = pd.read_csv(DLRS_CSV).drop_duplicates("patient", keep="first")
dlrs = dlrs.rename(columns=DLRS_COLUMN_RENAMES)[list(DLRS_COLUMN_RENAMES.values())]
dlrs["Subtype_Oncotype"] = pd.qcut(
    dlrs["oncotype_rs_dlrs"].rank(method="first"),
    q=3, labels=["Low", "Intermediate", "High"])
print(f"  {len(dlrs):,} unique DLRS patients")
print(dlrs["Subtype_Oncotype"].value_counts().rename_axis("Subtype_Oncotype").to_string())

# ---------------------------------------------------------------------------
# 3. Survival left-joined with DLRS (DLRS is partial; no PAM50 filter --
#    the manuscript Figure 5 BRCA analyses use OncotypeDX, not PAM50).
#    Event/time pairs interleaved to match the legacy schema.
# ---------------------------------------------------------------------------
out = clin.merge(dlrs, on="Patient_ID", how="left")
front = ["Patient_ID"]
for _, _, leg_evt, leg_t in ENDPOINT_RENAMES:
    front += [leg_evt, leg_t]
trail = [c for c in out.columns if c not in front]
out = out[front + trail]

n_dlrs_cov = out["oncotype_rs_dlrs"].notna().sum()
print(f"\nMerged: {len(out):,} BRCA patients with curated survival; "
      f"{n_dlrs_cov:,} also carry DLRS Oncotype RS")

out.to_csv(OUTPUT_CSV, index=False)
print(f"Wrote {OUTPUT_CSV}")
