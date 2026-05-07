"""Build ``glioma_clinical_metadata.csv`` for the TESSERA glioma cohort.

Joins the WHO 2021 molecular reclassification of TCGA gliomas (Leiria et
al., 2025, *Sci. Data* 12, 5117; ``Matrix_WHO2021.csv`` next to this
script) with the Liu 2018 curated TCGA Pan-Cancer Atlas survival
endpoints (``../../TCGA_PanCan/clinical.csv``) and writes one row per
glioma patient with their WHO 2021 primary class plus DSS, DFI, PFI
endpoints.

The output is the TCGA-side ground truth used by

  scripts/tumor_classification_snv_cna_infonce/tcga_concordance_paper_figures.py

to evaluate the manuscript's classifier-vs-pathology concordance in
glioma (Figure 4 g-h, Figure 5 a) under the WHO 2021 reclassification
that retired the historical GBM/LGG histologic split.

Inputs
------
``Matrix_WHO2021.csv`` (this directory)
    Per-patient WHO 2021 simplified labels (``classification.2021_simplified.labels``).
``../../TCGA_PanCan/clinical.csv``
    TCGA Pan-Cancer Atlas curated clinical resource (Liu 2018-style).
    The build reads the three curated endpoint pairs
    (``DSS_cr``/``DSS.time.cr``, ``DFI.cr``/``DFI.time.cr``,
    ``PFI.cr``/``PFI.time.cr``) and ``bcr_patient_barcode``.

Output
------
``glioma_clinical_metadata.csv`` (this directory)
    Columns: ``Patient_ID, WHO2021, DSS, DSS.time, DFI, DFI.time, PFI,
    PFI.time``. Rows: glioma patients with both a WHO 2021 label and a
    survival row.

Liu 2018 ambiguous-event handling
---------------------------------
The curated ``DSS_cr`` / ``DFI.cr`` / ``PFI.cr`` columns use a tri-state
encoding: ``0`` = censored, ``1`` = event, ``2`` = ambiguous cause-of-
death (uncodable). This build remaps ``2 -> 0`` before writing so the
output event columns stay strictly binary, matching the convention
``cph.fit`` and ``KaplanMeierFitter.fit`` expect. The same convention
is used by ``tcga_concordance_paper_figures.py`` (Figure 4 d-h),
keeping the entire pipeline aligned. The remap is conservative
(uncodable -> not an event), so the event set going into Cox / KM is
identical to what raw ``DSS`` / ``DFI`` / ``PFI`` from ``ncit.csv``
would have produced.

Citation
--------
Leiria, R. et al. (2025). Updated TCGA glioma classification according
to the 2021 WHO classification of CNS tumours. *Scientific Data* 12,
5117. ``s41597-025-05117-2``.

Liu, J. et al. (2018). An integrated TCGA pan-cancer clinical data
resource to drive high-quality survival outcome analytics. *Cell* 173,
400-416.e11.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent.resolve()
WHO2021_CSV = Path(os.environ.get("WHO2021_CSV", HERE / "Matrix_WHO2021.csv"))
CLINICAL_CSV = Path(os.environ.get(
    "CLINICAL_CSV", HERE / ".." / ".." / "TCGA_PanCan" / "clinical.csv"))
OUTPUT_CSV = Path(os.environ.get("OUTPUT_CSV", HERE / "glioma_clinical_metadata.csv"))

WHO2021_LABEL_COL = "classification.2021_simplified.labels"

# Liu 2018 -> legacy schema: each (curated_event_col, curated_time_col,
# legacy_event_col, legacy_time_col).
ENDPOINT_RENAMES = (
    ("DSS_cr", "DSS.time.cr", "DSS", "DSS.time"),
    ("DFI.cr", "DFI.time.cr", "DFI", "DFI.time"),
    ("PFI.cr", "PFI.time.cr", "PFI", "PFI.time"),
)

# ---------------------------------------------------------------------------
# 1. WHO 2021 simplified labels (one row per patient).
# ---------------------------------------------------------------------------
print(f"Loading WHO 2021 labels: {WHO2021_CSV}")
who = pd.read_csv(WHO2021_CSV)[["Patient_ID", WHO2021_LABEL_COL]].copy()
who = who.rename(columns={WHO2021_LABEL_COL: "WHO2021"})
who = who.drop_duplicates(subset=["Patient_ID"], keep="first")
print(f"  {len(who):,} unique patients")
print(who["WHO2021"].value_counts().rename_axis("WHO2021").to_string())

# ---------------------------------------------------------------------------
# 2. TCGA Pan-Cancer curated survival endpoints (one row per patient).
# ---------------------------------------------------------------------------
print(f"\nLoading TCGA Pan-Cancer curated clinical: {CLINICAL_CSV}")
clin = pd.read_csv(CLINICAL_CSV)
clin.columns = [c.lstrip("﻿") for c in clin.columns]
clin["Patient_ID"] = clin["bcr_patient_barcode"].astype(str).str[:12]
clin = clin.drop_duplicates(subset=["Patient_ID"], keep="first")
surv_cols_keep = ["Patient_ID"] + [src for src, _, _, _ in ENDPOINT_RENAMES] \
                                + [src for _, src, _, _ in ENDPOINT_RENAMES]
clin = clin[surv_cols_keep].copy()

# Liu 2018 ambiguous events (== 2) -> censored (== 0). Cast to Int64 so
# the event columns serialise as integers and survive any NaN entries.
for cur_evt, _, _, _ in ENDPOINT_RENAMES:
    clin[cur_evt] = clin[cur_evt].replace(2, 0).astype("Int64")

# Rename to legacy non-_cr column names.
rename_map = {}
for cur_evt, cur_t, leg_evt, leg_t in ENDPOINT_RENAMES:
    rename_map[cur_evt] = leg_evt
    rename_map[cur_t] = leg_t
clin = clin.rename(columns=rename_map)
print(f"  {len(clin):,} unique patient survival rows")

# ---------------------------------------------------------------------------
# 3. Inner-join on Patient_ID and write output (event/time pairs interleaved
#    to match the legacy schema downstream code expects).
# ---------------------------------------------------------------------------
out = who.merge(clin, on="Patient_ID", how="inner")
out_cols = ["Patient_ID", "WHO2021"]
for _, _, leg_evt, leg_t in ENDPOINT_RENAMES:
    out_cols += [leg_evt, leg_t]
out = out[out_cols]
print(f"\nMerged: {len(out):,} glioma patients with WHO2021 + curated survival")
print(f"  with DSS.time:  {out['DSS.time'].notna().sum():,}")

out.to_csv(OUTPUT_CSV, index=False)
print(f"\nWrote {OUTPUT_CSV}")
