"""Build ``glioma_clinical_metadata.csv`` for the TESSERA glioma cohort.

Joins the WHO 2021 molecular reclassification of TCGA gliomas (Leiria et
al., 2025, *Sci. Data* 12, 5117; ``Matrix_WHO2021.csv`` next to this
script) with the standard TCGA Pan-Cancer Atlas clinical / outcome
table (``ncit.csv``) and writes one row per glioma patient with their
WHO 2021 primary class plus OS, DSS, DFI, PFI endpoints.

The output is the TCGA-side ground truth used by

  scripts/tumor_classification_snv_cna_infonce/tcga_concordance_paper_figures.py

to evaluate the manuscript's classifier-vs-pathology concordance in
glioma (Figure 4 g-h, Figure 5 a) under the WHO 2021 reclassification
that retired the historical GBM/LGG histologic split.

Inputs
------
``Matrix_WHO2021.csv`` (this directory)
    Per-patient WHO 2021 simplified labels (``classification.2021_simplified.labels``).
``../../TCGA_PanCan/ncit.csv``
    TCGA Pan-Cancer Atlas clinical resource. We pull
    ``bcr_patient_barcode`` and the four standard endpoint pairs
    (``OS``/``OS.time``, ``DSS``/``DSS.time``, ``DFI``/``DFI.time``,
    ``PFI``/``PFI.time``).

Output
------
``glioma_clinical_metadata.csv`` (this directory)
    Columns: ``Patient_ID, WHO2021, OS, OS.time, DSS, DSS.time, DFI,
    DFI.time, PFI, PFI.time``. Rows: glioma patients with both a
    WHO 2021 label and a survival row.

Citation
--------
Leiria, R. et al. (2025). Updated TCGA glioma classification according
to the 2021 WHO classification of CNS tumours. *Scientific Data* 12,
5117. ``s41597-025-05117-2``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent.resolve()
WHO2021_CSV = Path(os.environ.get("WHO2021_CSV", HERE / "Matrix_WHO2021.csv"))
NCIT_CSV = Path(os.environ.get("NCIT_CSV", HERE / ".." / ".." / "TCGA_PanCan" / "ncit.csv"))
OUTPUT_CSV = Path(os.environ.get("OUTPUT_CSV", HERE / "glioma_clinical_metadata.csv"))

WHO2021_LABEL_COL = "classification.2021_simplified.labels"
EVENT_COLS = ("OS", "DSS", "DFI", "PFI")
SURV_COLS = (
    "Patient_ID",
    "OS", "OS.time",
    "DSS", "DSS.time",
    "DFI", "DFI.time",
    "PFI", "PFI.time",
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
# 2. TCGA Pan-Cancer survival endpoints (one row per patient).
# ---------------------------------------------------------------------------
print(f"\nLoading TCGA Pan-Cancer survival: {NCIT_CSV}")
surv = pd.read_csv(NCIT_CSV)
surv["Patient_ID"] = surv["bcr_patient_barcode"].astype(str).str[:12]
surv = surv[list(SURV_COLS)].drop_duplicates(subset=["Patient_ID"], keep="first")
for c in EVENT_COLS:
    surv[c] = surv[c].astype("Int64")
print(f"  {len(surv):,} unique patient survival rows")

# ---------------------------------------------------------------------------
# 3. Inner-join on Patient_ID and write output.
# ---------------------------------------------------------------------------
out = who.merge(surv, on="Patient_ID", how="inner")
print(f"\nMerged: {len(out):,} glioma patients with WHO2021 + survival")
print(f"  with OS.time:  {out['OS.time'].notna().sum():,}")
print(f"  with DSS.time: {out['DSS.time'].notna().sum():,}")

out.to_csv(OUTPUT_CSV, index=False)
print(f"\nWrote {OUTPUT_CSV}")
