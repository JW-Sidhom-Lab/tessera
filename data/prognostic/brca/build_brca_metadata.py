"""Build ``brca_clinical_metadata.csv`` for the TESSERA BRCA cohort.

Joins three sources into a single per-patient table:

1. **PAM50 intrinsic subtype** (``Subtype`` column) from the TCGA 2012
   Cell breast paper (TCGA Network 2012, *Nature* 490, 61-70). Source
   table: ``brca_subtype.tsv``.

2. **TCGA Pan-Cancer Atlas survival endpoints** -- OS, DSS, DFI, PFI
   plus AJCC pathologic stage. Source: ``../../TCGA_PanCan/ncit.csv``.

3. **Research-reconstructed OncotypeDX Recurrence Score and MammaPrint
   risk score** from the DLRS (Deep Learning for Recurrence Score)
   release that accompanies Howard et al. 2023 (*npj Breast Cancer*,
   ``s41587-020-0546-8``). DLRS computed Oncotype RS via the Paik 2004
   21-gene formula on TCGA-BRCA RNA-seq -- a research reconstruction,
   not Genomic Health's clinical score -- and binarized at the 85th
   percentile (``H`` = top 15%). For consistency with the PRAD pipeline
   we additionally tertile-bin the continuous score into Low /
   Intermediate / High. Source: ``DLRS_tcga_brca_complete.csv``.

The output is the TCGA-side ground truth used by

  scripts/tumor_classification_snv_cna_infonce/tcga_concordance_paper_figures.py

and by Figure 5 c-j of the manuscript (UMAP overlay + joint Cox of the
TESSERA score against the OncotypeDX comparator).

Inputs
------
``brca_subtype.tsv`` (this directory)
    PAM50 calls from TCGA 2012 (column ``PAM50Call_RNAseq``).
``DLRS_tcga_brca_complete.csv`` (this directory)
    DLRS Oncotype / MammaPrint reconstruction from Howard et al. 2023.
``../../TCGA_PanCan/ncit.csv``
    TCGA Pan-Cancer Atlas clinical resource.

Output
------
``brca_clinical_metadata.csv`` (this directory)

Citations
---------
TCGA Network. (2012). Comprehensive molecular portraits of human breast
tumours. *Nature* 490, 61-70.

Howard, F. M. et al. (2023). Multimodal prediction of breast cancer
recurrence assays and risk of recurrence. *npj Breast Cancer*. DLRS
release: github.com/fmhoward/DLRS.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent.resolve()
PAM50_TSV = Path(os.environ.get("PAM50_TSV", HERE / "brca_subtype.tsv"))
DLRS_CSV = Path(os.environ.get("DLRS_CSV", HERE / "DLRS_tcga_brca_complete.csv"))
NCIT_CSV = Path(os.environ.get("NCIT_CSV", HERE / ".." / ".." / "TCGA_PanCan" / "ncit.csv"))
OUTPUT_CSV = Path(os.environ.get("OUTPUT_CSV", HERE / "brca_clinical_metadata.csv"))

PATIENT_BARCODE_LEN = 12
EVENT_COLS = ("OS", "DSS", "DFI", "PFI")
SURV_COLS = (
    "Patient_ID",
    "OS", "OS.time",
    "DSS", "DSS.time",
    "DFI", "DFI.time",
    "PFI", "PFI.time",
    "ajcc_pathologic_tumor_stage",
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
# 1. PAM50 intrinsic subtype (one row per patient).
# ---------------------------------------------------------------------------
print(f"Loading PAM50 subtypes: {PAM50_TSV}")
pam50 = pd.read_csv(PAM50_TSV, sep="\t").dropna(subset=["PAM50Call_RNAseq"])
pam50["Patient_ID"] = pam50["sample"].astype(str).str[:PATIENT_BARCODE_LEN]
pam50 = (pam50[["Patient_ID", "PAM50Call_RNAseq"]]
         .rename(columns={"PAM50Call_RNAseq": "Subtype"})
         .drop_duplicates(subset=["Patient_ID"], keep="first"))
print(f"  {len(pam50):,} unique patients")
print(pam50["Subtype"].value_counts(dropna=False).rename_axis("Subtype").to_string())

# ---------------------------------------------------------------------------
# 2. TCGA Pan-Cancer survival endpoints (BRCA subset).
# ---------------------------------------------------------------------------
print(f"\nLoading TCGA Pan-Cancer survival: {NCIT_CSV}")
surv = pd.read_csv(NCIT_CSV)
surv["Patient_ID"] = surv["bcr_patient_barcode"].astype(str).str[:PATIENT_BARCODE_LEN]
surv = surv[surv["type"] == "BRCA"][list(SURV_COLS)].drop_duplicates(
    subset=["Patient_ID"], keep="first")
for c in EVENT_COLS:
    surv[c] = surv[c].astype("Int64")
# Normalise stage strings (drop AJCC A/B/C suffixes, blank out non-codes).
surv["ajcc_pathologic_tumor_stage"] = (
    surv["ajcc_pathologic_tumor_stage"]
        .replace({"[Discrepancy]": np.nan, "[Not Available]": np.nan, "Stage X": np.nan})
        .apply(lambda s: s[:-1] if isinstance(s, str) and s[-1] in ("A", "B", "C") else s))
print(f"  {len(surv):,} BRCA patient survival rows")

# ---------------------------------------------------------------------------
# 3. DLRS OncotypeDX + MammaPrint per-patient values.
# ---------------------------------------------------------------------------
print(f"\nLoading DLRS Oncotype/MammaPrint: {DLRS_CSV}")
dlrs = pd.read_csv(DLRS_CSV).drop_duplicates("patient", keep="first")
dlrs = dlrs.rename(columns=DLRS_COLUMN_RENAMES)[list(DLRS_COLUMN_RENAMES.values())]
# Tertile-bin the continuous Oncotype RS within the DLRS cohort.
dlrs["Subtype_Oncotype"] = pd.qcut(
    dlrs["oncotype_rs_dlrs"].rank(method="first"),
    q=3, labels=["Low", "Intermediate", "High"])
print(f"  {len(dlrs):,} unique DLRS patients")
print(dlrs["Subtype_Oncotype"].value_counts().rename_axis("Subtype_Oncotype").to_string())

# ---------------------------------------------------------------------------
# 4. PAM50 + survival inner-join, then DLRS left-join (DLRS is partial).
# ---------------------------------------------------------------------------
out = pam50.merge(surv, on="Patient_ID", how="inner") \
           .merge(dlrs, on="Patient_ID", how="left")
n_dlrs_cov = out["oncotype_rs_dlrs"].notna().sum()
print(f"\nMerged: {len(out):,} BRCA patients with PAM50 + survival; "
      f"{n_dlrs_cov:,} also carry DLRS Oncotype RS")

out.to_csv(OUTPUT_CSV, index=False)
print(f"Wrote {OUTPUT_CSV}")
