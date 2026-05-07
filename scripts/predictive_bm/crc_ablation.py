"""CRC three-way ablation: LOCAL vs LOCAL+GLOBAL vs FULL.

Tests whether the FM's predictive signal in CRC FOLFOX-vs-FOLFIRI lives in:
  - LOCAL only (intrinsic + local sequence attention) — context-INDEPENDENT
  - LOCAL+GLOBAL (adds within-modality self-attention) — patient-context
  - FULL (adds InfoNCE projection) — implicit SNV+CNA cross-modal alignment

Outputs:
  results/ablation/crc_predictions_{local,local_global,full}.csv
  results/ablation/crc_ablation_summary.tsv
  results/ablation/crc_ablation_figure.{png,pdf}

Sign convention (CRC):
  arm = 1 = FOLFOX (n=1204), arm = 0 = FOLFIRI (n=248)
  HIGH τ̂ → above τ̂_0 → FOLFOX-favored (arm=1)
  LOW  τ̂ → below τ̂_0 → FOLFIRI-favored (arm=0)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core.cohorts import build_crc_met
from _ablation_lib import run_cohort

DATA = ROOT / ".." / ".." / "data" / "msk_chord_2024"
GT = DATA / "GROUND_TRUTH_CRC_FOLFOX_FOLFIRI_STAGE4_TTNTD.csv"
OUT_DIR = ROOT / "results" / "ablation"

# Validated CRC settings (from crc_discovery.py)
CRC_CFG = dict(
    pca_var=0.99,
    K_mu1=2, sp_mu1=0.5,
    K_mu0=2, sp_mu0=0.85,
    K_e=3,   sp_e=0.0,
    K_tau=5, sp_tau=0.85,
)

if __name__ == "__main__":
    run_cohort(
        cohort_name="CRC",
        gt_path=GT,
        builder=build_crc_met,
        cfg=CRC_CFG,
        arm1_label="FOLFOX",
        arm0_label="FOLFIRI",
        out_dir=OUT_DIR,
    )
