"""PDAC three-way ablation: LOCAL vs LOCAL+GLOBAL vs FULL.

Tests whether the FM's predictive signal in PDAC FOLFIRINOX-vs-Gem/Abraxane
lives in:
  - LOCAL only (intrinsic + local sequence attention) — context-INDEPENDENT
  - LOCAL+GLOBAL (adds within-modality self-attention) — patient-context
  - FULL (adds InfoNCE projection) — implicit SNV+CNA cross-modal alignment

Outputs:
  results/ablation/pdac_predictions_{local,local_global,full}.csv
  results/ablation/pdac_ablation_summary.tsv
  results/ablation/pdac_ablation_figure.{png,pdf}

Sign convention (PDAC):
  arm = 1 = FFX (FOLFIRINOX), arm = 0 = GA (Gem/Abraxane)
  HIGH τ̂ → above τ̂_0 → FFX-favored (arm=1)
  LOW  τ̂ → below τ̂_0 → GA-favored (arm=0; super-responder direction)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core.cohorts import build_pdac_met
from _ablation_lib import run_cohort

DATA = ROOT / ".." / ".." / "data" / "msk_chord_2024"
GT = DATA / "GROUND_TRUTH_PANCREATIC_GEMABRA_FOLFIRINOX_STAGE4_FIRST_LINE_TTNTD.csv"
OUT_DIR = ROOT / "results" / "ablation"

# Validated PDAC settings (from pdac_discovery.py)
PDAC_CFG = dict(
    pca_var=0.99,
    K_mu1=2, sp_mu1=0.5,
    K_mu0=3, sp_mu0=0.85,
    K_e=2,   sp_e=0.0,
    K_tau=5, sp_tau=0.85,
)

if __name__ == "__main__":
    run_cohort(
        cohort_name="PDAC",
        gt_path=GT,
        builder=build_pdac_met,
        cfg=PDAC_CFG,
        arm1_label="FFX",
        arm0_label="GA",
        out_dir=OUT_DIR,
    )
