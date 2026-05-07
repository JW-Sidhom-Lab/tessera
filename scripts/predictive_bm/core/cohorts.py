"""Build CRC and PDAC analysis cohorts from MSK CHORD ground-truth tables.

Each builder:
  - Reads a TTNTD-format GT CSV
  - Filters to LINE_OF_THERAPY == 1
  - Maps REGIMEN to a binary arm column (1 = primary arm of interest)
  - Builds PFS time/event from DAYS_TO_PROGRESSION (capped at horizon)
  - Builds OS time/event from OS_MONTHS_FROM_TREATMENT_START (capped at 60mo)
  - Restricts to patients with foundation-model embedding features

Returns (df, X) where X = feat.loc[df['pid']] as a numpy array.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DAYS_PER_MO = 30.4375


def _pfs_time_event(days: np.ndarray, horizon_mo: float):
    horizon_d = horizon_mo * DAYS_PER_MO
    t = days.astype(float).copy()
    e = np.ones_like(t, dtype=int)
    over = t > horizon_d
    t[over] = horizon_d
    e[over] = 0
    return t / DAYS_PER_MO, e


def _is_dead(s: object) -> int:
    s_up = str(s).upper()
    return int("DECEASED" in s_up or s_up.startswith("1:") or s_up == "1")


def _build_one(gt_csv: Path, feat: pd.DataFrame, *,
                arm1_label: str, horizon_mo: float, os_horizon_mo: float = 60.0):
    g = pd.read_csv(gt_csv)
    g = g[g["LINE_OF_THERAPY"] == 1].copy()
    g["pid"] = g["PATIENT_ID"].str[:9]
    g["arm"] = (g["REGIMEN"] == arm1_label).astype(int)
    g["pfs_days"] = pd.to_numeric(g["DAYS_TO_PROGRESSION"], errors="coerce")
    g["os_mo"]    = pd.to_numeric(g["OS_MONTHS_FROM_TREATMENT_START"], errors="coerce")
    g["os_event"] = g["OS_STATUS"].apply(_is_dead)
    g = g[g["pid"].isin(feat.index)]
    g = g[g["pfs_days"].notna()].reset_index(drop=True)

    pfs_t, pfs_e = _pfs_time_event(g["pfs_days"].values, horizon_mo)
    g["pfs_t"] = pfs_t
    g["pfs_e"] = pfs_e
    os_clip = np.minimum(g["os_mo"].astype(float).values, os_horizon_mo)
    os_e_capped = ((g["os_mo"].astype(float).values <= os_horizon_mo) &
                    (g["os_event"].values == 1)).astype(int)
    g["os_t"] = os_clip
    g["os_e"] = os_e_capped

    X = feat.loc[g["pid"]].values.astype(float)
    return g, X


def build_crc_met(gt_csv, feat: pd.DataFrame, horizon_months: float = 36.0):
    """CRC stage IV first-line FOLFOX (arm=1) vs FOLFIRI (arm=0); PFS endpoint."""
    return _build_one(Path(gt_csv), feat, arm1_label="FOLFOX",
                       horizon_mo=horizon_months)


def build_pdac_met(gt_csv, feat: pd.DataFrame, horizon_months: float = 36.0):
    """PDAC stage IV first-line FOLFIRINOX (arm=1) vs Gem/Abraxane (arm=0); PFS endpoint."""
    return _build_one(Path(gt_csv), feat, arm1_label="FOLFIRINOX",
                       horizon_mo=horizon_months)
