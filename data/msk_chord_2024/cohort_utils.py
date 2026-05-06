"""Shared utilities for assembling TTNTD-based ground-truth cohort tables from MSK-CHORD.

The MSK-CHORD ground-truth cohort scripts (``create_ground_truth_*.py``) all
share the same skeleton: filter to a stage-restricted cancer-type cohort,
group the treatment timeline into lines of therapy, locate the first instance
of each named regimen per patient, then compute time-to-next-treatment-or-death
(TTNTD), event types, and progression dates. This module factors that common
pipeline out so each per-cohort script reduces to a small spec (cancer type,
regimen list, first-line-only flag) plus a single call to
``build_ttntd_cohort``.

TTNTD definition. For each (patient, regimen-instance) pair, TTNTD is the
number of days from regimen start to the earlier of: (a) the next treatment
start date observed in the timeline strictly after the regimen's stop date;
or (b) the patient's death date (computed as ``OS_MONTHS * 30.44`` when
``OS_STATUS`` is ``1:DECEASED``). Patients with neither event observed are
flagged ``TTNTD_IS_CENSORED = True``.

Response label. ``TTNTD_RESPONSE_LABEL`` is 1 if TTNTD > 180 days, 0 if
TTNTD <= 180 days. Censored patients with treatment duration >= 180 days who
are alive get a 1 (sustained-benefit responder); other censored patients get
NaN.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

# cBioPortal patient/sample tables prepend four metadata header lines.
CBIOPORTAL_CLINICAL_SKIPROWS = 4

# Mean days per month, used to convert OS_MONTHS to a date offset comparable
# with the timeline's day-offset units.
DAYS_PER_MONTH = 30.44

# Treatments started within this many days of the previous treatment are
# treated as part of the same line of therapy.
LINE_OF_THERAPY_GAP_DAYS = 30

# Threshold separating responders from non-responders on TTNTD (days).
RESPONDER_TTNTD_DAYS = 180

# Output column order for the ground-truth CSV.
OUTPUT_COLS = [
    "PATIENT_ID",
    "REGIMEN",
    "AGENT_COMBINATION",
    "LINE_OF_THERAPY",
    "TREATMENT_START",
    "TREATMENT_STOP",
    "TREATMENT_DURATION_DAYS",
    "STAGE",
    "AGE_AT_DIAGNOSIS",
    "GENDER",
    "PRIOR_MED_TO_MSK",
    "OS_MONTHS",
    "OS_MONTHS_FROM_TREATMENT_START",
    "OS_STATUS",
    "NEXT_TREATMENT_DATE",
    "NEXT_AGENT",
    "DEATH_DATE",
    "EVENT_TYPE",
    "TTNTD_DAYS",
    "TTNTD_MONTHS",
    "TTNTD_IS_CENSORED",
    "TTNTD_RESPONSE_LABEL",
    "TTNTD_RESPONSE_CATEGORY",
    "FIRST_PROGRESSION_DATE",
    "DAYS_TO_PROGRESSION",
    "MONTHS_TO_PROGRESSION",
]

logger = logging.getLogger(__name__)


@dataclass
class Regimen:
    """Definition of a chemotherapy regimen for cohort matching.

    Attributes:
        label: Short name written into the ``REGIMEN`` column (e.g. ``"FOLFOX"``).
        combination: Drug-combination string for the ``AGENT_COMBINATION``
            column (e.g. ``"FLUOROURACIL+LEUCOVORIN+OXALIPLATIN"``).
        required_agents: Component agents that must all appear within a single
            line of therapy for the regimen to match (uppercase). Matching uses
            ``set.issubset`` against the line's unique agent set.
        included_agents: The specific MSK-CHORD ``AGENT`` labels to aggregate
            when computing the regimen's ``tx_start`` (min) and ``tx_stop``
            (max) dates. Often equal to ``required_agents`` written as a list,
            but kept separate so a regimen can require a generic agent
            (``PACLITAXEL PROTEIN-BOUND``) while excluding others
            (``IRINOTECAN LIPOSOMAL``).
    """

    label: str
    combination: str
    required_agents: frozenset
    included_agents: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.included_agents:
            self.included_agents = list(self.required_agents)


def load_inputs(base_dir: Path) -> dict:
    """Load the four MSK-CHORD raw files needed for cohort assembly."""
    base_dir = Path(base_dir)
    return {
        "clinical_patient": pd.read_csv(
            base_dir / "data_clinical_patient.txt",
            sep="\t", skiprows=CBIOPORTAL_CLINICAL_SKIPROWS,
        ),
        "clinical_sample": pd.read_csv(
            base_dir / "data_clinical_sample.txt",
            sep="\t", skiprows=CBIOPORTAL_CLINICAL_SKIPROWS,
        ),
        "treatments": pd.read_csv(
            base_dir / "data_timeline_treatment.txt", sep="\t",
        ),
        "progression": pd.read_csv(
            base_dir / "data_timeline_progression.txt", sep="\t",
        ),
    }


def select_cohort_patients(clinical_patient: pd.DataFrame,
                           clinical_sample: pd.DataFrame,
                           cancer_type: str,
                           stage: str = "Stage 4") -> np.ndarray:
    """Return the unique patient IDs for the stage-restricted cancer-type cohort."""
    stage_patients = clinical_patient.loc[
        clinical_patient["STAGE_HIGHEST_RECORDED"] == stage, "PATIENT_ID"
    ]
    cohort_samples = clinical_sample[
        clinical_sample["PATIENT_ID"].isin(stage_patients) &
        (clinical_sample["CANCER_TYPE"] == cancer_type)
    ]
    return cohort_samples["PATIENT_ID"].unique()


def assign_lines_of_therapy(treatments: pd.DataFrame,
                            patient_ids: Iterable[str],
                            gap_days: int = LINE_OF_THERAPY_GAP_DAYS) -> pd.DataFrame:
    """Annotate each treatment row with its line-of-therapy index.

    Treatments that begin within ``gap_days`` of the previous treatment for
    the same patient are assigned to the same line.
    """
    df = treatments[treatments["PATIENT_ID"].isin(patient_ids)].copy()
    df = df.sort_values(["PATIENT_ID", "START_DATE"]).reset_index(drop=True)

    df["line_of_therapy"] = 0
    for _, group in df.groupby("PATIENT_ID", sort=False):
        idx = group.index.to_numpy()
        starts = group["START_DATE"].to_numpy()
        lines = np.empty(len(group), dtype=int)
        lines[0] = 1
        for i in range(1, len(group)):
            current, prev = starts[i], starts[i - 1]
            if pd.isna(current) or pd.isna(prev):
                lines[i] = lines[i - 1]
            elif abs(current - prev) <= gap_days:
                lines[i] = lines[i - 1]
            else:
                lines[i] = lines[i - 1] + 1
        df.loc[idx, "line_of_therapy"] = lines

    return df


def find_regimen_instances(treatments_with_lot: pd.DataFrame,
                           regimens: List[Regimen],
                           first_line_only: bool = False) -> pd.DataFrame:
    """Locate each patient's first instance of each named regimen.

    For each (patient, line-of-therapy) pair, checks whether any regimen's
    ``required_agents`` is a subset of the line's unique agents. The first
    matching regimen wins (regimens are evaluated in list order). When
    ``first_line_only`` is True, only ``line_of_therapy == 1`` is considered.
    """
    records = []
    for patient_id, patient_rows in treatments_with_lot.groupby("PATIENT_ID", sort=False):
        if first_line_only:
            patient_rows = patient_rows[patient_rows["line_of_therapy"] == 1]
            if patient_rows.empty:
                continue
            line_iter = [(1, patient_rows)]
        else:
            line_iter = list(patient_rows.groupby("line_of_therapy"))

        for line, line_rows in line_iter:
            agents_upper = {a.upper() for a in line_rows["AGENT"].dropna()}
            for regimen in regimens:
                if regimen.required_agents.issubset(agents_upper):
                    component_rows = line_rows[
                        line_rows["AGENT"].isin(regimen.included_agents)
                    ]
                    records.append({
                        "patient_id": patient_id,
                        "regimen": regimen.label,
                        "agent_combination": regimen.combination,
                        "line_of_therapy": line,
                        "tx_start": component_rows["START_DATE"].min(),
                        "tx_stop": component_rows["STOP_DATE"].max(),
                    })
                    break  # one regimen per line

    df = pd.DataFrame(records)
    if df.empty:
        return df
    df = df.sort_values(["patient_id", "tx_start"])
    df = df.drop_duplicates(subset=["patient_id", "regimen"], keep="first")
    return df.reset_index(drop=True)


def _compute_ttntd_for_patient(regimen_row: pd.Series,
                               patient_treatments: pd.DataFrame,
                               patient_clinical: pd.Series) -> dict:
    """Compute TTNTD and related fields for one (patient, regimen) instance."""
    tx_start = regimen_row["tx_start"]
    tx_stop = regimen_row["tx_stop"]

    next_treatments = patient_treatments[
        patient_treatments["START_DATE"] > tx_stop
    ].sort_values("START_DATE")
    next_treatment_date = (next_treatments["START_DATE"].iloc[0]
                           if not next_treatments.empty else None)
    next_agent = (next_treatments["AGENT"].iloc[0]
                  if not next_treatments.empty else None)

    os_months = patient_clinical.get("OS_MONTHS", np.nan)
    os_status = patient_clinical.get("OS_STATUS", "Unknown")
    is_deceased = (os_status == "1:DECEASED")
    death_date = (os_months * DAYS_PER_MONTH) if pd.notna(os_months) else None

    events = []
    if next_treatment_date is not None:
        events.append(("next_treatment", next_treatment_date))
    if is_deceased and death_date is not None:
        events.append(("death", death_date))

    if events:
        event_type, event_date = min(events, key=lambda x: x[1])
        ttntd_days = int(event_date - tx_start)
        ttntd_months = round(ttntd_days / DAYS_PER_MONTH, 2)
        ttntd_is_censored = False
    else:
        event_type = None
        ttntd_days = np.nan
        ttntd_months = np.nan
        ttntd_is_censored = True

    if pd.notna(ttntd_days):
        if ttntd_days > RESPONDER_TTNTD_DAYS:
            response_label, response_category = 1, "Responder"
        else:
            response_label, response_category = 0, "Non-responder"
    elif (ttntd_is_censored and pd.notna(tx_stop)
          and (tx_stop - tx_start) >= RESPONDER_TTNTD_DAYS
          and (os_status == "0:LIVING" or not is_deceased)):
        response_label, response_category = 1, "Responder (sustained on therapy)"
    else:
        response_label, response_category = np.nan, "Censored"

    os_from_tx = (os_months - tx_start / DAYS_PER_MONTH) if pd.notna(os_months) else np.nan
    duration_days = (tx_stop - tx_start) if pd.notna(tx_stop) else np.nan

    return {
        "TREATMENT_START": tx_start,
        "TREATMENT_STOP": tx_stop,
        "TREATMENT_DURATION_DAYS": duration_days,
        "OS_MONTHS": os_months,
        "OS_MONTHS_FROM_TREATMENT_START": os_from_tx,
        "OS_STATUS": os_status,
        "NEXT_TREATMENT_DATE": next_treatment_date,
        "NEXT_AGENT": next_agent,
        "DEATH_DATE": death_date,
        "EVENT_TYPE": event_type,
        "TTNTD_DAYS": ttntd_days,
        "TTNTD_MONTHS": ttntd_months,
        "TTNTD_IS_CENSORED": ttntd_is_censored,
        "TTNTD_RESPONSE_LABEL": response_label,
        "TTNTD_RESPONSE_CATEGORY": response_category,
    }


def _compute_progression(regimen_row: pd.Series,
                         patient_progression: pd.DataFrame) -> dict:
    """First post-treatment-start progression date and offsets for one patient."""
    tx_start = regimen_row["tx_start"]
    after_start = patient_progression[
        (patient_progression["START_DATE"] >= tx_start) &
        (patient_progression["PROGRESSION"] == "Y")
    ]
    first_prog = after_start["START_DATE"].min() if not after_start.empty else np.nan
    days_to_prog = int(first_prog - tx_start) if pd.notna(first_prog) else np.nan
    months_to_prog = round(days_to_prog / DAYS_PER_MONTH, 2) if pd.notna(days_to_prog) else np.nan
    return {
        "FIRST_PROGRESSION_DATE": first_prog,
        "DAYS_TO_PROGRESSION": days_to_prog,
        "MONTHS_TO_PROGRESSION": months_to_prog,
    }


def assemble_ground_truth(regimen_instances: pd.DataFrame,
                          all_treatments: pd.DataFrame,
                          clinical_patient: pd.DataFrame,
                          progression: pd.DataFrame) -> pd.DataFrame:
    """Compute TTNTD, progression, and clinical fields for every regimen instance."""
    clinical_by_patient = clinical_patient.set_index("PATIENT_ID")
    prior_meds = clinical_by_patient["PRIOR_MED_TO_MSK"]

    records = []
    for _, regimen_row in regimen_instances.iterrows():
        pid = regimen_row["patient_id"]
        clinical = clinical_by_patient.loc[pid]
        if isinstance(clinical, pd.DataFrame):
            clinical = clinical.iloc[0]
        patient_tx = all_treatments[all_treatments["PATIENT_ID"] == pid]
        patient_prog = progression[progression["PATIENT_ID"] == pid]

        record = {
            "PATIENT_ID": pid,
            "REGIMEN": regimen_row["regimen"],
            "AGENT_COMBINATION": regimen_row["agent_combination"],
            "LINE_OF_THERAPY": regimen_row["line_of_therapy"],
            "STAGE": clinical.get("STAGE_HIGHEST_RECORDED", "Unknown"),
            "AGE_AT_DIAGNOSIS": clinical.get("AGE_AT_DIAGNOSIS", np.nan),
            "GENDER": clinical.get("GENDER", "Unknown"),
            "PRIOR_MED_TO_MSK": prior_meds.get(pid, np.nan),
        }
        record.update(_compute_ttntd_for_patient(regimen_row, patient_tx, clinical))
        record.update(_compute_progression(regimen_row, patient_prog))
        records.append(record)

    df = pd.DataFrame(records)
    return df[OUTPUT_COLS] if not df.empty else df


def summarise(df: pd.DataFrame) -> None:
    """Log cohort statistics to INFO."""
    if df.empty:
        logger.info("Empty cohort.")
        return

    responders = (df["TTNTD_RESPONSE_LABEL"] == 1).sum()
    non_responders = (df["TTNTD_RESPONSE_LABEL"] == 0).sum()
    censored = df["TTNTD_IS_CENSORED"].sum()
    evaluable = responders + non_responders

    logger.info("Total regimen instances: %d (patients: %d)",
                len(df), df["PATIENT_ID"].nunique())
    if evaluable:
        logger.info("Evaluable: %d  responders=%d (%.1f%%)  non-responders=%d (%.1f%%)",
                    evaluable, responders, 100 * responders / evaluable,
                    non_responders, 100 * non_responders / evaluable)
    logger.info("Censored: %d (%.1f%% of total)",
                censored, 100 * censored / len(df))

    for regimen, group in df.groupby("REGIMEN"):
        ev = group[~group["TTNTD_IS_CENSORED"]]
        if len(ev):
            rate = 100 * (ev["TTNTD_RESPONSE_LABEL"] == 1).sum() / len(ev)
            logger.info("  %s: total=%d evaluable=%d response=%.1f%%",
                        regimen, len(group), len(ev), rate)


def build_ttntd_cohort(base_dir: Path,
                       cancer_type: str,
                       regimens: List[Regimen],
                       output_path: Path,
                       first_line_only: bool = False,
                       stage: str = "Stage 4") -> pd.DataFrame:
    """End-to-end cohort assembly: load inputs, build cohort, write CSV, return frame."""
    logger.info("Loading MSK-CHORD inputs from %s", base_dir)
    raw = load_inputs(base_dir)

    patient_ids = select_cohort_patients(
        raw["clinical_patient"], raw["clinical_sample"],
        cancer_type=cancer_type, stage=stage,
    )
    logger.info("Cohort patients (%s, %s): %d", stage, cancer_type, len(patient_ids))

    treatments_with_lot = assign_lines_of_therapy(raw["treatments"], patient_ids)
    logger.info("Treatments for cohort: %d (%d patients)",
                len(treatments_with_lot),
                treatments_with_lot["PATIENT_ID"].nunique())

    regimen_instances = find_regimen_instances(
        treatments_with_lot, regimens, first_line_only=first_line_only,
    )
    logger.info("Regimen instances: %d (%d unique patients)",
                len(regimen_instances),
                regimen_instances["patient_id"].nunique() if not regimen_instances.empty else 0)

    df = assemble_ground_truth(
        regimen_instances, raw["treatments"],
        raw["clinical_patient"], raw["progression"],
    )
    summarise(df)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Writing %d rows to %s", len(df), output_path)
    df.to_csv(output_path, index=False)
    return df
