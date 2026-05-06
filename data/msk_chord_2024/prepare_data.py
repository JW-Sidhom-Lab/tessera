"""Build the MSK-CHORD variant + clinical and treatment-regimen tables used downstream by TESSERA.

Reads the cBioPortal-format MSK-CHORD 2024 release and produces two CSVs:

1. ``msk_chord_2024.csv`` -- the somatic mutations table joined to sample-level
   and patient-level clinical metadata. This is the table read by
   ``methods/data/msk_chord/create_snv.py`` to build the per-sample SNV table
   for TESSERA inference.

2. ``msk_chord_2024_tx.csv`` -- the treatment-regimen timeline collapsed to one
   row per (patient, regimen-start-date), with ``line_of_therapy`` numbered
   per patient and the patient's ``PRIOR_MED_TO_MSK`` flag joined back in.
   This table is consumed by the predictive-biomarker scripts in
   ``methods/predictive_bm/`` for cohort assembly.

Inputs
------
``data_clinical_patient.txt``
    cBioPortal patient-level clinical table (four metadata header lines).
``data_clinical_sample.txt``
    cBioPortal sample-level clinical table (four metadata header lines).
``data_mutations.txt``
    Somatic-mutation MAF (no metadata header).
``data_timeline_treatment.txt``
    Treatment timeline (no metadata header). One row per administered agent
    per visit; collapsed here to one row per regimen start date.

All four files come from the MSK-CHORD 2024 release on Synapse and cBioPortal
(see Jee et al., 2024).

Outputs
-------
``msk_chord_2024.csv``
    All mutation rows with sample- and patient-level clinical columns merged in.
``msk_chord_2024_tx.csv``
    Columns: ``PATIENT_ID, START_DATE_REGIMEN, STOP_DATE_REGIMEN, AGENT_REGIMEN,
    TNT, line_of_therapy, PRIOR_MED_TO_MSK``.

Usage
-----
    python prepare_data.py \\
        --clinical-patient data_clinical_patient.txt \\
        --clinical-sample data_clinical_sample.txt \\
        --mutations data_mutations.txt \\
        --treatment-timeline data_timeline_treatment.txt \\
        --variant-output msk_chord_2024.csv \\
        --treatment-output msk_chord_2024_tx.csv
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

# cBioPortal clinical files prepend four metadata header lines (each starting
# with '#') before the column header. Same convention as GENIE.
CBIOPORTAL_CLINICAL_SKIPROWS = 4

# Final columns retained for the per-regimen treatment table. AGENT_REGIMEN is
# the underscore-joined, alphabetically-sorted set of agents administered
# concurrently within a single regimen start date; this canonical ordering
# avoids treating (drugA + drugB) as different from (drugB + drugA).
TREATMENT_OUTPUT_COLS = [
    "PATIENT_ID",
    "START_DATE_REGIMEN",
    "STOP_DATE_REGIMEN",
    "AGENT_REGIMEN",
    "TNT",
    "line_of_therapy",
    "PRIOR_MED_TO_MSK",
]

logger = logging.getLogger(__name__)


def build_variant_table(clinical_patient: pd.DataFrame,
                        clinical_sample: pd.DataFrame,
                        mutations: pd.DataFrame) -> pd.DataFrame:
    """Join mutations with sample-level then patient-level clinical metadata."""
    sample = clinical_sample.rename(columns={"SAMPLE_ID": "Tumor_Sample_Barcode"})
    df = mutations.merge(sample, on="Tumor_Sample_Barcode", how="left")
    df = df.merge(clinical_patient, on="PATIENT_ID", how="left")
    return df


def build_treatment_regimen_table(clinical_patient: pd.DataFrame,
                                  treatment_timeline: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-agent treatment events into per-regimen rows with line-of-therapy."""
    # Group by patient and regimen start date so concurrent agents collapse
    # into a single regimen entry.
    grouped = treatment_timeline.groupby(["PATIENT_ID", "START_DATE"]).agg({
        "STOP_DATE": "max",
        "AGENT": lambda x: "_".join(sorted(set(x))),
    }).reset_index()

    grouped["TNT"] = grouped["STOP_DATE"] - grouped["START_DATE"]
    grouped = grouped.rename(columns={
        "START_DATE": "START_DATE_REGIMEN",
        "STOP_DATE": "STOP_DATE_REGIMEN",
        "AGENT": "AGENT_REGIMEN",
    })
    grouped = grouped.sort_values(by=["PATIENT_ID", "START_DATE_REGIMEN"])
    grouped["line_of_therapy"] = grouped.groupby("PATIENT_ID").cumcount() + 1

    prior_med = clinical_patient.set_index("PATIENT_ID")["PRIOR_MED_TO_MSK"]
    grouped["PRIOR_MED_TO_MSK"] = grouped["PATIENT_ID"].map(prior_med)

    return grouped[TREATMENT_OUTPUT_COLS].reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build MSK-CHORD variant + clinical and treatment-regimen tables.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--clinical-patient", type=Path,
                        default=Path("data_clinical_patient.txt"),
                        help="cBioPortal patient-level clinical table.")
    parser.add_argument("--clinical-sample", type=Path,
                        default=Path("data_clinical_sample.txt"),
                        help="cBioPortal sample-level clinical table.")
    parser.add_argument("--mutations", type=Path,
                        default=Path("data_mutations.txt"),
                        help="Somatic-mutation MAF.")
    parser.add_argument("--treatment-timeline", type=Path,
                        default=Path("data_timeline_treatment.txt"),
                        help="Treatment timeline (one row per agent administration).")
    parser.add_argument("--variant-output", type=Path,
                        default=Path("msk_chord_2024.csv"),
                        help="Output path for the variant + clinical CSV.")
    parser.add_argument("--treatment-output", type=Path,
                        default=Path("msk_chord_2024_tx.csv"),
                        help="Output path for the treatment-regimen CSV.")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    for path in (args.clinical_patient, args.clinical_sample,
                 args.mutations, args.treatment_timeline):
        if not path.exists():
            raise FileNotFoundError(f"Input not found: {path}")

    logger.info("Loading clinical_patient: %s", args.clinical_patient)
    clinical_patient = pd.read_csv(args.clinical_patient, sep="\t",
                                   skiprows=CBIOPORTAL_CLINICAL_SKIPROWS)
    logger.info("clinical_patient rows: %d", len(clinical_patient))

    logger.info("Loading clinical_sample: %s", args.clinical_sample)
    clinical_sample = pd.read_csv(args.clinical_sample, sep="\t",
                                  skiprows=CBIOPORTAL_CLINICAL_SKIPROWS)
    logger.info("clinical_sample rows: %d", len(clinical_sample))

    logger.info("Loading mutations: %s", args.mutations)
    mutations = pd.read_csv(args.mutations, sep="\t", low_memory=False)
    logger.info("mutation rows: %d", len(mutations))

    logger.info("Loading treatment_timeline: %s", args.treatment_timeline)
    treatment_timeline = pd.read_csv(args.treatment_timeline, sep="\t")
    logger.info("treatment-event rows: %d", len(treatment_timeline))

    logger.info("Building variant + clinical table")
    variant_df = build_variant_table(clinical_patient, clinical_sample, mutations)
    args.variant_output.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Writing %d rows to %s", len(variant_df), args.variant_output)
    variant_df.to_csv(args.variant_output, index=False)

    logger.info("Building treatment-regimen table")
    treatment_df = build_treatment_regimen_table(clinical_patient, treatment_timeline)
    args.treatment_output.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Writing %d regimen rows to %s",
                len(treatment_df), args.treatment_output)
    treatment_df.to_csv(args.treatment_output, index=False)
    logger.info("Done.")


if __name__ == "__main__":
    main(parse_args())
