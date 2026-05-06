"""Build the Stage IV CRC FOLFOX-vs-FOLFIRI ground-truth cohort table.

Selects MSK-CHORD patients with Stage IV Colorectal Cancer who received
either FOLFOX or FOLFIRI at any line of therapy, computes time-to-next-
treatment-or-death (TTNTD) and binary response per regimen instance, and
writes the result to ``GROUND_TRUTH_CRC_FOLFOX_FOLFIRI_STAGE4_TTNTD.csv``.

This is the ground-truth cohort used in Figure 6, panels b-e of the
manuscript. The TTNTD framework follows Jee et al. (Nature 2024).

The shared cohort-assembly machinery lives in ``cohort_utils.py``; this
script only specifies the cohort-specific configuration.

Usage
-----
    python create_ground_truth_crc_folfox_folfiri_stage4_ttntd.py \\
        --base-dir . \\
        --output GROUND_TRUTH_CRC_FOLFOX_FOLFIRI_STAGE4_TTNTD.csv
"""

import argparse
import logging
from pathlib import Path

from cohort_utils import Regimen, build_ttntd_cohort

CANCER_TYPE = "Colorectal Cancer"

REGIMENS = [
    Regimen(
        label="FOLFOX",
        combination="FLUOROURACIL+LEUCOVORIN+OXALIPLATIN",
        required_agents=frozenset({"FLUOROURACIL", "LEUCOVORIN", "OXALIPLATIN"}),
    ),
    Regimen(
        label="FOLFIRI",
        combination="FLUOROURACIL+LEUCOVORIN+IRINOTECAN",
        required_agents=frozenset({"FLUOROURACIL", "LEUCOVORIN", "IRINOTECAN"}),
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base-dir", type=Path, default=Path("."),
                        help="Directory containing the MSK-CHORD raw release "
                             "(data_clinical_*.txt, data_timeline_*.txt).")
    parser.add_argument("--output", type=Path,
                        default=Path("GROUND_TRUTH_CRC_FOLFOX_FOLFIRI_STAGE4_TTNTD.csv"),
                        help="Output path for the ground-truth CSV.")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    build_ttntd_cohort(
        base_dir=args.base_dir,
        cancer_type=CANCER_TYPE,
        regimens=REGIMENS,
        output_path=args.output,
        first_line_only=False,
    )


if __name__ == "__main__":
    main(parse_args())
