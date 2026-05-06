"""Build the Stage IV PDAC FOLFIRINOX-vs-Gem/Abraxane first-line ground-truth cohort table.

Selects MSK-CHORD patients with Stage IV Pancreatic Cancer who received
either gemcitabine + nab-paclitaxel ('Gem/Abraxane') or FOLFIRINOX as
first-line therapy, computes time-to-next-treatment-or-death (TTNTD) and
binary response per regimen instance, and writes the result to
``GROUND_TRUTH_PANCREATIC_GEMABRA_FOLFIRINOX_STAGE4_FIRST_LINE_TTNTD.csv``.

This is the ground-truth cohort used in Figure 6, panels f-i of the
manuscript. The TTNTD framework follows Jee et al. (Nature 2024).

Note on the regimen specs: ``PACLITAXEL PROTEIN-BOUND`` (nab-paclitaxel,
Abraxane) is required, distinct from non-protein-bound paclitaxel; and
``IRINOTECAN`` selects the standard form, distinct from
``IRINOTECAN LIPOSOMAL``.

The shared cohort-assembly machinery lives in ``cohort_utils.py``; this
script only specifies the cohort-specific configuration.

Usage
-----
    python create_ground_truth_pancreatic_gemabra_folfirinox_stage4_first_line_ttntd.py \\
        --base-dir . \\
        --output GROUND_TRUTH_PANCREATIC_GEMABRA_FOLFIRINOX_STAGE4_FIRST_LINE_TTNTD.csv
"""

import argparse
import logging
from pathlib import Path

from cohort_utils import Regimen, build_ttntd_cohort

CANCER_TYPE = "Pancreatic Cancer"

REGIMENS = [
    Regimen(
        label="Gem/Abraxane",
        combination="GEMCITABINE+PACLITAXEL PROTEIN-BOUND",
        required_agents=frozenset({"GEMCITABINE", "PACLITAXEL PROTEIN-BOUND"}),
    ),
    Regimen(
        label="FOLFIRINOX",
        combination="FLUOROURACIL+LEUCOVORIN+IRINOTECAN+OXALIPLATIN",
        required_agents=frozenset({"FLUOROURACIL", "LEUCOVORIN",
                                   "IRINOTECAN", "OXALIPLATIN"}),
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
    parser.add_argument(
        "--output", type=Path,
        default=Path("GROUND_TRUTH_PANCREATIC_GEMABRA_FOLFIRINOX_STAGE4_FIRST_LINE_TTNTD.csv"),
        help="Output path for the ground-truth CSV.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    build_ttntd_cohort(
        base_dir=args.base_dir,
        cancer_type=CANCER_TYPE,
        regimens=REGIMENS,
        output_path=args.output,
        first_line_only=True,
    )


if __name__ == "__main__":
    main(parse_args())
