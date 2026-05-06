#!/usr/bin/env bash
# Train the 6 SNV tumor-type classifiers reported in Figure 3 b-c, one per
# variant model.
#
#   ./run_snv_classifier.sh                       # all 6 (default)
#   ./run_snv_classifier.sh global_25             # one model
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# 6 models (the manuscript Figure 3 b-c lineup): the 3 local + 3 global
# context widths. The no-context baseline appears in Figure 1 (masked-
# token accuracy) but not in the SNV tumour-classifier comparison.
DEFAULT_MODELS=(local_1 local_10 local_25 global_1 global_10 global_25)
MODELS=("${@:-${DEFAULT_MODELS[@]}}")

# Save plots without blocking on any GUI backend.
export MPLBACKEND=Agg

for MODEL_NAME in "${MODELS[@]}"; do
    echo "=== Training SNV tumor-type classifier: MODEL_NAME=${MODEL_NAME} ==="
    MODEL_NAME="$MODEL_NAME" python3 tumor_type_classifier_snv.py
done
