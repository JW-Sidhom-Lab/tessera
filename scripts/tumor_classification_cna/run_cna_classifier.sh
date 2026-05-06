#!/usr/bin/env bash
# Train the 3 CNA tumor-type classifiers reported in Figure 3, one per
# CNA attention-block model.
#
#   ./run_cna_classifier.sh                       # all 3 (default)
#   ./run_cna_classifier.sh attn_2                # one model
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# 3 CNA models from scripts/tcga_pancan_cna/ (attn_0, attn_1, attn_2).
DEFAULT_MODELS=(attn_0 attn_1 attn_2)
if [ "$#" -eq 0 ]; then
    MODELS=("${DEFAULT_MODELS[@]}")
else
    MODELS=("$@")
fi

# Save plots without blocking on any GUI backend.
export MPLBACKEND=Agg

for MODEL_NAME in "${MODELS[@]}"; do
    echo "=== Training CNA tumor-type classifier: MODEL_NAME=${MODEL_NAME} ==="
    MODEL_NAME="$MODEL_NAME" python3 tumor_type_classifier_cna.py
done
