#!/usr/bin/env bash
# Train the SNV+CNA multimodal tumor-type classifier reported in Figure 3.
# Default = global_25 SNV features + attn_2 CNA features (manuscript config).
#
#   ./run_snv_cna_classifier.sh                                                # manuscript default
#   SNV_MODEL_NAME=local_25 CNA_MODEL_NAME=attn_0 ./run_snv_cna_classifier.sh  # other combo
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# Save plots without blocking on any GUI backend.
export MPLBACKEND=Agg

SNV="${SNV_MODEL_NAME:-global_25}"
CNA="${CNA_MODEL_NAME:-attn_2}"

echo "=== Training SNV+CNA multimodal tumor-type classifier: SNV=${SNV}, CNA=${CNA} ==="
SNV_MODEL_NAME="$SNV" CNA_MODEL_NAME="$CNA" python3 tumor_type_classifier_snv_cna.py
