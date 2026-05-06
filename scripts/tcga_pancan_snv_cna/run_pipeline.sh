#!/usr/bin/env bash
# End-to-end joint SNV+CNA InfoNCE pretraining + feature extraction.
#
#   ./run_pipeline.sh          # train, then extract per-variant + per-segment features
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

echo "=== Training joint SNV+CNA InfoNCE model ==="
python3 fit_model.py

echo "=== Extracting joint features ==="
python3 get_cna_variant_features.py
