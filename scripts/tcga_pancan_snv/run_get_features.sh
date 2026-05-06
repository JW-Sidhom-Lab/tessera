#!/usr/bin/env bash
# Extract per-variant latent features from all 7 trained TESSERA SNV models.
#
# Same (config, context_len) defaults as run_all_configs.sh; override by
# passing custom run specs:
#
#   ./run_get_features.sh                     # all 7
#   ./run_get_features.sh global:25           # one model
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

DEFAULT_RUNS=(baseline:1 local:1 local:10 local:25 global:1 global:10 global:25)
RUNS=("${@:-${DEFAULT_RUNS[@]}}")

for run in "${RUNS[@]}"; do
    config="${run%%:*}"
    context_len="${run##*:}"
    echo "=== Extracting features: config=${config} context_len=${context_len} ==="
    CONFIG="$config" CONTEXT_LEN="$context_len" python3 get_variant_features.py
done
