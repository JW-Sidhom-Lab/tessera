#!/usr/bin/env bash
# Train all 7 TESSERA SNV models reported in Figure 1.
#
# Each line below is a (config, context_len) pair: 1 baseline + 3 local +
# 3 global. Override which subset to run by passing arguments, e.g.:
#
#   ./run_all_configs.sh                     # all 7
#   ./run_all_configs.sh baseline:1 local:25 # custom subset
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

DEFAULT_RUNS=(baseline:1 local:1 local:10 local:25 global:1 global:10 global:25)
RUNS=("${@:-${DEFAULT_RUNS[@]}}")

for run in "${RUNS[@]}"; do
    config="${run%%:*}"
    context_len="${run##*:}"
    echo "=== Training config=${config} context_len=${context_len} ==="
    CONFIG="$config" CONTEXT_LEN="$context_len" python3 fit_model.py
done
