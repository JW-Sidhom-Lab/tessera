#!/usr/bin/env bash
# Train the 3 TESSERA CNA models reported in Figure 2 (attn_0, attn_1, attn_2).
# Override the attention-block list by passing arguments:
#
#   ./run_all_configs.sh           # 0 1 2 (default)
#   ./run_all_configs.sh 1 2       # subset
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

ATTN_BLOCKS=("${@:-0 1 2}")

for n in "${ATTN_BLOCKS[@]}"; do
    echo "=== Training cna_attention_blocks=${n} ==="
    CNA_ATTN_BLOCKS="$n" python3 fit_model.py
done
