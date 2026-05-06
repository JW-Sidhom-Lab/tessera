#!/usr/bin/env bash
# Train the 3 TESSERA CNA NoLOH models (attn_0, attn_1, attn_2).
#
#   ./run_all_configs.sh           # 0 1 2 (default)
#   ./run_all_configs.sh 2         # one model
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [ "$#" -eq 0 ]; then
    ATTN_BLOCKS=(0 1 2)
else
    ATTN_BLOCKS=("$@")
fi

for n in "${ATTN_BLOCKS[@]}"; do
    echo "=== Training NoLOH cna_attention_blocks=${n} ==="
    CNA_ATTN_BLOCKS="$n" python3 fit_model.py
done
