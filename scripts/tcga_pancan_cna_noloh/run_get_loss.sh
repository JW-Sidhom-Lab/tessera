#!/usr/bin/env bash
# Extract per-segment predictions from the trained NoLOH CNA models.
#
#   ./run_get_loss.sh           # 0 1 2 (default)
#   ./run_get_loss.sh 2         # one model
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [ "$#" -eq 0 ]; then
    ATTN_BLOCKS=(0 1 2)
else
    ATTN_BLOCKS=("$@")
fi

for n in "${ATTN_BLOCKS[@]}"; do
    echo "=== Extracting NoLOH CNA loss/predictions: cna_attention_blocks=${n} ==="
    CNA_ATTN_BLOCKS="$n" python3 get_cna_loss_metrics.py
done
