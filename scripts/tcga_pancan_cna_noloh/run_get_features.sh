#!/usr/bin/env bash
# Extract per-segment latent features from the trained NoLOH CNA models.
#
#   ./run_get_features.sh           # 0 1 2 (default)
#   ./run_get_features.sh 2         # one model
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

ATTN_BLOCKS=("${@:-0 1 2}")

for n in "${ATTN_BLOCKS[@]}"; do
    echo "=== Extracting NoLOH CNA features: cna_attention_blocks=${n} ==="
    CNA_ATTN_BLOCKS="$n" python3 get_cna_features.py
done
