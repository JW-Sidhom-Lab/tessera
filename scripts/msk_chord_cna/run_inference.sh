#!/usr/bin/env bash
# Apply the 3 trained TESSERA CNA NoLOH models to MSK-CHORD panel-sequenced
# CNA data, then plot Figure 2 d.
#
#   ./run_inference.sh             # all 3 attention configs (default)
#   ./run_inference.sh 2           # one config only
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [ "$#" -eq 0 ]; then
    ATTN_BLOCKS=(0 1 2)
else
    ATTN_BLOCKS=("$@")
fi

for n in "${ATTN_BLOCKS[@]}"; do
    echo "=== MSK-CHORD CNA inference: attn_${n} ==="
    CNA_ATTN_BLOCKS="$n" python3 get_cna_loss_acc.py
done

echo "=== Plotting Figure 2 d ==="
python3 plot_cna_reconstruction.py
