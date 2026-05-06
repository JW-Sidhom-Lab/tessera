#!/usr/bin/env bash
# Run the full CNA pretraining and analysis pipeline:
#   1. Train all 3 attention configs (attn_0 / attn_1 / attn_2)
#   2. Extract per-segment loss + LOH predictions
#   3. Extract per-segment latent features
#
# Generate Figure 2 plots after this with `python plot_cna_reconstruction.py`.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

./run_all_configs.sh
./run_get_loss.sh
./run_get_features.sh
