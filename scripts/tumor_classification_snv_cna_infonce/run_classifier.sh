#!/usr/bin/env bash
# Train both InfoNCE-aligned multimodal classifier variants reported in
# the manuscript:
#   * LOH variant   (Figure 4 b-c headline classifier; macro-AUC 0.987)
#       -> models_macro/, plots/
#   * NoLOH variant (ablation; corresponding noLOH joint pretraining)
#       -> models_macro_noloh/, plots_noloh/
#
#   ./run_classifier.sh                 # both variants (default)
#   ./run_classifier.sh loh             # LOH only
#   ./run_classifier.sh noloh           # NoLOH only
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

DEFAULT_VARIANTS=(loh noloh)
if [ "$#" -eq 0 ]; then
    VARIANTS=("${DEFAULT_VARIANTS[@]}")
else
    VARIANTS=("$@")
fi

# Save plots without blocking on any GUI backend.
export MPLBACKEND=Agg

FEATURES_DIR="../tcga_pancan_snv_cna/multimodal_features"

for variant in "${VARIANTS[@]}"; do
    case "$variant" in
        loh)
            FEATURES_PATH="${FEATURES_DIR}/TCGA_SNV_CNA_InfoNCE_per_sample_loss_multimodal_features.pkl"
            OUTPUT_TAG=""
            ;;
        noloh)
            FEATURES_PATH="${FEATURES_DIR}/TCGA_SNV_CNA_InfoNCE_per_sample_loss_noLOH_multimodal_features.pkl"
            OUTPUT_TAG="_noloh"
            ;;
        *)
            echo "Unknown variant: $variant (expected: loh | noloh)" >&2
            exit 1
            ;;
    esac
    echo "=== Training InfoNCE multimodal classifier: variant=${variant} ==="
    FEATURES_PATH="$FEATURES_PATH" OUTPUT_TAG="$OUTPUT_TAG" \
        python3 tumor_type_classifier_snv_cna_infonce.py
done
