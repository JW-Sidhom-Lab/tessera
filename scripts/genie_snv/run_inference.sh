#!/bin/bash

################################################################################
# GENIE SNV Model Inference Pipeline
#
# Run loss and accuracy computation on pretrained TCGA PanCan SNV models using
# GENIE data. Evaluates multiple models with different context lengths to assess
# how model architecture generalises across panel-sequencing platforms (Fig 1 f-g).
#
# Usage:
#     ./run_inference.sh
#
# Output:
#     - var_loss/: Pickle files containing model predictions, logits, and loss
#       for each model
#
# Requirements:
#     - tessera package installed (pip install -e .)
#     - Pretrained models in ../tcga_pancan_snv/models/
#     - GENIE data prepared in ../data/genie/snv.csv
#
################################################################################

################################################################################
# Configuration
################################################################################

# List of models to evaluate
# Models are organized by architecture (baseline, local context, global context)
# and context length (1, 10, 25 bases on each side of variant)
#
# baseline: No local/global attention, baseline architecture
# local_*:  Local context attention with varying context windows
# global_*: Global context attention with varying context windows
MODELS=(
    "TCGA_PanCan_SNV_baseline"
    "TCGA_PanCan_SNV_local_1"
    "TCGA_PanCan_SNV_local_10"
    "TCGA_PanCan_SNV_local_25"
    "TCGA_PanCan_SNV_global_1"
    "TCGA_PanCan_SNV_global_10"
    "TCGA_PanCan_SNV_global_25"
)

################################################################################
# Helper Functions
################################################################################

# Print section header with visual formatting
print_header() {
    echo ""
    echo "=========================================="
    echo "$1"
    echo "=========================================="
    echo ""
}

# Print model separator
print_model_header() {
    echo ""
    echo "----------------------------------------"
    echo "Model: $1"
    echo "----------------------------------------"
}

################################################################################
# Main Pipeline
################################################################################

print_header "GENIE SNV Inference Pipeline"

# Counter for tracking progress
model_count=0
success_count=0
failed_count=0

# Iterate through all models
for MODEL in "${MODELS[@]}"; do
    model_count=$((model_count + 1))

    print_model_header "$MODEL"

    # Verify model directory exists
    # Exit early if model not found to avoid wasted computation
    MODEL_DIR="../tcga_pancan_snv/models/$MODEL"
    if [ ! -d "$MODEL_DIR" ]; then
        echo "⚠ WARNING: Model directory not found: $MODEL_DIR"
        echo "Skipping this model..."
        failed_count=$((failed_count + 1))
        continue
    fi

    # Run inference to compute loss and accuracy metrics
    # This evaluates the model on GENIE variants and saves predictions/logits
    echo "Running inference on GENIE data..."
    python3 get_variant_loss_acc.py "$MODEL"

    # Check if inference completed successfully
    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed inference for $MODEL"
        success_count=$((success_count + 1))
        echo "  Results saved to: var_loss/${MODEL}_genie_loss_variant.pkl"
    else
        echo "✗ Error during inference for $MODEL"
        failed_count=$((failed_count + 1))
    fi

done

################################################################################
# Summary
################################################################################

print_header "Pipeline Summary"

echo "Models processed: $model_count"
echo "  ✓ Successful: $success_count"
echo "  ✗ Failed: $failed_count"
echo ""
echo "Output directory: var_loss/"
echo ""
echo "Next steps:"
echo "  1. Run plot_accuracy.py to generate accuracy visualizations"
echo "  2. Analyze results in var_loss/ directory"
echo ""
print_header "Pipeline Complete"