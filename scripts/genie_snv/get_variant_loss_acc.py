"""Evaluate pretrained TCGA SNV models on GENIE for cross-platform validation.

Loads a TCGA-pretrained TESSERA SNV model and runs inference on the GENIE
panel-sequencing SNV dataset. Computes ALT and REF allele predictions,
logits, and loss; pickles them for ``plot_accuracy.py``. Backs Fig. 1 f-g.

Usage:
    python3 get_variant_loss_acc.py [MODEL_NAME]
    ./run_inference.sh                          # iterate over all 7 models

Output:
    Pickle files under var_loss/, each containing:
        y_pred, y_pred_ref       ALT and REF allele predictions
        logits, logits_ref       ALT and REF allele logits
        y_true, y_true_ref       ground-truth ALT and REF labels
        loss                     model loss on ALT predictions
        genie_data               filtered GENIE dataframe
        data_info                summary statistics
"""

import pickle
from tessera.model import TESSERA
import pandas as pd
import os
import sys


# ============================================================================
# Configuration
# ============================================================================

# Model name: path to pretrained model in ../tcga_pancan_snv/models/
# Can be overridden via command line argument
MODEL_NAME = 'TCGA_PanCan_SNV_baseline'

# Context length: number of flanking bases on each side (extracted from model name if available)
# Default: 1 (minimal context)
CONTEXT_LEN = 1

# Batch size for inference processing
BATCH_SIZE = 120

# Mutation type: 'SNV' for single nucleotide variants
MUT_TYPE = 'SNV'

# Distributed training flag: set to True for multi-GPU inference
USE_DISTRIBUTED = False

# ============================================================================
# Handle Command Line Arguments
# ============================================================================

# Allow model name override via command line argument
# Usage: python3 get_variant_loss_acc.py TCGA_PanCan_SNV_global_25
if len(sys.argv) > 1:
    MODEL_NAME = sys.argv[1]
    # Try to extract context length from model name (e.g., TCGA_PanCan_SNV_global_25 -> 25)
    try:
        CONTEXT_LEN = int(MODEL_NAME.split('_')[4])
    except (IndexError, ValueError):
        CONTEXT_LEN = 1
    print(f"Using model: {MODEL_NAME}")

# ============================================================================
# Setup
# ============================================================================

# Create output directory for results
dir_save = 'var_loss'
if not os.path.exists(dir_save):
    os.makedirs(dir_save)
    print(f"Created output directory: {dir_save}")


# ============================================================================
# Load and Filter Data
# ============================================================================

print(f"\nLoading GENIE data from ../data/genie/snv.csv...")
genie_data = pd.read_csv('../data/genie/snv.csv')
print(f"Total variants loaded: {len(genie_data)}")
print(f"Unique samples: {genie_data['Tumor_Sample_Barcode'].nunique()}")

# Filter out samples with excessive variant burden (>100 variants per sample)
# Rationale: High variant burden samples may have different mutational processes or be false positives
# This ensures we evaluate on samples with biologically reasonable mutation loads
variant_counts = genie_data['Tumor_Sample_Barcode'].value_counts()
samples_to_keep = variant_counts[variant_counts <= 100].index
n_before_filter = len(genie_data)
genie_data = genie_data[genie_data['Tumor_Sample_Barcode'].isin(samples_to_keep)]
n_after_filter = len(genie_data)

print(f"After filtering (>100 variants/sample): {n_after_filter} variants")
print(f"  Variants removed: {n_before_filter - n_after_filter}")
print(f"  Unique samples remaining: {genie_data['Tumor_Sample_Barcode'].nunique()}")


# ============================================================================
# Extract Genomic Features
# ============================================================================

# Extract variant genomic coordinates and annotations for model input
sample_ids = genie_data['Tumor_Sample_Barcode'].values
chr_vals = genie_data['Chromosome'].values
pos_vals = genie_data['Start_Position'].values
ref_vals = genie_data['Reference_Allele'].values
alt_vals = genie_data['Tumor_Seq_Allele2'].values
vaf_vals = genie_data['vaf'].values

print(f"\nExtracted genomic features for {len(sample_ids)} variants")


# ============================================================================
# Initialize and Load Model
# ============================================================================

print(f"\nInitializing model: {MODEL_NAME}")
model = TESSERA(
    model_dir='../tcga_pancan_snv/models/' + MODEL_NAME,
    use_distributed=USE_DISTRIBUTED,
    jit_compile=False,
    mixed_precision=False
)
print(f"Model loaded successfully from ../tcga_pancan_snv/models/{MODEL_NAME}")

# Set ref and alt allele lengths based on mutation type
# SNV: single base substitution (ref_len=1, alt_len=1)
# Other types: use longer sequences (ref_len=10, alt_len=10)
if MUT_TYPE == 'SNV':
    ref_len = 1
    alt_len = 1
else:
    ref_len = 10
    alt_len = 10


# ============================================================================
# Create GENIE Dataset
# ============================================================================

print(f"\nCreating inference dataset for {len(genie_data)} variants...")
model.create_sample_dataset(
    sample_ids, chr_vals, pos_vals, ref_vals, alt_vals,
    vaf=vaf_vals,
    context_len=CONTEXT_LEN,
    batch_size=BATCH_SIZE,
    name='genie_dataset',
    is_training=False,
    subsample=None,
    fixed_bag_size=True,
    ref_len=ref_len,
    alt_len=alt_len
)
print(f"Dataset created with batch size: {BATCH_SIZE}, context length: {CONTEXT_LEN}")


# ============================================================================
# Run Inference
# ============================================================================

print(f"\nRunning inference on GENIE variants...")
# Get predictions for both ALT and REF alleles with logits and loss
out = model.get_variant_probabilities(
    dataset_name='genie_dataset',
    return_logits=True,      # Return model logits for calibration analysis
    return_true_values=True,  # Return ground truth labels
    return_loss=True,         # Return loss values for each variant
    non_zero_only=False,      # Include all variants (don't filter zero predictions)
    return_ref=True           # Get predictions for REF allele as well
)
print(f"Inference complete")

# Unpack results
# out[0:4] = ALT predictions, logits, true labels, loss
# out[4:7] = REF predictions, logits, true labels
y_pred, logits, y_true, loss, y_pred_ref, logits_ref, y_true_ref = out


# ============================================================================
# Save Results
# ============================================================================

output_file = os.path.join(dir_save, f'{MODEL_NAME}_genie_loss_variant.pkl')
print(f"\nSaving results to {output_file}...")

with open(output_file, 'wb') as f:
    pickle.dump({
        # ALT allele predictions
        'y_pred': y_pred,              # Predicted probabilities for ALT allele
        'logits': logits,              # Raw logits before softmax
        'y_true': y_true,              # Ground truth ALT allele labels
        'loss': loss,                  # Model loss on ALT predictions

        # REF allele predictions
        'y_pred_ref': y_pred_ref,      # Predicted probabilities for REF allele
        'logits_ref': logits_ref,      # Raw logits for REF allele
        'y_true_ref': y_true_ref,      # Ground truth REF allele labels

        # Metadata and data
        'genie_data': genie_data,      # Filtered GENIE dataframe for traceability
        'model_name': MODEL_NAME,      # Model used for inference
        'data_info': {                 # Summary statistics
            'n_samples': len(genie_data),
            'n_patients': genie_data['Tumor_Sample_Barcode'].nunique(),
            'n_variants': len(genie_data)
        }
    }, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Results saved successfully!")
print(f"  Variants evaluated: {len(genie_data)}")
print(f"  Unique samples: {genie_data['Tumor_Sample_Barcode'].nunique()}")
print(f"Done!")