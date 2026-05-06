"""Generate per-model ClinVar pathogenicity predictions on the dedup train/valid split.

For each of the 7 trained TESSERA SNV models, fits a logistic-regression
classifier on per-variant TESSERA latent features (with optional PCA),
predicts pathogenic-vs-benign labels for both halves of the dedup split,
and writes one prediction CSV per (model, split-half) to
``clinvar_preds_dedup_<strategy>/``.

Inputs
------
``data/{train,valid}_data_snv_dedup{,_gene}.csv``       (from create_variant_dedup_split.py)
``data/index_mapping{,_gene}.pkl``                       (from create_index_mapping.py)
``../../data/clinvar/clinvar.pkl``                       (from data/clinvar/create_training_data.py)
``../tcga_pancan_snv/var_features/TCGA_PanCan_SNV_<config>_features.pkl``
``../tcga_pancan_snv/var_loss/TCGA_PanCan_SNV_<config>_loss_variant.pkl``

Outputs
-------
``clinvar_preds_dedup_<strategy>/<config>_{train,valid}_predictions.csv``

Usage
-----
    python generate_clinvar_predictions.py                       # variant (default)
    STRATEGY=gene python generate_clinvar_predictions.py
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ============================================================================
# Configuration (edit or set env vars to override)
# ============================================================================
STRATEGY = os.environ.get("STRATEGY", "variant")  # 'variant' | 'gene'


def load_mapping(mapping_file=None):
    """Load the index mapping from deduplicated rows to feature space."""
    if mapping_file is None:
        # Use strategy-specific mapping file
        if STRATEGY == 'gene':
            mapping_file = 'data/index_mapping_gene.pkl'
        else:
            mapping_file = 'data/index_mapping.pkl'
    with open(mapping_file, 'rb') as f:
        mapping = pickle.load(f)
    return mapping


def prepare_data(train_file=None, valid_file=None, clinvar_file='../../data/clinvar/clinvar.pkl'):
    """
    Load train and validation data and merge with ClinVar annotations.
    Create binary labels: Pathogenic vs Benign

    Returns:
    --------
    train_data : pd.DataFrame
        Training dataframe with ClinVar labels
    valid_data : pd.DataFrame
        Validation dataframe with ClinVar labels
    train_indices : np.ndarray
        Indices of train variants with ClinVar annotations
    valid_indices : np.ndarray
        Indices of validation variants with ClinVar annotations
    """
    # Use strategy-specific dedup files if not explicitly provided
    if train_file is None or valid_file is None:
        if STRATEGY == 'gene':
            train_file = train_file or 'data/train_data_snv_dedup_gene.csv'
            valid_file = valid_file or 'data/valid_data_snv_dedup_gene.csv'
        else:
            train_file = train_file or 'data/train_data_snv_dedup.csv'
            valid_file = valid_file or 'data/valid_data_snv_dedup.csv'

    print("=" * 80)
    print("PREPARING DATA WITH CLINVAR ANNOTATIONS")
    print(f"Strategy: {STRATEGY} | Train: {os.path.basename(train_file)}")
    print("=" * 80)

    # Load training data
    print("\nLoading training data...")
    train_data = pd.read_csv(train_file)
    print(f"Total train samples: {len(train_data):,}")

    # Load validation data
    print("Loading validation data...")
    valid_data = pd.read_csv(valid_file)
    print(f"Total validation samples: {len(valid_data):,}")

    # Load ClinVar
    print("Loading ClinVar annotations...")
    with open(clinvar_file, 'rb') as f:
        clinvar = pickle.load(f)
    print(f"Total ClinVar entries: {len(clinvar):,}")

    # Create mutation IDs for merging
    mut_id_cols_clinvar = ['CHROM', 'POS', 'REF', 'ALT']
    clinvar['mut_id'] = clinvar[mut_id_cols_clinvar].apply(lambda x: '_'.join(x.astype(str)), axis=1)
    clinvar = clinvar[['mut_id', 'CLNSIG']]

    mut_id_cols_data = ['Chromosome', 'Start_Position', 'Reference_Allele', 'Tumor_Seq_Allele2']
    train_data['mut_id'] = train_data[mut_id_cols_data].apply(lambda x: '_'.join(x.astype(str)), axis=1)
    valid_data['mut_id'] = valid_data[mut_id_cols_data].apply(lambda x: '_'.join(x.astype(str)), axis=1)

    # Merge with ClinVar
    train_data = train_data.merge(clinvar, on='mut_id', how='left')
    valid_data = valid_data.merge(clinvar, on='mut_id', how='left')

    # Create mapping to broader categories
    clinvar_mapping = {
        'Pathogenic': 'pathogenic',
        'Pathogenic/Likely_pathogenic': 'pathogenic',
        'Likely_pathogenic': 'pathogenic',
        'Benign': 'benign',
        'Benign/Likely_benign': 'benign',
        'Likely_benign': 'benign',
        'Uncertain_significance': 'uncertain',
        'Conflicting_classifications_of_pathogenicity': 'uncertain'
    }

    train_data['clinvar_broad'] = train_data['CLNSIG'].map(clinvar_mapping)
    valid_data['clinvar_broad'] = valid_data['CLNSIG'].map(clinvar_mapping)

    # Filter to only pathogenic and benign
    print("\nFiltering to Pathogenic and Benign variants...")
    train_data_filtered = train_data[train_data['clinvar_broad'].isin(['pathogenic', 'benign'])].reset_index(drop=False)
    valid_data_filtered = valid_data[valid_data['clinvar_broad'].isin(['pathogenic', 'benign'])].reset_index(drop=False)
    print(f"Train variants with ClinVar annotations: {len(train_data_filtered):,}")
    print(f"Validation variants with ClinVar annotations: {len(valid_data_filtered):,}")

    # Save original indices for feature subsetting
    train_indices = train_data_filtered['index'].values
    valid_indices = valid_data_filtered['index'].values

    # Drop the index column
    train_data_filtered = train_data_filtered.drop(columns=['index'])
    valid_data_filtered = valid_data_filtered.drop(columns=['index'])

    return train_data, valid_data, train_data_filtered, valid_data_filtered, train_indices, valid_indices


def load_features_for_model(model_name, var_features_dir='../tcga_pancan_snv/var_features'):
    """Load features for a single model from combined feature space (memory efficient - on-demand loading)."""
    model_file = f'TCGA_PanCan_SNV_{model_name}_features.pkl'
    model_path = os.path.join(var_features_dir, model_file)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Feature file not found: {model_path}")

    with open(model_path, 'rb') as f:
        out = pickle.load(f)

    # Concatenate train and valid into combined feature space
    # This allows indexing by combined indices from the mapping
    train_features = out['train']
    valid_features = out['valid']
    combined_features = np.vstack([train_features, valid_features])

    return combined_features


def load_all_features(var_features_dir='../tcga_pancan_snv/var_features'):
    """Return list of available models (features loaded on-demand to save memory)."""

    models = [
        'baseline',
        'local_1',
        'local_10',
        'local_25',
        'global_1',
        'global_10',
        'global_25',
    ]

    print("\n" + "=" * 80)
    print("LOADING MODEL FEATURES (ON-DEMAND TO SAVE MEMORY)")
    print("=" * 80)
    print(f"Available models: {models}")
    print("(Features will be loaded one model at a time)")

    return models


def load_prediction_accuracy(var_loss_dir='../tcga_pancan_snv/var_loss'):
    """
    Load pre-training prediction accuracy for all models.

    Returns:
    --------
    all_accuracy : dict
        Dictionary with model names as keys, each containing:
        - 'train': binary mask of correctly predicted train variants (ref AND alt correct)
        - 'valid': binary mask of correctly predicted validation variants (ref AND alt correct)
    """

    print("\n" + "=" * 80)
    print("LOADING PRE-TRAINING PREDICTION ACCURACY")
    print("=" * 80)

    models = [
        'TCGA_PanCan_SNV_baseline',
        'TCGA_PanCan_SNV_local_1',
        'TCGA_PanCan_SNV_local_10',
        'TCGA_PanCan_SNV_local_25',
        'TCGA_PanCan_SNV_global_1',
        'TCGA_PanCan_SNV_global_10',
        'TCGA_PanCan_SNV_global_25',
    ]

    all_accuracy = {}

    for model_name_full in models:
        model_name = model_name_full.replace('TCGA_PanCan_SNV_', '')
        loss_file = os.path.join(var_loss_dir, f'{model_name_full}_loss_variant.pkl')

        print(f"\nLoading {model_name}...")
        with open(loss_file, 'rb') as f:
            data = pickle.load(f)

        # Unpack train data
        y_pred_train, logits_train, y_true_train, loss_train, y_pred_ref_train, logits_ref_train, y_true_ref_train = data[0]

        # Unpack validation data
        y_pred_valid, logits_valid, y_true_valid, loss_valid, y_pred_ref_valid, logits_ref_valid, y_true_ref_valid = data[1]

        # Compute accuracy for train: both ref and alt must be completely correct
        train_pred_classes_alt = np.argmax(y_pred_train, axis=-1)
        train_pred_classes_ref = np.argmax(y_pred_ref_train, axis=-1)
        train_acc_alt = np.all(train_pred_classes_alt == y_true_train, axis=1)
        train_acc_ref = np.all(train_pred_classes_ref == y_true_ref_train, axis=1)
        train_acc_combined = train_acc_alt & train_acc_ref  # Both must be correct

        # Compute accuracy for validation: both ref and alt must be completely correct
        valid_pred_classes_alt = np.argmax(y_pred_valid, axis=-1)
        valid_pred_classes_ref = np.argmax(y_pred_ref_valid, axis=-1)
        valid_acc_alt = np.all(valid_pred_classes_alt == y_true_valid, axis=1)
        valid_acc_ref = np.all(valid_pred_classes_ref == y_true_ref_valid, axis=1)
        valid_acc_combined = valid_acc_alt & valid_acc_ref  # Both must be correct

        print(f"  Train accuracy: {train_acc_combined.mean():.3f}")
        print(f"  Valid accuracy: {valid_acc_combined.mean():.3f}")

        all_accuracy[model_name] = {
            'train': train_acc_combined,
            'valid': valid_acc_combined
        }

    return all_accuracy


def train_and_predict(X_train, X_test, y_train, use_scaler=True, use_pca=False, pca_variance=0.999):
    """
    Train binary classifier and generate predictions with probabilities.

    Parameters:
    -----------
    X_train : np.ndarray
        Training feature matrix
    X_test : np.ndarray
        Test feature matrix
    y_train : np.ndarray
        Training labels (0/1)
    use_scaler : bool
        Whether to apply StandardScaler
    use_pca : bool
        Whether to apply PCA
    pca_variance : float
        Target explained variance for PCA

    Returns:
    --------
    y_train_proba : np.ndarray
        Training set predictions (probabilities for class 1)
    y_test_proba : np.ndarray
        Test set predictions (probabilities for class 1)
    """
    # Standardize features
    if use_scaler:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test

    # Apply PCA
    if use_pca:
        pca = PCA(n_components=pca_variance, random_state=42)
        X_train_final = pca.fit_transform(X_train_scaled)
        X_test_final = pca.transform(X_test_scaled)
    else:
        X_train_final = X_train_scaled
        X_test_final = X_test_scaled

    # Train logistic regression
    lr = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42,
        verbose=0
    )
    lr.fit(X_train_final, y_train)

    # Get probabilities for both train and test
    y_train_proba = lr.predict_proba(X_train_final)[:, 1]
    y_test_proba = lr.predict_proba(X_test_final)[:, 1]

    return y_train_proba, y_test_proba


def predict_in_batches(model, scaler, pca, X, batch_size=50000, use_scaler=True, use_pca=True):
    """
    Predict probabilities in batches to avoid memory issues.

    Parameters:
    -----------
    model : trained classifier
    scaler : fitted StandardScaler (or None)
    pca : fitted PCA (or None)
    X : feature matrix to predict on
    batch_size : number of samples per batch
    use_scaler : whether to apply scaling
    use_pca : whether to apply PCA

    Returns:
    --------
    predictions : array of probabilities for class 1
    """
    n_samples = X.shape[0]
    predictions = np.zeros(n_samples)

    n_batches = int(np.ceil(n_samples / batch_size))

    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)

        X_batch = X[start_idx:end_idx]

        # Apply transformations
        if use_scaler and scaler is not None:
            X_batch = scaler.transform(X_batch)

        if use_pca and pca is not None:
            X_batch = pca.transform(X_batch)

        # Predict
        predictions[start_idx:end_idx] = model.predict_proba(X_batch)[:, 1]

        if (i + 1) % 10 == 0 or (i + 1) == n_batches:
            print(f"    Processed {end_idx:,} / {n_samples:,} samples ({end_idx/n_samples*100:.1f}%)")

    return predictions


def main(use_scaler=True, use_pca=True, pca_variance=0.999, output_dir='clinvar_preds_dedup',
         batch_size=50000, require_correct_prediction=False, save_models=True, models_dir='pathogenic_models_dedup',
         base_model_dir='../tcga_pancan_snv'):
    """
    Generate predictions for all models and save results.
    Trains on variants with ClinVar labels, but predicts on ALL variants.
    Uses batch processing to avoid memory issues.

    Parameters:
    -----------
    batch_size : int
        Number of samples to process at once during prediction (default: 50000)
    require_correct_prediction : bool
        If True, train only on variants where the model correctly predicted both ref and alt
        during pre-training. Files will be saved with "_correct_only" suffix. (default: False)
    save_models : bool
        If True, save trained models (lr, scaler, pca) to disk (default: True)
    models_dir : str
        Directory to save trained models (default: 'pathogenic_models')
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    if save_models:
        os.makedirs(models_dir, exist_ok=True)

    # Prepare data
    train_data_full, valid_data_full, train_data_filtered, valid_data_filtered, train_indices, valid_indices = prepare_data()

    # Load index mapping for deduplicated split
    print("\nLoading index mapping...")
    mapping = load_mapping()
    orig_train_size = mapping['orig_train_rows']
    train_idx_mapping = mapping['train_indices']
    valid_idx_mapping = mapping['valid_indices']
    print(f"  Mapping size: {len(train_idx_mapping):,} train + {len(valid_idx_mapping):,} valid → combined space ({len(train_idx_mapping) + len(valid_idx_mapping):,})")

    # Construct paths from base model directory
    var_features_dir = os.path.join(base_model_dir, 'var_features')
    var_loss_dir = os.path.join(base_model_dir, 'var_loss')

    print(f"Using model directory: {base_model_dir}")
    print(f"  Features: {var_features_dir}")
    print(f"  Loss: {var_loss_dir}")

    # Encode labels for training (only ClinVar-labeled variants)
    label_map = {'pathogenic': 1, 'benign': 0}
    y_train = np.array([label_map[label] for label in train_data_filtered['clinvar_broad'].values])

    # Load features for all models
    all_features = load_all_features(var_features_dir=var_features_dir)

    # Load prediction accuracy for all models (will use mapping to extract subset)
    print("\nLoading pre-training prediction accuracy...")
    all_accuracy = load_prediction_accuracy(var_loss_dir=var_loss_dir)

    # Process each model
    print("\n" + "=" * 80)
    print("GENERATING PREDICTIONS FOR ALL MODELS")
    print("Approach: Train on ClinVar-labeled variants, predict on ALL variants")
    print(f"Batch size: {batch_size:,}")
    if require_correct_prediction:
        print("Training filter: Only correctly predicted variants (ref AND alt correct)")
    else:
        print("Training filter: All labeled variants")
    print("=" * 80)

    for model_name in all_features:
        print(f"\n{'='*60}")
        print(f"Processing: {model_name}")
        print(f"{'='*60}")

        # Load features for this model only (memory efficient - from combined feature space)
        print(f"Loading features for {model_name}...")
        combined_features = load_features_for_model(model_name, var_features_dir=var_features_dir)
        print(f"  Combined features shape: {combined_features.shape}")

        # Extract features for dedup train and valid using mapping
        print(f"Extracting dedup train/valid features using mapping...")
        X_train_all = combined_features[train_idx_mapping]
        X_test_all = combined_features[valid_idx_mapping]
        print(f"  Dedup train shape: {X_train_all.shape}, Dedup valid shape: {X_test_all.shape}")

        # Extract correctness labels using mapping (map indices back to original positions)
        # Precomputed correctness arrays are indexed by original row position
        train_correct_orig = all_accuracy[model_name]['train']  # Shape: (1441936,)
        valid_correct_orig = all_accuracy[model_name]['valid']  # Shape: (479467,)

        # Extract correctness for dedup rows by mapping indices
        train_correct_all = np.zeros(len(train_idx_mapping), dtype=bool)
        for i, idx in enumerate(train_idx_mapping):
            if idx < orig_train_size:
                train_correct_all[i] = train_correct_orig[idx]
            else:
                orig_idx = idx - orig_train_size
                train_correct_all[i] = valid_correct_orig[orig_idx]

        valid_correct_all = np.zeros(len(valid_idx_mapping), dtype=bool)
        for i, idx in enumerate(valid_idx_mapping):
            if idx < orig_train_size:
                valid_correct_all[i] = train_correct_orig[idx]
            else:
                orig_idx = idx - orig_train_size
                valid_correct_all[i] = valid_correct_orig[orig_idx]

        # Get features only for ClinVar-labeled variants (for training)
        X_train_labeled = X_train_all[train_indices]
        train_correct_labeled = train_correct_all[train_indices]

        # Optionally filter to only correctly predicted variants
        if require_correct_prediction:
            train_filter = train_correct_labeled
            X_train_labeled = X_train_labeled[train_filter]
            y_train_filtered = y_train[train_filter]

            n_kept = len(X_train_labeled)
            n_total = len(train_indices)
            print(f"  Training on {n_kept:,} / {n_total:,} ClinVar-labeled variants ({n_kept/n_total*100:.1f}%)")
            print(f"    (filtered to correctly predicted variants only)")
        else:
            y_train_filtered = y_train
            print(f"  Training on {len(X_train_labeled):,} ClinVar-labeled variants")

        print(f"  Will predict on {len(X_train_all):,} dedup train + {len(X_test_all):,} dedup valid variants")

        # Fit scaler and PCA on labeled data only
        scaler = None
        pca = None

        if use_scaler:
            print("  Fitting StandardScaler...")
            scaler = StandardScaler()
            X_train_labeled_scaled = scaler.fit_transform(X_train_labeled)
        else:
            X_train_labeled_scaled = X_train_labeled

        if use_pca:
            print("  Fitting PCA...")
            pca = PCA(n_components=pca_variance, random_state=42)
            X_train_labeled_final = pca.fit_transform(X_train_labeled_scaled)
            print(f"    PCA components: {pca.n_components_}")
        else:
            X_train_labeled_final = X_train_labeled_scaled

        # Train logistic regression on (optionally filtered) labeled data
        print("  Training logistic regression...")
        lr = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42,
            verbose=0
        )
        lr.fit(X_train_labeled_final, y_train_filtered)

        # Save trained models
        if save_models:
            print("  Saving trained model...")
            model_save_path = os.path.join(models_dir, f'{model_name}.pkl')
            model_package = {
                'model': lr,
                'scaler': scaler,
                'pca': pca,
                'use_scaler': use_scaler,
                'use_pca': use_pca,
                'pca_variance': pca_variance if use_pca else None,
                'n_components': pca.n_components_ if use_pca and pca is not None else None
            }
            with open(model_save_path, 'wb') as f:
                pickle.dump(model_package, f)
            print(f"    Saved model to: {model_save_path}")

        # Predict on ALL variants using batch processing
        print("  Generating predictions for train set...")
        y_train_proba_all = predict_in_batches(
            lr, scaler, pca, X_train_all,
            batch_size=batch_size,
            use_scaler=use_scaler,
            use_pca=use_pca
        )

        print("  Generating predictions for validation set...")
        y_test_proba_all = predict_in_batches(
            lr, scaler, pca, X_test_all,
            batch_size=batch_size,
            use_scaler=use_scaler,
            use_pca=use_pca
        )

        # Add predictions and correctness indicator to data
        train_output = train_data_full.copy()
        train_output['P(pathogenic)'] = y_train_proba_all
        train_output['correctly_predicted'] = train_correct_all

        valid_output = valid_data_full.copy()
        valid_output['P(pathogenic)'] = y_test_proba_all
        valid_output['correctly_predicted'] = valid_correct_all

        # Save results with appropriate suffix
        suffix = '_correct_only' if require_correct_prediction else ''
        train_output_path = os.path.join(output_dir, f'{model_name}_train_predictions{suffix}.csv')
        valid_output_path = os.path.join(output_dir, f'{model_name}_valid_predictions{suffix}.csv')

        print("  Saving results...")
        train_output.to_csv(train_output_path, index=False)
        valid_output.to_csv(valid_output_path, index=False)

        print(f"  Saved: {train_output_path}")
        print(f"  Saved: {valid_output_path}")

        # Print summary statistics
        print(f"\n  Summary (all variants):")
        print(f"    Train - Mean P(pathogenic): {y_train_proba_all.mean():.3f}, Std: {y_train_proba_all.std():.3f}")
        print(f"    Valid - Mean P(pathogenic): {y_test_proba_all.mean():.3f}, Std: {y_test_proba_all.std():.3f}")
        print(f"    Train - Correctly predicted: {train_correct_all.mean()*100:.1f}%")
        print(f"    Valid - Correctly predicted: {valid_correct_all.mean()*100:.1f}%")

    print("\n" + "=" * 80)
    print("PREDICTION GENERATION COMPLETE!")
    print(f"All results saved to: {output_dir}/")
    print("=" * 80)


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 1:  # No command-line args = running from IDE
        # ================================================================
        # IDE-ONLY CONFIGURATION (ignored when run from command line)
        # ================================================================
        use_scaler = True
        use_pca = True
        pca_variance = 0.999
        output_dir = 'clinvar_preds_dedup_'+ STRATEGY
        require_correct_prediction = False
        save_models = True
        models_dir = 'pathogenic_models_dedup'
        base_model_dir = '../tcga_pancan_snv'  # or '../tcga_pancan_snv_sample_loss'

        # Run with IDE configuration
        main(use_scaler=use_scaler, use_pca=use_pca, pca_variance=pca_variance,
             output_dir=output_dir, require_correct_prediction=require_correct_prediction,
             save_models=save_models, models_dir=models_dir, base_model_dir=base_model_dir)