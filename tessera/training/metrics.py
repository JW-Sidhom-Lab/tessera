"""Metrics for variant sequence prediction evaluation."""

from typing import Optional
import tensorflow as tf
import numpy as np


def masked_accuracy(
    y_true: tf.Tensor,
    logits: tf.Tensor,
    non_zero_only: bool = False
) -> tf.Tensor:
    """
    Compute per-variant accuracy for sequence reconstruction.

    Calculates exact-match accuracy at the variant level: a variant is considered
    correct only if ALL positions in the sequence are predicted correctly.

    Args:
        y_true: True alteration sequences of shape [batch, num_variants, seq_len]
        logits: Raw model predictions (logits) of shape [batch, num_variants, seq_len, vocab_size]
        non_zero_only: If True, only require non-zero elements to be correct.
                      Padding positions (zeros) are automatically considered correct.
                      If False, ALL positions (including padding) must match.

    Returns:
        Tensor of shape [batch, num_variants] where each element is:
            - 1.0 if ALL positions in that variant are correctly predicted
            - 0.0 if ANY position is incorrectly predicted
            - 0.0 for completely padded variants (no non-zero positions)

    Example:
        >>> y_true = tf.constant([[[1, 2, 0], [3, 4, 0]]])  # [1, 2, 3]
        >>> logits = tf.random.normal([1, 2, 3, 5])
        >>> acc = masked_accuracy(y_true, logits, non_zero_only=True)
        >>> acc.shape
        TensorShape([1, 2])

    Note:
        This is a strict metric - partial matches are not rewarded. Use this to
        ensure complete sequence reconstruction accuracy.
    """
    y_true = tf.cast(y_true, dtype=tf.int64)

    # Get predicted classes first to determine the computation dtype
    y_pred_class = tf.argmax(logits, axis=-1)
    y_pred_class = tf.cast(y_pred_class, dtype=tf.int64)

    # Check if predictions are correct for each position
    correct_predictions = tf.equal(y_true, y_pred_class)

    if non_zero_only:
        # Only require non-zero positions to be correct
        # Create a mask for non-zero elements in y_true
        non_zero_mask = tf.not_equal(y_true, 0)
        
        # Apply mask: padding positions (zeros) are automatically considered "correct"
        # Only non-zero positions need to actually match
        masked_correct = tf.logical_or(correct_predictions, tf.logical_not(non_zero_mask))
        
        # For each variant, check if all relevant positions are correct
        all_positions_correct = tf.reduce_all(masked_correct, axis=2)
    else:
        # Require all positions (including padding) to be correct
        all_positions_correct = tf.reduce_all(correct_predictions, axis=2)
    
    # Cast to logits dtype for mixed precision compatibility
    all_positions_correct = tf.cast(all_positions_correct, logits.dtype)

    # Create mask for non-zero variants and cast to match computation dtype
    variant_has_nonzero = tf.reduce_any(tf.not_equal(y_true, 0), axis=2)  # [batch, num_variants]
    final_mask = tf.cast(variant_has_nonzero, logits.dtype)

    # Apply mask to exclude invalid variants (set to 0.0)
    per_variant_accuracy = all_positions_correct * final_mask

    return per_variant_accuracy  # Shape: [batch, num_variants]




@tf.keras.utils.register_keras_serializable(
    package="tessera.metrics",
    name="AUCLayer"
)
class AUCLayer(tf.keras.layers.Layer):
    """
    Keras layer for computing multi-class Area Under the ROC Curve (AUC).

    This layer computes per-class AUC scores and returns the macro-averaged AUC.
    Useful as a differentiable metric during model training, though note that
    it uses py_function internally which may impact graph optimization.

    Args:
        num_classes: Number of classes for multi-class AUC computation
        name: Layer name. Default: 'auc'
        **kwargs: Additional layer arguments

    Example:
        >>> auc_layer = AUCLayer(num_classes=3)
        >>> y_true = tf.constant([[0], [1], [2], [1]])
        >>> logits = tf.random.normal([4, 3])
        >>> auc_score = auc_layer([y_true, logits])
    """

    def __init__(self, num_classes: int, name: str = 'auc', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes

    def call(self, inputs):
        """
        Compute macro-averaged AUC across all classes.

        Args:
            inputs: Tuple of (y_true, logits) where:
                   - y_true: True labels of shape [batch_size, 1] or [batch_size]
                   - logits: Raw predictions of shape [batch_size, num_classes]

        Returns:
            Scalar tensor containing the macro-averaged AUC score
        """
        y_true, logits = inputs

        # Ensure y_true is 1D
        y_true = tf.squeeze(y_true, axis=-1)

        # Convert labels to one-hot encoding
        y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=self.num_classes)

        # Convert logits to probabilities
        y_pred_probs = tf.nn.softmax(logits)

        # Compute AUC for each class
        auc_scores = []
        for class_idx in range(self.num_classes):
            y_true_class = y_true_one_hot[:, class_idx]
            y_pred_class = y_pred_probs[:, class_idx]

            auc = tf.py_function(
                compute_auc,
                [y_true_class, y_pred_class],
                tf.float32
            )
            auc_scores.append(auc)

        # Compute macro-averaged AUC (mean across all classes)
        mean_auc = tf.reduce_mean(auc_scores)

        return mean_auc

    def get_config(self):
        """
        Get layer configuration for serialization.

        Returns:
            Dictionary containing layer configuration
        """
        config = super().get_config()
        config.update({"num_classes": self.num_classes})
        return config


def compute_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute binary Area Under the ROC Curve using trapezoidal rule.

    This function is designed to be called from TensorFlow's py_function and
    operates on numpy arrays. It handles edge cases where all samples belong
    to a single class.

    Args:
        y_true: Binary true labels (0 or 1) as numpy array
        y_pred: Predicted probabilities as numpy array

    Returns:
        AUC score as float. Returns 0.5 if all samples are of one class.

    Note:
        This is a simple implementation for demonstration. For production use,
        consider using sklearn.metrics.roc_auc_score which handles edge cases better.

    Example:
        >>> y_true = np.array([0, 0, 1, 1])
        >>> y_pred = np.array([0.1, 0.4, 0.35, 0.8])
        >>> auc = compute_auc(y_true, y_pred)
        >>> 0.0 <= auc <= 1.0
        True
    """
    # Convert tensors to numpy arrays if needed
    if hasattr(y_true, 'numpy'):
        y_true = y_true.numpy()
    if hasattr(y_pred, 'numpy'):
        y_pred = y_pred.numpy()

    # Handle edge case: all samples of one class
    if np.all(y_true == 1) or np.all(y_true == 0):
        return 0.5

    # Sort by predicted probabilities (descending)
    sorted_indices = np.argsort(y_pred)[::-1]
    y_true_sorted = y_true[sorted_indices]

    # Compute cumulative True Positive Rate and False Positive Rate
    n_positives = np.sum(y_true_sorted)
    n_negatives = len(y_true_sorted) - n_positives

    # Avoid division by zero
    if n_positives == 0 or n_negatives == 0:
        return 0.5

    tpr = np.cumsum(y_true_sorted) / n_positives
    fpr = np.cumsum(1 - y_true_sorted) / n_negatives

    # Compute AUC using trapezoidal rule
    auc = np.trapz(tpr, fpr)

    return float(auc)