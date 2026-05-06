"""
Pooling layers with masking support.

This module provides pooling layers that properly handle variable-length sequences
with padding masks, ensuring padding does not contribute to aggregated representations.
"""

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(
    package="tessera.layers.pooling",
    name="CreateMaskLayer"
)
class CreateMaskLayer(tf.keras.layers.Layer):
    """
    Creates a boolean mask from input tensor by checking for non-zero values.

    This layer is designed to work within Keras Functional API where TensorFlow
    operations cannot be used directly on KerasTensors during model construction.

    Args:
        squeeze: Whether to squeeze the last dimension. Default: True
        **kwargs: Standard layer arguments

    Input Shape:
        [batch_size, sequence_length] or [batch_size, sequence_length, 1]

    Output Shape:
        [batch_size, sequence_length] - Boolean mask where True indicates non-zero

    Example:
        >>> # Create mask from chromosome IDs where 0 = padding
        >>> mask_layer = CreateMaskLayer(squeeze=True)
        >>> chr_input = tf.keras.Input(shape=(100, 1), dtype=tf.int32)
        >>> mask = mask_layer(chr_input)  # [batch, 100] boolean mask
    """

    def __init__(self, squeeze=True, **kwargs):
        super(CreateMaskLayer, self).__init__(**kwargs)
        self.squeeze = squeeze

    def call(self, inputs):
        """
        Create boolean mask from inputs.

        Args:
            inputs: Tensor to create mask from

        Returns:
            Boolean mask where True indicates non-zero values
        """
        # Squeeze if needed
        if self.squeeze and len(inputs.shape) == 3:
            inputs = tf.squeeze(inputs, axis=-1)

        # Create mask: True where input is non-zero
        mask = tf.not_equal(inputs, 0)

        return mask

    def get_config(self):
        config = super().get_config()
        config.update({'squeeze': self.squeeze})
        return config


@tf.keras.utils.register_keras_serializable(
    package="tessera.layers.pooling",
    name="MaskedGlobalAveragePooling1D"
)
class MaskedGlobalAveragePooling1D(tf.keras.layers.Layer):
    """
    Global Average Pooling over temporal/sequence dimension with mask support.

    Unlike tf.keras.layers.GlobalAveragePooling1D, this layer explicitly handles
    padding masks by only averaging over valid (non-padded) positions.

    This is critical for applications like Multiple Instance Learning or contrastive
    learning where bags have variable lengths and padding should not contribute to
    the aggregated representation.

    Args:
        **kwargs: Standard layer arguments

    Input Shapes:
        features: Tensor of shape [batch_size, sequence_length, feature_dim]
        mask: Boolean tensor of shape [batch_size, sequence_length]
              True = valid position, False = padding

    Output Shape:
        [batch_size, feature_dim] - averaged features over valid positions only

    Example:
        >>> # Sample with 20 valid variants out of 100 positions
        >>> features = tf.random.normal([32, 100, 256])
        >>> mask = tf.concat([
        ...     tf.ones([32, 20], dtype=tf.bool),   # First 20 are valid
        ...     tf.zeros([32, 80], dtype=tf.bool)   # Rest are padding
        ... ], axis=1)
        >>>
        >>> pooling = MaskedGlobalAveragePooling1D()
        >>> output = pooling(features, mask=mask)
        >>> output.shape
        TensorShape([32, 256])
        >>> # output is average of first 20 positions only, not all 100

    Mathematical Formulation:
        For each feature dimension d:
        output[b, d] = sum_{i=1}^{N} (features[b, i, d] * mask[b, i]) / sum_{i=1}^{N} mask[b, i]

        where:
        - b = batch index
        - i = sequence position
        - N = sequence length
        - Only positions where mask[b, i] = True contribute to the average

    Note:
        - If all positions are masked (mask is all False), returns zeros with epsilon
        - Mask dtype is automatically cast to match features dtype for computation
        - Epsilon is added to denominator to prevent division by zero
    """

    def __init__(self, **kwargs):
        super(MaskedGlobalAveragePooling1D, self).__init__(**kwargs)

    def call(self, features, mask=None):
        """
        Forward pass with masked averaging.

        Args:
            features: Input features [batch, sequence_length, feature_dim]
            mask: Boolean mask [batch, sequence_length], True = valid position

        Returns:
            Averaged features [batch, feature_dim]
        """
        # If no mask provided, fall back to standard global average pooling
        if mask is None:
            return tf.reduce_mean(features, axis=1)

        eps = tf.keras.backend.epsilon()

        # Expand mask to match feature dimensions: [batch, seq_len] -> [batch, seq_len, 1]
        mask_expanded = tf.expand_dims(mask, axis=-1)

        # Cast mask to match features dtype (important for mixed precision training)
        mask_expanded = tf.cast(mask_expanded, dtype=features.dtype)

        # Apply mask: multiply features by mask to zero out padded positions
        masked_features = features * mask_expanded  # [batch, seq_len, feature_dim]

        # Sum over sequence dimension
        sum_features = tf.reduce_sum(masked_features, axis=1)  # [batch, feature_dim]

        # Count number of valid positions per sample
        count_valid = tf.reduce_sum(mask_expanded, axis=1)  # [batch, 1]

        # Compute average: divide sum by count (add epsilon to avoid division by zero)
        mean_features = sum_features / (count_valid + eps)

        return mean_features

    def compute_output_shape(self, input_shape):
        """Compute output shape: removes sequence dimension."""
        return (input_shape[0], input_shape[2])  # [batch, feature_dim]

    def get_config(self):
        """Get layer configuration for serialization."""
        return super().get_config()
