"""
Masking layers for attention mechanisms.

This module contains layers for creating and combining attention masks to control
which positions can attend to each other in attention mechanisms.
"""

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(
    package="tessera.layers.masking",
    name="CreateAttentionMaskLayer",
)
class CreateAttentionMaskLayer(tf.keras.layers.Layer):
    """
    Creates a boolean attention mask from padded input sequences.

    This layer generates a mask that prevents:
    1. Real tokens from attending to padding tokens
    2. Padding tokens from attending to anything

    Input shape:
        ref: (batch_size, seq_length, 1) - Reference sequence with 0 for padding

    Output shape:
        attention_mask: (batch_size, seq_length, seq_length) - Boolean mask

    Example:
        ```python
        mask_layer = CreateAttentionMaskLayer()
        attention_mask = mask_layer(ref_input)  # Shape: [B, T, T]
        ```
    """

    def call(self, ref):
        # Create boolean mask where non-padded values (non-zeros) are True
        # ref shape: [batch_size, seq_length, 1]
        mask = tf.not_equal(ref, 0)  # [batch_size, seq_length, 1]
        # mask = tf.squeeze(mask, axis=-1)  # [batch_size, seq_length]
        mask = tf.reduce_any(mask, axis=-1)  # [num_bags, num_variants]

        # Create attention mask that prevents both:
        # 1. Real tokens from attending to padding tokens
        # 2. Padding tokens from attending to anything
        query_mask = tf.expand_dims(mask, axis=2)  # [batch, T, 1]
        key_mask = tf.expand_dims(mask, axis=1)  # [batch, 1, S]

        # Attention is only allowed when both query and key positions are valid
        attention_mask = tf.logical_and(query_mask, key_mask)  # [batch, T, S]

        return attention_mask


@tf.keras.utils.register_keras_serializable(
    package="tessera.layers.masking",
    name="SelfAttentionMaskLayer",
)
class SelfAttentionMaskLayer(tf.keras.layers.Layer):
    """
    Creates a boolean mask for preventing self-attention in variant prediction.

    Generates a mask with True everywhere except on the diagonal, preventing
    each position from attending to itself. Useful for autoregressive models
    or when computing cross-variant relationships.

    Input shape:
        inputs: Any tensor with shape (batch_size, seq_len, ...)

    Output shape:
        self_mask: (batch_size, seq_len, seq_len) - Boolean mask with False on diagonal

    Example:
        ```python
        self_mask_layer = SelfAttentionMaskLayer()
        mask = self_mask_layer(embeddings)
        # mask[i, j, j] = False, mask[i, j, k] = True (for j != k)
        ```
    """

    def __init__(self, **kwargs):
        super(SelfAttentionMaskLayer, self).__init__(**kwargs)

    def call(self, inputs):
        # Get sequence length dynamically
        seq_len = tf.shape(inputs)[1]
        batch_size = tf.shape(inputs)[0]

        # Create initial mask matrix with True values
        ones_matrix = tf.ones((1, seq_len, seq_len), dtype=tf.bool)

        # Create diagonal mask with False values
        diag_mask = tf.eye(seq_len, dtype=tf.bool)[tf.newaxis, :, :]

        # Combine masks to get False on diagonal and True elsewhere
        self_mask = tf.math.logical_and(ones_matrix, tf.math.logical_not(diag_mask))

        # Broadcast the mask to match batch size
        self_mask = tf.tile(self_mask, [batch_size, 1, 1])

        return self_mask

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[1])

    def get_config(self):
        return super(SelfAttentionMaskLayer, self).get_config()


@tf.keras.utils.register_keras_serializable(
    package="tessera.layers.masking",
    name="CombineAttentionMasksLayer",
)
class CombineAttentionMasksLayer(tf.keras.layers.Layer):
    """
    Combines two boolean attention masks using logical AND.

    Useful for combining padding masks with causal or self-attention masks.
    The result is True only where both input masks are True.

    Input shapes:
        base_mask: (batch_size, seq_len, seq_len) - First boolean mask
        self_mask: (batch_size, seq_len, seq_len) - Second boolean mask

    Output shape:
        combined_mask: (batch_size, seq_len, seq_len) - Logical AND of both masks

    Example:
        ```python
        combine_layer = CombineAttentionMasksLayer()
        final_mask = combine_layer([padding_mask, self_attn_mask])
        ```
    """

    def __init__(self, **kwargs):
        super(CombineAttentionMasksLayer, self).__init__(**kwargs)

    def call(self, inputs):
        base_mask, self_mask = inputs
        return tf.math.logical_and(base_mask, self_mask)

    def compute_output_shape(self, input_shape):
        return input_shape[0]  # Same shape as first input

    def get_config(self):
        return super(CombineAttentionMasksLayer, self).get_config()


# ==============================================================================
# Multi-modal Masking Functions for CNA Integration
# ==============================================================================

def create_padding_mask(valid_flags, target_valid_flags=None):
    """
    Create attention mask to prevent attending to padded entries.

    Args:
        valid_flags: Boolean tensor indicating valid positions [batch, query_len]
                    True = valid position, False = padding
        target_valid_flags: Optional boolean tensor for target/key positions [batch, key_len]
                          If None, uses valid_flags for both query and key (self-attention)

    Returns:
        Attention mask of shape [batch, query_len, key_len]
        True = can attend, False = masked out (padding)

    Example:
        >>> # Self-attention: CNA segments attending to each other
        >>> cna_valid = tf.constant([[True, True, False], [True, False, False]])  # [2, 3]
        >>> mask = create_padding_mask(cna_valid)
        >>> mask.shape
        TensorShape([2, 3, 3])
        >>> # Each valid segment can attend to all valid segments (including itself)
        >>> # Padded segments are masked out in the key dimension
    """
    if target_valid_flags is None:
        target_valid_flags = valid_flags

    # Expand dimensions for broadcasting
    # Query dimension: [batch, query_len, 1]
    # Key dimension: [batch, 1, key_len]
    query_mask = tf.expand_dims(valid_flags, axis=-1)
    key_mask = tf.expand_dims(target_valid_flags, axis=1)

    # Broadcast to [batch, query_len, key_len]
    # A position can be attended to if BOTH query is valid AND key is valid
    # But typically we only mask out invalid keys (let invalid queries attend, they'll be masked later)
    attention_mask = key_mask  # Only mask invalid keys

    return attention_mask


def create_cna_self_attention_mask(cna_chr):
    """
    Create self-attention mask for CNA segments.

    Prevents padded CNA segments (cna_chr == 0) from being attended to.

    Args:
        cna_chr: CNA chromosome IDs [batch, cna_bag_size]
                 0 indicates padding

    Returns:
        Attention mask [batch, cna_bag_size, cna_bag_size]
        True = can attend, False = masked (padding)

    Example:
        >>> cna_chr = tf.constant([[1, 2, 0], [1, 0, 0]])  # [2, 3]
        >>> mask = create_cna_self_attention_mask(cna_chr)
        >>> mask.shape
        TensorShape([2, 3, 3])
        >>> # Valid segments can attend to all valid segments
        >>> # Padded segments (chr=0) are masked out
    """
    # Valid CNA segments have non-zero chromosome
    cna_valid = tf.not_equal(cna_chr, 0)  # [batch, cna_bag_size]

    # Create padding mask for self-attention
    mask = create_padding_mask(cna_valid, cna_valid)

    return mask


def create_snv_to_cna_mask(chr_input, cna_chr):
    """
    Create cross-modal attention mask for SNV variants attending to CNA segments.

    Prevents:
    - Padded variants (chr == 0) from contributing (though their outputs are ignored anyway)
    - Valid variants from attending to padded CNA segments (cna_chr == 0)

    Args:
        chr_input: Variant chromosome IDs [batch, variant_bag_size, 1]
                  0 indicates padding
        cna_chr: CNA chromosome IDs [batch, cna_bag_size]
                0 indicates padding

    Returns:
        Attention mask [batch, variant_bag_size, cna_bag_size]
        True = can attend, False = masked (padding)

    Example:
        >>> chr_input = tf.constant([[[1], [2], [0]], [[1], [0], [0]]])  # [2, 3, 1]
        >>> cna_chr = tf.constant([[1, 2, 0], [1, 0, 0]])  # [2, 3]
        >>> mask = create_snv_to_cna_mask(chr_input, cna_chr)
        >>> mask.shape
        TensorShape([2, 3, 3])
        >>> # Valid variants can attend to valid CNA segments only
    """
    # Squeeze chr_input to [batch, variant_bag_size]
    chr_squeezed = tf.squeeze(chr_input, axis=-1)

    # Valid variants and valid CNA segments
    variant_valid = tf.not_equal(chr_squeezed, 0)  # [batch, variant_bag_size]
    cna_valid = tf.not_equal(cna_chr, 0)  # [batch, cna_bag_size]

    # Create cross-modal padding mask
    mask = create_padding_mask(variant_valid, cna_valid)

    return mask


def create_cna_to_snv_mask(cna_chr, chr_input):
    """
    Create cross-modal attention mask for CNA segments attending to SNV variants.

    Prevents:
    - Padded CNA segments (cna_chr == 0) from contributing (though their outputs are ignored anyway)
    - Valid CNA segments from attending to padded variants (chr == 0)

    Args:
        cna_chr: CNA chromosome IDs [batch, cna_bag_size]
                0 indicates padding
        chr_input: Variant chromosome IDs [batch, variant_bag_size, 1]
                  0 indicates padding

    Returns:
        Attention mask [batch, cna_bag_size, variant_bag_size]
        True = can attend, False = masked (padding)

    Example:
        >>> cna_chr = tf.constant([[1, 2, 0], [1, 0, 0]])  # [2, 3]
        >>> chr_input = tf.constant([[[1], [2], [0]], [[1], [0], [0]]])  # [2, 3, 1]
        >>> mask = create_cna_to_snv_mask(cna_chr, chr_input)
        >>> mask.shape
        TensorShape([2, 3, 3])
        >>> # Valid CNA segments can attend to valid variants only
    """
    # Squeeze chr_input to [batch, variant_bag_size]
    chr_squeezed = tf.squeeze(chr_input, axis=-1)

    # Valid CNA segments and valid variants
    cna_valid = tf.not_equal(cna_chr, 0)  # [batch, cna_bag_size]
    variant_valid = tf.not_equal(chr_squeezed, 0)  # [batch, variant_bag_size]

    # Create cross-modal padding mask
    mask = create_padding_mask(cna_valid, variant_valid)

    return mask


def create_variant_self_attention_mask(chr_input):
    """
    Create self-attention mask for variants (for reference/compatibility).

    This function provides a consistent interface for variant self-attention masking,
    though variant self-attention masking is typically handled in the GlobalAttentionBlock.

    Prevents padded variants (chr == 0) from being attended to.

    Args:
        chr_input: Variant chromosome IDs [batch, variant_bag_size, 1]
                  0 indicates padding

    Returns:
        Attention mask [batch, variant_bag_size, variant_bag_size]
        True = can attend, False = masked (padding)

    Note:
        This does NOT include diagonal masking to prevent self-attention.
        Diagonal masking (preventing variant i from attending to itself)
        should be added separately if needed by the GlobalAttentionBlock.

    Example:
        >>> chr_input = tf.constant([[[1], [2], [0]], [[1], [0], [0]]])  # [2, 3, 1]
        >>> mask = create_variant_self_attention_mask(chr_input)
        >>> mask.shape
        TensorShape([2, 3, 3])
    """
    # Squeeze chr_input to [batch, variant_bag_size]
    chr_squeezed = tf.squeeze(chr_input, axis=-1)

    # Valid variants have non-zero chromosome
    variant_valid = tf.not_equal(chr_squeezed, 0)  # [batch, variant_bag_size]

    # Create padding mask for self-attention
    mask = create_padding_mask(variant_valid, variant_valid)

    return mask


# ==============================================================================
# Keras Layer Versions for Functional API Compatibility
# ==============================================================================

@tf.keras.utils.register_keras_serializable(
    package="tessera.layers.masking",
    name="CNASelfAttentionMaskLayer",
)
class CNASelfAttentionMaskLayer(tf.keras.layers.Layer):
    """
    Layer version of create_cna_self_attention_mask for use in Functional API.

    Creates self-attention mask for CNA segments, preventing padded CNA segments
    (cna_chr == 0) from being attended to.

    Input shape:
        cna_chr: CNA chromosome IDs [batch, cna_bag_size] or [batch, cna_bag_size, 1]
                 0 indicates padding

    Output shape:
        Attention mask [batch, cna_bag_size, cna_bag_size]
        True = can attend, False = masked (padding)

    Example:
        ```python
        cna_mask_layer = CNASelfAttentionMaskLayer()
        mask = cna_mask_layer(cna_chr_input)
        ```
    """

    def call(self, cna_chr):
        # Handle both [batch, cna_bag_size] and [batch, cna_bag_size, 1] shapes
        if len(cna_chr.shape) == 3:
            cna_chr = tf.squeeze(cna_chr, axis=-1)

        # Valid CNA segments have non-zero chromosome
        cna_valid = tf.not_equal(cna_chr, 0)  # [batch, cna_bag_size]

        # Expand dimensions for broadcasting
        query_mask = tf.expand_dims(cna_valid, axis=-1)  # [batch, cna_bag_size, 1]
        key_mask = tf.expand_dims(cna_valid, axis=1)  # [batch, 1, cna_bag_size]

        # Create padding mask: only mask invalid keys
        mask = key_mask  # [batch, cna_bag_size, cna_bag_size]

        return mask

    def get_config(self):
        return super().get_config()


@tf.keras.utils.register_keras_serializable(
    package="tessera.layers.masking",
    name="SNVToCNAMaskLayer",
)
class SNVToCNAMaskLayer(tf.keras.layers.Layer):
    """
    Layer version of create_snv_to_cna_mask for use in Functional API.

    Creates cross-modal attention mask for SNV variants attending to CNA segments.

    Prevents:
    - Valid variants from attending to padded CNA segments (cna_chr == 0)

    Input shapes:
        chr_input: Variant chromosome IDs [batch, variant_bag_size, 1]
                  0 indicates padding
        cna_chr: CNA chromosome IDs [batch, cna_bag_size] or [batch, cna_bag_size, 1]
                0 indicates padding

    Output shape:
        Attention mask [batch, variant_bag_size, cna_bag_size]
        True = can attend, False = masked (padding)

    Example:
        ```python
        snv_to_cna_mask_layer = SNVToCNAMaskLayer()
        mask = snv_to_cna_mask_layer([chr_input, cna_chr])
        ```
    """

    def call(self, inputs):
        chr_input, cna_chr = inputs

        # Squeeze chr_input to [batch, variant_bag_size]
        chr_squeezed = tf.squeeze(chr_input, axis=-1)

        # Handle both shapes for cna_chr
        if len(cna_chr.shape) == 3:
            cna_chr = tf.squeeze(cna_chr, axis=-1)

        # Valid variants and valid CNA segments
        variant_valid = tf.not_equal(chr_squeezed, 0)  # [batch, variant_bag_size]
        cna_valid = tf.not_equal(cna_chr, 0)  # [batch, cna_bag_size]

        # Expand dimensions for broadcasting
        query_mask = tf.expand_dims(variant_valid, axis=-1)  # [batch, variant_bag_size, 1]
        key_mask = tf.expand_dims(cna_valid, axis=1)  # [batch, 1, cna_bag_size]

        # Create cross-modal padding mask: only mask invalid keys
        mask = key_mask  # [batch, variant_bag_size, cna_bag_size]

        return mask

    def get_config(self):
        return super().get_config()


@tf.keras.utils.register_keras_serializable(
    package="tessera.layers.masking",
    name="CNAToSNVMaskLayer",
)
class CNAToSNVMaskLayer(tf.keras.layers.Layer):
    """
    Layer version of create_cna_to_snv_mask for use in Functional API.

    Creates cross-modal attention mask for CNA segments attending to SNV variants.

    Prevents:
    - Valid CNA segments from attending to padded variants (chr == 0)

    Input shapes:
        cna_chr: CNA chromosome IDs [batch, cna_bag_size] or [batch, cna_bag_size, 1]
                0 indicates padding
        chr_input: Variant chromosome IDs [batch, variant_bag_size, 1]
                  0 indicates padding

    Output shape:
        Attention mask [batch, cna_bag_size, variant_bag_size]
        True = can attend, False = masked (padding)

    Example:
        ```python
        cna_to_snv_mask_layer = CNAToSNVMaskLayer()
        mask = cna_to_snv_mask_layer([cna_chr, chr_input])
        ```
    """

    def call(self, inputs):
        cna_chr, chr_input = inputs

        # Squeeze chr_input to [batch, variant_bag_size]
        chr_squeezed = tf.squeeze(chr_input, axis=-1)

        # Handle both shapes for cna_chr
        if len(cna_chr.shape) == 3:
            cna_chr = tf.squeeze(cna_chr, axis=-1)

        # Valid CNA segments and valid variants
        cna_valid = tf.not_equal(cna_chr, 0)  # [batch, cna_bag_size]
        variant_valid = tf.not_equal(chr_squeezed, 0)  # [batch, variant_bag_size]

        # Expand dimensions for broadcasting
        query_mask = tf.expand_dims(cna_valid, axis=-1)  # [batch, cna_bag_size, 1]
        key_mask = tf.expand_dims(variant_valid, axis=1)  # [batch, 1, variant_bag_size]

        # Create cross-modal padding mask: only mask invalid keys
        mask = key_mask  # [batch, cna_bag_size, variant_bag_size]

        return mask

    def get_config(self):
        return super().get_config()