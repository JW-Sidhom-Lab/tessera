"""
Attention mechanism layers for variant sequence modeling.

This module contains all attention-related layers including:
- MultiHeadAttention: Custom multi-head attention with raw scores and L1 regularization
- LocalAttentionBlock: Local (within-sequence) attention mechanism
- GlobalAttentionBlock: Global (cross-sequence) attention mechanism with masking

All layers support XLA compilation and mixed precision training.
"""

import tensorflow as tf
from keras.src import ops


@tf.keras.utils.register_keras_serializable(
    package="tessera.layers.attention",
    name="CustomMultiHeadAttention",
)
class CustomMultiHeadAttention(tf.keras.layers.MultiHeadAttention):
    """
    Custom MultiHeadAttention that inherits from standard TensorFlow MultiHeadAttention.

    Modifications:
    - Returns raw attention scores before softmax for interpretability
    - Optional row-wise L1 regularization for sparse attention patterns

    Args:
        attention_l1_factor: L1 regularization factor for attention sparsity.
                            Row-wise L1 penalty encourages each variant to attend to fewer others.
                            0.0 = no regularization, 0.01 = light, 0.1 = strong
        **kwargs: Arguments passed to parent MultiHeadAttention

    Example:
        ```python
        attention_layer = CustomMultiHeadAttention(
            num_heads=8,
            key_dim=64,
            attention_l1_factor=0.01  # Light L1 regularization
        )
        output, scores = attention_layer(query, key, value, return_attention_scores=True)
        ```
    """

    def __init__(self, attention_l1_factor: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.attention_l1_factor = attention_l1_factor

    def _compute_attention(
            self, query, key, value, attention_mask=None, training=None, precomputed_attention_scores=None
    ):
        """
        Override _compute_attention to return raw attention scores.
        Identical to parent implementation except stores raw scores before softmax.
        """
        # Scale the query (matches original)
        query = ops.multiply(
            query, ops.cast(self._inverse_sqrt_key_dim, query.dtype)
        )

        # Use precomputed attention scores if provided, otherwise compute them
        if precomputed_attention_scores is not None:
            attention_scores = precomputed_attention_scores
        else:
            # Take the dot product between "query" and "key" to get the raw
            # attention scores.
            attention_scores = ops.einsum(self._dot_product_equation, key, query)

        # Store pre-softmax scores for interpretability
        raw_attention_scores = attention_scores

        # Add row-wise L1 regularization for sparse attention per variant
        if training and self.attention_l1_factor > 0:
            # Sum L1 across each row (each variant's attention pattern)
            # Shape: [batch, heads, seq_len, seq_len] -> [batch, heads, seq_len]
            row_l1 = tf.reduce_sum(tf.abs(raw_attention_scores), axis=-1)
            l1_loss = self.attention_l1_factor * tf.reduce_mean(row_l1)
            self.add_loss(l1_loss)

        attention_scores = self._masked_softmax(
            attention_scores, attention_mask
        )

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        if self.dropout:
            final_attn_scores = self._dropout_layer(
                attention_scores, training=training
            )
        else:
            final_attn_scores = attention_scores

        # `context_layer` = [B, T, N, H]
        attention_output = ops.einsum(
            self._combine_equation, final_attn_scores, value
        )

        # Return both attention output and raw scores
        return attention_output, raw_attention_scores

    def call(
        self,
        query,
        value,
        key=None,
        query_mask=None,
        value_mask=None,
        key_mask=None,
        attention_mask=None,
        return_attention_scores=False,
        training=None,
        use_causal_mask=False,
        precomputed_attention_scores=None,
    ):
        if key is None:
            key = value

        attention_mask = self._compute_attention_mask(
            query,
            value,
            query_mask=query_mask,
            value_mask=value_mask,
            key_mask=key_mask,
            attention_mask=attention_mask,
            use_causal_mask=use_causal_mask,
        )

        #   N = `num_attention_heads`
        #   H = `size_per_head`
        # `query` = [B, T, N ,H]
        query = self._query_dense(query)

        # `key` = [B, S, N, H]
        key = self._key_dense(key)

        # `value` = [B, S, N, H]
        value = self._value_dense(value)

        attention_output, attention_scores = self._compute_attention(
            query, key, value, attention_mask, training, precomputed_attention_scores
        )
        attention_output = self._output_dense(attention_output)

        if return_attention_scores:
            return attention_output, attention_scores
        return attention_output

    def get_config(self):
        """Get layer configuration including attention_l1_factor."""
        config = super().get_config()
        config.update({
            'attention_l1_factor': self.attention_l1_factor,
        })
        return config


@tf.keras.utils.register_keras_serializable(
    package="tessera.layers.attention",
    name="LocalAttentionBlock",
)
class LocalAttentionBlock(tf.keras.layers.Layer):
    """
    Local attention block for processing sequence context within variants.

    Applies cross-attention between a query sequence (variant) and key/value sequences (genomic context).
    Follows transformer best practices with pre-normalization and residual connections.

    Args:
        num_heads: Number of attention heads
        embed_dim: Embedding dimension for projections
        ff_dim: Feed-forward network dimension
        output_dim: Final output dimension
        attention_activation_type: Activation type for attention gates (currently unused)
        dropout_rate: Dropout rate for regularization
        use_bias: Whether to use bias in dense layers
        layer_norm_eps: Layer normalization epsilon
        use_head_scaling: Whether to use learnable head scaling (currently unused)
        head_scale_init: Initial value for head scales (currently unused)

    Input shapes:
        seq1: Query sequence (batch_size, num_sequences, embed_dim)
        seq2: Key/Value sequence (batch_size, num_sequences, key_dim)

    Output shapes:
        final_output: (batch_size, num_sequences, output_dim)
        attention_scores: (batch_size, num_heads, num_sequences, seq_length)

    Example:
        ```python
        local_block = LocalAttentionBlock(
            num_heads=4,
            embed_dim=128,
            ff_dim=256,
            output_dim=128,
            dropout_rate=0.1
        )
        output, attn_scores = local_block([query_seq, context_seq])
        ```
    """

    def __init__(
        self,
        num_heads: int,
        embed_dim: int,
        ff_dim: int,
        output_dim: int,
        attention_activation_type: str = 'sigmoid',
        dropout_rate: float = 0.1,
        use_bias: bool = True,
        layer_norm_eps: float = 1e-6,
        use_head_scaling: bool = False,
        head_scale_init: float = 1.0,
        **kwargs
    ):
        super(LocalAttentionBlock, self).__init__(**kwargs)

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.output_dim = output_dim
        self.attention_activation_type = attention_activation_type
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.layer_norm_eps = layer_norm_eps
        self.use_head_scaling = use_head_scaling
        self.head_scale_init = head_scale_init

        # Pre-norm layer normalization
        self.seq1_norm = tf.keras.layers.LayerNormalization(
            epsilon=layer_norm_eps,
            center=False,
            scale=False,
            name=f'{self.name}_seq1_norm'
        )

        self.seq2_norm = tf.keras.layers.LayerNormalization(
            epsilon=layer_norm_eps,
            center=False,
            scale=False,
            name=f'{self.name}_seq2_norm'
        )

        # Projection layers to match embed_dim
        self.seq1_proj = tf.keras.layers.Dense(
            embed_dim,
            use_bias=use_bias,
            kernel_initializer='glorot_uniform',
            name=f'{self.name}_seq1_projection'
        )

        self.seq2_proj = tf.keras.layers.Dense(
            embed_dim,
            use_bias=use_bias,
            kernel_initializer='glorot_uniform',
            name=f'{self.name}_seq2_projection'
        )

        # Custom MultiHead Attention
        self.attention = CustomMultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout_rate,
            use_bias=False,
            attention_axes=(2,),  # attend only over the tokens axis of (B, G, T, C)
            name=f'{self.name}_custom_local_attention'
        )

        # Dropout after attention
        self.attention_dropout = tf.keras.layers.Dropout(dropout_rate, name=f'{self.name}_attention_dropout')

        # Concatenation layer
        self.concat = tf.keras.layers.Concatenate(axis=-1, name=f'{self.name}_context_attention_concat')

        # Combined projection
        self.combined_proj = tf.keras.layers.Dense(
            ff_dim,
            activation='gelu',
            use_bias=use_bias,
            kernel_initializer='glorot_uniform',
            name=f'{self.name}_combined_projection'
        )

        # Pre-FFN norm
        self.pre_ffn_norm = tf.keras.layers.LayerNormalization(
            epsilon=layer_norm_eps,
            center=False,
            scale=False,
            name=f'{self.name}_pre_ffn_norm'
        )

        # Feed forward layers
        self.ffn_layers = []
        self.ffn_dropouts = []
        for i in range(3):
            self.ffn_layers.append(tf.keras.layers.Dense(
                ff_dim,
                activation='gelu',
                use_bias=use_bias,
                kernel_initializer='glorot_uniform',
                name=f'{self.name}_ffn_layer_{i+1}'
            ))
            self.ffn_dropouts.append(tf.keras.layers.Dropout(dropout_rate, name=f'{self.name}_ffn_dropout_{i+1}'))

        # Final projection
        self.ffn_output_layer = tf.keras.layers.Dense(
            output_dim,
            use_bias=use_bias,
            kernel_initializer='glorot_uniform',
            name=f'{self.name}_ffn_output'
        )
        self.ffn_final_dropout = tf.keras.layers.Dropout(dropout_rate, name=f'{self.name}_ffn_final_dropout')

        # Output projection for residual connection
        self.out1_projection = tf.keras.layers.Dense(
            output_dim,
            use_bias=use_bias,
            kernel_initializer='glorot_uniform',
            name=f'{self.name}_out1_projection'
        )

        # Final residual connection
        self.final_add = tf.keras.layers.Add(name=f'{self.name}_final_residual')

    def call(self, inputs, training=None, precomputed_attention_scores=None):
        """
        Forward pass through the local attention block.

        Args:
            inputs: List of [seq1, seq2] tensors
            training: Whether in training mode
            precomputed_attention_scores: Optional precomputed attention scores to reuse

        Returns:
            final_output: Processed sequence (batch_size, seq_len, output_dim)
            attention_scores: Attention scores for visualization
        """
        seq1, seq2 = inputs

        # Pre-norm layer normalization
        seq1_norm = self.seq1_norm(seq1)
        seq2_norm = self.seq2_norm(seq2)

        # Project both sequences to the same embed_dim
        seq1_proj = self.seq1_proj(seq1_norm)
        seq2_proj = self.seq2_proj(seq2_norm)

        # Reverse the post-processing on precomputed attention scores if provided
        raw_precomputed_scores = None
        if precomputed_attention_scores is not None:
            # Reverse the transpose: [B, num_heads, num_seq, 20] -> [B, num_seq, num_heads, 20]
            raw_precomputed_scores = tf.transpose(precomputed_attention_scores, [0, 2, 1, 3])
            # Reverse the squeeze: add back the dimension that was removed
            raw_precomputed_scores = tf.expand_dims(raw_precomputed_scores, axis=-2)

        # Use the pre-built attention layer
        attention_output, attention_scores = self.attention(
            seq1_proj, seq2_proj, seq2_proj,
            return_attention_scores=True,
            precomputed_attention_scores=raw_precomputed_scores
        )

        attention_scores = tf.squeeze(attention_scores, axis=-2)  # Remove the dimension with size 1
        attention_scores = tf.transpose(attention_scores, [0, 2, 1, 3])  # Reshape to [B, num_heads, num_seq, 20]

        # Add dropout after attention
        attention_output = self.attention_dropout(attention_output, training=training)

        # Concatenate seq1 with attention output
        combined = self.concat([seq1_norm, attention_output])

        # Project concatenated features
        out1 = self.combined_proj(combined)

        # Pre-norm for feed-forward network
        ffn_input = self.pre_ffn_norm(out1)

        # Feed forward network
        ffn_output = ffn_input
        for i in range(3):
            ffn_output = self.ffn_layers[i](ffn_output)
            ffn_output = self.ffn_dropouts[i](ffn_output, training=training)

        # Final projection
        ffn_output = self.ffn_output_layer(ffn_output)
        ffn_output = self.ffn_final_dropout(ffn_output, training=training)

        # Project out1 to output_dim for residual connection
        out1_projected = self.out1_projection(out1)

        # Final residual connection
        final_output = self.final_add([ffn_output, out1_projected])

        return final_output, attention_scores

    def get_config(self):
        config = super(LocalAttentionBlock, self).get_config()
        config.update({
            'num_heads': self.num_heads,
            'embed_dim': self.embed_dim,
            'ff_dim': self.ff_dim,
            'output_dim': self.output_dim,
            'attention_activation_type': self.attention_activation_type,
            'dropout_rate': self.dropout_rate,
            'use_bias': self.use_bias,
            'layer_norm_eps': self.layer_norm_eps,
            'use_head_scaling': self.use_head_scaling,
            'head_scale_init': self.head_scale_init
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable(
    package="tessera.layers.attention",
    name="GlobalAttentionBlock",
)
class GlobalAttentionBlock(tf.keras.layers.Layer):
    """
    Global attention block for variant-to-variant cross-attention.

    Enables variants to attend to other variants in the sequence with proper masking
    to prevent self-attention. Supports both pairwise and mean-based attention modes.

    Args:
        embed_dim: Dimension of embeddings
        num_heads: Number of attention heads
        ff_dim: Feed forward dimension
        attention_type: Type of attention mechanism ('pairwise' or 'mean_based')
        attention_activation_type: Activation type for attention gates
        dropout_rate: Dropout rate for regularization
        use_bias: Whether to use bias in dense layers
        layer_norm_eps: Layer normalization epsilon
        use_head_scaling: Whether to use learnable head scaling (currently unused)
        attention_l1_factor: L1 regularization factor for attention sparsity
        head_scale_init: Initial value for head scales (currently unused)

    Input shapes:
        ref_context: Reference sequence embeddings (batch_size, seq_len, embed_dim)
        mut_context: Mutation context embeddings (batch_size, seq_len, embed_dim)
        attention_mask: Boolean mask (batch_size, seq_len, seq_len)

    Output shapes:
        final_output: (batch_size, seq_len, embed_dim)
        attention_scores: (batch_size, num_heads, seq_len, seq_len)

    Example:
        ```python
        global_block = GlobalAttentionBlock(
            embed_dim=256,
            num_heads=8,
            ff_dim=512,
            attention_type='pairwise',
            dropout_rate=0.1
        )
        output, attn_scores = global_block([ref_ctx, mut_ctx, mask])
        ```
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        attention_type: str,
        attention_activation_type: str = 'sigmoid',
        dropout_rate: float = 0.1,
        use_bias: bool = True,
        layer_norm_eps: float = 1e-6,
        use_head_scaling: bool = False,
        attention_l1_factor: float = 0.0,
        head_scale_init: float = 1.0,
        **kwargs
    ):
        super(GlobalAttentionBlock, self).__init__(**kwargs)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.attention_type = attention_type
        self.attention_activation_type = attention_activation_type
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.layer_norm_eps = layer_norm_eps
        self.use_head_scaling = use_head_scaling
        self.head_scale_init = head_scale_init

        # Masking layers removed - masks will be created externally and passed in

        if attention_type == 'pairwise':
            # Pre-norm layer normalization
            self.ref_norm = tf.keras.layers.LayerNormalization(
                epsilon=layer_norm_eps,
                center=False,
                scale=False,
                name=f'{self.name}_ref_pre_norm'
            )

            self.mut_norm = tf.keras.layers.LayerNormalization(
                epsilon=layer_norm_eps,
                center=False,
                scale=False,
                name=f'{self.name}_mut_pre_norm'
            )

            # CustomMultiHeadAttention for safe masking
            self.attention = CustomMultiHeadAttention(
                num_heads=num_heads,
                key_dim=embed_dim // num_heads,
                dropout=dropout_rate,
                use_bias=False,
                attention_l1_factor=attention_l1_factor,
                name=f'{self.name}_standard_attention'
            )

        elif attention_type == 'mean_based':
            # Pre-norm layer normalization
            self.ref_mean_norm = tf.keras.layers.LayerNormalization(
                epsilon=layer_norm_eps,
                center=False,
                scale=False,
                name=f'{self.name}_ref_mean_norm'
            )

            self.mut_mean_norm = tf.keras.layers.LayerNormalization(
                epsilon=layer_norm_eps,
                center=False,
                scale=False,
                name=f'{self.name}_mut_mean_norm'
            )

            # Standard MultiHeadAttention for mean-based
            self.attention = tf.keras.layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=embed_dim // num_heads,
                dropout=dropout_rate,
                kernel_initializer='glorot_uniform',
                use_bias=False,
                name=f'{self.name}_mean_attention'
            )
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")

        # Common layers
        self.attention_dropout = tf.keras.layers.Dropout(dropout_rate, name=f'{self.name}_attention_dropout')
        self.concat = tf.keras.layers.Concatenate(axis=-1, name=f'{self.name}_context_attention_concat')

        # Combined projection
        self.combined_proj = tf.keras.layers.Dense(
            ff_dim,
            activation='gelu',
            use_bias=use_bias,
            kernel_initializer='glorot_uniform',
            name=f'{self.name}_combined_projection'
        )

        # Pre-FFN norm
        self.pre_ffn_norm = tf.keras.layers.LayerNormalization(
            epsilon=layer_norm_eps,
            center=False,
            scale=False,
            name=f'{self.name}_pre_ffn_norm'
        )

        # Feed forward layers
        self.ffn_layers = []
        self.ffn_dropouts = []
        for i in range(3):
            self.ffn_layers.append(tf.keras.layers.Dense(
                ff_dim,
                activation='gelu',
                use_bias=use_bias,
                kernel_initializer='glorot_uniform',
                name=f'{self.name}_ffn_layer_{i+1}'
            ))
            self.ffn_dropouts.append(tf.keras.layers.Dropout(dropout_rate, name=f'{self.name}_ffn_dropout_{i+1}'))

        # Final projection
        self.ffn_output_layer = tf.keras.layers.Dense(
            embed_dim,
            use_bias=use_bias,
            kernel_initializer='glorot_uniform',
            name=f'{self.name}_ffn_output'
        )
        self.ffn_final_dropout = tf.keras.layers.Dropout(dropout_rate, name=f'{self.name}_ffn_final_dropout')

        # Output projection for residual connection
        self.out1_projection = tf.keras.layers.Dense(
            embed_dim,
            use_bias=use_bias,
            kernel_initializer='glorot_uniform',
            name=f'{self.name}_out1_projection'
        )

        # Final residual connection
        self.final_add = tf.keras.layers.Add(name=f'{self.name}_final_residual')

    def call(self, inputs, training=None, precomputed_attention_scores=None):
        """
        Forward pass through the global attention block.

        Args:
            inputs: List of [ref_context, mut_context, attention_mask] tensors
            training: Whether in training mode
            precomputed_attention_scores: Optional precomputed attention scores to reuse

        Returns:
            final_output: Processed sequence (batch_size, seq_len, embed_dim)
            attention_scores: Attention scores for visualization
        """
        ref_context, mut_context, attention_mask = inputs

        if self.attention_type == 'pairwise':
            # Pre-norm layer normalization
            ref_context_norm = self.ref_norm(ref_context)
            mut_context_norm = self.mut_norm(mut_context)

            # Use CustomMultiHeadAttention
            attention_output, attention_scores = self.attention(
                ref_context_norm, mut_context_norm, mut_context_norm,
                return_attention_scores=True,
                attention_mask=attention_mask,
                precomputed_attention_scores=precomputed_attention_scores
            )

        elif self.attention_type == 'mean_based':
            # Pre-norm for inputs
            ref_context_norm = self.ref_mean_norm(ref_context)
            mut_context_norm = self.mut_mean_norm(mut_context)

            # For mean-based attention, extract self-mask from the attention_mask
            # The attention_mask contains both base mask and self-mask information
            # We can use the diagonal to extract self-attention information
            self_mask = tf.cast(attention_mask, tf.float32)
            self_mask = 1.0 - tf.linalg.diag_part(self_mask)  # Invert diagonal to get valid positions

            # Mask out self-attention before taking the mean
            masked_mut_context = mut_context_norm * tf.expand_dims(self_mask, -1)
            mut_context_mean = tf.reduce_sum(masked_mut_context, axis=1, keepdims=True) / \
                               tf.reduce_sum(self_mask, axis=-1, keepdims=True)[..., tf.newaxis]

            attention_output, attention_scores = self.attention(
                ref_context_norm, mut_context_mean, return_attention_scores=True
            )

        # Add dropout after attention
        attention_output = self.attention_dropout(attention_output, training=training)

        # Concatenate reference context with attention output
        if self.attention_type == 'pairwise':
            combined = self.concat([ref_context_norm, attention_output])
        else:  # mean_based
            combined = self.concat([ref_context_norm, attention_output])

        # Project concatenated features
        out1 = self.combined_proj(combined)

        # Pre-norm for feed-forward network
        ffn_input = self.pre_ffn_norm(out1)

        # Feed forward network
        ffn_output = ffn_input
        for i in range(3):
            ffn_output = self.ffn_layers[i](ffn_output)
            ffn_output = self.ffn_dropouts[i](ffn_output, training=training)

        # Final projection
        ffn_output = self.ffn_output_layer(ffn_output)
        ffn_output = self.ffn_final_dropout(ffn_output, training=training)

        # Project out1 to embed_dim for residual connection
        out1_projected = self.out1_projection(out1)

        # Final residual connection
        final_output = self.final_add([ffn_output, out1_projected])

        return final_output, attention_scores

    def get_config(self):
        config = super(GlobalAttentionBlock, self).get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'attention_type': self.attention_type,
            'attention_activation_type': self.attention_activation_type,
            'dropout_rate': self.dropout_rate,
            'use_bias': self.use_bias,
            'layer_norm_eps': self.layer_norm_eps,
            'use_head_scaling': self.use_head_scaling,
            'head_scale_init': self.head_scale_init
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)