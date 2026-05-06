"""
Cross-modal attention layers for multi-modal learning.

This module contains layers for cross-modal re-contextualization between different data modalities
(e.g., SNV mutations and CNA segments), allowing each modality to attend to and be informed by the other.
"""

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(
    package="tessera.layers.cross_modal",
    name="CrossModalAttentionBlock",
)
class CrossModalAttentionBlock(tf.keras.layers.Layer):
    """
    Cross-modal attention block for re-contextualizing one modality using another.

    Uses concat-based architecture matching LocalAttentionBlock and GlobalAttentionBlock.
    Enables one modality (query) to attend to another modality (key/value), allowing
    information flow between different data types (e.g., SNV mutations and CNA segments).

    Architecture:
        query → LayerNorm → query_proj(embed_dim) →
        key_value → LayerNorm → kv_proj(embed_dim) →
        CrossAttention(query_proj, kv_proj) → Dropout →
        Concat([query_norm, attention_output]) →
        combined_proj(ff_dim, gelu) → LayerNorm →
        3-layer FFN(ff_dim→ff_dim→ff_dim) →
        output_proj(output_dim) → Add(out1_projected) → output

    Design principles:
    - Input projections ensure consistent attention dimension regardless of input sizes
    - Concatenation preserves both original and attended representations
    - Pre-normalization for better gradient flow
    - 3-layer FFN for complex pattern capture (matches Local/Global blocks)
    - Residual connection for stable training

    Args:
        num_heads: Number of attention heads
        embed_dim: Internal attention projection dimension (key_dim = embed_dim // num_heads)
        ff_dim: Feed-forward network intermediate dimension (typically 2× embed_dim)
        output_dim: Final output dimension (must match query dimension for residual)
        attention_activation_type: Activation type for attention gates (currently unused, for API consistency)
        dropout_rate: Dropout rate for regularization
        use_bias: Whether to use bias in dense layers
        layer_norm_eps: Layer normalization epsilon
        use_head_scaling: Whether to use learnable head scaling (currently unused, for API consistency)
        attention_l1_factor: L1 regularization factor for attention sparsity
        head_scale_init: Initial value for head scales (currently unused, for API consistency)

    Input shapes:
        query_emb: Query embeddings (batch_size, query_seq_len, query_dim)
        key_value_emb: Key/Value embeddings (batch_size, kv_seq_len, kv_dim)
        attention_mask: Optional mask (batch_size, query_seq_len, kv_seq_len)

    Output shapes:
        final_output: (batch_size, query_seq_len, output_dim)
        attention_scores: (batch_size, num_heads, query_seq_len, kv_seq_len)

    Example:
        ```python
        # SNV attending to CNA (mutations informed by copy number context)
        snv_to_cna_attn = CrossModalAttentionBlock(
            num_heads=8,
            embed_dim=256,
            ff_dim=512,
            output_dim=256,  # Match SNV embedding dimension
            dropout_rate=0.1
        )
        snv_recontextualized, attn_scores = snv_to_cna_attn(
            query_emb=snv_embeddings,
            key_value_emb=cna_embeddings,
            attention_mask=mask
        )

        # CNA attending to SNV (copy numbers informed by mutation context)
        cna_to_snv_attn = CrossModalAttentionBlock(
            num_heads=8,
            embed_dim=64,
            ff_dim=128,
            output_dim=64,  # Match CNA embedding dimension
            dropout_rate=0.1
        )
        cna_recontextualized, attn_scores = cna_to_snv_attn(
            query_emb=cna_embeddings,
            key_value_emb=snv_embeddings,
            attention_mask=mask
        )
        ```
    """

    def __init__(
        self,
        num_heads: int,
        embed_dim: int,
        ff_dim: int,
        output_dim: int = None,
        attention_activation_type: str = 'sigmoid',
        dropout_rate: float = 0.1,
        use_bias: bool = False,
        layer_norm_eps: float = 1e-6,
        use_head_scaling: bool = False,
        attention_l1_factor: float = 0.0,
        head_scale_init: float = 1.0,
        **kwargs
    ):
        super(CrossModalAttentionBlock, self).__init__(**kwargs)

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        # Default output_dim to embed_dim for backward compatibility with saved models
        self.output_dim = output_dim if output_dim is not None else embed_dim
        self.attention_activation_type = attention_activation_type
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.layer_norm_eps = layer_norm_eps
        self.use_head_scaling = use_head_scaling
        self.attention_l1_factor = attention_l1_factor
        self.head_scale_init = head_scale_init

        # Pre-norm layer normalization for both inputs
        self.query_norm = tf.keras.layers.LayerNormalization(
            epsilon=layer_norm_eps,
            center=False,
            scale=False,
            name=f'{self.name}_query_norm'
        )

        self.key_value_norm = tf.keras.layers.LayerNormalization(
            epsilon=layer_norm_eps,
            center=False,
            scale=False,
            name=f'{self.name}_key_value_norm'
        )

        # Projection layers to match embed_dim (matches LocalAttentionBlock pattern)
        self.query_proj = tf.keras.layers.Dense(
            embed_dim,
            use_bias=use_bias,
            kernel_initializer='glorot_uniform',
            name=f'{self.name}_query_projection'
        )

        self.key_value_proj = tf.keras.layers.Dense(
            embed_dim,
            use_bias=use_bias,
            kernel_initializer='glorot_uniform',
            name=f'{self.name}_key_value_projection'
        )

        # Cross-attention mechanism (query from one modality, key/value from another)
        from tessera.layers.attention import CustomMultiHeadAttention
        self.cross_attention = CustomMultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout_rate,
            use_bias=False,
            attention_l1_factor=attention_l1_factor,
            name=f'{self.name}_cross_attention'
        )

        # Dropout after attention
        self.attention_dropout = tf.keras.layers.Dropout(
            dropout_rate,
            name=f'{self.name}_attention_dropout'
        )

        # Concatenation layer (matches Local/Global blocks)
        self.concat = tf.keras.layers.Concatenate(
            axis=-1,
            name=f'{self.name}_context_attention_concat'
        )

        # Combined projection after concatenation
        self.combined_proj = tf.keras.layers.Dense(
            ff_dim,
            activation='gelu',
            use_bias=use_bias,
            kernel_initializer='glorot_uniform',
            name=f'{self.name}_combined_projection'
        )

        # Pre-FFN normalization
        self.pre_ffn_norm = tf.keras.layers.LayerNormalization(
            epsilon=layer_norm_eps,
            center=False,
            scale=False,
            name=f'{self.name}_pre_ffn_norm'
        )

        # Feed forward layers (3-layer FFN matches Local/Global blocks)
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
            self.ffn_dropouts.append(tf.keras.layers.Dropout(
                dropout_rate,
                name=f'{self.name}_ffn_dropout_{i+1}'
            ))

        # Final projection to output_dim
        self.ffn_output_layer = tf.keras.layers.Dense(
            self.output_dim,
            use_bias=use_bias,
            kernel_initializer='glorot_uniform',
            name=f'{self.name}_ffn_output'
        )
        self.ffn_final_dropout = tf.keras.layers.Dropout(
            dropout_rate,
            name=f'{self.name}_ffn_final_dropout'
        )

        # Output projection for residual connection
        self.out1_projection = tf.keras.layers.Dense(
            self.output_dim,
            use_bias=use_bias,
            kernel_initializer='glorot_uniform',
            name=f'{self.name}_out1_projection'
        )

        # Final residual connection
        self.final_add = tf.keras.layers.Add(name=f'{self.name}_final_residual')

    def call(self, query_emb, key_value_emb, attention_mask=None, training=None):
        """
        Forward pass through the cross-modal attention block.

        Architecture (matches Local/Global blocks):
            query → norm → proj → attention → dropout →
            concat([query_norm, attention_output]) →
            combined_proj → norm → 3-layer FFN →
            output_proj → add(out1_projected) → output

        Args:
            query_emb: Query embeddings from one modality (batch_size, query_seq_len, query_dim)
            key_value_emb: Key/Value embeddings from another modality (batch_size, kv_seq_len, kv_dim)
            attention_mask: Optional boolean mask (batch_size, query_seq_len, kv_seq_len)
            training: Whether in training mode

        Returns:
            final_output: Re-contextualized query embeddings (batch_size, query_seq_len, output_dim)
            attention_scores: Cross-modal attention scores (batch_size, num_heads, query_seq_len, kv_seq_len)
        """
        # Pre-norm layer normalization
        query_norm = self.query_norm(query_emb)
        key_value_norm = self.key_value_norm(key_value_emb)

        # Project both to embed_dim (matches LocalAttentionBlock pattern)
        query_proj = self.query_proj(query_norm)
        kv_proj = self.key_value_proj(key_value_norm)

        # Cross-attention: query attends to key/value from different modality
        attention_output, attention_scores = self.cross_attention(
            query_proj, kv_proj, kv_proj,
            return_attention_scores=True,
            attention_mask=attention_mask,
            training=training
        )

        # Add dropout after attention
        attention_output = self.attention_dropout(attention_output, training=training)

        # Concatenate query_norm with attention output (matches Local/Global blocks)
        combined = self.concat([query_norm, attention_output])

        # Project concatenated features
        out1 = self.combined_proj(combined)

        # Pre-norm for feed-forward network
        ffn_input = self.pre_ffn_norm(out1)

        # Feed forward network (3-layer FFN)
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
        config = super(CrossModalAttentionBlock, self).get_config()
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
            'attention_l1_factor': self.attention_l1_factor,
            'head_scale_init': self.head_scale_init
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
