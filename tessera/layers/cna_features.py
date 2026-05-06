"""
CNA (Copy Number Alteration) feature extraction layers.

This module contains layers for processing CNA segments:
- CNAEmbedding: Embeds encoded CNA features into learned representations
- CNASelfAttentionBlock: Self-attention mechanism for CNA segments to attend to each other
"""

import tensorflow as tf


def create_cna_inputs(use_loh=False):
    """
    Create model inputs for CNA processing.

    Args:
        use_loh: Whether to include loss of heterozygosity feature

    Returns:
        Dictionary of input layers for CNA segments
    """
    cna_bag_size = None
    # 3D shapes for consistency with variant inputs: (batch, cna_bag_size, 1)
    cna_chr = tf.keras.layers.Input(shape=(cna_bag_size, 1), dtype=tf.int32, name='cna_chr')
    cna_start = tf.keras.layers.Input(shape=(cna_bag_size, 1), dtype=tf.float32, name='cna_start')
    cna_end = tf.keras.layers.Input(shape=(cna_bag_size, 1), dtype=tf.float32, name='cna_end')
    cna_length = tf.keras.layers.Input(shape=(cna_bag_size, 1), dtype=tf.float32, name='cna_length')
    cna_segment_mean = tf.keras.layers.Input(shape=(cna_bag_size, 1), dtype=tf.float32, name='cna_segment_mean')

    inputs = {
        'cna_chr': cna_chr,
        'cna_start': cna_start,
        'cna_end': cna_end,
        'cna_length': cna_length,
        'cna_segment_mean': cna_segment_mean
    }

    # Add optional CNA features if requested
    if use_loh:
        inputs['cna_loh'] = tf.keras.layers.Input(
            shape=(cna_bag_size, 1), dtype=tf.float32, name='cna_loh'
        )

    return inputs


@tf.keras.utils.register_keras_serializable(
    package="tessera.layers.cna_features",
    name="CNAEmbedding",
)
class CNAEmbedding(tf.keras.layers.Layer):
    """
    Layer that embeds CNA segment features into a learned representation.

    Takes encoded CNA features (chromosome, normalized start/end positions, log length, optionally segment mean)
    and projects them into an embedding space suitable for attention mechanisms.

    To prevent data leakage, this layer supports creating two types of embeddings:
    - Query embeddings (use_segment_mean=False): For predicting segment_mean
    - Key/Value embeddings (use_segment_mean=True): For attending to other segments

    Args:
        chr_encoder: Dictionary containing chromosome encoding information
        embed_dim: Output embedding dimension for CNA features
        use_segment_mean: Whether to include segment_mean in the embedding (default: True)
                         Set to False for query embeddings to prevent data leakage
        use_chr_one_hot: Whether to use one-hot encoding for chromosomes (default: True)
        chr_embed_dim: Embedding dimension for chromosomes if not using one-hot (default: 16)
        layer_norm_eps: Epsilon for layer normalization (default: 1e-6)
        use_bias: Whether to use bias in dense layers (default: False)
        n_fc_layers: Number of fully connected layers for processing (default: 3)
        use_loh: Whether to include loss of heterozygosity feature (default: False)

    Input shapes:
        cna_chr: (batch_size, cna_bag_size, 1) - Chromosome IDs
        cna_start: (batch_size, cna_bag_size, 1) - Normalized start positions [0,1]
        cna_end: (batch_size, cna_bag_size, 1) - Normalized end positions [0,1]
        cna_length: (batch_size, cna_bag_size, 1) - Log-transformed segment lengths
        cna_segment_mean: (batch_size, cna_bag_size, 1) - Segment mean values (log2 ratio) [only if use_segment_mean=True]
        cna_loh: (batch_size, cna_bag_size, 1) - Loss of heterozygosity (0/1) [optional]

    Output shape:
        (batch_size, cna_bag_size, embed_dim) - Embedded CNA features

    Example:
        ```python
        # Query embedding (without segment_mean - for prediction)
        cna_query_emb = CNAEmbedding(
            chr_encoder=chr_encoder,
            embed_dim=64,
            use_segment_mean=False,
            name='cna_query_embedding'
        )

        # Key/Value embedding (with segment_mean - for attention)
        cna_kv_emb = CNAEmbedding(
            chr_encoder=chr_encoder,
            embed_dim=64,
            use_segment_mean=True,
            name='cna_kv_embedding'
        )
        ```
    """

    def __init__(self, chr_encoder, embed_dim=64, use_segment_mean=True, use_chr_one_hot=True, chr_embed_dim=16,
                 layer_norm_eps=1e-6, use_bias=False, n_fc_layers=3, use_loh=False, **kwargs):
        super(CNAEmbedding, self).__init__(**kwargs)
        self.chr_encoder = chr_encoder
        self.embed_dim = embed_dim
        self.use_segment_mean = use_segment_mean
        self.use_chr_one_hot = use_chr_one_hot
        self.chr_embed_dim = chr_embed_dim
        self.layer_norm_eps = layer_norm_eps
        self.use_bias = use_bias
        self.n_fc_layers = n_fc_layers
        self.use_loh = use_loh

        # Chromosome embedding
        if self.use_chr_one_hot:
            # Store vocab size for one-hot encoding (using tf.one_hot instead of CategoryEncoding
            # to avoid graph mode issues with symbolic tensors)
            self.chr_vocab_size = self.chr_encoder['vocab_size']
        else:
            self.chr_embedding_layer = tf.keras.layers.Embedding(
                input_dim=self.chr_encoder['vocab_size'],
                output_dim=self.chr_embed_dim,
                name='cna_chr_embedding'
            )

        # No need to expand - inputs already have shape (batch, cna_bag_size, 1)
        # Only need to squeeze chromosome for one-hot encoding

        # Concatenate all CNA features
        self.cna_concat = tf.keras.layers.Concatenate(axis=-1, name='cna_feature_concat')

        # Dense layers for feature processing (will be built dynamically)
        self.cna_dense_layers = []
        self.cna_norms = []

    def build(self, input_shape):
        super().build(input_shape)

        # Calculate input dimension after concatenation
        if self.use_chr_one_hot:
            chr_dim = self.chr_encoder['vocab_size']
        else:
            chr_dim = self.chr_embed_dim

        # chr + start + end + length [+ segment_mean if use_segment_mean=True]
        # [+ optional features]
        base_features = 3  # chr + start + end + length (chr is chr_dim, others are 1 each)
        self.input_dim = chr_dim + base_features

        if self.use_segment_mean:
            self.input_dim += 1
        if self.use_loh:
            self.input_dim += 1

        # Create dense layers to project to embed_dim
        for i in range(self.n_fc_layers):
            self.cna_dense_layers.append(
                tf.keras.layers.Dense(
                    self.embed_dim,
                    activation='gelu',
                    use_bias=self.use_bias,
                    name=f'cna_dense_{i}'
                )
            )
            self.cna_norms.append(
                tf.keras.layers.LayerNormalization(
                    epsilon=self.layer_norm_eps,
                    center=False,
                    scale=False,
                    name=f'cna_norm_{i}'
                )
            )

    def call(self, inputs):
        """
        Process CNA inputs through embedding layers.

        Args:
            inputs: Dictionary with keys 'cna_chr', 'cna_start', 'cna_end', 'cna_length', 'cna_segment_mean'
                   All inputs have shape (batch, cna_bag_size, 1)

        Returns:
            Embedded CNA features (batch_size, cna_bag_size, embed_dim)
        """
        # Inputs are 3D: (batch, cna_bag_size, 1)
        cna_chr = inputs['cna_chr']
        cna_start = inputs['cna_start']
        cna_end = inputs['cna_end']
        cna_length = inputs['cna_length']

        # Process chromosome - squeeze for one-hot encoding
        if self.use_chr_one_hot:
            # Squeeze to (batch, cna_bag_size) for one-hot encoding
            cna_chr_squeezed = tf.squeeze(cna_chr, axis=-1)
            # One-hot: (batch, cna_bag_size, vocab_size)
            chr_emb = tf.one_hot(cna_chr_squeezed, depth=self.chr_vocab_size, dtype=tf.float32)
        else:
            # Embedding layer handles the 3D input directly
            chr_emb = self.chr_embedding_layer(cna_chr)  # Shape: (batch, cna_bag_size, chr_embed_dim)

        # Genomic features already have shape (batch, cna_bag_size, 1) - no need to expand
        start_emb = cna_start  # Shape: (batch, cna_bag_size, 1)
        end_emb = cna_end      # Shape: (batch, cna_bag_size, 1)
        length_emb = cna_length  # Shape: (batch, cna_bag_size, 1)

        # Build feature list dynamically based on configuration
        feature_list = [chr_emb, start_emb, end_emb, length_emb]

        # Conditionally include segment_mean to prevent data leakage
        if self.use_segment_mean:
            cna_segment_mean = inputs['cna_segment_mean']  # Shape: (batch, cna_bag_size, 1)
            feature_list.append(cna_segment_mean)

        # Add optional genomic features if enabled
        if self.use_loh:
            cna_loh = inputs['cna_loh']
            feature_list.append(cna_loh)

        # Concatenate all selected features
        cna_features = self.cna_concat(feature_list)

        # Apply dense layers with normalization
        for i in range(self.n_fc_layers):
            cna_features = self.cna_dense_layers[i](cna_features)
            cna_features = self.cna_norms[i](cna_features)

        return cna_features

    def get_config(self):
        config = super().get_config()
        config.update({
            'chr_encoder': self.chr_encoder,
            'embed_dim': self.embed_dim,
            'use_segment_mean': self.use_segment_mean,
            'use_chr_one_hot': self.use_chr_one_hot,
            'chr_embed_dim': self.chr_embed_dim,
            'layer_norm_eps': self.layer_norm_eps,
            'use_bias': self.use_bias,
            'n_fc_layers': self.n_fc_layers,
            'use_loh': self.use_loh
        })
        return config


@tf.keras.utils.register_keras_serializable(
    package="tessera.layers.cna_features",
    name="CNAEmbeddingUnified",
)
class CNAEmbeddingUnified(tf.keras.layers.Layer):
    """
    Unified CNA embedding layer that returns both query and key/value embeddings.
    Similar to MutEmbedding which returns both ref_emb and mut_emb from a single layer,
    this ensures the embedding weights are in the model graph when either output is
    connected to predictions.

    The key difference from using two separate CNAEmbedding layers:
    - Query embedding: base features WITHOUT segment_mean (for prediction)
    - KV embedding: base features WITH segment_mean (for attention/features)
    - Each has SEPARATE dense layers (like MutEmbedding has ref_dense_layers and mut_dense_layers)
    - Since both outputs come from the same layer call, weights are always in the model graph

    Args:
        chr_encoder: Dictionary containing chromosome encoding information
        embed_dim: Output embedding dimension for CNA features
        use_chr_one_hot: Whether to use one-hot encoding for chromosomes (default: True)
        chr_embed_dim: Embedding dimension for chromosomes if not using one-hot (default: 16)
        layer_norm_eps: Epsilon for layer normalization (default: 1e-6)
        use_bias: Whether to use bias in dense layers (default: False)
        n_fc_layers: Number of fully connected layers for processing (default: 3)
        use_loh: Whether to include loss of heterozygosity feature (default: False)

    Returns:
        Dictionary with:
            'query_emb': Embedding WITHOUT segment_mean (batch, cna_bag_size, embed_dim)
            'kv_emb': Embedding WITH segment_mean (batch, cna_bag_size, embed_dim)
    """

    def __init__(self, chr_encoder, embed_dim=64, use_chr_one_hot=True, chr_embed_dim=16,
                 layer_norm_eps=1e-6, use_bias=False, n_fc_layers=3, use_loh=False, predict_cna_loh=False, **kwargs):
        super(CNAEmbeddingUnified, self).__init__(**kwargs)
        self.chr_encoder = chr_encoder
        self.embed_dim = embed_dim
        self.use_chr_one_hot = use_chr_one_hot
        self.chr_embed_dim = chr_embed_dim
        self.layer_norm_eps = layer_norm_eps
        self.use_bias = use_bias
        self.n_fc_layers = n_fc_layers
        self.use_loh = use_loh
        self.predict_cna_loh = predict_cna_loh

        # Determine whether to include LOH in query (exclude if predicting it to prevent leakage)
        self.use_loh_in_query = use_loh and not predict_cna_loh
        self.use_loh_in_kv = use_loh

        # Chromosome embedding
        if self.use_chr_one_hot:
            self.chr_vocab_size = self.chr_encoder['vocab_size']
        else:
            self.chr_embedding_layer = tf.keras.layers.Embedding(
                input_dim=self.chr_encoder['vocab_size'],
                output_dim=self.chr_embed_dim,
                name='cna_chr_embedding'
            )

        # Concatenation layers for query and kv features
        self.query_concat = tf.keras.layers.Concatenate(axis=-1, name='cna_query_feature_concat')
        self.kv_concat = tf.keras.layers.Concatenate(axis=-1, name='cna_kv_feature_concat')

        # SEPARATE dense layers for query and kv (like MutEmbedding has ref_dense_layers and mut_dense_layers)
        self.query_dense_layers = []
        self.query_norms = []
        self.kv_dense_layers = []
        self.kv_norms = []

    def build(self, input_shape):
        super().build(input_shape)

        # Create SEPARATE dense layers for query and kv
        for i in range(self.n_fc_layers):
            self.query_dense_layers.append(
                tf.keras.layers.Dense(
                    self.embed_dim,
                    activation='gelu',
                    use_bias=self.use_bias,
                    name=f'cna_query_dense_{i}'
                )
            )
            self.query_norms.append(
                tf.keras.layers.LayerNormalization(
                    epsilon=self.layer_norm_eps,
                    center=False,
                    scale=False,
                    name=f'cna_query_norm_{i}'
                )
            )
            self.kv_dense_layers.append(
                tf.keras.layers.Dense(
                    self.embed_dim,
                    activation='gelu',
                    use_bias=self.use_bias,
                    name=f'cna_kv_dense_{i}'
                )
            )
            self.kv_norms.append(
                tf.keras.layers.LayerNormalization(
                    epsilon=self.layer_norm_eps,
                    center=False,
                    scale=False,
                    name=f'cna_kv_norm_{i}'
                )
            )

    def call(self, inputs):
        """
        Process CNA inputs and return both query and kv embeddings.

        Args:
            inputs: Dictionary with CNA feature tensors

        Returns:
            Dictionary with 'query_emb' and 'kv_emb'
        """
        # Extract inputs
        cna_chr = inputs['cna_chr']
        cna_start = inputs['cna_start']
        cna_end = inputs['cna_end']
        cna_length = inputs['cna_length']
        cna_segment_mean = inputs['cna_segment_mean']

        # Process chromosome
        if self.use_chr_one_hot:
            cna_chr_squeezed = tf.squeeze(cna_chr, axis=-1)
            chr_emb = tf.one_hot(cna_chr_squeezed, depth=self.chr_vocab_size, dtype=tf.float32)
        else:
            chr_emb = self.chr_embedding_layer(cna_chr)

        # Build base feature list (shared between query and kv, without LOH)
        base_feature_list = [chr_emb, cna_start, cna_end, cna_length]

        # Create query features (base features WITHOUT segment_mean)
        # LOH is included in query ONLY if use_loh=True AND we're NOT predicting LOH
        query_feature_list = base_feature_list.copy()
        if self.use_loh_in_query:
            query_feature_list.append(inputs['cna_loh'])
        query_features = self.query_concat(query_feature_list)

        # Create kv features (base features WITH segment_mean and LOH if enabled)
        # KV always gets LOH if use_loh=True (other segments can see LOH)
        kv_feature_list = base_feature_list.copy()
        if self.use_loh_in_kv:
            kv_feature_list.append(inputs['cna_loh'])
        kv_feature_list.append(cna_segment_mean)
        kv_features = self.kv_concat(kv_feature_list)

        # Apply SEPARATE dense layers to each
        query_emb = query_features
        kv_emb = kv_features

        for i in range(self.n_fc_layers):
            query_emb = self.query_dense_layers[i](query_emb)
            query_emb = self.query_norms[i](query_emb)

            kv_emb = self.kv_dense_layers[i](kv_emb)
            kv_emb = self.kv_norms[i](kv_emb)

        return {
            'query_emb': query_emb,
            'kv_emb': kv_emb
        }

    def get_config(self):
        config = super().get_config()
        config.update({
            'chr_encoder': self.chr_encoder,
            'embed_dim': self.embed_dim,
            'use_chr_one_hot': self.use_chr_one_hot,
            'chr_embed_dim': self.chr_embed_dim,
            'layer_norm_eps': self.layer_norm_eps,
            'use_bias': self.use_bias,
            'n_fc_layers': self.n_fc_layers,
            'use_loh': self.use_loh,
            'predict_cna_loh': self.predict_cna_loh
        })
        return config


@tf.keras.utils.register_keras_serializable(
    package="tessera.layers.cna_features",
    name="CNASelfAttentionBlock",
)
class CNASelfAttentionBlock(tf.keras.layers.Layer):
    """
    Self-attention block for CNA segments to attend to each other within a sample.

    To prevent data leakage when predicting segment_mean:
    - Query embeddings do NOT include the segment's own segment_mean
    - Key/Value embeddings DO include segment_mean from other segments
    - This allows each segment to learn from others while preventing leakage

    Follows transformer architecture with pre-normalization and residual connections.

    Args:
        num_heads: Number of attention heads
        embed_dim: Embedding dimension
        ff_dim: Feed-forward network dimension
        dropout_rate: Dropout rate for regularization (default: 0.1)
        use_bias: Whether to use bias in dense layers (default: False)
        layer_norm_eps: Layer normalization epsilon (default: 1e-6)
        attention_l1_factor: L1 regularization factor for attention sparsity (default: 0.0)

    Input shapes:
        cna_query_emb: Query embeddings WITHOUT segment_mean (batch_size, cna_bag_size, embed_dim)
        cna_kv_emb: Key/Value embeddings WITH segment_mean (batch_size, cna_bag_size, embed_dim)
        attention_mask: Optional mask (batch_size, cna_bag_size, cna_bag_size) - should include diagonal mask

    Output shapes:
        final_output: (batch_size, cna_bag_size, embed_dim)
        attention_scores: (batch_size, num_heads, cna_bag_size, cna_bag_size)

    Example:
        ```python
        cna_self_attn = CNASelfAttentionBlock(
            num_heads=8,
            embed_dim=64,
            ff_dim=128,
            dropout_rate=0.1
        )
        cna_output, attn_scores = cna_self_attn(
            cna_query_emb=query_emb,
            cna_kv_emb=kv_emb,
            attention_mask=mask
        )
        ```
    """

    def __init__(self, num_heads, embed_dim, ff_dim, dropout_rate=0.1, use_bias=False,
                 layer_norm_eps=1e-6, attention_l1_factor=0.0, **kwargs):
        super(CNASelfAttentionBlock, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.layer_norm_eps = layer_norm_eps
        self.attention_l1_factor = attention_l1_factor

        # Pre-norm layer normalization
        self.pre_attn_norm = tf.keras.layers.LayerNormalization(
            epsilon=layer_norm_eps,
            center=False,
            scale=False,
            name=f'{self.name}_pre_attn_norm'
        )

        # Multi-head self-attention
        from tessera.layers.attention import CustomMultiHeadAttention
        self.attention = CustomMultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout_rate,
            use_bias=False,
            attention_l1_factor=attention_l1_factor,
            name=f'{self.name}_self_attention'
        )

        # Dropout after attention
        self.attention_dropout = tf.keras.layers.Dropout(dropout_rate, name=f'{self.name}_attention_dropout')

        # Concatenation layer (matching GlobalAttentionBlock, LocalAttentionBlock, CrossModalAttentionBlock)
        self.concat = tf.keras.layers.Concatenate(axis=-1, name=f'{self.name}_context_attention_concat')

        # Combined projection (matching other attention blocks)
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

        # Feed-forward network
        self.ffn_layers = []
        self.ffn_dropouts = []
        for i in range(3):
            self.ffn_layers.append(
                tf.keras.layers.Dense(
                    ff_dim,
                    activation='gelu',
                    use_bias=use_bias,
                    kernel_initializer='glorot_uniform',
                    name=f'{self.name}_ffn_layer_{i+1}'
                )
            )
            self.ffn_dropouts.append(
                tf.keras.layers.Dropout(dropout_rate, name=f'{self.name}_ffn_dropout_{i+1}')
            )

        # Final projection
        self.ffn_output_layer = tf.keras.layers.Dense(
            embed_dim,
            use_bias=use_bias,
            kernel_initializer='glorot_uniform',
            name=f'{self.name}_ffn_output'
        )
        self.ffn_final_dropout = tf.keras.layers.Dropout(dropout_rate, name=f'{self.name}_ffn_final_dropout')

        # Output projection for residual connection (matching other attention blocks)
        self.out1_projection = tf.keras.layers.Dense(
            embed_dim,
            use_bias=use_bias,
            kernel_initializer='glorot_uniform',
            name=f'{self.name}_out1_projection'
        )

        # Final residual connection
        self.final_add = tf.keras.layers.Add(name=f'{self.name}_final_residual')

    def call(self, cna_query_emb, cna_kv_emb=None, attention_mask=None, training=None):
        """
        Forward pass through the CNA self-attention block.

        Args:
            cna_query_emb: Query embeddings WITHOUT segment_mean (batch_size, cna_bag_size, embed_dim)
            cna_kv_emb: Key/Value embeddings WITH segment_mean (batch_size, cna_bag_size, embed_dim)
                       If None, uses cna_query_emb for all (backward compatibility)
            attention_mask: Optional boolean mask (batch_size, cna_bag_size, cna_bag_size)
            training: Whether in training mode

        Returns:
            final_output: Processed CNA features (batch_size, cna_bag_size, embed_dim)
            attention_scores: Attention scores for visualization
        """
        # Backward compatibility: if no kv_emb provided, use query_emb for all
        if cna_kv_emb is None:
            cna_kv_emb = cna_query_emb

        # Pre-norm before attention
        cna_query_norm = self.pre_attn_norm(cna_query_emb)
        cna_kv_norm = self.pre_attn_norm(cna_kv_emb)

        # Self-attention: queries WITHOUT segment_mean attend to keys/values WITH segment_mean
        # This prevents each segment from seeing its own segment_mean
        attention_output, attention_scores = self.attention(
            cna_query_norm, cna_kv_norm, cna_kv_norm,  # Q uses query_emb, K/V use kv_emb
            return_attention_scores=True,
            attention_mask=attention_mask,
            training=training
        )

        # Dropout after attention
        attention_output = self.attention_dropout(attention_output, training=training)

        # Concatenate query with attention output (matching other attention blocks)
        combined = self.concat([cna_query_norm, attention_output])

        # Project concatenated features
        out1 = self.combined_proj(combined)

        # Pre-norm for feed-forward
        ffn_input = self.pre_ffn_norm(out1)

        # Feed-forward network
        ffn_output = ffn_input
        for i in range(3):
            ffn_output = self.ffn_layers[i](ffn_output)
            ffn_output = self.ffn_dropouts[i](ffn_output, training=training)

        # Final projection
        ffn_output = self.ffn_output_layer(ffn_output)
        ffn_output = self.ffn_final_dropout(ffn_output, training=training)

        # Project out1 to embed_dim for residual connection (matching other attention blocks)
        out1_projected = self.out1_projection(out1)

        # Final residual connection
        final_output = self.final_add([ffn_output, out1_projected])

        return final_output, attention_scores

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_heads': self.num_heads,
            'embed_dim': self.embed_dim,
            'ff_dim': self.ff_dim,
            'dropout_rate': self.dropout_rate,
            'use_bias': self.use_bias,
            'layer_norm_eps': self.layer_norm_eps,
            'attention_l1_factor': self.attention_l1_factor
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
