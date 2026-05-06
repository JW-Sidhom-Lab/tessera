"""Variant feature extraction and embedding layers.

End-to-end variant featurisation: chromosome and position embeddings (PosEmbedding),
reference and alternate allele embedding (MutEmbedding), genomic-context concatenation
(GenomicConcatLayer), local-to-global projection layers, and the composite
VariantFeaturizationLayer1 that produces the per-variant token representations
consumed by TESSERA's attention stack. Also exposes create_variant_inputs, the
helper used by model.py to declare Keras Input tensors of the right shapes.
"""
import tensorflow as tf
from tessera.layers.attention import LocalAttentionBlock
from tessera.layers.utils import FlattenLastTwoDims, Conv1DFor4D

def create_variant_inputs(self, use_vaf=False):
    """
    Create model inputs for variant processing.

    Args:
        self: Model instance with ref_len, alt_len, context_len attributes
        use_vaf: Whether to include VAF (Variant Allele Frequency) input

    Returns:
        Dictionary of input layers
    """
    bag_size = None
    ref = tf.keras.layers.Input(shape=(bag_size, self.ref_len), dtype=tf.int32, name='ref')
    alt = tf.keras.layers.Input(shape=(bag_size, self.alt_len), dtype=tf.int32, name='alt')
    context_5p = tf.keras.layers.Input(shape=(bag_size, self.context_len), dtype=tf.int32, name='context_5p')
    context_3p = tf.keras.layers.Input(shape=(bag_size, self.context_len), dtype=tf.int32, name='context_3p')
    chr_input = tf.keras.layers.Input(shape=(bag_size, 1), dtype=tf.int32, name='chr')
    pos_input = tf.keras.layers.Input(shape=(bag_size, 1), dtype=tf.float32, name='pos')
    inputs = {'ref': ref, 'alt': alt, 'context_5p': context_5p, 'context_3p': context_3p, 'chr': chr_input,
              'pos': pos_input}

    # Add VAF input if use_vaf is True
    if use_vaf:
        vaf_input = tf.keras.layers.Input(shape=(bag_size, 1), dtype=tf.float32, name='vaf')
        inputs['vaf'] = vaf_input

    return inputs

@tf.keras.utils.register_keras_serializable(
    package="tessera.layers.variant_features",
    name="PosEmbedding",
)
class PosEmbedding(tf.keras.layers.Layer):
    """
    Layer that handles chromosome and position embedding for genomic data.
    """
    
    def __init__(self, chr_encoder, use_chr_one_hot=True, chr_embed_dim=16, pos_embed_dim=16, 
                 layer_norm_eps=1e-6, use_bias=False, use_pos_mlp=False, **kwargs):
        super(PosEmbedding, self).__init__(**kwargs)
        self.chr_encoder = chr_encoder
        self.use_chr_one_hot = use_chr_one_hot
        self.chr_embed_dim = chr_embed_dim
        self.pos_embed_dim = pos_embed_dim
        self.layer_norm_eps = layer_norm_eps
        self.use_bias = use_bias
        self.use_pos_mlp = use_pos_mlp
        
        if self.use_chr_one_hot:
            self.chr_encoding = tf.keras.layers.CategoryEncoding(
                num_tokens=self.chr_encoder['vocab_size'], 
                output_mode="one_hot", 
                name='chr_embedding'
            )
            self.chr_expand = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-2))
        else:
            self.chr_embedding_layer = tf.keras.layers.Embedding(
                input_dim=self.chr_encoder['vocab_size'],
                output_dim=self.chr_embed_dim,
                name='chr_embedding'
            )
        
        self.pos_expand = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))
        
        # Optional position MLP layers
        if self.use_pos_mlp:
            self.pos_dense_layers = []
            for i in range(3):
                self.pos_dense_layers.append(
                    tf.keras.layers.Dense(self.pos_embed_dim, activation='gelu', use_bias=self.use_bias, name=f'pos_embedding_{i}')
                )
            self.pos_norm = tf.keras.layers.LayerNormalization(epsilon=self.layer_norm_eps, name='pos_norm')
        else:
            self.pos_dense_layers = []
            self.pos_norm = None
            
        self.genomic_concat = tf.keras.layers.Concatenate(axis=-1, name='genomic_embedding')
        self.genomic_norm = tf.keras.layers.LayerNormalization(epsilon=self.layer_norm_eps, name='genomic_norm')
        
        # Dense layers will be created in build() after genomic_dim is calculated
        self.genomic_dense_layers = []
    
    def build(self, input_shape):
        super().build(input_shape)
        # Calculate genomic dimension after knowing the shapes
        if self.use_chr_one_hot:
            chr_dim = self.chr_encoder['vocab_size']
        else:
            chr_dim = self.chr_embed_dim
        pos_dim = 1  # Position is just expanded by 1 dimension
        self.genomic_dim = chr_dim + pos_dim
        
        # Create dense layers with correct dimension now that genomic_dim is known
        for i in range(3):
            self.genomic_dense_layers.append(
                tf.keras.layers.Dense(self.genomic_dim, activation='gelu', use_bias=self.use_bias, name=f'genomic_dense_{i}')
            )
    
    def call(self, inputs):
        """
        Process chromosome and position inputs.
        
        Args:
            inputs: Dictionary with keys 'chr' and 'pos'
        
        Returns:
            Processed genomic embedding
        """
        chr_input = inputs['chr']
        pos_input = inputs['pos']
        
        # Process chromosome
        if self.use_chr_one_hot:
            chr_emb = self.chr_encoding(chr_input)
            chr_emb = self.chr_expand(chr_emb)  # Shape: (batch, variants, 1, vocab_size)
        else:
            chr_emb = self.chr_embedding_layer(chr_input)  # Shape: (batch, variants, 1, chr_embed_dim)
        
        # Process position
        pos_emb = self.pos_expand(pos_input)  # Shape: (batch, variants, 1, 1)
        
        # Apply position MLP if enabled
        if self.use_pos_mlp:
            for layer in self.pos_dense_layers:
                pos_emb = layer(pos_emb)
            pos_emb = self.pos_norm(pos_emb)
        
        # Combine chromosome and position embeddings
        genomic_emb = self.genomic_concat([chr_emb, pos_emb])
        
        # Apply dense layers
        for layer in self.genomic_dense_layers:
            genomic_emb = layer(genomic_emb)
        genomic_emb = self.genomic_norm(genomic_emb)
        
        return genomic_emb
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'chr_encoder': self.chr_encoder,
            'use_chr_one_hot': self.use_chr_one_hot,
            'chr_embed_dim': self.chr_embed_dim,
            'pos_embed_dim': self.pos_embed_dim,
            'layer_norm_eps': self.layer_norm_eps,
            'use_bias': self.use_bias,
            'use_pos_mlp': self.use_pos_mlp
        })
        return config

@tf.keras.utils.register_keras_serializable(
    package="tessera.layers.variant_features",
    name="MutEmbedding",
)
class MutEmbedding(tf.keras.layers.Layer):
    """
    Layer that processes ref and alt embeddings into mutation embeddings.
    Handles normalization, concatenation, and fully connected processing.
    """
    
    def __init__(self, ref_alt_dim, n_fc_layers=3, layer_norm_eps=1e-6, use_bias=False, **kwargs):
        super(MutEmbedding, self).__init__(**kwargs)
        self.ref_alt_dim = ref_alt_dim
        self.n_fc_layers = n_fc_layers
        self.layer_norm_eps = layer_norm_eps
        self.use_bias = use_bias
        
        # Initial normalization layers
        self.ref_norm_initial = tf.keras.layers.LayerNormalization(
            epsilon=self.layer_norm_eps, center=False, scale=False, name='ref_norm_initial'
        )
        self.alt_norm_initial = tf.keras.layers.LayerNormalization(
            epsilon=self.layer_norm_eps, center=False, scale=False, name='alt_norm_initial'
        )
        
        # Concatenation layer
        self.mut_concat = tf.keras.layers.Concatenate(axis=-1, name='mut_concat')
        
        # Dense layers for ref and mut processing
        self.ref_dense_layers = []
        self.mut_dense_layers = []
        for i in range(self.n_fc_layers):
            self.ref_dense_layers.append(
                tf.keras.layers.Dense(self.ref_alt_dim, activation='gelu', use_bias=self.use_bias, name=f'ref_dense_{i}')
            )
            self.mut_dense_layers.append(
                tf.keras.layers.Dense(self.ref_alt_dim, activation='gelu', use_bias=self.use_bias, name=f'mut_dense_{i}')
            )
        
        # Final normalization layers
        self.ref_norm_final = tf.keras.layers.LayerNormalization(
            epsilon=self.layer_norm_eps, center=False, scale=False, name='ref_norm_final'
        )
        self.mut_norm_final = tf.keras.layers.LayerNormalization(
            epsilon=self.layer_norm_eps, center=False, scale=False, name='mut_norm_final'
        )
    
    def call(self, inputs):
        """
        Process ref and alt embeddings into mutation embeddings.
        
        Args:
            inputs: Dictionary with keys 'ref_emb' and 'alt_emb'
        
        Returns:
            Dictionary with processed 'ref_emb' and 'mut_emb'
        """
        ref_emb = inputs['ref_emb']
        alt_emb = inputs['alt_emb']
        
        # Initial normalization
        ref_emb = self.ref_norm_initial(ref_emb)
        alt_emb = self.alt_norm_initial(alt_emb)
        
        # Create mutation embedding by concatenating ref and alt
        mut_emb = self.mut_concat([ref_emb, alt_emb])
        # mut_emb = alt_emb
        
        # Apply fully connected layers
        for i in range(self.n_fc_layers):
            ref_emb = self.ref_dense_layers[i](ref_emb)
            mut_emb = self.mut_dense_layers[i](mut_emb)
        
        # Final normalization
        ref_emb = self.ref_norm_final(ref_emb)
        mut_emb = self.mut_norm_final(mut_emb)
        
        return {
            'ref_emb': ref_emb,
            'mut_emb': mut_emb
        }
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'ref_alt_dim': self.ref_alt_dim,
            'n_fc_layers': self.n_fc_layers,
            'layer_norm_eps': self.layer_norm_eps,
            'use_bias': self.use_bias
        })
        return config

@tf.keras.utils.register_keras_serializable(
    package="tessera.layers.variant_features",
    name="GenomicConcatLayer",
)
class GenomicConcatLayer(tf.keras.layers.Layer):
    """
    Layer that concatenates genomic embeddings with ref/mut embeddings
    and applies final fully connected processing.
    """
    
    def __init__(self, ref_alt_dim, n_fc_layers=3, layer_norm_eps=1e-6, use_bias=False, **kwargs):
        super(GenomicConcatLayer, self).__init__(**kwargs)
        self.ref_alt_dim = ref_alt_dim
        self.n_fc_layers = n_fc_layers
        self.layer_norm_eps = layer_norm_eps
        self.use_bias = use_bias
        
        # Concatenation layers
        self.ref_concat = tf.keras.layers.Concatenate(axis=-1, name='ref_genomic_concat')
        self.mut_concat = tf.keras.layers.Concatenate(axis=-1, name='mut_genomic_concat')
        
        # Dense layers for final processing
        self.ref_dense_layers = []
        self.mut_dense_layers = []
        for i in range(self.n_fc_layers):
            self.ref_dense_layers.append(
                tf.keras.layers.Dense(self.ref_alt_dim, activation='gelu', use_bias=self.use_bias, name=f'ref_final_dense_{i}')
            )
            self.mut_dense_layers.append(
                tf.keras.layers.Dense(self.ref_alt_dim, activation='gelu', use_bias=self.use_bias, name=f'mut_final_dense_{i}')
            )
        
        # Final normalization layers
        self.ref_norm_final = tf.keras.layers.LayerNormalization(
            epsilon=self.layer_norm_eps, center=False, scale=False, name='ref_final_norm'
        )
        self.mut_norm_final = tf.keras.layers.LayerNormalization(
            epsilon=self.layer_norm_eps, center=False, scale=False, name='mut_final_norm'
        )
    
    def call(self, inputs):
        """
        Concatenate genomic embeddings with ref/mut embeddings and apply final processing.
        
        Args:
            inputs: Dictionary with keys 'ref_emb', 'mut_emb', 'genomic_emb'
        
        Returns:
            Dictionary with processed 'ref_emb' and 'mut_emb'
        """
        ref_emb = inputs['ref_emb']
        mut_emb = inputs['mut_emb']
        genomic_emb = inputs['genomic_emb']
        
        # Concatenate genomic features
        # ref_emb = self.ref_concat([ref_emb, genomic_emb])
        ref_emb = genomic_emb
        mut_emb = self.mut_concat([mut_emb, genomic_emb])

        # Apply final dense layers
        for i in range(self.n_fc_layers):
            ref_emb = self.ref_dense_layers[i](ref_emb)
            mut_emb = self.mut_dense_layers[i](mut_emb)
        
        # Final normalization
        ref_emb = self.ref_norm_final(ref_emb)
        mut_emb = self.mut_norm_final(mut_emb)
        
        return {
            'ref_emb': ref_emb,
            'mut_emb': mut_emb
        }
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'ref_alt_dim': self.ref_alt_dim,
            'n_fc_layers': self.n_fc_layers,
            'layer_norm_eps': self.layer_norm_eps,
            'use_bias': self.use_bias
        })
        return config

@tf.keras.utils.register_keras_serializable(
    package="tessera.layers.variant_features",
    name="SeqEmbedding",
)
class SeqEmbedding(tf.keras.layers.Layer):
    """
    Layer that handles sequence embedding for genomic data.
    Supports both one-hot encoding and learned embeddings.
    """
    
    def __init__(self, token_map, nuc_embed_dim=12, use_one_hot=True, **kwargs):
        super(SeqEmbedding, self).__init__(**kwargs)
        self.token_map = token_map
        self.nuc_embed_dim = nuc_embed_dim
        self.use_one_hot = use_one_hot
        
        if self.use_one_hot:
            self.ref_encoding = tf.keras.layers.CategoryEncoding(
                num_tokens=len(self.token_map), output_mode="one_hot", name='ref_embedding'
            )
            self.alt_encoding = tf.keras.layers.CategoryEncoding(
                num_tokens=len(self.token_map), output_mode="one_hot", name='alt_embedding'
            )
            self.context_5p_encoding = tf.keras.layers.CategoryEncoding(
                num_tokens=len(self.token_map), output_mode="one_hot", name='context_5p_embedding'
            )
            self.context_3p_encoding = tf.keras.layers.CategoryEncoding(
                num_tokens=len(self.token_map), output_mode="one_hot", name='context_3p_embedding'
            )
        else:
            self.embedding_layer = tf.keras.layers.Embedding(len(self.token_map), self.nuc_embed_dim)
            self.ref_activation = tf.keras.layers.Activation('linear', name='ref_embedding')
            self.alt_activation = tf.keras.layers.Activation('linear', name='alt_embedding')
            self.context_5p_activation = tf.keras.layers.Activation('linear', name='context_5p_embedding')
            self.context_3p_activation = tf.keras.layers.Activation('linear', name='context_3p_embedding')
    
    def call(self, inputs):
        """
        Process sequence inputs through embedding layers.
        
        Args:
            inputs: Dictionary with keys 'ref', 'alt', 'context_5p', 'context_3p'
        
        Returns:
            Dictionary with embedded sequences
        """
        ref = inputs['ref']
        alt = inputs['alt']
        context_5p = inputs['context_5p']
        context_3p = inputs['context_3p']
        
        if self.use_one_hot:
            ref_emb = self.ref_encoding(ref)
            alt_emb = self.alt_encoding(alt)
            context_5p_emb = self.context_5p_encoding(context_5p)
            context_3p_emb = self.context_3p_encoding(context_3p)
            
            # Ensure consistent dimensionality - add sequence dimension if missing
            if len(ref_emb.shape) == 3:  # (batch, seq_len, features) - missing sequence dimension
                ref_emb = tf.keras.layers.Reshape((-1, ref.shape[-1], ref_emb.shape[-1]))(ref_emb)
            if len(alt_emb.shape) == 3:  # (batch, seq_len, features) - missing sequence dimension  
                alt_emb = tf.keras.layers.Reshape((-1, alt.shape[-1], alt_emb.shape[-1]))(alt_emb)
            if len(context_5p_emb.shape) == 3:  # (batch, seq_len, features) - missing sequence dimension
                context_5p_emb = tf.keras.layers.Reshape((-1, context_5p.shape[-1], context_5p_emb.shape[-1]))(context_5p_emb)
            if len(context_3p_emb.shape) == 3:  # (batch, seq_len, features) - missing sequence dimension
                context_3p_emb = tf.keras.layers.Reshape((-1, context_3p.shape[-1], context_3p_emb.shape[-1]))(context_3p_emb)
        else:
            ref_emb = self.embedding_layer(ref)
            alt_emb = self.embedding_layer(alt)
            context_5p_emb = self.embedding_layer(context_5p)
            context_3p_emb = self.embedding_layer(context_3p)
            
            # Add named Activation layers for gradient attribution
            ref_emb = self.ref_activation(ref_emb)
            alt_emb = self.alt_activation(alt_emb)
            context_5p_emb = self.context_5p_activation(context_5p_emb)
            context_3p_emb = self.context_3p_activation(context_3p_emb)
        
        return {
            'ref_emb': ref_emb,
            'alt_emb': alt_emb,
            'context_5p_emb': context_5p_emb,
            'context_3p_emb': context_3p_emb
        }
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'token_map': self.token_map,
            'nuc_embed_dim': self.nuc_embed_dim,
            'use_one_hot': self.use_one_hot
        })
        return config

@tf.keras.utils.register_keras_serializable(
    package="tessera.layers.variant_features",
    name="LocalToGlobalProjection",
)
class LocalToGlobalProjection(tf.keras.layers.Layer):
    """
    Layer that projects features from local embedding dimension to global embedding dimension.
    Used between local and global attention blocks to ensure dimension compatibility.
    """
    
    def __init__(self, local_embed_dim, global_embed_dim, layer_norm_eps=1e-6, use_bias=False, **kwargs):
        super(LocalToGlobalProjection, self).__init__(**kwargs)
        self.local_embed_dim = local_embed_dim
        self.global_embed_dim = global_embed_dim
        self.layer_norm_eps = layer_norm_eps
        self.use_bias = use_bias
        
        # Projection layer from local to global dimensions
        self.projection_dense = tf.keras.layers.Dense(
            self.global_embed_dim, 
            activation='gelu', 
            use_bias=self.use_bias,
            name=f'{self.name}_projection'
        )
        
        # Layer normalization for stability
        self.layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=self.layer_norm_eps, 
            center=False, 
            scale=False,
            name=f'{self.name}_norm'
        )
    
    def call(self, inputs):
        """
        Project input from local embedding dimension to global embedding dimension.
        
        Args:
            inputs: Tensor with shape [..., local_embed_dim]
        
        Returns:
            Tensor with shape [..., global_embed_dim]
        """
        # Apply projection
        x = self.projection_dense(inputs)
        
        # Apply layer normalization
        x = self.layer_norm(x)
        
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'local_embed_dim': self.local_embed_dim,
            'global_embed_dim': self.global_embed_dim,
            'layer_norm_eps': self.layer_norm_eps,
            'use_bias': self.use_bias
        })
        return config

@tf.keras.utils.register_keras_serializable(
    package="tessera.layers.variant_features",
    name="PostGlobalAttentionLayer",
)
class PostGlobalAttentionLayer(tf.keras.layers.Layer):
    """
    Shared post-global attention processing layer for both real and WT variants.
    Performs layer normalization, 3 fully connected layers, and final layer normalization.
    """
    def __init__(self, ff_dim, layer_norm_eps=1e-6, **kwargs):
        super(PostGlobalAttentionLayer, self).__init__(**kwargs)
        self.ff_dim = ff_dim
        self.layer_norm_eps = layer_norm_eps
        
        # Create layers
        self.layer_norm_1 = tf.keras.layers.LayerNormalization(
            epsilon=layer_norm_eps, center=False, scale=False, name=f'{self.name}_ln1'
        )
        self.dense_layers = []
        for i in range(3):
            self.dense_layers.append(
                tf.keras.layers.Dense(ff_dim, activation='gelu', name=f'{self.name}_dense_{i}')
            )
        self.layer_norm_2 = tf.keras.layers.LayerNormalization(
            epsilon=layer_norm_eps, center=False, scale=False, name=f'{self.name}_ln2'
        )
    
    def call(self, ref_context):
        """
        Process ref_context through shared post-global attention layers.
        
        Args:
            ref_context: Context tensor from global attention
            
        Returns:
            Processed context tensor
        """
        # Initial layer normalization
        x = self.layer_norm_1(ref_context)
        
        # 3 fully connected layers
        for dense_layer in self.dense_layers:
            x = dense_layer(x)
        
        # Final layer normalization
        x = self.layer_norm_2(x)
        
        return x
    
    def get_config(self):
        config = super(PostGlobalAttentionLayer, self).get_config()
        config.update({
            "ff_dim": self.ff_dim,
            "layer_norm_eps": self.layer_norm_eps
        })
        return config

@tf.keras.utils.register_keras_serializable(
    package="tessera.layers.variant_features",
    name="VariantFeaturizationLayer1",
)
class VariantFeaturizationLayer1(tf.keras.layers.Layer):
    """
    Layer that handles variant featurization from embeddings through local attention blocks.
    This part is identical for both real and WT variants.
    """
    
    def __init__(self, model_instance, attention_activation_type='softmax',
                 layer_norm_eps=1e-6, padding='same', dropout_rate=0.0, use_bias=False, pos_mlp_dim=None, **kwargs):
        super(VariantFeaturizationLayer1, self).__init__(**kwargs)
        self.model_instance = model_instance
        self.attention_activation_type = attention_activation_type
        self.layer_norm_eps = layer_norm_eps
        self.padding = padding
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.pos_mlp_dim = pos_mlp_dim

        # Store use_vaf from model_instance
        self.use_vaf = hasattr(model_instance, 'use_vaf') and model_instance.use_vaf

        # Initialize all sub-layers
        self._build_layers()
    
    def _build_layers(self):
        """Initialize all the sub-layers used in the featurization pipeline."""
        # Sequence embedding layer
        self.seq_embedding_layer = SeqEmbedding(
            token_map=self.model_instance.token_map, 
            nuc_embed_dim=self.model_instance.nuc_embed_dim, 
            use_one_hot=True,
            name='seq_embedding'
        )
        
        # Position embedding layer
        self.pos_embedding_layer = PosEmbedding(
            chr_encoder=self.model_instance.chr_encoder,
            use_chr_one_hot=True,
            chr_embed_dim=16,
            pos_embed_dim=16,
            layer_norm_eps=self.layer_norm_eps,
            use_bias=self.use_bias,
            use_pos_mlp=False,
            name='pos_embedding'
        )
        
        # Mutation embedding layer
        self.mut_embedding_layer = MutEmbedding(
            ref_alt_dim=self.model_instance.ref_alt_dim,
            n_fc_layers=3,
            layer_norm_eps=self.layer_norm_eps,
            use_bias=self.use_bias,
            name='mut_embedding'
        )
        
        # Genomic concatenation layer
        self.genomic_concat_layer = GenomicConcatLayer(
            ref_alt_dim=self.model_instance.ref_alt_dim,
            n_fc_layers=3,
            layer_norm_eps=self.layer_norm_eps,
            use_bias=self.use_bias,
            name='genomic_concat'
        )

        # VAF processing MLP (only created if use_vaf is True)
        if self.use_vaf:
            self.vaf_mlp = []
            for i in range(3):
                self.vaf_mlp.append(
                    tf.keras.layers.Dense(12, activation='gelu', use_bias=self.use_bias, name=f'vaf_mlp_{i}')
                )

        # Context processing layers
        self.context_concat = tf.keras.layers.Concatenate(axis=-2, name='context_concat')
        self.context_reverse = tf.keras.layers.Lambda(lambda x: tf.reverse(x, axis=[-2]), name='context_reverse')
        self.context_reverse_back = tf.keras.layers.Lambda(lambda x: tf.reverse(x, axis=[-2]), name='context_reverse_back')
        self.context_final_concat = tf.keras.layers.Concatenate(axis=-1, name='context_final_concat')
        
        # Convolutional layers for context processing
        self.conv_layers = []
        self.conv_layers_reverse = []
        self.conv_dropouts = []
        self.conv_dropouts_reverse = []
        conv_kernels = self.model_instance.local_conv_kernel
        conv_dim = self.model_instance.local_conv_dim
        for i in range(len(conv_kernels)):
            # Use efficient Conv1DFor4D instead of TimeDistributed
            self.conv_layers.append(
                Conv1DFor4D(
                    filters=conv_dim[i],
                    kernel_size=conv_kernels[i],
                    padding=self.padding,
                    activation='gelu',
                    use_bias=self.use_bias,
                    # kernel_initializer='he_normal',
                    name=f'conv_{i}'
                )
            )
            self.conv_dropouts.append(
                tf.keras.layers.TimeDistributed(
                    tf.keras.layers.Dropout(self.dropout_rate),
                    name=f'conv_dropout_{i}'
                )
            )
            self.conv_layers_reverse.append(
                Conv1DFor4D(
                    filters=conv_dim[i],
                    kernel_size=conv_kernels[i],
                    padding=self.padding,
                    activation='gelu',
                    use_bias=self.use_bias,
                    # kernel_initializer='he_normal',
                    name=f'conv_reverse_{i}'
                )
            )
            self.conv_dropouts_reverse.append(
                tf.keras.layers.TimeDistributed(
                    tf.keras.layers.Dropout(self.dropout_rate),
                    name=f'conv_dropout_reverse_{i}'
                )
            )

            # Commented out original TimeDistributed approach:
            # self.conv_layers.append(
            #     tf.keras.layers.TimeDistributed(
            #         tf.keras.layers.Conv1D(
            #             conv_dim[i], conv_kernels[i],
            #             padding=self.padding, activation='gelu',
            #             use_bias=self.use_bias
            #         ),
            #         name=f'conv_{i}'
            #     )
            # )
            # self.conv_dropouts.append(
            #     tf.keras.layers.TimeDistributed(
            #         tf.keras.layers.Dropout(self.dropout_rate),
            #         name=f'conv_dropout_{i}'
            #     )
            # )
            # self.conv_layers_reverse.append(
            #     tf.keras.layers.TimeDistributed(
            #         tf.keras.layers.Conv1D(
            #             conv_dim[i], conv_kernels[i],
            #             padding=self.padding, activation='gelu',
            #             use_bias=self.use_bias
            #         ),
            #         name=f'conv_reverse_{i}'
            #     )
            # )
            # self.conv_dropouts_reverse.append(
            #     tf.keras.layers.TimeDistributed(
            #         tf.keras.layers.Dropout(self.dropout_rate),
            #         name=f'conv_dropout_reverse_{i}'
            #     )
            # )
        
        # Positional encoders
        from tessera.layers.positional import SimpleNucleotidePositionalEncoder, SinCosPositionalEncoder
        self.simple_pos_encoder = SimpleNucleotidePositionalEncoder(mlp_dim=self.pos_mlp_dim)
        self.sincos_pos_encoder = SinCosPositionalEncoder()
        
        # Local attention blocks
        self.mut_local_blocks = []
        self.ref_local_blocks = []
        for i in range(self.model_instance.local_attention_blocks):
            self.mut_local_blocks.append(
                LocalAttentionBlock(
                    num_heads=self.model_instance.local_num_heads,
                    embed_dim=self.model_instance.local_embed_dim,
                    ff_dim=self.model_instance.local_ff_dim,
                    output_dim=self.model_instance.local_embed_dim,
                    attention_activation_type=self.attention_activation_type,
                    use_bias=False,
                    use_head_scaling=False,
                    head_scale_init=1.0,
                    dropout_rate=self.dropout_rate,
                    name=f'mut_local_block_{i}'
                )
            )
            self.ref_local_blocks.append(
                LocalAttentionBlock(
                    num_heads=self.model_instance.local_num_heads,
                    embed_dim=self.model_instance.local_embed_dim,
                    ff_dim=self.model_instance.local_ff_dim,
                    output_dim=self.model_instance.local_embed_dim,
                    attention_activation_type=self.attention_activation_type,
                    use_bias=False,
                    use_head_scaling=False,
                    head_scale_init=1.0,
                    dropout_rate=self.dropout_rate,
                    name=f'ref_local_block_{i}'
                )
            )
        
        # XLA-compatible flatten layers
        self.flatten_layer = FlattenLastTwoDims(
            name='flatten_last_two_dims'
        )

        self.flatten_mut_layer = FlattenLastTwoDims(
            name='flatten_mut_dims'
        )
        
        # Normalization layers
        self.layer_norms = []
        for i in range(6):  # Multiple normalization points
            self.layer_norms.append(
                tf.keras.layers.LayerNormalization(
                    epsilon=self.layer_norm_eps, center=False, scale=False,
                    name=f'layer_norm_{i}'
                )
            )
        
        # Dense layers for final processing
        self.dense_layers = []
        self.dense_layers_ref = []
        for i in range(3):
            self.dense_layers.append(
                tf.keras.layers.Dense(
                    self.model_instance.local_embed_dim, activation='gelu',
                    name=f'dense_{i}'
                )
            )
            self.dense_layers_ref.append(
                tf.keras.layers.Dense(
                    self.model_instance.local_embed_dim, activation='gelu',
                    name=f'dense_ref_{i}'
                )
            )
        
        # Projection layers from local to global embedding dimensions
        self.ref_projection = LocalToGlobalProjection(
            local_embed_dim=self.model_instance.local_embed_dim,
            global_embed_dim=self.model_instance.global_embed_dim,
            layer_norm_eps=self.layer_norm_eps,
            use_bias=self.use_bias,
            name='ref_local_to_global_projection'
        )
        
        self.mut_projection = LocalToGlobalProjection(
            local_embed_dim=self.model_instance.local_embed_dim,
            global_embed_dim=self.model_instance.global_embed_dim,
            layer_norm_eps=self.layer_norm_eps,
            use_bias=self.use_bias,
            name='mut_local_to_global_projection'
        )
        
    
    def call(self, inputs, is_wt = False,training=None,):
        """
        Process variant inputs through the featurization pipeline.

        Args:
            inputs: Dictionary containing variant inputs
            is_wt: If True, this is processing WT variants (affects masking strategy)

        Returns:
            Processed variant features and attention weights
        """
        layer_out = []
        # Extract inputs
        ref = inputs['ref']
        alt = inputs['alt']
        context_5p = inputs['context_5p']
        context_3p = inputs['context_3p']
        chr_input = inputs['chr']
        pos_input = inputs['pos']

        # Only extract VAF if use_vaf is True
        if self.use_vaf:
            vaf_input = inputs['vaf']

        # Apply sequence embedding
        embeddings = self.seq_embedding_layer({
            'ref': ref, 'alt': alt, 'context_5p': context_5p, 'context_3p': context_3p
        })
        ref_emb = embeddings['ref_emb']
        alt_emb = embeddings['alt_emb'] 
        context_5p_emb = embeddings['context_5p_emb']
        context_3p_emb = embeddings['context_3p_emb']
        
        # Apply position embedding
        genomic_emb = self.pos_embedding_layer({'chr': chr_input, 'pos': pos_input})
        
        # Apply mutation embedding
        mut_results = self.mut_embedding_layer({'ref_emb': ref_emb, 'alt_emb': alt_emb})
        ref_emb = mut_results['ref_emb']
        mut_emb = mut_results['mut_emb']

        # Reshape ref_emb and mut_emb from (None, bag_size, ref_alt_dim, nuc_embed_dim*4) to (None, bag_size, ref_alt_dim * nuc_embed_dim*4)
        ref_emb = self.flatten_mut_layer(ref_emb)
        mut_emb = self.flatten_mut_layer(mut_emb)

        # Add dimension for genomic concatenation: (None, bag_size, flattened_dim) -> (None, bag_size, 1, flattened_dim)
        ref_emb = tf.expand_dims(ref_emb, axis=-2)
        mut_emb = tf.expand_dims(mut_emb, axis=-2)
        
        # Apply genomic concatenation
        final_results = self.genomic_concat_layer({
            'ref_emb': ref_emb, 'mut_emb': mut_emb, 'genomic_emb': genomic_emb
        })
        ref_emb = final_results['ref_emb']
        mut_emb = final_results['mut_emb']

        # Concatenate VAF if use_vaf is True
        # VAF shape: (batch, bag_size, 1) -> pass through MLP -> expand to (batch, bag_size, 1, 12)
        if self.use_vaf:
            # Pass VAF through small MLP for training stability
            vaf_processed = vaf_input  # (batch, bag_size, 1)
            # for vaf_layer in self.vaf_mlp:
            #     vaf_processed = vaf_layer(vaf_processed)  # Still (batch, bag_size, 12) after all layers

            # Expand to match ref_emb/mut_emb dimensions: (batch, bag_size, 12) -> (batch, bag_size, 1, 12)
            vaf_expanded = tf.expand_dims(vaf_processed, axis=-2)  # (batch, bag_size, 1, 12)

            # Concatenate along feature dimension
            ref_emb = tf.concat([ref_emb, vaf_expanded], axis=-1)  # (batch, bag_size, 1, ref_alt_dim + 12)
            mut_emb = tf.concat([mut_emb, vaf_expanded], axis=-1)  # (batch, bag_size, 1, ref_alt_dim + 12)

        # layer_out.append(FlattenLastTwoDims()(ref_emb))
        layer_out.append(FlattenLastTwoDims()(mut_emb))

        # Local attention processing
        attn_local_5p = []
        attn_local_3p = []
        
        if self.model_instance.variant_local_attention:
            # Process context embeddings
            context_5p_emb_pre_conv = context_5p_emb
            context_3p_emb_pre_conv = context_3p_emb

            # Concatenate pre-conv embeddings together
            combined_context_pre_conv = tf.keras.layers.Concatenate(axis=-2)([context_5p_emb_pre_conv, context_3p_emb_pre_conv])

            if len(self.conv_layers) != 0:
                combined_context = self.context_concat([context_5p_emb, context_3p_emb])
                combined_context_reverse = self.context_reverse(combined_context)

                # Apply convolutional layers
                for i in range(len(self.conv_layers)):
                    combined_context = self.conv_layers[i](combined_context)
                    # combined_context = self.conv_dropouts[i](combined_context, training=training)
                    combined_context_reverse = self.conv_layers_reverse[i](combined_context_reverse)
                    # combined_context_reverse = self.conv_dropouts_reverse[i](combined_context_reverse, training=training)

                combined_context_reverse = self.context_reverse_back(combined_context_reverse)
                combined_context = self.context_final_concat([combined_context, combined_context_reverse])

                combined_context = tf.keras.layers.Concatenate(axis=-1)([combined_context, combined_context_pre_conv])
            else:
                combined_context = combined_context_pre_conv

            combined_context = self.simple_pos_encoder(combined_context)
            combined_context = self.sincos_pos_encoder(combined_context)
            
            # Apply local attention blocks
            current_mut_emb = mut_emb
            current_ref_emb = ref_emb
            
            for i in range(self.model_instance.local_attention_blocks):
                if self.model_instance.variant_self_attention or self.model_instance.mil:
                    ref_emb_unified, attn_s_ref_unified = self.ref_local_blocks[i]([current_ref_emb, combined_context])
                    mut_emb_unified, attn_s_unified = self.mut_local_blocks[i]([current_mut_emb, combined_context],precomputed_attention_scores=attn_s_ref_unified)

                else:
                    ref_emb_unified, attn_s_ref_unified = self.ref_local_blocks[i]([current_ref_emb, combined_context])

                
                # Split attention scores for interpretability
                context_len = self.model_instance.context_len
                attn_s_5p = attn_s_ref_unified[..., :context_len]
                attn_s_3p = attn_s_ref_unified[..., context_len:]

                attn_local_5p.append(attn_s_5p)
                attn_local_3p.append(attn_s_3p)
                #
                
                # Update embeddings for next iteration
                if i < self.model_instance.local_attention_blocks - 1:
                    if self.model_instance.variant_self_attention or self.model_instance.mil:
                        current_mut_emb = mut_emb_unified
                    current_ref_emb = ref_emb_unified

                # Use Keras safe named identity layer for feature extraction
                # named_identity_layer = NamedIdentityLayer(name=f'local_block_{i}_output')
                # ref_emb_unified = named_identity_layer(ref_emb_unified)
                layer_out.append(FlattenLastTwoDims()(ref_emb_unified))
                # layer_out.append(ref_emb_unified)
            
            # Final local attention outputs
            if self.model_instance.variant_self_attention:
                mut_context = self.flatten_layer(mut_emb_unified)
            ref_context = self.flatten_layer(ref_emb_unified)
        else:
            ref_context = self.flatten_layer(ref_emb)
            if self.model_instance.variant_self_attention or self.model_instance.mil:
                mut_context = self.flatten_layer(mut_emb)
        
        # Normalization
        if self.model_instance.variant_self_attention or self.model_instance.mil:
            mut_context = self.layer_norms[0](mut_context)
        ref_context = self.layer_norms[1](ref_context)

        # Dense layers
        supervised_out = []
        
        for i in range(3):
            if self.model_instance.variant_self_attention or self.model_instance.mil:
                mut_context = self.dense_layers[i](mut_context)
                # layer_out.append(mut_context)
                supervised_out.append(mut_context)
            
            ref_context = self.dense_layers_ref[i](ref_context)
            # layer_out.append(ref_context)
            supervised_out.append(ref_context)
        
        # Apply projection from local to global embedding dimensions
        if self.model_instance.variant_self_attention:
            ref_context = self.ref_projection(ref_context)
        if self.model_instance.variant_self_attention or self.model_instance.mil:
            mut_context = self.mut_projection(mut_context)

        # Return outputs projected to global embedding dimension
        # When self-attention is disabled, create a dummy tensor with same shape as ref_context
        # but filled with zeros to avoid None values that cause TensorFlow layer errors
        if not (self.model_instance.variant_self_attention or self.model_instance.mil):
            mut_context = tf.zeros_like(ref_context)
        
        return {
            'ref_context': ref_context,
            'mut_context': mut_context,
            'attn_local_5p': attn_local_5p,
            'attn_local_3p': attn_local_3p,
            'layer_out': layer_out,
            'supervised_out': supervised_out
        }
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'attention_activation_type': self.attention_activation_type,
            'layer_norm_eps': self.layer_norm_eps,
            'padding': self.padding,
            'dropout_rate': self.dropout_rate,
            'use_bias': self.use_bias,
            'pos_mlp_dim': self.pos_mlp_dim,
            # Save all model_instance attributes used in _build_layers()
            'token_map': self.model_instance.token_map,
            'nuc_embed_dim': self.model_instance.nuc_embed_dim,
            'chr_encoder': self.model_instance.chr_encoder,
            'ref_alt_dim': self.model_instance.ref_alt_dim,
            'local_conv_kernel': self.model_instance.local_conv_kernel,
            'local_conv_dim': self.model_instance.local_conv_dim,
            'context_len': self.model_instance.context_len,
            'local_attention_blocks': self.model_instance.local_attention_blocks,
            'local_num_heads': self.model_instance.local_num_heads,
            'local_embed_dim': self.model_instance.local_embed_dim,
            'local_ff_dim': self.model_instance.local_ff_dim,
            'global_embed_dim': self.model_instance.global_embed_dim,
            'variant_local_attention': self.model_instance.variant_local_attention,
            'variant_self_attention': self.model_instance.variant_self_attention,
            'mil': getattr(self.model_instance, 'mil', False),  # Use getattr with default for backward compatibility
            'use_vaf': self.use_vaf,  # Save use_vaf flag for correct architecture reconstruction
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Extract model_instance attributes from config
        config = config.copy()

        # Create a dummy model_instance with the saved attributes
        class DummyModelInstance:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

        model_attrs = {
            'token_map': config.pop('token_map'),
            'nuc_embed_dim': config.pop('nuc_embed_dim'),
            'chr_encoder': config.pop('chr_encoder'),
            'ref_alt_dim': config.pop('ref_alt_dim'),
            'local_conv_kernel': config.pop('local_conv_kernel'),
            'local_conv_dim': config.pop('local_conv_dim'),
            'context_len': config.pop('context_len'),
            'local_attention_blocks': config.pop('local_attention_blocks'),
            'local_num_heads': config.pop('local_num_heads'),
            'local_embed_dim': config.pop('local_embed_dim'),
            'local_ff_dim': config.pop('local_ff_dim'),
            'global_embed_dim': config.pop('global_embed_dim'),
            'variant_local_attention': config.pop('variant_local_attention'),
            'variant_self_attention': config.pop('variant_self_attention'),
            'mil': config.pop('mil'),
            'use_vaf': config.pop('use_vaf'),  # Restore use_vaf flag
        }

        model_instance = DummyModelInstance(**model_attrs)

        # The remaining config parameters (attention_activation_type, layer_norm_eps, etc.)
        # are passed directly to the constructor
        return cls(model_instance=model_instance, **config)