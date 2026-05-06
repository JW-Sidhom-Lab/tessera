"""Positional encoding layers.

Sinusoidal, adaptive, and nucleotide-specific positional encodings used to inject
genomic-position information into TESSERA's attention layers. Includes a helper
(calculate_optimal_sincos_frequency) that picks sin/cos frequencies appropriate to
the encoded sequence length so the encoding doesn't alias.
"""

import tensorflow as tf
import numpy as np

def calculate_optimal_sincos_frequency(seq_length):
    """
    Calculate optimal base frequency for sin/cos positional encoding.
    
    For short sequences, we need a much smaller frequency than the traditional 10000
    to ensure positions are distinguishable and show sinusoidal variation.
    
    We want roughly 1-2 full cycles across the sequence length:
    - For 1 cycle: freq = seq_length / (2*pi) ≈ seq_length / 6.28
    - For 2 cycles: freq = seq_length / (4*pi) ≈ seq_length / 12.57
    """
    # Use frequency that gives about 1.5 cycles across sequence length
    # This provides good variation while maintaining distinguishability
    optimal_frequency = seq_length / (3.0 * 3.14159)  # ≈ seq_length / 9.42
    
    return optimal_frequency

@tf.keras.utils.register_keras_serializable(
    package="tessera.layers.positional",
    name="PositionalEncoding")
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, logp=False, use_concat=False, auto_frequency=True, **kwargs):
        super().__init__(**kwargs)
        self.logp = logp
        self.use_concat = use_concat
        self.auto_frequency = auto_frequency

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        seq_length, embed_dim = input_shape[-2], input_shape[-1]

        pos_encoding = self.get_positional_encoding(seq_length, embed_dim)
        
        # Cast to match input dtype for mixed precision compatibility
        pos_encoding = tf.cast(pos_encoding, inputs.dtype)
        
        # Add leading dimensions to match input rank: [seq_len, embed_dim] -> [1, 1, seq_len, embed_dim]
        for _ in range(inputs.shape.rank - 2):
            pos_encoding = tf.expand_dims(pos_encoding, axis=0)

        if self.use_concat:
            # Tile to match all batch dimensions
            batch_shape = input_shape[:-2]  # All dimensions except last 2
            tile_multiples = tf.concat([batch_shape, [1, 1]], axis=0)
            pos_encoding = tf.tile(pos_encoding, tile_multiples)
            return tf.concat([inputs, pos_encoding], axis=-1)
        else:
            return inputs + pos_encoding

    def get_positional_encoding(self, length, depth):
        if self.logp:
            positions = tf.math.log1p(tf.range(length, dtype=tf.float32))[:, tf.newaxis]
        else:
            positions = tf.range(length, dtype=tf.float32)[:, tf.newaxis]
        depths = tf.range(depth, dtype=tf.float32)[tf.newaxis, :] / tf.cast(depth, tf.float32)

        if self.auto_frequency:
            # Use optimal frequency for short sequences
            base_frequency = calculate_optimal_sincos_frequency(tf.cast(length, tf.float32))
        else:
            # Use traditional frequency
            base_frequency = 10000.0

        angle_rates = 1 / (base_frequency**depths)
        angle_rads = positions * angle_rates

        pos_encoding = tf.concat([tf.math.sin(angle_rads), tf.math.cos(angle_rads)], axis=-1)

        return pos_encoding[:, :depth]  # Ensure the last dimension matches the input depth

    def compute_output_shape(self, input_shape):
        if self.use_concat:
            # When concatenating, the last dimension doubles
            return input_shape[:-1] + (input_shape[-1] * 2,)
        else:
            return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            'logp': self.logp,
            'use_concat': self.use_concat,
            'auto_frequency': self.auto_frequency
        })
        return config

@tf.keras.utils.register_keras_serializable(
    package="tessera.layers.positional",
    name="AdaptivePositionalEncoding")
class AdaptivePositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, use_sinusoidal_init=False, use_concat=False, **kwargs):
        super().__init__(**kwargs)
        self.pe = None
        self.use_sinusoidal_init = use_sinusoidal_init
        self.use_concat = use_concat

    def build(self, input_shape):
        # Get the sequence length and embedding dimension from the input shape
        seq_length, embed_dim = input_shape[-2], input_shape[-1]

        if self.use_sinusoidal_init:
            # Initialize with traditional sinusoidal positional encoding
            def sinusoidal_initializer(shape, dtype=None):
                return self._get_sinusoidal_encoding(shape[0], shape[1])
            
            initializer = sinusoidal_initializer
        else:
            # Use random initialization (original behavior)
            initializer = "random_normal"

        # Initialize the positional encoding
        self.pe = self.add_weight(
            name="pos_encoding",
            shape=(seq_length, embed_dim),
            initializer=initializer,
            trainable=True,
        )

    def _get_sinusoidal_encoding(self, length, depth):
        """Generate sinusoidal positional encoding (same as PositionalEncoding)"""
        # Use float32 explicitly for mixed precision compatibility
        positions = tf.range(length, dtype=tf.float32)[:, tf.newaxis]
        depths = tf.range(depth, dtype=tf.float32)[tf.newaxis, :] / tf.cast(depth, tf.float32)

        angle_rates = 1 / (10000**depths)
        angle_rads = positions * angle_rates

        pos_encoding = tf.concat([tf.math.sin(angle_rads), tf.math.cos(angle_rads)], axis=-1)

        return tf.cast(pos_encoding[:, :depth], tf.float32)  # Ensure float32 output

    def call(self, inputs):
        # Cast positional encoding to match input dtype for mixed precision compatibility
        pe_casted = tf.cast(self.pe, inputs.dtype)
        
        # Get the actual sequence length from inputs
        input_shape = tf.shape(inputs)
        actual_seq_len = input_shape[-2]
        
        # Slice positional encoding to match actual sequence length
        pe_casted = pe_casted[:actual_seq_len, :]
        
        # Add leading dimensions to match input rank
        for _ in range(inputs.shape.rank - 2):
            pe_casted = tf.expand_dims(pe_casted, axis=0)
        
        if self.use_concat:
            # Tile to match all batch dimensions
            batch_shape = input_shape[:-2]  # All dimensions except last 2
            tile_multiples = tf.concat([batch_shape, [1, 1]], axis=0)
            pe_casted = tf.tile(pe_casted, tile_multiples)
            return tf.concat([inputs, pe_casted], axis=-1)
        else:
            return inputs + pe_casted

    def compute_output_shape(self, input_shape):
        if self.use_concat:
            # When concatenating, the last dimension doubles
            return input_shape[:-1] + (input_shape[-1] * 2,)
        else:
            return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            'use_sinusoidal_init': self.use_sinusoidal_init,
            'use_concat': self.use_concat
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

@tf.keras.utils.register_keras_serializable(
    package="tessera.layers.positional",
    name="NucleotidePositionalEncoding")
class NucleotidePositionalEncoding(tf.keras.layers.Layer):
    """
    Positional encoding specifically designed for nucleotide sequences.
    Uses learnable positional embeddings initialized with one-hot position encoding.
    """
    def __init__(self, use_concat=False, **kwargs):
        super().__init__(**kwargs)
        self.position_embeddings = None
        self.use_concat = use_concat
        
    def build(self, input_shape):
        # Get the sequence length and embedding dimension from the input shape
        seq_length, embed_dim = input_shape[-2], input_shape[-1]
        
        # Initialize with one-hot position encoding
        def one_hot_position_initializer(shape, dtype=None):
            seq_len, emb_dim = shape
            
            # Create one-hot encoding for positions
            if seq_len <= emb_dim:
                # If we have enough embedding dimensions, use pure one-hot
                embeddings = np.zeros((seq_len, emb_dim), dtype=np.float32)
                embeddings[:seq_len, :seq_len] = np.eye(seq_len)
            else:
                # If sequence is longer than embedding dim, use repeated one-hot pattern
                embeddings = np.zeros((seq_len, emb_dim), dtype=np.float32)
                for i in range(seq_len):
                    pos_idx = i % emb_dim  # Cycle through embedding dimensions
                    embeddings[i, pos_idx] = 1.0
            
            return embeddings
        
        # Create learnable positional embeddings initialized with one-hot
        self.position_embeddings = self.add_weight(
            name="position_embeddings",
            shape=(seq_length, embed_dim),
            initializer=one_hot_position_initializer,
            trainable=True,
        )
        
    def call(self, inputs):
        # Cast to match input dtype for mixed precision compatibility
        position_encodings = tf.cast(self.position_embeddings, inputs.dtype)
        
        # Get the actual sequence length from inputs
        input_shape = tf.shape(inputs)
        actual_seq_len = input_shape[-2]
        
        # Slice positional encoding to match actual sequence length
        position_encodings = position_encodings[:actual_seq_len, :]
        
        # Add leading dimensions to match input rank
        for _ in range(inputs.shape.rank - 2):
            position_encodings = tf.expand_dims(position_encodings, axis=0)
        
        if self.use_concat:
            # Tile to match all batch dimensions
            batch_shape = input_shape[:-2]  # All dimensions except last 2
            tile_multiples = tf.concat([batch_shape, [1, 1]], axis=0)
            position_encodings = tf.tile(position_encodings, tile_multiples)
            return tf.concat([inputs, position_encodings], axis=-1)
        else:
            return inputs + position_encodings

    def compute_output_shape(self, input_shape):
        if self.use_concat:
            # When concatenating, the last dimension doubles
            return input_shape[:-1] + (input_shape[-1] * 2,)
        else:
            return input_shape
        
    def get_config(self):
        config = super().get_config()
        config.update({
            'use_concat': self.use_concat
        })
        return config
        
    @classmethod
    def from_config(cls, config):
        return cls(**config)

@tf.keras.utils.register_keras_serializable(
    package="tessera.layers.positional",
    name="SimpleNucleotidePositionalEncoder")
class SimpleNucleotidePositionalEncoder(tf.keras.layers.Layer):
    """
    Simple nucleotide positional encoder that creates 1-hot position embeddings
    and concatenates them to token features. Each position gets a unique 1-hot vector.

    If mlp_dim is provided, uses sparse encoding -> embedding approach for efficiency:
    - Avoids creating large one-hot matrices for long contexts
    - Uses learnable embedding layer to map positions to compressed representations
    """
    def __init__(self, mlp_dim=None, **kwargs):
        super().__init__(**kwargs)
        self.mlp_dim = mlp_dim
        self.position_embeddings = None
        self.position_embedding_layer = None
        
    def build(self, input_shape):
        # Get the sequence length from the input shape
        seq_length = input_shape[-2]

        if self.mlp_dim is not None:
            # Use sparse encoding -> embedding approach for efficiency
            self.position_embedding_layer = tf.keras.layers.Embedding(
                input_dim=seq_length,
                output_dim=self.mlp_dim,
                name="position_embedding"
            )
        else:
            # Original approach: create 1-hot position embeddings
            # Shape: [seq_length, seq_length] where each row is a 1-hot vector
            position_matrix = np.eye(seq_length, dtype=np.float32)

            # Store as non-trainable weights (fixed 1-hot encoding)
            self.position_embeddings = self.add_weight(
                name="position_embeddings",
                shape=(seq_length, seq_length),
                initializer="zeros",
                trainable=False,
            )

            # Set the 1-hot values
            self.position_embeddings.assign(position_matrix)
        
    def call(self, inputs):
        # Get the actual sequence length from inputs
        input_shape = tf.shape(inputs)
        actual_seq_len = input_shape[-2]

        if self.mlp_dim is not None:
            # Use sparse encoding -> embedding approach
            # Create position indices: [0, 1, 2, ..., seq_len-1]
            position_indices = tf.range(actual_seq_len, dtype=tf.int32)

            # Get embeddings for each position: [seq_len, mlp_dim]
            position_encodings = self.position_embedding_layer(position_indices)

            # Cast to match input dtype for mixed precision compatibility
            position_encodings = tf.cast(position_encodings, inputs.dtype)
        else:
            # Original approach: use one-hot position embeddings
            position_encodings = tf.cast(self.position_embeddings, inputs.dtype)

            # Slice positional encoding to match actual sequence length
            position_encodings = position_encodings[:actual_seq_len, :actual_seq_len]

        # Add leading dimensions to match input rank
        for _ in range(inputs.shape.rank - 2):
            position_encodings = tf.expand_dims(position_encodings, axis=0)

        # Tile to match all batch dimensions
        batch_shape = input_shape[:-2]  # All dimensions except last 2
        tile_multiples = tf.concat([batch_shape, [1, 1]], axis=0)
        position_encodings = tf.tile(position_encodings, tile_multiples)

        # Always concatenate the position embeddings to token features
        return tf.concat([inputs, position_encodings], axis=-1)

    def compute_output_shape(self, input_shape):
        # Output shape: original features + position features
        original_dim = input_shape[-1]
        if self.mlp_dim is not None:
            # When using embedding approach, add mlp_dim
            return input_shape[:-1] + (original_dim + self.mlp_dim,)
        else:
            # When using one-hot approach, add seq_length
            seq_length = input_shape[-2]
            return input_shape[:-1] + (original_dim + seq_length,)
        
    def get_config(self):
        config = super().get_config()
        config.update({
            'mlp_dim': self.mlp_dim
        })
        return config
        
    @classmethod
    def from_config(cls, config):
        return cls(**config)

@tf.keras.utils.register_keras_serializable(
    package="tessera.layers.positional",
    name="SinCosPositionalEncoder")
class SinCosPositionalEncoder(tf.keras.layers.Layer):
    """
    Simple sin/cos positional encoder that concatenates 2D sin/cos position embeddings
    to token features. Adds exactly 2 dimensions: [sin(pos), cos(pos)].
    """
    def __init__(self, base_frequency=10000.0, auto_frequency=True, **kwargs):
        super().__init__(**kwargs)
        self.base_frequency = base_frequency
        self.auto_frequency = auto_frequency
        
    def call(self, inputs):
        # Get input shape
        input_shape = tf.shape(inputs)
        seq_length = input_shape[-2]
        
        # Create position indices
        positions = tf.range(seq_length, dtype=tf.float32)
        
        # Determine frequency to use
        if self.auto_frequency:
            # Use the optimal frequency calculation
            frequency = calculate_optimal_sincos_frequency(tf.cast(seq_length, tf.float32))
        else:
            frequency = self.base_frequency
        
        # Calculate sin/cos values for each position
        # Using frequency scaling: pos / frequency
        angle_rads = positions / frequency
        
        # Create sin and cos components
        sin_component = tf.sin(angle_rads)  # [seq_length]
        cos_component = tf.cos(angle_rads)  # [seq_length]
        
        # Stack to create 2D positional encoding
        position_encodings = tf.stack([sin_component, cos_component], axis=-1)  # [seq_length, 2]
        
        # Cast to match input dtype for mixed precision compatibility
        position_encodings = tf.cast(position_encodings, inputs.dtype)
        
        # Add leading dimensions to match input rank
        for _ in range(inputs.shape.rank - 2):
            position_encodings = tf.expand_dims(position_encodings, axis=0)
        
        # Tile to match all batch dimensions
        batch_shape = input_shape[:-2]  # All dimensions except last 2
        tile_multiples = tf.concat([batch_shape, [1, 1]], axis=0)
        position_encodings = tf.tile(position_encodings, tile_multiples)
        
        # Concatenate the 2D sin/cos position embeddings to token features
        return tf.concat([inputs, position_encodings], axis=-1)

    def compute_output_shape(self, input_shape):
        # Output shape: original features + 2 (sin/cos)
        original_dim = input_shape[-1]
        return input_shape[:-1] + (original_dim + 2,)
        
    def get_config(self):
        config = super().get_config()
        config.update({
            'base_frequency': self.base_frequency,
            'auto_frequency': self.auto_frequency
        })
        return config
        
    @classmethod
    def from_config(cls, config):
        return cls(**config)

