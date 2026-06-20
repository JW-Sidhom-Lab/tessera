"""
Keras Layers for TESSERA.

This package contains all TensorFlow/Keras layer implementations organized by functionality:

Modules:
    attention: Multi-head attention mechanisms (local and global)
    cross_modal: Cross-modal attention for multi-modal learning (SNV↔CNA)
    masking: Attention mask creation and combination
    positional: Positional encoding layers (sinusoidal, adaptive, nucleotide-specific)
    variant_features: Variant-specific feature extraction and embedding layers
    mil: Multiple Instance Learning aggregation layers
    utils: Common utility layers (reshape, flatten, identity, etc.)

Quick Reference:
---------------

Attention Layers:
    - CustomMultiHeadAttention: Multi-head attention with L1 regularization
    - LocalAttentionBlock: Local (within-sequence) attention
    - GlobalAttentionBlock: Global (cross-sequence) attention with masking
    - CrossModalAttentionBlock: Cross-modal attention (SNV↔CNA)

Masking Layers:
    - CreateAttentionMaskLayer: Create padding masks from sequences
    - SelfAttentionMaskLayer: Prevent self-attention (diagonal mask)
    - CombineAttentionMasksLayer: Combine multiple masks with logical AND

Positional Encoding:
    - PositionalEncoding: Sinusoidal positional encoding
    - AdaptivePositionalEncoding: Learnable positional encoding
    - NucleotidePositionalEncoding: Nucleotide-specific encoding
    - SimpleNucleotidePositionalEncoder: One-hot position encoding
    - SinCosPositionalEncoder: Simple 2D sin/cos encoding

Variant Feature Layers:
    - SeqEmbedding: Sequence embedding (one-hot or learned)
    - PosEmbedding: Chromosome and position embedding
    - MutEmbedding: Mutation embedding (ref + alt processing)
    - GenomicConcatLayer: Concatenate genomic features
    - VariantFeaturizationLayer1: Complete variant featurization pipeline
    - PostGlobalAttentionLayer: Post-attention processing
    - LocalToGlobalProjection: Dimension projection between layers
    - create_variant_inputs: Helper function to create model inputs

MIL Layers:
    - MILAttentionLayerAVG: Weighted average aggregation
    - MILAttentionLayerMAX: Maximum value aggregation
    - MILMaskingLayer: MIL bag masking
    - MILMultiClassAttentionLayer: Class-specific attention

Utility Layers:
    - UnflattenLastDim: Reshape last dimension to 2D
    - FlattenLastTwoDims: Flatten last two dimensions
    - ReverseLayer: Reverse tensor along axis
    - NamedIdentityLayer: Identity layer for intermediate outputs
    - Conv1DFor4D: Efficient Conv1D for 4D tensors
    - graph_object: Legacy compatibility object

Example Usage:
-------------
```python
# Import individual layers
from tessera.layers import CustomMultiHeadAttention, LocalAttentionBlock

# Or import from submodules
from tessera.layers.attention import GlobalAttentionBlock
from tessera.layers.positional import PositionalEncoding
from tessera.layers.variant_features import SeqEmbedding

# Create attention layers
local_attn = LocalAttentionBlock(
    num_heads=4,
    embed_dim=128,
    ff_dim=256,
    output_dim=128
)

# Create positional encoding
pos_enc = PositionalEncoding(auto_frequency=True)

# Build complete model
model = tf.keras.Sequential([
    SeqEmbedding(token_map={'A': 0, 'C': 1, 'G': 2, 'T': 3}),
    pos_enc,
    local_attn,
])
```

For backward compatibility, all layers can also be imported from their original locations
via tessera module aliases.
"""

# Pipeline functions (high-level orchestration)
from tessera.layers.pipelines import (
    variant_featurization_layer,
    multimodal_featurization_layer,
    expression_layer,
)

# Attention layers
from tessera.layers.attention import (
    CustomMultiHeadAttention,
    LocalAttentionBlock,
    GlobalAttentionBlock,
)

# Cross-modal attention layers
from tessera.layers.cross_modal import (
    CrossModalAttentionBlock,
)

# Masking layers
from tessera.layers.masking import (
    CreateAttentionMaskLayer,
    SelfAttentionMaskLayer,
    CombineAttentionMasksLayer,
    CNASelfAttentionMaskLayer,
    SNVToCNAMaskLayer,
    CNAToSNVMaskLayer,
)

# Positional encoding layers
from tessera.layers.positional import (
    PositionalEncoding,
    AdaptivePositionalEncoding,
    NucleotidePositionalEncoding,
    SimpleNucleotidePositionalEncoder,
    SinCosPositionalEncoder,
)

# Variant feature layers
from tessera.layers.variant_features import (
    create_variant_inputs,
    PosEmbedding,
    MutEmbedding,
    GenomicConcatLayer,
    SeqEmbedding,
    LocalToGlobalProjection,
    PostGlobalAttentionLayer,
    VariantFeaturizationLayer1,
)

# MIL layers
from tessera.layers.mil import (
    MILAttentionLayerAVG,
    MILAttentionLayerMAX,
    MILMaskingLayer,
    MILMultiClassAttentionLayer,
)

# Utility layers
from tessera.layers.utils import (
    UnflattenLastDim,
    FlattenLastTwoDims,
    Conv1DFor4D,
    ReverseLayer,
    NamedIdentityLayer,
    graph_object,
)

# Pooling / mask layers. Imported here so their @register_keras_serializable
# decorators run on `import tessera`. Otherwise `pooling` is only imported lazily
# during model building, and loading the saved reconstruction model
# (final_model.keras, which references CreateMaskLayer) fails on a fresh install.
from tessera.layers.pooling import (
    CreateMaskLayer,
    MaskedGlobalAveragePooling1D,
)

# Define public API
__all__ = [
    # Pipeline functions
    'variant_featurization_layer',
    'multimodal_featurization_layer',
    'expression_layer',
    # Attention
    'CustomMultiHeadAttention',
    'LocalAttentionBlock',
    'GlobalAttentionBlock',
    # Cross-modal attention
    'CrossModalAttentionBlock',
    # Masking
    'CreateAttentionMaskLayer',
    'SelfAttentionMaskLayer',
    'CombineAttentionMasksLayer',
    'CNASelfAttentionMaskLayer',
    'SNVToCNAMaskLayer',
    'CNAToSNVMaskLayer',
    # Positional
    'PositionalEncoding',
    'AdaptivePositionalEncoding',
    'NucleotidePositionalEncoding',
    'SimpleNucleotidePositionalEncoder',
    'SinCosPositionalEncoder',
    # Variant Features
    'create_variant_inputs',
    'PosEmbedding',
    'MutEmbedding',
    'GenomicConcatLayer',
    'SeqEmbedding',
    'LocalToGlobalProjection',
    'PostGlobalAttentionLayer',
    'VariantFeaturizationLayer1',
    # MIL
    'MILAttentionLayerAVG',
    'MILAttentionLayerMAX',
    'MILMaskingLayer',
    'MILMultiClassAttentionLayer',
    # Utils
    'UnflattenLastDim',
    'FlattenLastTwoDims',
    'Conv1DFor4D',
    'ReverseLayer',
    'NamedIdentityLayer',
    'graph_object',
    # Pooling / mask
    'CreateMaskLayer',
    'MaskedGlobalAveragePooling1D',
]