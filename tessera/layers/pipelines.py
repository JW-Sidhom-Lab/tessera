"""High-level featurisation pipelines.

Composite functions that wire together the lower-level layer modules
(attention, masking, positional, variant_features) into end-to-end featurisation
stacks for SNVs, CNAs, and joint multi-modal inputs. These pipelines are the
primary entry points used by the TESSERA model class to assemble its forward graph.
"""

import tensorflow as tf

# Import from new organized structure
from tessera.layers.attention import LocalAttentionBlock, GlobalAttentionBlock
from tessera.layers.masking import (
    CreateAttentionMaskLayer,
    SelfAttentionMaskLayer,
    CombineAttentionMasksLayer
)
from tessera.layers.variant_features import (
    create_variant_inputs,
    VariantFeaturizationLayer1,
    PostGlobalAttentionLayer
)
from tessera.layers.utils import (
    UnflattenLastDim,
    ReverseLayer,
    NamedIdentityLayer,
    graph_object
)

# Re-export for backward compatibility
__all__ = [
    'graph_object',
    'UnflattenLastDim',
    'ReverseLayer',
    'NamedIdentityLayer',
    'variant_featurization_layer',
    'expression_layer',
    'multimodal_featurization_layer',
    'LocalAttentionBlock',
    'GlobalAttentionBlock',
    'CreateAttentionMaskLayer',
    'SelfAttentionMaskLayer',
    'CombineAttentionMasksLayer',
    'VariantFeaturizationLayer1',
    'PostGlobalAttentionLayer',
]


def variant_featurization_layer(self,
                                inputs=None,
                                load_featurization_path=None,
                                freeze_featurization=False,
                                attention_activation_type='sigmoid',
                                layer_norm_eps=1e-6,
                                padding='same',
                                dropout_rate=0.0,
                                attention_l1_factor=0.0,
                                use_bias=False):
    """
    Legacy variant featurization layer function.

    This function maintains backward compatibility with existing code.
    For new code, use VariantFeaturizationLayer1 class directly.
    """

    # Use provided inputs or create them if not provided (for backward compatibility)
    if inputs is None:
        inputs = create_variant_inputs(self, use_vaf=hasattr(self, 'use_vaf') and self.use_vaf)

    # Extract individual inputs for processing
    ref = inputs['ref']
    alt = inputs['alt']
    context_5p = inputs['context_5p']
    context_3p = inputs['context_3p']
    chr_input = inputs['chr']
    pos_input = inputs['pos']

    # Use VariantFeaturizationLayer1 for the shared processing pipeline
    variant_feat_layer1 = VariantFeaturizationLayer1(
        model_instance=self,
        attention_activation_type=attention_activation_type,
        layer_norm_eps=layer_norm_eps,
        padding=padding,
        dropout_rate=dropout_rate,
        use_bias=use_bias,
        name='variant_featurization_layer1',
        pos_mlp_dim=None
    )

    # Process real variants - include VAF if present
    layer_inputs = {
        'ref': ref, 'alt': alt, 'context_5p': context_5p, 'context_3p': context_3p,
        'chr': chr_input, 'pos': pos_input
    }
    if 'vaf' in inputs:
        layer_inputs['vaf'] = inputs['vaf']

    real_results = variant_feat_layer1(layer_inputs, is_wt=False)

    ref_context = real_results['ref_context']
    mut_context = real_results['mut_context']
    attn_local_5p = real_results['attn_local_5p']
    attn_local_3p = real_results['attn_local_3p']
    layer_out = real_results['layer_out']
    supervised_out = real_results['supervised_out']

    ref_context_pre = ref_context

    # Initialize attention tracking variables
    attn = []
    attn_m = []
    global_layers = []  # Store global layers for reuse with WT variants

    if self.variant_self_attention:
        # Process real variants with self-attention + base masking
        self_mask_layer = SelfAttentionMaskLayer(name='global_self_mask')
        self_mask = self_mask_layer(ref_context)

        base_mask_layer = CreateAttentionMaskLayer(name='global_base_mask')
        base_attention_mask = base_mask_layer(ref)

        combine_mask_layer = CombineAttentionMasksLayer(name='global_combined_mask')
        attention_mask = combine_mask_layer([base_attention_mask, self_mask])

        # Store mut_context at each block for WT processing
        saved_mut_contexts = []

        # Create and store global attention layers for reuse
        for i in range(self.global_attention_blocks):
            # Layer for ref_context -> mut_context cross-attention
            global_layer = GlobalAttentionBlock(
                embed_dim=self.global_embed_dim,
                num_heads=self.global_num_heads,
                ff_dim=self.global_ff_dim,
                attention_type=self.attention_type,
                attention_activation_type=attention_activation_type,
                dropout_rate=dropout_rate,
                use_bias=False,
                use_head_scaling=False,
                head_scale_init=1.0,
                attention_l1_factor=attention_l1_factor,
                name=f'global_block_{i}'
            )
            global_layers.append(global_layer)

            ref_context, attn_s = global_layer([ref_context, mut_context, attention_mask])
            saved_mut_contexts.append(mut_context)  # Save current mut_context before updating

            attn.append(attn_s)
            attn_m.append(attention_mask)

            # Use Keras safe named identity layer for feature extraction
            # named_identity_layer = NamedIdentityLayer(name=f'global_block_{i}_output')
            # ref_context = named_identity_layer(ref_context)
            layer_out.append(ref_context)

        # Create shared post-global attention processing layer
        post_global_layer = PostGlobalAttentionLayer(
            ff_dim=self.global_ff_dim,
            layer_norm_eps=layer_norm_eps,
            name='post_global_attention'
        )

        # Apply shared post-global attention processing for real variants
        ref_context = post_global_layer(ref_context)

    else:
        ref_context = ref_context_pre

    # Create featurization submodel with appropriate inputs and outputs
    model_inputs = [ref, alt, context_5p, context_3p, chr_input, pos_input]
    if self.variant_self_attention or self.mil:
        model_outputs = [ref_context, mut_context]
    else:
        model_outputs = [ref_context]

    featurization_submodel = tf.keras.Model(
        inputs=model_inputs,
        outputs=model_outputs,
        name="featurization_submodel"
    )

    if load_featurization_path is not None:
        featurization_submodel.load_weights(
            load_featurization_path,
            by_name=True,
            skip_mismatch=True
        )

    if freeze_featurization:
        for layer in featurization_submodel.layers:
            layer.trainable = False

    # if self.variant_self_attention:
    #     layer_out = [ref_context,mut_context]
    # else:
    #     layer_out = [ref_context]

    if self.mil:
        supervised_out = tf.keras.layers.Concatenate(axis=-1)(supervised_out)
        return (supervised_out, layer_out, attn, attn_local_5p, attn_local_3p, attn_m)
    else:
        return (ref_context, layer_out, attn, attn_local_5p, attn_local_3p, attn_m)


def expression_layer(self):
    """
    Legacy expression layer function.

    Creates input and processing layers for gene expression data.
    """
    exp_input = tf.keras.layers.Input(shape=(None, self.n_exp), dtype=tf.float32, name='exp')

    n_fc_layers = 3
    x = exp_input
    for _ in range(n_fc_layers):
        x = tf.keras.layers.Dense(self.embed_dim, activation='relu')(x)
        x = tf.keras.layers.LayerNormalization()(x)

    return x, exp_input


def multimodal_featurization_layer(
    self,
    inputs=None,
    load_featurization_path=None,
    freeze_featurization=False,
    attention_activation_type='sigmoid',
    layer_norm_eps=1e-6,
    padding='same',
    dropout_rate=0.0,
    attention_l1_factor=0.0,
    use_bias=False,
    # CNA-specific parameters
    cna_embed_dim=64,
    cna_num_heads=8,
    cna_ff_dim=128,
    cna_attention_blocks=2,
    use_chr_one_hot=True,
    chr_embed_dim=16,
    use_cna_loh=False,
    predict_cna_loh=False,
    # Cross-modal parameters
    cross_modal_num_heads=8,
    cross_modal_ff_dim=256,
    cross_modal_blocks=2,
    cross_attention_explicit=False
):
    """
    Unified multi-modal featurization layer that adapts based on available data modalities.

    This function processes available genomic data modalities (mutations and/or CNAs) based on
    the self.use_mut and self.use_cna flags that are auto-detected during dataset creation.

    Args:
        self: Model instance with use_mut and use_cna flags
        inputs: Dictionary of input tensors (optional)
        load_featurization_path: Path to load pretrained weights (optional)
        freeze_featurization: Whether to freeze featurization layers
        attention_activation_type: Activation for attention ('sigmoid', 'softmax')
        layer_norm_eps: Epsilon for layer normalization
        padding: Padding type for convolutions
        dropout_rate: Dropout rate for regularization
        attention_l1_factor: L1 regularization for attention sparsity
        use_bias: Whether to use bias in dense layers

        # CNA parameters
        cna_embed_dim: Embedding dimension for CNA features
        cna_num_heads: Number of attention heads for CNA self-attention
        cna_ff_dim: Feed-forward dimension for CNA layers
        cna_attention_blocks: Number of CNA self-attention blocks
        use_chr_one_hot: Whether to use one-hot encoding for chromosomes
        chr_embed_dim: Embedding dimension for chromosomes (if not one-hot)

        # Cross-modal parameters
        cross_modal_num_heads: Number of attention heads for cross-modal attention
        cross_modal_ff_dim: Feed-forward dimension for cross-modal layers
        cross_modal_blocks: Number of cross-modal attention blocks
        cross_attention_explicit: Controls K/V representations in cross-modal attention (default: False)
                                 'concat': Concatenates learned + explicit representations for richer K/V
                                 True: CNAs see explicit ref/alt, mutations see explicit segment_mean
                                 False: Both see learned representations without explicit labels

    Returns:
        dict: Dictionary containing:
            - 'mut_features': Mutation features (if use_mut=True)
            - 'cna_features': CNA features (if use_cna=True)
            - 'mut_context': Mutation context for reconstruction (if use_mut=True)
            - 'attention_dict': Dictionary of attention scores for visualization
            - 'layer_outputs': List of intermediate layer outputs
            - 'inputs': Dictionary of all input layers

    Example:
        >>> # Multi-modal case (both mutations and CNAs)
        >>> results = multimodal_featurization_layer(
        ...     model, cna_embed_dim=64, cross_modal_blocks=2
        ... )
        >>> mut_features = results['mut_features']  # For mutation reconstruction
        >>> cna_features = results['cna_features']  # For CNA prediction

        >>> # Mutation-only case
        >>> results = multimodal_featurization_layer(model)
        >>> mut_features = results['mut_features']  # CNA features will be None
    """

    # Import required layers
    from tessera.layers.cna_features import CNAEmbedding, CNAEmbeddingUnified, CNASelfAttentionBlock, create_cna_inputs
    from tessera.layers.cross_modal import CrossModalAttentionBlock
    from tessera.layers.masking import (
        CNASelfAttentionMaskLayer,
        SNVToCNAMaskLayer,
        CNAToSNVMaskLayer
    )

    # Initialize result dictionary
    result = {
        'mut_features': None,
        'cna_features': None,
        'mut_context': None,
        'attention_dict': {},
        'mut_layer_out': [],  # Track mutation features through architecture
        'cna_layer_out': [],  # Track CNA features through architecture
        'inputs': {}
    }

    # Check which modalities are available
    if not self.use_mut and not self.use_cna:
        raise ValueError("No modalities detected. Either mutation or CNA data must be provided.")

    # =========================================================================
    # 1. Process Mutation Data (if available)
    # =========================================================================
    if self.use_mut:
        # Use provided inputs or create them
        if inputs is None or 'ref' not in inputs:
            mut_inputs = create_variant_inputs(self, use_vaf=hasattr(self, 'use_vaf') and self.use_vaf)
        else:
            mut_inputs = {k: v for k, v in inputs.items() if k in ['ref', 'alt', 'context_5p', 'context_3p', 'chr', 'pos', 'vaf']}

        result['inputs'].update(mut_inputs)

        # Extract individual inputs
        ref = mut_inputs['ref']
        alt = mut_inputs['alt']
        context_5p = mut_inputs['context_5p']
        context_3p = mut_inputs['context_3p']
        chr_input = mut_inputs['chr']
        pos_input = mut_inputs['pos']

        # Use VariantFeaturizationLayer1 for mutation processing
        variant_feat_layer1 = VariantFeaturizationLayer1(
            model_instance=self,
            attention_activation_type=attention_activation_type,
            layer_norm_eps=layer_norm_eps,
            padding=padding,
            dropout_rate=dropout_rate,
            use_bias=use_bias,
            name='variant_featurization_layer1',
            pos_mlp_dim=12
        )

        # Process mutations
        layer_inputs = {
            'ref': ref, 'alt': alt, 'context_5p': context_5p, 'context_3p': context_3p,
            'chr': chr_input, 'pos': pos_input
        }
        if 'vaf' in mut_inputs:
            layer_inputs['vaf'] = mut_inputs['vaf']

        mut_results = variant_feat_layer1(layer_inputs, is_wt=False)

        ref_context = mut_results['ref_context']
        mut_context = mut_results['mut_context']
        result['attention_dict']['attn_local_5p'] = mut_results['attn_local_5p']
        result['attention_dict']['attn_local_3p'] = mut_results['attn_local_3p']
        # Collect initial mutation layer outputs (from local attention blocks)
        result['mut_layer_out'].extend(mut_results['layer_out'])

        # Apply mutation self-attention if enabled
        mut_self_attn_scores = []
        if self.variant_self_attention:
            self_mask_layer = SelfAttentionMaskLayer(name='mut_self_mask')
            self_mask = self_mask_layer(ref_context)

            base_mask_layer = CreateAttentionMaskLayer(name='mut_base_mask')
            base_attention_mask = base_mask_layer(ref)

            combine_mask_layer = CombineAttentionMasksLayer(name='mut_combined_mask')
            attention_mask = combine_mask_layer([base_attention_mask, self_mask])

            for i in range(self.global_attention_blocks):
                global_layer = GlobalAttentionBlock(
                    embed_dim=self.global_embed_dim,
                    num_heads=self.global_num_heads,
                    ff_dim=self.global_ff_dim,
                    attention_type=self.attention_type,
                    attention_activation_type=attention_activation_type,
                    dropout_rate=dropout_rate,
                    use_bias=False,
                    use_head_scaling=False,
                    head_scale_init=1.0,
                    attention_l1_factor=attention_l1_factor,
                    name=f'mut_global_block_{i}'
                )

                ref_context, attn_s = global_layer([ref_context, mut_context, attention_mask])
                mut_self_attn_scores.append(attn_s)
                # Collect mutation features after each global attention block
                result['mut_layer_out'].append(ref_context)

            # Post-global attention processing
            post_global_layer = PostGlobalAttentionLayer(
                ff_dim=self.global_ff_dim,
                layer_norm_eps=layer_norm_eps,
                name='mut_post_global_attention'
            )
            ref_context = post_global_layer(ref_context)

        result['attention_dict']['mut_self_attention'] = mut_self_attn_scores
        mut_features = ref_context
        result['mut_context'] = mut_context
    else:
        mut_features = None
        chr_input = None  # Will be needed for masking if we have CNA

    # =========================================================================
    # 2. Process CNA Data (if available)
    # =========================================================================
    if self.use_cna:
        # Create or use provided CNA inputs
        if inputs is None or 'cna_chr' not in inputs:
            cna_inputs = create_cna_inputs(
                use_loh=use_cna_loh
            )
        else:
            cna_inputs = {k: v for k, v in inputs.items() if k.startswith('cna_')}

        result['inputs'].update(cna_inputs)

        # OLD APPROACH (commented out): Two separate CNAEmbedding layers
        # This caused weight transfer issues when cna_attention_blocks=0 because
        # cna_kv_embedding layer was not connected to prediction outputs, so its
        # weights were not in model_inf when trying to save the features model.
        #
        # # Create TWO separate CNA embedding layers to prevent data leakage:
        # # 1. Query embedding WITHOUT segment_mean (for prediction)
        # #    When predicting LOH, also exclude LOH from query to prevent leakage
        # cna_query_embedding_layer = CNAEmbedding(
        #     chr_encoder=self.chr_encoder,
        #     embed_dim=cna_embed_dim,
        #     use_segment_mean=False,  # ← NO segment_mean to prevent leakage
        #     use_chr_one_hot=use_chr_one_hot,
        #     chr_embed_dim=chr_embed_dim,
        #     layer_norm_eps=layer_norm_eps,
        #     use_bias=use_bias,
        #     n_fc_layers=3,
        #     use_breakpoint_density=use_cna_breakpoint_density,
        #     use_delta_cn=use_cna_delta_cn,
        #     use_loh=use_cna_loh and not predict_cna_loh,  # ← Exclude LOH from query if predicting it
        #     name='cna_query_embedding'
        # )
        #
        # # 2. Key/Value embedding WITH segment_mean (for attending to other segments)
        # cna_kv_embedding_layer = CNAEmbedding(
        #     chr_encoder=self.chr_encoder,
        #     embed_dim=cna_embed_dim,
        #     use_segment_mean=True,  # ← WITH segment_mean for other segments to see
        #     use_chr_one_hot=use_chr_one_hot,
        #     chr_embed_dim=chr_embed_dim,
        #     layer_norm_eps=layer_norm_eps,
        #     use_bias=use_bias,
        #     n_fc_layers=3,
        #     use_breakpoint_density=use_cna_breakpoint_density,
        #     use_delta_cn=use_cna_delta_cn,
        #     use_loh=use_cna_loh,
        #     name='cna_kv_embedding'
        # )
        #
        # # Create both embeddings
        # cna_query_emb = cna_query_embedding_layer(cna_inputs)
        # cna_kv_emb = cna_kv_embedding_layer(cna_inputs)

        # NEW APPROACH: Use UNIFIED CNA embedding layer that returns both query and kv embeddings.
        # Similar to MutEmbedding which returns both ref_emb and mut_emb from a single layer,
        # this ensures the embedding weights are in the model graph when either output is
        # connected to predictions (since both come from the same layer call).
        # - Query embedding: base features WITHOUT segment_mean (for prediction)
        # - KV embedding: base features WITH segment_mean (for attention/features)
        # - Each has SEPARATE dense layers (like MutEmbedding has ref_dense_layers and mut_dense_layers)
        cna_unified_embedding_layer = CNAEmbeddingUnified(
            chr_encoder=self.chr_encoder,
            embed_dim=cna_embed_dim,
            use_chr_one_hot=use_chr_one_hot,
            chr_embed_dim=chr_embed_dim,
            layer_norm_eps=layer_norm_eps,
            use_bias=use_bias,
            n_fc_layers=3,
            use_loh=use_cna_loh,
            predict_cna_loh=predict_cna_loh,  # Exclude LOH from query if predicting it
            name='cna_unified_embedding'
        )

        # Get both embeddings from the unified layer
        cna_emb_outputs = cna_unified_embedding_layer(cna_inputs)
        cna_query_emb = cna_emb_outputs['query_emb']
        cna_kv_emb = cna_emb_outputs['kv_emb']

        # Collect KV embedding for features (includes segment_mean)
        # Since both query and kv come from the same layer, and query is connected
        # to predictions, the embedding weights are guaranteed to be in model_inf
        result['cna_layer_out'].append(cna_kv_emb)

        # CNA self-attention with separate query and key/value embeddings
        cna_self_attn_scores = []
        if cna_attention_blocks > 0:
            # Create diagonal self-attention mask (prevents attending to self)
            cna_self_mask_layer = SelfAttentionMaskLayer(name='cna_self_mask')
            cna_self_mask = cna_self_mask_layer(cna_query_emb)

            # Create padding mask (prevents attending to padded segments)
            cna_padding_mask_layer = CNASelfAttentionMaskLayer(name='cna_padding_mask')
            cna_padding_mask = cna_padding_mask_layer(cna_inputs['cna_chr'])

            # Combine both masks for complete masking
            combine_cna_mask_layer = CombineAttentionMasksLayer(name='cna_combined_mask')
            cna_mask = combine_cna_mask_layer([cna_padding_mask, cna_self_mask])

            for i in range(cna_attention_blocks):
                cna_self_attn_layer = CNASelfAttentionBlock(
                    num_heads=cna_num_heads,
                    embed_dim=cna_embed_dim,
                    ff_dim=cna_ff_dim,
                    dropout_rate=dropout_rate,
                    use_bias=False,
                    layer_norm_eps=layer_norm_eps,
                    attention_l1_factor=attention_l1_factor,
                    name=f'cna_self_attention_block_{i}'
                )

                # Pass separate query and key/value embeddings
                # Query: embeddings WITHOUT segment_mean (what we're predicting)
                # K/V: embeddings WITH segment_mean (what other segments can see)
                # Combined mask prevents: 1) attending to padding, 2) attending to self
                # IMPORTANT: cna_kv_emb stays FIXED across all blocks (like mut_context in mutation self-attention)
                # This prevents information leakage from segment_mean through query→K/V recycling
                cna_query_emb, attn_scores = cna_self_attn_layer(
                    cna_query_emb=cna_query_emb,  # Query: no segment_mean (updated each block)
                    cna_kv_emb=cna_kv_emb,         # K/V: with segment_mean (FIXED across all blocks)
                    attention_mask=cna_mask         # Combined padding + diagonal mask
                )
                cna_self_attn_scores.append(attn_scores)

                # Collect CNA features after each self-attention block
                result['cna_layer_out'].append(cna_query_emb)

            # Post-CNA self-attention processing (matching mutation post-global attention)
            post_cna_self_attn_layer = PostGlobalAttentionLayer(
                ff_dim=cna_ff_dim,
                layer_norm_eps=layer_norm_eps,
                name='cna_post_self_attention'
            )
            cna_query_emb = post_cna_self_attn_layer(cna_query_emb)

        result['attention_dict']['cna_self_attention'] = cna_self_attn_scores

        # Use query embeddings as final CNA features (the ones without segment_mean leakage)
        cna_features = cna_query_emb
    else:
        cna_features = None

    # =========================================================================
    # 3. Cross-Modal Attention (if both modalities available)
    # =========================================================================
    # Save pre-cross-modal features for InfoNCE (before modalities see each other)
    result['mut_layer_out_pre_cross_modal'] = list(result['mut_layer_out'])
    result['cna_layer_out_pre_cross_modal'] = list(result['cna_layer_out'])

    if self.use_mut and self.use_cna and cross_modal_blocks > 0:
        # Bidirectional cross-modal attention
        mut_to_cna_attn_scores = []
        cna_to_mut_attn_scores = []

        # IMPORTANT: Save initial features as FIXED K/V embeddings across all blocks
        # This prevents information leakage through query→K/V recycling (same as CNA self-attention)

        if cross_attention_explicit == 'concat':
            # Concatenate learned + explicit along feature dimension for richer K/V
            mut_features_kv = tf.keras.layers.Concatenate(axis=-1)([mut_features, mut_context])
            cna_features_kv = tf.keras.layers.Concatenate(axis=-1)([cna_features, cna_kv_emb])
        elif cross_attention_explicit:
            # Use EXPLICIT representations in cross-modal K/V (safe because tasks are orthogonal):
            # - CNAs don't predict ref/alt, so they CAN see explicit ref/alt
            # - Mutations don't predict segment_mean, so they CAN see explicit segment_mean
            # This provides richer cross-modal information without leaking prediction targets
            mut_features_kv = mut_context    # Contains explicit ref/alt embeddings
            cna_features_kv = cna_kv_emb     # Contains explicit segment_mean
        else:
            # Use LEARNED representations in cross-modal K/V (no explicit labels):
            # - CNAs see learned mutation representations (no explicit ref/alt)
            # - Mutations see learned CNA representations (no explicit segment_mean)
            mut_features_kv = mut_features
            cna_features_kv = cna_features

        # Create cross-modal masks using layers
        snv_to_cna_mask_layer = SNVToCNAMaskLayer(name='snv_to_cna_mask')
        snv_to_cna_mask = snv_to_cna_mask_layer([chr_input, cna_inputs['cna_chr']])

        cna_to_snv_mask_layer = CNAToSNVMaskLayer(name='cna_to_snv_mask')
        cna_to_snv_mask = cna_to_snv_mask_layer([cna_inputs['cna_chr'], chr_input])

        for i in range(cross_modal_blocks):
            # SNV -> CNA: SNV features (query) attend to FIXED CNA features (K/V)
            mut_to_cna_layer = CrossModalAttentionBlock(
                num_heads=cross_modal_num_heads,
                embed_dim=self.global_embed_dim,  # Must match mutation embed_dim
                ff_dim=cross_modal_ff_dim,
                output_dim=self.global_embed_dim,  # Match query (mutation) dimension
                dropout_rate=dropout_rate,
                use_bias=False,
                layer_norm_eps=layer_norm_eps,
                attention_l1_factor=attention_l1_factor,
                name=f'mut_to_cna_cross_attention_{i}'
            )

            mut_features, mut_to_cna_scores = mut_to_cna_layer(
                query_emb=mut_features,         # Query: updated each block
                key_value_emb=cna_features_kv,  # K/V: FIXED across all blocks
                attention_mask=snv_to_cna_mask
            )
            mut_to_cna_attn_scores.append(mut_to_cna_scores)

            # CNA -> SNV: CNA features (query) attend to FIXED SNV features (K/V)
            cna_to_mut_layer = CrossModalAttentionBlock(
                num_heads=cross_modal_num_heads,
                embed_dim=cna_embed_dim,  # Must match CNA embed_dim
                ff_dim=cross_modal_ff_dim,
                output_dim=cna_embed_dim,  # Match query (CNA) dimension
                dropout_rate=dropout_rate,
                use_bias=False,
                layer_norm_eps=layer_norm_eps,
                attention_l1_factor=attention_l1_factor,
                name=f'cna_to_mut_cross_attention_{i}'
            )

            cna_features, cna_to_mut_scores = cna_to_mut_layer(
                query_emb=cna_features,         # Query: updated each block
                key_value_emb=mut_features_kv,  # K/V: FIXED across all blocks
                attention_mask=cna_to_snv_mask
            )
            cna_to_mut_attn_scores.append(cna_to_mut_scores)

            # Collect features after each cross-modal attention block
            result['mut_layer_out'].append(mut_features)
            result['cna_layer_out'].append(cna_features)

        result['attention_dict']['mut_to_cna_cross_attention'] = mut_to_cna_attn_scores
        result['attention_dict']['cna_to_mut_cross_attention'] = cna_to_mut_attn_scores

        # Post-cross-modal attention processing (matching post-global attention pattern)
        post_cross_modal_mut_layer = PostGlobalAttentionLayer(
            ff_dim=self.global_ff_dim,
            layer_norm_eps=layer_norm_eps,
            name='mut_post_cross_modal_attention'
        )
        mut_features = post_cross_modal_mut_layer(mut_features)

        post_cross_modal_cna_layer = PostGlobalAttentionLayer(
            ff_dim=cna_ff_dim,
            layer_norm_eps=layer_norm_eps,
            name='cna_post_cross_modal_attention'
        )
        cna_features = post_cross_modal_cna_layer(cna_features)

    # =========================================================================
    # 4. Set final features
    # =========================================================================
    result['mut_features'] = mut_features
    result['cna_features'] = cna_features

    # =========================================================================
    # 5. Create featurization submodel (optional)
    # =========================================================================
    if load_featurization_path is not None or freeze_featurization:
        # Build list of model inputs and outputs
        model_inputs = []
        model_outputs = []

        if self.use_mut:
            model_inputs.extend([
                result['inputs']['ref'],
                result['inputs']['alt'],
                result['inputs']['context_5p'],
                result['inputs']['context_3p'],
                result['inputs']['chr'],
                result['inputs']['pos']
            ])
            model_outputs.append(mut_features)
            if result['mut_context'] is not None:
                model_outputs.append(result['mut_context'])

        if self.use_cna:
            model_inputs.extend([
                result['inputs']['cna_chr'],
                result['inputs']['cna_start'],
                result['inputs']['cna_end'],
                result['inputs']['cna_length'],
                result['inputs']['cna_segment_mean']
            ])
            model_outputs.append(cna_features)

        featurization_submodel = tf.keras.Model(
            inputs=model_inputs,
            outputs=model_outputs,
            name="multimodal_featurization_submodel"
        )

        if load_featurization_path is not None:
            featurization_submodel.load_weights(
                load_featurization_path,
                by_name=True,
                skip_mismatch=True
            )

        if freeze_featurization:
            for layer in featurization_submodel.layers:
                layer.trainable = False

        result['featurization_submodel'] = featurization_submodel

    return result