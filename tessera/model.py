"""Joint SNV+CNA foundation model.

Defines the TESSERA class — a self-supervised foundation model jointly pretrained
on somatic single-nucleotide variants and copy-number alterations through masked-token
reconstruction within each modality and an InfoNCE contrastive objective across
modalities. The class inherits dataset-construction and inference infrastructure
from BaseModel (see tessera.base) and adds the model architecture, training loops,
and feature-extraction methods.
"""

import numpy as np
import pandas as pd
import os
import pickle
from pyfaidx import Fasta
import tensorflow as tf
import tessera.data.preprocessing
from tessera.training.models import CustomTrainingModel
from tessera.training.metrics import masked_accuracy
from tessera.training.callbacks import (
    CustomEarlyStopping,
    OptimizerResetCallback,
    TensorToFloatCallback
)
from tessera.training.schedules import CosineAnnealingWithWarmup
from tessera.layers import variant_featurization_layer, multimodal_featurization_layer, UnflattenLastDim
from tessera.layers.variant_features import create_variant_inputs
from tessera.layers.cna_features import CNAEmbedding, CNASelfAttentionBlock
import shutil
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from keras.config import enable_unsafe_deserialization
from tessera.base import BaseModel
from tessera.input_keys import get_input_keys

class TESSERA(BaseModel):
    def build_model(self,
                    recon_ref = True,
                    nuc_embed_dim=12,
                    local_conv_dim = [512,256,128],
                    local_conv_kernel =[25,100,500],
                    ref_alt_dim = 12,
                    local_embed_dim = 256,
                    global_embed_dim = 256,
                    local_num_heads=6,
                    local_ff_dim=256,
                    local_attention_blocks=3,
                    global_num_heads = 8,
                    global_ff_dim = 512,
                    global_attention_blocks = 3,
                    variant_local_attention = True,
                    variant_self_attention = True,
                    attention_type = 'pairwise',
                    attention_activation_type = 'softmax',
                    dropout_rate=0.1,
                    attention_l1_factor=0.0,
                    # CNA-specific parameters
                    cna_embed_dim=64,
                    cna_num_heads=8,
                    cna_ff_dim=128,
                    cna_self_attention=True,
                    cna_attention_blocks=3,
                    use_chr_one_hot=True,
                    chr_embed_dim=16,
                    # CNA dual-task parameters
                    predict_cna_loh=False,
                    # Cross-modal parameters
                    cross_modal_num_heads=8,
                    cross_modal_ff_dim=256,
                    cross_modal_blocks=0,
                    cross_attention_explicit=False,
                    # Training parameters
                    train_on_mutation_loss=True,
                    train_on_cna_loss=True,
                    # Other parameters
                    load_featurization_path=None,
                    freeze_featurization=False,
                    max_learning_rate=1e-3,
                    warmup_steps=1000,
                    min_learning_rate=1e-6,
                    use_cosine_schedule=False,
                    loss_non_zero_only=False,
                    accuracy_non_zero_only=False,
                    hinge_loss_t=None,
                    log_gradients=False,
                    intermediate_dim_1=512,
                    intermediate_dim_2=128,
                    per_sample_loss=False,
                    # InfoNCE loss parameters
                    use_infonce_loss=False,
                    infonce_projection_dim=128,
                    infonce_temperature=0.1,
                    infonce_loss_weight=1.0,
                    # Token-bag InfoNCE parameters (independent per modality)
                    use_mut_token_bag_infonce=False,
                    use_cna_token_bag_infonce=False,
                    token_bag_temperature=0.1,
                    token_bag_loss_weight=1.0,
                    # Projection MLP architecture
                    infonce_shared_projection=True,
                    infonce_projection_n_layers=1,
                    infonce_projection_dropout=None):
        """
        Build the TESSERA model with specified architecture and training parameters.

        Args:
            intermediate_dim_1 (int): Size of first intermediate dense layer for progressive
                dimensionality reduction. The model compresses features from >3000 dimensions
                to this size. Default: 512.

            intermediate_dim_2 (int): Size of second intermediate dense layer for progressive
                dimensionality reduction. Features are further compressed from intermediate_dim_1
                to this size before final output. Default: 128.

            dropout_rate (float): Dropout rate applied throughout the model for regularization.
                Controls dropout in attention layers, feed-forward networks, and dense layers.
                Default: 0.1.

            Other parameters: See method signature for full list of architecture and
                           training configuration parameters.
        """

        self.nuc_embed_dim = nuc_embed_dim
        self.local_conv_dim = local_conv_dim
        self.local_conv_kernel = local_conv_kernel
        self.ref_alt_dim = ref_alt_dim
        self.local_embed_dim = local_embed_dim
        self.global_embed_dim = global_embed_dim
        self.local_num_heads = local_num_heads
        self.local_ff_dim = local_ff_dim
        self.local_attention_blocks = local_attention_blocks
        self.global_num_heads = global_num_heads
        self.global_ff_dim = global_ff_dim
        self.global_attention_blocks = global_attention_blocks
        self.variant_local_attention = variant_local_attention
        self.variant_self_attention = variant_self_attention
        self.attention_type = attention_type
        self.attention_activation_type = attention_activation_type
        self.dropout_rate = dropout_rate
        self.attention_l1_factor = attention_l1_factor
        self.recon_ref = recon_ref
        self.intermediate_dim_1 = intermediate_dim_1
        self.intermediate_dim_2 = intermediate_dim_2
        self.cna_self_attention = cna_self_attention

        # CNA dual-task parameters
        self.predict_cna_loh = predict_cna_loh

        # Cross-modal parameters
        self.cross_modal_blocks = cross_modal_blocks

        # InfoNCE loss parameters
        self.use_infonce_loss = use_infonce_loss
        self.infonce_projection_dim = infonce_projection_dim
        self.infonce_temperature = infonce_temperature
        self.infonce_loss_weight = infonce_loss_weight
        # Dropout on the InfoNCE projection MLP; falls back to self.dropout_rate if None.
        self.infonce_projection_dropout = infonce_projection_dropout

        # Token-bag InfoNCE parameters (independent per modality)
        self.use_mut_token_bag_infonce = use_mut_token_bag_infonce
        self.use_cna_token_bag_infonce = use_cna_token_bag_infonce
        self.token_bag_temperature = token_bag_temperature
        self.token_bag_loss_weight = token_bag_loss_weight

        # Save model configuration for later use
        self._save_model_config()

        with self.strategy.scope():
            # Call multimodal featurization layer (handles input creation internally)
            feat_results = multimodal_featurization_layer(
                self,
                inputs=None,  # Let it create inputs internally
                load_featurization_path=load_featurization_path,
                freeze_featurization=freeze_featurization,
                attention_activation_type=attention_activation_type,
                dropout_rate=dropout_rate,
                attention_l1_factor=attention_l1_factor,
                # CNA parameters
                cna_embed_dim=cna_embed_dim,
                cna_num_heads=cna_num_heads,
                cna_ff_dim=cna_ff_dim,
                cna_attention_blocks=cna_attention_blocks,
                use_chr_one_hot=use_chr_one_hot,
                chr_embed_dim=chr_embed_dim,
                # CNA optional features (from auto-detection)
                use_cna_loh=self.use_cna_loh,
                predict_cna_loh=self.predict_cna_loh,
                # Cross-modal parameters
                cross_modal_num_heads=cross_modal_num_heads,
                cross_modal_ff_dim=cross_modal_ff_dim,
                cross_modal_blocks=cross_modal_blocks,
                cross_attention_explicit=cross_attention_explicit
            )

            # Extract features based on available modalities
            inputs = feat_results['inputs']
            mut_features = feat_results['mut_features']  # Will be None if self.use_mut=False
            cna_features = feat_results['cna_features']  # Will be None if self.use_cna=False
            mut_context = feat_results['mut_context']
            mut_layer_out = feat_results['mut_layer_out']  # Mutation features through architecture
            cna_layer_out = feat_results['cna_layer_out']  # CNA features through architecture
            # Pre-cross-modal features for InfoNCE (before modalities see each other)
            mut_layer_out_pre_cm = feat_results['mut_layer_out_pre_cross_modal']
            cna_layer_out_pre_cm = feat_results['cna_layer_out_pre_cross_modal']
            attention_dict = feat_results['attention_dict']

            # Initialize model outputs dictionary
            model_outputs = {}

            # Create mutation reconstruction heads if mutation data is available
            if self.use_mut and mut_features is not None:
                # Create shared progressive dimensionality reduction layers for mutations
                # Gradual compression: >3000 → intermediate_dim_1 → intermediate_dim_2 → final_output
                intermediate_1 = tf.keras.layers.Dense(intermediate_dim_1, activation='gelu', name="intermediate_dense_1")
                dropout_1 = tf.keras.layers.Dropout(self.dropout_rate, name="dropout_1")
                intermediate_2 = tf.keras.layers.Dense(intermediate_dim_2, activation='gelu', name="intermediate_dense_2")
                dropout_2 = tf.keras.layers.Dropout(self.dropout_rate, name="dropout_2")
                output_dense = tf.keras.layers.Dense(self.alt_len * len(self.token_map), name="output_dense")

                # Apply progressive compression to mutation features
                x_compressed = intermediate_1(mut_features)
                x_compressed = dropout_1(x_compressed)
                x_compressed = intermediate_2(x_compressed)
                x_compressed = dropout_2(x_compressed)
                logits = output_dense(x_compressed)
                logits = UnflattenLastDim(self.alt_len, len(self.token_map))(logits)
                model_outputs['logits'] = logits

                # Create reference reconstruction head if requested
                if recon_ref:
                    intermediate_1_ref = tf.keras.layers.Dense(intermediate_dim_1, activation='gelu', name="intermediate_dense_1_ref")
                    dropout_1_ref = tf.keras.layers.Dropout(self.dropout_rate, name="dropout_1_ref")
                    intermediate_2_ref = tf.keras.layers.Dense(intermediate_dim_2, activation='gelu', name="intermediate_dense_2_ref")
                    dropout_2_ref = tf.keras.layers.Dropout(self.dropout_rate, name="dropout_2_ref")
                    output_dense_ref = tf.keras.layers.Dense(self.ref_len * len(self.token_map), name="output_dense_ref")

                    x_compressed_ref = intermediate_1_ref(mut_features)
                    x_compressed_ref = dropout_1_ref(x_compressed_ref)
                    x_compressed_ref = intermediate_2_ref(x_compressed_ref)
                    x_compressed_ref = dropout_2_ref(x_compressed_ref)
                    logits_ref = output_dense_ref(x_compressed_ref)
                    logits_ref = UnflattenLastDim(self.ref_len, len(self.token_map))(logits_ref)
                    model_outputs['logits_ref'] = logits_ref

            # Create CNA prediction head(s) if CNA data is available
            if self.use_cna and cna_features is not None:
                # Shared encoder layers for CNA features
                cna_dense = tf.keras.layers.Dense(128, activation='relu', name="cna_dense_1")
                cna_dropout = tf.keras.layers.Dropout(self.dropout_rate, name="cna_dropout")

                # Apply shared encoder
                cna_compressed = cna_dense(cna_features)
                cna_compressed = cna_dropout(cna_compressed)

                # Task 1: Segment mean prediction (always present)
                cna_segment_mean_head = tf.keras.layers.Dense(1, activation='linear', name="cna_segment_mean_output")
                cna_segment_mean_pred = cna_segment_mean_head(cna_compressed)
                # Output shape: [batch, num_segments, 1] - matches dataset shape

                # Store segment_mean prediction with both new and legacy keys for backward compatibility
                model_outputs['cna_segment_mean_pred'] = cna_segment_mean_pred
                model_outputs['cna_pred'] = cna_segment_mean_pred  # Legacy key for backward compatibility

                # Task 2: LOH prediction (optional, enabled by predict_cna_loh flag)
                if self.predict_cna_loh:
                    cna_loh_head = tf.keras.layers.Dense(1, activation='sigmoid', name="cna_loh_output")
                    cna_loh_pred = cna_loh_head(cna_compressed)
                    # Output shape: [batch, num_segments, 1] - matches dataset shape
                    model_outputs['cna_loh_pred'] = cna_loh_pred

            # Create concatenated feature tensors for feature extraction (ALL layers including cross-modal)
            # Mutation features
            mut_features_concat = None
            if mut_layer_out:
                if len(mut_layer_out) > 1:
                    mut_features_concat = tf.keras.layers.Concatenate(axis=-1, name='mut_features_concat')(mut_layer_out)
                else:
                    mut_features_concat = mut_layer_out[0]

            # CNA features
            cna_features_concat = None
            if cna_layer_out:
                if len(cna_layer_out) > 1:
                    cna_features_concat = tf.keras.layers.Concatenate(axis=-1, name='cna_features_concat')(cna_layer_out)
                else:
                    cna_features_concat = cna_layer_out[0]

            # Create PRE-cross-modal concatenated features for InfoNCE
            # InfoNCE must operate on unimodal representations (before modalities see each other)
            # to prevent the contrastive task from being trivially solved via cross-modal attention
            mut_features_pre_cm = None
            if mut_layer_out_pre_cm:
                if len(mut_layer_out_pre_cm) > 1:
                    mut_features_pre_cm = tf.keras.layers.Concatenate(axis=-1, name='mut_features_pre_cm_concat')(mut_layer_out_pre_cm)
                else:
                    mut_features_pre_cm = mut_layer_out_pre_cm[0]

            cna_features_pre_cm = None
            if cna_layer_out_pre_cm:
                if len(cna_layer_out_pre_cm) > 1:
                    cna_features_pre_cm = tf.keras.layers.Concatenate(axis=-1, name='cna_features_pre_cm_concat')(cna_layer_out_pre_cm)
                else:
                    cna_features_pre_cm = cna_layer_out_pre_cm[0]

            # Add InfoNCE projection layers for contrastive learning
            # Both contrastive losses use PRE-cross-modal features to prevent
            # trivial optimization through cross-modal attention leakage:
            #   1. Cross-modal InfoNCE (mut↔cna): pre-CM so modalities can't see each other
            #   2. Token-bag InfoNCE (token↔sample, intra-modal): pre-CM so bag
            #      representations don't contain injected cross-modal information

            if use_infonce_loss or use_mut_token_bag_infonce or use_cna_token_bag_infonce:
                from tessera.layers.pooling import MaskedGlobalAveragePooling1D, CreateMaskLayer
                drop_rate = 0.2

                # Create masks (shared across cross-modal and token-bag)
                mut_mask = None
                cna_mask = None
                if self.use_mut:
                    mut_mask_layer = CreateMaskLayer(squeeze=True, name='mut_mask_creator')
                    mut_mask = mut_mask_layer(inputs['chr'])
                if self.use_cna:
                    cna_mask_layer = CreateMaskLayer(squeeze=True, name='cna_mask_creator')
                    cna_mask = cna_mask_layer(inputs['cna_chr'])

            # --- InfoNCE projection heads ---
            # Two modes controlled by infonce_shared_projection:
            #
            # Shared (True): tokens → MLP → projected_tokens; bag = mean(projected_tokens)
            #   One set of weights, project-then-mean. Used for both token-bag and bag-to-bag.
            #
            # Separate (False): tokens → Token MLP; mean(backbone) → Sample MLP (same dims)
            #   Two independent projection heads. Token MLP for token-bag loss;
            #   Sample MLP for bag-to-bag cross-modal loss.
            #
            # Both modes use pre-cross-modal backbone features so contrastive signals
            # are not contaminated by cross-modal attention shortcuts.

            # Dropout on projection: explicit override, else inherit backbone dropout_rate.
            _proj_dropout_rate = (self.infonce_projection_dropout
                                  if self.infonce_projection_dropout is not None
                                  else self.dropout_rate)

            def _proj_mlp(x, n_layers, proj_dim, name_prefix):
                """n-layer MLP: (n-1) hidden (GELU + Dropout) blocks + 1 final linear layer.
                n_layers=1 → single linear layer (no hidden, no dropout);
                n_layers=2 → 1 GELU + Dropout + 1 linear."""
                for i in range(n_layers - 1):
                    x = tf.keras.layers.Dense(proj_dim, activation='gelu',
                                              name=f'{name_prefix}_h{i}')(x)
                    if _proj_dropout_rate > 0:
                        x = tf.keras.layers.Dropout(_proj_dropout_rate,
                                                    name=f'{name_prefix}_h{i}_drop')(x)
                return tf.keras.layers.Dense(proj_dim, activation=None,
                                             name=f'{name_prefix}_out')(x)

            mut_tb_features = mut_features_pre_cm if mut_features_pre_cm is not None else mut_features_concat
            cna_tb_features = cna_features_pre_cm if cna_features_pre_cm is not None else cna_features_concat

            if infonce_shared_projection:
                # --- Mode 1: shared projection (project-then-mean) ---
                # Build mut projection whenever token-bag OR bag-to-bag InfoNCE is active
                if (use_mut_token_bag_infonce or use_infonce_loss) and self.use_mut and mut_tb_features is not None:
                    mut_token_emb = _proj_mlp(mut_tb_features, infonce_projection_n_layers,
                                              infonce_projection_dim, 'mut_token_proj')
                    mut_bag_emb = MaskedGlobalAveragePooling1D(name='mut_bag_pooling')(
                        mut_token_emb, mask=mut_mask)
                    if use_mut_token_bag_infonce:
                        model_outputs['mut_token_embedding'] = mut_token_emb
                        model_outputs['mut_token_bag_embedding'] = mut_bag_emb
                    if use_infonce_loss:
                        # Register projected tokens for enrichment concat in model_features_mut
                        model_outputs['mut_token_embedding'] = mut_token_emb
                        model_outputs['mut_sample_embedding'] = mut_bag_emb

                # Build cna projection whenever token-bag OR bag-to-bag InfoNCE is active
                if (use_cna_token_bag_infonce or use_infonce_loss) and self.use_cna and cna_tb_features is not None:
                    cna_token_emb = _proj_mlp(cna_tb_features, infonce_projection_n_layers,
                                              infonce_projection_dim, 'cna_token_proj')
                    cna_bag_emb = MaskedGlobalAveragePooling1D(name='cna_bag_pooling')(
                        cna_token_emb, mask=cna_mask)
                    if use_cna_token_bag_infonce:
                        model_outputs['cna_token_embedding'] = cna_token_emb
                        model_outputs['cna_token_bag_embedding'] = cna_bag_emb
                    if use_infonce_loss:
                        # Register projected tokens for enrichment concat in model_features_cna
                        model_outputs['cna_token_embedding'] = cna_token_emb
                        model_outputs['cna_sample_embedding'] = cna_bag_emb

            else:
                # --- Mode 2: separate token and sample projections ---
                if use_mut_token_bag_infonce and self.use_mut and mut_tb_features is not None:
                    mut_token_emb = _proj_mlp(mut_tb_features, infonce_projection_n_layers,
                                              infonce_projection_dim, 'mut_token_proj')
                    model_outputs['mut_token_embedding'] = mut_token_emb

                if self.use_mut and mut_tb_features is not None and (
                        use_infonce_loss or use_mut_token_bag_infonce):
                    mut_pooled = MaskedGlobalAveragePooling1D(name='mut_sample_pooling')(
                        mut_tb_features, mask=mut_mask)
                    mut_sample_emb = _proj_mlp(mut_pooled, infonce_projection_n_layers,
                                               infonce_projection_dim, 'mut_sample_proj')
                    if use_infonce_loss:
                        model_outputs['mut_sample_embedding'] = mut_sample_emb
                    if use_mut_token_bag_infonce:
                        model_outputs['mut_token_bag_embedding'] = mut_sample_emb

                if use_cna_token_bag_infonce and self.use_cna and cna_tb_features is not None:
                    cna_token_emb = _proj_mlp(cna_tb_features, infonce_projection_n_layers,
                                              infonce_projection_dim, 'cna_token_proj')
                    model_outputs['cna_token_embedding'] = cna_token_emb

                if self.use_cna and cna_tb_features is not None and (
                        use_infonce_loss or use_cna_token_bag_infonce):
                    cna_pooled = MaskedGlobalAveragePooling1D(name='cna_sample_pooling')(
                        cna_tb_features, mask=cna_mask)
                    cna_sample_emb = _proj_mlp(cna_pooled, infonce_projection_n_layers,
                                               infonce_projection_dim, 'cna_sample_proj')
                    if use_infonce_loss:
                        model_outputs['cna_sample_embedding'] = cna_sample_emb
                    if use_cna_token_bag_infonce:
                        model_outputs['cna_token_bag_embedding'] = cna_sample_emb

            # Create Keras models
            self.model = tf.keras.Model(inputs=inputs, outputs=model_outputs)
            self.model_inf = tf.keras.Model(inputs=inputs, outputs=model_outputs)

            # Create separate attention models for each attention type
            # 1. Mutation local attention (5' and 3' contexts)
            if variant_local_attention and 'attn_local_5p' in attention_dict and 'attn_local_3p' in attention_dict:
                self.model_attn_local = tf.keras.Model(
                    inputs=inputs,
                    outputs=[attention_dict['attn_local_5p'], attention_dict['attn_local_3p']],
                    name='mutation_local_attention_model'
                )

            # 2. Mutation self-attention
            if self.variant_self_attention and 'mut_self_attention' in attention_dict:
                mut_self_attn = attention_dict['mut_self_attention']
                if mut_self_attn:  # List of attention scores from each block
                    self.model_attn_mut_self = tf.keras.Model(
                        inputs=inputs,
                        outputs=mut_self_attn,  # List of tensors, one per block
                        name='mutation_self_attention_model'
                    )

            # 3. CNA self-attention
            if self.use_cna and 'cna_self_attention' in attention_dict:
                cna_self_attn = attention_dict['cna_self_attention']
                if cna_self_attn:  # List of attention scores from each block
                    self.model_attn_cna_self = tf.keras.Model(
                        inputs=inputs,
                        outputs=cna_self_attn,  # List of tensors, one per block
                        name='cna_self_attention_model'
                    )

            # 4. Mutation-to-CNA cross-modal attention
            if self.use_mut and self.use_cna and 'mut_to_cna_cross_attention' in attention_dict:
                mut_to_cna_attn = attention_dict['mut_to_cna_cross_attention']
                if mut_to_cna_attn:  # List of attention scores from each block
                    self.model_attn_mut_to_cna = tf.keras.Model(
                        inputs=inputs,
                        outputs=mut_to_cna_attn,  # List of tensors, one per block
                        name='mut_to_cna_cross_attention_model'
                    )

            # 5. CNA-to-mutation cross-modal attention
            if self.use_mut and self.use_cna and 'cna_to_mut_cross_attention' in attention_dict:
                cna_to_mut_attn = attention_dict['cna_to_mut_cross_attention']
                if cna_to_mut_attn:  # List of attention scores from each block
                    self.model_attn_cna_to_mut = tf.keras.Model(
                        inputs=inputs,
                        outputs=cna_to_mut_attn,  # List of tensors, one per block
                        name='cna_to_mut_cross_attention_model'
                    )

            # Create separate features models using the SAME concatenated features
            # Determine which inputs to use based on whether cross-attention is enabled
            from tessera.input_keys import get_input_keys

            has_cross_attention = self.use_mut and self.use_cna and cross_modal_blocks > 0

            # Mutation features model (reuses mut_features_concat created above)
            if mut_features_concat is not None:
                if has_cross_attention:
                    # With cross-attention, mutation features depend on both modalities
                    mut_input_keys = get_input_keys(
                        use_mut=self.use_mut,
                        use_vaf=self.use_vaf,
                        use_cna=self.use_cna,
                        use_cna_loh=self.use_cna_loh
                    )
                else:
                    # Without cross-attention, only use mutation-related inputs
                    mut_input_keys = get_input_keys(
                        use_mut=self.use_mut,
                        use_vaf=self.use_vaf,
                        use_cna=False
                    )

                mut_model_inputs = {k: v for k, v in inputs.items() if k in mut_input_keys}

                # Enrich mutation features with any per-instance InfoNCE
                # projections that were created above. Both the general-InfoNCE
                # path and the token-bag path can each contribute a per-instance
                # tensor; when both are enabled they share the same tensor
                # (alias), so we dedupe by identity to avoid double-concat.
                # Output dim: backbone + k*projection_dim where k is the number
                # of *distinct* per-instance projection tensors that exist.
                mut_enrichment_keys = ['mut_infonce_token_embedding', 'mut_token_embedding']
                mut_enrichments = []
                for _k in mut_enrichment_keys:
                    if _k in model_outputs:
                        _t = model_outputs[_k]
                        if not any(_t is _existing for _existing in mut_enrichments):
                            mut_enrichments.append(_t)
                if mut_enrichments:
                    mut_features_output = tf.keras.layers.Concatenate(
                        axis=-1,
                        name='mut_features_enriched'
                    )([mut_features_concat] + mut_enrichments)
                else:
                    mut_features_output = mut_features_concat

                self.model_features_mut = tf.keras.Model(
                    inputs=mut_model_inputs,
                    outputs=mut_features_output,
                    name='mutation_features_model'
                )

            # CNA features model (reuses cna_features_concat created above)
            if cna_features_concat is not None:
                if has_cross_attention:
                    # With cross-attention, CNA features depend on both modalities
                    cna_input_keys = get_input_keys(
                        use_mut=self.use_mut,
                        use_vaf=self.use_vaf,
                        use_cna=self.use_cna,
                        use_cna_loh=self.use_cna_loh
                    )
                else:
                    # Without cross-attention, only use CNA-related inputs
                    cna_input_keys = get_input_keys(
                        use_mut=False,
                        use_vaf=False,
                        use_cna=self.use_cna,
                        use_cna_loh=self.use_cna_loh
                    )

                cna_model_inputs = {k: v for k, v in inputs.items() if k in cna_input_keys}

                # Enrich CNA features with any per-instance InfoNCE projections
                # that were created above — see mutation branch for rationale.
                cna_enrichment_keys = ['cna_infonce_token_embedding', 'cna_token_embedding']
                cna_enrichments = []
                for _k in cna_enrichment_keys:
                    if _k in model_outputs:
                        _t = model_outputs[_k]
                        if not any(_t is _existing for _existing in cna_enrichments):
                            cna_enrichments.append(_t)
                if cna_enrichments:
                    cna_features_output = tf.keras.layers.Concatenate(
                        axis=-1,
                        name='cna_features_enriched'
                    )([cna_features_concat] + cna_enrichments)
                else:
                    cna_features_output = cna_features_concat

                self.model_features_cna = tf.keras.Model(
                    inputs=cna_model_inputs,
                    outputs=cna_features_output,
                    name='cna_features_model'
                )

            # Wrap model with CustomTrainingModel which handles loss computation internally
            # Loss logic is determined by use_mut and use_cna flags
            self.model = CustomTrainingModel(
                self.model,
                use_mut=self.use_mut,
                use_cna=self.use_cna,
                train_on_mutation_loss=train_on_mutation_loss,
                train_on_cna_loss=train_on_cna_loss,
                loss_non_zero_only=loss_non_zero_only,
                hinge_loss_t=hinge_loss_t,
                per_sample_loss=per_sample_loss,
                accuracy_non_zero_only=accuracy_non_zero_only,
                model_dir=self.model_dir,
                log_gradients=log_gradients,
                # CNA dual-task parameters
                predict_cna_loh=self.predict_cna_loh,
                # InfoNCE loss parameters
                use_infonce_loss=self.use_infonce_loss,
                infonce_temperature=self.infonce_temperature,
                infonce_loss_weight=self.infonce_loss_weight,
                # Token-bag InfoNCE parameters (independent per modality)
                use_mut_token_bag_infonce=self.use_mut_token_bag_infonce,
                use_cna_token_bag_infonce=self.use_cna_token_bag_infonce,
                token_bag_temperature=self.token_bag_temperature,
                token_bag_loss_weight=self.token_bag_loss_weight
            )

            # Store learning rate parameters for later use in train_model
            self.max_learning_rate = max_learning_rate
            self.warmup_steps = warmup_steps
            self.min_learning_rate = min_learning_rate
            self.use_cosine_schedule = use_cosine_schedule

            # Don't compile here - will be done in train_model with proper learning rate schedule
            # This avoids double compilation which is expensive

    def _create_callbacks(self, early_stopping_patience, epochs_min, min_relative_delta=0.0):
        """
        Create callbacks for model training.

        Args:
            early_stopping_patience (int): Number of epochs with no improvement after which training will be stopped
            epochs_min (int or None): Minimum number of epochs to train before early stopping can trigger

        Returns:
            list: List of Keras callbacks [checkpoint_callback, checkpoint_callback_h5, early_stopping, csv_logger]
        """
        has_validation = hasattr(self, 'valid_dataset')
        monitor = 'val_loss' if has_validation else 'loss'

        # Create checkpoint callback (h5 callback removed - layers are shared so
        # EarlyStopping(restore_best_weights=True) restores to all models)
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(self.model_dir, 'best_model.keras'),
            save_best_only=True,
            monitor=monitor,
            mode='min',
            verbose=1,
        )

        # Create early stopping callback
        if epochs_min is not None:
            early_stopping = CustomEarlyStopping(
                monitor=monitor,
                patience=early_stopping_patience,
                mode='min',
                verbose=1,
                restore_best_weights=True,
                min_epochs=epochs_min,
                min_relative_delta=min_relative_delta
            )
        else:
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor=monitor,
                patience=early_stopping_patience,
                mode='min',
                verbose=1,
                restore_best_weights=True
            )

        # TensorToFloatCallback must come BEFORE CSVLogger to convert tensor metrics to floats
        # This fixes "Cannot iterate over a scalar tensor" error with custom metrics like cna_pearson
        tensor_to_float = TensorToFloatCallback()

        # CSV logger to save training metrics
        csv_logger = tf.keras.callbacks.CSVLogger(
            filename=os.path.join(self.model_dir, 'training_log.csv'),
            separator=',',
            append=False
        )

        return [checkpoint_callback, early_stopping, tensor_to_float, csv_logger]

    def _create_optimizer(self, learning_rate=None, epochs_min=None, steps_per_epoch=None):
        """
        Create optimizer with optional learning rate schedule.

        Args:
            learning_rate (float or None): Learning rate to use. If None, uses self.max_learning_rate
            epochs_min (int or None): Minimum epochs for calculating total training steps (for schedule)
            steps_per_epoch (int or None): Steps per epoch for calculating total training steps (for schedule)

        Returns:
            tf.keras.optimizers.Optimizer: Configured optimizer (AdamW with optional mixed precision wrapping)
        """
        lr_to_use = learning_rate if learning_rate is not None else self.max_learning_rate

        if hasattr(self, 'use_cosine_schedule') and self.use_cosine_schedule:
            # Calculate total_steps based on realistic training duration
            if epochs_min is not None and steps_per_epoch is not None:
                total_steps = epochs_min * steps_per_epoch
            elif epochs_min is not None:
                total_steps = epochs_min * 100  # default steps_per_epoch
            elif steps_per_epoch is not None:
                total_steps = 1000 * steps_per_epoch  # reasonable default epochs
            else:
                total_steps = 100000  # conservative default: 1000 epochs * 100 steps

            lr_schedule = CosineAnnealingWithWarmup(
                max_learning_rate=lr_to_use,
                total_steps=total_steps,
                warmup_steps=self.warmup_steps,
                min_learning_rate=self.min_learning_rate
            )

            # Use AdamW optimizer with weight decay for better large model training
            opt = tf.keras.optimizers.AdamW(
                learning_rate=lr_schedule,
                beta_1=0.9,
                beta_2=0.97,
                weight_decay=0.001,  # Add weight decay for regularization
                clipnorm=10.0  # Gradient clipping by norm for stability
            )
            if learning_rate is None:  # Only print on initial creation
                print(f"Using cosine annealing schedule: max_lr={lr_to_use}, "
                      f"warmup_steps={self.warmup_steps}, total_steps={total_steps}")
        else:
            # Use AdamW with fixed learning rate
            opt = tf.keras.optimizers.AdamW(
                learning_rate=lr_to_use,
                beta_1=0.9,
                beta_2=0.97,
                weight_decay=0.001,
                clipnorm=10.0,
            )

        # Wrap optimizer with loss scaling for mixed precision training if needed
        if self.mixed_precision:
            opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)
            if learning_rate is None:  # Only print on initial creation
                print("Mixed precision enabled with automatic loss scaling")

        return opt

    def _save_trained_models(self):
        """
        Save trained models conditionally based on configuration.

        This method saves all models directly without explicit weight transfer because:
        - All sub-models (model_features_cna, model_attn_*, etc.) share layers with model_inf
        - EarlyStopping(restore_best_weights=True) automatically restores best weights
        - Since layers are shared by reference, all models get the restored weights automatically

        Conditionally saves only relevant models based on configuration:
           - Mutation models only if use_mut=True
           - CNA models only if use_cna=True
           - Cross-modal models only if both use_mut=True and use_cna=True
           - Attention models only if they were created (variant_local_attention, variant_self_attention, etc.)
        """
        # Re-save model configuration with updated flags
        # This is important because flags like use_vaf, use_cna_loh are set during create_sample_dataset()
        # which happens AFTER build_model() initially saves the config
        self._save_model_config()

        # Note: EarlyStopping(restore_best_weights=True) already restored best weights to model_inf.
        # Since all sub-models share layers with model_inf by reference, they already have best weights.
        print("Saving models with best weights (restored by EarlyStopping)...")

        # Always save the main inference model
        self.model_inf.save(os.path.join(self.model_dir, 'final_model.keras'))
        print("  ✓ Saved main inference model")

        # Save mutation-related models only if mutations are being used
        if self.use_mut:
            # Save mutation local attention model if enabled
            if self.variant_local_attention and hasattr(self, 'model_attn_local'):
                self.model_attn_local.save(os.path.join(self.model_dir, 'attn_local_model.keras'))
                print("  ✓ Saved mutation local attention model")

            # Save legacy mutation self-attention model if enabled
            if self.variant_self_attention and hasattr(self, 'model_attn'):
                self.model_attn.save(os.path.join(self.model_dir, 'attn_model.keras'))
                print("  ✓ Saved mutation self-attention model (legacy)")

            # Save mutation self-attention model if enabled
            if self.variant_self_attention and hasattr(self, 'model_attn_mut_self'):
                self.model_attn_mut_self.save(os.path.join(self.model_dir, 'attn_mut_self_model.keras'))
                print("  ✓ Saved mutation self-attention model")

            # Save mutation features model if it exists
            if hasattr(self, 'model_features_mut'):
                self.model_features_mut.save(os.path.join(self.model_dir, 'features_model_mut.keras'))
                print("  ✓ Saved mutation features model")

        # Save CNA-related models only if CNAs are being used
        if self.use_cna:
            # Save CNA self-attention model if it exists
            if hasattr(self, 'model_attn_cna_self'):
                self.model_attn_cna_self.save(os.path.join(self.model_dir, 'attn_cna_self_model.keras'))
                print("  ✓ Saved CNA self-attention model")

            # Save CNA features model if it exists
            if hasattr(self, 'model_features_cna'):
                self.model_features_cna.save(os.path.join(self.model_dir, 'features_model_cna.keras'))
                print("  ✓ Saved CNA features model")

        # Save cross-modal attention models only if both mutations and CNAs are being used
        if self.use_mut and self.use_cna:
            # Save mutation-to-CNA cross-modal attention model if it exists
            if hasattr(self, 'model_attn_mut_to_cna'):
                self.model_attn_mut_to_cna.save(os.path.join(self.model_dir, 'attn_mut_to_cna_model.keras'))
                print("  ✓ Saved mutation→CNA cross-attention model")

            # Save CNA-to-mutation cross-modal attention model if it exists
            if hasattr(self, 'model_attn_cna_to_mut'):
                self.model_attn_cna_to_mut.save(os.path.join(self.model_dir, 'attn_cna_to_mut_model.keras'))
                print("  ✓ Saved CNA→mutation cross-attention model")

    def train_model(self, epochs=10, steps_per_epoch=None,validation_steps=None,early_stopping_patience=5,epochs_min=None,
                    validation_freq=1, reset_optimizer_epoch=None, reset_learning_rate=None,prefetch=tf.data.AUTOTUNE,steps_per_execution=1,
                    early_stopping_min_relative_delta=0.0):

        # Determine training and validation data
        if hasattr(self,'train_dataset'):
            train_data = self.train_dataset
        else:
            train_data = self.dataset

        if hasattr(self,'valid_dataset'):
            validation_data = self.valid_dataset
        else:
            validation_data = None

        with self.strategy.scope():
            # Create initial optimizer
            optimizer = self._create_optimizer(
                learning_rate=None,
                epochs_min=epochs_min,
                steps_per_epoch=steps_per_epoch
            )
                
            # Single compilation here with final optimizer
            self.model.compile(
                optimizer=optimizer,
                jit_compile=self.jit_compile,
                steps_per_execution=steps_per_execution
            )
            
            # Reset gradient logging data for fresh training session
            if hasattr(self.model, 'reset_gradient_logging'):
                self.model.reset_gradient_logging()
            
            keys = get_input_keys(use_mut=self.use_mut, use_vaf=self.use_vaf, use_cna=self.use_cna,
                                  use_cna_loh=self.use_cna_loh)

            # Optimize training data pipeline — map first, prefetch last
            train_data = train_data.map(
                lambda x: tessera.data.preprocessing.filter_inputs(x, keys=keys),
                num_parallel_calls=tf.data.AUTOTUNE
            ).prefetch(prefetch)

            if validation_data is not None:
                validation_data = validation_data.map(
                    lambda x: tessera.data.preprocessing.filter_inputs(x, keys=keys),
                    num_parallel_calls=tf.data.AUTOTUNE
                ).prefetch(prefetch)

            # Create callbacks
            callbacks_list = self._create_callbacks(early_stopping_patience, epochs_min, min_relative_delta=early_stopping_min_relative_delta)
            
            # Add optimizer reset callback if requested
            if reset_optimizer_epoch is not None:
                # Create optimizer factory with reset learning rate if provided
                def optimizer_factory():
                    return self._create_optimizer(
                        learning_rate=reset_learning_rate,
                        epochs_min=epochs_min,
                        steps_per_epoch=steps_per_epoch
                    )
                
                optimizer_reset_callback = OptimizerResetCallback(
                    reset_epoch=reset_optimizer_epoch,
                    optimizer_factory=optimizer_factory,
                    verbose=1
                )
                callbacks_list.append(optimizer_reset_callback)
                print(f"Optimizer will be reset at epoch {reset_optimizer_epoch + 1}")
                if reset_learning_rate is not None:
                    print(f"Reset learning rate: {reset_learning_rate}")
                else:
                    print(f"Reset learning rate: {self.max_learning_rate} (same as initial)")

            # Train the model
            history = self.model.fit(
            train_data,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=callbacks_list,
            verbose=1,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            validation_freq=validation_freq
            )

        # Save all trained models
        self._save_trained_models()

        return history

    def save_best_model_h5(self):
        """
        Load best model weights and save all trained models.

        This is a convenience method that can be called after training to re-save
        the best model weights. It calls _save_trained_models() to perform the actual saving.
        """
        self._save_trained_models()

    def load_pretrained_for_finetune(self, model_dir):
        """
        Loads all pre-trained sub-models (model_inf, local/global attention models,
        features model) and wraps the main inference model with the custom
        training class so that you can resume fine-tuning.
        """

        # 1. Load the main inference model (this is your final forward-pass model).
        final_model_path = os.path.join(model_dir, 'final_model.keras')
        self.model_inf = tf.keras.models.load_model(final_model_path)
        print(f"Loaded main inference model from {final_model_path}")

        # 2. Load the local attention model (if it was saved).
        attn_local_path = os.path.join(model_dir, 'attn_local_model.keras')
        if os.path.exists(attn_local_path):
            self.model_attn_local = tf.keras.models.load_model(attn_local_path)
            print(f"Loaded local attention model from {attn_local_path}")
        else:
            self.model_attn_local = None
            print("Local attention model not found. Skipping...")

        # 3. Load the global attention model (if variant_self_attention is True).
        attn_model_path = os.path.join(model_dir, 'attn_model.keras')
        if os.path.exists(attn_model_path):
            self.model_attn = tf.keras.models.load_model(attn_model_path)
            print(f"Loaded global attention model from {attn_model_path}")
        else:
            self.model_attn = None
            print("Global attention model not found or variant_self_attention=False. Skipping...")

        # 3b. Load separate attention models for each attention type
        # Mutation self-attention model
        attn_mut_self_path = os.path.join(model_dir, 'attn_mut_self_model.keras')
        if os.path.exists(attn_mut_self_path):
            self.model_attn_mut_self = tf.keras.models.load_model(attn_mut_self_path)
            print(f"Loaded mutation self-attention model from {attn_mut_self_path}")
        else:
            self.model_attn_mut_self = None
            print("Mutation self-attention model not found. Skipping...")

        # CNA self-attention model
        attn_cna_self_path = os.path.join(model_dir, 'attn_cna_self_model.keras')
        if os.path.exists(attn_cna_self_path):
            self.model_attn_cna_self = tf.keras.models.load_model(attn_cna_self_path)
            print(f"Loaded CNA self-attention model from {attn_cna_self_path}")
        else:
            self.model_attn_cna_self = None
            print("CNA self-attention model not found. Skipping...")

        # Mutation-to-CNA cross-modal attention model
        attn_mut_to_cna_path = os.path.join(model_dir, 'attn_mut_to_cna_model.keras')
        if os.path.exists(attn_mut_to_cna_path):
            self.model_attn_mut_to_cna = tf.keras.models.load_model(attn_mut_to_cna_path)
            print(f"Loaded mutation-to-CNA cross-modal attention model from {attn_mut_to_cna_path}")
        else:
            self.model_attn_mut_to_cna = None
            print("Mutation-to-CNA cross-modal attention model not found. Skipping...")

        # CNA-to-mutation cross-modal attention model
        attn_cna_to_mut_path = os.path.join(model_dir, 'attn_cna_to_mut_model.keras')
        if os.path.exists(attn_cna_to_mut_path):
            self.model_attn_cna_to_mut = tf.keras.models.load_model(attn_cna_to_mut_path)
            print(f"Loaded CNA-to-mutation cross-modal attention model from {attn_cna_to_mut_path}")
        else:
            self.model_attn_cna_to_mut = None
            print("CNA-to-mutation cross-modal attention model not found. Skipping...")

        # 4. Load the separate features models for mutations and CNAs.
        # Mutation features model
        features_model_mut_path = os.path.join(model_dir, 'features_model_mut.keras')
        if os.path.exists(features_model_mut_path):
            self.model_features_mut = tf.keras.models.load_model(features_model_mut_path)
            print(f"Loaded mutation features model from {features_model_mut_path}")
        else:
            self.model_features_mut = None
            print("Mutation features model not found. Skipping...")

        # CNA features model
        features_model_cna_path = os.path.join(model_dir, 'features_model_cna.keras')
        if os.path.exists(features_model_cna_path):
            self.model_features_cna = tf.keras.models.load_model(features_model_cna_path)
            print(f"Loaded CNA features model from {features_model_cna_path}")
        else:
            self.model_features_cna = None
            print("CNA features model not found. Skipping...")

        # 5. Wrap the inference model in the CustomTrainingModel again,
        #    so you can continue training with the same logic used in build_model().
        #    NOTE: If you have a strategy scope, place these lines inside that scope as well.
        #    CustomTrainingModel handles loss computation internally based on use_mut/use_cna flags
        self.model = CustomTrainingModel(
            self.model_inf,
            use_mut=self.use_mut,
            use_cna=self.use_cna,
            loss_non_zero_only=False,
            hinge_loss_t=None,
            per_sample_loss=False,
            accuracy_non_zero_only=False,
            model_dir=self.model_dir,
            log_gradients=True,
            # CNA dual-task parameters
            predict_cna_loh=self.predict_cna_loh,
            # InfoNCE loss parameters
            use_infonce_loss=self.use_infonce_loss,
            infonce_temperature=self.infonce_temperature,
            infonce_loss_weight=self.infonce_loss_weight,
            # Token-bag InfoNCE parameters (independent per modality)
            use_mut_token_bag_infonce=self.use_mut_token_bag_infonce,
            use_cna_token_bag_infonce=self.use_cna_token_bag_infonce,
            token_bag_temperature=self.token_bag_temperature,
            token_bag_loss_weight=self.token_bag_loss_weight
        )

        # 6. Compile the newly wrapped model with the same optimizer settings used in build_model().
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0))

        # 7. (Optional) If you want to ensure you’re loading the exact optimizer state
        #    from the final checkpoint, you’ll need to load from a saved Keras checkpoint
        #    that includes the optimizer weights. (Simply calling `load_model` on
        #    final_model.keras might not preserve the optimizer state by default.)
        #
        #    If you have a checkpoint callback that saved the optimizer state,
        #    you could do something like:
        #
        #       self.model.load_weights(os.path.join(model_dir, 'best_model.keras'))
        #
        #    which (depending on how it was saved) may load the optimizer as well.

        print("All sub-models loaded and the custom training model is re-wrapped and compiled. "
              "You can now resume fine-tuning.")

    def _build_features_model_from_trained(self):
        """Build features model from trained model by extracting named layers."""

        if not hasattr(self, 'model_inf'):
            raise ValueError("Model must be trained first")

        # Extract the named layer outputs
        layer_outputs = []

        # Extract local attention block outputs
        if self.variant_local_attention:
            for i in range(self.local_attention_blocks):
                layer_name = f'local_block_{i}_output'
                try:
                    layer_output = self.model_inf.get_layer(layer_name).output
                    # Squeeze singleton dimension if present (axis=2)
                    if len(layer_output.shape) == 4:
                        squeezed_output = tf.squeeze(layer_output, axis=2)
                    else:
                        squeezed_output = layer_output
                    layer_outputs.append(squeezed_output)
                except ValueError:
                    print(f"Warning: Layer {layer_name} not found")

        # Extract global attention block outputs if they exist
        if self.variant_self_attention:
            for i in range(self.global_attention_blocks):
                layer_name = f'global_block_{i}_output'
                try:
                    layer_output = self.model_inf.get_layer(layer_name).output
                    layer_outputs.append(layer_output)
                except ValueError:
                    print(f"Warning: Layer {layer_name} not found")

        # Concatenate all layer outputs
        if len(layer_outputs) > 1:
            features_output = tf.keras.layers.Concatenate(axis=-1)(layer_outputs)
        else:
            features_output = layer_outputs[0] if layer_outputs else None

        if features_output is not None:
            # Build the features model
            self.model_features = tf.keras.Model(
                inputs=self.model_inf.input,
                outputs=features_output
            )
            print(f"Built features model with {len(layer_outputs)} layer outputs")
        else:
            print("Warning: No valid layer outputs found")

    def get_variant_features(self, dataset_name, downcast=False):
        with self.strategy.scope():
            # Load model configuration if cross_modal_blocks is not set
            # This is necessary when loading a pre-trained model for feature extraction
            if not hasattr(self, 'cross_modal_blocks'):
                self._load_model_config_if_needed()

            # Only load from disk if model doesn't exist in memory
            if not hasattr(self, 'model_features_mut') or self.model_features_mut is None:
                model_path = os.path.join(self.model_dir, 'features_model_mut.keras')
                if not os.path.exists(model_path):
                    raise ValueError(f"Mutation features model not found at {model_path}. Please train the model first using build_model().")
                self.model_features_mut = tf.keras.models.load_model(model_path, compile=False)

            # Ensure the dataset exists
            if not hasattr(self, dataset_name):
                raise ValueError(f"Dataset '{dataset_name}' not found. Please create it first using create_dataset method.")

            dataset = getattr(self, dataset_name)
            from tessera.input_keys import get_input_keys

            has_cross_attention = self.use_mut and self.use_cna and self.cross_modal_blocks > 0

            if has_cross_attention:
                keys = get_input_keys(use_mut=self.use_mut, use_vaf=self.use_vaf, use_cna=self.use_cna,
                                      use_cna_loh=self.use_cna_loh)
            else:
                keys = get_input_keys(use_mut=self.use_mut, use_vaf=self.use_vaf, use_cna=False,
                                      use_cna_loh=False)

            # Combined dataset: yields (filtered_inputs, idx) in one pass so we can do
            # inline unpadding during inference without a second dataset iteration.
            def _add_idx(batch):
                filtered = tessera.data.preprocessing.filter_inputs(batch, keys=keys)
                idx = batch.get('idx', tf.constant([], dtype=tf.int32))
                return filtered, idx

            dataset_combined = dataset.map(_add_idx, num_parallel_calls=tf.data.AUTOTUNE)
            dataset_combined = dataset_combined.prefetch(tf.data.AUTOTUNE)

            # Iterate batch-by-batch to avoid GPU OOM from predict() concatenating
            # the full output on-device.  Use predict() (not model()) per batch to
            # preserve the same compiled graph that was used during training — calling
            # model() directly uses a different XLA path that breaks FlattenLastTwoDims.
            # For 3-D outputs (token-level), unpad inline so we never hold the padded
            # (n_samples × max_tokens × feature_dim) cube in RAM — only valid tokens.
            all_features = []
            all_indices = []
            is_3d = None
            dtype = np.float16 if downcast else np.float32

            for batch_inputs, batch_idx in dataset_combined:
                _bf = self.model_features_mut.predict(
                    tf.data.Dataset.from_tensors(batch_inputs), verbose=0
                )
                if isinstance(_bf, list) and len(_bf) > 1:
                    _bf = np.concatenate(_bf, axis=-1)
                _bf_np = np.asarray(_bf).astype(dtype)

                if is_3d is None:
                    is_3d = (_bf_np.ndim == 3)

                if is_3d:
                    indices = batch_idx.numpy()  # (batch_size, max_tokens)
                    for j in range(_bf_np.shape[0]):
                        mask = indices[j] != -1
                        all_features.append(_bf_np[j, mask])
                        all_indices.append(indices[j, mask])
                else:
                    all_features.append(_bf_np)

            out = np.concatenate(all_features, axis=0)
            if is_3d:
                flat_indices = np.concatenate(all_indices)
                out = out[np.argsort(flat_indices)]

        return out

    def get_variant_probabilities(self, dataset_name, return_logits=False, return_true_values=False, return_loss=False, non_zero_only=False, return_ref=False):
        """
        Gets predicted probabilities (and optionally logits, true values, and per-variant loss) for each variant in the original input order.

        Args:
            dataset_name: Name of the dataset to process
            return_logits: If True, return both probabilities and logits. If False, return only probabilities.
            return_true_values: If True, return true values along with predictions
            return_loss: If True, return per-variant loss values
            non_zero_only: If True and return_loss=True, compute loss only on non-zero positions
            return_ref: If True and recon_ref was used, return reference probabilities/logits as well

        Returns:
            If all flags are False:
                Array of probabilities in original variant order
            Otherwise:
                Tuple containing requested outputs in order:
                (probabilities, logits?, true_values?, loss?, probs_ref?, logits_ref?, true_values_ref?)

                When recon_ref=True and return_ref=True, reference data is appended at the end:
                - probs_ref: Reference sequence probabilities (if return_ref=True)
                - logits_ref: Reference sequence logits (if return_ref=True and return_logits=True)
                - true_values_ref: Reference sequence true values (if return_ref=True and return_true_values=True)
        """
        # Load model configuration to check if recon_ref was used
        import json
        config_path = os.path.join(self.model_dir, 'model_config.json')
        recon_ref = False
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                recon_ref = config.get('recon_ref', False)

        with self.strategy.scope():
            self.model = tf.keras.models.load_model(os.path.join(self.model_dir, 'final_model.keras'), compile=False)

            if not hasattr(self, dataset_name):
                raise ValueError(f"Dataset '{dataset_name}' not found. Please create it first using create_dataset method.")

            dataset = getattr(self, dataset_name)
            from tessera.input_keys import get_input_keys
            keys = get_input_keys(use_mut=self.use_mut, use_vaf=self.use_vaf, use_cna=self.use_cna,
                                  use_cna_loh=self.use_cna_loh)
            dataset_inf = dataset.map(
                lambda x: tessera.data.preprocessing.filter_inputs(x, keys=keys),
                num_parallel_calls=tf.data.AUTOTUNE
            )
            dataset_inf = dataset_inf.prefetch(tf.data.AUTOTUNE)

            outputs = self.model.predict(dataset_inf)

        # Handle dual reconstruction outputs
        logits = outputs['logits']
        probabilities = tf.nn.softmax(logits, axis=-1).numpy()

        # Extract reference logits and probabilities if available
        logits_ref = None
        probabilities_ref = None
        if recon_ref and 'logits_ref' in outputs:
            logits_ref = outputs['logits_ref']
            probabilities_ref = tf.nn.softmax(logits_ref, axis=-1).numpy()

        dataset = list(dataset)

        all_indices = []
        all_probs = []
        all_logits = [] if return_logits else None
        all_true_values = [] if return_true_values else None
        all_losses = [] if return_loss else None

        # Reference data tracking (only if recon_ref was used and requested)
        all_probs_ref = [] if return_ref and recon_ref and probabilities_ref is not None else None
        all_logits_ref = [] if return_ref and return_logits and recon_ref and logits_ref is not None else None
        all_true_values_ref = [] if return_ref and return_true_values and recon_ref else None
        pos = 0

        for batch in dataset:
            batch_indices = batch['idx'].numpy()
            if return_true_values or return_loss:
                batch_true = batch['alt'].numpy()
            if return_ref and return_true_values and recon_ref:
                batch_true_ref = batch['ref'].numpy()

            for i, sample_indices in enumerate(batch_indices):
                mask = sample_indices != -1
                valid_length = np.sum(mask)

                sample_probs = probabilities[pos][:valid_length]
                if return_logits:
                    sample_logits = logits[pos][:valid_length]
                if return_true_values:
                    sample_true = batch_true[i][:valid_length]

                # Reference data extraction (if available and requested)
                if all_probs_ref is not None:
                    sample_probs_ref = probabilities_ref[pos][:valid_length]
                if all_logits_ref is not None:
                    sample_logits_ref = logits_ref[pos][:valid_length]
                if all_true_values_ref is not None:
                    sample_true_ref = batch_true_ref[i][:valid_length]
                if return_loss:
                    if recon_ref and logits_ref is not None:
                        # Dual reconstruction: compute both alt and ref losses
                        sample_true_alt = batch_true[i][:valid_length]
                        sample_logits_alt = logits[pos][:valid_length]
                        sample_true_ref = batch_true_ref[i][:valid_length]
                        sample_logits_ref_loss = logits_ref[pos][:valid_length]

                        # Calculate alt loss
                        alt_loss = tf.keras.losses.sparse_categorical_crossentropy(
                            sample_true_alt, sample_logits_alt, from_logits=True
                        ).numpy()

                        # Calculate ref loss
                        ref_loss = tf.keras.losses.sparse_categorical_crossentropy(
                            sample_true_ref, sample_logits_ref_loss, from_logits=True
                        ).numpy()

                        # Combined loss (average of alt and ref)
                        position_loss = (alt_loss + ref_loss) / 2.0
                    else:
                        # Standard single reconstruction
                        sample_true_loss = batch_true[i][:valid_length]
                        sample_logits_loss = logits[pos][:valid_length]

                        # Calculate per-variant loss using sparse categorical crossentropy
                        position_loss = tf.keras.losses.sparse_categorical_crossentropy(
                            sample_true_loss, sample_logits_loss, from_logits=True
                        ).numpy()
                    
                    if non_zero_only:
                        # Only compute loss on non-zero positions for each variant
                        # Use alt sequence for masking regardless of reconstruction mode
                        sample_true_for_mask = sample_true_alt if recon_ref and logits_ref is not None else sample_true_loss
                        non_zero_mask = sample_true_for_mask != 0
                        # Apply mask and compute mean per variant, handling variants with no non-zero positions
                        masked_loss = np.where(non_zero_mask, position_loss, np.nan)
                        sample_loss = np.nanmean(masked_loss, axis=-1)
                    else:
                        # Average loss over all positions (axis=-1 averages across positions for each variant)
                        sample_loss = np.mean(position_loss, axis=-1)

                all_indices.append(sample_indices[:valid_length])
                all_probs.append(sample_probs)
                if return_logits:
                    all_logits.append(sample_logits)
                if return_true_values:
                    all_true_values.append(sample_true)
                if return_loss:
                    all_losses.append(sample_loss)

                # Append reference data if available
                if all_probs_ref is not None:
                    all_probs_ref.append(sample_probs_ref)
                if all_logits_ref is not None:
                    all_logits_ref.append(sample_logits_ref)
                if all_true_values_ref is not None:
                    all_true_values_ref.append(sample_true_ref)

                pos += 1

        all_indices = np.concatenate(all_indices)
        all_probs = np.concatenate(all_probs)
        if return_logits:
            all_logits = np.concatenate(all_logits)
        if return_true_values:
            all_true_values = np.concatenate(all_true_values)
        if return_loss:
            all_losses = np.concatenate(all_losses)

        # Concatenate reference data if available
        if all_probs_ref is not None:
            all_probs_ref = np.concatenate(all_probs_ref)
        if all_logits_ref is not None:
            all_logits_ref = np.concatenate(all_logits_ref)
        if all_true_values_ref is not None:
            all_true_values_ref = np.concatenate(all_true_values_ref)

        sorted_idx = np.argsort(all_indices)

        # Build return tuple dynamically based on requested outputs
        results = [all_probs[sorted_idx]]
        
        if return_logits:
            results.append(all_logits[sorted_idx])
        if return_true_values:
            results.append(all_true_values[sorted_idx])
        if return_loss:
            results.append(all_losses[sorted_idx])

        # Add reference data if available and requested
        if all_probs_ref is not None:
            results.append(all_probs_ref[sorted_idx])
        if all_logits_ref is not None:
            results.append(all_logits_ref[sorted_idx])
        if all_true_values_ref is not None:
            results.append(all_true_values_ref[sorted_idx])

        # Return single array if only probabilities requested, otherwise return tuple
        return results[0] if len(results) == 1 else tuple(results)

    def assess_mask_predictions(self, dataset_name='test_dataset', num_batches=None, non_zero_only=True):
        """
        Vectorized version of mask prediction assessment.
        
        Args:
            dataset_name: Name of dataset to evaluate
            num_batches: Number of batches to process (None for all)
            non_zero_only: If True, only evaluate non-zero positions; if False, include padding
        """
        with self.strategy.scope():
            self.model = tf.keras.models.load_model(os.path.join(self.model_dir, 'final_model.keras'), compile=False)

            if not hasattr(self, dataset_name):
                raise ValueError(f"Dataset '{dataset_name}' not found")

            dataset = getattr(self, dataset_name)
            from tessera.input_keys import get_input_keys
            keys = get_input_keys(use_mut=self.use_mut, use_vaf=self.use_vaf, use_cna=self.use_cna,
                                  use_cna_loh=self.use_cna_loh)
            dataset_inf = dataset.map(
                lambda x: tessera.data.preprocessing.filter_inputs(x, keys=keys),
                num_parallel_calls=tf.data.AUTOTUNE
            )
            dataset_inf = dataset_inf.prefetch(tf.data.AUTOTUNE)

        total_correct = 0
        total_positions = 0
        position_correct = np.zeros(self.alt_len)
        position_total = np.zeros(self.alt_len)
        variants_correct = 0
        total_variants = 0

        for batch_idx, batch in enumerate(dataset_inf):
            if num_batches and batch_idx >= num_batches:
                break

            outputs = self.model(batch, training=False)
            logits = outputs['logits'].numpy()
            y_true = batch['alt'].numpy()  # (batch_size, num_variants, alt_len)
            predictions = np.argmax(logits, axis=-1)  # (batch_size, num_variants, alt_len)

            # Identify non-padded variants
            non_padding = np.any(y_true != 0, axis=2)  # (batch_size, num_variants)

            # Extract only non-padded variants
            flat_true = y_true[non_padding]  # (num_valid_variants, alt_len)
            flat_pred = predictions[non_padding]  # (num_valid_variants, alt_len)

            if non_zero_only:
                # Only evaluate non-zero positions
                valid_positions = (flat_true != 0)  # (num_valid_variants, alt_len)
                
                # Position-wise metrics
                correct_at_pos = (flat_true == flat_pred) & valid_positions  # Only count non-zero positions
                position_correct += np.sum(correct_at_pos, axis=0)  # (alt_len,)
                position_total += np.sum(valid_positions, axis=0)  # (alt_len,)
                
                # Variant-level metrics: all non-zero positions must be correct
                variant_all_correct = np.all(correct_at_pos, axis=1)  # (num_valid_variants,)
                
                # Overall metrics
                total_correct += np.sum(correct_at_pos)
                total_positions += np.sum(valid_positions)
            else:
                # Evaluate all positions (including padding)
                correct_at_pos = (flat_true == flat_pred)  # All positions
                position_correct += np.sum(correct_at_pos, axis=0)  # (alt_len,)
                position_total += flat_true.shape[0]  # Count all positions for each alt_len position
                
                # Variant-level metrics: all positions (including padding) must be correct
                variant_all_correct = np.all(correct_at_pos, axis=1)  # (num_valid_variants,)
                
                # Overall metrics
                total_correct += np.sum(correct_at_pos)
                total_positions += flat_true.size  # Total number of positions (including padding)
                
            variants_correct += np.sum(variant_all_correct)
            total_variants += len(flat_true)

            # Calculate final metrics
        overall_accuracy = total_correct / total_positions if total_positions > 0 else 0
        variant_accuracy = variants_correct / total_variants if total_variants > 0 else 0
        position_accuracy = np.divide(position_correct, position_total,
                                      out=np.zeros_like(position_correct),
                                      where=position_total != 0)

        return {
            'overall_accuracy': overall_accuracy,
            'variant_accuracy': variant_accuracy,
            'position_accuracy': position_accuracy,
            'total_predictions': total_positions,
            'total_variants': total_variants
        }

    def get_local_attention_weights(self, dataset_name):
        """
        Extract and aggregate local attention weights for each variant, maintaining original order.
        Returns attention weights for 5' and 3' contexts with both average and max attention across heads.

        Args:
            dataset_name (str): Name of dataset to extract attention weights from

        Returns:
            tuple: (attention_weights_5p, attention_weights_3p) where each contains a dictionary
            with 'avg' and 'max' matrices, ordered by original variant index
        """
        with self.strategy.scope():
            self.model_attn_local = tf.keras.models.load_model(os.path.join(self.model_dir, 'attn_local_model.keras'), compile=False)

            # Ensure the dataset exists
            if not hasattr(self, dataset_name):
                raise ValueError(f"Dataset '{dataset_name}' not found. Please create it first using create_dataset method.")

            dataset = getattr(self, dataset_name)
            from tessera.input_keys import get_input_keys
            keys = get_input_keys(use_mut=self.use_mut, use_vaf=self.use_vaf, use_cna=self.use_cna,
                                  use_cna_loh=self.use_cna_loh)
            dataset_inf = dataset.map(
                lambda x: tessera.data.preprocessing.filter_inputs(x, keys=keys),
                num_parallel_calls=tf.data.AUTOTUNE
            )
            dataset_inf = dataset_inf.prefetch(tf.data.AUTOTUNE)

            # Get predictions
            attention_weights_5p, attention_weights_3p = self.model_attn_local.predict(dataset_inf)

        # If the outputs are lists, convert them to numpy arrays by combining across batches
        if isinstance(attention_weights_5p, list):
            attention_weights_5p = np.concatenate(attention_weights_5p, axis=1)
        # Also handle if it's already a tensor
        elif isinstance(attention_weights_5p, tf.Tensor):
            attention_weights_5p = attention_weights_5p.numpy()

        if isinstance(attention_weights_3p, list):
            attention_weights_3p = np.concatenate(attention_weights_3p, axis=1)
        elif isinstance(attention_weights_3p, tf.Tensor):
            attention_weights_3p = attention_weights_3p.numpy()

        # Initialize scalers for normalization
        scaler_list_5p = []
        scaler_list_3p = []

        # Fit scalers for 5' context
        for head_idx in range(attention_weights_5p.shape[1]):
            head_attn = attention_weights_5p[:, head_idx, :, :]
            scaler = MinMaxScaler()
            scaler.fit(head_attn.reshape(-1, 1))
            # check = scaler.transform(head_attn.reshape(-1, 1))
            scaler_list_5p.append(scaler)

        # Fit scalers for 3' context
        for head_idx in range(attention_weights_3p.shape[1]):
            head_attn = attention_weights_3p[:, head_idx, :, :]
            scaler = MinMaxScaler()
            scaler.fit(head_attn.reshape(-1, 1))
            scaler_list_3p.append(scaler)

        def process_attention_weights(dataset, weights_5p, weights_3p):
            # Convert dataset to list first so we don't consume it
            dataset = list(dataset)

            all_indices = []
            all_avg_weights_5p = []
            all_max_weights_5p = []
            all_avg_weights_3p = []
            all_max_weights_3p = []
            pos = 0
            max_values = []

            for batch in dataset:
                batch_indices = batch['idx'].numpy()

                # Process each sample in the batch
                for sample_indices in batch_indices:
                    mask = sample_indices != -1
                    valid_length = np.sum(mask)

                    if valid_length > 0:
                        # Initialize with correct shapes using weights shape info
                        avg_attention_5p = np.zeros((valid_length, weights_5p.shape[-1]))  # Use -1 to get key_length
                        max_attention_5p = np.full((valid_length, weights_5p.shape[-1]), -np.inf)
                        avg_attention_3p = np.zeros((valid_length, weights_3p.shape[-1]))
                        max_attention_3p = np.full((valid_length, weights_3p.shape[-1]), -np.inf)

                        # Process 5' context attention weights across all heads
                        for head_idx in range(weights_5p.shape[1]):
                            head_attention = weights_5p[pos, head_idx]
                            head_attention_valid = head_attention[:valid_length, :]
                            head_attention_valid_flat = head_attention_valid.flatten().reshape(-1, 1)
                            head_attention_valid_scaled_flat = scaler_list_5p[head_idx].transform(
                                head_attention_valid_flat)
                            head_attention_valid_scaled = head_attention_valid_scaled_flat.reshape(
                                head_attention_valid.shape)

                            avg_attention_5p += head_attention_valid_scaled
                            max_attention_5p = np.maximum(max_attention_5p, head_attention_valid_scaled)

                        # Process 3' context attention weights across all heads
                        for head_idx in range(weights_3p.shape[1]):
                            head_attention = weights_3p[pos, head_idx]
                            head_attention_valid = head_attention[:valid_length, :]
                            head_attention_valid_flat = head_attention_valid.flatten().reshape(-1, 1)
                            head_attention_valid_scaled_flat = scaler_list_3p[head_idx].transform(
                                head_attention_valid_flat)
                            head_attention_valid_scaled = head_attention_valid_scaled_flat.reshape(
                                head_attention_valid.shape)
                            max_values.append(np.max(head_attention_valid_scaled))

                            avg_attention_3p += head_attention_valid_scaled
                            max_attention_3p = np.maximum(max_attention_3p, head_attention_valid_scaled)

                        # Average across heads
                        avg_attention_5p /= weights_5p.shape[1]
                        avg_attention_3p /= weights_3p.shape[1]

                        all_indices.append(sample_indices[:valid_length])
                        all_avg_weights_5p.append(avg_attention_5p)
                        all_max_weights_5p.append(max_attention_5p)
                        all_avg_weights_3p.append(avg_attention_3p)
                        all_max_weights_3p.append(max_attention_3p)

                    pos += 1

            # Convert to arrays and concatenate
            all_indices = np.concatenate(all_indices)
            all_avg_weights_5p = np.concatenate(all_avg_weights_5p)
            all_max_weights_5p = np.concatenate(all_max_weights_5p)
            all_avg_weights_3p = np.concatenate(all_avg_weights_3p)
            all_max_weights_3p = np.concatenate(all_max_weights_3p)

            # Sort by original indices
            sorted_idx = np.argsort(all_indices)

            return ({
                        'avg': all_avg_weights_5p[sorted_idx],
                        'max': all_max_weights_5p[sorted_idx]
                    }, {
                        'avg': all_avg_weights_3p[sorted_idx],
                        'max': all_max_weights_3p[sorted_idx]
                    })

        return process_attention_weights(dataset, attention_weights_5p, attention_weights_3p)

    def get_global_attention_weights(self, dataset_name, agg=None):
        """
        Extract global attention weights from the model with flexible aggregation options.

        Args:
            dataset_name (str): Name of the dataset to extract attention weights from
            agg (str or None): Aggregation mode for attention weights. Options:
                - None: No aggregation - return raw attention matrices for each head in each block
                  Output structure: {'sample_id': {'blocks': [{'block_idx': i, 'heads': [matrix_per_head]}], 'variant_indices': [...]}}

                - 'head': Aggregate attention across heads for each block separately
                  Returns avg and max attention matrices per block
                  Output structure: {'sample_id': {'avg_attention_matrices': [per_block], 'max_attention_matrices': [per_block], 'variant_indices': [...]}}

                - 'all': Aggregate attention across all heads and all blocks
                  Returns single avg and max attention matrices
                  Output structure: {'sample_id': {'avg_attention_matrix': matrix, 'max_attention_matrix': matrix, 'variant_indices': [...]}}

        Returns:
            dict: Dictionary with sample IDs as keys, containing attention data in the format
                  specified by the aggregation mode
        """
        with self.strategy.scope():
            self.model_attn = tf.keras.models.load_model(os.path.join(self.model_dir,'attn_model.keras'), compile=False)

            # Ensure the dataset exists
            if not hasattr(self, dataset_name):
                raise ValueError(f"Dataset '{dataset_name}' not found. Please create it first using create_dataset method.")

            dataset = getattr(self, dataset_name)
            from tessera.input_keys import get_input_keys
            keys = get_input_keys(use_mut=self.use_mut, use_vaf=self.use_vaf, use_cna=self.use_cna,
                                  use_cna_loh=self.use_cna_loh)
            dataset_inf = dataset.map(
                lambda x: tessera.data.preprocessing.filter_inputs(x, keys=keys),
                num_parallel_calls=tf.data.AUTOTUNE
            )
            dataset_inf = dataset_inf.prefetch(tf.data.AUTOTUNE)

            attention_blocks,attn_mask = self.model_attn.predict(dataset_inf)
        dataset = list(dataset)

        if isinstance(attn_mask, list):
            attn_mask = attn_mask[0]

        #check if attn_mask is ragged, if so, convert to dense tensor
        if isinstance(attn_mask,tf.RaggedTensor):
            attn_mask = attn_mask.to_tensor(default_value=False)
        #check if attn_mask is tensorflow tensor, if so, convert to numpy
        if isinstance(attn_mask,tf.Tensor):
            attn_mask = attn_mask.numpy()
        #if not bool, convert to bool
        if not isinstance(attn_mask,bool):
            attn_mask = attn_mask.astype(bool)

        scaler_list = []
        for block_idx, attention_block in enumerate(attention_blocks):
            #check if attention block is ragged, if so, convert to dense tensor
            if isinstance(attention_block,tf.RaggedTensor):
                attention_block = attention_block.to_tensor(default_value=-1e9)
            #check if attention block is tensorflow tensor, if so, convert to numpy
            if isinstance(attention_block,tf.Tensor):
                attention_block = attention_block.numpy()
            for head_idx in range(attention_block.shape[1]):
                head_attn = attention_block[:,head_idx,:,:]
                attn_values = head_attn[attn_mask]
                scaler = MinMaxScaler()
                scaler.fit(attn_values.reshape(-1,1))
                scaler_list.append(scaler)
        scaler_list = np.array(scaler_list)

        # Dictionary to store attention matrices for each sample
        sample_attention_matrices = {}

        pos = 0
        for batch in dataset:
            batch_indices = batch['idx'].numpy()
            sample_ids = batch['sample_ids'].numpy()

            # Process each sample in the batch
            for sample_idx, sample_indices in enumerate(batch_indices):
                # Get valid variants for this sample
                mask = sample_indices != -1
                valid_indices = sample_indices[mask]
                valid_length = np.sum(mask)

                if valid_length == 0:
                    continue

                sample_id = sample_ids[sample_idx].decode('utf-8')

                n_blocks = len(attention_blocks)
                n_heads = attention_blocks[0].shape[1]

                if agg is None:
                    # Mode 1: No aggregation - return all heads for all blocks
                    blocks_data = []
                    head_pos = 0

                    for block_idx, attention_block in enumerate(attention_blocks):
                        if isinstance(attention_block, tf.RaggedTensor):
                            attention_block = attention_block.to_tensor(default_value=-1e9)
                        if isinstance(attention_block, tf.Tensor):
                            attention_block = attention_block.numpy()

                        heads_data = []
                        for head_idx in range(attention_block.shape[1]):
                            head_attention = attention_block[pos, head_idx]
                            head_attention_valid = head_attention[:valid_length, :valid_length]
                            head_attention_valid_flat = head_attention_valid.flatten().reshape(-1, 1)
                            head_attention_valid_scaled_flat = scaler_list[head_pos].transform(head_attention_valid_flat)
                            head_attention_valid_scaled = head_attention_valid_scaled_flat.reshape(head_attention_valid.shape)

                            heads_data.append(head_attention_valid_scaled)
                            head_pos += 1

                        blocks_data.append({
                            'block_idx': block_idx,
                            'heads': heads_data
                        })

                    sample_attention_matrices[sample_id] = {
                        'blocks': blocks_data,
                        'variant_indices': valid_indices
                    }

                elif agg == 'head':
                    # Mode 2: Aggregate heads per block
                    avg_attention_matrices = []
                    max_attention_matrices = []
                    head_pos = 0

                    for block_idx, attention_block in enumerate(attention_blocks):
                        if isinstance(attention_block, tf.RaggedTensor):
                            attention_block = attention_block.to_tensor(default_value=-1e9)
                        if isinstance(attention_block, tf.Tensor):
                            attention_block = attention_block.numpy()

                        avg_attention_matrix = np.zeros((valid_length, valid_length))
                        max_attention_matrix = np.full((valid_length, valid_length), -np.inf)

                        for head_idx in range(attention_block.shape[1]):
                            head_attention = attention_block[pos, head_idx]
                            head_attention_valid = head_attention[:valid_length, :valid_length]
                            head_attention_valid_flat = head_attention_valid.flatten().reshape(-1, 1)
                            head_attention_valid_scaled_flat = scaler_list[head_pos].transform(head_attention_valid_flat)
                            head_attention_valid_scaled = head_attention_valid_scaled_flat.reshape(head_attention_valid.shape)

                            avg_attention_matrix += head_attention_valid_scaled
                            max_attention_matrix = np.maximum(max_attention_matrix, head_attention_valid_scaled)
                            head_pos += 1

                        # Average across heads for this block
                        avg_attention_matrix /= n_heads

                        avg_attention_matrices.append(avg_attention_matrix)
                        max_attention_matrices.append(max_attention_matrix)

                    sample_attention_matrices[sample_id] = {
                        'avg_attention_matrices': avg_attention_matrices,
                        'max_attention_matrices': max_attention_matrices,
                        'variant_indices': valid_indices
                    }

                elif agg == 'all':
                    # Mode 3: Aggregate all heads and all blocks
                    avg_attention_matrix = np.zeros((valid_length, valid_length))
                    max_attention_matrix = np.full((valid_length, valid_length), -np.inf)
                    head_pos = 0

                    for block_idx, attention_block in enumerate(attention_blocks):
                        if isinstance(attention_block, tf.RaggedTensor):
                            attention_block = attention_block.to_tensor(default_value=-1e9)
                        if isinstance(attention_block, tf.Tensor):
                            attention_block = attention_block.numpy()

                        for head_idx in range(attention_block.shape[1]):
                            head_attention = attention_block[pos, head_idx]
                            head_attention_valid = head_attention[:valid_length, :valid_length]
                            head_attention_valid_flat = head_attention_valid.flatten().reshape(-1, 1)
                            head_attention_valid_scaled_flat = scaler_list[head_pos].transform(head_attention_valid_flat)
                            head_attention_valid_scaled = head_attention_valid_scaled_flat.reshape(head_attention_valid.shape)

                            avg_attention_matrix += head_attention_valid_scaled
                            max_attention_matrix = np.maximum(max_attention_matrix, head_attention_valid_scaled)
                            head_pos += 1

                    # Average across all blocks and heads
                    avg_attention_matrix /= (n_blocks * n_heads)

                    sample_attention_matrices[sample_id] = {
                        'avg_attention_matrix': avg_attention_matrix,
                        'max_attention_matrix': max_attention_matrix,
                        'variant_indices': valid_indices
                    }

                else:
                    raise ValueError(f"Invalid aggregation mode: {agg}. Choose from None, 'head', or 'all'.")

                pos += 1
        return sample_attention_matrices

    def get_cna_features(self, dataset_name, downcast=False):
        """
        Extract CNA embeddings/features from the trained features model.

        Args:
            dataset_name: Name of the dataset to process
            downcast: If True, downcast features to float16 to save memory

        Returns:
            Array of CNA features in original segment order
        """
        with self.strategy.scope():
            # Load model configuration if cross_modal_blocks is not set
            # This is necessary when loading a pre-trained model for feature extraction
            if not hasattr(self, 'cross_modal_blocks'):
                self._load_model_config_if_needed()

            # Only load from disk if model doesn't exist in memory
            if not hasattr(self, 'model_features_cna') or self.model_features_cna is None:
                model_path = os.path.join(self.model_dir, 'features_model_cna.keras')
                if not os.path.exists(model_path):
                    raise ValueError(f"CNA features model not found at {model_path}. Please train the model first using build_model().")
                self.model_features_cna = tf.keras.models.load_model(model_path, compile=False)

            # Ensure the dataset exists
            if not hasattr(self, dataset_name):
                raise ValueError(f"Dataset '{dataset_name}' not found. Please create it first using create_dataset method.")

            dataset = getattr(self, dataset_name)
            from tessera.input_keys import get_input_keys

            # Determine which inputs to use based on whether cross-attention was enabled during training
            # If cross_modal_blocks = 0, CNA features model expects ONLY CNA inputs
            # If cross_modal_blocks > 0, CNA features model expects ALL inputs (because of cross-attention)
            has_cross_attention = self.use_mut and self.use_cna and self.cross_modal_blocks > 0

            if has_cross_attention:
                keys = get_input_keys(use_mut=self.use_mut, use_vaf=self.use_vaf, use_cna=self.use_cna,
                                      use_cna_loh=self.use_cna_loh)
            else:
                keys = get_input_keys(use_mut=False, use_vaf=False, use_cna=self.use_cna,
                                      use_cna_loh=self.use_cna_loh)

            # Combined dataset: yields (filtered_inputs, cna_idx) in one pass so we can do
            # inline unpadding during inference without a second dataset iteration.
            def _add_cna_idx(batch):
                filtered = tessera.data.preprocessing.filter_inputs(batch, keys=keys)
                idx = batch.get('cna_idx', tf.constant([], dtype=tf.int32))
                return filtered, idx

            dataset_combined = dataset.map(_add_cna_idx, num_parallel_calls=tf.data.AUTOTUNE)
            dataset_combined = dataset_combined.prefetch(tf.data.AUTOTUNE)

            # Iterate batch-by-batch to avoid GPU OOM from predict() concatenating
            # the full output on-device.  Use predict() (not model()) per batch to
            # preserve the same compiled graph that was used during training.
            # For 3-D outputs (segment-level), unpad inline — never hold the full padded
            # (n_samples × max_segs × feature_dim) cube in RAM.
            all_features = []
            all_indices = []
            is_3d = None
            dtype = np.float16 if downcast else np.float32

            for batch_inputs, batch_idx in dataset_combined:
                _bf = self.model_features_cna.predict(
                    tf.data.Dataset.from_tensors(batch_inputs), verbose=0
                )
                if isinstance(_bf, list) and len(_bf) > 1:
                    _bf = np.concatenate(_bf, axis=-1)
                _bf_np = np.asarray(_bf).astype(dtype)

                if is_3d is None:
                    is_3d = (_bf_np.ndim == 3)

                if is_3d:
                    indices = batch_idx.numpy()  # (batch_size, cna_bag_size)
                    for j in range(_bf_np.shape[0]):
                        mask = indices[j] != -1
                        all_features.append(_bf_np[j, mask])
                        all_indices.append(indices[j, mask])
                else:
                    all_features.append(_bf_np)

            out = np.concatenate(all_features, axis=0)
            if is_3d:
                flat_indices = np.concatenate(all_indices)
                out = out[np.argsort(flat_indices)]

        return out

    def get_cna_predictions(self, dataset_name, return_true_values=False, return_loh=False):
        """
        Get CNA predictions (segment_mean and optionally LOH) from the model.

        Args:
            dataset_name: Name of the dataset to process
            return_true_values: If True, return true values along with predictions
            return_loh: If True, also return LOH predictions (only if model was trained with predict_cna_loh=True)

        Returns:
            Multiple return patterns based on flags:
            - return_true_values=False, return_loh=False: segment_mean_pred
            - return_true_values=True, return_loh=False: (segment_mean_pred, segment_mean_true)
            - return_true_values=False, return_loh=True: (segment_mean_pred, loh_pred)
            - return_true_values=True, return_loh=True: (segment_mean_pred, segment_mean_true, loh_pred, loh_true)
        """
        with self.strategy.scope():
            self.model_inf = tf.keras.models.load_model(
                os.path.join(self.model_dir, 'final_model.keras'), compile=False
            )

            if not hasattr(self, dataset_name):
                raise ValueError(f"Dataset '{dataset_name}' not found. Please create it first using create_dataset method.")

            dataset = getattr(self, dataset_name)
            from tessera.input_keys import get_input_keys
            keys = get_input_keys(use_mut=self.use_mut, use_vaf=self.use_vaf, use_cna=self.use_cna,
                                  use_cna_loh=self.use_cna_loh)
            dataset_inf = dataset.map(
                lambda x: tessera.data.preprocessing.filter_inputs(x, keys=keys),
                num_parallel_calls=tf.data.AUTOTUNE
            )
            dataset_inf = dataset_inf.prefetch(tf.data.AUTOTUNE)

            outputs = self.model_inf.predict(dataset_inf)

        # Extract CNA predictions from model outputs
        # Try new key first, fallback to legacy key for backward compatibility
        cna_segment_mean_pred = outputs.get('cna_segment_mean_pred', outputs.get('cna_pred'))
        cna_loh_pred = outputs.get('cna_loh_pred') if return_loh else None

        dataset = list(dataset)

        all_indices = []
        all_segment_mean_predictions = []
        all_segment_mean_true_values = [] if return_true_values else None
        all_loh_predictions = [] if return_loh and cna_loh_pred is not None else None
        all_loh_true_values = [] if return_loh and return_true_values else None
        pos = 0

        for batch in dataset:
            batch_idx = batch['cna_idx'].numpy()  # shape: (batch_size, cna_bag_size)
            if return_true_values:
                batch_segment_mean_true = batch['cna_segment_mean'].numpy()  # shape: (batch_size, cna_bag_size)
            if return_loh and return_true_values:
                batch_loh_true = batch.get('cna_loh')  # shape: (batch_size, cna_bag_size)
                if batch_loh_true is not None:
                    batch_loh_true = batch_loh_true.numpy()

            for i, sample_indices in enumerate(batch_idx):
                # CNA segments are masked by checking cna_idx != -1 (padding has idx=-1)
                mask = sample_indices != -1
                valid_length = np.sum(mask)

                # Extract segment_mean predictions for valid segments
                sample_segment_mean_pred = cna_segment_mean_pred[pos][:valid_length]

                if return_true_values:
                    sample_segment_mean_true = batch_segment_mean_true[i][:valid_length]

                # Extract LOH predictions if requested
                if all_loh_predictions is not None:
                    sample_loh_pred = cna_loh_pred[pos][:valid_length]

                if all_loh_true_values is not None and batch_loh_true is not None:
                    sample_loh_true = batch_loh_true[i][:valid_length]

                all_indices.append(sample_indices[:valid_length])
                all_segment_mean_predictions.append(sample_segment_mean_pred)

                if return_true_values:
                    all_segment_mean_true_values.append(sample_segment_mean_true)

                if all_loh_predictions is not None:
                    all_loh_predictions.append(sample_loh_pred)

                if all_loh_true_values is not None and batch_loh_true is not None:
                    all_loh_true_values.append(sample_loh_true)

                pos += 1

        # Convert to arrays and concatenate
        all_indices = np.concatenate(all_indices)
        all_segment_mean_predictions = np.concatenate(all_segment_mean_predictions)

        # Squeeze to 1D if needed (model outputs [batch, num_segments, 1])
        if all_segment_mean_predictions.ndim == 2 and all_segment_mean_predictions.shape[1] == 1:
            all_segment_mean_predictions = np.squeeze(all_segment_mean_predictions, axis=1)

        # Sort by original CNA indices
        sorted_idx = np.argsort(all_indices)
        all_segment_mean_predictions = all_segment_mean_predictions[sorted_idx]

        # Handle segment_mean true values
        if all_segment_mean_true_values is not None:
            all_segment_mean_true_values = np.concatenate(all_segment_mean_true_values)
            # Squeeze to 1D if needed (dataset format is [batch, num_segments, 1])
            if all_segment_mean_true_values.ndim == 2 and all_segment_mean_true_values.shape[1] == 1:
                all_segment_mean_true_values = np.squeeze(all_segment_mean_true_values, axis=1)
            all_segment_mean_true_values = all_segment_mean_true_values[sorted_idx]

        # Handle LOH predictions
        if all_loh_predictions is not None:
            all_loh_predictions = np.concatenate(all_loh_predictions)
            # Squeeze to 1D if needed (model outputs [batch, num_segments, 1])
            if all_loh_predictions.ndim == 2 and all_loh_predictions.shape[1] == 1:
                all_loh_predictions = np.squeeze(all_loh_predictions, axis=1)
            all_loh_predictions = all_loh_predictions[sorted_idx]

        # Handle LOH true values
        if all_loh_true_values is not None:
            all_loh_true_values = np.concatenate(all_loh_true_values)
            # Squeeze to 1D if needed (dataset format is [batch, num_segments, 1])
            if all_loh_true_values.ndim == 2 and all_loh_true_values.shape[1] == 1:
                all_loh_true_values = np.squeeze(all_loh_true_values, axis=1)
            all_loh_true_values = all_loh_true_values[sorted_idx]

        # Build return tuple based on requested outputs
        if return_true_values and return_loh:
            # Return all four: segment_mean_pred, segment_mean_true, loh_pred, loh_true
            return (all_segment_mean_predictions, all_segment_mean_true_values,
                    all_loh_predictions, all_loh_true_values)
        elif return_true_values:
            # Return segment_mean only: segment_mean_pred, segment_mean_true
            return (all_segment_mean_predictions, all_segment_mean_true_values)
        elif return_loh:
            # Return segment_mean and LOH predictions: segment_mean_pred, loh_pred
            return (all_segment_mean_predictions, all_loh_predictions)
        else:
            # Return only segment_mean predictions
            return all_segment_mean_predictions

    def _load_model_config_if_needed(self):
        """
        Load model configuration from saved JSON file if it exists.
        This is used when loading a pre-trained model for feature extraction
        without calling build_model().
        """
        import json

        config_path = os.path.join(self.model_dir, 'model_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)

            # Set architectural parameters
            if 'cross_modal_blocks' in config:
                self.cross_modal_blocks = config['cross_modal_blocks']
            else:
                self.cross_modal_blocks = 0  # Default to no cross-attention

            # Set modality flags if available
            if 'use_mut' in config:
                # Don't override if already set by create_sample_dataset
                if not hasattr(self, 'use_mut') or not self.use_mut:
                    self.use_mut = config['use_mut']

            if 'use_cna' in config:
                if not hasattr(self, 'use_cna') or not self.use_cna:
                    self.use_cna = config['use_cna']

            if 'use_vaf' in config:
                if not hasattr(self, 'use_vaf') or not self.use_vaf:
                    self.use_vaf = config['use_vaf']

            if 'use_cna_loh' in config:
                if not hasattr(self, 'use_cna_loh') or not self.use_cna_loh:
                    self.use_cna_loh = config['use_cna_loh']

            print(f"Loaded model configuration from {config_path}")
            print(f"  cross_modal_blocks: {self.cross_modal_blocks}")
            print(f"  use_mut: {self.use_mut}, use_cna: {self.use_cna}, use_vaf: {self.use_vaf}")
            print(f"  use_cna_loh: {self.use_cna_loh}")
        else:
            print(f"Warning: Model configuration not found at {config_path}")
            print("Using default values: cross_modal_blocks=0")
            self.cross_modal_blocks = 0

    def _save_model_config(self):
        """
        Save model configuration to a JSON file in the model directory.
        This allows the configuration to be loaded later when using pre-trained models.
        """
        import json
        
        # Ensure model directory exists
        if hasattr(self, 'model_dir') and self.model_dir is not None:
            os.makedirs(self.model_dir, exist_ok=True)
            
            config = {
                'nuc_embed_dim': self.nuc_embed_dim,
                'local_conv_dim': self.local_conv_dim,
                'local_conv_kernel': self.local_conv_kernel,
                'ref_alt_dim': self.ref_alt_dim,
                'local_embed_dim': self.local_embed_dim,
                'global_embed_dim': self.global_embed_dim,
                'local_num_heads': self.local_num_heads,
                'local_ff_dim': self.local_ff_dim,
                'local_attention_blocks': self.local_attention_blocks,
                'global_num_heads': self.global_num_heads,
                'global_ff_dim': self.global_ff_dim,
                'global_attention_blocks': self.global_attention_blocks,
                'variant_local_attention': self.variant_local_attention,
                'variant_self_attention': self.variant_self_attention,
                'attention_type': self.attention_type,
                'attention_activation_type': self.attention_activation_type,
                'dropout_rate': self.dropout_rate,
                'attention_l1_factor': self.attention_l1_factor,
                'recon_ref': self.recon_ref,
                'intermediate_dim_1': self.intermediate_dim_1,
                'intermediate_dim_2': self.intermediate_dim_2,
                # CNA dual-task parameters
                'predict_cna_loh': self.predict_cna_loh,
                # Cross-modal parameters
                'cross_modal_blocks': self.cross_modal_blocks,
                # InfoNCE loss parameters
                'use_infonce_loss': self.use_infonce_loss,
                'infonce_projection_dim': self.infonce_projection_dim,
                'infonce_temperature': self.infonce_temperature,
                'infonce_loss_weight': self.infonce_loss_weight,
                # Token-bag InfoNCE parameters (independent per modality)
                'use_mut_token_bag_infonce': self.use_mut_token_bag_infonce,
                'use_cna_token_bag_infonce': self.use_cna_token_bag_infonce,
                'token_bag_temperature': self.token_bag_temperature,
                'token_bag_loss_weight': self.token_bag_loss_weight,
                # Modality flags (determined at dataset creation time)
                'use_mut': self.use_mut,
                'use_cna': self.use_cna,
                'use_vaf': self.use_vaf,
                'use_cna_loh': self.use_cna_loh,
            }

            config_path = os.path.join(self.model_dir, 'model_config.json')
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"Model configuration saved to {config_path}")
        else:
            print("Warning: model_dir not set. Cannot save model configuration.")

    @staticmethod
    def load_model_config(model_path):
        """
        Load model configuration from a saved model directory or config file.
        
        Parameters:
        -----------
        model_path : str
            Path to either the model directory containing model_config.json 
            or direct path to model_config.json file
            
        Returns:
        --------
        dict
            Dictionary containing model configuration parameters
        """
        import json
        
        # Determine if path points to directory or file
        if os.path.isdir(model_path):
            config_path = os.path.join(model_path, 'model_config.json')
        elif model_path.endswith('.json'):
            config_path = model_path
        else:
            # Assume it's a model file path, look for config in same directory
            model_dir = os.path.dirname(model_path)
            config_path = os.path.join(model_dir, 'model_config.json')
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"Loaded model configuration from {config_path}")
            return config
        else:
            print(f"Warning: No model configuration found at {config_path}")
            print("Returning default configuration. You may need to specify parameters manually.")
            return {
                'nuc_embed_dim': 12,
                'local_conv_dim': [512, 256, 128],
                'local_conv_kernel': [25, 100, 500],
                'ref_alt_dim': 12,
                'local_embed_dim': 256,
                'global_embed_dim': 256,
                'local_num_heads': 6,
                'local_ff_dim': 256,
                'local_attention_blocks': 1,
                'global_num_heads': 8,
                'global_ff_dim': 512,
                'global_attention_blocks': 6,
                'variant_local_attention': True,
                'variant_self_attention': True,
                'attention_type': 'pairwise',
                'attention_activation_type': 'softmax',
                'dropout_rate': 0.1,
                'attention_l1_factor': 0.0,
                'recon_ref': False,
                'intermediate_dim_1': 512,
                'intermediate_dim_2': 128,
                # InfoNCE loss parameters
                'use_infonce_loss': False,
                'infonce_projection_dim': 128,
                'infonce_temperature': 0.1,
                'infonce_loss_weight': 1.0,
                # Token-bag InfoNCE parameters (independent per modality)
                'use_mut_token_bag_infonce': False,
                'use_cna_token_bag_infonce': False,
                'token_bag_temperature': 0.1,
                'token_bag_loss_weight': 1.0,
            }


