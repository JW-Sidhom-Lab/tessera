"""Loss functions for variant sequence reconstruction."""

from typing import Optional
import tensorflow as tf


def compute_loss(
    y_true: tf.Tensor,
    logits: tf.Tensor,
    non_zero_only: bool = False,
    hinge_loss_t: Optional[float] = None,
    include_all_in_loss: bool = True,
    per_sample_loss: bool = False
) -> tf.Tensor:
    """
    Compute reconstruction loss for variant sequence prediction.

    This function implements a flexible loss computation with several modes:
    - Position-level masking (non_zero_only): Only compute loss on actual variant positions
    - Hinge loss thresholding: Focus learning on difficult variants above a loss threshold
    - Per-sample vs per-variant weighting: Control how loss is aggregated across samples

    Args:
        y_true: True alteration sequences of shape [batch, num_variants, seq_len]
        logits: Raw model predictions (logits) of shape [batch, num_variants, seq_len, vocab_size]
        non_zero_only: If True, only compute loss for non-zero elements in y_true.
                      This is useful when sequences are padded with zeros.
        hinge_loss_t: Threshold for hinge loss. Variants with loss below this threshold
                     won't contribute to the total loss. If None, all variants contribute.
        include_all_in_loss: Whether to include variants with zero loss in the mean calculation.
                           If False, only non-zero loss variants contribute to the mean.
        per_sample_loss: If True, compute per-sample loss by averaging variant losses within
                        each sample first, then averaging across samples. This gives equal
                        weight to each sample regardless of number of variants.
                        If False (default), all variants are weighted equally.

    Returns:
        Scalar loss tensor

    Example:
        >>> y_true = tf.constant([[[1, 2, 0], [3, 4, 0]]])  # [1, 2, 3]
        >>> logits = tf.random.normal([1, 2, 3, 5])  # [1, 2, 3, vocab_size]
        >>> loss = compute_loss(y_true, logits, non_zero_only=True)
        >>> loss.shape
        TensorShape([])

    Note:
        Length normalization is applied to prevent longer variants (deletions) from
        dominating the loss. Variant loss is divided by sqrt(variant_length).
    """
    y_true = tf.cast(y_true, dtype=tf.int32)

    # Create mask for variants with any non-zero values
    variant_mask = tf.reduce_any(tf.not_equal(y_true, 0), axis=-1)  # [batch, num_variants]
    variant_mask = tf.expand_dims(variant_mask, axis=-1)  # [batch, num_variants, 1]
    variant_mask = tf.cast(variant_mask, tf.float32)

    # Compute the standard loss per position
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, logits, from_logits=True)  # [batch, num_variants, seq_len]

    if non_zero_only:
        # Create position-level mask for non-zero elements
        non_zero_mask = tf.not_equal(y_true, 0)  # [batch, num_variants, seq_len]
        non_zero_mask = tf.cast(non_zero_mask, loss.dtype)  # Match loss dtype for mixed precision
        
        # Apply position-level masking to loss before averaging
        masked_loss = loss * non_zero_mask  # [batch, num_variants, seq_len]
        
        # Compute variant loss as average of only non-zero positions
        position_counts = tf.reduce_sum(non_zero_mask, axis=-1)  # [batch, num_variants]
        variant_loss = tf.reduce_sum(masked_loss, axis=-1) / (position_counts + tf.keras.backend.epsilon())  # [batch, num_variants]
    else:
        # Compute loss per variant (average across all sequence positions)
        variant_loss = tf.reduce_mean(loss, axis=-1)  # [batch, num_variants]

        # Length-normalize to prevent longer variants (deletions) from having higher loss
        variant_length = tf.reduce_sum(tf.cast(tf.not_equal(y_true, 0), loss.dtype), axis=-1)  # [batch, num_variants]
        variant_loss = variant_loss / tf.sqrt(variant_length + tf.keras.backend.epsilon())

    # Apply variant mask to exclude padding variants
    variant_mask_2d = tf.squeeze(variant_mask, axis=-1)  # [batch, num_variants]
    variant_mask_2d = tf.cast(variant_mask_2d, loss.dtype)
    
    # Apply threshold if specified - only variants with loss above threshold contribute
    if hinge_loss_t is not None:
        hinge_mask = tf.cast(variant_loss > hinge_loss_t, variant_mask_2d.dtype)
        active_variants = variant_mask_2d * hinge_mask
    else:
        active_variants = variant_mask_2d

    # Apply mask to variant losses
    masked_variant_loss = variant_loss * active_variants

    # Calculate final loss based on per_sample_loss and include_all_in_loss settings
    if per_sample_loss:
        # Per-sample loss: average variant losses within each sample, then average across samples
        # This gives more weight to variants in samples with fewer variants

        if hinge_loss_t is not None and include_all_in_loss:
            # Use all variants (even zeroed ones) for per-sample averaging
            sample_variant_loss = variant_loss * variant_mask_2d  # [batch, num_variants]
            sample_variant_counts = tf.reduce_sum(variant_mask_2d, axis=-1)  # [batch]
        else:
            # Use only active variants for per-sample averaging
            sample_variant_loss = masked_variant_loss  # [batch, num_variants]
            sample_variant_counts = tf.reduce_sum(active_variants, axis=-1)  # [batch]

        # Average loss per sample (sum of variant losses / number of variants per sample)
        sample_loss_sum = tf.reduce_sum(sample_variant_loss, axis=-1)  # [batch]
        sample_mean_loss = sample_loss_sum / (sample_variant_counts + tf.keras.backend.epsilon())  # [batch]

        # Final loss is mean across samples
        mean_loss = tf.reduce_mean(sample_mean_loss)

    else:
        # Original per-variant loss behavior
        if hinge_loss_t is not None and include_all_in_loss:
            # Include all variants in loss computation (even zeroed ones)
            # This gives a loss that decreases as training progresses
            masked_all_variant_loss = variant_loss * variant_mask_2d
            mean_loss = tf.reduce_mean(masked_all_variant_loss)
        else:
            # Original behavior: only active variants contribute
            # Sum of weighted losses divided by sum of active variants (with epsilon to avoid div by zero)
            denominator = tf.reduce_sum(active_variants) + tf.keras.backend.epsilon()
            mean_loss = tf.reduce_sum(masked_variant_loss) / denominator

    return mean_loss


def cna_logfold_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    cna_chr: tf.Tensor,
    per_sample_loss: bool = False
) -> tf.Tensor:
    """
    Compute regression loss for CNA segment mean (log-fold change) prediction.

    This function computes the Mean Absolute Error (MAE) between predicted and true
    CNA segment means, with proper masking for padded segments based on chromosome IDs.

    Args:
        y_true: True CNA segment means of shape [batch, num_segments] or [batch, num_segments, 1]
        y_pred: Predicted CNA segment means of shape [batch, num_segments] or [batch, num_segments, 1]
        cna_chr: CNA chromosome IDs of shape [batch, num_segments] or [batch, num_segments, 1]
                Used to identify padding (cna_chr == 0)
        per_sample_loss: If True, compute per-sample loss by averaging segment losses within
                        each sample first, then averaging across samples. This gives equal
                        weight to each sample regardless of number of segments.
                        If False (default), all segments are weighted equally.

    Returns:
        Scalar loss tensor

    Example:
        >>> y_true = tf.constant([[0.5, -0.3, 0.0], [0.8, 0.0, 0.0]])  # [2, 3]
        >>> y_pred = tf.constant([[0.4, -0.2, 0.0], [0.7, 0.0, 0.0]])  # [2, 3]
        >>> cna_chr = tf.constant([[1, 2, 0], [1, 0, 0]])  # [2, 3], 0 = padding
        >>> loss = cna_logfold_loss(y_true, y_pred, cna_chr)
        >>> loss.shape
        TensorShape([])

    Note:
        - Uses MAE (L1) instead of MSE (L2) for robustness to outliers
        - Segments with cna_chr=0 are treated as padding and excluded
        - Real CNA segments can have segment_mean ≈ 0.0 (neutral copy number)
        - The loss is computed only on non-padded CNA segments (cna_chr != 0)
    """
    # All tensors have shape [batch, num_segments, 1] - no reshaping needed
    # Binary operations will broadcast correctly

    # Create mask for non-padded CNA segments using chromosome IDs
    # IMPORTANT: We use cna_chr != 0 instead of segment_mean != 0.0
    # because real CNA segments CAN have segment_mean ≈ 0 (neutral copy number)
    segment_mask = tf.not_equal(cna_chr, 0)  # [batch, num_segments]
    segment_mask = tf.cast(segment_mask, tf.float32)

    # Compute absolute error per segment
    segment_loss = tf.abs(y_true - y_pred)  # [batch, num_segments]

    # Apply mask to exclude padded segments
    masked_segment_loss = segment_loss * segment_mask

    # Calculate final loss based on per_sample_loss setting
    if per_sample_loss:
        # Per-sample loss: average segment losses within each sample, then average across samples
        # Sum over both num_segments and trailing dimension to get [batch]
        sample_loss_sum = tf.reduce_sum(masked_segment_loss, axis=[1, 2])  # [batch]
        sample_segment_counts = tf.reduce_sum(segment_mask, axis=[1, 2])  # [batch]
        sample_mean_loss = sample_loss_sum / (sample_segment_counts + tf.keras.backend.epsilon())  # [batch]

        # Final loss is mean across samples
        mean_loss = tf.reduce_mean(sample_mean_loss)
    else:
        # Per-segment loss: all segments weighted equally
        # Sum of weighted losses divided by sum of active segments (with epsilon to avoid div by zero)
        denominator = tf.reduce_sum(segment_mask) + tf.keras.backend.epsilon()
        mean_loss = tf.reduce_sum(masked_segment_loss) / denominator

    return mean_loss


def cna_loh_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    cna_chr: tf.Tensor,
    per_sample_loss: bool = False
) -> tf.Tensor:
    """
    Compute binary cross-entropy loss for CNA LOH prediction.

    Args:
        y_true: True LOH labels [batch, num_segments] or [batch, num_segments, 1]
                Values should be 0 (no LOH) or 1 (LOH present)
        y_pred: Predicted LOH probabilities [batch, num_segments] or [batch, num_segments, 1]
                Values should be in [0, 1] from sigmoid activation
        cna_chr: CNA chromosome IDs [batch, num_segments] or [batch, num_segments, 1]
                Used to identify padding (cna_chr == 0)
        per_sample_loss: If True, compute per-sample loss for fair weighting

    Returns:
        Scalar loss tensor

    Example:
        >>> y_true = tf.constant([[1, 0, 0], [1, 1, 0]])  # [2, 3]
        >>> y_pred = tf.constant([[0.9, 0.2, 0.1], [0.8, 0.7, 0.3]])  # [2, 3]
        >>> cna_chr = tf.constant([[1, 2, 0], [1, 2, 3]])  # [2, 3], 0 = padding
        >>> loss = cna_loh_loss(y_true, y_pred, cna_chr)

    Note:
        - Uses binary cross-entropy for binary classification
        - Segments with cna_chr=0 are treated as padding and excluded
        - Handles class imbalance through proper weighting
    """
    # All tensors have shape [batch, num_segments, 1] - no reshaping needed
    # Binary operations will broadcast correctly

    # Create mask for non-padded CNA segments
    segment_mask = tf.not_equal(cna_chr, 0)  # [batch, num_segments]
    segment_mask = tf.cast(segment_mask, tf.float32)

    # Convert y_true to float for BCE
    y_true_float = tf.cast(y_true, tf.float32)

    # Clip predictions to avoid log(0) issues
    y_pred_clipped = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)

    # Compute binary cross-entropy per segment
    # Note: Using from_logits=False since y_pred comes from sigmoid
    # binary_crossentropy reduces over last dim, so [batch, num_segments, 1] → [batch, num_segments]
    segment_loss = tf.keras.losses.binary_crossentropy(
        y_true_float, y_pred_clipped, from_logits=False
    )  # [batch, num_segments]

    # Expand dimensions to match mask shape for proper broadcasting
    segment_loss = tf.expand_dims(segment_loss, axis=-1)  # [batch, num_segments, 1]

    # Apply mask to exclude padded segments
    masked_segment_loss = segment_loss * segment_mask

    # Calculate final loss based on per_sample_loss setting
    if per_sample_loss:
        # Per-sample loss: average segment losses within each sample
        # Sum over both num_segments and trailing dimension to get [batch]
        sample_loss_sum = tf.reduce_sum(masked_segment_loss, axis=[1, 2])  # [batch]
        sample_segment_counts = tf.reduce_sum(segment_mask, axis=[1, 2])  # [batch]
        sample_mean_loss = sample_loss_sum / (sample_segment_counts + tf.keras.backend.epsilon())

        # Final loss is mean across samples
        mean_loss = tf.reduce_mean(sample_mean_loss)
    else:
        # Per-segment loss: all segments weighted equally
        denominator = tf.reduce_sum(segment_mask) + tf.keras.backend.epsilon()
        mean_loss = tf.reduce_sum(masked_segment_loss) / denominator

    return mean_loss


def multimodal_cna_loss(
    segment_mean_y_true: tf.Tensor,
    segment_mean_y_pred: tf.Tensor,
    loh_y_true: Optional[tf.Tensor],
    loh_y_pred: Optional[tf.Tensor],
    cna_chr: tf.Tensor,
    per_sample_loss: bool = False
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Compute combined multi-task loss for CNA segment_mean + LOH prediction.

    This function enables joint training on both CNA tasks, allowing the model
    to learn from complementary supervision signals.

    Args:
        segment_mean_y_true: True segment means [batch, num_segments]
        segment_mean_y_pred: Predicted segment means [batch, num_segments]
        loh_y_true: True LOH labels [batch, num_segments] or None
        loh_y_pred: Predicted LOH probabilities [batch, num_segments] or None
        cna_chr: CNA chromosome IDs [batch, num_segments] for masking
        per_sample_loss: If True, compute per-sample loss for fair weighting

    Returns:
        tuple of (total_loss, segment_mean_loss, loh_loss)
        - total_loss: Sum of both losses
        - segment_mean_loss: Segment_mean loss
        - loh_loss: LOH loss (0.0 if no LOH data)

    Example:
        >>> seg_true = tf.constant([[0.5, -0.3], [0.8, -0.1]])
        >>> seg_pred = tf.constant([[0.4, -0.2], [0.7, -0.05]])
        >>> loh_true = tf.constant([[1, 0], [0, 1]])
        >>> loh_pred = tf.constant([[0.9, 0.2], [0.3, 0.8]])
        >>> cna_chr = tf.constant([[1, 2], [1, 2]])
        >>> total, seg_loss, loh_loss = multimodal_cna_loss(
        ...     seg_true, seg_pred, loh_true, loh_pred, cna_chr
        ... )

    Note:
        - If LOH data is None, only segment_mean loss is computed
        - Both losses are simply added together with no weighting
    """
    # Compute segment_mean regression loss (MAE)
    segment_mean_loss = cna_logfold_loss(
        y_true=segment_mean_y_true,
        y_pred=segment_mean_y_pred,
        cna_chr=cna_chr,
        per_sample_loss=per_sample_loss
    )

    # Compute LOH classification loss if LOH data is provided
    if loh_y_true is not None and loh_y_pred is not None:
        loh_loss = cna_loh_loss(
            y_true=loh_y_true,
            y_pred=loh_y_pred,
            cna_chr=cna_chr,
            per_sample_loss=per_sample_loss
        )
    else:
        # No LOH data - set LOH loss to 0
        loh_loss = tf.constant(0.0, dtype=segment_mean_loss.dtype)

    # Simply add the losses (no weighting)
    total_loss = segment_mean_loss + loh_loss

    return total_loss, segment_mean_loss, loh_loss


def multimodal_loss(
    mutation_y_true: tf.Tensor,
    mutation_logits: tf.Tensor,
    cna_y_true: Optional[tf.Tensor],
    cna_y_pred: Optional[tf.Tensor],
    mutation_loss_weight: float = 0.5,
    cna_loss_weight: float = 0.5,
    non_zero_only: bool = False,
    hinge_loss_t: Optional[float] = None,
    include_all_in_loss: bool = True,
    per_sample_loss: bool = False
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Compute combined multi-modal loss for mutation reconstruction and CNA prediction.

    This function enables joint training on both somatic mutations and copy number alterations,
    allowing the model to learn from both data types simultaneously. The losses are weighted
    to balance their contributions to the total loss.

    Args:
        mutation_y_true: True alteration sequences for mutations [batch, num_variants, seq_len]
        mutation_logits: Raw model predictions for mutations [batch, num_variants, seq_len, vocab_size]
        cna_y_true: True CNA segment means [batch, num_segments] or None if no CNA data
        cna_y_pred: Predicted CNA segment means [batch, num_segments] or None if no CNA data
        mutation_loss_weight: Weight for mutation reconstruction loss (default: 0.5)
        cna_loss_weight: Weight for CNA prediction loss (default: 0.5)
        non_zero_only: If True, only compute mutation loss for non-zero elements
        hinge_loss_t: Threshold for hinge loss on mutations. If None, all variants contribute.
        include_all_in_loss: Whether to include variants with zero loss in the mean calculation
        per_sample_loss: If True, compute per-sample loss for fair weighting across samples

    Returns:
        tuple of (total_loss, mutation_loss, cna_loss)
        - total_loss: Weighted combination of mutation and CNA losses
        - mutation_loss: Unweighted mutation reconstruction loss
        - cna_loss: Unweighted CNA prediction loss (0.0 if no CNA data)

    Example:
        >>> mutation_y_true = tf.random.uniform([4, 100, 1], 0, 5, dtype=tf.int32)
        >>> mutation_logits = tf.random.normal([4, 100, 1, 5])
        >>> cna_y_true = tf.random.normal([4, 50])
        >>> cna_y_pred = tf.random.normal([4, 50])
        >>> total_loss, mut_loss, cna_loss = multimodal_loss(
        ...     mutation_y_true, mutation_logits, cna_y_true, cna_y_pred,
        ...     mutation_loss_weight=0.5, cna_loss_weight=0.5
        ... )

    Note:
        - If CNA data is None, only mutation loss is computed (CNA loss = 0)
        - Weights should sum to 1.0 for interpretability, but this is not enforced
        - Both losses are computed independently before weighting
    """
    # Compute mutation reconstruction loss
    mutation_loss = compute_loss(
        y_true=mutation_y_true,
        logits=mutation_logits,
        non_zero_only=non_zero_only,
        hinge_loss_t=hinge_loss_t,
        include_all_in_loss=include_all_in_loss,
        per_sample_loss=per_sample_loss
    )

    # Compute CNA prediction loss if CNA data is provided
    if cna_y_true is not None and cna_y_pred is not None:
        cna_loss = cna_logfold_loss(
            y_true=cna_y_true,
            y_pred=cna_y_pred,
            per_sample_loss=per_sample_loss
        )
    else:
        # No CNA data - set CNA loss to 0
        cna_loss = tf.constant(0.0, dtype=mutation_loss.dtype)

    # Compute weighted total loss
    total_loss = (mutation_loss_weight * mutation_loss) + (cna_loss_weight * cna_loss)

    return total_loss, mutation_loss, cna_loss


def create_adaptive_loss_fn(
    use_mut: bool,
    use_cna: bool,
    mutation_loss_weight: float = 0.5,
    cna_loss_weight: float = 0.5,
    loss_non_zero_only: bool = False,
    hinge_loss_t: Optional[float] = None,
    per_sample_loss: bool = False
):
    """
    Factory function to create an adaptive loss function based on available data modalities.

    This function returns a custom loss function that automatically adapts to the available
    data types (mutations, CNA, or both). The returned function has the signature required
    by CustomTrainingModel.

    Args:
        use_mut: Whether mutation data is available
        use_cna: Whether CNA data is available
        mutation_loss_weight: Weight for mutation reconstruction loss (default: 0.5)
        cna_loss_weight: Weight for CNA prediction loss (default: 0.5)
        loss_non_zero_only: If True, only compute mutation loss for non-zero elements
        hinge_loss_t: Threshold for hinge loss on mutations. If None, all variants contribute.
        per_sample_loss: If True, compute per-sample loss for fair weighting across samples

    Returns:
        A loss function with signature: loss_fn(y_true: dict, model_outputs: dict) -> tf.Tensor

    Example:
        >>> # Multi-modal case
        >>> loss_fn = create_adaptive_loss_fn(use_mut=True, use_cna=True)
        >>> loss = loss_fn(
        ...     y_true={'alt': alt_data, 'cna_segment_mean': cna_data},
        ...     model_outputs={'logits': logits, 'cna_pred': cna_pred}
        ... )

        >>> # Mutation-only case
        >>> loss_fn = create_adaptive_loss_fn(use_mut=True, use_cna=False)
        >>> loss = loss_fn(
        ...     y_true={'alt': alt_data},
        ...     model_outputs={'logits': logits}
        ... )

    Note:
        The returned function expects:
        - y_true: Dictionary with keys 'alt' (mutations) and/or 'cna_segment_mean' (CNA)
        - model_outputs: Dictionary with keys 'logits' (mutations) and/or 'cna_pred' (CNA)
    """
    if use_mut and use_cna:
        # Multi-modal training: use multimodal_loss for both mutations and CNA
        def loss_fn(y_true, model_outputs):
            """Multi-modal loss function for joint mutation and CNA training."""
            total_loss, _, _ = multimodal_loss(
                mutation_y_true=y_true['alt'],
                mutation_logits=model_outputs['logits'],
                cna_y_true=y_true.get('cna_segment_mean', None),
                cna_y_pred=model_outputs.get('cna_pred', None),
                mutation_loss_weight=mutation_loss_weight,
                cna_loss_weight=cna_loss_weight,
                non_zero_only=loss_non_zero_only,
                hinge_loss_t=hinge_loss_t,
                include_all_in_loss=True,
                per_sample_loss=per_sample_loss
            )
            return total_loss

    elif use_cna:
        # CNA-only training: use cna_logfold_loss
        def loss_fn(y_true, model_outputs):
            """CNA-only loss function."""
            return cna_logfold_loss(
                y_true=y_true['cna_segment_mean'],
                y_pred=model_outputs['cna_pred'],
                per_sample_loss=per_sample_loss
            )

    else:
        # Mutation-only training (default): use compute_loss
        def loss_fn(y_true, model_outputs):
            """Mutation-only loss function."""
            return compute_loss(
                y_true=y_true['alt'],
                logits=model_outputs['logits'],
                non_zero_only=loss_non_zero_only,
                hinge_loss_t=hinge_loss_t,
                per_sample_loss=per_sample_loss
            )

    return loss_fn


def compute_infonce_loss(
    mut_embeddings: tf.Tensor,
    cna_embeddings: tf.Tensor,
    temperature: float = 0.1,
    valid_pairs_mask: Optional[tf.Tensor] = None
) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Compute InfoNCE contrastive loss between mutation and CNA sample embeddings.

    This loss encourages the model to learn aligned representations where:
    - Positive pairs (mutation and CNA from the same sample) have high similarity
    - Negative pairs (mutation and CNA from different samples) have low similarity

    The loss is computed bidirectionally (mut→cna and cna→mut) and averaged for
    symmetric alignment.

    Args:
        mut_embeddings: Mutation sample embeddings of shape [batch, projection_dim]
        cna_embeddings: CNA sample embeddings of shape [batch, projection_dim]
        temperature: Temperature parameter for scaling similarities (default: 0.1)
                    Lower values (0.05-0.1) make the loss focus more on hard negatives
                    Higher values (0.3-0.5) provide softer, more stable training
        valid_pairs_mask: Optional boolean mask of shape [batch] indicating which samples
                         have both mutation and CNA data. Only samples where mask is True
                         will contribute to the loss. If None, all samples contribute.

    Returns:
        tuple of (loss, margin):
        - loss: Scalar loss value (InfoNCE loss)
        - margin: Scalar alignment margin = avg(positive_sim) - avg(negative_sim)
                 Dimension-invariant metric measuring alignment quality

    Mathematical Formula:
        For each sample i in the batch:
        L_i = -log(exp(sim(mut_i, cna_i) / τ) / Σ_j exp(sim(mut_i, cna_j) / τ))

        where:
        - sim(a, b) = cosine similarity between vectors a and b
        - τ = temperature parameter
        - Sum is over all samples j in the batch

    Margin Computation:
        - Positive similarities: diagonal of normalized similarity matrix
        - Negative similarities: off-diagonal elements
        - Margin = mean(positive_sims) - mean(negative_sims)
        - Only computed for valid sample pairs when mask is provided

    Example:
        >>> mut_emb = tf.random.normal([32, 128])  # 32 samples, 128-dim embeddings
        >>> cna_emb = tf.random.normal([32, 128])
        >>> loss, margin = compute_infonce_loss(mut_emb, cna_emb, temperature=0.1)
        >>> loss.shape
        TensorShape([])
        >>> margin.shape
        TensorShape([])

    Note:
        - Embeddings are L2-normalized to unit sphere for cosine similarity
        - Temperature controls the hardness of negative samples
        - Requires batch_size >= 2 for meaningful contrastive learning
        - Loss is symmetric: average of mut→cna and cna→mut directions
        - Margin is dimension-invariant and provides interpretable alignment metric
        - When valid_pairs_mask is provided, only samples with both modalities contribute
    """
    # Normalize embeddings to unit sphere for cosine similarity
    # This ensures sim(a, b) = cosine_similarity(a, b)
    mut_embeddings = tf.nn.l2_normalize(mut_embeddings, axis=-1)  # [batch, proj_dim]
    cna_embeddings = tf.nn.l2_normalize(cna_embeddings, axis=-1)  # [batch, proj_dim]

    # Compute similarity matrix: [batch, batch]
    # similarity_matrix[i,j] = cosine_similarity(mut_i, cna_j)
    similarity_matrix = tf.matmul(mut_embeddings, cna_embeddings, transpose_b=True)

    # Store raw similarities (before temperature scaling) for margin computation
    similarity_matrix_raw = similarity_matrix

    # Apply temperature scaling to control hardness of negatives
    # Lower temperature → sharper distribution → focus on hard negatives
    # Higher temperature → softer distribution → more stable training
    similarity_matrix = similarity_matrix / temperature

    # Compute InfoNCE loss manually for XLA compatibility
    # Manual implementation avoids XLA fusion issues with built-in cross-entropy ops

    # Direction 1: mut → cna (for each mutation, find its matching CNA)
    # Compute softmax manually
    exp_sim = tf.exp(similarity_matrix)  # [batch, batch]
    sum_exp = tf.reduce_sum(exp_sim, axis=1, keepdims=True)  # [batch, 1]
    softmax = exp_sim / sum_exp  # [batch, batch]

    # Extract positive pair probabilities (diagonal elements)
    positive_probs_mut = tf.linalg.diag_part(softmax)  # [batch]

    # Compute negative log likelihood
    loss_mut_to_cna = -tf.math.log(positive_probs_mut + 1e-7)  # [batch]

    # Direction 2: cna → mut (for each CNA, find its matching mutation)
    # Transpose and repeat
    similarity_matrix_t = tf.transpose(similarity_matrix)  # [batch, batch]
    exp_sim_t = tf.exp(similarity_matrix_t)  # [batch, batch]
    sum_exp_t = tf.reduce_sum(exp_sim_t, axis=1, keepdims=True)  # [batch, 1]
    softmax_t = exp_sim_t / sum_exp_t  # [batch, batch]

    # Extract positive pair probabilities (diagonal elements)
    positive_probs_cna = tf.linalg.diag_part(softmax_t)  # [batch]

    # Compute negative log likelihood
    loss_cna_to_mut = -tf.math.log(positive_probs_cna + 1e-7)  # [batch]

    # Apply valid pairs mask if provided
    # This ensures only samples with both modalities contribute to the loss
    if valid_pairs_mask is not None:
        valid_pairs_mask = tf.cast(valid_pairs_mask, loss_mut_to_cna.dtype)  # [batch]

        # Apply mask to per-sample losses
        loss_mut_to_cna = loss_mut_to_cna * valid_pairs_mask
        loss_cna_to_mut = loss_cna_to_mut * valid_pairs_mask

        # Compute mean only over valid pairs
        num_valid_pairs = tf.reduce_sum(valid_pairs_mask) + tf.keras.backend.epsilon()
        loss_mut_to_cna_mean = tf.reduce_sum(loss_mut_to_cna) / num_valid_pairs
        loss_cna_to_mut_mean = tf.reduce_sum(loss_cna_to_mut) / num_valid_pairs
    else:
        loss_mut_to_cna_mean = tf.reduce_mean(loss_mut_to_cna)
        loss_cna_to_mut_mean = tf.reduce_mean(loss_cna_to_mut)

    # Symmetric loss: average both directions
    # This ensures balanced alignment in both mut→cna and cna→mut spaces
    loss = (loss_mut_to_cna_mean + loss_cna_to_mut_mean) / 2.0

    # Compute alignment margin (dimension-invariant metric)
    # Extract diagonal (positive pairs) from raw similarity matrix
    positive_sims = tf.linalg.diag_part(similarity_matrix_raw)  # [batch]

    # Extract off-diagonal (negative pairs) from raw similarity matrix
    # Create mask for off-diagonal elements
    batch_size = tf.shape(similarity_matrix_raw)[0]
    off_diag_mask = 1.0 - tf.eye(batch_size)  # [batch, batch] with 0s on diagonal

    # Apply mask and compute mean of negative similarities
    masked_sims = similarity_matrix_raw * off_diag_mask  # [batch, batch]
    num_negatives = tf.reduce_sum(off_diag_mask)  # Total number of off-diagonal elements
    negative_sims_sum = tf.reduce_sum(masked_sims)
    negative_sims_mean = negative_sims_sum / (num_negatives + 1e-7)

    # Margin: positive similarity should be high, negative should be low
    # Apply valid pairs mask to margin computation as well
    if valid_pairs_mask is not None:
        masked_positive_sims = positive_sims * valid_pairs_mask
        positive_sims_mean = tf.reduce_sum(masked_positive_sims) / (tf.reduce_sum(valid_pairs_mask) + tf.keras.backend.epsilon())
    else:
        positive_sims_mean = tf.reduce_mean(positive_sims)

    margin = positive_sims_mean - negative_sims_mean

    return loss, margin


def compute_token_bag_infonce_loss(
    token_embeddings: tf.Tensor,
    bag_embeddings: tf.Tensor,
    token_mask: tf.Tensor,
    temperature: float = 0.1,
    per_sample_loss: bool = False
) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Compute InfoNCE contrastive loss between tokens and their bag representations.

    This loss maximizes mutual information I(Z_tok; Z_bag) by encouraging:
    - High similarity between tokens and their own sample's bag (positive pairs)
    - Low similarity between tokens and other samples' bags (negative pairs)

    Args:
        token_embeddings: Token-level embeddings [batch, num_tokens, projection_dim]
        bag_embeddings: Sample-level bag embeddings [batch, projection_dim]
        token_mask: Boolean mask for valid tokens [batch, num_tokens]
                   True = valid token, False = padding
        temperature: Temperature parameter for scaling similarities (default: 0.1)
        per_sample_loss: If True, compute per-sample loss by averaging token losses within
                        each sample first, then averaging across samples. This gives equal
                        weight to each sample regardless of number of tokens.
                        If False (default), all tokens are weighted equally.

    Returns:
        tuple of (loss, margin):
        - loss: Scalar InfoNCE loss value
        - margin: Alignment margin = avg(positive_sim) - avg(negative_sim)

    Mathematical Formula:
        For each valid token j in sample i:
        L_j = -log(exp(sim(z_tok_j, z_bag_i) / τ) / Σ_k exp(sim(z_tok_j, z_bag_k) / τ))

        where:
        - sim(a, b) = cosine similarity
        - τ = temperature
        - k ranges over all samples in batch

    Example:
        >>> token_emb = tf.random.normal([32, 100, 128])  # 32 samples, 100 tokens, 128-dim
        >>> bag_emb = tf.random.normal([32, 128])
        >>> mask = tf.constant([[True]*50 + [False]*50] * 32)  # 50 valid tokens per sample
        >>> loss, margin = compute_token_bag_infonce_loss(token_emb, bag_emb, mask)

    Note:
        - Embeddings are L2-normalized to unit sphere for cosine similarity
        - Temperature controls the hardness of negative samples
        - Requires batch_size >= 2 for meaningful contrastive learning
        - Properly handles masking for padded tokens throughout computation
        - XLA-compatible manual softmax computation
        - With per_sample_loss=True, samples with fewer tokens are weighted equally to
          samples with more tokens, preventing high-burden samples from dominating training
    """
    # Normalize embeddings to unit sphere for cosine similarity
    token_embeddings = tf.nn.l2_normalize(token_embeddings, axis=-1)  # [batch, num_tokens, proj_dim]
    bag_embeddings = tf.nn.l2_normalize(bag_embeddings, axis=-1)      # [batch, proj_dim]

    # Get dimensions
    batch_size = tf.shape(token_embeddings)[0]
    num_tokens = tf.shape(token_embeddings)[1]
    proj_dim = tf.shape(token_embeddings)[2]

    # Compute similarity matrix: [batch, num_tokens, batch]
    # For each token in each sample, compute similarity to all bag embeddings
    # Reshape tokens: [batch * num_tokens, proj_dim]
    tokens_flat = tf.reshape(token_embeddings, [batch_size * num_tokens, proj_dim])

    # Compute similarity: [batch * num_tokens, proj_dim] @ [proj_dim, batch] = [batch * num_tokens, batch]
    bag_embeddings_transposed = tf.transpose(bag_embeddings)  # [proj_dim, batch]
    similarity_matrix_flat = tf.matmul(tokens_flat, bag_embeddings_transposed)  # [batch * num_tokens, batch]

    # Reshape back: [batch, num_tokens, batch]
    similarity_matrix = tf.reshape(similarity_matrix_flat, [batch_size, num_tokens, batch_size])

    # Store raw similarities for margin computation
    similarity_matrix_raw = similarity_matrix

    # Apply temperature scaling
    similarity_matrix = similarity_matrix / temperature

    # Extract positive similarities (diagonal): for each token, its own sample's bag
    # Create batch indices for gathering: [batch, num_tokens]
    batch_indices = tf.range(batch_size)  # [batch]
    batch_indices = tf.expand_dims(batch_indices, axis=1)  # [batch, 1]
    batch_indices = tf.tile(batch_indices, [1, num_tokens])  # [batch, num_tokens]

    # Create token indices: [batch, num_tokens]
    token_indices = tf.range(num_tokens)  # [num_tokens]
    token_indices = tf.expand_dims(token_indices, axis=0)  # [1, num_tokens]
    token_indices = tf.tile(token_indices, [batch_size, 1])  # [batch, num_tokens]

    # For each position [i, j], we want similarity_matrix[i, j, i] (token j from sample i vs bag i)
    # Use advanced indexing with gather_nd, reusing batch_indices for diagonal
    indices = tf.stack([
        batch_indices,   # batch dim (reuse computed indices)
        token_indices,   # token dim
        batch_indices    # bag dim (diagonal - same as batch)
    ], axis=-1)  # [batch, num_tokens, 3]

    positive_sims = tf.gather_nd(similarity_matrix, indices)  # [batch, num_tokens]

    # Compute InfoNCE loss manually for XLA compatibility
    # exp_sim: [batch, num_tokens, batch]
    exp_sim = tf.exp(similarity_matrix)
    sum_exp = tf.reduce_sum(exp_sim, axis=2)  # [batch, num_tokens] - sum over bags

    # Positive probabilities: exp(positive_sim) / sum_exp
    exp_positive = tf.exp(positive_sims)  # [batch, num_tokens]
    positive_probs = exp_positive / (sum_exp + 1e-7)  # [batch, num_tokens]

    # Loss per token: -log(positive_prob)
    loss_per_token = -tf.math.log(positive_probs + 1e-7)  # [batch, num_tokens]

    # Apply mask to exclude padded tokens
    token_mask_float = tf.cast(token_mask, tf.float32)  # [batch, num_tokens]
    masked_loss = loss_per_token * token_mask_float

    # Calculate final loss based on per_sample_loss setting
    if per_sample_loss:
        # Per-sample loss: average token losses within each sample, then average across samples
        # This gives equal weight to each sample regardless of number of tokens
        sample_loss_sum = tf.reduce_sum(masked_loss, axis=1)  # [batch]
        sample_token_counts = tf.reduce_sum(token_mask_float, axis=1)  # [batch]
        sample_mean_loss = sample_loss_sum / (sample_token_counts + tf.keras.backend.epsilon())  # [batch]

        # Final loss is mean across samples
        loss = tf.reduce_mean(sample_mean_loss)

        # For margin computation with per-sample loss, also normalize per-sample
        positive_sims_raw = tf.gather_nd(similarity_matrix_raw, indices)  # [batch, num_tokens]
        masked_positive_sims = positive_sims_raw * token_mask_float

        # Compute mean positive similarity per sample, then across samples
        sample_positive_sims_sum = tf.reduce_sum(masked_positive_sims, axis=1)  # [batch]
        sample_positive_sims_mean = sample_positive_sims_sum / (sample_token_counts + tf.keras.backend.epsilon())  # [batch]
        positive_sims_mean = tf.reduce_mean(sample_positive_sims_mean)
    else:
        # Per-token loss: all tokens weighted equally
        num_valid_tokens = tf.reduce_sum(token_mask_float) + tf.keras.backend.epsilon()
        loss = tf.reduce_sum(masked_loss) / num_valid_tokens

        # Compute margin (dimension-invariant alignment metric)
        # Positive similarities: diagonal elements from raw similarity matrix
        positive_sims_raw = tf.gather_nd(similarity_matrix_raw, indices)  # [batch, num_tokens]

        # Apply mask and compute mean positive similarity
        masked_positive_sims = positive_sims_raw * token_mask_float
        positive_sims_mean = tf.reduce_sum(masked_positive_sims) / num_valid_tokens

    # Negative similarities: off-diagonal elements
    # Create mask for off-diagonal: [batch, num_tokens, batch]
    # For each token, negative bags are all bags except its own
    diagonal_mask = tf.eye(batch_size)  # [batch, batch]
    diagonal_mask = tf.expand_dims(diagonal_mask, axis=1)  # [batch, 1, batch]
    off_diagonal_mask = 1.0 - diagonal_mask  # [batch, 1, batch]

    # Expand token mask for broadcasting
    token_mask_expanded = tf.expand_dims(token_mask_float, axis=2)  # [batch, num_tokens, 1]

    # Combined mask: valid tokens AND off-diagonal
    negative_mask = token_mask_expanded * off_diagonal_mask  # [batch, num_tokens, batch]

    # Apply mask and compute mean negative similarity
    masked_negative_sims = similarity_matrix_raw * negative_mask
    num_negative_pairs = tf.reduce_sum(negative_mask) + 1e-7
    negative_sims_mean = tf.reduce_sum(masked_negative_sims) / num_negative_pairs

    # Margin: positive should be high, negative should be low
    margin = positive_sims_mean - negative_sims_mean

    return loss, margin