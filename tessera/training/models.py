"""Custom Keras Model subclasses with TESSERA training logic.

Defines CustomTrainingModel — the standard joint-SNV/CNA training graph that
applies masked-token reconstruction loss (variant + CNA) plus optional InfoNCE
contrastive loss in a single XLA-compilable train_step / test_step — and
CustomTrainingModelMIL, an MIL variant for bag-level supervision. Also exposes
PearsonCorrelationMetric for evaluating CNA segment-mean regression.
"""

import tensorflow as tf
from tessera.training.losses import compute_loss, cna_logfold_loss, multimodal_cna_loss, compute_infonce_loss


class PearsonCorrelationMetric(tf.keras.metrics.Metric):
    """
    Custom metric that computes Pearson correlation using incremental statistics.

    This metric uses incremental computation to calculate the EXACT Pearson correlation
    across all data points, making it fully compatible with XLA/JIT compilation.

    Unlike Spearman (which requires global ranking), Pearson can be computed accurately
    by accumulating sufficient statistics (sum_x, sum_y, sum_x², sum_y², sum_xy, count)
    across batches and then computing the final correlation.

    This metric is well-aligned with MAE loss, as both measure linear relationships:
    - MAE measures linear distance between predictions and truth
    - Pearson measures strength of linear correlation

    Formula:
        r = (n*Σxy - Σx*Σy) / sqrt((n*Σx² - (Σx)²) * (n*Σy² - (Σy)²))

    Args:
        name: Name of the metric (default: 'pearson')
    """

    def __init__(self, name='pearson', **kwargs):
        super().__init__(name=name, **kwargs)
        # Accumulate sufficient statistics for exact Pearson computation
        # Use shape=() explicitly to avoid macOS HDF5 serialization issues
        self.sum_x = self.add_weight(name='sum_x', shape=(), initializer='zeros', dtype=tf.float32)
        self.sum_y = self.add_weight(name='sum_y', shape=(), initializer='zeros', dtype=tf.float32)
        self.sum_x2 = self.add_weight(name='sum_x2', shape=(), initializer='zeros', dtype=tf.float32)
        self.sum_y2 = self.add_weight(name='sum_y2', shape=(), initializer='zeros', dtype=tf.float32)
        self.sum_xy = self.add_weight(name='sum_xy', shape=(), initializer='zeros', dtype=tf.float32)
        self.count = self.add_weight(name='count', shape=(), initializer='zeros', dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Accumulate statistics from this batch for Pearson computation.

        Args:
            y_true: True values (already filtered for valid segments)
            y_pred: Predicted values (already filtered for valid segments)
            sample_weight: Ignored (not used for Pearson)
        """
        # Flatten to 1D and cast to float32
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)

        # Accumulate sufficient statistics
        self.sum_x.assign_add(tf.reduce_sum(y_true))
        self.sum_y.assign_add(tf.reduce_sum(y_pred))
        self.sum_x2.assign_add(tf.reduce_sum(tf.square(y_true)))
        self.sum_y2.assign_add(tf.reduce_sum(tf.square(y_pred)))
        self.sum_xy.assign_add(tf.reduce_sum(y_true * y_pred))
        self.count.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))

    def result(self):
        """
        Compute exact Pearson correlation from accumulated statistics.

        Returns:
            Pearson correlation coefficient (scalar) in range [-1, 1]
        """
        # Pearson formula: r = (n*Σxy - Σx*Σy) / sqrt((n*Σx² - (Σx)²) * (n*Σy² - (Σy)²))
        n = self.count
        numerator = n * self.sum_xy - self.sum_x * self.sum_y
        denominator = tf.sqrt(
            tf.maximum(
                (n * self.sum_x2 - tf.square(self.sum_x)) *
                (n * self.sum_y2 - tf.square(self.sum_y)),
                1e-20  # Prevent negative values under sqrt
            )
        )

        # Compute Pearson, handling edge cases with tf.where
        # Use safe division: if denominator is too small, return 0
        safe_denom = tf.maximum(denominator, 1e-10)
        pearson = numerator / safe_denom

        # Return 0 if no data accumulated or denominator is too small
        has_enough_data = self.count > 1.0
        denom_is_valid = denominator > 1e-10

        # Use tf.where for conditional selection (more compatible with Keras)
        result = tf.where(
            tf.logical_and(has_enough_data, denom_is_valid),
            pearson,
            0.0  # tf.where automatically broadcasts scalars
        )

        # Wrap in tf.identity and ensure it's a scalar float32 tensor
        # This helps Keras process the metric correctly
        return tf.identity(tf.cast(result, tf.float32))

    def reset_state(self):
        """Reset all accumulators at the start of each epoch."""
        self.sum_x.assign(0.0)
        self.sum_y.assign(0.0)
        self.sum_x2.assign(0.0)
        self.sum_y2.assign(0.0)
        self.sum_xy.assign(0.0)
        self.count.assign(0.0)

    def get_config(self):
        """Return configuration for serialization."""
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        """Create metric from configuration."""
        return cls(**config)


class CustomTrainingModel(tf.keras.Model):
    """
    Custom training model with adaptive multi-modal loss computation.

    This model wraps a base model and automatically adapts its loss computation
    based on the available data modalities (mutations, CNA, or both). It handles
    the training step, gradient computation, and metric tracking.

    All losses are simply added together with no weighting for simplicity.

    Args:
        base_model (tf.keras.Model): The underlying model to train
        use_mut (bool): Whether mutation data is available
        use_cna (bool): Whether CNA data is available
        train_on_mutation_loss (bool): Whether to train on mutation loss. If False, mutation loss
                                      is computed for metrics but does not contribute to gradients (default: True)
        train_on_cna_loss (bool): Whether to train on CNA loss. If False, CNA loss
                                 is computed for metrics but does not contribute to gradients (default: True)
        loss_non_zero_only (bool): If True, only compute mutation loss for non-zero elements
        hinge_loss_t (float, optional): Threshold for hinge loss on mutations. If None, all variants contribute.
        per_sample_loss (bool): If True, compute per-sample loss for fair weighting across samples
        accuracy_non_zero_only (bool): Whether to only compute accuracy on non-zero positions
        model_dir (str, optional): Directory for saving gradient logs
        log_gradients (bool): Whether to log gradient norms for debugging.
                             Requires jit_compile=False (XLA/JIT compilation must be disabled)
        predict_cna_loh (bool): Whether to predict CNA LOH in addition to segment_mean (default: False)
        **kwargs: Additional arguments passed to the parent tf.keras.Model
    """
    def __init__(self, base_model, use_mut, use_cna,
                 train_on_mutation_loss=True, train_on_cna_loss=True,
                 loss_non_zero_only=False, hinge_loss_t=None, per_sample_loss=False,
                 accuracy_non_zero_only=False, model_dir=None, log_gradients=False,
                 # CNA dual-task parameters
                 predict_cna_loh=False,
                 # InfoNCE loss parameters
                 use_infonce_loss=False,
                 infonce_temperature=0.1,
                 infonce_loss_weight=1.0,
                 # Token-bag InfoNCE parameters (independent per modality)
                 use_mut_token_bag_infonce=False,
                 use_cna_token_bag_infonce=False,
                 token_bag_temperature=0.1,
                 token_bag_loss_weight=1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.use_mut = use_mut
        self.use_cna = use_cna
        self.train_on_mutation_loss = train_on_mutation_loss
        self.train_on_cna_loss = train_on_cna_loss
        self.loss_non_zero_only = loss_non_zero_only
        self.hinge_loss_t = hinge_loss_t
        self.per_sample_loss = per_sample_loss
        self.accuracy_non_zero_only = accuracy_non_zero_only

        # CNA dual-task parameters
        self.predict_cna_loh = predict_cna_loh

        # InfoNCE loss parameters
        self.use_infonce_loss = use_infonce_loss
        self.infonce_temperature = infonce_temperature
        self.infonce_loss_weight = infonce_loss_weight

        # Token-bag InfoNCE parameters (independent per modality)
        self.use_mut_token_bag_infonce = use_mut_token_bag_infonce
        self.use_cna_token_bag_infonce = use_cna_token_bag_infonce
        self.token_bag_temperature = token_bag_temperature
        self.token_bag_loss_weight = token_bag_loss_weight

        # Import accuracy functions directly
        from tessera.training.metrics import masked_accuracy
        self.masked_accuracy_fn = masked_accuracy
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

        # InfoNCE loss and alignment metrics (only if using both modalities and InfoNCE is enabled)
        if self.use_mut and self.use_cna and self.use_infonce_loss:
            self.infonce_loss_tracker = tf.keras.metrics.Mean(name="infonce_loss")
            self.infonce_margin_tracker = tf.keras.metrics.Mean(name="infonce_margin")

        # Token-bag InfoNCE metrics (independent per modality)
        if self.use_mut and self.use_mut_token_bag_infonce:
            self.token_bag_mut_loss_tracker = tf.keras.metrics.Mean(name="token_bag_mut_loss")
            self.token_bag_mut_margin_tracker = tf.keras.metrics.Mean(name="token_bag_mut_margin")

        if self.use_cna and self.use_cna_token_bag_infonce:
            self.token_bag_cna_loss_tracker = tf.keras.metrics.Mean(name="token_bag_cna_loss")
            self.token_bag_cna_margin_tracker = tf.keras.metrics.Mean(name="token_bag_cna_margin")

        # Create mutation loss and accuracy trackers (only if using mutations)
        if self.use_mut:
            self.mutation_loss_tracker = tf.keras.metrics.Mean(name="mutation_loss")
            self.accuracy_tracker = tf.keras.metrics.Mean(name="accuracy")
            self.sample_avg_accuracy_tracker = tf.keras.metrics.Mean(name="sample_avg_accuracy")

        # Create CNA metric trackers (only if using CNA)
        if self.use_cna:
            self.cna_loss_tracker = tf.keras.metrics.Mean(name="cna_loss")
            self.cna_segment_mean_loss_tracker = tf.keras.metrics.Mean(name="cna_segment_mean_loss")
            self.cna_pearson_tracker = PearsonCorrelationMetric(name="cna_pearson")

            # Separate validation tracker (Keras doesn't auto-create for custom metrics)
            self.val_cna_pearson_tracker = PearsonCorrelationMetric(name="cna_pearson")

            # LOH-specific metrics (only if predicting LOH)
            if self.predict_cna_loh:
                self.cna_loh_loss_tracker = tf.keras.metrics.Mean(name="cna_loh_loss")
                # self.cna_loh_accuracy_tracker = tf.keras.metrics.Mean(name="cna_loh_accuracy")
                self.cna_loh_auc_tracker = tf.keras.metrics.AUC(name="cna_loh_auc")

        self.model_dir = model_dir  # Store model directory for gradient logging
        self.log_gradients = log_gradients  # Enable/disable gradient logging
        self.print_to_terminal = True

        # For gradient norm logging - use add_weight for proper device placement
        if self.log_gradients:
            self.step_count = self.add_weight(
                name="gradient_step_count",
                shape=(),
                dtype=tf.int32,
                initializer='zeros',
                trainable=False
            )
            self.log_every_n_steps = 50
            self.gradient_df = None  # Pandas DataFrame to store gradient norms
            self.layer_names = None  # Will be set on first call

            # Print requirement about XLA
            print("\n" + "="*80)
            print("Gradient logging is ENABLED.")
            print("")
            print("IMPORTANT: Gradient logging requires jit_compile=False")
            print("           (XLA/JIT compilation must be disabled)")
            print("")
            print("Gradient norms will be logged to: %s" % (
                self.model_dir + '/gradient_norms.csv' if self.model_dir else 'gradient_norms.csv'))
            print("="*80 + "\n")

    def call(self, inputs, training=False):
        """
        Forward pass through the base model.

        Args:
            inputs: Input data to the model
            training (bool): Whether the model is in training mode. Default: False

        Returns:
            Output from the base model
        """
        return self.base_model(inputs, training=training)

    def _compute_mutation_loss(self, y_true_alt, logits):
        """
        Compute mutation reconstruction loss.

        Args:
            y_true_alt: True mutation labels [batch, num_variants, num_features]
            logits: Predicted mutation logits [batch, num_variants, num_features]

        Returns:
            tf.Tensor: Mutation loss value (scalar)
        """
        return compute_loss(
            y_true=y_true_alt,
            logits=logits,
            non_zero_only=self.loss_non_zero_only,
            hinge_loss_t=self.hinge_loss_t,
            per_sample_loss=self.per_sample_loss
        )

    def _compute_cna_loss(self, x, y_pred):
        """
        Compute CNA prediction loss (dual-task: segment_mean + optional LOH).

        Args:
            x: Input data dictionary containing 'cna_segment_mean', 'cna_chr', and optionally 'cna_loh'
            y_pred: Model predictions dictionary containing 'cna_segment_mean_pred' and optionally 'cna_loh_pred'

        Returns:
            tuple: (total_cna_loss, segment_mean_loss, loh_loss) where loh_loss is 0.0 if not predicting LOH
        """
        # Extract predictions
        segment_mean_pred = y_pred.get('cna_segment_mean_pred', y_pred.get('cna_pred'))  # Try new key, fallback to legacy
        loh_pred = y_pred.get('cna_loh_pred') if self.predict_cna_loh else None

        # Extract ground truth - loss functions will handle shape normalization
        segment_mean_true = x['cna_segment_mean']
        loh_true = x.get('cna_loh') if self.predict_cna_loh else None

        # Compute multi-modal loss (segment_mean + LOH losses are simply added)
        total_loss, segment_mean_loss, loh_loss = multimodal_cna_loss(
            segment_mean_y_true=segment_mean_true,
            segment_mean_y_pred=segment_mean_pred,
            loh_y_true=loh_true,
            loh_y_pred=loh_pred,
            cna_chr=x['cna_chr'],
            per_sample_loss=self.per_sample_loss
        )

        return total_loss, segment_mean_loss, loh_loss

    def _compute_dual_accuracy(self, per_variant_acc, variant_mask):
        """
        Compute both overall accuracy and sample average accuracy from per-variant accuracies.

        Args:
            per_variant_acc: Tensor of shape [batch, num_variants] with per-variant accuracies
            variant_mask: Tensor of shape [batch, num_variants] indicating valid variants

        Returns:
            tuple: (overall_accuracy, sample_avg_accuracy)
                - overall_accuracy: Mean accuracy across all valid variants (each variant weighted equally)
                - sample_avg_accuracy: Mean of per-sample accuracies (each sample weighted equally)
        """
        # Overall accuracy: each variant weighted equally
        overall_accuracy = tf.reduce_sum(per_variant_acc * variant_mask) / (
            tf.reduce_sum(variant_mask) + tf.keras.backend.epsilon())

        # Sample average accuracy: each sample weighted equally
        # First compute accuracy per sample: sum of variant accuracies / number of valid variants per sample
        sample_accuracy_sum = tf.reduce_sum(per_variant_acc * variant_mask, axis=-1)  # [batch]
        sample_variant_counts = tf.reduce_sum(variant_mask, axis=-1)  # [batch]
        sample_accuracy = sample_accuracy_sum / (sample_variant_counts + tf.keras.backend.epsilon())  # [batch]

        # Then average across samples
        sample_avg_accuracy = tf.reduce_mean(sample_accuracy)

        return overall_accuracy, sample_avg_accuracy

    def train_step(self, data):
        """
        Performs one training step with custom loss and accuracy computation.

        Args:
            data (dict): Training data dictionary containing input features.
                        Expected to have 'alt' key for variant labels and potentially
                        'cna_segment_mean' for CNA labels.

        Returns:
            dict: Dictionary containing computed loss and accuracy metrics
        """
        x = data  # Assume `x` contains 'ref', 'alt', etc.

        with tf.GradientTape() as tape:
            # Forward pass through the base model
            y_pred = self.base_model(x, training=True)

            # Compute mutation and CNA losses separately, then combine
            mutation_loss = 0.0
            cna_loss = 0.0

            # Check for dual reconstruction case (recon_ref=True)
            if isinstance(y_pred, dict) and 'logits_ref' in y_pred:
                # Dual reconstruction: compute mutation loss for both alt and ref (only if training on it)
                if self.use_mut and self.train_on_mutation_loss:
                    mut_loss_alt = self._compute_mutation_loss(x['alt'], y_pred['logits'])
                    mut_loss_ref = self._compute_mutation_loss(x['ref'], y_pred['logits_ref'])
                    mutation_loss = (mut_loss_alt + mut_loss_ref) / 2.0

                # CNA loss is computed once (not duplicated for dual reconstruction, only if training on it)
                if self.use_cna and self.train_on_cna_loss:
                    cna_total_loss, cna_segment_mean_loss, cna_loh_loss = self._compute_cna_loss(x, y_pred)
                    cna_loss = cna_total_loss
            else:
                # Standard single output case (only compute losses if training on them)
                if self.use_mut and self.train_on_mutation_loss:
                    mutation_loss = self._compute_mutation_loss(x['alt'], y_pred['logits'])

                if self.use_cna and self.train_on_cna_loss:
                    cna_total_loss, cna_segment_mean_loss, cna_loh_loss = self._compute_cna_loss(x, y_pred)
                    cna_loss = cna_total_loss

            # Combine losses by simple addition (no weighting)
            # Note: Losses are still computed above for metrics even if not being trained on
            if self.train_on_mutation_loss and self.train_on_cna_loss:
                total_loss = mutation_loss + cna_loss
            elif self.train_on_mutation_loss:
                total_loss = mutation_loss
            elif self.train_on_cna_loss:
                total_loss = cna_loss
            else:
                raise ValueError("At least one of train_on_mutation_loss or train_on_cna_loss must be True")

            # Compute InfoNCE loss if enabled and both modal embeddings are present
            infonce_loss = 0.0
            infonce_margin = 0.0
            if self.use_infonce_loss and 'mut_sample_embedding' in y_pred and 'cna_sample_embedding' in y_pred:
                mut_embedding = y_pred['mut_sample_embedding']
                cna_embedding = y_pred['cna_sample_embedding']

                # Create mask for samples with both mutation and CNA data
                # A sample has mutation data if it has at least one non-padding variant (chr != 0)
                # A sample has CNA data if it has at least one non-padding segment (cna_chr != 0)
                batch_size = tf.shape(x['chr'])[0]

                # Check if each sample has mutation data
                has_mut_data = tf.reduce_any(tf.not_equal(x['chr'], 0), axis=[1, 2])  # [batch]

                # Check if each sample has CNA data
                has_cna_data = tf.reduce_any(tf.not_equal(x['cna_chr'], 0), axis=[1, 2])  # [batch]

                # Mask for samples with BOTH modalities
                valid_pairs_mask = tf.logical_and(has_mut_data, has_cna_data)  # [batch]

                infonce_loss, infonce_margin = compute_infonce_loss(
                    mut_embedding, cna_embedding, self.infonce_temperature,
                    valid_pairs_mask=valid_pairs_mask
                )
                total_loss = total_loss + self.infonce_loss_weight * infonce_loss

            # Compute token-bag InfoNCE losses (independent per modality)
            token_bag_mut_loss = 0.0
            token_bag_mut_margin = 0.0
            token_bag_cna_loss = 0.0
            token_bag_cna_margin = 0.0

            # Import the token-bag loss function if needed
            if self.use_mut_token_bag_infonce or self.use_cna_token_bag_infonce:
                from tessera.training.losses import compute_token_bag_infonce_loss

            # Mutation token-bag InfoNCE
            if self.use_mut_token_bag_infonce and 'mut_token_embedding' in y_pred and 'mut_token_bag_embedding' in y_pred:
                mut_token_mask = tf.not_equal(x['chr'], 0)
                if len(mut_token_mask.shape) == 3:
                    mut_token_mask = tf.squeeze(mut_token_mask, axis=-1)
                token_bag_mut_loss, token_bag_mut_margin = compute_token_bag_infonce_loss(
                    token_embeddings=y_pred['mut_token_embedding'],
                    bag_embeddings=y_pred['mut_token_bag_embedding'],
                    token_mask=mut_token_mask,
                    temperature=self.token_bag_temperature,
                    per_sample_loss=self.per_sample_loss
                )
                total_loss = total_loss + self.token_bag_loss_weight * token_bag_mut_loss

            # CNA token-bag InfoNCE
            if self.use_cna_token_bag_infonce and 'cna_token_embedding' in y_pred and 'cna_token_bag_embedding' in y_pred:
                cna_token_mask = tf.not_equal(x['cna_chr'], 0)
                if len(cna_token_mask.shape) == 3:
                    cna_token_mask = tf.squeeze(cna_token_mask, axis=-1)
                token_bag_cna_loss, token_bag_cna_margin = compute_token_bag_infonce_loss(
                    token_embeddings=y_pred['cna_token_embedding'],
                    bag_embeddings=y_pred['cna_token_bag_embedding'],
                    token_mask=cna_token_mask,
                    temperature=self.token_bag_temperature,
                    per_sample_loss=self.per_sample_loss
                )
                total_loss = total_loss + self.token_bag_loss_weight * token_bag_cna_loss

            # Add regularization losses (L1 attention regularization, etc.)
            if self.base_model.losses:
                total_loss_print = total_loss
                regularization_loss = tf.add_n(self.base_model.losses)
                total_loss = total_loss + regularization_loss
            else:
                total_loss_print = total_loss

        # Update the custom loss metric
        self.loss_tracker.update_state(total_loss_print)

        # Update separate loss trackers for mutations and CNAs (only if computed)
        if self.use_mut and self.train_on_mutation_loss:
            self.mutation_loss_tracker.update_state(mutation_loss)
        if self.use_cna and self.train_on_cna_loss:
            self.cna_loss_tracker.update_state(cna_total_loss)
            self.cna_segment_mean_loss_tracker.update_state(cna_segment_mean_loss)
            if self.predict_cna_loh:
                self.cna_loh_loss_tracker.update_state(cna_loh_loss)

        # Update InfoNCE loss and margin trackers if computed
        if self.use_infonce_loss:
            self.infonce_loss_tracker.update_state(infonce_loss)
            self.infonce_margin_tracker.update_state(infonce_margin)

        # Update token-bag InfoNCE loss and margin trackers if computed (independent per modality)
        if hasattr(self, 'token_bag_mut_loss_tracker'):
            self.token_bag_mut_loss_tracker.update_state(token_bag_mut_loss)
            self.token_bag_mut_margin_tracker.update_state(token_bag_mut_margin)
        if hasattr(self, 'token_bag_cna_loss_tracker'):
            self.token_bag_cna_loss_tracker.update_state(token_bag_cna_loss)
            self.token_bag_cna_margin_tracker.update_state(token_bag_cna_margin)

        # Compute gradients - LossScaleOptimizer handles scaling automatically
        gradients = tape.gradient(total_loss, self.trainable_variables)

        # Log gradient norms for debugging (if enabled)
        # Note: Requires jit_compile=False (XLA/JIT must be disabled)
        if self.log_gradients:
            self._log_gradient_norms(gradients)

        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Build result metrics dict starting with loss (always present)
        result_metrics = {"loss": self.loss_tracker.result()}

        # Add separate loss metrics if applicable (only if training on them)
        if self.use_mut and self.train_on_mutation_loss:
            result_metrics["mutation_loss"] = self.mutation_loss_tracker.result()
        if self.use_cna and self.train_on_cna_loss:
            result_metrics["cna_loss"] = self.cna_loss_tracker.result()
            result_metrics["cna_segment_mean_loss"] = self.cna_segment_mean_loss_tracker.result()

        # Add InfoNCE loss and margin metrics if computed
        if self.use_infonce_loss:
            result_metrics["infonce_loss"] = self.infonce_loss_tracker.result()
            result_metrics["infonce_margin"] = self.infonce_margin_tracker.result()

        # Add token-bag InfoNCE metrics if computed (independent per modality)
        if hasattr(self, 'token_bag_mut_loss_tracker'):
            result_metrics["token_bag_mut_loss"] = self.token_bag_mut_loss_tracker.result()
            result_metrics["token_bag_mut_margin"] = self.token_bag_mut_margin_tracker.result()
        if hasattr(self, 'token_bag_cna_loss_tracker'):
            result_metrics["token_bag_cna_loss"] = self.token_bag_cna_loss_tracker.result()
            result_metrics["token_bag_cna_margin"] = self.token_bag_cna_margin_tracker.result()

        # Compute mutation accuracy metrics (only if using mutations AND training on them)
        if self.use_mut and self.train_on_mutation_loss:
            # Check for dual reconstruction case (recon_ref=True)
            if isinstance(y_pred, dict) and 'logits_ref' in y_pred:
                # Dual reconstruction: compute accuracy for both alt and ref
                alt_acc_per_variant = self.masked_accuracy_fn(x['alt'], y_pred['logits'], non_zero_only=self.accuracy_non_zero_only)
                ref_acc_per_variant = self.masked_accuracy_fn(x['ref'], y_pred['logits_ref'], non_zero_only=self.accuracy_non_zero_only)

                # Variant is correct only if BOTH alt and ref are correct
                dual_acc_per_variant = alt_acc_per_variant * ref_acc_per_variant

                # Compute both overall and sample-average accuracy
                variant_mask = tf.reduce_any(tf.not_equal(x['alt'], 0), axis=-1)  # [batch, num_variants]
                variant_mask = tf.cast(variant_mask, dual_acc_per_variant.dtype)
                overall_accuracy, sample_avg_accuracy = self._compute_dual_accuracy(dual_acc_per_variant, variant_mask)
            else:
                # Standard single reconstruction
                y_true = x['alt']
                per_variant_acc = self.masked_accuracy_fn(y_true, y_pred['logits'], non_zero_only=self.accuracy_non_zero_only)

                # Compute both overall and sample-average accuracy
                variant_mask = tf.reduce_any(tf.not_equal(y_true, 0), axis=-1)  # [batch, num_variants]
                variant_mask = tf.cast(variant_mask, per_variant_acc.dtype)
                overall_accuracy, sample_avg_accuracy = self._compute_dual_accuracy(per_variant_acc, variant_mask)

            self.accuracy_tracker.update_state(overall_accuracy)
            self.sample_avg_accuracy_tracker.update_state(sample_avg_accuracy)

            result_metrics["accuracy"] = self.accuracy_tracker.result()
            result_metrics["sample_avg_accuracy"] = self.sample_avg_accuracy_tracker.result()

        # Compute CNA metrics (only if training on CNA to save computation)
        if self.use_cna and 'cna_pred' in y_pred and self.train_on_cna_loss:
            # Extract segment_mean predictions and ground truth
            segment_mean_pred = y_pred.get('cna_segment_mean_pred', y_pred.get('cna_pred'))
            segment_mean_true = x['cna_segment_mean']
            cna_chr = x['cna_chr']

            # Flatten to 2D and mask valid segments
            if len(segment_mean_true.shape) == 3:
                segment_mean_true = tf.squeeze(segment_mean_true, axis=-1)
            if len(segment_mean_pred.shape) == 3:
                segment_mean_pred = tf.squeeze(segment_mean_pred, axis=-1)
            if len(cna_chr.shape) == 3:
                cna_chr = tf.squeeze(cna_chr, axis=-1)

            # Create mask for valid CNA segments (non-padding)
            valid_mask = tf.not_equal(cna_chr, 0)  # [batch, num_segments]

            # Filter to only valid segments (flatten across batch)
            y_true_valid = tf.boolean_mask(segment_mean_true, valid_mask)
            y_pred_valid = tf.boolean_mask(segment_mean_pred, valid_mask)

            # Update Pearson metric with filtered values
            self.cna_pearson_tracker.update_state(y_true_valid, y_pred_valid)
            result_metrics["cna_pearson"] = self.cna_pearson_tracker.result()

        # Compute LOH metrics if predicting LOH (only if training on CNA)
        if self.predict_cna_loh and 'cna_loh_pred' in y_pred and self.train_on_cna_loss:
            # Get predictions and ground truth - all have shape [batch, num_segments, 1]
            loh_pred = y_pred['cna_loh_pred']  # [batch, num_segments, 1], sigmoid output
            loh_true = x.get('cna_loh')  # [batch, num_segments, 1], binary 0/1
            cna_chr = x['cna_chr']  # [batch, num_segments, 1], for masking

            # Squeeze to 2D [batch, num_segments] for metrics computation
            loh_pred = tf.squeeze(loh_pred, axis=-1)  # [batch, num_segments]
            loh_true = tf.squeeze(loh_true, axis=-1)  # [batch, num_segments]
            cna_chr_flat = tf.squeeze(cna_chr, axis=-1)  # [batch, num_segments]

            # Mask to exclude padding segments (cna_chr != 0)
            valid_mask = tf.cast(tf.not_equal(cna_chr_flat, 0), tf.float32)  # [batch, num_segments]

            # # Compute binary accuracy (rounded predictions vs true labels)
            # loh_pred_binary = tf.round(loh_pred)  # Round 0.5 threshold
            # loh_correct = tf.cast(tf.equal(loh_pred_binary, loh_true), tf.float32)
            # loh_accuracy = tf.reduce_sum(loh_correct * valid_mask) / (tf.reduce_sum(valid_mask) + tf.keras.backend.epsilon())
            #
            # # Update metrics
            # self.cna_loh_accuracy_tracker.update_state(loh_accuracy)

            # Flatten for AUC computation
            loh_pred_flat = tf.reshape(loh_pred, [-1])
            loh_true_flat = tf.reshape(loh_true, [-1])
            valid_mask_flat = tf.reshape(valid_mask, [-1])

            # Filter to only valid segments
            loh_pred_valid = tf.boolean_mask(loh_pred_flat, tf.cast(valid_mask_flat, tf.bool))
            loh_true_valid = tf.boolean_mask(loh_true_flat, tf.cast(valid_mask_flat, tf.bool))

            # Update AUC
            self.cna_loh_auc_tracker.update_state(loh_true_valid, loh_pred_valid)

            # Add to result metrics
            result_metrics["cna_loh_loss"] = self.cna_loh_loss_tracker.result()
            # result_metrics["cna_loh_accuracy"] = self.cna_loh_accuracy_tracker.result()
            result_metrics["cna_loh_auc"] = self.cna_loh_auc_tracker.result()

        return result_metrics

    def test_step(self, data):
        """
        Performs one validation/test step.
        Uses same trackers as train_step so Keras automatically creates val_ prefixed metrics.

        Args:
            data: Validation batch data containing input features and labels

        Returns:
            dict: Validation metrics
        """
        x = data

        # Forward pass in inference mode
        y_pred = self.base_model(x, training=False)

        # Compute mutation and CNA losses separately, then combine
        mutation_loss = 0.0
        cna_loss = 0.0

        # Check for dual reconstruction case (recon_ref=True)
        if isinstance(y_pred, dict) and 'logits_ref' in y_pred:
            # Dual reconstruction: compute mutation loss for both alt and ref (only if training on it)
            if self.use_mut and self.train_on_mutation_loss:
                mut_loss_alt = self._compute_mutation_loss(x['alt'], y_pred['logits'])
                mut_loss_ref = self._compute_mutation_loss(x['ref'], y_pred['logits_ref'])
                mutation_loss = (mut_loss_alt + mut_loss_ref) / 2.0

            # CNA loss is computed once (not duplicated for dual reconstruction, only if training on it)
            if self.use_cna and self.train_on_cna_loss:
                cna_total_loss, cna_segment_mean_loss, cna_loh_loss = self._compute_cna_loss(x, y_pred)
                cna_loss = cna_total_loss
        else:
            # Standard single output case (only compute losses if training on them)
            if self.use_mut and self.train_on_mutation_loss:
                mutation_loss = self._compute_mutation_loss(x['alt'], y_pred['logits'])

            # Only compute CNA loss if we're training on it (saves computation)
            if self.use_cna and self.train_on_cna_loss:
                cna_total_loss, cna_segment_mean_loss, cna_loh_loss = self._compute_cna_loss(x, y_pred)
                cna_loss = cna_total_loss

        # Combine losses by simple addition (no weighting)
        if self.train_on_mutation_loss and self.train_on_cna_loss:
            total_loss = mutation_loss + cna_loss
        elif self.train_on_mutation_loss:
            total_loss = mutation_loss
        elif self.train_on_cna_loss:
            total_loss = cna_loss
        else:
            raise ValueError("At least one of train_on_mutation_loss or train_on_cna_loss must be True")

        # Compute InfoNCE loss if enabled and both modal embeddings are present
        infonce_loss = 0.0
        infonce_margin = 0.0
        if self.use_infonce_loss and 'mut_sample_embedding' in y_pred and 'cna_sample_embedding' in y_pred:
            mut_embedding = y_pred['mut_sample_embedding']
            cna_embedding = y_pred['cna_sample_embedding']

            # Create mask for samples with both mutation and CNA data
            # A sample has mutation data if it has at least one non-padding variant (chr != 0)
            # A sample has CNA data if it has at least one non-padding segment (cna_chr != 0)
            batch_size = tf.shape(x['chr'])[0]

            # Check if each sample has mutation data
            has_mut_data = tf.reduce_any(tf.not_equal(x['chr'], 0), axis=[1, 2])  # [batch]

            # Check if each sample has CNA data
            has_cna_data = tf.reduce_any(tf.not_equal(x['cna_chr'], 0), axis=[1, 2])  # [batch]

            # Mask for samples with BOTH modalities
            valid_pairs_mask = tf.logical_and(has_mut_data, has_cna_data)  # [batch]

            infonce_loss, infonce_margin = compute_infonce_loss(
                mut_embedding, cna_embedding, self.infonce_temperature,
                valid_pairs_mask=valid_pairs_mask
            )
            total_loss = total_loss + self.infonce_loss_weight * infonce_loss

        # Compute token-bag InfoNCE losses (independent per modality)
        token_bag_mut_loss = 0.0
        token_bag_mut_margin = 0.0
        token_bag_cna_loss = 0.0
        token_bag_cna_margin = 0.0

        # Import the token-bag loss function if needed
        if self.use_mut_token_bag_infonce or self.use_cna_token_bag_infonce:
            from tessera.training.losses import compute_token_bag_infonce_loss

        # Mutation token-bag InfoNCE
        if self.use_mut_token_bag_infonce and 'mut_token_embedding' in y_pred and 'mut_token_bag_embedding' in y_pred:
            mut_token_mask = tf.not_equal(x['chr'], 0)
            if len(mut_token_mask.shape) == 3:
                mut_token_mask = tf.squeeze(mut_token_mask, axis=-1)
            token_bag_mut_loss, token_bag_mut_margin = compute_token_bag_infonce_loss(
                token_embeddings=y_pred['mut_token_embedding'],
                bag_embeddings=y_pred['mut_token_bag_embedding'],
                token_mask=mut_token_mask,
                temperature=self.token_bag_temperature,
                per_sample_loss=self.per_sample_loss
            )
            total_loss = total_loss + self.token_bag_loss_weight * token_bag_mut_loss

        # CNA token-bag InfoNCE
        if self.use_cna_token_bag_infonce and 'cna_token_embedding' in y_pred and 'cna_token_bag_embedding' in y_pred:
            cna_token_mask = tf.not_equal(x['cna_chr'], 0)
            if len(cna_token_mask.shape) == 3:
                cna_token_mask = tf.squeeze(cna_token_mask, axis=-1)
            token_bag_cna_loss, token_bag_cna_margin = compute_token_bag_infonce_loss(
                token_embeddings=y_pred['cna_token_embedding'],
                bag_embeddings=y_pred['cna_token_bag_embedding'],
                token_mask=cna_token_mask,
                temperature=self.token_bag_temperature,
                per_sample_loss=self.per_sample_loss
            )
            total_loss = total_loss + self.token_bag_loss_weight * token_bag_cna_loss

        # Update the custom loss metric
        self.loss_tracker.update_state(total_loss)

        # Update separate loss trackers for mutations and CNAs (only if computed)
        if self.use_mut and self.train_on_mutation_loss:
            self.mutation_loss_tracker.update_state(mutation_loss)
        if self.use_cna and self.train_on_cna_loss:
            self.cna_loss_tracker.update_state(cna_total_loss)
            self.cna_segment_mean_loss_tracker.update_state(cna_segment_mean_loss)
            if self.predict_cna_loh:
                self.cna_loh_loss_tracker.update_state(cna_loh_loss)

        # Update InfoNCE loss and margin trackers if computed
        if self.use_infonce_loss:
            self.infonce_loss_tracker.update_state(infonce_loss)
            self.infonce_margin_tracker.update_state(infonce_margin)

        # Update token-bag InfoNCE loss and margin trackers if computed (independent per modality)
        if hasattr(self, 'token_bag_mut_loss_tracker'):
            self.token_bag_mut_loss_tracker.update_state(token_bag_mut_loss)
            self.token_bag_mut_margin_tracker.update_state(token_bag_mut_margin)
        if hasattr(self, 'token_bag_cna_loss_tracker'):
            self.token_bag_cna_loss_tracker.update_state(token_bag_cna_loss)
            self.token_bag_cna_margin_tracker.update_state(token_bag_cna_margin)

        # Build result metrics dict starting with loss (always present)
        result_metrics = {"loss": self.loss_tracker.result()}

        # Add separate loss metrics if applicable (only if training on them)
        if self.use_mut and self.train_on_mutation_loss:
            result_metrics["mutation_loss"] = self.mutation_loss_tracker.result()
        if self.use_cna and self.train_on_cna_loss:
            result_metrics["cna_loss"] = self.cna_loss_tracker.result()
            result_metrics["cna_segment_mean_loss"] = self.cna_segment_mean_loss_tracker.result()

        # Add InfoNCE loss and margin metrics if computed
        if self.use_infonce_loss:
            result_metrics["infonce_loss"] = self.infonce_loss_tracker.result()
            result_metrics["infonce_margin"] = self.infonce_margin_tracker.result()

        # Add token-bag InfoNCE metrics if computed (independent per modality)
        if hasattr(self, 'token_bag_mut_loss_tracker'):
            result_metrics["token_bag_mut_loss"] = self.token_bag_mut_loss_tracker.result()
            result_metrics["token_bag_mut_margin"] = self.token_bag_mut_margin_tracker.result()
        if hasattr(self, 'token_bag_cna_loss_tracker'):
            result_metrics["token_bag_cna_loss"] = self.token_bag_cna_loss_tracker.result()
            result_metrics["token_bag_cna_margin"] = self.token_bag_cna_margin_tracker.result()

        # Compute mutation accuracy metrics (only if using mutations AND training on them)
        if self.use_mut and self.train_on_mutation_loss:
            # Check for dual reconstruction case (recon_ref=True)
            if isinstance(y_pred, dict) and 'logits_ref' in y_pred:
                # Dual reconstruction: compute accuracy for both alt and ref
                alt_acc_per_variant = self.masked_accuracy_fn(x['alt'], y_pred['logits'], non_zero_only=self.accuracy_non_zero_only)
                ref_acc_per_variant = self.masked_accuracy_fn(x['ref'], y_pred['logits_ref'], non_zero_only=self.accuracy_non_zero_only)

                # Variant is correct only if BOTH alt and ref are correct
                dual_acc_per_variant = alt_acc_per_variant * ref_acc_per_variant

                # Compute both overall and sample-average accuracy
                variant_mask = tf.reduce_any(tf.not_equal(x['alt'], 0), axis=-1)  # [batch, num_variants]
                variant_mask = tf.cast(variant_mask, dual_acc_per_variant.dtype)
                overall_accuracy, sample_avg_accuracy = self._compute_dual_accuracy(dual_acc_per_variant, variant_mask)
            else:
                # Standard single reconstruction
                y_true = x['alt']
                per_variant_acc = self.masked_accuracy_fn(y_true, y_pred['logits'], non_zero_only=self.accuracy_non_zero_only)

                # Compute both overall and sample-average accuracy
                variant_mask = tf.reduce_any(tf.not_equal(y_true, 0), axis=-1)  # [batch, num_variants]
                variant_mask = tf.cast(variant_mask, per_variant_acc.dtype)
                overall_accuracy, sample_avg_accuracy = self._compute_dual_accuracy(per_variant_acc, variant_mask)

            self.accuracy_tracker.update_state(overall_accuracy)
            self.sample_avg_accuracy_tracker.update_state(sample_avg_accuracy)

            result_metrics["accuracy"] = self.accuracy_tracker.result()
            result_metrics["sample_avg_accuracy"] = self.sample_avg_accuracy_tracker.result()

        # Compute CNA metrics (only if training on CNA to save computation)
        if self.use_cna and 'cna_pred' in y_pred and self.train_on_cna_loss:
            # Extract segment_mean predictions and ground truth
            segment_mean_pred = y_pred.get('cna_segment_mean_pred', y_pred.get('cna_pred'))
            segment_mean_true = x['cna_segment_mean']
            cna_chr = x['cna_chr']

            # Flatten to 2D and mask valid segments
            if len(segment_mean_true.shape) == 3:
                segment_mean_true = tf.squeeze(segment_mean_true, axis=-1)
            if len(segment_mean_pred.shape) == 3:
                segment_mean_pred = tf.squeeze(segment_mean_pred, axis=-1)
            if len(cna_chr.shape) == 3:
                cna_chr = tf.squeeze(cna_chr, axis=-1)

            # Create mask for valid CNA segments (non-padding)
            valid_mask = tf.not_equal(cna_chr, 0)  # [batch, num_segments]

            # Filter to only valid segments (flatten across batch)
            y_true_valid = tf.boolean_mask(segment_mean_true, valid_mask)
            y_pred_valid = tf.boolean_mask(segment_mean_pred, valid_mask)

            # Update Pearson metric with filtered values (use validation tracker)
            self.val_cna_pearson_tracker.update_state(y_true_valid, y_pred_valid)
            result_metrics["cna_pearson"] = self.val_cna_pearson_tracker.result()

        # Compute LOH metrics if predicting LOH (only if training on CNA)
        if self.predict_cna_loh and 'cna_loh_pred' in y_pred and self.train_on_cna_loss:
            # Get predictions and ground truth - all have shape [batch, num_segments, 1]
            loh_pred = y_pred['cna_loh_pred']  # [batch, num_segments, 1], sigmoid output
            loh_true = x.get('cna_loh')  # [batch, num_segments, 1], binary 0/1
            cna_chr = x['cna_chr']  # [batch, num_segments, 1], for masking

            # Squeeze to 2D [batch, num_segments] for metrics computation
            loh_pred = tf.squeeze(loh_pred, axis=-1)  # [batch, num_segments]
            loh_true = tf.squeeze(loh_true, axis=-1)  # [batch, num_segments]
            cna_chr_flat = tf.squeeze(cna_chr, axis=-1)  # [batch, num_segments]

            # Mask to exclude padding segments (cna_chr != 0)
            valid_mask = tf.cast(tf.not_equal(cna_chr_flat, 0), tf.float32)  # [batch, num_segments]

            # # Compute binary accuracy (rounded predictions vs true labels)
            # loh_pred_binary = tf.round(loh_pred)  # Round 0.5 threshold
            # loh_correct = tf.cast(tf.equal(loh_pred_binary, loh_true), tf.float32)
            # loh_accuracy = tf.reduce_sum(loh_correct * valid_mask) / (tf.reduce_sum(valid_mask) + tf.keras.backend.epsilon())
            #
            # # Update metrics
            # self.cna_loh_accuracy_tracker.update_state(loh_accuracy)

            # Flatten for AUC computation
            loh_pred_flat = tf.reshape(loh_pred, [-1])
            loh_true_flat = tf.reshape(loh_true, [-1])
            valid_mask_flat = tf.reshape(valid_mask, [-1])

            # Filter to only valid segments
            loh_pred_valid = tf.boolean_mask(loh_pred_flat, tf.cast(valid_mask_flat, tf.bool))
            loh_true_valid = tf.boolean_mask(loh_true_flat, tf.cast(valid_mask_flat, tf.bool))

            # Update AUC
            self.cna_loh_auc_tracker.update_state(loh_true_valid, loh_pred_valid)

            # Add to result metrics
            result_metrics["cna_loh_loss"] = self.cna_loh_loss_tracker.result()
            # result_metrics["cna_loh_accuracy"] = self.cna_loh_accuracy_tracker.result()
            result_metrics["cna_loh_auc"] = self.cna_loh_auc_tracker.result()

        return result_metrics

    def reset_metrics(self):
        """
        Reset all metric trackers to their initial state.

        Called at the beginning of each epoch to reset accumulated metrics.
        """
        self.loss_tracker.reset_state()
        if self.use_mut:
            self.mutation_loss_tracker.reset_state()
            self.accuracy_tracker.reset_state()
            self.sample_avg_accuracy_tracker.reset_state()
        if self.use_cna:
            self.cna_loss_tracker.reset_state()
            self.cna_segment_mean_loss_tracker.reset_state()
            self.cna_pearson_tracker.reset_state()
            self.val_cna_pearson_tracker.reset_state()

            if self.predict_cna_loh:
                self.cna_loh_loss_tracker.reset_state()
                # self.cna_loh_accuracy_tracker.reset_state()
                self.cna_loh_auc_tracker.reset_state()

        # Reset InfoNCE loss and margin trackers if they exist
        if self.use_mut and self.use_cna and self.use_infonce_loss:
            self.infonce_loss_tracker.reset_state()
            self.infonce_margin_tracker.reset_state()

        # Reset token-bag InfoNCE trackers if they exist (independent per modality)
        if hasattr(self, 'token_bag_mut_loss_tracker'):
            self.token_bag_mut_loss_tracker.reset_state()
            self.token_bag_mut_margin_tracker.reset_state()
        if hasattr(self, 'token_bag_cna_loss_tracker'):
            self.token_bag_cna_loss_tracker.reset_state()
            self.token_bag_cna_margin_tracker.reset_state()

        super().reset_metrics()

    @property
    def metrics(self):
        """
        Return list of all metrics tracked by this model.

        Returns:
            list: List of metric objects for Keras to track
        """
        metrics_list = [self.loss_tracker]

        # Only include mutation metrics if training on them
        if self.use_mut and self.train_on_mutation_loss:
            metrics_list.extend([self.mutation_loss_tracker, self.accuracy_tracker, self.sample_avg_accuracy_tracker])

        # Only include CNA metrics if training on them
        if self.use_cna and self.train_on_cna_loss:
            metrics_list.extend([
                self.cna_loss_tracker,
                self.cna_segment_mean_loss_tracker
                # NOTE: Pearson trackers are NOT included - they're managed manually
                # to avoid Keras lifecycle interference with custom metrics
            ])

            if self.predict_cna_loh:
                metrics_list.extend([
                    self.cna_loh_loss_tracker,
                    # self.cna_loh_accuracy_tracker,
                    self.cna_loh_auc_tracker
                ])

        # Include InfoNCE loss and margin trackers if using both modalities and InfoNCE is enabled
        if self.use_mut and self.use_cna and self.use_infonce_loss:
            metrics_list.append(self.infonce_loss_tracker)
            metrics_list.append(self.infonce_margin_tracker)

        # Include token-bag InfoNCE trackers if they exist (independent per modality)
        if hasattr(self, 'token_bag_mut_loss_tracker'):
            metrics_list.extend([
                self.token_bag_mut_loss_tracker,
                self.token_bag_mut_margin_tracker
            ])
        if hasattr(self, 'token_bag_cna_loss_tracker'):
            metrics_list.extend([
                self.token_bag_cna_loss_tracker,
                self.token_bag_cna_margin_tracker
            ])

        return metrics_list
    
    def reset_gradient_logging(self):
        """
        Reset gradient logging data for a fresh training session.
        Call this before starting a new training run.
        """
        if self.log_gradients:
            self.step_count.assign(0)
            self.gradient_df = None
            self.layer_names = None
    
    def _log_gradient_norms(self, gradients):
        """
        Save gradient norms to CSV file for debugging training stability.

        Args:
            gradients: List of gradient tensors from tape.gradient()

        Note:
            Uses tf.py_function to force eager execution for gradient logging,
            which allows access to .numpy() even when train_step runs in graph mode.
        """
        self.step_count.assign_add(1)

        # Compute gradient norms in graph mode
        grad_norms_list = []
        for grad in gradients:
            if grad is not None:
                grad_norms_list.append(tf.norm(grad))
            else:
                grad_norms_list.append(tf.constant(0.0))

        # Stack into a single tensor
        grad_norms_tensor = tf.stack(grad_norms_list)

        # Use tf.py_function to run logging in eager mode
        # This forces eager execution where .numpy() is available
        tf.py_function(
            func=self._log_gradients_eager,
            inp=[grad_norms_tensor, self.step_count],
            Tout=[]
        )
    
    def _log_gradients_eager(self, grad_norms_tensor, step_tensor):
        """
        Eager execution wrapper for gradient logging.
        Called via tf.py_function to enable .numpy() access.

        Args:
            grad_norms_tensor: Tensor of gradient norms
            step_tensor: Tensor with step count
        """
        # Convert tensors to Python values (now safe in eager mode)
        step_val = int(step_tensor.numpy())
        grad_norms = grad_norms_tensor.numpy().tolist()

        # Check if we should log (first 3 steps, then every N steps)
        should_log = (step_val <= 3) or (step_val % self.log_every_n_steps == 0)

        if should_log:
            self._save_gradient_norms_to_csv_eager(grad_norms, step_val)

    def _save_gradient_norms_to_csv_eager(self, grad_norms, step):
        """
        Helper function to save gradient norms per layer over time using pandas.
        Format: rows = layers, columns = time steps

        Args:
            grad_norms: Python list of gradient norm values
            step: Python int of current step number
        """
        import pandas as pd
        import numpy as np
        import os

        try:
            # Values are already Python types
            current_grad_norms = grad_norms
            step_int = step
            
            # Get layer names from trainable variables
            current_layer_names = []
            for var in self.trainable_variables:
                layer_name = var.path.replace('/', '_').replace(':', '_')  # Make filename-safe
                current_layer_names.append(layer_name)
            
            # Initialize gradient data storage on first call
            if not hasattr(self, 'gradient_data') or self.gradient_data is None:
                self.layer_names = current_layer_names
                self.gradient_data = {}  # Dictionary to store gradient data
            
            # Add new column for this step - handle mismatched lengths
            column_name = f'step_{step_int}'
            
            # Verify gradient/layer count consistency
            if len(current_grad_norms) != len(self.layer_names):
                print(f"ERROR: Step {step_int} - Gradient count mismatch!")
                print(f"  Expected {len(self.layer_names)} layers, got {len(current_grad_norms)} gradients")
                print(f"  First time layer count: {len(self.layer_names)}")
                print(f"  Current trainable vars: {len(current_layer_names)}")
                print(f"  This indicates a serious issue - layers are disappearing/appearing during training")
                raise ValueError(f"Gradient count mismatch at step {step_int}: expected {len(self.layer_names)}, got {len(current_grad_norms)}")
                
            # Store gradient data efficiently (no DataFrame operations yet)
            self.gradient_data[column_name] = current_grad_norms
            
            # Create DataFrame efficiently from all collected data
            self.gradient_df = pd.DataFrame(self.gradient_data, index=self.layer_names)
            self.gradient_df.index.name = 'layer_name'
            
            # Get model directory and construct CSV path
            if self.model_dir is not None:
                gradient_log_file = os.path.join(self.model_dir, 'gradient_norms.csv')
            else:
                gradient_log_file = 'gradient_norms.csv'  # Fallback
            
            # Write DataFrame to CSV
            self.gradient_df.to_csv(gradient_log_file)
            
            # Print summary for immediate feedback
            max_grad_norm = max(current_grad_norms)
            avg_grad_norm = sum(current_grad_norms) / len(current_grad_norms)
            min_grad_norm = min(current_grad_norms)
            if self.print_to_terminal:
                # Get current learning rate from optimizer
                current_lr = float(self.optimizer.learning_rate.numpy())
                print(f"Step {step_int}: Max grad: {max_grad_norm:.6f}, Avg grad: {avg_grad_norm:.6f}, Min grad: {min_grad_norm:.6f}, LR: {current_lr:.6f}")

            # Warn about problematic gradients
            if max_grad_norm > 10.0:
                print(f"  WARNING: Large gradients detected!")
            nan_count = sum(1 for x in current_grad_norms if not np.isfinite(x))
            if nan_count > 0:
                print(f"  WARNING: {nan_count} layers have NaN/Inf gradients!")

        except Exception as e:
            print(f"Error logging gradients: {e}")
            import traceback
            traceback.print_exc()

    def get_config(self):
        """
        Get the configuration dictionary for model serialization.

        Returns:
            dict: Configuration dictionary containing all init parameters needed to reconstruct the model
        """
        config = super().get_config()

        # Convert model_dir to string if it's a Path object
        model_dir = self.model_dir
        if hasattr(model_dir, '__fspath__'):  # Check if it's a Path-like object
            model_dir = str(model_dir)

        config.update({
            "base_model": tf.keras.utils.serialize_keras_object(self.base_model),
            "use_mut": self.use_mut,
            "use_cna": self.use_cna,
            "train_on_mutation_loss": self.train_on_mutation_loss,
            "train_on_cna_loss": self.train_on_cna_loss,
            "loss_non_zero_only": self.loss_non_zero_only,
            "hinge_loss_t": self.hinge_loss_t,
            "per_sample_loss": self.per_sample_loss,
            "accuracy_non_zero_only": self.accuracy_non_zero_only,
            "model_dir": model_dir,
            "log_gradients": self.log_gradients,
            # CNA dual-task parameters
            "predict_cna_loh": self.predict_cna_loh,
            # InfoNCE loss parameters
            "use_infonce_loss": self.use_infonce_loss,
            "infonce_temperature": self.infonce_temperature,
            "infonce_loss_weight": self.infonce_loss_weight,
            # Token-bag InfoNCE parameters
            "use_mut_token_bag_infonce": self.use_mut_token_bag_infonce,
            "use_cna_token_bag_infonce": self.use_cna_token_bag_infonce,
            "token_bag_temperature": self.token_bag_temperature,
            "token_bag_loss_weight": self.token_bag_loss_weight,
        })
        return config

    @classmethod
    def from_config(cls, config):
        """
        Create a model instance from a configuration dictionary.

        Args:
            config (dict): Configuration dictionary from get_config()

        Returns:
            CustomTrainingModel: New model instance with the specified configuration
        """
        # Deserialize base_model
        base_model = tf.keras.utils.deserialize_keras_object(config.pop("base_model"))

        # Create and return the model with all config parameters
        return cls(base_model=base_model, **config)

class CustomTrainingModelMIL(tf.keras.Model):
    """
    Custom Multiple Instance Learning (MIL) training model with advanced loss computation.
    
    This model implements MIL-specific training with optional hinge loss thresholding,
    class weighting, and separate metric tracking for training and validation phases.
    Handles sparse categorical cross-entropy loss with custom aggregation methods.
    
    Args:
        base_model (tf.keras.Model): The underlying MIL model to train
        hinge_loss_t (float, optional): Hinge loss threshold. Only samples with loss > threshold
                                       contribute to the final loss. If None, all samples contribute.
        class_weights (array-like, optional): Per-class weights for handling class imbalance.
                                            Shape should be [num_classes]. If None, all classes have equal weight.
        include_all_in_loss (bool): Whether to include samples with zero loss in the mean calculation.
                                  If False, only non-zero loss samples contribute to the mean. Default: True
        **kwargs: Additional arguments passed to the parent tf.keras.Model
    
    Attributes:
        num_classes (int): Number of output classes
        sparse_categorical_loss: SparseCategoricalCrossentropy loss function with no reduction
        loss_tracker, accuracy_tracker, auc_macro_tracker, auc_micro_tracker: Training metrics
        val_loss_tracker, val_accuracy_tracker, val_auc_macro_tracker, val_auc_micro_tracker: Validation metrics (lazy)
    """
    def __init__(self, base_model,hinge_loss_t=None,class_weights=None, include_all_in_loss=True, **kwargs):
        super().__init__(**kwargs)
        self.base_model = base_model
        num_classes = self.base_model.outputs[0].shape[1]
        # Training metrics
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.accuracy_tracker = tf.keras.metrics.Mean(name="accuracy")
        # For proper multi-class AUC computation
        if num_classes == 2:
            # Binary classification - use standard AUC
            self.auc_macro_tracker = tf.keras.metrics.AUC(name="auc_macro")
            self.auc_micro_tracker = tf.keras.metrics.AUC(name="auc_micro")
        else:
            # Multi-class - use one-vs-rest approach
            self.auc_macro_tracker = tf.keras.metrics.AUC(
                name="auc_macro",
                multi_label=True,
                num_labels=num_classes
            )
            self.auc_micro_tracker = tf.keras.metrics.AUC(
                name="auc_micro",
                multi_label=False
            )
        
        # Separate validation metrics (always created but only included in metrics list when used)
        # Note: Keras automatically adds "val_" prefix to test_step metrics, so we use base names
        self.val_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.val_accuracy_tracker = tf.keras.metrics.Mean(name="accuracy")
        # Validation AUC metrics (matching training configuration)
        if num_classes == 2:
            # Binary classification - use standard AUC
            self.val_auc_macro_tracker = tf.keras.metrics.AUC(name="auc_macro")
            self.val_auc_micro_tracker = tf.keras.metrics.AUC(name="auc_micro")
        else:
            # Multi-class - use one-vs-rest approach
            self.val_auc_macro_tracker = tf.keras.metrics.AUC(
                name="auc_macro",
                multi_label=True,
                num_labels=num_classes
            )
            self.val_auc_micro_tracker = tf.keras.metrics.AUC(
                name="auc_micro",
                multi_label=False
            )
        self._validation_used = False  # Flag to track if validation has been used

        # self.sparse_categorical_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.sparse_categorical_loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
        self.hinge_loss_t = hinge_loss_t
        self.class_weights = class_weights
        self.num_classes = num_classes
        self.include_all_in_loss = include_all_in_loss

    def call(self, inputs, training=False):
        """
        Forward pass through the base model.
        
        Args:
            inputs: Input data to the model
            training (bool): Whether the model is in training mode. Default: False
        
        Returns:
            tuple: (logits, labels) from the base model
        """
        # Forward pass through the base model
        logits,y = self.base_model(inputs, training=training)
        return logits,y

    def compute_loss(self, y_true, logits):
        """
        Compute the MIL loss with optional hinge thresholding and class weighting.
        
        This method implements sophisticated loss computation including:
        - Sparse categorical cross-entropy with no reduction
        - Optional hinge loss thresholding to focus on difficult samples
        - Per-class weighting for handling class imbalance
        - Configurable inclusion/exclusion of zero-loss samples in mean computation
        
        Args:
            y_true (tf.Tensor): True labels of shape [batch_size, 1] or [batch_size]
            logits (tf.Tensor): Raw model predictions of shape [batch_size, num_classes]
        
        Returns:
            tf.Tensor: Computed scalar loss value
        
        Note:
            - If hinge_loss_t is provided, only samples with loss > threshold contribute
            - If class_weights is provided, losses are weighted by class
            - If include_all_in_loss is False, zero losses are excluded from mean
        """
        # Compute the sparse categorical cross entropy loss
        sample_losses = self.sparse_categorical_loss(y_true, logits)

        # Apply threshold if specified - only samples with loss above threshold contribute
        if self.hinge_loss_t is not None:
            # Create a mask for samples with loss above threshold
            hinge_mask = tf.cast(sample_losses > self.hinge_loss_t, tf.float32)

            # Apply mask to sample losses (losses below threshold become zero)
            masked_sample_losses = sample_losses * hinge_mask

            # Create a binary mask to track which samples are still contributing
            active_samples = hinge_mask
        else:
            # If no threshold, all samples are active
            masked_sample_losses = sample_losses
            active_samples = tf.ones_like(sample_losses)

        # Apply class weights if specified
        if self.class_weights is not None:
            # Convert class weights to a tensor if it's not already
            if not isinstance(self.class_weights, tf.Tensor):
                class_weights_tensor = tf.constant(self.class_weights, dtype=tf.float32)
            else:
                class_weights_tensor = self.class_weights

            # Get the weights corresponding to the true classes
            weights = tf.gather(class_weights_tensor, tf.cast(y_true, tf.int32))

            # Apply weights to individual sample losses
            masked_sample_losses = masked_sample_losses * weights
            
            # Also apply weights to original sample losses if we're including all
            if self.hinge_loss_t is not None and self.include_all_in_loss:
                sample_losses = sample_losses * weights

        # Calculate final loss
        if self.hinge_loss_t is not None and self.include_all_in_loss:
            # Include all samples in loss computation (even zeroed ones)
            # This gives a loss that decreases as training progresses
            loss = tf.reduce_mean(sample_losses)
        else:
            # Original behavior: only active samples contribute
            # Sum of weighted losses divided by sum of active samples (with epsilon to avoid div by zero)
            denominator = tf.reduce_sum(active_samples) + tf.keras.backend.epsilon()
            loss = tf.reduce_sum(masked_sample_losses) / denominator

        return loss

    def train_step(self, data):
        """
        Performs one training step for the MIL model.
        
        Executes forward pass, computes loss with custom MIL logic, performs
        gradient descent, and updates all training metrics including accuracy and AUC.
        
        Args:
            data (dict): Training batch containing:
                - Input features (ref, alt, context_5p, context_3p, vaf, etc.)
                - 'label': True class labels of shape [batch_size, 1]
        
        Returns:
            dict: Training metrics with keys:
                - 'loss': Training loss value
                - 'accuracy': Training accuracy
                - 'auc_macro': Macro-averaged AUC across classes
                - 'auc_micro': Micro-averaged AUC
        """
        x = data
        y_true = x['label']  # Labels from 'data'

        with tf.GradientTape() as tape:
            # Forward pass through the base model
            logits,_ = self(x, training=True)

            # Compute the sparse categorical cross entropy loss
            loss = self.compute_loss(y_true, logits)

        # Update the custom loss metric
        self.loss_tracker.update_state(loss)

        # Compute gradients and apply them
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Compute the accuracy
        accuracy = tf.keras.metrics.sparse_categorical_accuracy(y_true, logits)
        self.accuracy_tracker.update_state(accuracy)

        # Convert logits -> probabilities with softmax
        probs = tf.nn.softmax(logits, axis=-1)  # shape: [batch_size, num_classes]

        # Update AUC metrics based on number of classes
        if self.num_classes == 2:
            # Binary classification: use positive class probability and true labels
            # Ensure proper shape handling for tensors
            y_true_flat = tf.reshape(y_true, [-1])  # Flatten to 1D
            y_true_binary = tf.cast(y_true_flat, tf.float32)  # Convert to float for AUC
            probs_positive = tf.reshape(probs[:, 1], [-1])  # Flatten probability to 1D
            
            self.auc_macro_tracker.update_state(y_true_binary, probs_positive)
            self.auc_micro_tracker.update_state(y_true_binary, probs_positive)
        else:
            # Multi-class: use one-hot encoding
            y_true_1hot = tf.one_hot(
                tf.cast(y_true, tf.int32),
                depth=self.num_classes
            )
            
            self.auc_macro_tracker.update_state(y_true_1hot, probs)
            self.auc_micro_tracker.update_state(y_true_1hot, probs)

        # Return the total loss and accuracy for logging
        return {"loss": self.loss_tracker.result(),
                "accuracy": self.accuracy_tracker.result(),
                "auc_macro": self.auc_macro_tracker.result(),
                "auc_micro": self.auc_micro_tracker.result()}

    
    def test_step(self, data):
        """
        Performs one validation/test step for the MIL model.
        
        Executes forward pass in inference mode, computes validation loss,
        and updates validation-specific metrics. Automatically creates validation
        metrics on first call if they don't exist.
        
        Args:
            data (dict): Validation batch containing:
                - Input features (ref, alt, context_5p, context_3p, vaf, etc.)
                - 'label': True class labels of shape [batch_size, 1]
        
        Returns:
            dict: Validation metrics with keys:
                - 'loss': Validation loss value (mapped from val_loss_tracker)
                - 'accuracy': Validation accuracy (mapped from val_accuracy_tracker)
                - 'auc_macro': Validation macro-averaged AUC
                - 'auc_micro': Validation micro-averaged AUC
        
        Note:
            - Validation metrics are completely separate from training metrics
            - Returns metrics with standard Keras naming for callback compatibility
        """
        # Mark that validation is being used
        self._validation_used = True
        
        x = data
        y_true = x['label']  # Labels from 'data'

        logits,_ = self(x, training=False)  # Forward pass during validation
        loss = self.compute_loss(y_true, logits)

        # Track validation metrics separately
        self.val_loss_tracker.update_state(loss)

        accuracy = tf.keras.metrics.sparse_categorical_accuracy(y_true, logits)
        self.val_accuracy_tracker.update_state(accuracy)

        # Convert logits -> probabilities with softmax
        probs = tf.nn.softmax(logits, axis=-1)
        
        # Update validation AUC metrics based on number of classes
        if self.num_classes == 2:
            # Binary classification: use positive class probability and true labels
            # Ensure proper shape handling for tensors
            y_true_flat = tf.reshape(y_true, [-1])  # Flatten to 1D
            y_true_binary = tf.cast(y_true_flat, tf.float32)  # Convert to float for AUC
            probs_positive = tf.reshape(probs[:, 1], [-1])  # Flatten probability to 1D
            
            self.val_auc_macro_tracker.update_state(y_true_binary, probs_positive)
            self.val_auc_micro_tracker.update_state(y_true_binary, probs_positive)
        else:
            # Multi-class: use one-hot encoding
            y_true_1hot = tf.one_hot(
                tf.cast(y_true, tf.int32),
                depth=self.num_classes
            )
            
            self.val_auc_macro_tracker.update_state(y_true_1hot, probs)
            self.val_auc_micro_tracker.update_state(y_true_1hot, probs)

        # Return validation metrics (Keras will automatically add "val_" prefix)
        return {"loss": self.val_loss_tracker.result(),
                "accuracy": self.val_accuracy_tracker.result(),
                "auc_macro": self.val_auc_macro_tracker.result(),
                "auc_micro": self.val_auc_micro_tracker.result()}

    def reset_metrics(self):
        """
        Reset all metric trackers to their initial state.
        
        Resets both training and validation metrics (if validation metrics exist).
        Called automatically by Keras at the start of each epoch.
        
        Note:
            - Training metrics are always reset
            - Validation metrics are only reset if they have been created
            - Calls parent class reset_metrics for any additional metrics
        """
        # Reset training metrics
        self.loss_tracker.reset_state()
        self.accuracy_tracker.reset_state()
        self.auc_macro_tracker.reset_state()
        self.auc_micro_tracker.reset_state()
        
        # Always reset validation metrics (they're always created now)
        self.val_loss_tracker.reset_state()
        self.val_accuracy_tracker.reset_state()
        self.val_auc_macro_tracker.reset_state()
        self.val_auc_micro_tracker.reset_state()
        
        super().reset_metrics()

    @property
    def metrics(self):
        """
        Return list of all metrics tracked by this model.
        
        Keras uses this property to automatically reset metrics at epoch boundaries
        and to track metric states. Metrics are displayed in the order they appear here.
        
        Returns:
            list: List of metric objects in display order:
                - Training: [loss, accuracy, auc_macro, auc_micro]
                - Validation (if used): [val_loss, val_accuracy, val_auc_macro, val_auc_micro]
        """
        # Group training metrics first, then validation metrics for cleaner display
        metrics = [
            self.loss_tracker, 
            self.accuracy_tracker,
            self.auc_macro_tracker, 
            self.auc_micro_tracker
        ]
        
        # Only include validation metrics if validation has been used
        if self._validation_used:
            metrics.extend([
                self.val_loss_tracker,
                self.val_accuracy_tracker, 
                self.val_auc_macro_tracker,
                self.val_auc_micro_tracker
            ])
        
        return metrics

    @property
    def inputs(self):
        """
        Return the input specification of the base model.
        
        Returns:
            Input specification from the wrapped base model
        """
        return self.base_model.inputs

    def get_config(self):
        """
        Get the configuration dictionary for model serialization.
        
        Returns:
            dict: Configuration dictionary containing:
                - base_model: Serialized base model configuration
                - hinge_loss_t: Hinge loss threshold value
                - class_weights: Class weight array (converted to list)
                - include_all_in_loss: Boolean flag for loss inclusion
                - num_classes: Number of output classes
        
        Note:
            - Used by Keras for model saving/loading
            - Handles conversion of numpy arrays to lists for JSON serialization
        """
        # First get the config from the parent class
        config = super().get_config()

        # Convert class_weights to list if it's a tensor
        class_weights = self.class_weights
        if isinstance(class_weights, tf.Tensor):
            class_weights = class_weights.numpy().tolist()

        config.update({
            "base_model": tf.keras.utils.serialize_keras_object(self.base_model),
            "hinge_loss_t": self.hinge_loss_t,
            "class_weights": class_weights,
            "include_all_in_loss": self.include_all_in_loss
        })
        return config

    @classmethod
    def from_config(cls, config):
        """
        Create a model instance from a configuration dictionary.
        
        Args:
            config (dict): Configuration dictionary from get_config()
        
        Returns:
            CustomTrainingModelMIL: New model instance with the specified configuration
        
        Note:
            - Used by Keras for model loading
            - Reconstructs the base model from its configuration
            - Converts class_weights back from list to numpy array if present
        """
        # Deserialize `base_model` and pass it as an argument
        base_model = tf.keras.utils.deserialize_keras_object(config.pop("base_model"))
        hinge_loss_t = config.pop("hinge_loss_t", None)
        class_weights = config.pop("class_weights", None)
        include_all_in_loss = config.pop("include_all_in_loss", False)
        return cls(base_model=base_model,
                   hinge_loss_t=hinge_loss_t,
                   class_weights=class_weights,
                   include_all_in_loss=include_all_in_loss, **config)