"""Multiple Instance Learning aggregation layers.

Sample-level pooling layers that aggregate per-token (per-variant or per-segment)
embeddings into bag-level representations: attention-weighted average pooling
(MILAttentionLayerAVG), max pooling (MILAttentionLayerMAX), explicit MIL bag
masking (MILMaskingLayer), and class-conditional attention pooling for
multi-class supervision (MILMultiClassAttentionLayer).
"""

import tensorflow as tf
import tessera.layers.act_functions

@tf.keras.utils.register_keras_serializable(
    package="tessera.layers.mil",
    name="MILAttentionLayerAVG")
class MILAttentionLayerAVG(tf.keras.layers.Layer):
    """
    Multiple Instance Learning attention layer that computes weighted average aggregation.
    
    This layer learns attention weights for each instance in a bag and computes a weighted
    average of the instances. Supports both single-class and multi-class attention mechanisms.
    
    Args:
        attention_dim (list): Dimensions for the attention MLP layers. Default: [128, 128]
        use_weights (bool): Whether to use learned attention weights. Default: True
        temperature (float): Temperature parameter for Gumbel-Softmax. Default: 0.5
        use_multiclass_attention (bool): Whether to use multi-class attention. Default: False
        num_classes (int): Number of classes for multi-class attention. Required if use_multiclass_attention=True
        activation_type (str): Type of activation function ['softplus', 'ASU', 'gumbel']. Default: 'ASU'
        **kwargs: Additional layer arguments
    
    Raises:
        ValueError: If activation_type is not one of the supported types
        ValueError: If use_multiclass_attention=True but num_classes is None
    """
    def __init__(self, attention_dim=[128,128], use_weights=True, temperature=0.5, 
                 use_multiclass_attention=False, num_classes=None, 
                 activation_type='ASU', **kwargs):
        super(MILAttentionLayerAVG, self).__init__(**kwargs)
        self.attention_dim = attention_dim
        self.use_weights = use_weights
        self.temperature = temperature
        self.use_multiclass_attention = use_multiclass_attention
        self.num_classes = num_classes
        self.activation_type = activation_type
        self.hard = False

        # Validate activation type
        if activation_type not in ['softplus', 'ASU', 'gumbel']:
            raise ValueError(f"activation_type must be one of ['softplus', 'ASU', 'gumbel'], got {activation_type}")

        if self.use_multiclass_attention:
            if self.num_classes is None:
                raise ValueError("num_classes must be provided when use_multiclass_attention=True")
            # Use the existing MILMultiClassAttentionLayer for multi-class attention
            self.multiclass_attention = MILMultiClassAttentionLayer(
                num_classes=self.num_classes,
                attention_dim=self.attention_dim,
                activation_type=activation_type,
                temperature=self.temperature
            )
        else:
            # Build the attention MLP as a Sequential block
            self.attention_mlp = tf.keras.Sequential(name="attention_mlp")
            for ii in self.attention_dim:
                self.attention_mlp.add(tf.keras.layers.Dense(ii, activation='relu'))

            # Gate dense layer (always needed)
            self.gate_dense = tf.keras.layers.Dense(1, activation=None)
            
            # Set up activation based on type
            if activation_type == 'ASU':
                self.gate_activation = tessera.layers.act_functions.Activations.ASU()
            elif activation_type == 'softplus':
                self.gate_activation = tf.keras.layers.Activation('softplus')
            elif activation_type == 'gumbel':
                # For gumbel, we'll use the gumbel_softmax method instead
                self.gate_activation = None
            
            self.batch_norm = tf.keras.layers.BatchNormalization()

        self.temperature_var = tf.Variable(temperature, trainable=True, dtype=tf.float32)

    def gumbel_softmax(self, logits, mask=None):
        """Apply Gumbel-Softmax to get differentiable discrete attention weights."""
        # Reshape logits for softmax along instance dimension
        batch_size = tf.shape(logits)[0]
        num_instances = tf.shape(logits)[1]

        # Apply mask if provided (set masked logits to large negative value)
        if mask is not None:
            mask_expanded = tf.expand_dims(mask, axis=-1)
            logits = tf.where(mask_expanded, logits, -1e9 * tf.ones_like(logits))

        # Reshape to 2D for softmax
        logits_2d = tf.reshape(logits, [batch_size, num_instances])

        # Add Gumbel noise for sampling
        gumbel_noise = -tf.math.log(-tf.math.log(tf.random.uniform(tf.shape(logits_2d)) + 1e-10) + 1e-10)
        noisy_logits = (logits_2d + gumbel_noise) / self.temperature_var

        # Apply softmax to get probabilities
        weights = tf.nn.softmax(noisy_logits, axis=1)

        if self.hard:
            # Straight-through estimator for hard (one-hot) samples
            indices = tf.argmax(weights, axis=1)
            one_hot = tf.one_hot(indices, num_instances)
            # Use straight-through estimator for gradient
            hard_weights = tf.stop_gradient(one_hot - weights) + weights
            weights = hard_weights

        # Reshape back to original shape
        weights = tf.reshape(weights, [batch_size, num_instances, 1])

        return weights

    def call(self, inputs, vaf, mask=None, training=None):
        """
        Forward pass through the attention layer.
        
        Computes attention weights for each instance and returns a weighted average aggregation.
        The behavior depends on whether single-class or multi-class attention is used.
        
        Args:
            inputs (tf.Tensor): Instance features of shape [batch_size, num_instances, feature_dim]
            vaf (tf.Tensor): Variant allele frequencies of shape [batch_size, num_instances, 1]
            mask (tf.Tensor, optional): Boolean mask of shape [batch_size, num_instances]
                                       to ignore padded instances
            training (bool, optional): Whether in training mode (affects dropout, batch norm, etc.)

        Returns:
            tuple: A tuple containing:
                - output (tf.Tensor): Aggregated features of shape [batch_size, feature_dim]
                                     For multi-class: concatenated class-specific features
                - attention_weights (tf.Tensor): Attention weights of shape:
                                                 Single-class: [batch_size, num_instances, 1]
                                                 Multi-class: [batch_size, num_instances, num_classes]
        """

        if self.use_multiclass_attention:
            # Use multi-class attention layer
            class_outputs, attention_weights = self.multiclass_attention(inputs, vaf, mask=mask, training=training)
            # For compatibility, return a concatenated output and all attention weights
            # Concatenate all class outputs to form a single output vector
            output = tf.concat(class_outputs, axis=-1)
            return output, attention_weights
        else:
            # Use original single-class attention
            eps = tf.keras.backend.epsilon()

            # 1) Compute attention features
            self.inputs = inputs
            # attention_features = self.attention_mlp(inputs)
            attention_features = self.attention_mlp(tf.keras.layers.Concatenate()([inputs,vaf]))

            # attention_features = self.attention_dense(inputs)
            # attention_features = self.batch_norm(attention_features)

            # 2) Compute gates (attention scores) and apply activation
            gates = self.gate_dense(attention_features)  # shape: [B, N, 1]
            
            if self.activation_type == 'gumbel':
                gates = self.gumbel_softmax(gates, mask)  # Apply Gumbel-Softmax
            else:
                gates = self.gate_activation(gates)  # shape: [B, N, 1]

            # 3) If a mask is provided, expand mask to [B, N, 1] and apply it
            if mask is not None and self.activation_type != 'gumbel':  # gumbel handles mask internally
                mask_expanded = tf.expand_dims(mask, axis=-1)  # [B, N, 1]
                # Zero out gates for masked instances
                gates = tf.where(mask_expanded, gates, tf.zeros_like(gates))
                # Zero out vaf for masked instances
                vaf = tf.where(mask_expanded, vaf, tf.zeros_like(vaf))

            # ---------------------------------------------------------------------
            # Option A: use_weights = True  => Weighted by (gates * vaf)
            # Option B: use_weights = False => Weighted only by vaf (ignore gates)
            # ---------------------------------------------------------------------

            if self.use_weights:
                # Combine gates and vaf => final weight = gates * vaf
                # combined_weight = gates * vaf  # [B, N, 1]
                combined_weight = gates

                # Weighted inputs
                weighted_inputs = inputs * combined_weight

                # Sum across instances
                output = tf.reduce_sum(weighted_inputs, axis=1)

                # Normalize by total combined weight
                sum_weights = tf.reduce_sum(combined_weight, axis=1) + eps
                output = output / sum_weights

            else:
                # Weighted only by vaf (ignore gates)
                weighted_inputs = inputs * vaf

                # Sum across instances
                output = tf.reduce_sum(weighted_inputs, axis=1)

                # Normalize by total vaf
                sum_vaf = tf.reduce_sum(vaf, axis=1) + eps
                output = output / sum_vaf

                output = tf.reduce_mean(inputs, axis=1)

            return output, gates

    def set_temperature(self, value):
        """
        Update the temperature parameter during training.
        
        Temperature controls the sharpness of the Gumbel-Softmax distribution.
        Lower temperatures result in more discrete (one-hot-like) attention weights,
        while higher temperatures result in more uniform distributions.
        
        Args:
            value (float): New temperature value. Should be positive.
        """
        self.temperature_var.assign(value)
        # Also update the multiclass attention layer if using it
        if self.use_multiclass_attention:
            self.multiclass_attention.set_temperature(value)

@tf.keras.utils.register_keras_serializable(
    package="tessera.layers.mil",
    name="MILAttentionLayerMAX")
class MILAttentionLayerMAX(tf.keras.layers.Layer):
    """
    Multiple Instance Learning layer that computes maximum values across instances.
    
    This layer finds the maximum value for each feature across all instances in a bag,
    along with the indices where these maxima occur. This provides a complementary
    aggregation method to attention-based averaging.
    
    Args:
        use_weights (bool): Kept for API compatibility, but not used in computation. Default: True
        **kwargs: Additional layer arguments
    """
    def __init__(self, use_weights=True, **kwargs):
        super(MILAttentionLayerMAX, self).__init__(**kwargs)
        self.use_weights = use_weights

    def call(self, inputs, vaf, mask=None):
        """
        Forward pass to compute maximum values across instances.
        
        For each feature dimension, finds the maximum value across all instances in each bag.
        Properly handles masking to ignore padded instances.
        
        Args:
            inputs (tf.Tensor): Instance features of shape [batch_size, num_instances, feature_dim]
            vaf (tf.Tensor): Variant allele frequencies of shape [batch_size, num_instances, 1]
                           (unused, kept for API compatibility with attention layers)
            mask (tf.Tensor, optional): Boolean mask of shape [batch_size, num_instances]
                                       to ignore padded instances

        Returns:
            tuple: A tuple containing:
                - output (tf.Tensor): Maximum values of shape [batch_size, feature_dim]
                - argmax_indices (tf.Tensor): Indices of maxima of shape [batch_size, feature_dim]
                                              Values are -1 for completely masked samples
        """
        if mask is not None:
            # Create mask for inputs [B, N, D]
            mask_expanded = tf.expand_dims(mask, axis=-1)  # [B, N, 1]
            mask_expanded = tf.broadcast_to(mask_expanded, tf.shape(inputs))  # [B, N, D]

            # Mask out invalid positions with very low values
            masked_inputs = tf.where(
                mask_expanded,
                inputs,
                tf.ones_like(inputs) * -1e9
            )

            # Get max values and indices
            argmax_indices = tf.argmax(masked_inputs, axis=1)  # [B, D]
            output = tf.reduce_max(masked_inputs, axis=1)  # [B, D]

            # Handle completely masked samples
            any_unmasked = tf.reduce_any(mask, axis=1, keepdims=True)  # [B, 1]
            any_unmasked = tf.broadcast_to(any_unmasked, tf.shape(output))  # [B, D]
            
            output = tf.where(any_unmasked, output, tf.zeros_like(output))
            argmax_indices = tf.where(any_unmasked, argmax_indices, tf.ones_like(argmax_indices) * -1)
        else:
            # No mask - simple max operation
            argmax_indices = tf.argmax(inputs, axis=1)  # [B, D]
            output = tf.reduce_max(inputs, axis=1)  # [B, D]

        return output, argmax_indices

@tf.keras.utils.register_keras_serializable(
    package="tessera.layers.mil",
    name="MILMaskingLayer")
class MILMaskingLayer(tf.keras.layers.Layer):
    """
    Creates a boolean mask for Multiple Instance Learning bags.
    
    This layer generates a mask that identifies valid (non-zero) instances in a bag,
    which is used to ignore padded instances during aggregation operations.
    """
    def call(self, inputs):
        """
        Generate boolean mask for valid instances.
        
        Args:
            inputs (tf.Tensor): Input tensor of any shape, typically sequence inputs
        
        Returns:
            tf.Tensor: Boolean mask indicating non-zero instances
        """
        mask = tf.not_equal(inputs, 0)  # Create mask where inputs are non-zero
        return tf.reduce_any(mask, axis=-1)

@tf.keras.utils.register_keras_serializable(
    package="tessera.layers.mil",
    name="MILMultiClassAttentionLayer")
class MILMultiClassAttentionLayer(tf.keras.layers.Layer):
    """
    Multiple Instance Learning layer with class-specific attention weights.
    
    This layer learns separate attention mechanisms for each class, allowing the model
    to focus on different instances for different classes. Each class has its own
    attention MLP and gating mechanism.
    
    Args:
        num_classes (int): Number of classes for which to learn separate attention weights
        attention_dim (list): Dimensions for the attention MLP layers. Default: [128, 128]
        activation_type (str): Type of activation function ['softplus', 'ASU', 'gumbel']. Default: 'gumbel'
        temperature (float): Temperature parameter for Gumbel-Softmax. Default: 0.5
        **kwargs: Additional layer arguments
    
    Raises:
        ValueError: If activation_type is not one of the supported types
    """
    def __init__(self, num_classes, attention_dim=[128, 128], activation_type='gumbel', temperature=0.5, **kwargs):
        super(MILMultiClassAttentionLayer, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.attention_dim = attention_dim
        self.activation_type = activation_type
        self.temperature = temperature
        self.hard = False  # Whether to use hard (one-hot) samples in Gumbel-Softmax
        
        # Validate activation type
        if activation_type not in ['softplus', 'ASU', 'gumbel']:
            raise ValueError(f"activation_type must be one of ['softplus', 'ASU', 'gumbel'], got {activation_type}")
        
        # Build separate attention MLPs for each class (instead of shared)
        self.attention_mlps = []
        for class_idx in range(num_classes):
            # Create a distinct MLP for each class with different initializer seeds
            mlp = tf.keras.Sequential(name=f"attention_mlp_class_{class_idx}")
            for dim in self.attention_dim:
                initializer = tf.keras.initializers.GlorotUniform(seed=class_idx*100 + dim)
                mlp.add(tf.keras.layers.Dense(dim, activation='relu', 
                                            kernel_initializer=initializer))
            self.attention_mlps.append(mlp)
        
        # Class-specific attention heads (one per class)
        self.class_gates = []
        self.class_activations = []
        for class_idx in range(num_classes):
            # Use different initializers for each class gate to further encourage diversity
            initializer = tf.keras.initializers.GlorotUniform(seed=class_idx*1000)
            self.class_gates.append(tf.keras.layers.Dense(1, activation=None, 
                                                        kernel_initializer=initializer))
            
            # Set up activation for each class
            if activation_type == 'ASU':
                self.class_activations.append(tessera.layers.act_functions.Activations.ASU())
            elif activation_type == 'softplus':
                self.class_activations.append(tf.keras.layers.Activation('softplus'))
            elif activation_type == 'gumbel':
                # For gumbel, we'll use the gumbel_softmax method instead
                self.class_activations.append(None)
        
        # Temperature parameter for Gumbel-Softmax
        self.temperature_var = tf.Variable(temperature, trainable=True, dtype=tf.float32)
    
    def gumbel_softmax(self, logits, mask=None):
        """Apply Gumbel-Softmax to get differentiable discrete attention weights."""
        # Reshape logits for softmax along instance dimension
        batch_size = tf.shape(logits)[0]
        num_instances = tf.shape(logits)[1]

        # Apply mask if provided (set masked logits to large negative value)
        if mask is not None:
            mask_expanded = tf.expand_dims(mask, axis=-1)
            logits = tf.where(mask_expanded, logits, -1e9 * tf.ones_like(logits))

        # Reshape to 2D for softmax
        logits_2d = tf.reshape(logits, [batch_size, num_instances])

        # Add Gumbel noise for sampling
        gumbel_noise = -tf.math.log(-tf.math.log(tf.random.uniform(tf.shape(logits_2d)) + 1e-10) + 1e-10)
        noisy_logits = (logits_2d + gumbel_noise) / self.temperature_var

        # Apply softmax to get probabilities
        weights = tf.nn.softmax(noisy_logits, axis=1)

        if self.hard:
            # Straight-through estimator for hard (one-hot) samples
            indices = tf.argmax(weights, axis=1)
            one_hot = tf.one_hot(indices, num_instances)
            # Use straight-through estimator for gradient
            hard_weights = tf.stop_gradient(one_hot - weights) + weights
            weights = hard_weights

        # Reshape back to original shape
        weights = tf.reshape(weights, [batch_size, num_instances, 1])
        return weights
    
    def call(self, inputs, vaf, mask=None, training=None):
        """
        Args:
            inputs: Tensor of shape [batch_size, num_instances, feature_dim]
            vaf: Tensor of shape [batch_size, num_instances, 1]; variant allele frequencies
            mask: (Optional) Boolean mask of shape [batch_size, num_instances]
            training: Whether in training mode (controls dropout, etc.)

        Returns:
            outputs: List of class-specific aggregated MIL embeddings, each of shape [batch_size, feature_dim]
            attention_weights: Tensor of all class-specific attention weights [batch_size, num_instances, num_classes]
        """
        eps = tf.keras.backend.epsilon()
        
        # Prepare inputs with VAF
        combined_input = tf.keras.layers.Concatenate()([inputs, vaf])
        
        # Compute class-specific attention weights
        all_class_gates = []
        for class_idx in range(self.num_classes):
            # Process through class-specific MLP
            attention_features = self.attention_mlps[class_idx](
                combined_input, 
                training=training
            )
            
            # Get raw attention scores for this class
            class_gate = self.class_gates[class_idx](attention_features)  # [B, N, 1]
            
            # Apply activation based on type
            if self.activation_type == 'gumbel':
                class_gate = self.gumbel_softmax(class_gate, mask)
            elif self.activation_type in ['ASU', 'softplus']:
                # Apply activation function
                class_gate = self.class_activations[class_idx](class_gate)
                
                # Apply mask if provided
                if mask is not None:
                    mask_expanded = tf.expand_dims(mask, axis=-1)
                    class_gate = tf.where(mask_expanded, class_gate, tf.zeros_like(class_gate))
                
                # Apply softmax normalization over instances dimension
                batch_size = tf.shape(class_gate)[0]
                num_instances = tf.shape(class_gate)[1]
                gate_2d = tf.reshape(class_gate, [batch_size, num_instances])
                weights = tf.nn.softmax(gate_2d, axis=1)
            all_class_gates.append(class_gate)
        
        # Combine all class gates into a single tensor [B, N, C]
        attention_weights = tf.concat(all_class_gates, axis=-1)
        
        # Apply mask if provided
        if mask is not None:
            mask_expanded = tf.expand_dims(mask, axis=-1)  # [B, N, 1]
            # Zero out gates for masked instances (expand mask to match attention_weights)
            mask_for_gates = tf.broadcast_to(
                mask_expanded, 
                [tf.shape(attention_weights)[0], tf.shape(attention_weights)[1], self.num_classes]
            )
            attention_weights = tf.where(mask_for_gates, attention_weights, tf.zeros_like(attention_weights))
            
        # Apply attention for each class
        class_outputs = []
        for class_idx in range(self.num_classes):
            # Extract gates for this class [B, N, 1]
            class_gate = tf.expand_dims(attention_weights[:, :, class_idx], axis=-1)
            
            # Apply attention weights to inputs
            weighted_inputs = inputs * class_gate
            
            # Sum across instances
            output = tf.reduce_sum(weighted_inputs, axis=1)
            
            # Normalize by total weight for this class
            sum_weights = tf.reduce_sum(class_gate, axis=1) + eps
            normalized_output = output / sum_weights
            
            class_outputs.append(normalized_output)
        
        return class_outputs, attention_weights
    
    def set_temperature(self, value):
        """Update temperature parameter during training."""
        self.temperature_var.assign(value)