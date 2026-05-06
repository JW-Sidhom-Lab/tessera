"""
Utility layers for common tensor operations.

This module contains general-purpose layers for reshaping, transformations,
and other common operations used throughout the variant model architecture.
"""

import tensorflow as tf
from tensorflow.keras.layers import Lambda


class graph_object(object):
    """Legacy graph object for compatibility."""
    def __init__(self):
        self.init = 0


@tf.keras.utils.register_keras_serializable(
    package="tessera.layers.utils",
    name="UnflattenLastDim",
)
class UnflattenLastDim(tf.keras.layers.Layer):
    """
    Reshapes [B, N, F] -> [B, N, dim1, dim2], where F == dim1*dim2.

    XLA compatible: uses tf.shape + tf.reshape (no Python branching on tensors).
    Optional runtime check to assert F == dim1*dim2.

    Args:
        dim1: First dimension to unflatten to
        dim2: Second dimension to unflatten to
        validate: Whether to perform runtime validation (default: False)

    Input shape:
        (batch_size, seq_len, dim1 * dim2)

    Output shape:
        (batch_size, seq_len, dim1, dim2)

    Example:
        ```python
        unflatten = UnflattenLastDim(dim1=20, dim2=4)
        output = unflatten(inputs)  # [B, N, 80] -> [B, N, 20, 4]
        ```
    """

    def __init__(self, dim1: int, dim2: int, validate: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.dim1 = int(dim1)
        self.dim2 = int(dim2)
        self.validate = bool(validate)
        self.input_spec = tf.keras.layers.InputSpec(ndim=3)  # enforce rank 3

    def build(self, input_shape):
        # If static shape is known, verify at build-time.
        last = input_shape[-1]
        if last is not None:
            expected = self.dim1 * self.dim2
            if int(last) != expected:
                raise ValueError(
                    f"UnflattenLastDim: input last dim={int(last)} != dim1*dim2={expected}"
                )
        super().build(input_shape)

    def call(self, inputs):
        s = tf.shape(inputs)  # [B, N, F]
        b, n, f = s[0], s[1], s[2]
        expected = self.dim1 * self.dim2

        if self.validate:
            # Dynamic check (optional). Generally safe with XLA in tf.function.
            tf.debugging.assert_equal(
                f, expected, message="UnflattenLastDim: last dim != dim1*dim2"
            )

        out = tf.reshape(inputs, [b, n, self.dim1, self.dim2])
        # Annotate static shape to aid downstream shape inference.
        out = tf.ensure_shape(out, [None, None, self.dim1, self.dim2])
        return out

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0], input_shape[1], self.dim1, self.dim2])

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"dim1": self.dim1, "dim2": self.dim2, "validate": self.validate})
        return cfg


@tf.keras.utils.register_keras_serializable(
    package="tessera.layers.utils",
    name="ReverseLayer",
)
class ReverseLayer(tf.keras.layers.Layer):
    """
    Reverses a tensor along the specified axis.

    Args:
        axis: Axis or axes along which to reverse

    Example:
        ```python
        reverse = ReverseLayer(axis=[-2])
        reversed_seq = reverse(sequence)  # Reverse sequence dimension
        ```
    """

    def __init__(self, axis, **kwargs):
        super(ReverseLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.reverse(inputs, axis=self.axis)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"axis": self.axis})
        return cfg


@tf.keras.utils.register_keras_serializable(
    package="tessera.layers.utils",
    name="NamedIdentityLayer",
)
class NamedIdentityLayer(tf.keras.layers.Layer):
    """
    A Keras layer that acts as identity (pass-through) but provides a named output
    for later extraction. This is Keras serialization safe.

    Useful for extracting intermediate activations from a model without modifying
    the computation graph.

    Example:
        ```python
        identity = NamedIdentityLayer(name='intermediate_output')
        output = identity(hidden_layer)  # Same as hidden_layer, but named
        ```
    """

    def __init__(self, **kwargs):
        super(NamedIdentityLayer, self).__init__(**kwargs)

    def call(self, inputs):
        """Pass inputs through unchanged."""
        return inputs

    def get_config(self):
        config = super(NamedIdentityLayer, self).get_config()
        return config


@tf.keras.utils.register_keras_serializable(
    package="tessera.layers.utils",
    name="FlattenLastTwoDims",
)
class FlattenLastTwoDims(tf.keras.layers.Layer):
    """
    Flattens the last two dimensions: [B, N, D1, D2] -> [B, N, D1*D2]

    XLA-safe: uses tensor shapes in `call`, no Python control flow on tensors.
    Will expose static output dim if D1 and D2 are known at build time.

    Input shape:
        (batch_size, seq_len, dim1, dim2)

    Output shape:
        (batch_size, seq_len, dim1 * dim2)

    Example:
        ```python
        flatten = FlattenLastTwoDims()
        output = flatten(inputs)  # [B, N, 20, 4] -> [B, N, 80]
        ```
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = tf.keras.layers.InputSpec(ndim=4)  # enforce rank 4
        self._static_flat_dim = None  # cached when statically known

    def build(self, input_shape):
        # input_shape is a tf.TensorShape; dims may be None
        d1 = input_shape[-2]
        d2 = input_shape[-1]
        if d1 is not None and d2 is not None:
            self._static_flat_dim = int(d1) * int(d2)
        super().build(input_shape)

    def call(self, inputs):
        s = tf.shape(inputs)  # 1-D: [B, N, D1, D2]
        # Build the target shape as a 1-D tensor via tf.concat so XLA never needs
        # to Pack scalar tensors with mismatched shapes (which breaks when this layer
        # is instantiated as a throwaway object inside a traced function).
        flat_dim = tf.expand_dims(s[-2] * s[-1], axis=0)  # shape [1]
        new_shape = tf.concat([s[:-2], flat_dim], axis=0)  # [B, N, D1*D2]
        out = tf.reshape(inputs, new_shape)

        # Annotate static output dim when known to aid downstream shape inference.
        if self._static_flat_dim is not None:
            out = tf.ensure_shape(out, [None, None, self._static_flat_dim])
        return out

    def compute_output_shape(self, input_shape):
        b, n, d1, d2 = input_shape
        flat = (d1 * d2) if (d1 is not None and d2 is not None) else None
        return tf.TensorShape([b, n, flat])

    def get_config(self):
        cfg = super().get_config()
        # No runtime tensors stored; just persist whether we had a static flat dim for clarity
        cfg.update({"static_flat_dim": self._static_flat_dim})
        return cfg


@tf.keras.utils.register_keras_serializable(
    package="tessera.layers.utils",
    name="Conv1DFor4D",
)
class Conv1DFor4D(tf.keras.layers.Layer):
    """
    Efficient Conv1D layer for 4D tensors that avoids TimeDistributed overhead.

    Input shape: (batch, bag_size, sequence, features)
    Output shape: (batch, bag_size, sequence, filters)

    This layer reshapes 4D input to 3D, applies Conv1D, then reshapes back to 4D.
    Much faster than TimeDistributed(Conv1D) for compilation and execution.

    Args:
        filters: Number of output filters
        kernel_size: Size of the convolution kernel
        strides: Stride of the convolution
        padding: Padding mode ('valid' or 'same')
        activation: Activation function
        use_bias: Whether to use bias
        **kwargs: Other Conv1D arguments

    Example:
        ```python
        conv = Conv1DFor4D(filters=64, kernel_size=3, padding='same')
        output = conv(inputs)  # [B, N, L, F] -> [B, N, L, 64]
        ```
    """

    def __init__(self, filters, kernel_size, strides=1, padding='valid',
                 data_format='channels_last', dilation_rate=1, groups=1,
                 activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                 bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                 activity_regularizer=None, kernel_constraint=None, bias_constraint=None,
                 **kwargs):
        super(Conv1DFor4D, self).__init__(**kwargs)

        # Store parameters for Conv1D layer
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.groups = groups
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint

        # Create the Conv1D layer
        self.conv1d = tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint
        )

    def call(self, inputs):
        """
        Apply Conv1D to 4D tensor efficiently.

        Args:
            inputs: 4D tensor with shape (batch, bag_size, sequence, features)

        Returns:
            4D tensor with shape (batch, bag_size, new_sequence, filters)
        """
        # Get input shape
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        bag_size = input_shape[1]
        seq_len = input_shape[2]
        features = input_shape[3]

        # Reshape to 3D: (batch*bag_size, sequence, features)
        inputs_reshaped = tf.reshape(inputs, [batch_size * bag_size, seq_len, features])

        # Apply Conv1D
        conv_output = self.conv1d(inputs_reshaped)

        # Get output shape info
        conv_shape = tf.shape(conv_output)
        new_seq_len = conv_shape[1]

        # Reshape back to 4D: (batch, bag_size, new_sequence, filters)
        output = tf.reshape(conv_output, [batch_size, bag_size, new_seq_len, self.filters])

        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'groups': self.groups,
            'activation': self.activation,
            'use_bias': self.use_bias,
            'kernel_initializer': self.kernel_initializer,
            'bias_initializer': self.bias_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'bias_regularizer': self.bias_regularizer,
            'activity_regularizer': self.activity_regularizer,
            'kernel_constraint': self.kernel_constraint,
            'bias_constraint': self.bias_constraint
        })
        return config