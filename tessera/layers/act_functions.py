"""Custom activation function layers.

Defines the Activations dispatcher and two specialised activations (ASU, ARU) used
by other TESSERA layer modules where standard ReLU / SiLU give either over-aggressive
zeroing or insufficiently smooth gradients near the origin.
"""

import tensorflow as tf

class Activations:

    @staticmethod
    def asu_activation_function(x, lower_asymptote, upper_asymptote, lower_alpha, upper_alpha):
        x_2 = x ** 2
        lower_sqrt = (lower_alpha + x_2) ** (1 / 2)
        upper_sqrt = (upper_alpha + x_2) ** (1 / 2)
        return lower_asymptote + ((upper_asymptote - lower_asymptote) * ((x + lower_sqrt) / (lower_sqrt + upper_sqrt)))

    @staticmethod
    def aru_activation_function(x, alpha):
        return (x + ((alpha + (x ** 2)) ** (1 / 2))) / 2

    class ASU(tf.keras.layers.Layer):
        def __init__(self, trainable=True, lower_asymptote=0., upper_asymptote=1., alpha_init=1e-3, bias_init=None, use_exp_alpha=False, **kwargs):
            # Extract name from kwargs if provided, otherwise use default
            layer_name = kwargs.pop('name', 'ASU')
            super(Activations.ASU, self).__init__(name=layer_name, **kwargs)
            self.trainable = trainable
            self.lower_asymptote = lower_asymptote
            self.upper_asymptote = upper_asymptote
            self.alpha_init = alpha_init
            self.lower_alpha = None
            self.upper_alpha = None
            self.bias_init = bias_init
            self.bias = None
            self.use_exp_alpha = use_exp_alpha  # Option to use original exponential behavior

        def build(self, input_shape):
            # Use much smaller default initialization to prevent saturation
            self.lower_alpha = self.add_weight(shape=[input_shape[-1], ],
                                               initializer=tf.keras.initializers.constant(self.alpha_init),
                                               dtype=tf.float32, trainable=self.trainable, name = 'lower_alpha')
            self.upper_alpha = self.add_weight(shape=[input_shape[-1], ],
                                               initializer=tf.keras.initializers.constant(self.alpha_init),
                                               dtype=tf.float32, trainable=self.trainable, name = 'upper_alpha')
            if self.bias_init is not None:
                # Fix: use bias_init instead of alpha_init for bias initialization
                self.bias = self.add_weight(shape=[input_shape[-1], ], 
                                            initializer=tf.keras.initializers.constant(self.bias_init),
                                            dtype=tf.float32, trainable=self.trainable, name = 'bias')
            super().build(input_shape)

        def call(self, inputs, **kwargs):
            if self.use_exp_alpha:
                # Original exponential behavior (can cause saturation)
                lower_alpha_processed = tf.exp(self.lower_alpha)
                upper_alpha_processed = tf.exp(self.upper_alpha)
            else:
                # Improved: Use softplus activation to ensure positivity without explosive growth
                # softplus(x) = log(1 + exp(x)) ≈ x for large x, but smoother near 0
                lower_alpha_processed = tf.nn.softplus(self.lower_alpha) + 1e-3  # Small epsilon for numerical stability
                upper_alpha_processed = tf.nn.softplus(self.upper_alpha) + 1e-3
            
            return Activations.asu_activation_function(inputs + self.bias if self.bias is not None else inputs,
                                                       lower_asymptote=self.lower_asymptote,
                                                       upper_asymptote=self.upper_asymptote,
                                                       lower_alpha=lower_alpha_processed,
                                                       upper_alpha=upper_alpha_processed)

        def get_config(self):
            config = super(Activations.ASU, self).get_config()
            config.update(
                {
                    "trainable": self.trainable,
                    "lower_asymptote": self.lower_asymptote,
                    "upper_asymptote": self.upper_asymptote,
                    "alpha_init": self.alpha_init,
                    "bias_init": self.bias_init,
                    "use_exp_alpha": self.use_exp_alpha,
                }
            )
            return config

        @classmethod
        def from_config(cls, config):
            return cls(**config)

    class ARU(tf.keras.layers.Layer):
        def __init__(self, trainable=True, alpha_init=0., bias_init=None, **kwargs):
            super(Activations.ARU, self).__init__(**kwargs,name='ARU')
            self.trainable = trainable
            self.alpha_init = alpha_init
            self.alpha = None
            self.bias_init = bias_init
            self.bias = None

        def build(self, input_shape):
            self.alpha = self.add_weight(shape=[input_shape[-1], ], initializer=tf.keras.initializers.constant(self.alpha_init), dtype=tf.float32, trainable=self.trainable,
                                         name='alpha')
            if self.bias_init is not None:
                self.bias = self.add_weight(shape=[input_shape[-1], ], initializer=tf.keras.initializers.constant(self.bias_init), dtype=tf.float32, trainable=True,
                                            name='bias')
            super().build(input_shape)

        def call(self, inputs, **kwargs):
            # Compute exp value - original exponential operation
            alpha_exp = tf.exp(self.alpha)
            return Activations.aru_activation_function(inputs + self.bias if self.bias is not None else inputs, alpha=alpha_exp)

        def get_config(self):
            config = super(Activations.ARU, self).get_config()
            config.update(
                {
                    "trainable": self.trainable,
                    "alpha_init": self.alpha_init,
                    "bias_init": self.bias_init,
                }
            )
            return config

        @classmethod
        def from_config(cls, config):
            return cls(**config)

