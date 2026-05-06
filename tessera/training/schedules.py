"""Learning rate schedules for training optimization.

This module provides custom learning rate schedules for TensorFlow/Keras optimizers,
including warmup and cosine annealing strategies.
"""

from typing import Callable
import tensorflow as tf
import numpy as np


class WarmUpThenDecaySchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Learning rate schedule with linear warmup followed by decay.

    This schedule linearly increases the learning rate from 0 to initial_learning_rate
    over warmup_steps, then applies a custom decay function.

    Args:
        initial_learning_rate: Target learning rate after warmup
        decay_schedule_fn: Function that takes step number and returns learning rate
                          for the decay phase
        warmup_steps: Number of steps for linear warmup

    Example:
        >>> decay_fn = tf.keras.optimizers.schedules.ExponentialDecay(
        ...     initial_learning_rate=1e-3, decay_steps=1000, decay_rate=0.96
        ... )
        >>> schedule = WarmUpThenDecaySchedule(
        ...     initial_learning_rate=1e-3,
        ...     decay_schedule_fn=decay_fn,
        ...     warmup_steps=500
        ... )
        >>> optimizer = tf.keras.optimizers.Adam(learning_rate=schedule)
    """

    def __init__(
        self,
        initial_learning_rate: float,
        decay_schedule_fn: Callable[[int], float],
        warmup_steps: int
    ):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_schedule_fn = decay_schedule_fn
        self.warmup_steps = warmup_steps

    def __call__(self, step: tf.Tensor) -> tf.Tensor:
        """
        Compute learning rate for the given step.

        Args:
            step: Current training step

        Returns:
            Learning rate for this step
        """
        global_step_float = tf.cast(step, tf.float32)
        warmup_steps_float = tf.cast(self.warmup_steps, tf.float32)

        # Linear warmup from 0 to initial_learning_rate
        warmup_percent_done = global_step_float / warmup_steps_float
        warmup_learning_rate = self.initial_learning_rate * warmup_percent_done

        # Switch between warmup and decay based on current step
        is_warmup = tf.cast(global_step_float < warmup_steps_float, tf.float32)
        return ((1.0 - is_warmup) * self.decay_schedule_fn(step) +
                is_warmup * warmup_learning_rate)

    def get_config(self):
        """Get configuration for serialization."""
        return {
            'initial_learning_rate': self.initial_learning_rate,
            'warmup_steps': self.warmup_steps,
            # Note: decay_schedule_fn may not be serializable
        }


class CosineAnnealingWithWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Cosine annealing learning rate schedule with linear warmup.

    This schedule implements:
    1. Linear warmup: 0 -> max_learning_rate over warmup_steps
    2. Cosine annealing: max_learning_rate -> min_learning_rate over remaining steps

    The cosine annealing provides smooth decay that often works better than
    exponential decay for fine-tuning and long training runs.

    Args:
        max_learning_rate: Peak learning rate after warmup
        total_steps: Total number of training steps
        warmup_steps: Number of steps for linear warmup
        min_learning_rate: Minimum learning rate at end of schedule. Default: 0.0

    Example:
        >>> schedule = CosineAnnealingWithWarmup(
        ...     max_learning_rate=1e-3,
        ...     total_steps=10000,
        ...     warmup_steps=500,
        ...     min_learning_rate=1e-6
        ... )
        >>> optimizer = tf.keras.optimizers.Adam(learning_rate=schedule)

    Note:
        This is particularly effective for transformer-based models and
        when you want the learning rate to smoothly decrease to near-zero
        by the end of training.
    """

    def __init__(
        self,
        max_learning_rate: float,
        total_steps: int,
        warmup_steps: int,
        min_learning_rate: float = 0.0
    ):
        super().__init__()
        self.max_learning_rate = max_learning_rate
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_learning_rate = min_learning_rate

    def __call__(self, step: tf.Tensor) -> tf.Tensor:
        """
        Compute learning rate for the given step.

        Args:
            step: Current training step

        Returns:
            Learning rate for this step
        """
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        total_steps = tf.cast(self.total_steps, tf.float32)

        # Warmup phase: linear increase from 0 to max_learning_rate
        warmup_lr = (step / warmup_steps) * self.max_learning_rate

        # Cosine annealing phase: from max_learning_rate to min_learning_rate
        cosine_step = (step - warmup_steps) / (total_steps - warmup_steps)
        cosine_step = tf.clip_by_value(cosine_step, 0.0, 1.0)
        cosine_lr = self.min_learning_rate + (self.max_learning_rate - self.min_learning_rate) * \
                   0.5 * (1.0 + tf.cos(np.pi * cosine_step))

        # Choose warmup or cosine annealing based on current step
        is_warmup = tf.cast(step < warmup_steps, tf.float32)
        learning_rate = is_warmup * warmup_lr + (1.0 - is_warmup) * cosine_lr

        return learning_rate

    def get_config(self):
        """Get configuration for serialization."""
        return {
            'max_learning_rate': self.max_learning_rate,
            'total_steps': self.total_steps,
            'warmup_steps': self.warmup_steps,
            'min_learning_rate': self.min_learning_rate
        }
