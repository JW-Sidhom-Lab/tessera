"""Lightweight gradient and training logging via TensorBoard.

Provides per-layer gradient-norm tracking (SimpleGradientLogger) and a thin
GradientTrackingModel wrapper that integrates with Keras training loops without
requiring a custom train_step rewrite. Useful for diagnosing exploding/vanishing
gradients during long pretraining runs.
"""

import tensorflow as tf
import os
from typing import Optional


class SimpleGradientLogger(tf.keras.callbacks.Callback):
    """Simple gradient logger that tracks gradients to TensorBoard during training."""
    
    def __init__(self, log_dir: str, track_frequency: int = 10, log_histograms: bool = True):
        """
        Initialize the gradient logger.
        
        Args:
            log_dir: Directory to save TensorBoard logs
            track_frequency: How often to log gradients (every N batches)
            log_histograms: Whether to log gradient histograms (can be memory intensive)
        """
        super().__init__()
        self.log_dir = log_dir
        self.track_frequency = track_frequency
        self.log_histograms = log_histograms
        self.batch_count = 0
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Create TensorBoard writer
        self.writer = tf.summary.create_file_writer(log_dir)
        
        print(f"Simple gradient logger initialized. Logs will be saved to: {log_dir}")
        print(f"Tracking frequency: every {track_frequency} batches")
    
    def on_train_batch_end(self, batch, logs=None):
        """Track gradients at the end of each training batch."""
        self.batch_count += 1
        
        # Only track at specified frequency
        if self.batch_count % self.track_frequency != 0:
            return
        
        # Note: This callback relies on the CustomTrainingModel to call log_gradients()
        # directly from its train_step method. This callback just increments counters.
    
    def log_gradients(self, gradients, step: Optional[int] = None):
        """
        Manually log gradients. Call this from your training loop.
        
        Args:
            gradients: List of gradient tensors
            step: Current training step (if None, uses internal counter)
        """
        if step is None:
            step = self.batch_count
        
        # Get the model (handle wrapped models)
        model = self.model
        if hasattr(model, 'base_model'):
            model = model.base_model
        
        with self.writer.as_default():
            for i, (grad, var) in enumerate(zip(gradients, model.trainable_variables)):
                if grad is not None:
                    var_name = var.name.replace(':', '_')
                    
                    # Log gradient norms (always)
                    grad_norm = tf.norm(grad)
                    tf.summary.scalar(f"grad_norms/{var_name}", grad_norm, step=step)
                    
                    # Log gradient histograms (optional, memory intensive)
                    if self.log_histograms:
                        tf.summary.histogram(f"gradients/{var_name}", grad, step=step)
                    
                    # Log some basic statistics
                    tf.summary.scalar(f"grad_mean/{var_name}", tf.reduce_mean(tf.abs(grad)), step=step)
                    tf.summary.scalar(f"grad_max/{var_name}", tf.reduce_max(tf.abs(grad)), step=step)
                    
                    # Check for problematic gradients
                    if tf.reduce_any(tf.math.is_nan(grad)):
                        tf.summary.scalar(f"grad_nan_detected/{var_name}", 1.0, step=step)
                    if tf.reduce_any(tf.math.is_inf(grad)):
                        tf.summary.scalar(f"grad_inf_detected/{var_name}", 1.0, step=step)
            
            # Log overall gradient statistics
            valid_grads = [g for g in gradients if g is not None]
            if valid_grads:
                all_grad_norms = [tf.norm(g) for g in valid_grads]
                mean_grad_norm = tf.reduce_mean(all_grad_norms)
                max_grad_norm = tf.reduce_max(all_grad_norms)
                min_grad_norm = tf.reduce_min(all_grad_norms)
                
                tf.summary.scalar("gradient_stats/mean_norm", mean_grad_norm, step=step)
                tf.summary.scalar("gradient_stats/max_norm", max_grad_norm, step=step)
                tf.summary.scalar("gradient_stats/min_norm", min_grad_norm, step=step)
                tf.summary.scalar("gradient_stats/norm_ratio", max_grad_norm / (min_grad_norm + 1e-8), step=step)
        
        self.writer.flush()
    
    def on_train_end(self, logs=None):
        """Close the writer when training ends."""
        self.writer.close()
        print(f"Gradient logging complete. View logs with: tensorboard --logdir {self.log_dir}")


class GradientTrackingModel(tf.keras.Model):
    """
    Model wrapper that tracks gradients during training.
    
    This is a simpler alternative to the complex CustomTrainingModel approach.
    """
    
    def __init__(self, base_model, gradient_logger: SimpleGradientLogger):
        """
        Initialize the gradient tracking model.
        
        Args:
            base_model: The base Keras model to wrap
            gradient_logger: SimpleGradientLogger instance
        """
        super().__init__()
        self.base_model = base_model
        self.gradient_logger = gradient_logger
        self.global_step = 0
    
    def call(self, inputs, training=None, mask=None):
        """Forward pass through the base model."""
        return self.base_model(inputs, training=training, mask=mask)
    
    @property
    def trainable_variables(self):
        return self.base_model.trainable_variables
    
    @property
    def non_trainable_variables(self):
        return self.base_model.non_trainable_variables
    
    def train_step(self, data):
        """Custom training step that tracks gradients."""
        # Extract inputs and targets
        if isinstance(data, dict):
            x = data
            y_true = data.get('alt')  # Assuming 'alt' contains the labels
        else:
            x, y_true = data
        
        with tf.GradientTape() as tape:
            # Forward pass
            y_pred = self.base_model(x, training=True)
            
            # Compute loss
            if hasattr(self.base_model, 'custom_loss_fn'):
                loss = self.base_model.custom_loss_fn(y_true, y_pred)
            else:
                # Default loss computation for your model structure
                if isinstance(y_pred, dict) and 'logits' in y_pred:
                    logits = y_pred['logits']
                    target_shape = tf.shape(logits)[:-1]
                    y_true_dummy = tf.ones(target_shape, dtype=tf.int32)
                    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
                    loss = loss_fn(y_true_dummy, logits)
                else:
                    loss = tf.reduce_mean(tf.square(y_true - y_pred))
        
        # Compute gradients
        gradients = tape.gradient(loss, self.trainable_variables)
        
        # Log gradients
        if self.global_step % self.gradient_logger.track_frequency == 0:
            self.gradient_logger.log_gradients(gradients, self.global_step)
        
        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        self.global_step += 1
        
        return {'loss': loss}


def create_simple_gradient_logger(log_dir: str, track_frequency: int = 10, 
                                log_histograms: bool = True) -> SimpleGradientLogger:
    """
    Factory function to create a simple gradient logger.
    
    Args:
        log_dir: Directory to save TensorBoard logs
        track_frequency: How often to log gradients (every N batches)
        log_histograms: Whether to log gradient histograms
    
    Returns:
        SimpleGradientLogger instance
    """
    return SimpleGradientLogger(
        log_dir=log_dir,
        track_frequency=track_frequency,
        log_histograms=log_histograms
    )


def wrap_model_with_gradient_tracking(model, gradient_logger: SimpleGradientLogger):
    """
    Wrap a model with gradient tracking capability.
    
    Args:
        model: The base Keras model
        gradient_logger: SimpleGradientLogger instance
    
    Returns:
        GradientTrackingModel wrapper
    """
    wrapped_model = GradientTrackingModel(model, gradient_logger)
    
    # Copy compilation settings
    if hasattr(model, 'optimizer'):
        compile_kwargs = {'optimizer': model.optimizer}
        
        if hasattr(model, 'custom_loss_fn'):
            compile_kwargs['loss'] = model.custom_loss_fn
        elif hasattr(model, 'loss'):
            compile_kwargs['loss'] = model.loss
        
        if hasattr(model, 'metrics'):
            compile_kwargs['metrics'] = model.metrics
        
        wrapped_model.compile(**compile_kwargs)
    
    return wrapped_model