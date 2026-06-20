"""Custom Keras callbacks for TESSERA training.

Includes early-stopping variants (LossStopCallback, CustomEarlyStopping) that monitor
multi-objective losses, model-saving utilities (SaveModelH5Callback), an attention-
distribution monitor used during long pretraining runs, a temperature-annealing
callback for the InfoNCE objective, and an optimiser-reset callback for staged
training where one phase's optimiser state should not carry over.
"""

import os
import tensorflow as tf
import io
import numpy as np
from tessera.input_keys import get_base_input_keys

class LossStopCallback(tf.keras.callbacks.Callback):
    def __init__(self, loss_threshold,loss_name='recon_loss'):
        super(LossStopCallback, self).__init__()
        self.loss_threshold = loss_threshold
        self.loss_name = loss_name

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get(self.loss_name)
        if current_loss is not None and current_loss < self.loss_threshold:
            print(f"Loss reached the threshold of {self.loss_threshold}. Stopping training.")
            self.model.stop_training = True

class CustomEarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, monitor='val_loss', min_delta=0, patience=0, min_epochs=0, mode='min', verbose=0, restore_best_weights=False, min_relative_delta=0.0):
        super(CustomEarlyStopping, self).__init__()
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.min_epochs = min_epochs
        self.mode = mode
        self.verbose = verbose
        self.restore_best_weights = restore_best_weights
        self.min_relative_delta = min_relative_delta
        self.best_weights = None
        self.wait = 0
        self.stopped_epoch = 0
        self.best = float('inf') if self.mode == 'min' else -float('inf')

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            if self.verbose > 0:
                print(f"Warning: Early stopping metric '{self.monitor}' not found in logs. Available metrics: {list(logs.keys()) if logs else []}")
            return

        # Convert tensor to Python float if necessary (for newer TensorFlow/Keras)
        if isinstance(current, tf.Tensor):
            current = float(current.numpy())

        # Track patience counter from the start
        if self.mode == 'min':
            if self.best == float('inf'):
                comparison = True  # any finite value beats inf
            elif self.min_relative_delta > 0:
                comparison = current < self.best - abs(self.best) * self.min_relative_delta
            else:
                comparison = current < self.best - self.min_delta
        else:
            if self.best == -float('inf'):
                comparison = True  # any finite value beats -inf
            elif self.min_relative_delta > 0:
                comparison = current > self.best + abs(self.best) * self.min_relative_delta
            else:
                comparison = current > self.best + self.min_delta

        if comparison:
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1

        # Only enforce stopping after reaching minimum epochs
        if epoch >= self.min_epochs:
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.restore_best_weights and self.best_weights is not None:
                    if self.verbose > 0:
                        print('Restoring model weights from the end of the best epoch')
                    self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print(f'Epoch {self.stopped_epoch + 1}: early stopping')


class SaveModelH5Callback(tf.keras.callbacks.Callback):
    """
    Custom callback that saves your `model_inf` as a full .h5 model
    at the end of each epoch when the monitored metric improves.
    """

    def __init__(
        self,
        model_inf,            # The model you want to save (self.model_inf)
        model_dir,            # Directory to save the .h5 file
        monitor="val_loss",   # Metric to monitor
        mode="min",           # 'min' or 'max'
        verbose=1,
        save_best_only=True
    ):
        super().__init__()
        self.model_inf = model_inf
        self.model_dir = model_dir
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose
        self.save_best_only = save_best_only

        # Initialize best_value depending on 'mode'
        if self.mode == "min":
            self.best_value = float("inf")
        else:  # mode == "max"
            self.best_value = -float("inf")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_value = logs.get(self.monitor)

        # If the monitored metric is missing, do nothing
        if current_value is None:
            return

        # Check if there's improvement
        improved = False
        if self.mode == "min" and current_value < self.best_value:
            improved = True
        elif self.mode == "max" and current_value > self.best_value:
            improved = True

        if self.save_best_only:
            # Save only if improvement
            if improved:
                if self.verbose > 0:
                    print(f"Epoch {epoch + 1}: {self.monitor} improved "
                          f"from {self.best_value:.6f} to {current_value:.6f}. Saving model ...")
                self.best_value = current_value
                # <-- IMPORTANT: Save the model in .h5 format
                self.model_inf.save(os.path.join(self.model_dir, "best_model.h5"))
        else:
            # Save on every epoch
            if self.verbose > 0:
                if improved:
                    print(f"Epoch {epoch + 1}: {self.monitor} improved "
                          f"from {self.best_value:.6f} to {current_value:.6f}.")
                    self.best_value = current_value
                print(f"Saving model to 'best_model.h5' (save_best_only=False).")

            self.model_inf.save(os.path.join(self.model_dir, "best_model.h5"))


class AttentionDistributionCallback(tf.keras.callbacks.Callback):
    """Callback to monitor attention weight distribution during training."""

    def __init__(self, model_attn_mil, dataset, log_dir='./logs/attention_dist',
                 update_freq=10, sample_size=None, save_dir=None,print_stats=True):
        """
        Args:
            model_attn_mil: Model that outputs attention weights
            dataset: Dataset to extract attention weights from
            log_dir: Directory for TensorBoard logs
            update_freq: Update frequency (in batches)
            sample_size: Number of batches to use (None for all)
            save_dir: Directory to save plots directly as files (if None, uses model_dir)
        """
        super(AttentionDistributionCallback, self).__init__()
        self.model_attn_mil = model_attn_mil
        self.dataset = dataset
        self.log_dir = log_dir
        self.update_freq = update_freq
        self.sample_size = sample_size
        self.save_dir = save_dir

        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        self.batch_count = 0
        self.print_stats = print_stats

    def on_train_begin(self, logs=None):
        # Initialize the list for attention weights
        self.att_weights_hist = []
        print("AttentionDistributionCallback initialized. Will update every", self.update_freq, "batches.")

        # Test the attention model once to make sure it works
        print("Testing attention model...")
        try:
            test_batch = next(iter(self.dataset))
            test_weights = self._extract_attention_from_batch(test_batch)
            print(f"Test successful! Attention shape: {test_weights.shape}")

            # Save a test histogram to verify plotting works
            self._save_histogram(test_weights, "test_attention_hist.png")
            print(f"Saved test histogram to {os.path.join(self.save_dir, 'test_attention_hist.png')}")
        except Exception as e:
            print(f"Error in test run: {str(e)}")
            import traceback
            traceback.print_exc()

    def on_batch_end(self, batch, logs=None):
        """Update attention weights histogram periodically."""
        self.batch_count += 1
        if self.batch_count % self.update_freq == 0:
            if self.print_stats:
                print(f"\nCollecting attention weights at batch {self.batch_count}...")

            # Calculate attention distributions
            try:
                attention_weights = self._get_attention_weights()
                if self.print_stats:
                    print(f"Collected {len(attention_weights)} attention weights.")

                if len(attention_weights) == 0:
                    if self.print_stats:
                        print("Warning: No attention weights collected. Check your model outputs.")
                    return

                # Print the range of values for debugging
                if self.print_stats:
                    print(f"Attention weight range: {np.min(attention_weights)} to {np.max(attention_weights)}")

                # Calculate basic statistics for console output
                mean_val = np.mean(attention_weights)
                median_val = np.median(attention_weights)
                min_val = np.min(attention_weights)
                max_val = np.max(attention_weights)
                if self.print_stats:
                    print(
                        f"Attention stats - Mean: {mean_val:.4f}, Median: {median_val:.4f}, Min: {min_val:.4f}, Max: {max_val:.4f}")

                epoch = logs.get('epoch', 0)

                # Save the figure directly as a file if save_dir is provided
                if self.save_dir:
                    save_path = os.path.join(self.save_dir, f'attention_dist_batch_{self.batch_count}.png')
                    self._save_histogram(attention_weights, f'attention_dist_batch_{self.batch_count}.png')
                    if self.print_stats:
                        print(f"Saved attention plot to {save_path}")

                # Store for later analysis
                self.att_weights_hist.append({
                    'batch': self.batch_count,
                    'epoch': epoch,
                    'weights': attention_weights,
                    'stats': {
                        'mean': mean_val,
                        'median': median_val,
                        'min': min_val,
                        'max': max_val,
                        'std': np.std(attention_weights)
                    }
                })

            except Exception as e:
                print(f"Error collecting attention weights: {str(e)}")
                import traceback
                traceback.print_exc()

    def _extract_attention_from_batch(self, batch):
        """Extract attention weights from a single batch."""
        # Filter inputs for the attention model
        keys = get_base_input_keys()
        if 'vaf' in batch:
            keys.append('vaf')

        filtered_batch = {k: batch[k] for k in keys if k in batch}

        # Get attention weights
        weights = self.model_attn_mil(filtered_batch)

        # Handle different return types
        if isinstance(weights, list) or isinstance(weights, tuple):
            weights = weights[0]  # Assume first output is attention weights

        weights = weights.numpy()

        # Flatten all non-zero weights
        all_weights = []
        for sample_weights in weights:
            valid_weights = sample_weights[sample_weights != 0]
            if len(valid_weights) > 0:
                all_weights.extend(valid_weights.flatten())

        return np.array(all_weights)

    def _get_attention_weights(self):
        """Collect attention weights from the model."""
        all_weights = []

        # Limit the number of batches for efficiency
        dataset_iter = iter(self.dataset)
        count = 0

        for batch in dataset_iter:
            try:
                weights = self._extract_attention_from_batch(batch)
                all_weights.extend(weights)

                count += 1
                if self.sample_size is not None and count >= self.sample_size:
                    break

            except Exception as e:
                print(f"Error processing batch: {str(e)}")
                continue

        return np.array(all_weights)

    def _save_histogram(self, weights, filename):
        """Directly save a histogram to a file."""
        if not self.save_dir:
            return

        # matplotlib is an optional, training-only diagnostic dependency: import it
        # lazily and skip the plot (with a warning) if it isn't installed, so a
        # missing plotting library never crashes a training run.
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            import warnings
            warnings.warn(
                "matplotlib is not installed; skipping the attention-weight "
                "histogram. Run `pip install matplotlib` to enable this diagnostic.",
                stacklevel=2,
            )
            return

        plt.figure(figsize=(10, 6))
        plt.hist(weights, bins=50, alpha=0.7, color='blue', density=True)
        plt.title(f'Attention Weights Distribution (Batch {self.batch_count})')
        plt.xlabel('Attention Weight Value')
        plt.ylabel('Density')
        plt.grid(True, alpha=0.3)

        # Add distribution statistics
        if len(weights) > 0:
            mean = np.mean(weights)
            median = np.median(weights)
            std = np.std(weights)
            max_val = np.max(weights)
            min_val = np.min(weights)

            stats_text = (f'Mean: {mean:.4f}\nMedian: {median:.4f}\nStd: {std:.4f}\n'
                          f'Min: {min_val:.4f}\nMax: {max_val:.4f}\nCount: {len(weights)}')

            plt.figtext(0.15, 0.7, stats_text, bbox=dict(facecolor='white', alpha=0.5))

        plt.savefig(os.path.join(self.save_dir, filename))
        plt.close()

class TemperatureAnnealingCallback(tf.keras.callbacks.Callback):
    def __init__(self, layer, initial_temp, min_temp, epochs):
        super().__init__()
        self.layer = layer
        self.initial_temp = initial_temp
        self.min_temp = min_temp
        self.epochs = epochs

    def on_epoch_begin(self, epoch, logs=None):
        # Exponential decay
        temperature = max(self.initial_temp * np.exp(-3.0 * epoch / self.epochs), self.min_temp)
        self.layer.set_temperature(temperature)
        print(f"\nEpoch {epoch + 1}: Setting temperature to {temperature:.4f}")


class OptimizerResetCallback(tf.keras.callbacks.Callback):
    def __init__(self, reset_epoch, optimizer_factory, verbose=1):
        """
        Callback to reset optimizer at a specified epoch.
        
        Args:
            reset_epoch (int): Epoch number at which to reset the optimizer (0-indexed)
            optimizer_factory (callable): Function that returns a new optimizer instance
            verbose (int): Verbosity level (0 = silent, 1 = progress messages)
        """
        super().__init__()
        self.reset_epoch = reset_epoch
        self.optimizer_factory = optimizer_factory
        self.verbose = verbose
        
    def on_epoch_begin(self, epoch, logs=None):
        # Reset at the beginning of the target epoch
        if epoch == self.reset_epoch:
            if self.verbose > 0:
                print(f"\nEpoch {epoch + 1}: Resetting optimizer state...")
                
            # Create new optimizer to get fresh configuration
            new_optimizer = self.optimizer_factory()
            
            # Handle mixed precision wrapper
            current_optimizer = self.model.optimizer
            is_mixed_precision = hasattr(current_optimizer, 'inner_optimizer')
            
            if is_mixed_precision:
                # For mixed precision, we need to work with the inner optimizer
                target_optimizer = current_optimizer.inner_optimizer
                new_optimizer = tf.keras.mixed_precision.LossScaleOptimizer(new_optimizer)
                new_inner_optimizer = new_optimizer.inner_optimizer
                if self.verbose > 0:
                    print("Resetting mixed precision optimizer state...")
            else:
                target_optimizer = current_optimizer
                new_inner_optimizer = new_optimizer
            
            # Reset the learning rate
            if hasattr(new_inner_optimizer, 'learning_rate'):
                target_optimizer.learning_rate.assign(new_inner_optimizer.learning_rate)
                if self.verbose > 0:
                    print(f"Learning rate reset to: {new_inner_optimizer.learning_rate}")
            
            # Reset optimizer state variables (momentum, etc.)
            # This is the key part - we reset the internal state without recompiling
            try:
                # Clear any existing optimizer state
                for var in target_optimizer.variables:
                    var.assign(tf.zeros_like(var))
                
                if self.verbose > 0:
                    print("Optimizer state variables reset to zero")
                    
            except Exception as e:
                if self.verbose > 0:
                    print(f"Warning: Could not reset all optimizer variables: {e}")
            
            # If we have mixed precision, reset the loss scale as well
            if is_mixed_precision:
                try:
                    # Reset loss scaling state
                    if hasattr(current_optimizer, 'loss_scale'):
                        current_optimizer.loss_scale.assign(new_optimizer.loss_scale)
                    if hasattr(current_optimizer, '_loss_scale_manager'):
                        # Reset loss scale manager state if possible
                        current_optimizer._loss_scale_manager = new_optimizer._loss_scale_manager
                    if self.verbose > 0:
                        print("Mixed precision loss scaling reset")
                except Exception as e:
                    if self.verbose > 0:
                        print(f"Warning: Could not fully reset mixed precision state: {e}")
            
            if self.verbose > 0:
                print("Optimizer reset completed without recompilation")
                if is_mixed_precision:
                    print(f"Current learning rate: {current_optimizer.inner_optimizer.learning_rate}")
                else:
                    print(f"Current learning rate: {current_optimizer.learning_rate}")


class TensorToFloatCallback(tf.keras.callbacks.Callback):
    """
    Callback that converts tensor values in logs to Python floats.

    This is necessary because CSVLogger (and potentially other callbacks) cannot
    handle TensorFlow tensors directly. CSVLogger checks if a value is Iterable,
    and tf.Tensor is Iterable but iterating over a scalar tensor fails with
    "Cannot iterate over a scalar tensor".

    Place this callback BEFORE CSVLogger in the callback list to ensure
    all metric values are converted to floats before CSVLogger processes them.

    Example:
        callbacks = [
            TensorToFloatCallback(),  # Must come before CSVLogger
            CSVLogger('training.log'),
            ...
        ]
    """

    def on_epoch_end(self, epoch, logs=None):
        """Convert any tensor values in logs to Python floats."""
        if logs is None:
            return

        for key in logs:
            value = logs[key]
            # Convert TensorFlow tensors to Python floats
            if isinstance(value, tf.Tensor):
                try:
                    logs[key] = float(value.numpy())
                except Exception:
                    # If conversion fails, try to convert to string
                    try:
                        logs[key] = str(value.numpy())
                    except Exception:
                        pass  # Leave as-is if all conversions fail
