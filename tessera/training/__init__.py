"""Training utilities for TESSERA.

This package consolidates all training-related functionality:
- callbacks: Custom Keras callbacks for training control
- losses: Loss functions for variant reconstruction
- metrics: Evaluation metrics
- schedules: Learning rate schedules
- logging: Gradient and training logging utilities
- utils: Weight management and other training utilities
- models: Custom Keras training models (CustomTrainingModel, CustomTrainingModelMIL)
"""

# ===== Callbacks =====
from .callbacks import (
    LossStopCallback,
    CustomEarlyStopping,
    SaveModelH5Callback,
    AttentionDistributionCallback,
    TemperatureAnnealingCallback,
    OptimizerResetCallback,
)

# ===== Losses =====
from .losses import compute_loss

# ===== Metrics =====
from .metrics import (
    masked_accuracy,
    AUCLayer,
    compute_auc,
)

# ===== Learning Rate Schedules =====
from .schedules import (
    WarmUpThenDecaySchedule,
    CosineAnnealingWithWarmup,
)

# ===== Gradient Logging =====
from .logging import (
    SimpleGradientLogger,
    GradientTrackingModel,
    create_simple_gradient_logger,
    wrap_model_with_gradient_tracking,
)

# ===== Utilities =====
from .utils import assign_subset_weights

# ===== Training Models =====
from .models import (
    CustomTrainingModel,
    CustomTrainingModelMIL,
)

__all__ = [
    # Callbacks
    'LossStopCallback',
    'CustomEarlyStopping',
    'SaveModelH5Callback',
    'AttentionDistributionCallback',
    'TemperatureAnnealingCallback',
    'OptimizerResetCallback',
    # Losses
    'compute_loss',
    # Metrics
    'masked_accuracy',
    'AUCLayer',
    'compute_auc',
    # Schedules
    'WarmUpThenDecaySchedule',
    'CosineAnnealingWithWarmup',
    # Logging
    'SimpleGradientLogger',
    'GradientTrackingModel',
    'create_simple_gradient_logger',
    'wrap_model_with_gradient_tracking',
    # Utils
    'assign_subset_weights',
    # Models
    'CustomTrainingModel',
    'CustomTrainingModelMIL',
]
