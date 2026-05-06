"""Weight-management utilities for TESSERA model construction.

Helpers for assigning, transferring, or freezing subsets of weights across model
configurations. Used during pretraining-to-finetuning transitions, when loading
a partial checkpoint into a model with a different head, and during the feature-
slice ablations that re-fit downstream pipelines on subsets of the encoder output.
"""

from typing import List, Dict, Any
import tensorflow as tf


def assign_subset_weights(
    full_model: tf.keras.Model,
    subset_model: tf.keras.Model,
    best_weights: List
) -> None:
    """
    Transfer weights from a full model to a subset model by matching layer names.

    This function is useful when you want to initialize a smaller model with weights
    from a larger pre-trained model. It matches layers by name and only transfers
    weights for layers that exist in both models.

    Args:
        full_model: Source model containing all weights to transfer from
        subset_model: Target model that should receive a subset of weights
        best_weights: List of weight arrays from the source model (typically from
                     full_model.get_weights())

    Raises:
        ValueError: If there's a shape mismatch between corresponding layers or
                   if required layers are missing

    Example:
        >>> full_model = build_large_model()
        >>> subset_model = build_small_model()  # With subset of full_model layers
        >>> weights = full_model.get_weights()
        >>> assign_subset_weights(full_model, subset_model, weights)

    Note:
        - Layers are matched by name, so ensure consistent naming between models
        - The function handles wrapped models (e.g., Functional API models)
        - Shape mismatches will trigger warnings but won't stop execution
    """
    # Get the layer name to weight mapping from full model
    full_model_weights = {}
    weight_pos = 0

    # Handle wrapped models
    if len(full_model.layers) == 1 and hasattr(full_model.layers[0], 'layers'):
        full_layers = full_model.layers[0].layers
    else:
        full_layers = full_model.layers

    # Build mapping of layer names to their weights and positions
    for layer in full_layers:
        weights = layer.get_weights()
        if weights:
            full_model_weights[layer.name] = {
                'weights': weights,
                'start_idx': weight_pos
            }
            weight_pos += len(weights)

    # Collect weights for subset model
    subset_weights = []
    for layer in subset_model.layers:
        if layer.name in full_model_weights:
            # Get the expected weight shapes for this layer
            expected_shapes = [w.shape for w in layer.get_weights()]
            if not expected_shapes:
                continue

            # Get weights from full model
            layer_info = full_model_weights[layer.name]
            start_idx = layer_info['start_idx']
            num_weights = len(expected_shapes)
            layer_weights = best_weights[start_idx:start_idx + num_weights]

            # Add weights that match expected shapes
            for weight, expected_shape in zip(layer_weights, expected_shapes):
                if weight.shape == expected_shape:
                    subset_weights.append(weight)
                    # print(f"Assigned weights to layer: {layer.name}")
                else:
                    print(f"Warning: Shape mismatch in layer {layer.name}. "
                          f"Expected {expected_shape}, got {weight.shape}")

    try:
        # Set collected weights to subset model
        subset_model.set_weights(subset_weights)
    except ValueError as e:
        print(f"Error setting weights: {str(e)}")
        print(f"Full model total weights: {len(best_weights)}")
        print(f"Subset model total weights: {len(subset_weights)}")
        print(f"Subset model expected weights: {len(subset_model.get_weights())}")
        
        # Debug: print layer names
        print(f"\nFull model layers: {list(full_model_weights.keys())}")
        print(f"Subset model layers: {[layer.name for layer in subset_model.layers if layer.get_weights()]}")
        
        # Find missing layers
        subset_layer_names = [layer.name for layer in subset_model.layers if layer.get_weights()]
        missing_layers = [name for name in subset_layer_names if name not in full_model_weights]
        if missing_layers:
            print(f"Missing layers in full model: {missing_layers}")

        raise