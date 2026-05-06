"""Input-key registries for TESSERA's named-input dictionaries.

Returns the list of input names the Keras model expects given a particular modality
and feature combination (mutations only, mutations + VAF, CNA + LOH, joint multi-modal,
etc.). Used by training scripts and inference paths to filter batched data down to
the keys the model graph actually consumes, so a single dataset object can serve
multiple model configurations without rebuilding tensors.
"""

def get_base_input_keys():
    """
    Get base input keys for mutation/variant data (without VAF).

    This is an alias for get_mutation_input_keys() for backward compatibility.

    Returns:
        list: Base input keys ['ref', 'alt', 'context_5p', 'context_3p', 'chr', 'pos']
    """
    return get_mutation_input_keys()

def get_mutation_input_keys():
    """Get input keys required for mutation/variant data."""
    return ['ref', 'alt', 'context_5p', 'context_3p', 'chr', 'pos']

def get_cna_input_keys(use_loh=False):
    """
    Get input keys required for CNA (Copy Number Alteration) data.

    Args:
        use_loh (bool): Whether to include LOH feature

    Returns:
        list: CNA input keys including optional features
    """
    keys = ['cna_chr', 'cna_start', 'cna_end', 'cna_length', 'cna_segment_mean']

    # Add optional features
    if use_loh:
        keys.append('cna_loh')

    return keys

def get_input_keys(use_mut=True, use_vaf=False, use_cna=False, include_label=False,
                   use_cna_loh=False):
    """
    Get input keys based on model configuration.

    This function builds the list of required input keys based on the data modalities
    being used. Supports mutation-only, CNA-only, or multi-modal configurations.

    Args:
        use_mut (bool): Whether to include mutation/variant data. Default: True
        use_vaf (bool): Whether to include VAF (Variant Allele Frequency) data
        use_cna (bool): Whether to include CNA (Copy Number Alteration) data
        include_label (bool): Whether to include label data for supervised learning
        use_cna_loh (bool): Whether to include CNA LOH feature

    Returns:
        list: List of input keys for the model

    Examples:
        >>> # Mutation-only model (default)
        >>> get_input_keys()
        ['ref', 'alt', 'context_5p', 'context_3p', 'chr', 'pos']

        >>> # CNA-only model
        >>> get_input_keys(use_mut=False, use_cna=True)
        ['cna_chr', 'cna_start', 'cna_end', 'cna_segment_mean']

        >>> # Multi-modal model with LOH feature
        >>> get_input_keys(use_mut=True, use_cna=True, use_cna_loh=True)
        ['ref', 'alt', 'context_5p', 'context_3p', 'chr', 'pos', 'cna_chr', 'cna_start',
         'cna_end', 'cna_length', 'cna_segment_mean', 'cna_loh']
    """
    keys = []

    # Add mutation keys if using mutation data
    if use_mut:
        keys.extend(get_mutation_input_keys())

    # Add VAF if requested (only relevant for mutations)
    if use_vaf:
        keys.append('vaf')

    # Add CNA keys if using CNA data
    if use_cna:
        keys.extend(get_cna_input_keys(
            use_loh=use_cna_loh
        ))

    # Add label for supervised learning
    if include_label:
        keys.append('label')

    return keys