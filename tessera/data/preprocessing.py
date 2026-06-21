"""Preprocessing utilities for TESSERA inputs.

Transforms raw somatic data into the tensors consumed by the model: FASTA-context
lookup for variants, token-level sequence encoding, chromosome and position
normalisation for CNA segments, sample-level bagging, and helpers for converting
attention attributions into pandas DataFrames downstream of inference.
"""

from typing import Tuple, Dict, Any, Optional, Sequence, Union
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import gaussian_kde, rankdata


def create_dataset_indices(
    dataset_size: int,
    validation_split: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split dataset indices into training, validation, and test sets.

    Args:
        dataset_size: Total number of samples in the dataset
        validation_split: Fraction of data to use for validation+test (e.g., 0.2 for 20%)

    Returns:
        Tuple containing (train_indices, validation_indices, test_indices)

    Example:
        >>> train_idx, val_idx, test_idx = create_dataset_indices(1000, 0.2)
        >>> len(train_idx) + len(val_idx) + len(test_idx) == 1000
        True
    """
    val_test_size = int(dataset_size * validation_split)
    val_size = test_size = val_test_size // 2

    # Adjust for odd numbers to ensure all samples are included
    if val_test_size % 2 != 0:
        test_size += 1

    # Generate shuffled indices for random splitting
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)

    # Split indices into train, validation, and test sets
    train_end = dataset_size - val_test_size
    valid_end = dataset_size - test_size

    train_indices = indices[:train_end]
    valid_indices = indices[train_end:valid_end]
    test_indices = indices[valid_end:]

    return train_indices, valid_indices, test_indices

def gkde_smooth_and_sort(
    x: np.ndarray,
    y: np.ndarray,
    weights: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, gaussian_kde, np.ndarray]:
    """
    Apply Gaussian Kernel Density Estimation and sort by density.

    Computes a kernel density estimate for 2D data and returns the data
    sorted by increasing density values.

    Args:
        x: X-coordinates of data points
        y: Y-coordinates of data points
        weights: Optional weights for each data point

    Returns:
        Tuple containing:
            - x_sorted: X-coordinates sorted by density
            - y_sorted: Y-coordinates sorted by density
            - z_sorted: Density values at each point (sorted)
            - kernel: The fitted Gaussian KDE object
            - sort_indices: Indices used for sorting

    Example:
        >>> x = np.random.rand(100)
        >>> y = np.random.rand(100)
        >>> x_s, y_s, z_s, kde, idx = gkde_smooth_and_sort(x, y)
    """
    xy = np.vstack([x, y])
    kernel = gaussian_kde(xy, weights=weights)
    density = kernel(xy)
    sort_indices = np.argsort(density)

    x_sorted = x[sort_indices]
    y_sorted = y[sort_indices]
    z_sorted = density[sort_indices]

    return x_sorted, y_sorted, z_sorted, kernel, sort_indices

class ChromosomeMapper:
    def __init__(self, genome_version='GRCh37'):
        self.genome_version = genome_version.upper()
        if self.genome_version not in ['GRCH37', 'GRCH38']:
            raise ValueError("Genome version must be either 'GRCh37' or 'GRCh38'")
        self.chr_to_refseq = self._get_chromosome_mapping()
        self.chr_to_refseq.update({f'chr{k}': v for k, v in self.chr_to_refseq.items()})

    def _get_chromosome_mapping(self):
        if self.genome_version == 'GRCH37':
            return {
                '1': 'NC_000001.10', '2': 'NC_000002.11', '3': 'NC_000003.11',
                '4': 'NC_000004.11', '5': 'NC_000005.9', '6': 'NC_000006.11',
                '7': 'NC_000007.13', '8': 'NC_000008.10', '9': 'NC_000009.11',
                '10': 'NC_000010.10', '11': 'NC_000011.9', '12': 'NC_000012.11',
                '13': 'NC_000013.10', '14': 'NC_000014.8', '15': 'NC_000015.9',
                '16': 'NC_000016.9', '17': 'NC_000017.10', '18': 'NC_000018.9',
                '19': 'NC_000019.9', '20': 'NC_000020.10', '21': 'NC_000021.8',
                '22': 'NC_000022.10', 'X': 'NC_000023.10', 'Y': 'NC_000024.9',
                'MT': 'NC_012920.1'
            }
        else:  # GRCh38
            return {
                '1': 'NC_000001.11', '2': 'NC_000002.12', '3': 'NC_000003.12',
                '4': 'NC_000004.12', '5': 'NC_000005.10', '6': 'NC_000006.12',
                '7': 'NC_000007.14', '8': 'NC_000008.11', '9': 'NC_000009.12',
                '10': 'NC_000010.11', '11': 'NC_000011.10', '12': 'NC_000012.12',
                '13': 'NC_000013.11', '14': 'NC_000014.9', '15': 'NC_000015.10',
                '16': 'NC_000016.10', '17': 'NC_000017.11', '18': 'NC_000018.10',
                '19': 'NC_000019.10', '20': 'NC_000020.11', '21': 'NC_000021.9',
                '22': 'NC_000022.11', 'X': 'NC_000023.11', 'Y': 'NC_000024.10',
                'MT': 'NC_012920.1'
            }

def encode_sequence(
    sequence: str,
    token_map: Dict[str, int]
) -> np.ndarray:
    """
    Encode DNA sequence into integer tokens.

    Args:
        sequence: DNA sequence string (e.g., "ATCG")
        token_map: Dictionary mapping nucleotides to integer tokens
                  Should contain an 'X' key for unknown nucleotides

    Returns:
        Numpy array of integer tokens

    Example:
        >>> token_map = {'A': 1, 'T': 2, 'C': 3, 'G': 4, 'X': 0}
        >>> encode_sequence("ATCG", token_map)
        array([1, 2, 3, 4], dtype=int32)
    """
    default_token = token_map['X']
    return np.array(
        [token_map.get(nuc.upper(), default_token) for nuc in sequence],
        dtype=np.int32
    )


def get_sequence_context(
    fasta: Any,
    chrom: str,
    pos: int,
    context_len: int,
    chromosome_mapper: ChromosomeMapper
) -> Tuple[str, str]:
    """
    Extract sequence context around a genomic position.

    Args:
        fasta: Indexed FASTA file object (e.g., from pysam.FastaFile)
        chrom: Chromosome identifier (e.g., "chr1" or "1")
        pos: 1-based genomic position
        context_len: Number of bases to extract on each side
        chromosome_mapper: ChromosomeMapper instance for RefSeq conversion

    Returns:
        Tuple of (upstream_context, downstream_context) as uppercase strings

    Example:
        >>> upstream, downstream = get_sequence_context(fasta, "chr1", 1000, 10, mapper)
        >>> len(upstream) == 10 and len(downstream) == 10
        True
    """
    mapped_chrom = chromosome_mapper.chr_to_refseq.get(chrom, chrom)
    start = max(0, pos - context_len)
    end = pos + context_len

    upstream = str(fasta[mapped_chrom][start - 1:pos - 1]).upper()
    downstream = str(fasta[mapped_chrom][pos:end]).upper()

    return upstream, downstream


def filter_inputs(
    inputs: Dict[str, Any],
    keys: list = None
) -> Dict[str, Any]:
    """
    Filter input dictionary to include only specified keys.

    Args:
        inputs: Dictionary of input tensors/arrays
        keys: List of keys to include. Default: ['ref', 'alt', 'context_5p', 'context_3p']

    Returns:
        Filtered dictionary containing only the specified keys

    Example:
        >>> data = {'ref': [1,2], 'alt': [3,4], 'extra': [5,6]}
        >>> filtered = filter_inputs(data, keys=['ref', 'alt'])
        >>> 'extra' in filtered
        False
    """
    if keys is None:
        keys = ['ref', 'alt', 'context_5p', 'context_3p']

    return {key: inputs[key] for key in keys if key in inputs}


def detect_aggregation_mode(attn_data):
    """
    Detect the aggregation mode used in get_global_attention_weights.

    Parameters:
    -----------
    attn_data : dict
        Attention data dictionary for a single sample

    Returns:
    --------
    str or None
        - None if agg=None (no aggregation)
        - 'head' if agg='head' (head aggregation per block)
        - 'all' if agg='all' (full aggregation)
        - 'legacy' if old format (attention_matrix key)
    """
    if 'blocks' in attn_data:
        return None  # agg=None
    elif 'avg_attention_matrices' in attn_data:
        return 'head'  # agg='head'
    elif 'avg_attention_matrix' in attn_data:
        return 'all'  # agg='all'
    elif 'attention_matrix' in attn_data:
        return 'legacy'  # old format
    else:
        raise ValueError("Unknown attention data format. Expected keys: 'blocks', 'avg_attention_matrices', 'avg_attention_matrix', or 'attention_matrix'")


def create_attention_df(attention_matrices, data):
    """
    Create attention DataFrame compatible with all aggregation modes from get_global_attention_weights.

    Automatically detects the format and creates appropriate columns:
    - agg=None: columns like 'block_0_head_0', 'block_0_head_1', etc. (one per head per block)
    - agg='head': columns like 'block_0_avg', 'block_0_max', etc. (avg/max per block)
    - agg='all': columns 'avg' and 'max' (single aggregated values)
    - legacy: columns like 'head_0_weight', 'head_1_weight', etc. (backward compatibility)

    Parameters:
    -----------
    attention_matrices : dict
        Dictionary with sample IDs as keys and attention data as values.
        Format depends on aggregation mode used in get_global_attention_weights.
    data : pandas.DataFrame
        Variant data with gene annotations. Required columns:
        - 'Hugo_Symbol', 'Chromosome', 'Start_Position', 'Reference_Allele',
          'Tumor_Seq_Allele2', 'vaf', 'VARIANT_CLASS', 'HGVSp_Short',
          'Tumor_Sample_Barcode', 'type'

    Returns:
    --------
    pandas.DataFrame
        Long-format DataFrame with pairwise variants and attention weight columns
    """
    import pandas as pd

    # Detect aggregation mode from first sample
    first_sample_data = next(iter(attention_matrices.values()))
    agg_mode = detect_aggregation_mode(first_sample_data)

    # Pre-fetch all data columns we'll need
    data_dict = {
        'Hugo_Symbol': data['Hugo_Symbol'].values,
        'Chromosome': data['Chromosome'].values,
        'Start_Position': data['Start_Position'].values,
        'Reference_Allele': data['Reference_Allele'].values,
        'Tumor_Seq_Allele2': data['Tumor_Seq_Allele2'].values,
        'vaf': data['vaf'].values,
        'VARIANT_CLASS': data['VARIANT_CLASS'].values,
        'HGVSp_Short': data['HGVSp_Short'].values
    }

    tumor_type_dict = data.groupby('Tumor_Sample_Barcode')['type'].first().to_dict()

    # Initialize lists for the final dataframe
    sample_ids = []
    tumor_type = []
    v1_genes, v1_chrs, v1_pos, v1_refs, v1_alts, v1_vafs, v1_var_class, v1_prot = [], [], [], [], [], [], [], []
    v2_genes, v2_chrs, v2_pos, v2_refs, v2_alts, v2_vafs, v2_var_class, v2_prot = [], [], [], [], [], [], [], []

    # Dictionary to store attention weights
    attention_weights = {}

    # Process each sample
    for sample_id, attn in attention_matrices.items():
        variant_indices = attn['variant_indices']
        n_variants = len(variant_indices)

        # Create mesh grid of all variant pairs
        var1_idx, var2_idx = np.meshgrid(np.arange(n_variants), np.arange(n_variants))

        # Flatten the meshgrid and remove self-attention pairs
        mask = var1_idx.flatten() != var2_idx.flatten()
        var1_indices = var1_idx.flatten()[mask]
        var2_indices = var2_idx.flatten()[mask]

        # Get actual variant indices
        variant1_indices = variant_indices[var1_indices]
        variant2_indices = variant_indices[var2_indices]

        # Process attention weights based on aggregation mode
        if agg_mode is None:
            # Mode: agg=None - no aggregation, return all heads per block
            blocks_data = attn['blocks']

            # Initialize columns on first sample
            if len(attention_weights) == 0:
                for block in blocks_data:
                    block_idx = block['block_idx']
                    n_heads = len(block['heads'])
                    for head_idx in range(n_heads):
                        col_name = f'block_{block_idx}_head_{head_idx}'
                        attention_weights[col_name] = []

            # Extract weights for each head in each block
            for block in blocks_data:
                block_idx = block['block_idx']
                for head_idx, head_matrix in enumerate(block['heads']):
                    col_name = f'block_{block_idx}_head_{head_idx}'
                    weights = head_matrix[var1_indices, var2_indices]
                    attention_weights[col_name].extend(weights)

        elif agg_mode == 'head':
            # Mode: agg='head' - aggregate heads per block
            avg_matrices = attn['avg_attention_matrices']
            max_matrices = attn['max_attention_matrices']
            n_blocks = len(avg_matrices)

            # Initialize columns on first sample
            if len(attention_weights) == 0:
                for block_idx in range(n_blocks):
                    attention_weights[f'block_{block_idx}_avg'] = []
                    attention_weights[f'block_{block_idx}_max'] = []

            # Extract weights for each block
            for block_idx in range(n_blocks):
                avg_matrix = avg_matrices[block_idx]
                max_matrix = max_matrices[block_idx]

                avg_weights = avg_matrix[var1_indices, var2_indices]
                max_weights = max_matrix[var1_indices, var2_indices]

                attention_weights[f'block_{block_idx}_avg'].extend(avg_weights)
                attention_weights[f'block_{block_idx}_max'].extend(max_weights)

        elif agg_mode == 'all':
            # Mode: agg='all' - aggregate all heads and blocks
            avg_matrix = attn['avg_attention_matrix']
            max_matrix = attn['max_attention_matrix']

            # Initialize columns on first sample
            if len(attention_weights) == 0:
                attention_weights['avg'] = []
                attention_weights['max'] = []

            # Extract weights
            avg_weights = avg_matrix[var1_indices, var2_indices]
            max_weights = max_matrix[var1_indices, var2_indices]

            attention_weights['avg'].extend(avg_weights)
            attention_weights['max'].extend(max_weights)

        elif agg_mode == 'legacy':
            # Legacy mode: old format with 'attention_matrix' key
            attn_matrix = attn['attention_matrix']
            n_heads = attn_matrix.shape[0]

            # Initialize columns on first sample
            if len(attention_weights) == 0:
                for head_idx in range(n_heads):
                    attention_weights[f'head_{head_idx}_weight'] = []

            # Extract weights for each head
            for head_idx in range(n_heads):
                head_matrix = attn_matrix[head_idx]
                weights = head_matrix[var1_indices, var2_indices]
                attention_weights[f'head_{head_idx}_weight'].extend(weights)

        # Extend lists with vectorized operations
        n_pairs = len(var1_indices)
        sample_ids.extend([sample_id] * n_pairs)
        tumor_type.extend([tumor_type_dict[sample_id]] * n_pairs)

        # Variant 1 info - vectorized
        v1_genes.extend(data_dict['Hugo_Symbol'][variant1_indices])
        v1_chrs.extend(data_dict['Chromosome'][variant1_indices])
        v1_pos.extend(data_dict['Start_Position'][variant1_indices])
        v1_refs.extend(data_dict['Reference_Allele'][variant1_indices])
        v1_alts.extend(data_dict['Tumor_Seq_Allele2'][variant1_indices])
        v1_vafs.extend(data_dict['vaf'][variant1_indices])
        v1_var_class.extend(data_dict['VARIANT_CLASS'][variant1_indices])
        v1_prot.extend(data_dict['HGVSp_Short'][variant1_indices])

        # Variant 2 info - vectorized
        v2_genes.extend(data_dict['Hugo_Symbol'][variant2_indices])
        v2_chrs.extend(data_dict['Chromosome'][variant2_indices])
        v2_pos.extend(data_dict['Start_Position'][variant2_indices])
        v2_refs.extend(data_dict['Reference_Allele'][variant2_indices])
        v2_alts.extend(data_dict['Tumor_Seq_Allele2'][variant2_indices])
        v2_vafs.extend(data_dict['vaf'][variant2_indices])
        v2_var_class.extend(data_dict['VARIANT_CLASS'][variant2_indices])
        v2_prot.extend(data_dict['HGVSp_Short'][variant2_indices])

    # Create DataFrame all at once
    df_dict = {
        'sample_id': sample_ids,
        'tumor_type': tumor_type,
        'variant1_gene': v1_genes,
        'variant1_chr': v1_chrs,
        'variant1_pos': v1_pos,
        'variant1_ref': v1_refs,
        'variant1_alt': v1_alts,
        'variant1_vaf': v1_vafs,
        'variant1_var_class': v1_var_class,
        'variant1_prot': v1_prot,
        'variant2_gene': v2_genes,
        'variant2_chr': v2_chrs,
        'variant2_pos': v2_pos,
        'variant2_ref': v2_refs,
        'variant2_alt': v2_alts,
        'variant2_vaf': v2_vafs,
        'variant2_var_class': v2_var_class,
        'variant2_prot': v2_prot
    }

    # Add all attention weight columns
    df_dict.update(attention_weights)

    attention_df = pd.DataFrame(df_dict)

    return attention_df


def create_attention_edgenorm_df(attention_matrices, data):
    """
    Create attention DataFrame with edge normalization (ratio of outgoing to incoming attention).

    This function computes the ratio of outgoing attention (v1->v2) to incoming attention (v2->v1)
    for each variant pair, which can reveal directional influence patterns.

    Automatically detects the format and creates appropriate edge ratio columns:
    - agg=None: columns like 'block_0_head_0_ratio', 'block_0_head_1_ratio', etc. (one per head per block)
    - agg='head': columns like 'block_0_avg_ratio', 'block_0_max_ratio', etc. (avg/max per block)
    - agg='all': columns 'avg_ratio' and 'max_ratio' (single aggregated ratios)
    - legacy: columns like 'head_0_ratio', 'head_1_ratio', etc. (backward compatibility)

    Parameters:
    -----------
    attention_matrices : dict
        Dictionary with sample IDs as keys and attention data as values.
        Format depends on aggregation mode used in get_global_attention_weights.
    data : pandas.DataFrame
        Variant data with gene annotations. Required columns:
        - 'Hugo_Symbol', 'Chromosome', 'Start_Position', 'Reference_Allele',
          'Tumor_Seq_Allele2', 'vaf', 'VARIANT_CLASS', 'HGVSp_Short',
          'Tumor_Sample_Barcode', 'type'

    Returns:
    --------
    pandas.DataFrame
        Long-format DataFrame with pairwise variants and attention edge ratio columns.
        Edge ratio = outgoing_attention / incoming_attention.
        High ratio means v1 attends strongly to v2, but v2 doesn't attend back.

    Example:
        >>> edge_df = create_attention_edgenorm_df(attention_matrices, data)
        >>> # Auto-detects format and creates appropriate edge ratio columns
    """
    import pandas as pd

    # Detect aggregation mode from first sample
    first_sample_data = next(iter(attention_matrices.values()))
    agg_mode = detect_aggregation_mode(first_sample_data)

    # Pre-fetch all data columns we'll need
    data_dict = {
        'Hugo_Symbol': data['Hugo_Symbol'].values,
        'Chromosome': data['Chromosome'].values,
        'Start_Position': data['Start_Position'].values,
        'Reference_Allele': data['Reference_Allele'].values,
        'Tumor_Seq_Allele2': data['Tumor_Seq_Allele2'].values,
        'vaf': data['vaf'].values,
        'VARIANT_CLASS': data['VARIANT_CLASS'].values,
        'HGVSp_Short': data['HGVSp_Short'].values
    }

    tumor_type_dict = data.groupby('Tumor_Sample_Barcode')['type'].first().to_dict()

    # Initialize lists for the final dataframe
    sample_ids = []
    tumor_type = []
    v1_genes, v1_chrs, v1_pos, v1_refs, v1_alts, v1_vafs, v1_var_class, v1_prot = [], [], [], [], [], [], [], []
    v2_genes, v2_chrs, v2_pos, v2_refs, v2_alts, v2_vafs, v2_var_class, v2_prot = [], [], [], [], [], [], [], []

    # Dictionary to store edge ratios
    edge_ratios = {}
    epsilon = 1e-8

    # Process each sample
    for sample_id, attn in attention_matrices.items():
        variant_indices = attn['variant_indices']
        n_variants = len(variant_indices)

        # Create mesh grid of all variant pairs
        var1_idx, var2_idx = np.meshgrid(np.arange(n_variants), np.arange(n_variants))

        # Flatten the meshgrid and remove self-attention pairs
        mask = var1_idx.flatten() != var2_idx.flatten()
        var1_indices = var1_idx.flatten()[mask]
        var2_indices = var2_idx.flatten()[mask]

        # Get actual variant indices
        variant1_indices = variant_indices[var1_indices]
        variant2_indices = variant_indices[var2_indices]

        # Process edge ratios based on aggregation mode
        if agg_mode is None:
            # Mode: agg=None - no aggregation, compute ratios for all heads per block
            blocks_data = attn['blocks']

            # Initialize columns on first sample
            if len(edge_ratios) == 0:
                for block in blocks_data:
                    block_idx = block['block_idx']
                    n_heads = len(block['heads'])
                    for head_idx in range(n_heads):
                        col_name = f'block_{block_idx}_head_{head_idx}_ratio'
                        edge_ratios[col_name] = []

            # Calculate edge ratios for each head in each block
            for block in blocks_data:
                block_idx = block['block_idx']
                for head_idx, head_matrix in enumerate(block['heads']):
                    col_name = f'block_{block_idx}_head_{head_idx}_ratio'
                    outgoing = head_matrix[var1_indices, var2_indices]
                    incoming = head_matrix[var2_indices, var1_indices]
                    ratios = outgoing / (incoming + epsilon)
                    edge_ratios[col_name].extend(ratios)

        elif agg_mode == 'head':
            # Mode: agg='head' - aggregate heads per block, compute ratios per block
            avg_matrices = attn['avg_attention_matrices']
            max_matrices = attn['max_attention_matrices']
            n_blocks = len(avg_matrices)

            # Initialize columns on first sample
            if len(edge_ratios) == 0:
                for block_idx in range(n_blocks):
                    edge_ratios[f'block_{block_idx}_avg_ratio'] = []
                    edge_ratios[f'block_{block_idx}_max_ratio'] = []

            # Calculate edge ratios for each block
            for block_idx in range(n_blocks):
                avg_matrix = avg_matrices[block_idx]
                max_matrix = max_matrices[block_idx]

                # Average matrix ratios
                avg_outgoing = avg_matrix[var1_indices, var2_indices]
                avg_incoming = avg_matrix[var2_indices, var1_indices]
                avg_ratios = avg_outgoing / (avg_incoming + epsilon)

                # Max matrix ratios
                max_outgoing = max_matrix[var1_indices, var2_indices]
                max_incoming = max_matrix[var2_indices, var1_indices]
                max_ratios = max_outgoing / (max_incoming + epsilon)

                edge_ratios[f'block_{block_idx}_avg_ratio'].extend(avg_ratios)
                edge_ratios[f'block_{block_idx}_max_ratio'].extend(max_ratios)

        elif agg_mode == 'all':
            # Mode: agg='all' - aggregate all heads and blocks, compute single ratio
            avg_matrix = attn['avg_attention_matrix']
            max_matrix = attn['max_attention_matrix']

            # Initialize columns on first sample
            if len(edge_ratios) == 0:
                edge_ratios['avg_ratio'] = []
                edge_ratios['max_ratio'] = []

            # Calculate edge ratios
            avg_outgoing = avg_matrix[var1_indices, var2_indices]
            avg_incoming = avg_matrix[var2_indices, var1_indices]
            avg_ratios = avg_outgoing / (avg_incoming + epsilon)

            max_outgoing = max_matrix[var1_indices, var2_indices]
            max_incoming = max_matrix[var2_indices, var1_indices]
            max_ratios = max_outgoing / (max_incoming + epsilon)

            edge_ratios['avg_ratio'].extend(avg_ratios)
            edge_ratios['max_ratio'].extend(max_ratios)

        elif agg_mode == 'legacy':
            # Legacy mode: old format with 'attention_matrix' key
            attn_matrix = attn['attention_matrix']
            n_heads = attn_matrix.shape[0]

            # Initialize columns on first sample
            if len(edge_ratios) == 0:
                for head_idx in range(n_heads):
                    edge_ratios[f'head_{head_idx}_ratio'] = []

            # Calculate edge ratios for each head
            for head_idx in range(n_heads):
                head_matrix = attn_matrix[head_idx]
                outgoing = head_matrix[var1_indices, var2_indices]
                incoming = head_matrix[var2_indices, var1_indices]
                ratios = outgoing / (incoming + epsilon)
                edge_ratios[f'head_{head_idx}_ratio'].extend(ratios)

        # Extend lists with vectorized operations
        n_pairs = len(var1_indices)
        sample_ids.extend([sample_id] * n_pairs)
        tumor_type.extend([tumor_type_dict[sample_id]] * n_pairs)

        # Variant 1 info - vectorized
        v1_genes.extend(data_dict['Hugo_Symbol'][variant1_indices])
        v1_chrs.extend(data_dict['Chromosome'][variant1_indices])
        v1_pos.extend(data_dict['Start_Position'][variant1_indices])
        v1_refs.extend(data_dict['Reference_Allele'][variant1_indices])
        v1_alts.extend(data_dict['Tumor_Seq_Allele2'][variant1_indices])
        v1_vafs.extend(data_dict['vaf'][variant1_indices])
        v1_var_class.extend(data_dict['VARIANT_CLASS'][variant1_indices])
        v1_prot.extend(data_dict['HGVSp_Short'][variant1_indices])

        # Variant 2 info - vectorized
        v2_genes.extend(data_dict['Hugo_Symbol'][variant2_indices])
        v2_chrs.extend(data_dict['Chromosome'][variant2_indices])
        v2_pos.extend(data_dict['Start_Position'][variant2_indices])
        v2_refs.extend(data_dict['Reference_Allele'][variant2_indices])
        v2_alts.extend(data_dict['Tumor_Seq_Allele2'][variant2_indices])
        v2_vafs.extend(data_dict['vaf'][variant2_indices])
        v2_var_class.extend(data_dict['VARIANT_CLASS'][variant2_indices])
        v2_prot.extend(data_dict['HGVSp_Short'][variant2_indices])

    # Create DataFrame all at once
    df_dict = {
        'sample_id': sample_ids,
        'tumor_type': tumor_type,
        'variant1_gene': v1_genes,
        'variant1_chr': v1_chrs,
        'variant1_pos': v1_pos,
        'variant1_ref': v1_refs,
        'variant1_alt': v1_alts,
        'variant1_vaf': v1_vafs,
        'variant1_var_class': v1_var_class,
        'variant1_prot': v1_prot,
        'variant2_gene': v2_genes,
        'variant2_chr': v2_chrs,
        'variant2_pos': v2_pos,
        'variant2_ref': v2_refs,
        'variant2_alt': v2_alts,
        'variant2_vaf': v2_vafs,
        'variant2_var_class': v2_var_class,
        'variant2_prot': v2_prot
    }

    # Add all edge ratio columns
    df_dict.update(edge_ratios)

    attention_edgenorm_df = pd.DataFrame(df_dict)

    return attention_edgenorm_df


def encode_cna_segment(chromosome, start, end, chr_encoder, chr_sizes=None):
    """
    Encode CNA segment information for model input with chromosome-specific normalization.

    Args:
        chromosome (str or int): Chromosome identifier (e.g., '17', 17, 'X', 'Y', 'MT')
        start (int): Segment start position (genomic coordinate)
        end (int): Segment end position (genomic coordinate)
        chr_encoder (dict): Chromosome encoder dictionary from model instance
                           Expected keys: 'chr_to_int', 'int_to_chr', 'vocab_size'
        chr_sizes (dict, optional): Dictionary mapping chromosome names to sizes in bp
                                    If provided, uses chromosome-specific normalization
                                    If None, uses global normalization (250M, backward compatible)

    Returns:
        tuple: (chr_encoded, start_norm, end_norm, length_log)
            - chr_encoded (int): Encoded chromosome ID
            - start_norm (float): Normalized start position [0, 1]
            - end_norm (float): Normalized end position [0, 1]
            - length_log (float): Log10 of segment length

    Example:
        >>> chr_encoder = {'chr_to_int': {'17': 17, 'X': 23}, 'int_to_chr': {17: '17', 23: 'X'}}
        >>> chr_sizes = {'chr17': 83257441, 'chrX': 156040895}
        >>> chr, start_n, end_n, len_log = encode_cna_segment('17', 7000000, 8000000, chr_encoder, chr_sizes)
        >>> print(f"Chr: {chr}, Start: {start_n:.4f}, End: {end_n:.4f}, Length: {len_log:.2f}")
        Chr: 17, Start: 0.0841, End: 0.0961, Length: 6.00

    Note:
        - Uses chromosome-specific normalization for consistency with mutation encoding
        - Position 0.5 always means "middle of chromosome" regardless of chromosome size
        - Length is log-transformed to handle wide range of segment sizes (100bp to 100Mbp)
        - Add 1 to length before log to avoid log(0) for very small segments
        - Returns 0 for unknown chromosomes (padding value)
        - Falls back to global normalization (250M) if chr_sizes not provided (backward compatible)
    """
    # Encode chromosome to integer using dictionary lookup
    chr_encoded = chr_encoder['chr_to_int'].get(str(chromosome), 0)

    # Normalize positions to [0, 1] range
    if chr_sizes is not None:
        # Chromosome-specific normalization (consistent with mutation encoding)
        # Convert chromosome name to match chr_sizes format (add 'chr' prefix if needed)
        if isinstance(chromosome, str) and not chromosome.startswith('chr'):
            chr_key = f'chr{chromosome}'
        else:
            chr_key = str(chromosome)

        # Get chromosome size
        if chr_key in chr_sizes:
            chr_size = chr_sizes[chr_key]
            start_norm = min(1.0, max(0.0, start / chr_size))
            end_norm = min(1.0, max(0.0, end / chr_size))
        else:
            # Unknown chromosome - use default normalization
            start_norm = 0.5
            end_norm = 0.5
    else:
        # Global normalization (backward compatible with old behavior)
        max_position = 250_000_000
        start_norm = start / max_position
        end_norm = end / max_position

    # Compute log-transformed length
    length = end - start
    length_log = np.log10(length + 1)  # Add 1 to avoid log(0)

    return chr_encoded, start_norm, end_norm, length_log


# ----------------------------------------------------------------------------
# CNA quantile normalisation against the TCGA training distribution.
#
# Panel-sequencing or cell-line CNA segments cover a different copy-number
# distribution than the whole-exome TCGA cohort the model was pretrained on.
# Mapping a user's segment-mean values onto TCGA-distribution quantiles
# reconciles the shift before inference.
# ----------------------------------------------------------------------------


def _load_tcga_segment_mean(data_paths: Sequence[Union[str, Path]]) -> np.ndarray:
    """Concatenate the ``Segment_Mean`` column from one or more TCGA CNA CSVs."""
    vals = []
    for p in data_paths:
        df = pd.read_csv(p, usecols=["Segment_Mean"])
        vals.append(df["Segment_Mean"].astype(float).values)
    return np.concatenate(vals)


def get_tcga_cna_stats(stats_path: Union[str, Path],
                       data_paths: Sequence[Union[str, Path]]) -> Tuple[float, float]:
    """Return (mean, std) of the TCGA CNA Segment_Mean reference distribution.

    Caches the result to ``stats_path`` (JSON) so subsequent calls skip the
    full re-read of the underlying CSVs.
    """
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            s = json.load(f)
        return float(s["mean"]), float(s["std"])
    print(f"Computing TCGA CNA stats (cache miss: {stats_path})", flush=True)
    arr = _load_tcga_segment_mean(data_paths)
    mean, std = float(arr.mean()), float(arr.std())
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump({"mean": mean, "std": std, "n_segments": int(arr.size)}, f, indent=2)
    print(f"  Cached to {stats_path}: mean={mean:.4f} std={std:.4f} (n={arr.size:,})", flush=True)
    return mean, std


def get_tcga_cna_sorted(sorted_path: Union[str, Path],
                        data_paths: Sequence[Union[str, Path]]) -> np.ndarray:
    """Return a sorted float32 array of TCGA Segment_Mean values.

    The sorted array is the lookup table used by :func:`quantile_normalize_to_tcga`.
    Cached to ``sorted_path`` (npy) on first build; subsequent calls just load.
    """
    if os.path.exists(sorted_path):
        return np.load(sorted_path)
    print(f"Computing sorted TCGA CNA array (cache miss: {sorted_path})", flush=True)
    arr = np.sort(_load_tcga_segment_mean(data_paths)).astype(np.float32)
    os.makedirs(os.path.dirname(sorted_path), exist_ok=True)
    np.save(sorted_path, arr)
    print(f"  Cached to {sorted_path}: n={arr.size:,}", flush=True)
    return arr


def quantile_normalize_to_tcga(vals: np.ndarray, tcga_sorted: np.ndarray) -> np.ndarray:
    """Map ``vals`` onto the empirical quantiles of the TCGA reference.

    For each input value, finds its rank-based quantile, then looks up the
    corresponding value in the sorted TCGA reference. This collapses any
    panel-sequencing / cell-line distribution shift onto the TCGA
    pretraining distribution. Use with the array returned by
    :func:`get_tcga_cna_sorted`.

    Args:
        vals: 1-D array of user-provided Segment_Mean values.
        tcga_sorted: 1-D sorted array of TCGA reference Segment_Means.

    Returns:
        1-D float array, same length as ``vals``, whose values lie in the
        TCGA reference distribution.
    """
    n = len(vals)
    ranks = rankdata(vals, method="average")
    q = (ranks - 0.5) / n
    tcga_q = np.linspace(0.0, 1.0, len(tcga_sorted))
    return np.interp(q, tcga_q, tcga_sorted)


#: Locus columns that uniquely identify a single-base substitution. Used to derive
#: cohort recurrence when the caller has not supplied an explicit ``mut_id`` column.
SNV_LOCUS_COLS = ["Chromosome", "Start_Position", "Reference_Allele", "Tumor_Seq_Allele2"]


def subsample_snv(
    snv_df: pd.DataFrame,
    max_variants: int = 1000,
    min_recurrence: int = 5,
    sample_col: str = "Tumor_Sample_Barcode",
    random_state: Optional[int] = 42,
) -> pd.DataFrame:
    """Cap variants per sample, preserving variants recurrent across the cohort.

    TESSERA featurizes each sample as a *bag* of its alterations, padded to the
    cohort's largest sample (``fixed_bag_size``), so attention cost and peak memory
    scale with the maximum per-sample count -- O(bag^2). Capping each sample keeps
    featurization tractable on large cohorts or hypermutators with a negligible
    effect on the pooled embedding. This is the exact SNV subsampler used to build
    the TESSERA pretraining set (``scripts/data/tcga``).

    For each sample, every variant whose locus recurs in at least ``min_recurrence``
    samples cohort-wide is kept (a low-cost proxy for recurrent / driver
    alterations); the remaining budget up to ``max_variants`` is filled with a
    uniform random draw from that sample's other variants. Samples with at most
    ``max_variants`` variants are returned unchanged.

    The cap is **soft**: recurrent variants are never dropped, so a sample whose
    recurrent variants alone exceed ``max_variants`` keeps all of them and its
    returned count exceeds ``max_variants``. If you need a hard ceiling to bound
    peak memory, check per-sample counts on the result and raise ``min_recurrence``
    until the recurrent set fits.

    Args:
        snv_df: One row per variant, using :func:`tessera.featurize` column names:
            ``sample_col`` plus either an explicit ``mut_id`` column or the locus
            columns ``Chromosome``, ``Start_Position``, ``Reference_Allele``,
            ``Tumor_Seq_Allele2`` (from which recurrence is derived). Extra columns
            (``vaf``, annotations, ...) are carried through unchanged.
        max_variants: Maximum variants kept per sample (TESSERA pretraining value);
            a soft cap -- see above.
        min_recurrence: Minimum number of records a locus must appear in to be
            preserved as recurrent (counted by record, matching the pretraining
            pipeline; for one-row-per-variant-per-sample tables this equals the
            number of samples). Recurrence is relative to the cohort in ``snv_df``;
            with a single sample nothing is recurrent and the cap is a pure random
            draw. Set above the cohort size to disable preservation. Must be > 0.
        sample_col: Column identifying the sample.
        random_state: Seed for the within-sample random draw (reproducible default).

    Returns:
        Subsampled copy, reset-indexed, with the same columns as ``snv_df`` (a
        ``mut_id`` derived internally is not added to the output).

    Note:
        If ``snv_df`` already has a ``mut_id`` column it is used verbatim as the
        recurrence key and the locus columns are ignored; otherwise ``mut_id`` is
        derived from the locus columns. ``Hugo_Symbol`` is intentionally excluded
        (not part of the ``featurize`` schema and redundant with locus), so
        recurrence is locus-based.

    Example:
        >>> import tessera                                                       # doctest: +SKIP
        >>> small = tessera.data.preprocessing.subsample_snv(snv_df, max_variants=1000)  # doctest: +SKIP
        >>> result = tessera.featurize(snv_df=small, cna_df=cna_df)              # doctest: +SKIP
    """
    if sample_col not in snv_df.columns:
        raise KeyError(f"snv_df is missing the sample column {sample_col!r}")
    if max_variants <= 0:
        raise ValueError(f"max_variants must be positive, got {max_variants}")
    if min_recurrence <= 0:
        raise ValueError(f"min_recurrence must be positive, got {min_recurrence}")
    if len(snv_df) == 0:
        return snv_df.reset_index(drop=True)

    # Recurrence key: an explicit ``mut_id`` if present, else a locus key built
    # vectorized as a Series -- never added to the frame, so the caller's columns
    # and memory footprint are untouched.
    if "mut_id" in snv_df.columns:
        key = snv_df["mut_id"].astype(str)
    else:
        missing = [c for c in SNV_LOCUS_COLS if c not in snv_df.columns]
        if missing:
            raise KeyError(
                "subsample_snv needs either a 'mut_id' column or the locus columns "
                f"{SNV_LOCUS_COLS}; missing {missing}."
            )
        key = (snv_df["Chromosome"].astype(str) + ":"
               + snv_df["Start_Position"].astype(str) + ":"
               + snv_df["Reference_Allele"].astype(str) + ":"
               + snv_df["Tumor_Seq_Allele2"].astype(str))

    counts = key.value_counts()
    common = set(counts[counts >= min_recurrence].index)
    is_common = key.isin(common).to_numpy()

    # Select kept rows per sample by positional index, then slice the original
    # frame. This keeps every column (incl. ``sample_col``) and does not depend on
    # pandas' groupby.apply grouping-column inclusion, whose default changes across
    # pandas versions.
    kept = []
    for _, pos in snv_df.groupby(sample_col, sort=False).indices.items():
        common_pos = pos[is_common[pos]]
        slots = max_variants - len(common_pos)
        if slots <= 0:
            kept.append(common_pos)
            continue
        rest_pos = pos[~is_common[pos]]
        if len(rest_pos) == 0:
            kept.append(common_pos)
            continue
        n = min(len(rest_pos), slots)
        sampled = pd.Series(rest_pos).sample(
            n=n, replace=False, random_state=random_state).to_numpy()
        kept.append(np.concatenate([common_pos, sampled]))

    keep_pos = np.concatenate(kept) if kept else np.empty(0, dtype=int)
    return snv_df.iloc[keep_pos].reset_index(drop=True)


def subsample_cna(
    cna_df: pd.DataFrame,
    max_segments: int = 1000,
    loh_weight: float = 0.5,
    sample_col: str = "Tumor_Sample_Barcode",
) -> pd.DataFrame:
    """Cap segments per sample by copy-number alteration magnitude.

    The CNA counterpart to :func:`subsample_snv` (same bag-size / memory rationale).
    Each segment is scored by ``|Segment_Mean| + loh_weight * LOH`` and the top
    ``max_segments`` per sample are kept; samples with at most ``max_segments``
    segments are returned unchanged. This is the exact CNA subsampler used to build
    the TESSERA pretraining set. The ``LOH`` term is optional: when the column is
    absent the score is simply ``|Segment_Mean|`` (appropriate for the no-LoH model
    variant or copy-number calls without an LoH flag).

    Args:
        cna_df: One row per segment, using :func:`tessera.featurize` column names:
            ``sample_col``, ``Segment_Mean`` (log2 ratio), optionally ``LOH`` (0/1).
            Extra columns (``Chromosome``, ``Start``, ``End``, ...) are carried through.
        max_segments: Maximum segments kept per sample (TESSERA pretraining value).
        loh_weight: Bonus added to ``|Segment_Mean|`` for LoH segments. Ignored when
            there is no ``LOH`` column.
        sample_col: Column identifying the sample.

    Returns:
        Subsampled copy, reset-indexed, with the same columns as ``cna_df``.

    Note:
        ``Segment_Mean`` must be finite. Non-finite values are not sanitized:
        ``+/-inf`` ranks as the most-altered (always kept) and ``NaN`` sinks to the
        bottom (dropped first when a sample is over the cap, passed through
        otherwise). Clip or drop non-finite segments before calling.

    Example:
        >>> import tessera                                                       # doctest: +SKIP
        >>> small = tessera.data.preprocessing.subsample_cna(cna_df, max_segments=1000)  # doctest: +SKIP
        >>> result = tessera.featurize(snv_df=snv_df, cna_df=small)              # doctest: +SKIP
    """
    if sample_col not in cna_df.columns:
        raise KeyError(f"cna_df is missing the sample column {sample_col!r}")
    if "Segment_Mean" not in cna_df.columns:
        raise KeyError("cna_df is missing the required 'Segment_Mean' column")
    if max_segments <= 0:
        raise ValueError(f"max_segments must be positive, got {max_segments}")
    if len(cna_df) == 0:
        return cna_df.reset_index(drop=True)

    # Score each segment without writing a column to the frame (so a caller column
    # named like the internal score is never shadowed or dropped). LOH is optional.
    score = cna_df["Segment_Mean"].abs().to_numpy()
    if "LOH" in cna_df.columns:
        score = score + cna_df["LOH"].astype(float).to_numpy() * loh_weight

    # Keep the top ``max_segments`` per sample by score, selected by positional
    # index (independent of groupby.apply grouping-column semantics).
    kept = []
    for _, pos in cna_df.groupby(sample_col, sort=False).indices.items():
        if len(pos) <= max_segments:
            kept.append(pos)
            continue
        top = pd.Series(score[pos], index=pos).nlargest(max_segments).index.to_numpy()
        kept.append(top)

    keep_pos = np.concatenate(kept) if kept else np.empty(0, dtype=int)
    return cna_df.iloc[keep_pos].reset_index(drop=True)
