"""Shared base class for TESSERA-family models.

Defines BaseModel, which provides dataset construction, FASTA-context lookup,
chromosome encoding, position normalisation, and per-sample bagging utilities
inherited by TESSERA and any future model variants in this package. Subclasses
add the model architecture and task-specific training logic on top of this
shared input-pipeline infrastructure.
"""

import numpy as np
import os
import time
from tqdm import tqdm
from pyfaidx import Fasta
import tensorflow as tf
import tessera.data.preprocessing
import shutil
from sklearn.preprocessing import LabelEncoder
from keras.config import enable_unsafe_deserialization

class BaseModel(object):
    def __init__(self,name='TESSERA',genome_version='GRCh37',use_distributed=False,model_dir=None,jit_compile=True,mixed_precision=False):

        self.name = name
        if model_dir is None:
            self.model_dir = os.path.join('models', self.name)
        else:
            self.model_dir = model_dir

        os.makedirs(self.model_dir, exist_ok=True)
        self.use_distributed = use_distributed
        self.jit_compile = jit_compile
        self.mixed_precision = mixed_precision
        self.genome_version = genome_version

        # Set mixed precision policy if enabled
        if self.mixed_precision:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("Mixed precision enabled: using float16 for most operations")

        self.token_map = {
            '[PAD]': 0,
            'A': 1, 'C': 2, 'G': 3, 'T': 4,'-':5,
            'X': 6
        }
        self.reverse_token_map = {v: k for k, v in self.token_map.items()}

        self.model = None

        if genome_version == 'GRCh37':
            self.fasta_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ref_genomes', 'GCF_000001405.25_GRCh37.p13_genomic.fna')
        elif genome_version == 'GRCh38':
            self.fasta_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ref_genomes', 'GCF_000001405.40_GRCh38.p14_genomic.fna')

        # Load chromosome sizes for position normalization
        self.chr_sizes = self._load_chromosome_sizes()

        # Create chromosome encoder (string -> integer mapping)
        self.chr_encoder = self._create_chromosome_encoder()

        if self.use_distributed:
            self.strategy = tf.distribute.MirroredStrategy()
        else:
            self.strategy = tf.distribute.get_strategy()

        self.use_sample_exp = False
        self.use_vaf = False
        self.mil = False

        # Multi-modal flags - auto-detected from provided data
        self.use_mut = False  # Set to True when mutation data is provided
        self.use_cna = False  # Set to True when CNA data is provided

        # Optional CNA feature flags
        self.use_cna_loh = False

        enable_unsafe_deserialization()

    def _load_chromosome_sizes(self):
        """Load chromosome sizes based on genome version"""
        if self.genome_version == 'GRCh37':
            chr_sizes_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ref_genomes', 'GRCh37_chr_sizes.txt')
        elif self.genome_version == 'GRCh38':
            chr_sizes_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ref_genomes', 'GRCh38_chr_sizes.txt')
        else:
            raise ValueError(f"Unsupported genome version: {self.genome_version}")

        chr_sizes = {}
        with open(chr_sizes_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split('\t')
                    if len(parts) == 2:
                        chr_name, size = parts
                        chr_sizes[chr_name] = int(size)
        return chr_sizes

    def _create_chromosome_encoder(self):
        """Create chromosome encoder mapping chromosome names to integers"""
        # Standard human chromosomes (autosomes + sex chromosomes)
        chromosomes = [str(i) for i in range(1, 23)] + ['X', 'Y', 'MT']

        # Create mapping: chromosome name -> integer ID
        chr_to_int = {}
        for i, chrom in enumerate(chromosomes, 1):  # Start from 1, 0 reserved for padding
            chr_to_int[chrom] = i

        # Reverse mapping: integer ID -> chromosome name
        int_to_chr = {v: k for k, v in chr_to_int.items()}
        int_to_chr[0] = 'PAD'  # Padding value

        return {
            'chr_to_int': chr_to_int,
            'int_to_chr': int_to_chr,
            'vocab_size': len(chromosomes) + 1  # +1 for padding
        }

    def _normalize_position(self, chrom, pos):
        """Normalize genomic position to [0, 1] range based on chromosome size"""
        # Convert chromosome name to match chr_sizes format (add 'chr' prefix if needed)
        if isinstance(chrom, str) and not chrom.startswith('chr'):
            chrom_key = f'chr{chrom}'
        else:
            chrom_key = chrom

        if chrom_key in self.chr_sizes:
            return min(1.0, max(0.0, pos / self.chr_sizes[chrom_key]))
        else:
            # For unknown chromosomes, return 0.5 as default
            return 0.5

    def _reset_models(self):
        self.models_dir = os.path.join(self.Name, 'models')
        if os.path.exists(self.models_dir):
            shutil.rmtree(self.models_dir)
        os.makedirs(self.models_dir)

    @staticmethod
    def _pad_variants(batch_list, target_size, pad_val=0, item_shape=None):
        """
        Pad variant lists to target size.

        Args:
            batch_list: List of variant lists to pad
            target_size: Target size for padding
            pad_val: Value to use for padding
            item_shape: Expected shape of each item (used when batch_list contains empty lists)
                       Use () for scalar items (creates 2D output)

        Returns:
            Padded numpy array
        """
        padded = []
        for variants in batch_list:
            count = len(variants)
            if count < target_size:
                # shape reference from the first item if it exists
                if count > 0:
                    shape_item = np.array(variants[0]).shape
                elif item_shape is not None:
                    # Use provided shape
                    shape_item = item_shape
                else:
                    # if no variants at all and no shape provided, fallback
                    shape_item = (1,)

                # build the padding - handle scalars specially
                if shape_item == ():
                    # For scalars, use plain Python values instead of 0-d arrays
                    pad_list = [pad_val] * (target_size - count)
                else:
                    pad_list = [np.full(shape_item, pad_val) for _ in range(target_size - count)]
                new_variants = variants + pad_list
            else:
                new_variants = variants[:target_size]  # Truncate if too long
            padded.append(new_variants)

        # Convert to numpy array more carefully
        try:
            return np.array(padded)
        except ValueError:
            # If we still have shape issues, handle manually
            # This can happen with idx arrays of different types
            max_len = max(len(row) for row in padded)
            result = np.full((len(padded), max_len) + np.array(padded[0][0]).shape, pad_val)
            for i, row in enumerate(padded):
                for j, item in enumerate(row):
                    if j < max_len:
                        result[i, j] = item
            return result

    def _precompute_variant_encodings(self, chromosomes, positions, refs, alts,
                                       context_len, ref_len, alt_len, vaf,
                                       fasta, chromosome_mapper):
        """
        Precompute all variant encodings once before training starts.

        Iterates all variants, performs FASTA lookups and sequence encoding,
        and stores results in pre-allocated numpy arrays. Uses a local context
        cache to deduplicate same-(chrom, pos) FASTA reads.

        Returns:
            dict of numpy arrays keyed by 'ref', 'alt', 'context_5p', 'context_3p',
            'chr', 'pos', and optionally 'vaf'.
        """
        n_variants = len(chromosomes)
        print(f"Precomputing variant encodings for {n_variants:,} variants...")
        t0 = time.time()

        # Pre-allocate arrays
        ref_arr = np.zeros((n_variants, ref_len), dtype=np.int32)
        alt_arr = np.zeros((n_variants, alt_len), dtype=np.int32)
        c5p_arr = np.zeros((n_variants, context_len), dtype=np.int32)
        c3p_arr = np.zeros((n_variants, context_len), dtype=np.int32)
        chr_arr = np.zeros((n_variants, 1), dtype=np.int32)
        pos_arr = np.zeros((n_variants, 1), dtype=np.float32)
        vaf_arr = None
        if vaf is not None:
            vaf_arr = np.zeros((n_variants, 1), dtype=np.float32)

        # Local context cache — discarded after precomputation
        context_cache = {}

        for i in tqdm(range(n_variants), desc="  Precomputing variants", unit="var"):
            chrom = chromosomes[i]
            pos = positions[i]

            # Context lookup with deduplication
            cache_key = (chrom, pos)
            if cache_key in context_cache:
                context_5p, context_3p = context_cache[cache_key]
            else:
                context_5p, context_3p = tessera.data.preprocessing.get_sequence_context(
                    fasta, chrom, pos, context_len, chromosome_mapper
                )
                context_cache[cache_key] = (context_5p, context_3p)

            # Encode sequences
            ref_enc = tessera.data.preprocessing.encode_sequence(refs[i], self.token_map)
            alt_enc = tessera.data.preprocessing.encode_sequence(alts[i], self.token_map)
            c5p_enc = tessera.data.preprocessing.encode_sequence(context_5p, self.token_map)
            c3p_enc = tessera.data.preprocessing.encode_sequence(context_3p, self.token_map)

            # Pad/truncate to fixed length and store
            ref_arr[i, :min(len(ref_enc), ref_len)] = ref_enc[:ref_len]
            alt_arr[i, :min(len(alt_enc), alt_len)] = alt_enc[:alt_len]
            c5p_arr[i, :min(len(c5p_enc), context_len)] = c5p_enc[:context_len]
            c3p_arr[i, :min(len(c3p_enc), context_len)] = c3p_enc[:context_len]

            # Chromosome and position
            chr_arr[i, 0] = self.chr_encoder['chr_to_int'].get(chrom, 0)
            pos_arr[i, 0] = self._normalize_position(chrom, pos)

            if vaf is not None:
                vaf_arr[i, 0] = vaf[i]

        elapsed = time.time() - t0
        print(f"  Precompute complete: {n_variants:,} variants in {elapsed:.1f}s, "
              f"cache hit rate {100 * (1 - len(context_cache) / max(1, n_variants)):.1f}%")

        result = {
            'ref': ref_arr,
            'alt': alt_arr,
            'context_5p': c5p_arr,
            'context_3p': c3p_arr,
            'chr': chr_arr,
            'pos': pos_arr,
        }
        if vaf_arr is not None:
            result['vaf'] = vaf_arr
        return result

    def _precompute_cna_encodings(self, cna_chromosomes, cna_starts, cna_ends,
                                   cna_segment_means, cna_lohs):
        """
        Precompute all CNA segment encodings once before training starts.

        Returns:
            dict of numpy arrays keyed by 'chr', 'start', 'end', 'length',
            'segment_mean', and optionally 'loh'.
        """
        n_segments = len(cna_chromosomes)
        print(f"Precomputing CNA encodings for {n_segments:,} segments...")
        t0 = time.time()

        cna_chr_arr = np.zeros((n_segments, 1), dtype=np.int32)
        cna_start_arr = np.zeros((n_segments, 1), dtype=np.float32)
        cna_end_arr = np.zeros((n_segments, 1), dtype=np.float32)
        cna_length_arr = np.zeros((n_segments, 1), dtype=np.float32)
        cna_segment_mean_arr = np.zeros((n_segments, 1), dtype=np.float32)
        cna_loh_arr = None
        if cna_lohs is not None:
            cna_loh_arr = np.zeros((n_segments, 1), dtype=np.float32)

        for i in tqdm(range(n_segments), desc="  Precomputing CNA", unit="seg"):
            chrom = str(cna_chromosomes[i])
            start = cna_starts[i]
            end = cna_ends[i]

            chr_enc, start_norm, end_norm, len_log = tessera.data.preprocessing.encode_cna_segment(
                chrom, start, end, self.chr_encoder, self.chr_sizes
            )

            cna_chr_arr[i, 0] = chr_enc
            cna_start_arr[i, 0] = start_norm
            cna_end_arr[i, 0] = end_norm
            cna_length_arr[i, 0] = len_log
            cna_segment_mean_arr[i, 0] = cna_segment_means[i]

            if cna_lohs is not None:
                cna_loh_arr[i, 0] = float(cna_lohs[i])

        elapsed = time.time() - t0
        print(f"  CNA precompute complete: {n_segments:,} segments in {elapsed:.1f}s")

        result = {
            'chr': cna_chr_arr,
            'start': cna_start_arr,
            'end': cna_end_arr,
            'length': cna_length_arr,
            'segment_mean': cna_segment_mean_arr,
        }
        if cna_loh_arr is not None:
            result['loh'] = cna_loh_arr
        return result

    def _prepare_dataset_inputs(self, sample_ids, chromosomes, positions, refs, alts,
                                context_len, vaf, Y, sample_exp, ref_len, alt_len,
                                subsample, fixed_bag_size, set_class_weights,
                                use_cna=False, cna_sample_dict=None):
        """
        Prepare inputs and compute derived values for dataset creation.

        Returns:
            dict: Dictionary containing prepared inputs and metadata
        """
        # Handle mutation data (may be None for CNA-only datasets)
        if sample_ids is not None and len(sample_ids) > 0:
            # Determine ref_len, alt_len if not specified
            if ref_len is None:
                ref_len = max(np.vectorize(len)(refs))
            if alt_len is None:
                alt_len = max(np.vectorize(len)(alts))
        else:
            # CNA-only: set default values
            ref_len = ref_len or 1
            alt_len = alt_len or 1
            sample_ids = np.array([])
            chromosomes = np.array([])
            positions = np.array([])
            refs = np.array([])
            alts = np.array([])

        self.ref_len = ref_len
        self.alt_len = alt_len
        self.context_len = context_len or 0
        self.sample_ids = sample_ids
        self.subsample = subsample

        # Set flags based on optional inputs
        if vaf is not None:
            self.use_vaf = True
        if sample_exp is not None:
            self.use_sample_exp = True
            self.n_exp = sample_exp.shape[1]

        # Handle label encoding and class weights
        Y_encoded, class_counts, class_weights = self._compute_class_weights(
            sample_ids, Y, set_class_weights
        )

        # Prepare sample-to-label mapping
        sample_to_label = {}
        labels = Y_encoded if Y_encoded is not None else [None] * len(sample_ids)
        for s_id, lab in zip(sample_ids, labels):
            sample_to_label[s_id] = lab

        # Group variants by sample
        sample_dict = {}
        for i, s_id in enumerate(sample_ids):
            if s_id not in sample_dict:
                sample_dict[s_id] = []
            sample_dict[s_id].append(i)

        # If using CNA, include samples that only have CNA data (no mutations)
        if use_cna and cna_sample_dict is not None:
            cna_only_samples = set(cna_sample_dict.keys()) - set(sample_dict.keys())
            # Add CNA-only samples to sample_dict with empty variant lists
            for s_id in cna_only_samples:
                sample_dict[s_id] = []
            print(f"   Including {len(cna_only_samples)} samples with CNA data only (no mutations)")

        unique_samples = list(sample_dict.keys())

        # Compute global bag size if using fixed bag size
        global_bag_size = None
        global_cna_bag_size = None

        if fixed_bag_size:
            if subsample is not None:
                global_bag_size = subsample
            else:
                variant_counts = [len(v) for v in sample_dict.values()]
                global_bag_size = max(variant_counts) if variant_counts else 1

            # Compute global CNA bag size
            if use_cna and cna_sample_dict is not None:
                cna_counts = [len(v) for v in cna_sample_dict.values()]
                global_cna_bag_size = max(cna_counts) if cna_counts else 1
                print(f"   Global CNA bag size: {global_cna_bag_size}")

        return {
            'ref_len': ref_len,
            'alt_len': alt_len,
            'context_len': self.context_len,  # Return corrected context_len
            'Y_encoded': Y_encoded,
            'sample_to_label': sample_to_label,
            'sample_dict': sample_dict,
            'unique_samples': unique_samples,
            'global_bag_size': global_bag_size,
            'global_cna_bag_size': global_cna_bag_size,
        }

    def _compute_class_weights(self, sample_ids, Y, set_class_weights):
        """
        Compute class weights based on sample distribution.

        Returns:
            tuple: (Y_encoded, class_counts, class_weights)
        """
        if Y is None:
            return None, None, None

        self.label_encoder = LabelEncoder()
        Y_encoded = self.label_encoder.fit_transform(Y)

        # Count samples per class (not just raw instances)
        counted_samples = {}
        unique_classes = np.unique(Y_encoded)
        sample_class_counts = {cls: 0 for cls in unique_classes}

        for s_id, lab in zip(sample_ids, Y_encoded):
            if s_id not in counted_samples:
                sample_class_counts[lab] += 1
                counted_samples[s_id] = True

        class_counts = sample_class_counts

        # Compute class weights (inverse frequency)
        total_unique_samples = len(counted_samples)
        n_classes = len(unique_classes)
        class_weights = {
            cls: total_unique_samples / (n_classes * count)
            for cls, count in class_counts.items()
        }

        # Log class distribution
        print(f"Sample class distribution: {class_counts}")
        print(f"Computed class weights: {class_weights}")

        if set_class_weights:
            self.class_counts = class_counts
            self.class_weights = class_weights

        return Y_encoded, class_counts, class_weights

    def _create_batch_processor(self, chromosomes, positions, refs, alts, context_len,
                                vaf, sample_exp, ref_len, alt_len, subsample,
                                fixed_bag_size, global_bag_size, Y_encoded,
                                sample_to_label, sample_dict, use_context_cache,
                                fasta, chromosome_mapper, context_cache,
                                use_mut=False, use_cna=False, cna_sample_dict=None,
                                cna_chromosomes=None, cna_starts=None,
                                cna_ends=None, cna_segment_means=None,
                                cna_lohs=None,
                                cna_subsample=None, global_cna_bag_size=None,
                                precomputed_variants=None, precomputed_cna=None):
        """
        Create the batch processing function with optional CNA data support.

        Args:
            use_mut: Whether mutation data is present
            use_cna: Whether CNA data is present
            precomputed_variants: dict of numpy arrays from _precompute_variant_encodings(), or None
            precomputed_cna: dict of numpy arrays from _precompute_cna_encodings(), or None

        Returns:
            callable: Function that processes a batch of samples
        """
        def process_batch(batch_samples):
            """Build a single batch from a subset of samples -> dictionary of Tensors."""
            label_batch, sample_ids_batch = [], []

            # Initialize mutation batch lists only if using mutations
            if use_mut:
                ref_batch, alt_batch = [], []
                context_5p_batch, context_3p_batch = [], []
                idx_batch = []
                sample_exp_batch, vaf_batch = [], []
                chr_batch, pos_batch = [], []

            # CNA batch lists
            cna_chr_batch, cna_start_batch, cna_end_batch = [], [], []
            cna_length_batch, cna_segment_mean_batch, cna_idx_batch = [], [], []
            cna_breakpoint_density_batch, cna_delta_cn_batch, cna_loh_batch = [], [], []

            for s_id in batch_samples:
                # Process mutations if present
                if use_mut:
                    variant_indices = sample_dict[s_id]

                    # Select variants (with optional subsampling)
                    if subsample is not None:
                        if len(variant_indices) > subsample:
                            chosen_indices = np.random.choice(variant_indices, subsample, replace=False)
                        else:
                            chosen_indices = variant_indices
                    else:
                        chosen_indices = variant_indices

                    if precomputed_variants is not None:
                        # FAST PATH: numpy fancy indexing (~microseconds)
                        chosen_arr = np.asarray(chosen_indices, dtype=np.intp)
                        local_ref = list(precomputed_variants['ref'][chosen_arr])
                        local_alt = list(precomputed_variants['alt'][chosen_arr])
                        local_c5p = list(precomputed_variants['context_5p'][chosen_arr])
                        local_c3p = list(precomputed_variants['context_3p'][chosen_arr])
                        local_chr = list(precomputed_variants['chr'][chosen_arr])
                        local_pos = list(precomputed_variants['pos'][chosen_arr])
                        if vaf is not None:
                            local_vaf = list(precomputed_variants['vaf'][chosen_arr])
                        else:
                            local_vaf = []
                    else:
                        # ORIGINAL PATH: per-variant FASTA I/O (completely unchanged)
                        local_ref, local_alt = [], []
                        local_c5p, local_c3p = [], []
                        local_vaf = []
                        local_chr, local_pos = [], []

                        for idx in chosen_indices:
                            chrom = chromosomes[idx]
                            pos = positions[idx]
                            ref_seq, alt_seq = refs[idx], alts[idx]

                            # Get sequence context with optional caching
                            if use_context_cache:
                                cache_key = (chrom, pos)
                                if cache_key in context_cache:
                                    context_5p, context_3p = context_cache[cache_key]
                                else:
                                    context_5p, context_3p = tessera.data.preprocessing.get_sequence_context(
                                        fasta, chrom, pos, context_len, chromosome_mapper
                                    )
                                    context_cache[cache_key] = (context_5p, context_3p)
                            else:
                                context_5p, context_3p = tessera.data.preprocessing.get_sequence_context(
                                    fasta, chrom, pos, context_len, chromosome_mapper
                                )

                            # Encode sequences
                            ref_enc = np.array(tessera.data.preprocessing.encode_sequence(ref_seq, self.token_map))
                            alt_enc = np.array(tessera.data.preprocessing.encode_sequence(alt_seq, self.token_map))
                            c5p_enc = np.array(tessera.data.preprocessing.encode_sequence(context_5p, self.token_map))
                            c3p_enc = np.array(tessera.data.preprocessing.encode_sequence(context_3p, self.token_map))

                            # Pad sequences to fixed length
                            ref_enc = np.pad(ref_enc, (0, ref_len - len(ref_enc)))[:ref_len]
                            alt_enc = np.pad(alt_enc, (0, alt_len - len(alt_enc)))[:alt_len]
                            c5p_enc = np.pad(c5p_enc, (0, context_len - len(c5p_enc)))[:context_len]
                            c3p_enc = np.pad(c3p_enc, (0, context_len - len(c3p_enc)))[:context_len]

                            local_ref.append(ref_enc)
                            local_alt.append(alt_enc)
                            local_c5p.append(c5p_enc)
                            local_c3p.append(c3p_enc)

                            # Add chromosome and normalized position
                            chr_encoded = self.chr_encoder['chr_to_int'].get(chrom, 0)
                            local_chr.append([chr_encoded])
                            local_pos.append([self._normalize_position(chrom, pos)])

                            if vaf is not None:
                                local_vaf.append([vaf[idx]])

                    # Add to batch lists
                    ref_batch.append(local_ref)
                    alt_batch.append(local_alt)
                    context_5p_batch.append(local_c5p)
                    context_3p_batch.append(local_c3p)
                    chr_batch.append(local_chr)
                    pos_batch.append(local_pos)

                    if vaf is not None:
                        vaf_batch.append(local_vaf)

                    idx_batch.append(list(chosen_indices))

                    if sample_exp is not None:
                        exp_vals = sample_exp.loc[s_id].values
                        sample_exp_batch.append(np.tile(exp_vals, (len(chosen_indices), 1)))

                # Add labels and sample IDs (common to all modalities)
                if Y_encoded is not None:
                    label_batch.append(sample_to_label[s_id])
                sample_ids_batch.append(s_id)

                # Process CNA segments for this sample
                if use_cna and s_id in cna_sample_dict:
                    cna_indices = cna_sample_dict[s_id]

                    # Select CNA segments (with optional subsampling)
                    if cna_subsample is not None and len(cna_indices) > cna_subsample:
                        chosen_cna_indices = np.random.choice(cna_indices, cna_subsample, replace=False)
                    else:
                        chosen_cna_indices = cna_indices

                    if precomputed_cna is not None:
                        # FAST PATH: numpy fancy indexing (~microseconds)
                        chosen_cna_arr = np.asarray(chosen_cna_indices, dtype=np.intp)
                        local_cna_chr = list(precomputed_cna['chr'][chosen_cna_arr])
                        local_cna_start = list(precomputed_cna['start'][chosen_cna_arr])
                        local_cna_end = list(precomputed_cna['end'][chosen_cna_arr])
                        local_cna_length = list(precomputed_cna['length'][chosen_cna_arr])
                        local_cna_segment_mean = list(precomputed_cna['segment_mean'][chosen_cna_arr])
                        local_cna_idx = list(chosen_cna_indices)
                        if cna_lohs is not None:
                            local_cna_loh = list(precomputed_cna['loh'][chosen_cna_arr])
                        else:
                            local_cna_loh = []
                    else:
                        # ORIGINAL PATH: per-segment encoding (completely unchanged)
                        local_cna_chr, local_cna_start, local_cna_end = [], [], []
                        local_cna_length, local_cna_segment_mean, local_cna_idx = [], [], []
                        local_cna_loh = []

                        for cna_idx in chosen_cna_indices:
                            # Get CNA segment information
                            chrom = str(cna_chromosomes[cna_idx])
                            start = cna_starts[cna_idx]
                            end = cna_ends[cna_idx]
                            segment_mean = cna_segment_means[cna_idx]

                            # Encode using the encode_cna_segment function with chromosome-specific normalization
                            chr_enc, start_norm, end_norm, len_log = tessera.data.preprocessing.encode_cna_segment(
                                chrom, start, end, self.chr_encoder, self.chr_sizes
                            )

                            # Store with extra dimension for consistency with variant inputs
                            local_cna_chr.append([chr_enc])
                            local_cna_start.append([start_norm])
                            local_cna_end.append([end_norm])
                            local_cna_length.append([len_log])
                            local_cna_segment_mean.append([segment_mean])
                            local_cna_idx.append(cna_idx)  # Store original CNA index

                            # Add optional CNA features if provided
                            if cna_lohs is not None:
                                local_cna_loh.append([float(cna_lohs[cna_idx])])  # Convert bool to float

                    # Add to batch lists
                    cna_chr_batch.append(local_cna_chr)
                    cna_start_batch.append(local_cna_start)
                    cna_end_batch.append(local_cna_end)
                    cna_length_batch.append(local_cna_length)
                    cna_segment_mean_batch.append(local_cna_segment_mean)
                    cna_idx_batch.append(local_cna_idx)

                    if cna_lohs is not None:
                        cna_loh_batch.append(local_cna_loh)

                elif use_cna:
                    # Sample has no CNA data - add empty lists
                    cna_chr_batch.append([])
                    cna_start_batch.append([])
                    cna_end_batch.append([])
                    cna_length_batch.append([])
                    cna_segment_mean_batch.append([])
                    cna_idx_batch.append([])
                    if cna_lohs is not None:
                        cna_loh_batch.append([])

            # Build batch data dictionary - always include sample_ids
            batch_data = {
                'sample_ids': tf.constant(sample_ids_batch, dtype=tf.string),
            }

            # Add mutation data if present
            if use_mut:
                # Calculate bag size
                if fixed_bag_size:
                    bag_size = global_bag_size
                else:
                    bag_size = max(len(x) for x in ref_batch) if any(len(x) > 0 for x in ref_batch) else 1

                # Pad all variant sequences with appropriate shapes
                ref_pad = self._pad_variants(ref_batch, bag_size, item_shape=(ref_len,))
                alt_pad = self._pad_variants(alt_batch, bag_size, item_shape=(alt_len,))
                c5p_pad = self._pad_variants(context_5p_batch, bag_size, item_shape=(context_len,))
                c3p_pad = self._pad_variants(context_3p_batch, bag_size, item_shape=(context_len,))
                idx_pad = self._pad_variants(idx_batch, bag_size, pad_val=-1, item_shape=())
                chr_pad = self._pad_variants(chr_batch, bag_size, pad_val=0, item_shape=(1,))
                pos_pad = self._pad_variants(pos_batch, bag_size, pad_val=0.0, item_shape=(1,))

                # Add mutation tensors to batch data
                batch_data['ref'] = tf.constant(ref_pad, dtype=tf.int32)
                batch_data['alt'] = tf.constant(alt_pad, dtype=tf.int32)
                batch_data['context_5p'] = tf.constant(c5p_pad, dtype=tf.int32)
                batch_data['context_3p'] = tf.constant(c3p_pad, dtype=tf.int32)
                batch_data['chr'] = tf.constant(chr_pad, dtype=tf.int32)
                batch_data['pos'] = tf.constant(pos_pad, dtype=tf.float32)
                batch_data['idx'] = tf.constant(idx_pad, dtype=tf.int32)

                if vaf is not None:
                    vaf_pad = self._pad_variants(vaf_batch, bag_size, pad_val=0.0)
                    batch_data['vaf'] = tf.constant(vaf_pad, dtype=tf.float32)

                if sample_exp is not None:
                    exp_pad = self._pad_variants(sample_exp_batch, bag_size)
                    batch_data['exp'] = tf.constant(exp_pad, dtype=tf.float32)

            if Y_encoded is not None:
                batch_data['label'] = np.array(label_batch)

            # Add CNA data if present
            if use_cna:
                # Calculate CNA bag size
                if fixed_bag_size and global_cna_bag_size is not None:
                    cna_bag_size = global_cna_bag_size
                else:
                    cna_bag_size = max(len(x) for x in cna_chr_batch) if any(len(x) > 0 for x in cna_chr_batch) else 1

                # Pad CNA segments with shape (1,) for consistency with variant inputs
                cna_chr_pad = self._pad_variants(cna_chr_batch, cna_bag_size, pad_val=0, item_shape=(1,))
                cna_start_pad = self._pad_variants(cna_start_batch, cna_bag_size, pad_val=0.0, item_shape=(1,))
                cna_end_pad = self._pad_variants(cna_end_batch, cna_bag_size, pad_val=0.0, item_shape=(1,))
                cna_length_pad = self._pad_variants(cna_length_batch, cna_bag_size, pad_val=0.0, item_shape=(1,))
                cna_segment_mean_pad = self._pad_variants(cna_segment_mean_batch, cna_bag_size, pad_val=0.0, item_shape=(1,))
                cna_idx_pad = self._pad_variants(cna_idx_batch, cna_bag_size, pad_val=-1, item_shape=())

                # Add to batch data (CNA features have shape (batch, num_segments, 1))
                batch_data['cna_chr'] = tf.constant(cna_chr_pad, dtype=tf.int32)
                batch_data['cna_start'] = tf.constant(cna_start_pad, dtype=tf.float32)
                batch_data['cna_end'] = tf.constant(cna_end_pad, dtype=tf.float32)
                batch_data['cna_length'] = tf.constant(cna_length_pad, dtype=tf.float32)
                batch_data['cna_segment_mean'] = tf.constant(cna_segment_mean_pad, dtype=tf.float32)
                batch_data['cna_idx'] = tf.constant(cna_idx_pad, dtype=tf.int32)
                batch_data['cna_sample_ids'] = tf.constant(sample_ids_batch, dtype=tf.string)

                # Add optional CNA features if provided
                if cna_lohs is not None:
                    cna_loh_pad = self._pad_variants(cna_loh_batch, cna_bag_size, pad_val=0.0, item_shape=(1,))
                    batch_data['cna_loh'] = tf.constant(cna_loh_pad, dtype=tf.float32)

            return batch_data

        return process_batch

    def _create_dataset_generator(self, unique_samples, sample_dict, process_batch,
                                  batch_size, batch_size_num_variants, is_training,
                                  expand_samples_n=1):
        """
        Create the dataset generator function.

        Args:
            expand_samples_n: If > 1, pre-expand samples by shuffling n times and concatenating.
                             Eliminates the while True loop overhead. Default=1 (no expansion).

        Returns:
            callable: Generator function for the dataset
        """
        def get_batch_samples(samples_list, target_variants):
            """Yields subsets of samples so that total # of variants >= target_variants."""
            batch_samples = []
            current_variants = 0
            for s in samples_list:
                batch_samples.append(s)
                current_variants += len(sample_dict[s])
                if current_variants >= target_variants:
                    yield batch_samples
                    batch_samples = []
                    current_variants = 0
            if batch_samples:
                yield batch_samples

        # Pre-expand samples if requested
        if expand_samples_n > 1 and is_training:
            expanded_samples = []
            for _ in range(expand_samples_n):
                shuffled = unique_samples.copy()
                np.random.shuffle(shuffled)
                expanded_samples.extend(shuffled)
            use_expanded = True
        else:
            expanded_samples = None
            use_expanded = False

        if batch_size_num_variants is not None:
            # Batch by number of variants
            if is_training:
                if use_expanded:
                    # Single pass through pre-expanded samples
                    def generator():
                        for group in get_batch_samples(expanded_samples, batch_size_num_variants):
                            yield process_batch(group)
                else:
                    # Original while True loop with runtime shuffling
                    def generator():
                        while True:
                            s = unique_samples.copy()
                            np.random.shuffle(s)
                            for group in get_batch_samples(s, batch_size_num_variants):
                                yield process_batch(group)
            else:
                def generator():
                    s = unique_samples.copy()
                    for group in get_batch_samples(s, batch_size_num_variants):
                        yield process_batch(group)
        else:
            # Batch by number of samples
            if is_training:
                if use_expanded:
                    # Single pass through pre-expanded samples
                    def generator():
                        max_batches = len(expanded_samples) // batch_size
                        for i in range(0, max_batches * batch_size, batch_size):
                            yield process_batch(expanded_samples[i:i + batch_size])
                else:
                    # Original while True loop with runtime shuffling
                    def generator():
                        while True:
                            s = unique_samples.copy()
                            np.random.shuffle(s)
                            max_batches = len(s) // batch_size
                            for i in range(0, max_batches * batch_size, batch_size):
                                yield process_batch(s[i:i + batch_size])
            else:
                def generator():
                    s = unique_samples.copy()
                    for i in range(0, len(s), batch_size):
                        yield process_batch(s[i:i + batch_size])

        return generator

    def create_sample_dataset(
            self,
            sample_ids=None,
            chromosomes=None,
            positions=None,
            refs=None,
            alts=None,
            context_len=25,
            vaf=None,
            Y=None,
            sample_exp=None,
            batch_size=24,
            subsample=None,
            name='dataset',
            is_training=True,
            batch_size_num_variants=None,
            ref_len=1,
            alt_len=1,
            fixed_bag_size=True,
            set_class_weights=False,
            use_context_cache=False,
            expand_samples_n=1,
            # CNA parameters
            cna_sample_ids=None,
            cna_chromosomes=None,
            cna_starts=None,
            cna_ends=None,
            cna_segment_means=None,
            cna_lohs=None,
            cna_subsample=None,
            z_score_cna=False,
            z_score_clip=None,
            precompute=True,
            precompute_batches=None
    ):
        """
        Creates a tf.data.Dataset of batches, each batch containing bags of variants per sample.
        Optionally includes CNA (Copy Number Alteration) data for multi-modal learning.

        Parameters
        ----------
        sample_ids : array-like
            Sample identifiers
        chromosomes : array-like
            Chromosome names for each variant
        positions : array-like
            Genomic positions for each variant
        refs : array-like
            Reference alleles
        alts : array-like
            Alternate alleles
        context_len : int, default=25
            Length of sequence context to include on each side of the
            variant. ``25`` matches the TCGA Pan-Cancer pretraining run.
        vaf : array-like, optional
            Variant allele frequencies
        Y : array-like, optional
            Labels for classification
        sample_exp : array-like, optional
            Sample expression data
        batch_size : int, default=24
            Number of samples per batch. ``24`` matches the TCGA
            pretraining run.
        subsample : int, optional
            Maximum number of variants to keep per sample
        name : str, default='dataset'
            Name for the dataset attribute
        is_training : bool, default=True
            Whether this is a training dataset (enables shuffling)
        batch_size_num_variants : int, optional
            If provided, batch by number of variants instead of samples
        ref_len : int, default=1
            Fixed length for reference sequences. ``1`` is correct for
            single-nucleotide variants (the model was pretrained on
            SNVs only); raise to e.g. ``10`` for indel-aware datasets.
        alt_len : int, default=1
            Fixed length for alternate sequences. See ``ref_len``.
        fixed_bag_size : bool, default=True
            If True, use a fixed bag size across all batches. Required
            for ``jit_compile=True`` training.
        set_class_weights : bool, default=False
            If True, store class weights in the model
        use_context_cache : bool, default=False
            If True, cache sequence contexts to avoid redundant FASTA lookups
        expand_samples_n : int, default=1
            If > 1, pre-expand samples by shuffling n times and concatenating.
            Eliminates generator cycling overhead for small datasets. Default=1 (no expansion).
        cna_sample_ids : array-like, optional
            Sample identifiers for CNA segments
        cna_chromosomes : array-like, optional
            Chromosome names for each CNA segment
        cna_starts : array-like, optional
            Start positions for CNA segments
        cna_ends : array-like, optional
            End positions for CNA segments
        cna_segment_means : array-like, optional
            Log-fold change values for CNA segments
        cna_subsample : int, optional
            Maximum number of CNA segments to keep per sample
        z_score_cna : bool, default=False
            If True, z-score normalize cna_segment_means using dataset
            mean/std. ``False`` matches the TCGA pretraining run, which
            consumed raw log2 ratios.
        z_score_clip : float, default=None
            Clip normalized values to ±z_score_clip std. None disables clipping.
        precompute : bool, default=True
            If True, precompute all variant/CNA encodings before training starts.
            Eliminates per-batch FASTA I/O and encoding overhead (~50-200ms/batch → ~1-5ms/batch).
            Set to False to use the original per-batch FASTA-based code path.
        precompute_batches : int, optional
            If set, eagerly materializes this many batches into memory as stacked tensors,
            bypassing the Python generator entirely during training/validation.
            Ideal for validation datasets where data doesn't need to change between epochs.
        """
        # Load FASTA and chromosome mapper
        fasta = Fasta(self.fasta_path)
        chromosome_mapper = tessera.data.preprocessing.ChromosomeMapper()
        context_cache = {}

        # Check if CNA data is provided
        use_cna = (cna_sample_ids is not None and
                   cna_chromosomes is not None and
                   cna_starts is not None and
                   cna_ends is not None and
                   cna_segment_means is not None)

        # Organize CNA data by sample if provided
        cna_sample_dict = {}
        if use_cna:
            self.use_cna = True  # Auto-detect CNA modality
            for i, s_id in enumerate(cna_sample_ids):
                if s_id not in cna_sample_dict:
                    cna_sample_dict[s_id] = []
                cna_sample_dict[s_id].append(i)
            print(f"CNA data detected: {len(cna_sample_ids)} segments from {len(cna_sample_dict)} samples")

            # Z-score normalize segment_mean values
            if z_score_cna:
                seg_mean = np.mean(cna_segment_means)
                seg_std = np.std(cna_segment_means)
                cna_segment_means = (cna_segment_means - seg_mean) / seg_std
                if z_score_clip is not None:
                    n_clipped = np.sum(np.abs(cna_segment_means) > z_score_clip)
                    cna_segment_means = np.clip(cna_segment_means, -z_score_clip, z_score_clip)
                    print(f"  [Z-SCORE NORMALIZED] original mean={seg_mean:.4f}, std={seg_std:.4f} → normalized to mean=0, std=1, clipped {n_clipped:,} values to ±{z_score_clip}")
                else:
                    print(f"  [Z-SCORE NORMALIZED] original mean={seg_mean:.4f}, std={seg_std:.4f} → normalized to mean=0, std=1")

            # Detect optional CNA features
            if cna_lohs is not None:
                self.use_cna_loh = True

        # Check if mutation data is provided
        if sample_ids is not None and len(sample_ids) > 0:
            self.use_mut = True  # Auto-detect mutation modality
            print(f"Mutation data detected: {len(sample_ids)} variants")

        # Prepare inputs and compute derived values
        prep_data = self._prepare_dataset_inputs(
            sample_ids, chromosomes, positions, refs, alts, context_len,
            vaf, Y, sample_exp, ref_len, alt_len, subsample, fixed_bag_size,
            set_class_weights, use_cna, cna_sample_dict
        )

        # Precompute variant and CNA encodings if enabled
        precomputed_variants = None
        precomputed_cna = None

        if precompute and self.use_mut:
            precomputed_variants = self._precompute_variant_encodings(
                chromosomes, positions, refs, alts,
                prep_data['context_len'], prep_data['ref_len'], prep_data['alt_len'],
                vaf, fasta, chromosome_mapper
            )

        if precompute and use_cna:
            precomputed_cna = self._precompute_cna_encodings(
                cna_chromosomes, cna_starts, cna_ends,
                cna_segment_means, cna_lohs
            )

        # Create batch processor
        process_batch = self._create_batch_processor(
            chromosomes, positions, refs, alts, prep_data['context_len'], vaf, sample_exp,
            prep_data['ref_len'], prep_data['alt_len'], subsample, fixed_bag_size,
            prep_data['global_bag_size'], prep_data['Y_encoded'],
            prep_data['sample_to_label'], prep_data['sample_dict'],
            use_context_cache, fasta, chromosome_mapper, context_cache,
            # Modality flags
            self.use_mut, use_cna,
            # CNA parameters
            cna_sample_dict, cna_chromosomes, cna_starts,
            cna_ends, cna_segment_means,
            cna_lohs,
            cna_subsample, prep_data['global_cna_bag_size'],
            # Precomputed data
            precomputed_variants, precomputed_cna
        )

        # Create dataset generator
        generator = self._create_dataset_generator(
            prep_data['unique_samples'], prep_data['sample_dict'], process_batch,
            batch_size, batch_size_num_variants, is_training, expand_samples_n
        )

        # Define output signature - always include sample_ids
        bag_dim = None
        output_signature = {
            'sample_ids': tf.TensorSpec(shape=(None,), dtype=tf.string),
        }

        # Add mutation output signature if mutation data is present
        if self.use_mut:
            output_signature['ref'] = tf.TensorSpec(shape=(None, bag_dim, prep_data['ref_len']), dtype=tf.int32)
            output_signature['alt'] = tf.TensorSpec(shape=(None, bag_dim, prep_data['alt_len']), dtype=tf.int32)
            output_signature['context_5p'] = tf.TensorSpec(shape=(None, bag_dim, prep_data['context_len']), dtype=tf.int32)
            output_signature['context_3p'] = tf.TensorSpec(shape=(None, bag_dim, prep_data['context_len']), dtype=tf.int32)
            output_signature['chr'] = tf.TensorSpec(shape=(None, bag_dim, 1), dtype=tf.int32)
            output_signature['pos'] = tf.TensorSpec(shape=(None, bag_dim, 1), dtype=tf.float32)
            output_signature['idx'] = tf.TensorSpec(shape=(None, bag_dim), dtype=tf.int32)

            if vaf is not None:
                output_signature['vaf'] = tf.TensorSpec(shape=(None, bag_dim, 1), dtype=tf.float32)
            if sample_exp is not None:
                output_signature['exp'] = tf.TensorSpec(shape=(None, bag_dim, sample_exp.shape[1]), dtype=tf.float32)

        if Y is not None:
            output_signature['label'] = tf.TensorSpec(shape=(None,), dtype=tf.int32)

        # Add CNA output signature if CNA data is provided (3D tensors like variants)
        if use_cna:
            cna_bag_dim = None  # Variable CNA bag size
            output_signature['cna_chr'] = tf.TensorSpec(shape=(None, cna_bag_dim, 1), dtype=tf.int32)
            output_signature['cna_start'] = tf.TensorSpec(shape=(None, cna_bag_dim, 1), dtype=tf.float32)
            output_signature['cna_end'] = tf.TensorSpec(shape=(None, cna_bag_dim, 1), dtype=tf.float32)
            output_signature['cna_length'] = tf.TensorSpec(shape=(None, cna_bag_dim, 1), dtype=tf.float32)
            output_signature['cna_segment_mean'] = tf.TensorSpec(shape=(None, cna_bag_dim, 1), dtype=tf.float32)
            output_signature['cna_idx'] = tf.TensorSpec(shape=(None, cna_bag_dim), dtype=tf.int32)
            output_signature['cna_sample_ids'] = tf.TensorSpec(shape=(None,), dtype=tf.string)

            # Add optional CNA features to output signature if provided
            if cna_lohs is not None:
                output_signature['cna_loh'] = tf.TensorSpec(shape=(None, cna_bag_dim, 1), dtype=tf.float32)

        # Create TensorFlow dataset
        if precompute_batches is not None and precompute_batches > 0:
            # Eagerly materialize batches — eliminates Python generator overhead entirely
            print(f"Precomputing {precompute_batches} batches into memory...")
            t0 = time.time()
            cached = []
            gen = generator()
            for _ in tqdm(range(precompute_batches), desc="  Caching batches", unit="batch"):
                cached.append(next(gen))

            # Stack into a single dict of tensors (each slice = one pre-built batch)
            stacked = {}
            for key in cached[0]:
                stacked[key] = tf.stack([b[key] for b in cached])
            del cached

            elapsed = time.time() - t0
            print(f"  {precompute_batches} batches cached in {elapsed:.1f}s")

            with self.strategy.scope():
                dataset = tf.data.Dataset.from_tensor_slices(stacked).repeat().prefetch(tf.data.AUTOTUNE)
        else:
            with self.strategy.scope():
                dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)

        # Attach to self with given name
        setattr(self, name, dataset)
