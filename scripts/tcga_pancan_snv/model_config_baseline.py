"""Baseline TESSERA SNV model: no attention modules.

Used by ``fit_model.py``, ``get_variant_features.py``,
``get_variant_loss_acc.py``, and ``get_variant_self_attn.py`` via the
``--config baseline`` CLI flag. ``context_len`` is set on the CLI; the
model name is derived from ``(config, context_len)``.
"""

# Mutation type passed to the dataset builder (controls ref/alt token length).
mut_type = "SNV"

# Default sequence-context window in bp on each side of the variant. The
# CLI's ``--context-len`` flag overrides this at run time.
context_len = 1

# Training schedule.
batch_size = 24
epochs = 100
epochs_min = 15
steps_per_epoch = 7500 // batch_size
validation_steps = 2500 // batch_size
validation_freq = 1

# Sample-bag construction.
subsample = None
use_distributed = False

# Model topology toggles (the difference across baseline / local / global).
variant_local_attention = False
variant_self_attention = False
recon_ref = True
