"""Global-attention TESSERA SNV model: per-variant local attention plus inter-variant self-attention.

Used by ``fit_model.py``, ``get_variant_features.py``,
``get_variant_loss_acc.py``, and ``get_variant_self_attn.py`` via the
``--config global`` CLI flag. ``context_len`` is set on the CLI; the
model name is derived from ``(config, context_len)``.
"""

mut_type = "SNV"
context_len = 25

batch_size = 24
epochs = 100
epochs_min = 15
steps_per_epoch = 7500 // batch_size
validation_steps = 2500 // batch_size
validation_freq = 1

subsample = None
use_distributed = False

# Local attention (per-variant flanking-sequence) + global self-attention
# (inter-variant attention across the sample's variant bag).
variant_local_attention = True
variant_self_attention = True
recon_ref = True
