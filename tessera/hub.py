"""Load pretrained TESSERA weights from the Hugging Face Hub.

The :func:`load_pretrained` function is the user-facing one-liner; it
returns a ready-to-use :class:`tessera.model.TESSERA` instance.

The :func:`download_pretrained` helper returns just the local path of
the cached weights without instantiating a model. Useful when you need
the directory for some other purpose (custom loader, multiple models,
etc.).

Example::

    from tessera import load_pretrained

    model = load_pretrained("joint_snv_cna_noloh")
"""
from __future__ import annotations

from pathlib import Path

DEFAULT_REPO_ID = "JW-Sidhom-Lab/tessera-foundation"

#: Names of the pretrained variants published on the Hub. The names are
#: also the subdirectory names inside the model repo.
VARIANTS = (
    "joint_snv_cna",         # joint SNV + CNA InfoNCE-aligned, with the LoH head
    "joint_snv_cna_noloh",   # same architecture without the LoH head
)


def download_pretrained(
    variant: str = "joint_snv_cna_noloh",
    repo_id: str = DEFAULT_REPO_ID,
) -> str:
    """Download a TESSERA variant's weights from the Hub and return the
    local path. Cached under ``~/.cache/huggingface/hub/`` on subsequent
    calls.

    Parameters
    ----------
    variant
        ``"joint_snv_cna"`` (with-LoH; needs allele-specific CNA calls)
        or ``"joint_snv_cna_noloh"`` (without LoH; default; used for all
        published Figure 4-6 results).
    repo_id
        Hugging Face Hub model repo to fetch from. Override only if you
        host a fork of the weights elsewhere.

    Returns
    -------
    str
        Local path of the variant's directory (containing
        ``best_model.keras``, ``features_model_*.keras``, etc.). Pass
        this to :class:`tessera.model.TESSERA`'s ``model_dir`` argument.
    """
    if variant not in VARIANTS:
        raise ValueError(
            f"variant must be one of {VARIANTS}, got {variant!r}"
        )
    from huggingface_hub import snapshot_download
    weights_root = snapshot_download(
        repo_id=repo_id,
        allow_patterns=f"{variant}/*",
    )
    return str(Path(weights_root) / variant)


def load_pretrained(
    variant: str = "joint_snv_cna_noloh",
    repo_id: str = DEFAULT_REPO_ID,
    **kwargs,
):
    """Download and instantiate a pretrained TESSERA model from the Hub.

    Parameters
    ----------
    variant
        ``"joint_snv_cna"`` (with-LoH) or ``"joint_snv_cna_noloh"``
        (default; used for all published Figure 4-6 results).
    repo_id
        Hugging Face Hub model repo to fetch from.
    **kwargs
        Forwarded to :class:`tessera.model.TESSERA`. Sensible inference
        defaults are supplied for ``use_distributed``, ``jit_compile``
        and ``mixed_precision`` if not provided.

    Returns
    -------
    tessera.model.TESSERA
        A ready-to-use TESSERA model.
    """
    from tessera.model import TESSERA

    model_dir = download_pretrained(variant=variant, repo_id=repo_id)
    kwargs.setdefault("use_distributed", False)
    kwargs.setdefault("jit_compile", False)
    kwargs.setdefault("mixed_precision", False)
    return TESSERA(model_dir=model_dir, **kwargs)


# Module-level cache so repeat featurize() calls reuse the loaded model.
_FEATURIZE_MODEL_CACHE: dict = {}


def featurize(
    snv_df=None,
    cna_df=None,
    variant: str = "joint_snv_cna_noloh",
    from_assembly: str = "GRCh37",
    repo_id: str = DEFAULT_REPO_ID,
    return_predictions: bool = False,
    chain_file=None,
    **load_kwargs,
):
    """Single-call inference: dataframes in, embeddings out.

    Loads (or reuses a cached) pretrained TESSERA variant from the Hub
    and runs :meth:`tessera.model.TESSERA.featurize` on the supplied
    dataframes. Coordinates are lifted to GRCh37 first if
    ``from_assembly`` is anything else.

    Parameters
    ----------
    snv_df, cna_df
        Pandas DataFrames with the column conventions documented on
        :meth:`tessera.model.TESSERA.featurize`. At least one is
        required.
    variant
        Hub variant name (``"joint_snv_cna"`` or
        ``"joint_snv_cna_noloh"``).
    from_assembly
        Source reference assembly. ``"GRCh37"`` / ``"hg19"`` is a no-op;
        ``"GRCh38"`` / ``"hg38"`` triggers UCSC liftover.
    repo_id
        Hugging Face Hub repo to fetch from.
    return_predictions
        If True, also run the masked-token / segment-mean reconstruction
        heads and populate the predictions fields on the result.
    chain_file
        Explicit UCSC chain file path; falls through to
        ``TESSERA_LIFTOVER_CHAIN`` env var or pyliftover's auto-download
        otherwise.
    **load_kwargs
        Forwarded to :func:`load_pretrained` on first use of a given
        ``(variant, repo_id)`` pair.

    Returns
    -------
    tessera.model.FeaturizeResult
        Bundle holding ``snv_features`` / ``cna_features``, the
        post-liftover input tables, optional predictions, and
        ``liftover_stats``.
    """
    cache_key = (variant, repo_id)
    if cache_key not in _FEATURIZE_MODEL_CACHE:
        _FEATURIZE_MODEL_CACHE[cache_key] = load_pretrained(
            variant=variant, repo_id=repo_id, **load_kwargs
        )
    model = _FEATURIZE_MODEL_CACHE[cache_key]
    return model.featurize(
        snv_df=snv_df,
        cna_df=cna_df,
        from_assembly=from_assembly,
        return_predictions=return_predictions,
        chain_file=chain_file,
    )
