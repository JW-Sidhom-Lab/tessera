"""Compatibility utilities for loading .keras checkpoints trained with the
legacy ``pkg.var_llm.VariantLLM`` stack against the new ``tessera`` package.

A ``.keras`` archive is a zip of ``metadata.json`` + ``config.json`` +
``model.weights.h5``. The custom-layer registration paths are baked into
``config.json``'s ``"module"`` fields and the keras serialization
registry name (e.g., ``"pkg.functions.layers.attention>GlobalAttentionBlock"``).
The class definitions are unchanged between the two packages; only the
import paths moved (``pkg.functions.`` -> ``tessera.``). Substituting that
prefix everywhere it appears in ``config.json`` is sufficient to make a
legacy checkpoint load with the new package; ``model.weights.h5`` and
``metadata.json`` carry no module references.

Two interchangeable strategies are exposed:

1. :func:`relocate_keras_file` / :func:`relocate_keras_tree` -- rewrite
   ``config.json`` inside the ``.keras`` archive and write a new file.
   Preferred for a published checkpoint: the relocated archive loads
   under vanilla ``keras.models.load_model`` with no shim.

2. :func:`install_legacy_aliases` -- a runtime ``sys.modules`` shim that
   makes the legacy import paths resolve to the new modules. Useful for
   inspecting an archive in place without writing a copy.

CLI::

    python -m tessera._legacy <src_dir_or_file> <dst_dir_or_file>
"""

from __future__ import annotations

import json
import shutil
import sys
import types
import zipfile
from pathlib import Path
from typing import Any, List, Union

# Every legacy reference that needs rewriting starts with this exact
# prefix; replacing it with the new prefix handles both the bare
# ``"module"`` form (``pkg.functions.layers.X``) and the registered-
# serializable name form (``pkg.functions.layers.X>ClassName``) in a
# single string substitution.
_LEGACY_PREFIX = "pkg.functions."
_NEW_PREFIX = "tessera."

# Layer-config keys that newer Keras emits but older Keras versions do not
# accept as ``__init__`` kwargs. We strip these only when their saved value
# is ``None`` (the default), so removal cannot change loaded behaviour.
# Encountered so far:
#   ``quantization_config``  (Keras >=3.7 emits; older Keras rejects it)
_NONE_VALUED_KEYS_TO_STRIP = ("quantization_config",)


def _strip_none_keys(node: Any) -> None:
    """Recursively remove ``None``-valued legacy-only keys from a parsed config."""
    if isinstance(node, dict):
        for key in _NONE_VALUED_KEYS_TO_STRIP:
            if node.get(key) is None and key in node:
                node.pop(key, None)
        for value in node.values():
            _strip_none_keys(value)
    elif isinstance(node, list):
        for item in node:
            _strip_none_keys(item)


def rewrite_config_json(payload: bytes) -> bytes:
    """Rewrite a legacy config.json blob for the new tessera package.

    Two transformations are applied:

    1. ``pkg.functions.`` -> ``tessera.`` everywhere (covers both
       ``"module"`` paths and ``"registered_name"`` keys).
    2. Strip any ``quantization_config: null`` entries -- newer Keras
       emits the field but older Keras versions reject it as a Dense
       kwarg. Only ``None``-valued occurrences are removed; if a future
       checkpoint actually held a quantization configuration it would
       be preserved, but the manuscript checkpoints don't.
    """
    text = payload.decode("utf-8").replace(_LEGACY_PREFIX, _NEW_PREFIX)
    cfg = json.loads(text)
    _strip_none_keys(cfg)
    return json.dumps(cfg).encode("utf-8")


def relocate_keras_file(src: Union[Path, str], dst: Union[Path, str]) -> None:
    """Copy a legacy ``.keras`` archive to ``dst`` with config.json rewritten."""
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(src, "r") as zin, \
         zipfile.ZipFile(dst, "w") as zout:
        for item in zin.infolist():
            payload = zin.read(item.filename)
            if item.filename == "config.json":
                payload = rewrite_config_json(payload)
            # Pass the ZipInfo object so per-file compression settings carry over.
            zout.writestr(item, payload)


def relocate_keras_tree(src_dir: Union[Path, str],
                        dst_dir: Union[Path, str]) -> List[Path]:
    """Mirror ``src_dir`` to ``dst_dir`` with every ``*.keras`` file relocated.

    Non-keras files are copied byte-for-byte. Returns the list of relocated
    ``.keras`` destination paths.
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    relocated: List[Path] = []
    for src in src_dir.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(src_dir)
        dst = dst_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.suffix == ".keras":
            relocate_keras_file(src, dst)
            relocated.append(dst)
        else:
            shutil.copy2(src, dst)
    return relocated


# ---------------------------------------------------------------------
# Runtime-shim alternative.
# ---------------------------------------------------------------------

_OLD_TO_NEW_MODULES = {
    "pkg.functions.layers.attention":        "tessera.layers.attention",
    "pkg.functions.layers.cna_features":     "tessera.layers.cna_features",
    "pkg.functions.layers.masking":          "tessera.layers.masking",
    "pkg.functions.layers.pooling":          "tessera.layers.pooling",
    "pkg.functions.layers.utils":            "tessera.layers.utils",
    "pkg.functions.layers.variant_features": "tessera.layers.variant_features",
    "pkg.functions.training.models":         "tessera.training.models",
}


def install_legacy_aliases() -> None:
    """Register sys.modules aliases so legacy import paths resolve to tessera.

    Idempotent. Call once before ``keras.models.load_model`` on a legacy
    checkpoint::

        from tessera._legacy import install_legacy_aliases
        install_legacy_aliases()
        model = keras.models.load_model("legacy_best_model.keras")
    """
    import importlib

    for parent in ("pkg", "pkg.functions", "pkg.functions.layers",
                   "pkg.functions.training"):
        if parent not in sys.modules:
            mod = types.ModuleType(parent)
            mod.__path__ = []  # mark as a package so submodule imports work
            sys.modules[parent] = mod

    for old, new in _OLD_TO_NEW_MODULES.items():
        sys.modules[old] = importlib.import_module(new)


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Rewrite legacy pkg.functions.* references in .keras "
                    "checkpoints to the new tessera.* import paths.",
    )
    parser.add_argument("src", type=Path,
                        help="Source .keras file or directory tree.")
    parser.add_argument("dst", type=Path,
                        help="Destination .keras file or directory.")
    args = parser.parse_args()

    if args.src.is_dir():
        moved = relocate_keras_tree(args.src, args.dst)
        print(f"Relocated {len(moved)} .keras files into {args.dst}")
    else:
        relocate_keras_file(args.src, args.dst)
        print(f"Relocated {args.src} -> {args.dst}")


if __name__ == "__main__":
    _cli()
