"""Coordinate liftover from non-hg19 assemblies to GRCh37 (hg19).

TESSERA was pretrained on the TCGA Pan-Cancer Atlas in GRCh37/hg19, so any
input in a different assembly must be lifted before inference. Otherwise the
reference-sequence lookups during SNV tokenization will hit the wrong bases
and CNA segment positions will not align with the gene-context features the
encoder learned.

Two public helpers are exposed:

    lift_snv(df, from_assembly="GRCh38") -> (df, stats)
    lift_cna(df, from_assembly="GRCh38") -> (df, stats)

Both accept the standard SNV / CNA column conventions used elsewhere in
the package and return the lifted DataFrame plus a small dict reporting
how many rows were kept vs. dropped.

Chain files are resolved in this order:

  1. The optional ``chain_file=`` argument, if given.
  2. The ``TESSERA_LIFTOVER_CHAIN`` environment variable, if set.
  3. ``pyliftover`` auto-downloads the appropriate UCSC chain on first
     use and caches it at ``~/.cache/pyliftover/``.

Calling code that needs offline behaviour (e.g. inside a sandboxed Spaces
runtime) can bundle a chain file and point either env var or argument at
it; otherwise the default of letting pyliftover fetch on demand is fine.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# Aliases pyliftover understands. Source values map onto its UCSC names.
_ASSEMBLY_ALIASES = {
    "grch38": "hg38",
    "hg38":   "hg38",
    "grch37": "hg19",
    "hg19":   "hg19",
    "grch36": "hg18",
    "hg18":   "hg18",
}

_DEST_NAME = "hg19"   # TESSERA's training assembly; not a parameter on purpose.

_LIFTERS: dict = {}   # cache: (src, dst, chain_path_or_None) -> LiftOver instance


def _normalize_assembly(name: str) -> str:
    key = name.strip().lower()
    if key not in _ASSEMBLY_ALIASES:
        raise ValueError(
            f"Unknown reference assembly {name!r}. Supported: "
            "'GRCh38'/'hg38', 'GRCh37'/'hg19', 'GRCh36'/'hg18'."
        )
    return _ASSEMBLY_ALIASES[key]


def _get_lifter(from_assembly: str, chain_file: Optional[str] = None):
    src = _normalize_assembly(from_assembly)
    if src == _DEST_NAME:
        return None  # identity: nothing to lift.

    chain = chain_file or os.environ.get("TESSERA_LIFTOVER_CHAIN")
    cache_key = (src, _DEST_NAME, chain)
    if cache_key in _LIFTERS:
        return _LIFTERS[cache_key]

    from pyliftover import LiftOver
    if chain:
        path = Path(chain)
        if not path.exists():
            raise FileNotFoundError(f"Liftover chain file not found: {path}")
        lifter = LiftOver(str(path))
    else:
        lifter = LiftOver(src, _DEST_NAME)
    _LIFTERS[cache_key] = lifter
    return lifter


def _strip_chr(chrom: str) -> str:
    return chrom[3:] if chrom.lower().startswith("chr") else chrom


def lift_snv(
    df: pd.DataFrame,
    from_assembly: str = "GRCh38",
    chain_file: Optional[str] = None,
) -> Tuple[pd.DataFrame, dict]:
    """Lift SNV coordinates to GRCh37/hg19.

    Args:
        df: DataFrame with at least ``Chromosome`` (without a ``chr``
            prefix) and ``Start_Position`` (1-based int) columns. All
            other columns are passed through unchanged on kept rows.
        from_assembly: Source assembly name. ``'GRCh37'`` / ``'hg19'``
            is a no-op (returns ``df`` unchanged).
        chain_file: Optional explicit path to a UCSC chain file
            (e.g. ``hg38ToHg19.over.chain.gz``). When omitted,
            pyliftover downloads the UCSC chain on first use.

    Returns:
        Tuple ``(out_df, stats)`` where ``stats`` is
        ``{'n_in', 'n_out', 'n_dropped'}``. Variants whose coordinates
        cannot be lifted are dropped from ``out_df``.
    """
    lift = _get_lifter(from_assembly, chain_file=chain_file)
    if lift is None:
        return df.copy(), {"n_in": len(df), "n_out": len(df), "n_dropped": 0}

    chrom_in = "chr" + df["Chromosome"].astype(str)
    pos_in   = df["Start_Position"].astype(int)

    new_chrom: list = []
    new_pos:   list = []
    keep_mask: list = []
    for c, p in zip(chrom_in, pos_in):
        result = lift.convert_coordinate(c, int(p) - 1)   # pyliftover is 0-based
        if result:
            tc, tp, _, _ = result[0]
            new_chrom.append(_strip_chr(tc))
            new_pos.append(int(tp) + 1)
            keep_mask.append(True)
        else:
            new_chrom.append(None)
            new_pos.append(None)
            keep_mask.append(False)
    keep_mask_arr = np.array(keep_mask)
    out = df.loc[keep_mask_arr].copy().reset_index(drop=True)
    out["Chromosome"]     = [c for c, k in zip(new_chrom, keep_mask) if k]
    out["Start_Position"] = [p for p, k in zip(new_pos,   keep_mask) if k]
    return out, {
        "n_in": len(df),
        "n_out": len(out),
        "n_dropped": int((~keep_mask_arr).sum()),
    }


def lift_cna(
    df: pd.DataFrame,
    from_assembly: str = "GRCh38",
    chain_file: Optional[str] = None,
) -> Tuple[pd.DataFrame, dict]:
    """Lift CNA segment coordinates to GRCh37/hg19.

    Both segment endpoints must lift to the same hg19 chromosome or the
    segment is dropped. Returned ``Start`` / ``End`` are re-sorted so
    ``Start <= End`` even if the chain inverts orientation.

    Args:
        df: DataFrame with at least ``Chromosome``, ``Start``, ``End``
            columns. Other columns (``Tumor_Sample_Barcode``,
            ``Segment_Mean``, ``LOH``, ...) are passed through.
        from_assembly: Source assembly name. ``'GRCh37'`` / ``'hg19'``
            is a no-op.
        chain_file: Optional explicit path to a UCSC chain file.

    Returns:
        Tuple ``(out_df, stats)``.
    """
    lift = _get_lifter(from_assembly, chain_file=chain_file)
    if lift is None:
        return df.copy(), {"n_in": len(df), "n_out": len(df), "n_dropped": 0}

    rows_out: list = []
    n_dropped = 0
    for _, row in df.iterrows():
        chrom_in = "chr" + str(row["Chromosome"])
        s = lift.convert_coordinate(chrom_in, int(row["Start"]) - 1)
        e = lift.convert_coordinate(chrom_in, int(row["End"])   - 1)
        if not s or not e:
            n_dropped += 1
            continue
        sc, sp, _, _ = s[0]
        ec, ep, _, _ = e[0]
        if sc != ec:
            n_dropped += 1
            continue
        new = row.copy()
        new["Chromosome"] = _strip_chr(sc)
        a, b = sorted([int(sp) + 1, int(ep) + 1])
        new["Start"] = a
        new["End"]   = b
        rows_out.append(new)
    out = pd.DataFrame(rows_out).reset_index(drop=True) if rows_out else df.iloc[0:0].copy()
    return out, {"n_in": len(df), "n_out": len(out), "n_dropped": n_dropped}


__all__ = ["lift_snv", "lift_cna"]
