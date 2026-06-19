"""Reference-genome provisioning for TESSERA.

TESSERA needs a reference FASTA for SNV sequence-context lookups. The FASTA is
large (~3 GB uncompressed) so it is **not** shipped inside the package wheel;
instead it is fetched on first use and cached locally — mirroring how the
pretrained weights cache under ``~/.cache/huggingface`` and the liftover chain
files under ``~/.cache/pyliftover``.

Resolution order for a given genome build (first hit wins):

1. ``$TESSERA_REF_GENOME_DIR`` — a directory you populate yourself (offline
   clusters, shared read-only mounts, custom FASTA location).
2. The user cache dir: ``$XDG_CACHE_HOME/tessera/ref_genomes`` or
   ``~/.cache/tessera/ref_genomes``.
3. The package's own ``ref_genomes/`` directory (back-compat with
   ``download_ref_genomes.sh``, which writes there).

A candidate is only accepted if it meets a minimum-size sanity check, so a
truncated or partially written FASTA (an interrupted download, a disk-full
event) is treated as absent and re-provisioned rather than trusted forever.

If the FASTA is not found in any of these, :func:`ensure_reference_genome`
downloads it from NCBI into the user cache dir (or ``$TESSERA_REF_GENOME_DIR``
if that is set). The download verifies the byte count against the server's
``Content-Length`` and the decompressed size against the expected minimum, and
writes atomically so an interrupted run never leaves a truncated FASTA in place.

This module also exposes a small CLI used by
``ref_genomes/download_ref_genomes.sh``::

    python -m tessera.ref_genome [grch37 | grch38 | both] [--dir DIR]

Concurrency note: provisioning is safe to run from multiple processes (each
downloads to a unique temp and the final rename is atomic), but the pyfaidx
``.fai`` index is built without a cross-process lock; pre-provision the genome
(or its index) when launching many jobs simultaneously on a cold cache.
"""

from __future__ import annotations

import argparse
import gzip
import logging
import os
import shutil
import tempfile
import urllib.request

from pyfaidx import Fasta
from tqdm import tqdm

_log = logging.getLogger(__name__)

#: Environment variable pointing at a directory that holds the FASTA files.
_ENV_DIR_VAR = "TESSERA_REF_GENOME_DIR"

#: Per-read socket timeout (seconds). Bounds a stalled connection instead of
#: hanging forever on a multi-minute, multi-GB transfer.
_DOWNLOAD_TIMEOUT = 60

#: genome_version -> (FASTA filename, NCBI gzipped-FASTA URL, minimum plausible
#: uncompressed size in bytes). The size floor guards against truncated files
#: being trusted as complete (the real GRCh37/38 FASTAs are ~3.3 GB).
GENOMES = {
    "GRCh37": (
        "GCF_000001405.25_GRCh37.p13_genomic.fna",
        "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/"
        "GCF_000001405.25_GRCh37.p13/GCF_000001405.25_GRCh37.p13_genomic.fna.gz",
        3_000_000_000,
    ),
    "GRCh38": (
        "GCF_000001405.40_GRCh38.p14_genomic.fna",
        "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/"
        "GCF_000001405.40_GRCh38.p14/GCF_000001405.40_GRCh38.p14_genomic.fna.gz",
        3_000_000_000,
    ),
}

_PACKAGE_REF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref_genomes")


def _env_dir() -> str | None:
    """The user-configured override directory, if set."""
    return os.environ.get(_ENV_DIR_VAR)


def _user_cache_dir() -> str:
    """Persistent per-user cache directory for reference FASTAs."""
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "tessera", "ref_genomes")


def _writable_target_dir() -> str:
    """Directory new downloads are written to (env override or user cache)."""
    return _env_dir() or _user_cache_dir()


def _candidate_dirs() -> list[str]:
    """Directories searched for an already-present FASTA, in priority order."""
    dirs = [d for d in (_env_dir(), _user_cache_dir(), _PACKAGE_REF_DIR) if d]
    seen: set[str] = set()
    return [d for d in dirs if not (d in seen or seen.add(d))]


def _entry(genome_version: str) -> tuple[str, str, int]:
    try:
        return GENOMES[genome_version]
    except KeyError:
        raise ValueError(
            f"Unsupported genome_version {genome_version!r}; expected one of {tuple(GENOMES)}"
        ) from None


def _is_complete(path: str, min_bytes: int) -> bool:
    """True if ``path`` exists and is at least ``min_bytes`` (i.e. not truncated)."""
    try:
        return os.path.getsize(path) >= min_bytes
    except OSError:
        return False


def _find_existing(filename: str, min_bytes: int) -> str | None:
    for d in _candidate_dirs():
        path = os.path.join(d, filename)
        if _is_complete(path, min_bytes):
            return path
    return None


def resolved_fasta_path(genome_version: str) -> str | None:
    """Return the path to an already-available reference FASTA, or ``None``.

    Never downloads. Returns the first candidate location that holds a complete
    FASTA (passing the minimum-size check); returns ``None`` when the genome is
    not present locally yet, in which case :func:`ensure_reference_genome`
    downloads it on first use. A ``None`` return is the honest answer to
    "where is the FASTA right now?" — callers should not treat it as a path.
    """
    filename, _url, min_bytes = _entry(genome_version)
    return _find_existing(filename, min_bytes)


def ensure_reference_genome(genome_version: str) -> str:
    """Return a local path to the reference FASTA, downloading it if necessary.

    Searches the candidate locations first; on a miss, streams the gzipped FASTA
    from NCBI into the cache directory and decompresses it. The companion
    ``.fai`` index is created lazily by pyfaidx on first access (see
    :func:`open_fasta` for read-only-directory handling).
    """
    filename, url, min_bytes = _entry(genome_version)
    existing = _find_existing(filename, min_bytes)
    if existing is not None:
        return existing

    target_dir = _writable_target_dir()
    os.makedirs(target_dir, exist_ok=True)
    dest = os.path.join(target_dir, filename)
    _download_and_extract(url, dest, min_bytes)
    return dest


def _download_and_extract(url: str, dest: str, min_bytes: int) -> None:
    """Stream a gzipped FASTA from ``url`` and decompress it to ``dest`` atomically.

    Downloads to a unique temp ``.gz``, verifies the byte count against the
    server's ``Content-Length``, decompresses to a unique temp FASTA, checks the
    result meets ``min_bytes``, then atomically renames it into place — so an
    interrupted or truncated run never leaves a FASTA that looks complete. Temp
    files are always cleaned up.
    """
    _log.warning(
        "TESSERA: reference genome not found locally; downloading from NCBI "
        "(~3 GB, one-time, cached for future runs). source=%s dest=%s",
        url,
        dest,
    )
    target_dir = os.path.dirname(dest)
    gz_tmp = None
    fasta_tmp = None
    try:
        # --- download the gzipped FASTA to a unique temp file ---
        fd, gz_tmp = tempfile.mkstemp(suffix=".fna.gz.part", dir=target_dir)
        bytes_written = 0
        expected = None
        with os.fdopen(fd, "wb") as out:  # owns fd; closes even if the body raises
            request = urllib.request.Request(url, headers={"User-Agent": "tessera-foundation"})
            with urllib.request.urlopen(request, timeout=_DOWNLOAD_TIMEOUT) as resp:  # nosec B310 - trusted https NCBI URL
                expected = int(resp.headers.get("Content-Length", 0)) or None
                with tqdm(total=expected, unit="B", unit_scale=True, desc="Downloading genome") as bar:
                    for chunk in iter(lambda: resp.read(1024 * 1024), b""):
                        out.write(chunk)
                        bytes_written += len(chunk)
                        bar.update(len(chunk))
        if expected is not None and bytes_written != expected:
            raise IOError(
                f"truncated download: received {bytes_written} of {expected} bytes "
                f"(Content-Length) from {url}"
            )

        # --- decompress to a unique temp, validate, then atomically place ---
        fd2, fasta_tmp = tempfile.mkstemp(suffix=".fna.part", dir=target_dir)
        with os.fdopen(fd2, "wb") as fa, gzip.open(gz_tmp, "rb") as gz:
            shutil.copyfileobj(gz, fa, length=1024 * 1024)
        if not _is_complete(fasta_tmp, min_bytes):
            raise IOError(
                f"decompressed reference genome is implausibly small "
                f"({os.path.getsize(fasta_tmp)} bytes, expected >= {min_bytes}); "
                f"the download from {url} is likely corrupt or truncated"
            )
        os.replace(fasta_tmp, dest)
        fasta_tmp = None
    except Exception as e:  # noqa: BLE001 - re-raised with remediation context below
        raise RuntimeError(
            f"TESSERA: failed to provision the reference genome from {url} -> {dest}. "
            f"Set ${_ENV_DIR_VAR} to a directory containing the FASTA, or run "
            f"tessera/ref_genomes/download_ref_genomes.sh, to pre-provision it. Cause: {e}"
        ) from e
    finally:
        for tmp in (gz_tmp, fasta_tmp):
            if tmp and os.path.exists(tmp):
                os.remove(tmp)


def _index_path_for(fasta_path: str) -> str | None:
    """Where pyfaidx should read/write the ``.fai`` index.

    Returns ``None`` to use pyfaidx's default (the index sits next to the FASTA)
    when that is fine: either an index already exists there, or the FASTA's
    directory is writable. When the FASTA lives in a read-only directory
    (system/conda site-packages, or a shared read-only mount), redirect the
    index into the writable user cache so first-use indexing does not fail.
    """
    if os.path.exists(fasta_path + ".fai"):
        return None
    fasta_dir = os.path.dirname(fasta_path) or "."
    if os.access(fasta_dir, os.W_OK):
        return None
    cache = _user_cache_dir()
    os.makedirs(cache, exist_ok=True)
    return os.path.join(cache, os.path.basename(fasta_path) + ".fai")


def open_fasta(fasta_path: str) -> Fasta:
    """Open ``fasta_path`` with pyfaidx, keeping the ``.fai`` index writable.

    Equivalent to ``pyfaidx.Fasta(fasta_path)`` but routes the index to the user
    cache when the FASTA's own directory is read-only, so featurisation works
    out of the box even when the genome was provisioned into site-packages or a
    read-only mount.
    """
    indexname = _index_path_for(fasta_path)
    if indexname is None:
        return Fasta(fasta_path)
    return Fasta(fasta_path, indexname=indexname)


def _download_into(genome_version: str, target_dir: str) -> str:
    """Download ``genome_version`` into ``target_dir`` (skip if already complete)."""
    filename, url, min_bytes = _entry(genome_version)
    os.makedirs(target_dir, exist_ok=True)
    dest = os.path.join(target_dir, filename)
    if _is_complete(dest, min_bytes):
        _log.warning("TESSERA: %s already present at %s; skipping.", genome_version, dest)
        return dest
    _download_and_extract(url, dest, min_bytes)
    return dest


_CLI_TARGETS = {"grch37": ["GRCh37"], "grch38": ["GRCh38"], "both": ["GRCh37", "GRCh38"]}


def main(argv: list[str] | None = None) -> None:
    """CLI: download reference genome(s). Single source of truth for the
    filenames and URLs that ``download_ref_genomes.sh`` wraps."""
    parser = argparse.ArgumentParser(
        prog="python -m tessera.ref_genome",
        description="Download TESSERA reference genome(s) from NCBI.",
    )
    parser.add_argument(
        "target", nargs="?", default="grch37",
        help="grch37 | grch38 | both (default: grch37)",
    )
    parser.add_argument(
        "--dir", default=None,
        help="Directory to download into (default: the user cache dir, "
             "$TESSERA_REF_GENOME_DIR or ~/.cache/tessera/ref_genomes).",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    key = args.target.lower()
    if key not in _CLI_TARGETS:
        parser.error(f"unknown target {args.target!r}; choose grch37 | grch38 | both")
    target_dir = args.dir or _writable_target_dir()
    for genome_version in _CLI_TARGETS[key]:
        _download_into(genome_version, target_dir)


if __name__ == "__main__":
    main()
