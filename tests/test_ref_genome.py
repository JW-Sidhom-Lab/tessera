"""Hermetic tests for tessera.ref_genome — no network, no 3 GB download.

The genome download is exercised by registering a temporary "FAKE" genome whose
URL is a local ``file://`` pointing at a tiny gzipped FASTA, so the full
download -> verify -> decompress -> atomic-rename path runs in milliseconds.
"""

import gzip
import os
import stat

import pytest

import tessera.ref_genome as rg

FNAME = "test_genome.fna"
CONTENT = b">chrT test\n" + b"ACGT" * 64 + b"\n"  # small but a valid FASTA record


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Isolate all resolver locations to temp dirs and register a FAKE genome."""
    cache_root = tmp_path / "xdg"
    pkg_dir = tmp_path / "pkg_ref"
    pkg_dir.mkdir()
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache_root))
    monkeypatch.delenv("TESSERA_REF_GENOME_DIR", raising=False)
    monkeypatch.setattr(rg, "_PACKAGE_REF_DIR", str(pkg_dir))

    gz = tmp_path / "src.fna.gz"
    with gzip.open(gz, "wb") as f:
        f.write(CONTENT)
    # min_bytes=1 so the tiny content counts as "complete"
    monkeypatch.setitem(rg.GENOMES, "FAKE", (FNAME, gz.as_uri(), 1))

    return {
        "tmp": tmp_path,
        "cache": cache_root / "tessera" / "ref_genomes",
        "pkg": pkg_dir,
        "gz_uri": gz.as_uri(),
    }


def test_unsupported_version_raises():
    with pytest.raises(ValueError, match="Unsupported genome_version"):
        rg.resolved_fasta_path("nope")
    with pytest.raises(ValueError, match="Unsupported genome_version"):
        rg.ensure_reference_genome("nope")


def test_resolved_is_none_when_absent(env):
    assert rg.resolved_fasta_path("FAKE") is None


def test_resolver_precedence(env, tmp_path, monkeypatch):
    cache = env["cache"]
    cache.mkdir(parents=True)
    pkg = env["pkg"]

    # only the package dir has it (candidate #3)
    (pkg / FNAME).write_bytes(CONTENT)
    assert rg.resolved_fasta_path("FAKE") == str(pkg / FNAME)

    # user cache (candidate #2) beats package dir
    (cache / FNAME).write_bytes(CONTENT)
    assert rg.resolved_fasta_path("FAKE") == str(cache / FNAME)

    # env override (candidate #1) beats user cache
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    (env_dir / FNAME).write_bytes(CONTENT)
    monkeypatch.setenv("TESSERA_REF_GENOME_DIR", str(env_dir))
    assert rg.resolved_fasta_path("FAKE") == str(env_dir / FNAME)


def test_too_small_file_is_ignored(env, monkeypatch):
    # raise the size floor so the tiny cached file is treated as truncated
    monkeypatch.setitem(rg.GENOMES, "FAKE", (FNAME, env["gz_uri"], 10_000))
    cache = env["cache"]
    cache.mkdir(parents=True)
    (cache / FNAME).write_bytes(b"partial")
    assert rg.resolved_fasta_path("FAKE") is None


def test_ensure_downloads_then_is_idempotent(env, monkeypatch):
    path = rg.ensure_reference_genome("FAKE")
    assert path == str(env["cache"] / FNAME)
    with open(path, "rb") as f:
        assert f.read() == CONTENT
    assert not list(env["cache"].glob("*.part"))  # temps cleaned up

    # second call returns the cached file without re-downloading
    monkeypatch.setattr(
        rg, "_download_and_extract", lambda *a, **k: pytest.fail("re-downloaded")
    )
    assert rg.ensure_reference_genome("FAKE") == path


def test_decompressed_too_small_raises_and_cleans(env, monkeypatch):
    monkeypatch.setitem(rg.GENOMES, "FAKE", (FNAME, env["gz_uri"], 10_000))
    with pytest.raises(RuntimeError, match="implausibly small|corrupt|truncated"):
        rg.ensure_reference_genome("FAKE")
    cache = env["cache"]
    assert not (cache / FNAME).exists()
    assert not list(cache.glob("*.part"))


def test_bad_gzip_raises_and_cleans(env, tmp_path, monkeypatch):
    bad = tmp_path / "bad.gz"
    bad.write_bytes(b"this is not gzip")
    monkeypatch.setitem(rg.GENOMES, "FAKE", (FNAME, bad.as_uri(), 1))
    with pytest.raises(RuntimeError):
        rg.ensure_reference_genome("FAKE")
    cache = env["cache"]
    assert not (cache / FNAME).exists()
    assert not list(cache.glob("*.part"))


def test_open_fasta_routes_index_when_dir_readonly(env, tmp_path):
    ro_dir = tmp_path / "ro"
    ro_dir.mkdir()
    fasta = ro_dir / "small.fna"
    fasta.write_bytes(CONTENT)
    os.chmod(ro_dir, stat.S_IRUSR | stat.S_IXUSR)  # read-only directory
    try:
        fa = rg.open_fasta(str(fasta))
        assert str(fa["chrT"][:4]) == "ACGT"
        # index must NOT be written next to the read-only FASTA ...
        assert not (ro_dir / "small.fna.fai").exists()
        # ... it goes to the writable user cache instead
        assert (env["cache"] / "small.fna.fai").exists()
    finally:
        os.chmod(ro_dir, stat.S_IRWXU)  # restore so temp cleanup can remove it
