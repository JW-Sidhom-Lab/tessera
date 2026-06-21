"""Tests for coordinate liftover — in particular, dropping coordinates that lift onto
non-canonical contigs absent from TESSERA's GRCh37 reference (the crash reported on
BostonGene/GRCh38 input), rather than passing them through to the FASTA lookup.

Uses a fake LiftOver injected into the module cache so no chain download / network is
needed.
"""

import pandas as pd
import pytest

import tessera.data.liftover as lo
from tessera.data.liftover import lift_cna, lift_snv


class _FakeLift:
    # chr1 -> chr1 (canonical, kept); chr2 -> chrUn_gl000211 (non-canonical, dropped);
    # anything else -> [] (unliftable, dropped).
    def convert_coordinate(self, chrom, pos):
        if chrom == "chr1":
            return [("chr1", pos + 1000, "+", 0)]
        if chrom == "chr2":
            return [("chrUn_gl000211", pos + 5, "+", 0)]
        return []


@pytest.fixture
def fake_lifter():
    key = ("hg38", "hg19", None)
    lo._LIFTERS[key] = _FakeLift()
    yield
    lo._LIFTERS.pop(key, None)


def test_snv_drops_noncanonical_and_unliftable(fake_lifter):
    df = pd.DataFrame({"Tumor_Sample_Barcode": ["S", "S", "S"],
                       "Chromosome": ["1", "2", "3"], "Start_Position": [100, 200, 300]})
    out, st = lift_snv(df, from_assembly="GRCh38")
    assert list(out["Chromosome"]) == ["1"]                 # chr2 (alt contig), chr3 (unliftable) gone
    assert int(out["Start_Position"].iloc[0]) == 1100        # 100 -> 0-based 99 -> +1000 -> +1
    assert st["n_in"] == 3 and st["n_out"] == 1
    assert st["n_dropped"] == 2 and st["n_noncanonical"] == 1


def test_cna_drops_noncanonical(fake_lifter):
    df = pd.DataFrame({"Tumor_Sample_Barcode": ["S", "S"], "Chromosome": ["1", "2"],
                       "Start": [100, 200], "End": [150, 250], "Segment_Mean": [0.1, 0.2]})
    out, st = lift_cna(df, from_assembly="GRCh38")
    assert list(out["Chromosome"]) == ["1"]
    assert st["n_out"] == 1 and st["n_noncanonical"] == 1


def test_grch37_is_noop():
    df = pd.DataFrame({"Chromosome": ["1", "Un_gl000211"], "Start_Position": [100, 200]})
    out, st = lift_snv(df, from_assembly="GRCh37")
    assert len(out) == 2 and st["n_dropped"] == 0 and st["n_noncanonical"] == 0
