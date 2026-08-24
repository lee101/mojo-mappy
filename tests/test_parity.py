"""Parity tests for minimap2-compatible seeding and chain coordinates."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

import mappy as mm
import mappy.core as core
from mappy._lib import addr
from mappy.core import _Anchor


def reference_sketch(sequence: str, k: int, w: int):
    """Direct Python transcription of minimap2's non-HPC sketch.c loop."""
    table = {"A": 0, "C": 1, "G": 2, "T": 3, "U": 3}
    mask = (1 << (2 * k)) - 1

    def hash64(key):
        key = (~key + (key << 21)) & mask
        key ^= key >> 24
        key = ((key + (key << 3)) + (key << 8)) & mask
        key ^= key >> 14
        key = ((key + (key << 2)) + (key << 4)) & mask
        key ^= key >> 28
        return (key + (key << 31)) & mask

    ring = [(None, None)] * w
    forward = reverse = valid = ring_pos = min_pos = 0
    minimum = (None, None)
    output = []
    for i, char in enumerate(sequence.upper()):
        c = table.get(char, 4)
        info = (None, None)
        if c < 4:
            forward = ((forward << 2) | c) & mask
            reverse = (reverse >> 2) | ((3 ^ c) << (2 * (k - 1)))
            if forward == reverse:
                continue
            strand = 0 if forward < reverse else 1
            valid += 1
            if valid >= k:
                info = (hash64(forward if strand == 0 else reverse), (i, strand))
        else:
            valid = 0
        ring[ring_pos] = info
        if valid == w + k - 1 and minimum[0] is not None:
            for entry in ring[ring_pos + 1:] + ring[:ring_pos]:
                if entry[0] == minimum[0] and entry[1] != minimum[1]:
                    output.append(entry)
        if minimum[0] is None or info[0] is not None and info[0] <= minimum[0]:
            if valid >= w + k and minimum[0] is not None:
                output.append(minimum)
            minimum, min_pos = info, ring_pos
        elif ring_pos == min_pos:
            if valid >= w + k - 1 and minimum[0] is not None:
                output.append(minimum)
            candidates = ring[ring_pos + 1:] + ring[:ring_pos + 1]
            minimum = (None, None)
            for offset, entry in enumerate(candidates):
                if entry[0] is not None and (minimum[0] is None or minimum[0] >= entry[0]):
                    minimum = entry
                    min_pos = (ring_pos + 1 + offset) % w
            if valid >= w + k - 1 and minimum[0] is not None:
                for entry in candidates:
                    if entry[0] == minimum[0] and entry[1] != minimum[1]:
                        output.append(entry)
        ring_pos = (ring_pos + 1) % w
    if minimum[0] is not None:
        output.append(minimum)
    return [(value, position, strand, k) for value, (position, strand) in output]


def upstream(code: str) -> str:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    return subprocess.check_output(
        ["pixi", "run", "python", "-c", code], cwd="/tmp", env=env, text=True
    )


@pytest.mark.parametrize("sequence,k,w", [
    ("ACGTTGCATGTCGCATGATGCATGAGAGCT", 5, 4),
    ("ACGTNNACGTTTACGTACGTNNNACGTAC", 4, 3),
    ("A" * 20 + "CGTACGTTGCA" * 4, 7, 5),
])
def test_sketch_matches_reference(sequence, k, w):
    expected = reference_sketch(sequence, k, w)
    observed = [(x.hash, x.pos, x.strand, x.span) for x in mm.sketch(sequence, k, w)]
    assert observed == expected


def test_chain_matches_minimap2_recurrence():
    anchors = [(14, 14, 15), (28, 28, 15), (42, 43, 15), (70, 70, 15), (84, 84, 15)]
    score, path = mm.chain(anchors, max_gap=100, bandwidth=20, max_skip=25)
    assert score == 72
    assert path == [0, 1, 2, 3, 4]


def test_chain_accepts_its_documented_anchor_record():
    score, path = mm.chain([_Anchor(14, 14, 15)])
    assert (score, path) == (15, [0])


def test_exact_mapping_coordinates_agree_with_upstream(tmp_path: Path):
    rng = np.random.default_rng(31)
    reference = "".join(rng.choice(np.array(list("ACGT")), size=12_000))
    query = reference[3_000:9_000]
    fasta = tmp_path / "reference.fa"
    fasta.write_text(f">reference\n{reference}\n")
    code = (
        "import json,mappy; a=mappy.Aligner(" + repr(str(fasta)) + ", k=15, w=10); "
        "h=next(a.map(" + repr(query) + ")); "
        "print(json.dumps([h.ctg,h.r_st,h.r_en,h.q_st,h.q_en,h.strand]))"
    )
    expected = json.loads(upstream(code))
    observed = next(mm.Aligner(fasta, k=15, w=10).map(query))
    assert [observed.ctg, observed.strand] == [expected[0], expected[5]]
    assert expected[1] <= observed.r_st <= expected[1] + 15
    assert expected[2] - 15 <= observed.r_en <= expected[2]
    assert expected[3] <= observed.q_st <= expected[3] + 15
    assert expected[4] - 15 <= observed.q_en <= expected[4]


def test_reverse_mapping_coordinates_agree_with_upstream(tmp_path: Path):
    rng = np.random.default_rng(47)
    reference = "".join(rng.choice(np.array(list("ACGT")), size=12_000))
    query = mm.revcomp(reference[3_000:9_000])
    fasta = tmp_path / "reference.fa"
    fasta.write_text(f">reference\n{reference}\n")
    code = (
        "import json,mappy; a=mappy.Aligner(" + repr(str(fasta)) + ", k=15, w=10); "
        "h=next(a.map(" + repr(query) + ")); "
        "print(json.dumps([h.ctg,h.r_st,h.r_en,h.q_st,h.q_en,h.strand]))"
    )
    expected = json.loads(upstream(code))
    observed = next(mm.Aligner(fasta, k=15, w=10).map(query))
    assert [observed.ctg, observed.strand] == [expected[0], expected[5]]
    assert expected[1] <= observed.r_st <= expected[1] + 15
    assert expected[2] - 15 <= observed.r_en <= expected[2]
    assert expected[3] <= observed.q_st <= expected[3] + 15
    assert expected[4] - 15 <= observed.q_en <= expected[4]


def test_revcomp_and_fastx_match_upstream(tmp_path: Path):
    sequence = "ACGTNRYKMSWBDHVUacgtnrykmswbdhvu"
    actual = upstream("import mappy; print(mappy.revcomp(" + repr(sequence) + "))").strip()
    assert mm.revcomp(sequence) == actual
    fasta = tmp_path / "reads.fa"
    fasta.write_text(">a comment\nACG\nTT\n>b\nGGG\n")
    assert list(mm.fastx_read(fasta, read_comment=True)) == [
        ("a", "ACGTT", None, "comment"), ("b", "GGG", None, None)
    ]


def test_public_aligner_surface(tmp_path: Path):
    fasta = tmp_path / "reference.fa"
    fasta.write_text(">chr1\nACGTACGTACGTACGTACGTACGTACGT\n")
    aligner = mm.Aligner(fasta, k=5, w=3, min_cnt=1, min_chain_score=1)
    assert aligner and aligner.n_seq == 1 and aligner.seq_names == ["chr1"]
    assert aligner.seq("chr1", 2, 8) == "GTACGT"
    assert aligner.seq("missing") is None
    hit = next(aligner.map("ACGTACGTACGTACGTACGT"))
    assert hit.ctg == "chr1" and hit.ctg_len == 28
    assert 0 <= hit.q_st < hit.q_en <= 20
    assert 0 <= hit.r_st < hit.r_en <= hit.ctg_len
    assert hit.strand in (-1, 1) and hit.is_primary
    assert hit.mapq == 60 and hit.seed_count >= 1
    assert hit.mlen >= 1 and hit.blen >= hit.mlen
    assert hit.NM == 0 and hit.trans_strand == 0 and hit.read_num == 1
    assert "tp:A:P" in str(hit)
    assert hit.cigar == [] and hit.cigar_str == hit.cs == hit.ds == hit.MD == ""


def test_index_handles_singleton_and_repeated_minimizers():
    sequence = "ACGTTGCATGTCGCATGATGCATGAGAGCT"
    aligner = mm.Aligner(seq=sequence, k=5, w=4, min_cnt=1, min_chain_score=1)
    assert any(isinstance(value, tuple) for value in aligner._index.values())
    assert any(isinstance(value, list) for value in aligner._index.values())
    hit = next(aligner.map(sequence))
    assert (hit.q_st, hit.q_en) == (hit.r_st, hit.r_en)
    assert 0 <= hit.q_st < hit.q_en <= len(sequence)


def test_parallel_reference_sketch_matches_serial(tmp_path: Path, monkeypatch):
    fasta = tmp_path / "references.fa"
    fasta.write_text(">a\n" + "ACGTTGCA" * 100 + "\n>b\n" + "TGCACGTT" * 100 + "\n")
    monkeypatch.setattr(core, "_PARALLEL_SKETCH_THRESHOLD", 1)
    serial = mm.Aligner(fasta, k=7, w=5, n_threads=1)
    parallel = mm.Aligner(fasta, k=7, w=5, n_threads=2)
    assert parallel._index == serial._index


def test_vectorized_seed_expansion_handles_remainder_and_no_matches():
    rng = np.random.default_rng(59)
    reference = "".join(rng.choice(np.array(list("ACGT")), size=1021))
    aligner = mm.Aligner(seq=reference, k=7, w=5, min_cnt=1, min_chain_score=1)
    query = reference[317:430]
    hit = next(aligner.map(query))
    assert hit.strand == 1
    assert 317 <= hit.r_st <= 324
    assert 423 <= hit.r_en <= 430
    assert list(aligner.map("N" * 113)) == []


def test_ffi_rejects_wrong_dtype_and_non_writable_outputs():
    with pytest.raises(TypeError, match="uint8"):
        addr(np.ones(2, dtype=np.int64), np.uint8)
    output = np.ones(2, dtype=np.int64)
    output.flags.writeable = False
    with pytest.raises(ValueError, match="writable"):
        addr(output, np.int64, writable=True)


def test_integer_parameters_do_not_silently_narrow():
    with pytest.raises(TypeError, match="k must be an integer"):
        mm.sketch("ACGT", k=3.5)
    with pytest.raises(TypeError, match="anchor coordinates"):
        mm.chain([(1.5, 2, 3)])
