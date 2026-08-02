"""Minimap2-compatible seed construction and chain reporting."""

from __future__ import annotations

from collections import defaultdict, namedtuple
from dataclasses import dataclass
import gzip
import math
import operator
import os
from typing import Iterator

import numpy as np

from ._lib import addr, lib

Minimizer = namedtuple("Minimizer", "hash pos strand span")


def _integer(value, name: str) -> int:
    """Accept integer-like API values without silently truncating floats."""
    try:
        return operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error

_COMPLEMENT = str.maketrans(
    "ACGTRYKMSWBDHVNUacgtrykmswbdhvnu",
    "TGCAYRMKSWVHDBNA tgcayrmkswvhdbna".replace(" ", ""),
)


def revcomp(seq: str | bytes) -> str | bytes:
    """Return the IUB-aware reverse complement, matching ``mappy.revcomp``."""
    if isinstance(seq, bytes):
        return revcomp(seq.decode()).encode()
    return seq.translate(_COMPLEMENT)[::-1]


def _sketch_arrays(seq: str | bytes, k: int, w: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k = _integer(k, "k")
    w = _integer(w, "w")
    if not 1 <= k <= 28:
        raise ValueError("k must be in 1..28")
    if not 1 <= w <= 255:
        raise ValueError("w must be in 1..255")
    raw = seq.encode() if isinstance(seq, str) else bytes(seq)
    if not raw:
        empty = np.empty(0, dtype=np.int64)
        return np.empty(0, dtype=np.uint64), empty, empty
    text = np.frombuffer(raw, dtype=np.uint8)
    n = len(text)
    hashes = np.empty(n, dtype=np.uint64)
    positions = np.empty(n, dtype=np.int64)
    strands = np.empty(n, dtype=np.int64)
    scratch_hash = np.empty(w, dtype=np.uint64)
    scratch_meta = np.empty(w, dtype=np.uint64)
    count = lib().mmp_sketch(
        addr(text, np.uint8), n, k, w, addr(hashes, np.uint64, writable=True),
        addr(positions, np.int64, writable=True), addr(strands, np.int64, writable=True),
        addr(scratch_hash, np.uint64, writable=True), addr(scratch_meta, np.uint64, writable=True), n,
    )
    if not 0 <= count <= n:
        raise RuntimeError("Mojo sketch kernel returned an invalid output length")
    return hashes[:count], positions[:count], strands[:count]


def sketch(seq: str | bytes, k: int = 15, w: int = 10) -> list[Minimizer]:
    """Return minimap2's non-HPC canonical ``(w, k)`` minimizers."""
    hashes, positions, strands = _sketch_arrays(seq, k, w)
    return [Minimizer(int(h), int(p), int(s), k) for h, p, s in zip(hashes, positions, strands)]


@dataclass(frozen=True)
class _Anchor:
    ref_pos: int
    query_pos: int
    span: int


def chain(
    anchors: list[tuple[int, int, int]] | list[_Anchor],
    max_gap: int = 5000,
    bandwidth: int = 500,
    max_skip: int = 25,
    gap_scale: float = 0.8,
) -> tuple[int, list[int]]:
    """Score sorted anchors with minimap2's long-chain DP recurrence.

    ``anchors`` are ``(reference_end, query_end, kmer_span)`` records, sorted
    by reference coordinate. The return value is ``(score, anchor_indices)``.
    """
    if not anchors:
        return 0, []
    values = [
        (a.ref_pos, a.query_pos, a.span) if isinstance(a, _Anchor) else tuple(a)
        for a in anchors
    ]
    if any(len(a) < 3 for a in values):
        raise ValueError("anchors must contain reference_end, query_end, and kmer_span")
    values = [tuple(_integer(value, "anchor coordinates") for value in a[:3]) for a in values]
    if any(reference_end < 0 or query_end < 0 or span <= 0
           for reference_end, query_end, span in values):
        raise ValueError("anchor coordinates must be non-negative and spans positive")
    reference = np.asarray([a[0] for a in values], dtype=np.int64)
    query = np.asarray([a[1] for a in values], dtype=np.int64)
    spans = np.asarray([a[2] for a in values], dtype=np.int64)
    if np.any(reference[1:] < reference[:-1]):
        raise ValueError("anchors must be sorted by reference coordinate")
    max_gap = _integer(max_gap, "max_gap")
    bandwidth = _integer(bandwidth, "bandwidth")
    max_skip = _integer(max_skip, "max_skip")
    if max_gap < 0 or bandwidth < 0 or max_skip < 0 or not math.isfinite(gap_scale) or gap_scale < 0:
        raise ValueError("chain parameters must be non-negative")
    scores = np.empty(len(values), dtype=np.int64)
    predecessors = np.empty(len(values), dtype=np.int64)
    last = lib().mmp_chain_dp(
        addr(reference, np.int64), addr(query, np.int64), addr(spans, np.int64), len(values), max_gap,
        bandwidth, max_skip, int(round(gap_scale * int(spans.max()))),
        addr(scores, np.int64, writable=True), addr(predecessors, np.int64, writable=True),
    )
    if not 0 <= last < len(values):
        raise RuntimeError("Mojo chaining kernel returned an invalid predecessor index")
    path: list[int] = []
    while last >= 0:
        path.append(int(last))
        last = int(predecessors[last])
    path.reverse()
    return int(scores[path[-1]]), path


class Alignment:
    """A seed-chain alignment with mappy's public coordinate properties."""

    def __init__(self, ctg: str, ctg_len: int, q_st: int, q_en: int, r_st: int,
                 r_en: int, strand: int, score: int, count: int, primary: bool) -> None:
        self._ctg, self._ctg_len = ctg, ctg_len
        self._q_st, self._q_en, self._r_st, self._r_en = q_st, q_en, r_st, r_en
        self._strand, self._score, self._count, self._primary = strand, score, count, primary

    @property
    def ctg(self): return self._ctg
    @property
    def ctg_len(self): return self._ctg_len
    @property
    def q_st(self): return self._q_st
    @property
    def q_en(self): return self._q_en
    @property
    def r_st(self): return self._r_st
    @property
    def r_en(self): return self._r_en
    @property
    def strand(self): return self._strand
    @property
    def mapq(self): return 60 if self._primary else 0
    @property
    def mlen(self): return self._score
    @property
    def blen(self): return max(self._q_en - self._q_st, self._r_en - self._r_st)
    @property
    def NM(self): return 0
    @property
    def trans_strand(self): return 0
    @property
    def is_primary(self): return self._primary
    @property
    def read_num(self): return 1
    @property
    def cigar(self): return []
    @property
    def cigar_str(self): return ""
    @property
    def cs(self): return ""
    @property
    def ds(self): return ""
    @property
    def MD(self): return ""
    @property
    def seed_count(self): return self._count

    def __str__(self) -> str:
        strand = "+" if self._strand > 0 else "-"
        return "\t".join(map(str, (self._q_st, self._q_en, strand, self._ctg,
            self._ctg_len, self._r_st, self._r_en, self.mlen, self.blen,
            self.mapq, "tp:A:P" if self._primary else "tp:A:S", "ts:A:.")))


class Aligner:
    """In-memory FASTA/single-sequence minimizer index with mappy's signature."""

    def __init__(self, fn_idx_in=None, preset=None, k=None, w=None, min_cnt=None,
                 min_chain_score=None, min_dp_score=None, bw=None, bw_long=None,
                 best_n=None, n_threads=3, fn_idx_out=None, max_frag_len=None,
                 extra_flags=None, seq=None, scoring=None, sc_ambi=None,
                 max_chain_skip=None):
        del min_dp_score, bw_long, n_threads, fn_idx_out, max_frag_len, extra_flags, scoring, sc_ambi
        defaults = {"sr": (21, 11), "map-pb": (19, 10), "map-ont": (15, 10),
                    "asm5": (19, 19), "asm10": (19, 19), "splice": (15, 5)}
        if preset is not None and preset not in defaults:
            raise ValueError(f"unsupported preset {preset!r}")
        pk, pw = defaults.get(preset, (15, 10))
        self.k = _integer(pk if k is None else k, "k")
        self.w = _integer(pw if w is None else w, "w")
        self.min_cnt = _integer(3 if min_cnt is None else min_cnt, "min_cnt")
        self.min_chain_score = _integer(40 if min_chain_score is None else min_chain_score, "min_chain_score")
        self.bw = _integer(500 if bw is None else bw, "bw")
        self.best_n = _integer(5 if best_n is None else best_n, "best_n")
        self.max_chain_skip = _integer(25 if max_chain_skip is None else max_chain_skip, "max_chain_skip")
        if not 1 <= self.k <= 28 or not 1 <= self.w <= 255:
            raise ValueError("k must be in 1..28 and w must be in 1..255")
        if self.min_cnt < 0 or self.min_chain_score < 0 or self.bw < 0 or self.best_n < 0 or self.max_chain_skip < 0:
            raise ValueError("mapping thresholds must be non-negative")
        self._references = self._load_references(fn_idx_in, seq)
        self._index: dict[int, tuple[int, int, int] | list[tuple[int, int, int]]] = {}
        for rid, (_, reference) in enumerate(self._references):
            hashes, positions, strands = _sketch_arrays(reference, self.k, self.w)
            for seed_hash, position, strand in zip(hashes.tolist(), positions.tolist(), strands.tolist()):
                hit = (rid, position, strand)
                previous = self._index.get(seed_hash)
                if previous is None:
                    self._index[seed_hash] = hit
                elif isinstance(previous, tuple):
                    self._index[seed_hash] = [previous, hit]
                else:
                    previous.append(hit)

    @staticmethod
    def _load_references(fn_idx_in, seq):
        if seq is not None:
            return [("N/A", str(seq))]
        if fn_idx_in is None:
            return []
        records = [(name, sequence) for name, sequence, _ in fastx_read(fn_idx_in)]
        return records

    def __bool__(self):
        return bool(self._references)

    @property
    def n_seq(self): return len(self._references)
    @property
    def seq_names(self): return [name for name, _ in self._references]

    def seq(self, name: str, start: int = 0, end: int = 0x7FFFFFFF):
        for ref_name, reference in self._references:
            if ref_name == name and 0 <= start <= end <= len(reference):
                return reference[start:end]
        return None

    def map(self, seq, seq2=None, name=None, buf=None, cs=False, ds=False, MD=False,
            max_frag_len=None, extra_flags=None):
        del name, buf, cs, ds, MD, max_frag_len, extra_flags
        if seq2 is not None:
            raise NotImplementedError("paired-end mapping is outside the chaining subset")
        query = seq.decode() if isinstance(seq, bytes) else str(seq)
        grouped: dict[tuple[int, int], list[tuple[int, int, int, int]]] = defaultdict(list)
        for qseed in sketch(query, self.k, self.w):
            matches = self._index.get(qseed.hash)
            if matches is None:
                continue
            if isinstance(matches, tuple):
                matches = (matches,)
            for rid, rpos, rstrand in matches:
                strand = 1 if rstrand == qseed.strand else -1
                # The DP requires increasing coordinates.  For reverse hits use
                # the query's reverse-complement k-mer end coordinate, while
                # retaining the original coordinate for the public result.
                chain_qpos = qseed.pos if strand > 0 else len(query) - qseed.pos + self.k - 1
                grouped[(rid, strand)].append((rpos, chain_qpos, self.k, qseed.pos))
        hits = []
        for (rid, strand), anchors in grouped.items():
            anchors.sort()
            score, path = chain(anchors, bandwidth=self.bw, max_skip=self.max_chain_skip)
            if len(path) < self.min_cnt or score < self.min_chain_score:
                continue
            selected = [anchors[i] for i in path]
            r_st, r_en = selected[0][0] - self.k + 1, selected[-1][0] + 1
            q_positions = [a[3] for a in selected]
            q_st, q_en = min(q_positions) - self.k + 1, max(q_positions) + 1
            ctg, reference = self._references[rid]
            hits.append(Alignment(ctg, len(reference), max(0, q_st), q_en,
                                  max(0, r_st), r_en, strand, score, len(path), True))
        hits.sort(key=lambda hit: hit.mlen, reverse=True)
        for i, hit in enumerate(hits[:self.best_n]):
            hit._primary = i == 0
            yield hit


def fastx_read(fn, read_comment: bool = False) -> Iterator[tuple]:
    """Yield FASTA/FASTQ records in the same tuple shapes as ``mappy.fastx_read``."""
    opener = gzip.open if os.fspath(fn).endswith(".gz") else open
    with opener(fn, "rt") as handle:
        line = handle.readline()
        while line:
            line = line.rstrip("\r\n")
            if not line:
                line = handle.readline()
                continue
            if line[0] not in ">@":
                raise ValueError("expected FASTA/FASTQ header")
            header = line[1:]
            fields = header.split(maxsplit=1)
            name, comment = fields[0], fields[1] if len(fields) > 1 else None
            if line[0] == "@":
                sequence = handle.readline().strip()
                plus = handle.readline()
                if not plus.startswith("+"):
                    raise ValueError("expected FASTQ '+' separator")
                quality = handle.readline().strip()
                line = handle.readline()
            else:
                chunks: list[str] = []
                line = handle.readline()
                while line and not line.startswith(">"):
                    if line.startswith("@"):
                        raise ValueError("FASTA sequence cannot contain a FASTQ header")
                    chunks.append(line.strip())
                    line = handle.readline()
                sequence, quality = "".join(chunks), None
            yield (name, sequence, quality, comment) if read_comment else (name, sequence, quality)
