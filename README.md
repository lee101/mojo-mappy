# mojo-mappy

`mojo-mappy` is a standalone Mojo port of the compute-bound seeding and
chaining core of [mappy/minimap2](https://github.com/lh3/minimap2). It builds a
canonical DNA minimizer index, expands query minimizers into seed hits, and
uses minimap2's long-chain dynamic-programming recurrence to report coarse
mappings. The Python package is named `mappy` so code using the covered API
can change the import without changing call sites.

This is intentionally a chain-only mapper. It does not include minimap2's
base-level alignment extension, splice logic, RMQ rechaining, `.mmi` indexes,
homopolymer-compressed minimizers, paired-end mapping, or SAM/PAF emission.
Consequently `Alignment.cigar`, `cigar_str`, `cs`, `MD`, and `NM` are empty or
seed-derived rather than base-alignment results. Use upstream `mappy` when
those features are required.

## Install

```bash
pixi install
pixi run build
pixi run test
```

The test environment installs upstream `mappy==2.31` from PyPI for parity
checks. `pixi run test` also builds the shared library, so it is safe to run on
a fresh checkout.

## Usage

```python
import mappy as mp

aligner = mp.Aligner(seq="TTTACGTTGCATGTCGCATGATGCATGAGAGCTAAA", k=5, w=4,
                     min_cnt=1, min_chain_score=1)
for hit in aligner.map("ACGTTGCATGTCGCATGATGCATGAGAGCT"):
    print(hit.ctg, hit.r_st, hit.r_en, hit.q_st, hit.q_en, hit.strand)

print(mp.revcomp("ACGTN"))
```

Covered: an in-memory `Aligner` built from a FASTA/FASTQ path or `seq=`, its
single-query `map(seq)` chain results, `seq()`, `k`, `w`, `n_seq`, `seq_names`,
the documented `Alignment` coordinate and seed-chain properties, `revcomp()`,
`fastx_read()`, plus `sketch(seq, k=15, w=10)` and `chain(anchors, ...)`.

Not covered: `.mmi` input/output, paired reads (`seq2`), base-level alignment
or its derived fields, splice/RMQ/HPC modes, SAM/PAF output, and mapping-option
arguments other than `bw`, `best_n`, and `max_chain_skip`. Unsupported paired
reads fail explicitly; ignored compatibility arguments do not enable upstream
features.

## Performance

Measured with `pixi run bench` on this machine. Each result is the fastest of
three runs; mapping measures an already-built index for both implementations.
The ratio is upstream elapsed time divided by mojo-mappy elapsed time.

| case | mojo-mappy | upstream mappy | ratio |
| --- | ---: | ---: | ---: |
| index build (500 kb, k=15/w=10) | 24.1 ms | 30.2 ms | 1.25x |
| exact-read map (200 kb) | 27.5 ms | 63.7 ms | 2.32x |

The reference index keeps minimizer fields in sorted contiguous NumPy buffers.
Query lookup and repeated-hit expansion use vectorized array operations, while
the Python dictionary-shaped debug view is created only if `_index` is
inspected. Large multi-reference inputs sketch independent references
concurrently; inputs below one million total bases stay serial to avoid thread
launch overhead. Mapping includes upstream's base-level alignment, which this
chain-only port deliberately does not perform.

The array lookup, expansion, grouping, and coordinate reductions run in
NumPy's native vectorized kernels. The Mojo sketch and chain loops remain
scalar: sketching is a rolling state machine and the chain recurrence depends
on prior scores and skip state, so forcing either across SIMD lanes would
change ordering semantics or add a losing staging pass. No GPU path is used.
Both kernels perform integer and control-flow work with low useful arithmetic
intensity; host/device transfers and divergent execution would dominate, so a
GPU path is not justified.

## How it works

```text
NumPy uint8 sequence buffer
        | address passed as Int through ctypes
        v
src/capi.mojo: canonical (w,k) minimizer sketch
        | uint64 hashes and int64 positions
        v
Python: reference hash index and seed expansion
        | sorted int64 anchor buffers
        v
src/capi.mojo: minimap2 long-chain DP recurrence
```

The shared library is `dist/libmojo-mappy.so`, produced by
`mojo build --emit shared-lib`. Buffers cross the C ABI as integer addresses,
then Mojo reconstructs `UnsafePointer[..., AnyOrigin[mut=True]]`; no pointer
origin is exported. The kernels allocate nothing. Python owns all contiguous
NumPy output and scratch arrays, so lifetimes and memory layout stay explicit.

## Validation

`tests/test_parity.py` checks the Mojo sketch against a direct transcription
of minimap2's `sketch.c`, validates the chain recurrence, compares exact-read
mapping coordinates and IUB reverse complements with upstream PyPI `mappy`,
and covers FASTA parsing plus the public `Aligner` surface.

## License

MIT
