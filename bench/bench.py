"""Measured chain-only mappy subset benchmarks against upstream mappy."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "python"))
import mappy as mojo_mappy


def timed(fn, repeat: int = 3) -> float:
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def upstream_times(fasta: Path, query_file: Path) -> dict[str, float]:
    code = r"""import json,sys,time,mappy
fasta,query_file=sys.argv[1:]
query=open(query_file).read()
ref=''.join(open(fasta).read().splitlines()[1:])
def measure(fn):
    best=float('inf')
    for _ in range(3):
        start=time.perf_counter(); fn(); best=min(best,time.perf_counter()-start)
    return best
index = mappy.Aligner(fasta, k=15, w=10)
print(json.dumps({'index':measure(lambda:mappy.Aligner(seq=ref,k=15,w=10)),
'map':measure(lambda:list(index.map(query)))}))
"""
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    output = subprocess.check_output(
        ["pixi", "run", "python", "-c", code, str(fasta), str(query_file)],
        cwd="/tmp", env=env, text=True,
    )
    return json.loads(output)


def main() -> None:
    rng = np.random.default_rng(7)
    reference = "".join(rng.choice(np.array(list("ACGT")), size=500_000))
    query = reference[150_000:350_000]
    with tempfile.TemporaryDirectory() as directory:
        fasta = Path(directory) / "reference.fa"
        query_file = Path(directory) / "query.txt"
        fasta.write_text(f">reference\n{reference}\n")
        query_file.write_text(query)
        ours_index = timed(lambda: mojo_mappy.Aligner(seq=reference, k=15, w=10))
        index = mojo_mappy.Aligner(fasta, k=15, w=10)
        ours_map = timed(lambda: list(index.map(query)))
        theirs = upstream_times(fasta, query_file)
    print(f"machine: {platform.platform()} | Python {platform.python_version()}")
    print("| case | mojo-mappy | upstream mappy | ratio |")
    print("| --- | ---: | ---: | ---: |")
    cases = [
        ("index build (500 kb, k=15/w=10)", ours_index, theirs["index"]),
        ("exact-read map (200 kb)", ours_map, theirs["map"]),
    ]
    for name, ours, upstream_time in cases:
        print(f"| {name} | {ours * 1e3:.1f} ms | {upstream_time * 1e3:.1f} ms | {upstream_time / ours:.2f}x |")


if __name__ == "__main__":
    main()
