"""A Mojo-backed, chain-only subset of minimap2's :mod:`mappy` API."""

from .core import Alignment, Aligner, Minimizer, chain, fastx_read, revcomp, sketch

__version__ = "0.1.0"
__all__ = ["Alignment", "Aligner", "Minimizer", "chain", "fastx_read", "revcomp", "sketch"]
