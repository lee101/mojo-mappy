"""ctypes bindings for the Mojo seeding and chaining kernels."""

from __future__ import annotations

import ctypes
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.path.join(ROOT, "dist", "libmojo-mappy.so")
I = ctypes.c_int64

_SIGNATURES = {
    "mmp_sketch": ([I] * 10, I),
    "mmp_chain_dp": ([I] * 10, I),
}

_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        if not os.path.exists(LIB):
            raise RuntimeError("Mojo library not built; run `pixi run build`")
        _library = ctypes.CDLL(LIB)
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_library, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _library


def addr(array: np.ndarray, dtype: np.dtype, *, writable: bool = False) -> int:
    """Return a checked address for a live, one-dimensional NumPy buffer."""
    if not isinstance(array, np.ndarray) or array.ndim != 1 or not array.flags.c_contiguous:
        raise TypeError("FFI buffers must be one-dimensional contiguous NumPy arrays")
    if array.dtype != np.dtype(dtype):
        raise TypeError(f"FFI buffer must have dtype {np.dtype(dtype)}")
    if not array.size:
        raise ValueError("FFI buffers must be non-empty")
    if writable and not array.flags.writeable:
        raise ValueError("FFI output buffers must be writable")
    return int(array.ctypes.data)
