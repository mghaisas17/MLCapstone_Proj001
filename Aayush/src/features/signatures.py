"""Truncated path signatures via Chen's identity, in pure NumPy.

This machine's C++ toolchain can't build iisignature/esig/pySigLib (missing
standard library headers in the installed Command Line Tools), so
signatures are computed directly here instead of via a native extension.
The algorithm is exactly what those libraries do internally, just without
the C++ speed-up -- entirely adequate at the depths (3-4) and channel
counts (2-4) this project uses.

Math: for a piecewise-linear path with increments delta_1, ..., delta_L in
R^d, the signature of one linear segment with increment `delta`, truncated
at level N, is exp(delta) in the tensor algebra:
    seg_k = delta^{tensor k} / k!   for k = 0..N   (seg_0 = 1)
Chen's identity says the signature of the concatenated path is the tensor
algebra *product* of each segment's signature, level-truncated after every
step:
    (A * B)_k = sum_{i=0}^{k} A_i (tensor) B_{k-i}
Accumulating this product one increment at a time gives the full path's
truncated signature.
"""
from __future__ import annotations

import numpy as np


def _segment_exp(delta: np.ndarray, depth: int) -> list[np.ndarray]:
    """[delta^{tensor k} / k! for k = 0..depth], each level k flattened to
    shape (d**k,)."""
    d = delta.shape[0]
    levels = [np.array([1.0])]
    cur = np.array([1.0])
    for k in range(1, depth + 1):
        cur = np.outer(cur, delta).reshape(-1) / k
        levels.append(cur)
    return levels


def _tensor_multiply(a: list[np.ndarray], b: list[np.ndarray], depth: int) -> list[np.ndarray]:
    """Truncated tensor-algebra product of two elements represented as
    per-level flattened arrays [level0, level1, ..., levelN]."""
    result = []
    for k in range(depth + 1):
        # sum_{i=0}^{k} a_i (tensor) b_{k-i}, flattened to shape (d**k,)
        pieces = [np.outer(a[i], b[k - i]).reshape(-1) for i in range(k + 1)]
        result.append(np.sum(pieces, axis=0))
    return result


def signature(path: np.ndarray, depth: int, include_level_zero: bool = False) -> np.ndarray:
    """Truncated signature of a piecewise-linear path.

    Parameters
    ----------
    path : (L, d) array of points defining the piecewise-linear path.
    depth : truncation level N.
    include_level_zero : if True, prepend the constant 1 (level-0 term).

    Returns
    -------
    Flat array concatenating levels 1..depth (or 0..depth), length
    sum_{k=1}^{N} d**k (or +1).
    """
    if path.ndim != 2:
        raise ValueError("path must be a 2D (L, d) array")
    L, d = path.shape
    if L < 2:
        raise ValueError("path must have at least 2 points to define an increment")

    increments = np.diff(path, axis=0)
    running = [np.array([1.0])] + [np.zeros(d ** k) for k in range(1, depth + 1)]
    for inc in increments:
        seg = _segment_exp(inc, depth)
        running = _tensor_multiply(running, seg, depth)

    levels = running if include_level_zero else running[1:]
    return np.concatenate(levels)


def signature_dim(d: int, depth: int, include_level_zero: bool = False) -> int:
    start = 0 if include_level_zero else 1
    return sum(d ** k for k in range(start, depth + 1))


def batch_signatures(paths: list[np.ndarray], depth: int) -> np.ndarray:
    """Signature feature matrix (n_windows, feature_dim) for a list of
    equal-channel-count paths (lengths may vary)."""
    return np.vstack([signature(p, depth) for p in paths])
