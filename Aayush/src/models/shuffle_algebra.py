"""Shuffle-product algebra on the truncated tensor algebra, needed for the
paper's Theorem 2.6 (CVaR-OCSVM via the expected signature).

A tensor-algebra element is represented as `levels`, a list where
`levels[k]` is a flat NumPy array of length `d**k` holding the order-k
tensor (levels[0] is always a length-1 array, the scalar/constant term).
This mirrors the representation used in `src/features/signatures.py`.

Shuffle product of an order-p and an order-q tensor produces an order-(p+q)
tensor: for basis words it is the sum, over all C(p+q, p) ways of
interleaving the two letter sequences while preserving each one's internal
order, of the corresponding term. This is exactly the algebraic identity
`<w1, S(x)><w2, S(x)> = <w1 shuffle w2, S(x)>` used by the paper.

Cost warning: the number of interleavings is C(p+q, p), and each one costs
O(d**(p+q)) to accumulate. This is fine at the small depths this project
uses for the CVaR-OCSVM (kept low deliberately, see cvar_ocsvm.py) but
grows fast -- this is not a general-purpose fast library implementation.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np


def shuffle_product_level(u: np.ndarray, p: int, v: np.ndarray, q: int, d: int) -> np.ndarray:
    """Shuffle product of one order-p tensor (flat, length d**p) and one
    order-q tensor (flat, length d**q). Returns a flat order-(p+q) tensor."""
    if p == 0:
        return u[0] * v
    if q == 0:
        return v[0] * u

    U = u.reshape((d,) * p)
    V = v.reshape((d,) * q)
    outer = np.multiply.outer(U, V)  # axes 0..p-1 are U's, axes p..p+q-1 are V's
    n = p + q
    result = np.zeros((d,) * n)
    for u_positions in combinations(range(n), p):
        u_pos_set = set(u_positions)
        v_positions = [i for i in range(n) if i not in u_pos_set]
        perm = [0] * n
        for i, pos in enumerate(u_positions):
            perm[pos] = i
        for j, pos in enumerate(v_positions):
            perm[pos] = p + j
        result += np.transpose(outer, axes=perm)
    return result.reshape(-1)


def shuffle_element(a_levels: list[np.ndarray], b_levels: list[np.ndarray], d: int) -> list[np.ndarray]:
    """Full shuffle product of two tensor-algebra elements (each a list of
    per-level flat arrays, level 0 first). Result has levels
    0 .. (len(a_levels) - 1) + (len(b_levels) - 1)."""
    max_a, max_b = len(a_levels) - 1, len(b_levels) - 1
    result = [np.zeros(d ** k) for k in range(max_a + max_b + 1)]
    for i in range(max_a + 1):
        for j in range(max_b + 1):
            result[i + j] += shuffle_product_level(a_levels[i], i, b_levels[j], j, d)
    return result


def identity_element(d: int) -> list[np.ndarray]:
    """The unit `1` of the tensor algebra: level 0 = 1, nothing else."""
    return [np.array([1.0])]


def shuffle_power(a_levels: list[np.ndarray], power: int, d: int) -> list[np.ndarray]:
    """a^{shuffle power}, i.e. a shuffled with itself `power` times."""
    if power == 0:
        return identity_element(d)
    result = a_levels
    for _ in range(power - 1):
        result = shuffle_element(result, a_levels, d)
    return result


def pair(a_levels: list[np.ndarray], b_levels: list[np.ndarray]) -> float:
    """The natural pairing <a, b> = sum_k <a_k, b_k> between two
    tensor-algebra elements truncated at the same max level (shorter one is
    treated as zero-padded)."""
    n = min(len(a_levels), len(b_levels))
    return float(sum(np.dot(a_levels[k], b_levels[k]) for k in range(n)))
