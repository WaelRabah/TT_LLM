"""Dimension factorization and TT weight-decomposition utilities.

- :func:`factorize_dim` decomposes an integer feature size into ``num_factors``
  balanced factors whose product equals the original dim. (Bug-fixed: the
  notebook version chose the largest divisor below the geometric mean, leaving a
  large leftover final factor and inflating parameter count; this version picks
  the divisor *closest* to the geometric-mean target at each step so the cores
  stay balanced.)

- :func:`tt_svd` performs the classic Tensor-Train SVD (TT-SVD) decomposition of
  a 2D weight matrix into ``d`` TT cores. Each core has shape
  ``[r_{k-1}, i_k, o_k, r_k]``; boundary ranks equal 1.

- :func:`svd` performs a *global* low-rank SVD of the weight matrix
  (``W ~= U_r S_r V_r^T``) and then formats the rank-``r`` approximation as
  ``d`` TT cores via :func:`tt_svd`. This gives the best Frobenius-norm rank-``r``
  approximation of ``W`` packaged as a TT chain — distinct from
  :func:`tt_svd`, which performs locally-optimal per-unfolding truncation.

- :func:`reconstruct_matrix` contracts TT cores back into a 2D matrix
  ``[out_features, in_features]`` (used by tests and weight loading).
"""

from __future__ import annotations

import math
from typing import List, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Dimension factorization
# ---------------------------------------------------------------------------
def factorize_dim(dim: int, num_factors: int = 3) -> List[int]:
    """Decompose ``dim`` into exactly ``num_factors`` balanced factors.

    At each step, picks the divisor of the remaining value that is closest to
    the geometric-mean target ``val ** (1/(remaining+1))``. Searches all
    divisors (not just a narrow window) so that the result never degenerates
    to ``1`` when a balanced non-trivial factorisation exists.
    """
    if num_factors < 1:
        raise ValueError("num_factors must be >= 1")
    if dim < 1:
        raise ValueError("dim must be >= 1")
    if num_factors == 1:
        return [int(dim)]

    def _divisors(n: int) -> List[int]:
        ds = []
        i = 1
        while i * i <= n:
            if n % i == 0:
                ds.append(i)
                if i != n // i:
                    ds.append(n // i)
            i += 1
        return ds

    factors: List[int] = []
    val = int(dim)
    for i in range(num_factors - 1, 0, -1):
        target = val ** (1.0 / (i + 1))
        divs = _divisors(val)
        best = min(divs, key=lambda f: abs(f - target))
        factors.append(best)
        val //= best
    factors.append(val)

    while len(factors) < num_factors:
        factors.append(1)
    factors = factors[:num_factors]
    assert int(np.prod(factors)) == dim, f"factorize_dim: product {np.prod(factors)} != {dim}"
    return sorted(factors)


# ---------------------------------------------------------------------------
# TT-SVD
# ---------------------------------------------------------------------------
def _interleave_to_tensor(matrix: np.ndarray, in_shapes: Sequence[int], out_shapes: Sequence[int]) -> np.ndarray:
    """Reshape ``matrix`` [out, in] into the interleaved TT-matrix tensor order.

    Produces a tensor of shape ``(i_1, o_1, i_2, o_2, ..., i_d, o_d)`` such that
    flattening in row-major order recovers ``matrix`` (output-major). This is
    the canonical TT-matrix representation.
    """
    d = len(in_shapes)
    out_features = int(np.prod(out_shapes))
    in_features = int(np.prod(in_shapes))
    assert matrix.shape == (out_features, in_features), (
        f"matrix shape {matrix.shape} != ({out_features}, {in_features})"
    )
    W = matrix.reshape(*out_shapes, *in_shapes)  # [o_1..o_d, i_1..i_d]
    perm = []
    for k in range(d):
        perm.append(d + k)  # i_k
        perm.append(k)       # o_k
    W = W.transpose(perm)
    return np.ascontiguousarray(W)


def _tt_svd_on_interleaved(
    tensor: np.ndarray,
    in_shapes: Sequence[int],
    out_shapes: Sequence[int],
    max_rank: int,
    eps: float,
) -> List[np.ndarray]:
    """Run core TT-SVD on an interleaved ``(i_1, o_1, ..., i_d, o_d)`` tensor.

    Returns ``d`` cores of shape ``[r_{k-1}, i_k, o_k, r_k]``.
    """
    d = len(in_shapes)
    n_dims = [in_shapes[k] * out_shapes[k] for k in range(d)]
    folded = tensor.reshape(n_dims)  # [n_1, n_2, ..., n_d]

    cores: List[np.ndarray] = []
    r_prev = 1
    remaining = folded.reshape(1, -1)  # [1, n_1*n_2*...*n_d]
    total_size = folded.size

    for k in range(d):
        n_k = n_dims[k]
        if k < d - 1:
            rest = int(np.prod(n_dims[k + 1:]))
            remaining = remaining.reshape(r_prev * n_k, rest)
            u, s, vt = np.linalg.svd(remaining, full_matrices=False)
            if eps > 0 and s.size > 0 and s[0] > 0:
                tol = eps * float(s[0]) * math.sqrt(max(1, total_size))
                keep = max(int((s > tol).sum()), 1)
            else:
                keep = s.size
            keep = min(keep, max_rank)
            u = u[:, :keep]
            s = s[:keep]
            vt = vt[:keep, :]
            r_k = keep
            core = u.reshape(r_prev, in_shapes[k], out_shapes[k], r_k)
            remaining = vt * s[:, None]  # [r_k, rest]
        else:
            # last core: reshape the residual, no SVD/truncation needed
            core = remaining.reshape(r_prev, in_shapes[k], out_shapes[k], 1)
            r_k = 1
        cores.append(core)
        r_prev = r_k
    return cores


def tt_svd(
    matrix: np.ndarray,
    in_shapes: Sequence[int],
    out_shapes: Sequence[int],
    max_rank: int,
    eps: float = 1e-6,
) -> List[np.ndarray]:
    """TT-SVD decomposition of ``matrix`` into ``d`` TT cores.

    Parameters
    ----------
    matrix : np.ndarray, shape ``[out_features, in_features]``
    in_shapes, out_shapes : factor shapes with ``prod`` matching the matrix dims
    max_rank : upper bound on each internal TT rank
    eps : relative singular-value truncation tolerance
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    tensor = _interleave_to_tensor(matrix, in_shapes, out_shapes)
    return _tt_svd_on_interleaved(tensor, in_shapes, out_shapes, max_rank, eps)


def svd(
    matrix: np.ndarray,
    in_shapes: Sequence[int],
    out_shapes: Sequence[int],
    max_rank: int,
    eps: float = 0.0,
    tt_max_rank: int = 0,
) -> List[np.ndarray]:
    """Global low-rank SVD, then format as ``d`` TT cores.

    Computes the best rank-``max_rank`` approximation ``W_r = U_r S_r V_r^T`` and
    runs :func:`tt_svd` on ``W_r`` so the result is a valid TT chain that
    reconstructs ``W_r``. TT internal ranks are left un-truncated by default
    (the global SVD already provides the rank constraint), but can be capped via
    ``tt_max_rank`` to enforce a parameter budget.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    keep = min(max_rank, s.size)
    if eps > 0 and s.size > 0 and s[0] > 0:
        tol = eps * float(s[0])
        keep = max(int((s > tol).sum()), 1)
        keep = min(keep, max_rank)
    u = u[:, :keep]
    s = s[:keep]
    vt = vt[:keep, :]
    low_rank = (u * s[None, :]) @ vt
    tt_cap = tt_max_rank if tt_max_rank > 0 else 10**9
    return tt_svd(low_rank, in_shapes, out_shapes, max_rank=tt_cap, eps=0.0)


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------
def reconstruct_matrix(cores: Sequence[np.ndarray]) -> np.ndarray:
    """Contract ``d`` TT cores back into a 2D matrix ``[out_features, in_features]``.

    Cores are shaped ``[r_{k-1}, i_k, o_k, r_k]``. The full contraction produces
    the interleaved tensor ``(i_1, o_1, i_2, o_2, ..., i_d, o_d)`` which is then
    transposed to ``(o_1, ..., o_d, i_1, ..., i_d)`` and flattened to
    ``[out_features, in_features]`` (matching ``nn.Linear.weight``).
    """
    d = len(cores)
    in_shapes = [c.shape[1] for c in cores]
    out_shapes = [c.shape[2] for c in cores]
    # contract cores sequentially over the right/left rank axes
    result = cores[0]  # [1, i_1, o_1, r_1]
    for k in range(1, d):
        result = np.tensordot(result, cores[k], axes=([-1], [0]))
    # result: [1, i_1, o_1, ..., i_d, o_d, 1] -> drop boundary rank dims explicitly
    result = result.reshape([v for k in range(d) for v in (in_shapes[k], out_shapes[k])])
    # interleaved (i_1, o_1, ..., i_d, o_d) -> (o_1, ..., o_d, i_1, ..., i_d)
    out_axes = [2 * k + 1 for k in range(d)]
    in_axes = [2 * k for k in range(d)]
    result = result.transpose(out_axes + in_axes)
    out_features = int(np.prod(out_shapes))
    in_features = int(np.prod(in_shapes))
    return np.ascontiguousarray(result).reshape(out_features, in_features)
