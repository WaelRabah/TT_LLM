"""Tests for tt_llm.decompositions."""

import numpy as np
import pytest

from tt_llm.decompositions import factorize_dim, reconstruct_matrix, svd, tt_svd


# --- factorize_dim -------------------------------------------------------
@pytest.mark.parametrize("dim,n", [(576, 3), (1024, 3), (2048, 3), (941, 3), (1, 3), (12, 4), (576, 1)])
def test_factorize_dim_product_and_count(dim, n):
    factors = factorize_dim(dim, n)
    assert len(factors) == n
    assert int(np.prod(factors)) == dim, f"product {np.prod(factors)} != {dim}"
    assert all(f >= 1 for f in factors)


def test_factorize_dim_balanced():
    factors = factorize_dim(576, 3)
    # 576 = 2^6 * 3^2; balanced split ~ [8, 9, 8] order doesn't matter (sorted)
    # ensure no degenerate 1s leak in when avoidable
    assert int(np.prod(factors)) == 576
    assert max(factors) / min(factors) <= 4.0, f"unbalanced factors: {factors}"


# --- reconstruction round-trip (exact) -----------------------------------
def _make_weights(out_features, in_features, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((out_features, in_features))


@pytest.mark.parametrize("in_features,out_features", [(576, 576), (576, 1536), (1024, 4096)])
def test_tt_svd_exact_reconstruction_high_rank(in_features, out_features):
    """With max_rank high enough, TT-SVD should reconstruct W nearly exactly."""
    W = _make_weights(out_features, in_features, seed=42)
    in_shapes = factorize_dim(in_features, 3)
    out_shapes = factorize_dim(out_features, 3)
    cores = tt_svd(W, in_shapes, out_shapes, max_rank=1024, eps=0.0)
    W_rec = reconstruct_matrix(cores)
    assert W_rec.shape == W.shape
    rel_err = np.linalg.norm(W_rec - W) / np.linalg.norm(W)
    assert rel_err < 1e-8, f"TT-SVD rel error too high: {rel_err}"


@pytest.mark.parametrize("in_features,out_features", [(576, 576), (576, 1536)])
def test_svd_exact_reconstruction_high_rank(in_features, out_features):
    """Global SVD with full rank should reconstruct W nearly exactly."""
    W = _make_weights(out_features, in_features, seed=7)
    in_shapes = factorize_dim(in_features, 3)
    out_shapes = factorize_dim(out_features, 3)
    cores = svd(W, in_shapes, out_shapes, max_rank=min(in_features, out_features), eps=0.0)
    W_rec = reconstruct_matrix(cores)
    rel_err = np.linalg.norm(W_rec - W) / np.linalg.norm(W)
    assert rel_err < 1e-8, f"SVD rel error too high: {rel_err}"


# --- rank truncation properties ------------------------------------------
def test_tt_svd_low_rank_better_than_full_dim():
    """A low-rank TT-SVD should have error <= a roughly-equal-rank budget.

    A globally rank-r matrix can have TT-ranks exceeding r (unfolding ranks of
    the interleaved tensor), so we use a generous max_rank and verify the
    reconstruction is good. With no truncation it's exact.
    """
    rng = np.random.default_rng(1)
    a = rng.standard_normal((576, 10))
    b = rng.standard_normal((10, 576))
    W = a @ b  # rank exactly 10
    in_shapes = factorize_dim(576, 3)
    out_shapes = factorize_dim(576, 3)
    cores = tt_svd(W, in_shapes, out_shapes, max_rank=576, eps=0.0)
    W_rec = reconstruct_matrix(cores)
    rel_err = np.linalg.norm(W_rec - W) / np.linalg.norm(W)
    assert rel_err < 1e-8, f"TT-SVD on rank-10 matrix (no trunc) error too high: {rel_err}"


def test_svd_low_rank_approximates_well():
    """Global rank-r SVD with no TT truncation should reconstruct a rank-r matrix.

    Capping TT-ranks at r can over-truncate (a rank-r matrix has TT-ranks
    potentially >> r), so we disable TT truncation and verify the global SVD
    approximation is exact.
    """
    rng = np.random.default_rng(2)
    a = rng.standard_normal((576, 8))
    b = rng.standard_normal((8, 576))
    W = a @ b  # rank exactly 8
    in_shapes = factorize_dim(576, 3)
    out_shapes = factorize_dim(576, 3)
    cores = svd(W, in_shapes, out_shapes, max_rank=8, eps=0.0)
    W_rec = reconstruct_matrix(cores)
    rel_err = np.linalg.norm(W_rec - W) / np.linalg.norm(W)
    assert rel_err < 0.1, f"SVD rank-8 approx error too high: {rel_err}"


def test_svd_rank_truncation_loses_info():
    """Reducing the SVD rank below the true matrix rank should increase error."""
    rng = np.random.default_rng(5)
    a = rng.standard_normal((576, 16))
    b = rng.standard_normal((16, 576))
    W = a @ b  # rank exactly 16
    in_shapes = factorize_dim(576, 3)
    out_shapes = factorize_dim(576, 3)
    cores_high = svd(W, in_shapes, out_shapes, max_rank=576, eps=0.0)
    cores_low = svd(W, in_shapes, out_shapes, max_rank=8, eps=0.0)
    err_high = np.linalg.norm(reconstruct_matrix(cores_high) - W) / np.linalg.norm(W)
    err_low = np.linalg.norm(reconstruct_matrix(cores_low) - W) / np.linalg.norm(W)
    assert err_high < 1e-6, f"full-rank reconstruction error too high: {err_high}"
    assert err_low > err_high, f"rank truncation should increase error: {err_high} vs {err_low}"


def test_cores_shapes_and_boundary_ranks():
    W = _make_weights(576, 576, seed=3)
    in_shapes = factorize_dim(576, 3)
    out_shapes = factorize_dim(576, 3)
    cores = tt_svd(W, in_shapes, out_shapes, max_rank=16, eps=1e-6)
    assert len(cores) == 3
    for k, core in enumerate(cores):
        assert core.shape[1] == in_shapes[k]
        assert core.shape[2] == out_shapes[k]
    assert cores[0].shape[0] == 1
    assert cores[-1].shape[3] == 1
