"""Tests for tt_llm.layers.

Verifies that both TensorLinear and LinearTensorLinear:

- produce shapes matching nn.Linear for 2D and 3D inputs, and
- reconstruct the original nn.Linear weight matrix when cores are loaded from
  TT-SVD (i.e. forward(x) ~= nn.Linear(x) up to SVD truncation error).
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from tt_llm.decompositions import factorize_dim, reconstruct_matrix, svd, tt_svd
from tt_llm.layers import LinearTensorLinear, TensorLinear


def _make_linear(in_features, out_features, seed=0):
    torch.manual_seed(seed)
    return nn.Linear(in_features, out_features, bias=True)


def _load_cores_tensor(layer, cores, dtype, device):
    with torch.no_grad():
        for param, core in zip(layer.cores, cores):
            param.copy_(torch.from_numpy(np.ascontiguousarray(core)).to(dtype=dtype, device=device))


def _load_cores_linear(layer, cores, dtype, device):
    from tt_llm.compress import _load_cores

    _load_cores(layer, cores, dtype, device, "LinearTensorLinear")


@pytest.mark.parametrize("LayerCls,loader", [
    (TensorLinear, _load_cores_tensor),
    (LinearTensorLinear, _load_cores_linear),
])
class TestLayerForward:
    IN_FEATURES = 576
    OUT_FEATURES = 576

    def _build(self, LayerCls, loader, init_method="tt_svd", max_rank=64):
        linear = _make_linear(self.IN_FEATURES, self.OUT_FEATURES, seed=11)
        in_shapes = factorize_dim(self.IN_FEATURES, 3)
        out_shapes = factorize_dim(self.OUT_FEATURES, 3)
        W = linear.weight.detach().numpy().astype(np.float64)
        cores = (
            tt_svd(W, in_shapes, out_shapes, max_rank=max_rank, eps=0.0)
            if init_method == "tt_svd"
            else svd(W, in_shapes, out_shapes, max_rank=max_rank, eps=0.0)
        )
        ranks = [cores[0].shape[0]] + [c.shape[3] for c in cores]
        layer = LayerCls(self.IN_FEATURES, self.OUT_FEATURES, in_shapes, out_shapes, ranks, bias=True)
        loader(layer, cores, torch.float32, "cpu")
        if layer.bias is not None:
            layer.bias.data.copy_(linear.bias.data)
        return linear, layer

    def test_output_shape_2d(self, LayerCls, loader):
        _, layer = self._build(LayerCls, loader)
        x = torch.randn(4, self.IN_FEATURES)
        y = layer(x)
        assert y.shape == (4, self.OUT_FEATURES)

    def test_output_shape_3d(self, LayerCls, loader):
        _, layer = self._build(LayerCls, loader)
        x = torch.randn(2, 5, self.IN_FEATURES)
        y = layer(x)
        assert y.shape == (2, 5, self.OUT_FEATURES)

    def test_matches_linear_high_rank(self, LayerCls, loader):
        """With full rank, TT forward should match nn.Linear closely."""
        linear, layer = self._build(LayerCls, loader, max_rank=512)
        torch.manual_seed(99)
        x = torch.randn(3, self.IN_FEATURES)
        y_ref = linear(x).detach().numpy()
        y_tt = layer(x).detach().numpy()
        rel_err = np.linalg.norm(y_tt - y_ref) / max(np.linalg.norm(y_ref), 1e-12)
        assert rel_err < 1e-5, f"{LayerCls.__name__} vs Linear rel error: {rel_err}"

    @pytest.mark.parametrize("init_method", ["tt_svd", "svd"])
    def test_low_rank_approx(self, LayerCls, loader, init_method):
        """TT layers should approximate a low-rank weight matrix better than random.

        ``svd`` gives the optimal rank-r approximation (TT-ranks left unbounded,
        so it recovers a rank-8 matrix almost exactly).
        ``tt_svd`` performs per-unfolding truncation; with max_rank >= the
        matrix rank it can still recover well, so we use a generous rank budget.
        Random init would produce ~1.0 relative error, so we just require < 0.5.
        """
        rng = np.random.default_rng(123)
        a = rng.standard_normal((self.OUT_FEATURES, 8))
        b = rng.standard_normal((8, self.IN_FEATURES))
        W_low = a @ b  # rank exactly 8
        in_shapes = factorize_dim(self.IN_FEATURES, 3)
        out_shapes = factorize_dim(self.OUT_FEATURES, 3)
        # give tt_svd a high enough rank budget to recover the rank-8 matrix
        max_rank = 576 if init_method == "tt_svd" else 8
        cores = (
            tt_svd(W_low, in_shapes, out_shapes, max_rank=max_rank, eps=0.0)
            if init_method == "tt_svd"
            else svd(W_low, in_shapes, out_shapes, max_rank=max_rank, eps=0.0)
        )
        ranks = [cores[0].shape[0]] + [c.shape[3] for c in cores]
        layer = LayerCls(self.IN_FEATURES, self.OUT_FEATURES, in_shapes, out_shapes, ranks, bias=False)
        loader(layer, cores, torch.float32, "cpu")
        layer.eval()
        x = torch.randn(3, self.IN_FEATURES)
        y_ref = x.detach().numpy() @ W_low.T
        with torch.no_grad():
            y_tt = layer(x).detach().numpy()
        rel_err = np.linalg.norm(y_tt - y_ref) / max(np.linalg.norm(y_ref), 1e-12)
        assert rel_err < 0.5, f"{LayerCls.__name__} [{init_method}] rel error too high: {rel_err}"
