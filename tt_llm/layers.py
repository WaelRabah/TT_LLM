"""Tensor-Train (TT) linear layers for PyTorch.

Two interchangeable implementations of a TT-decomposed ``nn.Linear``:

- :class:`TensorLinear` stores the TT cores directly as ``[r_{k-1}, I_k, O_k, r_k]``
  parameters and contracts them with matmul-based sequential contraction.
- :class:`LinearTensorLinear` stores the cores as standard ``nn.Linear`` modules of
  shape ``(r_{k-1} * I_k) -> (O_k * r_k)`` and extracts the core tensor on the fly,
  so it remains a valid TT layer while using native ``nn.Linear`` plumbing.

Both layers implement the same TT-matrix contraction::

    y[b, o_1..o_d] = sum_{i_1..i_d} G_1[1, i_1, o_1, r_1]
                                   * G_2[r_1, i_2, o_2, r_2]
                                   * ...
                                   * G_d[r_{d-1}, i_d, o_d, 1]
                                   * x[b, i_1..i_d]

The forward pass uses a two-matmul strategy:
1. Merge cores 0..d-2 into a single [prod(I_0..I_{d-2}), prod(O_0..O_{d-2}) * r_{d-1}]
   matrix via a 2-operand einsum chain (cheap).
2. Matmul the reshaped input with the merged matrix, permute, then matmul with
   the last core reshaped to [r_{d-1} * I_d, O_d].
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class TensorLinear(nn.Module):
    """TT linear layer whose cores are stored as raw parameters."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        in_shapes,
        out_shapes,
        ranks,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.in_shapes = [int(s) for s in in_shapes]
        self.out_shapes = [int(s) for s in out_shapes]
        self.ranks = [int(r) for r in ranks]
        self.num_cores = len(self.in_shapes)

        _validate_tt_structure(
            self.in_features,
            self.out_features,
            self.in_shapes,
            self.out_shapes,
            self.ranks,
            self.num_cores,
        )

        self.cores = nn.ParameterList(
            [
                nn.Parameter(
                    torch.randn(self.ranks[i], self.in_shapes[i], self.out_shapes[i], self.ranks[i + 1]) * 0.1
                )
                for i in range(self.num_cores)
            ]
        )
        self.bias = nn.Parameter(torch.zeros(self.out_features)) if bias else None
        self._cache = None

    def _core_tensor(self, k: int) -> torch.Tensor:
        """Return core k in the standard ``[r_{k-1}, I_k, O_k, r_k]`` layout."""
        return self.cores[k]

    def _get_cache(self):
        if self.training or self._cache is None:
            d = self.num_cores
            cores = [self._core_tensor(k) for k in range(d)]
            merged, merged_in, _ = _merge_cores(cores, self.in_shapes, self.out_shapes, self.ranks, d)
            last_i = self.in_shapes[d - 1]
            last_o = self.out_shapes[d - 1]
            r_last = self.ranks[d - 1]
            last_2d = cores[d - 1].squeeze(-1).reshape(r_last * last_i, last_o)
            if not self.training:
                self._cache = (merged, merged_in, last_2d)
            return merged, merged_in, last_2d
        return self._cache

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = list(x.shape)
        x_flat = x.reshape(-1, self.in_features)
        batch = x_flat.shape[0]
        d = self.num_cores
        i_sh = self.in_shapes
        o_sh = self.out_shapes
        rk = self.ranks

        if d == 1:
            c0 = self._core_tensor(0).squeeze(0).squeeze(-1)  # [i0, o0]
            out = x_flat @ c0
        else:
            merged, merged_in, last_2d = self._get_cache()
            last_i = i_sh[d - 1]
            r_last = rk[d - 1]
            prod_out_prev = int(np.prod(o_sh[:d - 1]))
            x_r = x_flat.reshape(batch, merged_in, last_i)
            s = torch.matmul(x_r.transpose(1, 2), merged)
            s = s.reshape(batch, last_i, prod_out_prev, r_last)
            s = s.permute(0, 2, 3, 1).contiguous()
            s = s.reshape(batch * prod_out_prev, r_last * last_i)
            out = s @ last_2d

        out = out.reshape(batch, self.out_features)
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*(orig_shape[:-1] + [self.out_features]))


class LinearTensorLinear(nn.Module):
    """TT linear layer whose cores are stored as ``nn.Linear`` modules.

    Each ``nn.Linear(r_{k-1} * I_k, O_k * r_k, bias=False)`` encodes the same
    TT core as :class:`TensorLinear` (just a permuted view of its weight), so the
    forward pass is mathematically identical.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        in_shapes,
        out_shapes,
        ranks,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.in_shapes = [int(s) for s in in_shapes]
        self.out_shapes = [int(s) for s in out_shapes]
        self.ranks = [int(r) for r in ranks]
        self.num_cores = len(self.in_shapes)

        _validate_tt_structure(
            self.in_features,
            self.out_features,
            self.in_shapes,
            self.out_shapes,
            self.ranks,
            self.num_cores,
        )

        self.core_layers = nn.ModuleList(
            [
                nn.Linear(
                    in_features=self.ranks[i] * self.in_shapes[i],
                    out_features=self.out_shapes[i] * self.ranks[i + 1],
                    bias=False,
                )
                for i in range(self.num_cores)
            ]
        )
        self.bias = nn.Parameter(torch.zeros(self.out_features)) if bias else None
        self._cache = None

    def _core_tensor(self, k: int) -> torch.Tensor:
        """Reshape ``core_layers[k].weight`` into the TT-core layout
        ``[r_{k-1}, I_k, O_k, r_k]``."""
        layer = self.core_layers[k]
        r_prev, i_k, o_k, r_next = (
            self.ranks[k],
            self.in_shapes[k],
            self.out_shapes[k],
            self.ranks[k + 1],
        )
        # weight: [O_k * r_k, r_{k-1} * I_k] -> [r_{k-1}, I_k, O_k, r_k]
        return layer.weight.reshape(o_k, r_next, r_prev, i_k).permute(2, 3, 0, 1)

    def _get_cache(self):
        if self.training or self._cache is None:
            d = self.num_cores
            cores = [self._core_tensor(k) for k in range(d)]
            merged, merged_in, _ = _merge_cores(cores, self.in_shapes, self.out_shapes, self.ranks, d)
            last_i = self.in_shapes[d - 1]
            last_o = self.out_shapes[d - 1]
            r_last = self.ranks[d - 1]
            last_2d = cores[d - 1].squeeze(-1).reshape(r_last * last_i, last_o)
            if not self.training:
                self._cache = (merged, merged_in, last_2d)
            return merged, merged_in, last_2d
        return self._cache

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = list(x.shape)
        x_flat = x.reshape(-1, self.in_features)
        batch = x_flat.shape[0]
        d = self.num_cores
        i_sh = self.in_shapes
        o_sh = self.out_shapes
        rk = self.ranks

        if d == 1:
            c0 = self._core_tensor(0).squeeze(0).squeeze(-1)
            out = x_flat @ c0
        else:
            merged, merged_in, last_2d = self._get_cache()
            last_i = i_sh[d - 1]
            r_last = rk[d - 1]
            prod_out_prev = int(np.prod(o_sh[:d - 1]))
            x_r = x_flat.reshape(batch, merged_in, last_i)
            s = torch.matmul(x_r.transpose(1, 2), merged)
            s = s.reshape(batch, last_i, prod_out_prev, r_last)
            s = s.permute(0, 2, 3, 1).contiguous()
            s = s.reshape(batch * prod_out_prev, r_last * last_i)
            out = s @ last_2d

        out = out.reshape(batch, self.out_features)
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*(orig_shape[:-1] + [self.out_features]))


def _validate_tt_structure(in_features, out_features, in_shapes, out_shapes, ranks, num_cores):
    assert len(in_shapes) == len(out_shapes), "Dimension shapes mismatch."
    assert len(ranks) == num_cores + 1, "Ranks length must be num_cores + 1."
    assert ranks[0] == 1 and ranks[-1] == 1, "Boundary ranks must equal 1."
    assert int(np.prod(in_shapes)) == in_features, "Input shape product must equal in_features."
    assert int(np.prod(out_shapes)) == out_features, "Output shape product must equal out_features."


def _merge_cores(cores, i_sh, o_sh, rk, d):
    """Merge cores 0..d-2 into a single 2D matrix.

    Returns ``(merged_2d, merged_in, merged_out)`` where:
    - ``merged_in  = prod(i_0..i_{d-2})``
    - ``merged_out = prod(o_0..o_{d-2}) * r_{d-1}``
    """
    merged = cores[0].squeeze(0)  # [i0, o0, r1]
    prod_in = i_sh[0]
    prod_out = o_sh[0]

    for k in range(1, d - 1):
        merged = torch.einsum('abc,cdef->adbef', merged, cores[k])
        merged = merged.reshape(prod_in * i_sh[k], prod_out * o_sh[k], rk[k + 1])
        prod_in *= i_sh[k]
        prod_out *= o_sh[k]

    merged_2d = merged.reshape(prod_in, prod_out * rk[d - 1])
    return merged_2d, prod_in, prod_out * rk[d - 1]
