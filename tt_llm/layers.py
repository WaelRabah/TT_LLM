"""Tensor-Train (TT) linear layers for PyTorch.

Two interchangeable implementations of a TT-decomposed ``nn.Linear``:

- :class:`TensorLinear` stores the TT cores directly as ``[r_{k-1}, I_k, O_k, r_k]``
  parameters and contracts them with ``torch.einsum``.
- :class:`LinearTensorLinear` stores the cores as standard ``nn.Linear`` modules of
  shape ``(r_{k-1} * I_k) -> (O_k * r_k)`` and extracts the core tensor on the fly,
  so it remains a valid TT layer while using native ``nn.Linear`` plumbing.

Both layers implement the same TT-matrix contraction::

    y[b, o_1..o_d] = sum_{i_1..i_d} G_1[1, i_1, o_1, r_1]
                                   * G_2[r_1, i_2, o_2, r_2]
                                   * ...
                                   * G_d[r_{d-1}, i_d, o_d, 1]
                                   * x[b, i_1..i_d]

where the input/output feature dims are factored into ``in_shapes`` / ``out_shapes``
(with ``prod(in_shapes) == in_features``).
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = list(x.shape)
        x_flat = x.reshape(-1, self.in_features)
        batch = x_flat.shape[0]
        x_r = x_flat.reshape(batch, *self.in_shapes)  # [b, i_1, ..., i_d]

        # state after step k: [b, o_1..o_{k-1}, r_{k-1}, i_k..i_d]
        state = x_r.unsqueeze(1)  # [b, r_0=1, i_1..i_d]

        for k, core in enumerate(self.cores):
            # contract r_{k-1} (axis k+1) and i_k (axis k+2) with core axes 0, 1
            state = torch.tensordot(state, core, dims=([k + 1, k + 2], [0, 1]))
            # state: [b, o_1..o_{k-1}, i_{k+1}..i_d, o_k, r_k]
            # move o_k, r_k from the tail to positions k+1, k+2
            state = state.movedim([-2, -1], [k + 1, k + 2])

        # state: [b, o_1..o_d, r_d=1] -> squeeze
        out = state.squeeze(-1).reshape(batch, self.out_features)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = list(x.shape)
        x_flat = x.reshape(-1, self.in_features)
        batch = x_flat.shape[0]
        x_r = x_flat.reshape(batch, *self.in_shapes)

        state = x_r.unsqueeze(1)  # [b, r_0=1, i_1..i_d]
        for k in range(self.num_cores):
            core = self._core_tensor(k)
            state = torch.tensordot(state, core, dims=([k + 1, k + 2], [0, 1]))
            state = state.movedim([-2, -1], [k + 1, k + 2])

        out = state.squeeze(-1).reshape(batch, self.out_features)
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*(orig_shape[:-1] + [self.out_features]))


def _validate_tt_structure(in_features, out_features, in_shapes, out_shapes, ranks, num_cores):
    assert len(in_shapes) == len(out_shapes), "Dimension shapes mismatch."
    assert len(ranks) == num_cores + 1, "Ranks length must be num_cores + 1."
    assert ranks[0] == 1 and ranks[-1] == 1, "Boundary ranks must equal 1."
    assert int(np.prod(in_shapes)) == in_features, "Input shape product must equal in_features."
    assert int(np.prod(out_shapes)) == out_features, "Output shape product must equal out_features."
