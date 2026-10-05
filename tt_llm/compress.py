"""Model compression: replace ``nn.Linear`` layers with TT-decomposed layers.

The original ``compress_model_inplace`` swapped each ``nn.Linear`` for a
:class:`~tt_llm.layers.TensorLinear` whose cores were *randomly initialized*
(``torch.randn * 0.1``). This destroyed the pretrained weights and produced
garbage output after compression.

This module keeps that behaviour as ``init_method="random"`` for ablation, and
adds two faithful initializations that reconstruct the original weight matrix:

- ``init_method="svd"``: global rank-``r`` SVD of ``W``; the best Frobenius-norm
  rank-``r`` approximation, embedded as a TT chain. Ranked modes per layer are
  searched to hit ``target_ratio``.
- ``init_method="tt_svd"``: canonical per-unfolding TT-SVD with relative eps;
  same rank search.

The rank search is a linear scan over candidate ranks, picking the rank closest
to (but not exceeding) the target parameter budget — matching the original
notebook's goal of a user-specified compression ratio.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from .decompositions import factorize_dim, reconstruct_matrix, svd, tt_svd
from .layers import LinearTensorLinear, TensorLinear


def _tt_param_count(
    in_shapes: Iterable[int],
    out_shapes: Iterable[int],
    ranks: Iterable[int],
    layer_type: str,
) -> int:
    """Estimate parameter count for a TT layer (cores only, no bias)."""
    in_shapes = list(in_shapes)
    out_shapes = list(out_shapes)
    ranks = list(ranks)
    n = len(in_shapes)
    if layer_type == "linear":
        return sum(ranks[i] * in_shapes[i] * out_shapes[i] * ranks[i + 1] for i in range(n))
    return sum(ranks[i] * in_shapes[i] * out_shapes[i] * ranks[i + 1] for i in range(n))


def _search_rank(
    in_features: int,
    out_features: int,
    in_shapes: List[int],
    out_shapes: List[int],
    target_params: float,
    num_factors: int,
    layer_type: str,
) -> int:
    """Find the rank closest to (but <=) the target param budget."""
    best_rank, best_diff = 1, float("inf")
    upper = min(in_features, out_features)
    for r in range(1, upper):
        ranks = [1] + [r] * (num_factors - 1) + [1]
        total = _tt_param_count(in_shapes, out_shapes, ranks, layer_type)
        diff = abs(total - target_params)
        if diff < best_diff:
            best_diff = diff
            best_rank = r
        if total > target_params:
            break
    return best_rank


def _cores_to_tensor(cores: List[np.ndarray], dtype: torch.dtype, device: torch.device) -> List[torch.Tensor]:
    return [torch.from_numpy(np.ascontiguousarray(c)).to(dtype=dtype, device=device) for c in cores]


def compress_model_inplace(
    model: nn.Module,
    target_ratio: float = 1.42857,
    num_factors: int = 3,
    layer_type: str = "tensor",
    init_method: str = "svd",
    eps: float = 1e-6,
    verbose: bool = True,
) -> None:
    """Recursively replace every ``nn.Linear`` in ``model`` with a TT layer.

    Parameters
    ----------
    model : nn.Module
    target_ratio : desired ``orig_params / tt_params`` ratio (1.42857 ~= 30% cut).
    num_factors : number of TT cores.
    layer_type : ``"tensor"`` -> :class:`TensorLinear`, ``"linear"`` -> :class:`LinearTensorLinear`.
    init_method : ``"random"``, ``"svd"``, or ``"tt_svd"``.
    eps : relative truncation tolerance for SV/TT-SVD.
    """
    if init_method not in {"random", "svd", "tt_svd"}:
        raise ValueError(f"init_method must be random|svd|tt_svd, got {init_method!r}")
    if layer_type not in {"tensor", "linear"}:
        raise ValueError(f"layer_type must be tensor|linear, got {layer_type!r}")

    for name, module in list(model.named_children()):
        if isinstance(module, nn.Linear):
            compressed = _compress_linear(
                module, target_ratio, num_factors, layer_type, init_method, eps
            )
            setattr(model, name, compressed)
            if verbose:
                orig = module.in_features * module.out_features + (module.out_features if module.bias is not None else 0)
                new = sum(p.numel() for p in compressed.parameters())
                print(
                    f"Modified '{name}': Linear -> {type(compressed).__name__} "
                    f"[{init_method}] | Comp: {orig / max(new, 1):.2f}x"
                )
        else:
            compress_model_inplace(
                module, target_ratio, num_factors, layer_type, init_method, eps, verbose
            )


def _compress_linear(
    module: nn.Linear,
    target_ratio: float,
    num_factors: int,
    layer_type: str,
    init_method: str,
    eps: float,
) -> nn.Module:
    in_features = module.in_features
    out_features = module.out_features
    has_bias = module.bias is not None
    dtype = module.weight.dtype
    device = module.weight.device

    in_shapes = factorize_dim(in_features, num_factors)
    out_shapes = factorize_dim(out_features, num_factors)

    orig_params = in_features * out_features + (out_features if has_bias else 0)
    target_params = orig_params / target_ratio
    best_rank = _search_rank(in_features, out_features, in_shapes, out_shapes, target_params, num_factors, layer_type)
    ranks = [1] + [best_rank] * (num_factors - 1) + [1]

    if init_method == "random":
        if layer_type == "linear":
            layer = LinearTensorLinear(in_features, out_features, in_shapes, out_shapes, ranks, bias=has_bias)
        else:
            layer = TensorLinear(in_features, out_features, in_shapes, out_shapes, ranks, bias=has_bias)
        return layer.to(device=device, dtype=dtype)

    # decompose the original weight and load TT cores
    weight_np = module.weight.detach().cpu().numpy().astype(np.float64)
    if init_method == "svd":
        cores = svd(weight_np, in_shapes, out_shapes, max_rank=best_rank, eps=eps, tt_max_rank=best_rank)
    else:  # tt_svd
        cores = tt_svd(weight_np, in_shapes, out_shapes, max_rank=best_rank, eps=eps)

    # build the layer using the ACTUAL ranks produced by the decomposition
    actual_ranks = [cores[0].shape[0]] + [c.shape[3] for c in cores]
    if layer_type == "linear":
        layer = LinearTensorLinear(in_features, out_features, in_shapes, out_shapes, actual_ranks, bias=has_bias)
        cls_name = "LinearTensorLinear"
    else:
        layer = TensorLinear(in_features, out_features, in_shapes, out_shapes, actual_ranks, bias=has_bias)
        cls_name = "TensorLinear"
    layer = layer.to(device=device, dtype=dtype)

    _load_cores(layer, cores, dtype, device, cls_name)
    if has_bias:
        layer.bias.data.copy_(module.bias.data)
    return layer


def _load_cores(layer: nn.Module, cores: List[np.ndarray], dtype, device, cls_name: str) -> None:
    """Copy TT cores (np [r_{k-1}, i_k, o_k, r_k]) into the layer's parameters."""
    if cls_name == "TensorLinear":
        with torch.no_grad():
            for param, core in zip(layer.cores, cores):
                param.copy_(_cores_to_tensor([core], dtype, device)[0])
    else:  # LinearTensorLinear: reshape core back into nn.Linear.weight
        with torch.no_grad():
            for sub_layer, core in zip(layer.core_layers, cores):
                # core: [r_{k-1}, i_k, o_k, r_k]
                # _core_tensor builds core via:
                #   weight.reshape(o_k, r_next, r_prev, i_k).permute(2, 3, 0, 1)
                # so the inverse is permute(2, 3, 0, 1).reshape(o_k * r_next,
                # r_prev * i_k), producing nn.Linear.weight layout
                # [O_k * r_k, r_{k-1} * I_k].
                o_k = core.shape[2]
                r_next = core.shape[3]
                r_prev = core.shape[0]
                i_k = core.shape[1]
                weight = np.ascontiguousarray(
                    core.transpose(2, 3, 0, 1)
                    if isinstance(core, np.ndarray)
                    else core.permute(2, 3, 0, 1)
                ).reshape(o_k * r_next, r_prev * i_k)
                sub_layer.weight.data.copy_(
                    torch.from_numpy(np.ascontiguousarray(weight)).to(dtype=dtype, device=device)
                )
