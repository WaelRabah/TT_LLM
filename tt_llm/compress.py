"""Model compression: replace ``nn.Linear`` layers with TT-decomposed layers.

Two compression modes:

- ``mode="uniform"``: every ``nn.Linear`` is compressed by the same
  ``compression_pct`` (e.g. 30 for a 30% parameter cut), matching the original
  notebook's behaviour but with faithful init.
- ``mode="targeted"``: run the model on a set of calibration prompts, capture
  per-layer input activations via forward hooks, score each layer's importance
  by the Frobenius norm of its activations, and allocate the global parameter
  budget proportionally — important layers keep more parameters (less
  compression), less-important layers are compressed more aggressively or
  skipped entirely.

The user-facing knob is now ``compression_pct`` (0-100, e.g. 30 = "reduce
parameters by 30%"), replacing the old ``target_ratio`` inverse ratio.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .decompositions import factorize_dim, reconstruct_matrix, svd, tt_svd
from .layers import LinearTensorLinear, TensorLinear


# ---------------------------------------------------------------------------
# Parameter bookkeeping
# ---------------------------------------------------------------------------
def _linear_params(module: nn.Linear) -> int:
    return int(module.in_features * module.out_features + (module.out_features if module.bias is not None else 0))


def _tt_param_count(
    in_shapes: Sequence[int],
    out_shapes: Sequence[int],
    ranks: Sequence[int],
) -> int:
    """Parameter count for a TT layer's cores (bias is added separately)."""
    n = len(in_shapes)
    return sum(int(ranks[i] * in_shapes[i] * out_shapes[i] * ranks[i + 1]) for i in range(n))


def _rank_for_budget(
    in_features: int,
    out_features: int,
    in_shapes: Sequence[int],
    out_shapes: Sequence[int],
    num_factors: int,
    target_params: float,
) -> int:
    """Find the uniform rank whose TT param count is closest to (but <=) budget.

    Falls back to 1 if even rank-1 exceeds the budget (degenerate case).
    """
    best_rank, best_diff = 1, float("inf")
    upper = max(1, min(in_features, out_features))
    for r in range(1, upper):
        ranks = [1] + [r] * (num_factors - 1) + [1]
        total = _tt_param_count(in_shapes, out_shapes, ranks)
        diff = abs(total - target_params)
        if diff < best_diff:
            best_diff = diff
            best_rank = r
        if total > target_params:
            break
    return best_rank


# ---------------------------------------------------------------------------
# Core compression primitives
# ---------------------------------------------------------------------------
def _build_tt_layer(
    module: nn.Linear,
    num_factors: int,
    layer_type: str,
    init_method: str,
    eps: float,
    target_params: Optional[float],
    rank_override: Optional[int],
) -> Optional[nn.Module]:
    """Build a TT replacement for ``module``.

    If ``target_params`` is given, the rank is searched to fit that budget.
    If ``rank_override`` is given, it is used directly (targeted path).
    Returns ``None`` if the layer should be skipped (rank 0).
    """
    in_features = module.in_features
    out_features = module.out_features
    has_bias = module.bias is not None
    dtype = module.weight.dtype
    device = module.weight.device

    in_shapes = factorize_dim(in_features, num_factors)
    out_shapes = factorize_dim(out_features, num_factors)

    if rank_override is not None:
        best_rank = max(1, int(rank_override))
    elif target_params is not None:
        best_rank = _rank_for_budget(in_features, out_features, in_shapes, out_shapes, num_factors, target_params)
    else:
        best_rank = max(1, min(in_features, out_features) // 2)

    ranks = [1] + [best_rank] * (num_factors - 1) + [1]

    if init_method == "random":
        cls = LinearTensorLinear if layer_type == "linear" else TensorLinear
        layer = cls(in_features, out_features, in_shapes, out_shapes, ranks, bias=has_bias)
        return layer.to(device=device, dtype=dtype)

    weight_np = module.weight.detach().cpu().numpy().astype(np.float64)
    if init_method == "svd":
        cores = svd(weight_np, in_shapes, out_shapes, max_rank=best_rank, eps=eps, tt_max_rank=best_rank)
    else:  # tt_svd
        cores = tt_svd(weight_np, in_shapes, out_shapes, max_rank=best_rank, eps=eps)

    actual_ranks = [cores[0].shape[0]] + [c.shape[3] for c in cores]
    cls = LinearTensorLinear if layer_type == "linear" else TensorLinear
    cls_name = cls.__name__
    layer = cls(in_features, out_features, in_shapes, out_shapes, actual_ranks, bias=has_bias)
    layer = layer.to(device=device, dtype=dtype)

    _load_cores(layer, cores, dtype, device, cls_name)
    if has_bias:
        layer.bias.data.copy_(module.bias.data)
    return layer


def _load_cores(layer: nn.Module, cores: List[np.ndarray], dtype, device, cls_name: str) -> None:
    """Copy TT cores (np ``[r_{k-1}, i_k, o_k, r_k]``) into the layer's params."""
    if cls_name == "TensorLinear":
        with torch.no_grad():
            for param, core in zip(layer.cores, cores):
                param.copy_(torch.from_numpy(np.ascontiguousarray(core)).to(dtype=dtype, device=device))
    else:  # LinearTensorLinear: reshape core back into nn.Linear.weight
        with torch.no_grad():
            for sub_layer, core in zip(layer.core_layers, cores):
                o_k, r_next = core.shape[2], core.shape[3]
                r_prev, i_k = core.shape[0], core.shape[1]
                weight = np.ascontiguousarray(core.transpose(2, 3, 0, 1)).reshape(o_k * r_next, r_prev * i_k)
                sub_layer.weight.data.copy_(
                    torch.from_numpy(np.ascontiguousarray(weight)).to(dtype=dtype, device=device)
                )


# ---------------------------------------------------------------------------
# Collect all nn.Linear modules (name, parent, module) by recursive walk
# ---------------------------------------------------------------------------
def _collect_linears(model: nn.Module) -> List[Tuple[str, nn.Module, nn.Linear]]:
    """Return list of (dotted_name, parent_module, linear_module) for every
    ``nn.Linear`` in ``model``, excluding the internal core layers of
    ``LinearTensorLinear`` (which are themselves ``nn.Linear``)."""
    found: List[Tuple[str, nn.Module, nn.Linear]] = []
    for name, child in model.named_modules():
        if isinstance(child, (TensorLinear, LinearTensorLinear)):
            continue  # don't recurse into already-compressed layers
        for attr_name, sub in child.named_children():
            if isinstance(sub, nn.Linear):
                found.append((f"{name}.{attr_name}" if name else attr_name, child, sub))
    return found


# ---------------------------------------------------------------------------
# Uniform compression
# ---------------------------------------------------------------------------
def compress_model_inplace(
    model: nn.Module,
    compression_pct: float = 30.0,
    num_factors: int = 3,
    layer_type: str = "tensor",
    init_method: str = "svd",
    eps: float = 1e-6,
    verbose: bool = True,
    skip_layers: Optional[Sequence[str]] = None,
) -> dict:
    """Recursively replace every ``nn.Linear`` in ``model`` with a TT layer.

    Parameters
    ----------
    model : nn.Module
    compression_pct : float in [0, 100]
        Percentage of parameters to remove (e.g. 30 = 30% cut -> 70% remain).
    num_factors : number of TT cores.
    layer_type : ``"tensor"`` -> :class:`TensorLinear`, ``"linear"`` -> :class:`LinearTensorLinear`.
    init_method : ``"random"``, ``"svd"``, or ``"tt_svd"``.
    eps : relative truncation tolerance for SVD / TT-SVD.
    skip_layers : dotted module names to leave untouched.
    """
    if not 0 <= compression_pct <= 100:
        raise ValueError(f"compression_pct must be in [0, 100], got {compression_pct}")
    if init_method not in {"random", "svd", "tt_svd"}:
        raise ValueError(f"init_method must be random|svd|tt_svd, got {init_method!r}")
    if layer_type not in {"tensor", "linear"}:
        raise ValueError(f"layer_type must be tensor|linear, got {layer_type!r}")

    skip = set(skip_layers or [])
    ratio = 1.0 / (1.0 - compression_pct / 100.0)  # 30% -> 1/0.7 ≈ 1.4286
    total_orig, total_new = 0, 0
    for name, parent, module in _collect_linears(model):
        if name in skip:
            if verbose:
                print(f"Skipped '{name}' (explicit skip)")
            continue
        orig = _linear_params(module)
        target = orig / ratio
        compressed = _build_tt_layer(module, num_factors, layer_type, init_method, eps, target_params=target, rank_override=None)
        if compressed is None:
            continue
        child_name = name.split(".")[-1]
        setattr(parent, child_name, compressed)
        new = sum(p.numel() for p in compressed.parameters())
        total_orig += orig
        total_new += new
        if verbose:
            print(f"Modified '{name}': Linear -> {type(compressed).__name__} [{init_method}] | Comp: {orig / max(new, 1):.2f}x")
    if verbose and total_new:
        print(f"\nTotal: {total_orig} -> {total_new} params | Comp: {total_orig / total_new:.2f}x (target {compression_pct:.0f}%)")
    return {"orig": total_orig, "new": total_new, "ratio": total_orig / max(total_new, 1)}


# ---------------------------------------------------------------------------
# Targeted (activation-aware) compression
# ---------------------------------------------------------------------------
def compress_model_targeted(
    model: nn.Module,
    tokenizer,
    calibration_prompts: Sequence[str],
    compression_pct: float = 30.0,
    num_factors: int = 3,
    layer_type: str = "tensor",
    init_method: str = "svd",
    eps: float = 1e-6,
    max_length: int = 128,
    device: Optional[str] = None,
    min_importance: float = 1e-6,
    verbose: bool = True,
) -> dict:
    """Activation-aware compression: important layers keep more parameters.

    Steps:
    1. Run ``calibration_prompts`` through ``model`` with forward hooks on every
       ``nn.Linear``; score each layer's importance = Frobenius norm of its
       input activations (summed across prompts/tokens).
    2. Compute the global parameter budget: ``orig_total * (1 - pct/100)``.
    3. Allocate budget across layers proportional to importance. Layers whose
       importance is below ``min_importance`` are skipped (left untouched) —
       this is the "compress unimportant layers aggressively" lever, but here
       we keep them uncompressed rather than destroying them, since they
       contribute little to the budget anyway.
    4. For each non-skipped layer, search the TT rank that fits its allocated
       budget and build the TT replacement.
    """
    if not 0 < compression_pct < 100:
        raise ValueError(f"compression_pct must be in (0, 100), got {compression_pct}")
    if init_method not in {"random", "svd", "tt_svd"}:
        raise ValueError(f"init_method must be random|svd|tt_svd, got {init_method!r}")
    if layer_type not in {"tensor", "linear"}:
        raise ValueError(f"layer_type must be tensor|linear, got {layer_type!r}")

    from .activations import capture_activations, compute_importance

    linears = _collect_linears(model)
    if not linears:
        return {"orig": 0, "new": 0, "ratio": 1.0}

    dev = device or next(model.parameters()).device
    model.eval()

    # 1. capture activations
    acts = capture_activations(model, linears, tokenizer, calibration_prompts, max_length, dev)
    importance = compute_importance(acts)

    # 2. global budget
    orig_params = {name: _linear_params(m) for name, _, m in linears}
    total_orig = sum(orig_params.values())
    total_budget = total_orig * (1.0 - compression_pct / 100.0)

    # 3. allocate budget proportional to importance
    names = [n for n, _, _ in linears]
    total_imp = sum(importance[n] for n in names)
    if total_imp <= 0:
        # degenerate: fall back to uniform
        if verbose:
            print("Total importance is zero; falling back to uniform compression.")
        return compress_model_inplace(model, compression_pct, num_factors, layer_type, init_method, eps, verbose)

    # a small floor so every non-skipped layer can at least form rank-1 TT cores
    floor_frac = 0.0
    floor_budget = sum(
        _tt_param_count(factorize_dim(m.in_features, num_factors), factorize_dim(m.out_features, num_factors), [1] * (num_factors + 1))
        for n, _, m in linears if importance[n] >= min_importance
    )
    # budget above the rank-1 floor is distributed proportional to importance
    above_floor = max(total_budget - floor_budget, 0.0)
    imp_above = {n: (importance[n] if importance[n] >= min_importance else 0.0) for n in names}
    imp_total_above = sum(imp_above.values())

    budgets: dict = {}
    for n, _, m in linears:
        if importance[n] < min_importance:
            budgets[n] = None  # skip
            continue
        base = _tt_param_count(factorize_dim(m.in_features, num_factors), factorize_dim(m.out_features, num_factors), [1] * (num_factors + 1))
        extra = above_floor * (imp_above[n] / imp_total_above) if imp_total_above > 0 else 0.0
        budgets[n] = base + extra

    # 4. compress each layer to its budget
    total_new = 0
    for name, parent, module in linears:
        b = budgets[name]
        if b is None:
            if verbose:
                print(f"Skipped '{name}' (importance {importance[name]:.4e} < {min_importance})")
            total_new += _linear_params(module)
            continue
        compressed = _build_tt_layer(module, num_factors, layer_type, init_method, eps, target_params=b, rank_override=None)
        if compressed is None:
            total_new += _linear_params(module)
            continue
        child_name = name.split(".")[-1]
        setattr(parent, child_name, compressed)
        new = sum(p.numel() for p in compressed.parameters())
        total_new += new
        orig = orig_params[name]
        if verbose:
            print(
                f"Modified '{name}': Linear -> {type(compressed).__name__} [{init_method}] | "
                f"imp={importance[name]:.3e} | Comp: {orig / max(new, 1):.2f}x"
            )

    if verbose:
        print(
            f"\nTotal: {total_orig} -> {total_new} params | "
            f"Comp: {total_orig / max(total_new, 1):.2f}x (target {compression_pct:.0f}%)"
        )
    return {"orig": total_orig, "new": total_new, "ratio": total_orig / max(total_new, 1)}
