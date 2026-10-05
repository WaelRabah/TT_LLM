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
def _svd_rank_for_budget(
    weight_np: np.ndarray,
    in_shapes: Sequence[int],
    out_shapes: Sequence[int],
    target_params: float,
    eps: float,
) -> Tuple[int, int]:
    """Find SVD rank and TT-rank cap whose TT params best fit the budget.

    Strategy: compute the full SVD once, then walk SVD ranks from high to low.
    For each SVD rank ``r``, build the rank-``r`` approximation and run TT-SVD
    once (uncapped). If uncapped params exceed the budget, find the TT-rank cap
    via the direct formula (TT params are quadratic in uniform rank), not by
    re-running TT-SVD. Pick the pair ``(r, t)`` with the lowest error.
    """
    from .decompositions import _interleave_to_tensor, _tt_svd_on_interleaved

    u, s, vt = np.linalg.svd(weight_np, full_matrices=False)
    max_sv = s.shape[0]

    best_r, best_t, best_diff = 1, 1, float("inf")

    # Walk SVD ranks from high (best quality) to low (fewest params)
    for r in [max_sv, max_sv // 2, max_sv // 4, max_sv // 8, max_sv // 16, 1]:
        if r < 1:
            continue
        r = min(r, s.shape[0])
        low_rank = (u[:, :r] * s[:r]) @ vt[:r, :]
        tensor = _interleave_to_tensor(low_rank, in_shapes, out_shapes)
        cores = _tt_svd_on_interleaved(tensor, in_shapes, out_shapes, max_rank=10**9, eps=eps)
        uncapped_params = sum(c.size for c in cores)

        if uncapped_params <= target_params:
            diff = abs(uncapped_params - target_params)
            if diff < best_diff:
                best_diff = diff
                best_r, best_t = r, 0
            continue

        # Need TT-rank cap; search it directly using _rank_for_budget
        n = len(in_shapes)
        tt_cap = _rank_for_budget(
            weight_np.shape[1], weight_np.shape[0],
            in_shapes, out_shapes, n, target_params,
        )
        capped = _tt_svd_on_interleaved(tensor, in_shapes, out_shapes, max_rank=tt_cap, eps=eps)
        p = sum(c.size for c in capped)
        diff = abs(p - target_params)
        if diff < best_diff:
            best_diff = diff
            best_r, best_t = r, tt_cap

    return best_r, best_t


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

    For ``init_method='svd'`` the search optimises both SVD rank and TT-rank
    cap to fit the budget (two independent knobs: SVD rank controls quality,
    TT-rank cap controls param count). For ``init_method='tt_svd'`` the
    ``max_rank`` directly controls both.
    """
    in_features = module.in_features
    out_features = module.out_features
    has_bias = module.bias is not None
    dtype = module.weight.dtype
    device = module.weight.device

    in_shapes = factorize_dim(in_features, num_factors)
    out_shapes = factorize_dim(out_features, num_factors)

    weight_np = None
    if init_method != "random":
        weight_np = module.weight.detach().cpu().numpy().astype(np.float64)

    svd_best_rank = 0  # SVD rank (0 = N/A for non-svd paths)
    if rank_override is not None:
        best_rank = max(1, int(rank_override))
    elif target_params is not None:
        if init_method == "svd":
            svd_best_rank, best_rank = _svd_rank_for_budget(weight_np, in_shapes, out_shapes, target_params, eps)
        else:  # tt_svd
            best_rank = _rank_for_budget(in_features, out_features, in_shapes, out_shapes, num_factors, target_params)
    else:
        best_rank = max(1, min(in_features, out_features) // 2)

    ranks = [1] + [best_rank] * (num_factors - 1) + [1]

    if init_method == "random":
        cls = LinearTensorLinear if layer_type == "linear" else TensorLinear
        layer = cls(in_features, out_features, in_shapes, out_shapes, ranks, bias=has_bias)
        return layer.to(device=device, dtype=dtype)

    if init_method == "svd":
        cores = svd(weight_np, in_shapes, out_shapes, max_rank=svd_best_rank, eps=eps, tt_max_rank=best_rank)
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
    importance_cutoff: Optional[float] = None,
    verbose: bool = True,
) -> dict:
    """Activation-aware compression: important layers keep more parameters.

    Parameters
    ----------
    importance_cutoff : float in (0, 1) or None
        If set, only the bottom ``cutoff`` fraction of layers by importance
        (e.g. 0.1 = the 10% least important) are compressed; the rest are
        left untouched. If None (default), all layers are compressed with
        budget allocated proportional to importance.

    Steps:
    1. Run ``calibration_prompts`` through ``model`` with forward hooks on every
       ``nn.Linear``; score each layer's importance = Frobenius norm of its
       input activations (summed across prompts/tokens).
    2. If ``importance_cutoff`` is set, select only the least-important
       ``cutoff`` fraction of layers for compression.
    3. Compute the global parameter budget: ``orig_total * (1 - pct/100)``.
    4. Allocate budget across selected layers proportional to importance.
       Layers with importance below ``min_importance`` are skipped.
    5. For each selected layer, search the TT rank that fits its allocated
       budget and build the TT replacement.
    """
    if not 0 < compression_pct < 100:
        raise ValueError(f"compression_pct must be in (0, 100), got {compression_pct}")
    if init_method not in {"random", "svd", "tt_svd"}:
        raise ValueError(f"init_method must be random|svd|tt_svd, got {init_method!r}")
    if layer_type not in {"tensor", "linear"}:
        raise ValueError(f"layer_type must be tensor|linear, got {layer_type!r}")
    if importance_cutoff is not None and not 0 < importance_cutoff <= 1.0:
        raise ValueError(f"importance_cutoff must be in (0, 1] or None, got {importance_cutoff}")

    from .activations import capture_activations, compute_importance

    linears = _collect_linears(model)
    if not linears:
        return {"orig": 0, "new": 0, "ratio": 1.0}

    dev = device or next(model.parameters()).device
    model.eval()

    # 1. capture activations
    acts = capture_activations(model, linears, tokenizer, calibration_prompts, max_length, dev)
    importance = compute_importance(acts)

    names = [n for n, _, _ in linears]
    orig_params = {name: _linear_params(m) for name, _, m in linears}
    total_orig = sum(orig_params.values())

    # 2. select which layers to compress
    if importance_cutoff is not None:
        # rank layers by importance ascending; pick the bottom cutoff fraction
        sorted_names = sorted(names, key=lambda n: importance[n])
        n_select = max(1, int(len(sorted_names) * importance_cutoff))
        selected = set(sorted_names[:n_select])
        if verbose:
            print(f"Importance cutoff {importance_cutoff}: compressing {n_select}/{len(names)} least important layers")
    else:
        selected = set(names)

    # 3. budget: compression_pct applies to the SELECTED layers' params
    selected_orig = sum(orig_params[n] for n in names if n in selected)
    selected_budget = selected_orig * (1.0 - compression_pct / 100.0)

    total_imp = sum(importance[n] for n in names if n in selected and importance[n] >= min_importance)
    if total_imp <= 0:
        if verbose:
            print("Total importance of selected layers is zero; falling back to uniform among selected.")
        total_imp = 1.0
        imp_norm = {n: 1.0 for n in names if n in selected}
    else:
        imp_norm = {n: importance[n] for n in names if n in selected}

    # 4. allocate budget proportional to importance among selected layers
    # rank-1 floor so each compressed layer can at least form valid cores
    floor_total = 0.0
    floor_per = {}
    for n, _, m in linears:
        if n not in selected:
            continue
        base = _tt_param_count(
            factorize_dim(m.in_features, num_factors),
            factorize_dim(m.out_features, num_factors),
            [1] * (num_factors + 1),
        )
        floor_per[n] = base
        floor_total += base

    above_floor = max(selected_budget - floor_total, 0.0)

    budgets: dict = {}
    for n, _, m in linears:
        if n not in selected or importance[n] < min_importance:
            budgets[n] = None
            continue
        extra = above_floor * (imp_norm[n] / total_imp) if total_imp > 0 else 0.0
        budgets[n] = floor_per[n] + extra

    # 5. compress each selected layer to its budget
    total_new = 0
    for name, parent, module in linears:
        b = budgets[name]
        if b is None:
            if verbose:
                reason = "not in cutoff selection" if importance_cutoff is not None and name not in selected else f"importance {importance[name]:.4e} < {min_importance}"
                print(f"Skipped '{name}' ({reason})")
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
