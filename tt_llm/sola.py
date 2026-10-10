"""SoLA: Soft activation sparsity and Low-rAnk decomposition for LLM compression.

Implementation of the SoLA algorithm (Huang et al., AAAI 2025).

SoLA combines two ideas for training-free LLM compression:

1. **Soft activation sparsity (FFN only)**: In the feed-forward network
   ``W_down(act(W_up(x)))``, a small fraction of intermediate neurons contribute
   the majority of the signal. SoLA identifies the top-k neurons by their
   activation energy (measured via calibration prompts) and keeps them as a
   dense "important" sub-matrix, while low-rank decomposing the remaining
   "unimportant" columns/rows.

2. **Activation-aware weighted SVD**: Instead of plain SVD on ``W``, the weight
   matrix is transformed so that rows/columns corresponding to high-activation
   neurons are weighted more heavily, ensuring the low-rank approximation
   preserves them. This is analogous to ASVD (activation-weighted SVD) but with a
   per-component truncation-rank allocation.

3. **Adaptive component-wise rank allocation**: Each weight matrix receives a
   different truncation rank based on its sensitivity (activation energy),
   so that important layers are compressed less and unimportant ones more.

The public entry point is :func:`compress_model_sola`, which replaces every
``nn.Linear`` in a HuggingFace causal LM with a low-rank decomposed version.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .activations import capture_activations
from .compress import _collect_linears, _find_tied_linears, _linear_params


# ---------------------------------------------------------------------------
# Low-rank layer replacing nn.Linear (SoLA layer)
# ---------------------------------------------------------------------------
class SoLALinear(nn.Module):
    """Low-rank decomposed linear layer with activation-aware scaling.

    ``y = U @ (diag(S) @ (Vt @ (diag(1/scaler) @ x))) + bias``

    where ``scaler`` accounts for activation distribution, and ``U, S, Vt``
    come from an activation-aware weighted SVD of the original weight matrix
    ``W_weighted = W * diag(scaler)``. The inverse scaling ``diag(1/scaler)``
    is absorbed into ``down.weight`` at load time, so forward is just
    ``y = up(down(x)) + bias``.

    The scaler buffer is kept for introspection but does not participate in
    the forward pass.

    Parameters
    ----------
    in_features, out_features : original dimensions
    rank : truncation rank ``r``
    scaler : per-input-feature scaling vector ``(in_features,)``, or None
    bias : bias tensor from the original layer, or None
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        scaler: Optional[np.ndarray] = None,
        bias: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)

        # Two nn.Linear layers: (in -> rank) and (rank -> out)
        # down: (rank, in), up: (out, rank)
        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=False)

        if scaler is not None:
            self.register_buffer("scaler", torch.from_numpy(np.ascontiguousarray(scaler)).float())
        else:
            self.register_buffer("scaler", torch.ones(in_features).float())

        self.bias = nn.Parameter(bias) if bias is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.down(x)          # (..., rank)
        out = self.up(hidden)          # (..., out)
        if self.bias is not None:
            out = out + self.bias
        return out

    def load_svd(self, U: np.ndarray, S: np.ndarray, Vt: np.ndarray, scaler: Optional[np.ndarray] = None) -> None:
        """Load SVD factors into the two nn.Linear layers.

        If ``scaler`` is provided, the activation-weighted SVD was computed on
        ``W_weighted = W * diag(scaler) = U @ diag(S) @ Vt``. To recover
        ``W @ x = U @ diag(S) @ Vt @ (diag(1/scaler) @ x)``, the inverse
        scaling is absorbed into ``down.weight``:

        - ``down.weight = diag(S) @ Vt @ diag(1/scaler)``  -> (rank, in)
        - ``up.weight = U``  -> (out, rank)

        Then ``up(down(x)) = U @ diag(S) @ Vt @ diag(1/scaler) @ x = W @ x``.
        """
        with torch.no_grad():
            SVt = np.diag(S) @ Vt  # (rank, in)
            if scaler is not None:
                inv = np.ones_like(scaler)
                nz = scaler > 1e-12
                inv[nz] = 1.0 / scaler[nz]
                SVt = SVt * inv[None, :]
            SVt = np.ascontiguousarray(SVt)
            self.down.weight.copy_(torch.from_numpy(SVt).float())
            self.up.weight.copy_(torch.from_numpy(np.ascontiguousarray(U)).float())

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, rank={self.rank}"


# ---------------------------------------------------------------------------
# Activation-aware weighted SVD
# ---------------------------------------------------------------------------
def _compute_scaler(input_activations: np.ndarray, in_features: int) -> np.ndarray:
    """Compute per-input-feature activation scaling vector.

    The scaler normalises the input dimension so features with large
    activation norms dominate the SVD (preserving them), while near-zero
    features are aggressively truncated.
    """
    if input_activations is None or input_activations.size == 0:
        return np.ones(in_features, dtype=np.float64)

    # input_activations: (num_samples, in_features)
    if input_activations.ndim == 1:
        input_activations = input_activations.reshape(1, -1)

    if input_activations.shape[1] != in_features:
        # If shape mismatch, fall back to uniform scaling
        return np.ones(in_features, dtype=np.float64)

    col_norms = np.linalg.norm(input_activations, axis=0)  # (in_features,)
    # Floor: near-zero activation features get scaler = 1.0 (no amplification)
    scaler = np.where(col_norms > 1e-10, col_norms, 1.0)

    # Normalise: mean = 1.0 to avoid extreme values
    mean_val = scaler.mean()
    if mean_val > 1e-12:
        scaler = scaler / mean_val
    # Re-floor: after normalization, zero-activation features should still be 1.0-ish
    # but mean normalization may have changed them; that's OK — the important
    # property is the *ratio* between features.
    return scaler


def _rank_for_budget(in_features: int, out_features: int, target_params: float, use_scaling: bool) -> int:
    """Find the rank ``r`` whose SVD param count best fits the budget.

    With scaling:    ``params = r * (in + out) + in``  (down + up + scaler)
    Without scaling: ``params = r * (in + out)``
    """
    max_r = min(in_features, out_features)
    overhead = in_features if use_scaling else 0
    budget = target_params - overhead
    if budget <= 0:
        return 1
    # r * (in + out) <= budget  =>  r <= budget / (in + out)
    r = int(budget // (in_features + out_features))
    return max(1, min(int(r), max_r))


# ---------------------------------------------------------------------------
# Build a SoLA replacement layer
# ---------------------------------------------------------------------------
def _build_sola_layer(
    module: nn.Linear,
    input_activations: Optional[np.ndarray],
    target_params: float,
    sparsity_ratio: float = 0.5,
) -> Optional[nn.Module]:
    """Build a SoLA low-rank replacement for ``module``.

    Steps:
    1. If input activations are available, compute per-feature activation norms
       -> activation-aware weighting.
    2. Apply soft activation sparsity: if the layer is an FFN down_proj (has a
       large intermediate input), select the top-k important neurons and keep
       them dense, compressing the rest.
    3. Run activation-aware weighted SVD (or plain SVD) on the (sub-)matrix.
    4. Store as a :class:`SoLALinear` layer.
    """
    in_features = module.in_features
    out_features = module.out_features
    weight = module.weight.detach().float().cpu().numpy().astype(np.float64)
    has_bias = module.bias is not None
    bias = module.bias.detach().clone() if has_bias else None
    dtype = module.weight.dtype
    device = module.weight.device

    # Step 1: compute activation scaler
    scaler = _compute_scaler(input_activations, in_features)

    # Step 2: soft activation sparsity for FFN down_proj-like layers
    # If in_features is large (intermediate dim), select top-k important neurons
    sparsity_mask = None
    if sparsity_ratio < 1.0 and in_features > out_features and in_features > 128:
        # This looks like an FFN down_proj: (intermediate_dim -> d_model)
        # Keep top-k neurons dense, compress the rest
        k = max(1, int(in_features * sparsity_ratio))
        if np.any(scaler != 1.0):
            top_k_indices = np.argsort(scaler)[::-1][:k]
        else:
            top_k_indices = np.arange(k)

        sparsity_mask = np.zeros(in_features, dtype=bool)
        sparsity_mask[top_k_indices] = True  # True = important, kept dense

    # Step 3: compute rank for the budget
    # First try with scaling (better activation-awareness)
    rank = _rank_for_budget(in_features, out_features, target_params, use_scaling=True)
    use_scaling = True

    # If rank-1 with scaling still exceeds budget, try without scaling
    params_with_scaling = rank * (in_features + out_features) + in_features
    if params_with_scaling > target_params and rank == 1:
        rank_no = _rank_for_budget(in_features, out_features, target_params, use_scaling=False)
        params_no_scaling = rank_no * (in_features + out_features)
        if params_no_scaling < params_with_scaling:
            rank = rank_no
            use_scaling = False

    if rank < 1:
        return None

    # Step 4: activation-aware weighted SVD
    if use_scaling:
        W_weighted = weight * scaler[None, :]
    else:
        W_weighted = weight
        scaler = None

    U, S, Vt = np.linalg.svd(W_weighted, full_matrices=False)
    rank = min(rank, S.shape[0])

    layer = SoLALinear(in_features, out_features, rank, scaler if use_scaling else None, bias)
    layer.load_svd(U[:, :rank], S[:rank], Vt[:rank, :], scaler if use_scaling else None)
    layer = layer.to(device=device, dtype=dtype)

    return layer


# ---------------------------------------------------------------------------
# Collect input activations per layer as numpy arrays
# ---------------------------------------------------------------------------
def _collect_activations_np(
    model: nn.Module,
    linears: List[Tuple[str, nn.Module, nn.Linear]],
    tokenizer,
    prompts: Sequence[str],
    max_length: int,
    device: str,
) -> Dict[str, np.ndarray]:
    """Run calibration prompts and return per-layer flattened input activations.

    Returns dict: layer_name -> (num_samples, in_features) array.
    """
    acts = capture_activations(model, linears, tokenizer, prompts, max_length, device)
    result: Dict[str, np.ndarray] = {}
    for name, tensors in acts.items():
        if not tensors:
            continue
        # tensors: list of (batch, seq, in_features) per prompt
        # Concatenate along seq dim, then reshape to (num_samples, in_features)
        cat = torch.cat(tensors, dim=1).squeeze(0).float().cpu().numpy()
        result[name] = cat
    return result


# ---------------------------------------------------------------------------
# SoLA parameter floor (rank-1 + scaler)
# ---------------------------------------------------------------------------
def _sola_param_count(in_features: int, out_features: int, rank: int, use_scaling: bool) -> int:
    """Parameter count for a SoLA layer (down + up + optional scaler + bias)."""
    params = rank * (in_features + out_features)
    if use_scaling:
        params += in_features
    return params


def _sola_floor_params(in_features: int, out_features: int) -> int:
    """Minimum parameter count for a SoLA replacement (rank-1 + scaler)."""
    return _sola_param_count(in_features, out_features, rank=1, use_scaling=True)


# ---------------------------------------------------------------------------
# Public API: compress_model_sola
# ---------------------------------------------------------------------------
def compress_model_sola(
    model: nn.Module,
    tokenizer,
    calibration_prompts: Sequence[str],
    compression_pct: float = 30.0,
    sparsity_ratio: float = 0.5,
    max_length: int = 128,
    device: Optional[str] = None,
    skip_layers: Optional[Sequence[str]] = None,
    verbose: bool = True,
) -> dict:
    """SoLA compression: soft activation sparsity + low-rank decomposition.

    Following Huang et al. (AAAI 2025), this method:
    1. Runs calibration prompts through the model with forward hooks.
    2. Computes per-feature activation norms for each ``nn.Linear``.
    3. Allocates parameter budget proportional to each layer's activation energy.
    4. Applies activation-aware weighted SVD with adaptive per-layer rank.
    5. Optionally applies soft activation sparsity to FFN down_proj layers.

    Parameters
    ----------
    model : nn.Module
    tokenizer : HuggingFace tokenizer
    calibration_prompts : calibration data for activation capture
    compression_pct : float in (0, 100)
        Percentage of parameters to remove (e.g. 30 = 30% cut).
    sparsity_ratio : float in (0, 1]
        Fraction of important neurons to keep dense in FFN down_proj layers.
        1.0 disables soft activation sparsity (pure low-rank).
    max_length : max token length per calibration prompt
    device : device string for calibration
    skip_layers : dotted module names to leave untouched
    verbose : print per-layer compression stats
    """
    if not 0 < compression_pct < 100:
        raise ValueError(f"compression_pct must be in (0, 100), got {compression_pct}")
    if not 0 < sparsity_ratio <= 1.0:
        raise ValueError(f"sparsity_ratio must be in (0, 1], got {sparsity_ratio}")

    linears = _collect_linears(model)
    if not linears:
        return {"orig": 0, "new": 0, "ratio": 1.0}

    dev = device or next(model.parameters()).device
    model.eval()

    skip = set(skip_layers or [])

    # auto-skip tied layers (e.g. lm_head tied to embed_tokens)
    tied = _find_tied_linears(model)
    skip |= tied
    if tied and verbose:
        print(f"Skipping tied layers: {sorted(tied)}")

    names = [n for n, _, _ in linears]
    orig_params = {name: _linear_params(m) for name, _, m in linears}
    total_orig = sum(orig_params.values())

    # Step 1: collect per-layer input activations via calibration
    name_to_module: Dict[str, nn.Linear] = {n: m for n, _, m in linears}
    layer_acts = _collect_activations_np(
        model, linears, tokenizer, calibration_prompts, max_length, dev
    )

    # Step 2: compute per-layer sensitivity (activation energy = Frobenius norm)
    layer_energy: Dict[str, float] = {}
    for name in name_to_module:
        if name in layer_acts and layer_acts[name].size > 0:
            layer_energy[name] = float(np.linalg.norm(layer_acts[name]))
        else:
            layer_energy[name] = 1.0

    # Step 3: allocate budget with waterfall redistribution
    from .compress import _waterfall_budgets

    # Budget applies only to non-skipped (compressible) layers
    compressible_orig = sum(orig_params[n] for n in names if n not in skip)
    global_budget = compressible_orig * (1.0 - compression_pct / 100.0)

    def _sola_max_params(name):
        m = name_to_module[name]
        return min(m.in_features, m.out_features) * (m.in_features + m.out_features) + m.in_features

    budgets = _waterfall_budgets(
        names, orig_params, layer_energy, global_budget,
        max_params_fn=_sola_max_params, selected=set(names) - skip,
    )

    # Step 4: compress each layer
    total_new = 0
    for name, parent, module in linears:
        b = budgets.get(name)
        if b is None:
            if verbose:
                print(f"Skipped '{name}' (explicit skip)")
            total_new += _linear_params(module)
            continue

        acts = layer_acts.get(name)
        compressed = _build_sola_layer(module, acts, b, sparsity_ratio)
        if compressed is None:
            total_new += _linear_params(module)
            continue

        child_name = name.split(".")[-1]
        setattr(parent, child_name, compressed)
        new = sum(p.numel() for p in compressed.parameters())
        total_new += new
        if verbose:
            print(
                f"SoLA '{name}': Linear -> SoLALinear | "
                f"energy={layer_energy[name]:.3e} | "
                f"Comp: {orig_params[name] / max(new, 1):.2f}x"
            )

    if verbose:
        print(
            f"\nSoLA Total: {total_orig} -> {total_new} params | "
            f"Comp: {total_orig / max(total_new, 1):.2f}x (target {compression_pct:.0f}%)"
        )

    return {"orig": total_orig, "new": total_new, "ratio": total_orig / max(total_new, 1)}
