"""Activation capture for targeted compression.

Forward hooks are registered on every ``nn.Linear`` in the model; the input
activations are collected across a set of calibration prompts and aggregated
into a per-layer importance score (Frobenius norm). Layers with low importance
contribute little signal and are candidates for aggressive compression.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn


@contextmanager
def _hooks(modules: Sequence[Tuple[str, nn.Linear]], activations: Dict[str, List[torch.Tensor]]):
    """Register forward hooks that capture input tensors; cleanup on exit."""
    handles = []

    def _make_hook(name):
        def _hook(module, inp, out):
            x = inp[0] if isinstance(inp, tuple) else inp
            activations.setdefault(name, []).append(x.detach().to("cpu", dtype=torch.float32))
        return _hook

    for name, mod in modules:
        handles.append(mod.register_forward_hook(_make_hook(name)))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def capture_activations(
    model: nn.Module,
    linears: Sequence[Tuple[str, nn.Module, nn.Linear]],
    tokenizer,
    prompts: Sequence[str],
    max_length: int = 128,
    device: str = "cpu",
) -> Dict[str, List[torch.Tensor]]:
    """Run ``prompts`` through ``model`` and capture input activations per layer.

    Returns a dict mapping dotted layer name -> list of input activation tensors
    (one per prompt). Each tensor has shape ``[batch, seq, in_features]``.
    """
    model.eval()
    activations: Dict[str, List[torch.Tensor]] = {}
    hook_targets = [(name, mod) for name, _, mod in linears]

    with _hooks(hook_targets, activations), torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            model(**inputs, use_cache=False)

    return activations


def compute_importance(activations: Dict[str, List[torch.Tensor]]) -> Dict[str, float]:
    """Score each layer's importance as the Frobenius norm of its activations.

    Aggregates across all captured prompts and tokens via sum of squared
    elements, then takes the square root (so the score is energy, not magnitude).
    """
    importance: Dict[str, float] = {}
    for name, acts in activations.items():
        if not acts:
            importance[name] = 0.0
            continue
        sq = sum(float((a ** 2).sum()) for a in acts)
        importance[name] = math.sqrt(sq)
    return importance
