"""Tests for tt_llm.compress."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from tt_llm import compress_model_inplace
from tt_llm.layers import LinearTensorLinear, TensorLinear


def _make_toy_model(dims=(576, 576, 576), seed=0):
    torch.manual_seed(seed)

    class ToyMLP(nn.Module):
        def __init__(self, dims):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
            self.proj = nn.Linear(dims[-1], dims[0])

        def forward(self, x):
            for layer in self.layers:
                x = torch.relu(layer(x))
            return self.proj(x)

    return ToyMLP(list(dims))


@pytest.mark.parametrize("layer_type,LayerCls", [
    ("tensor", TensorLinear),
    ("linear", LinearTensorLinear),
])
@pytest.mark.parametrize("init_method", ["svd", "tt_svd"])
def test_compress_replaces_all_linears(layer_type, LayerCls, init_method):
    model = _make_toy_model(dims=(576, 576, 576), seed=1)
    compress_model_inplace(model, target_ratio=1.43, layer_type=layer_type, init_method=init_method, verbose=False)
    # No top-level nn.Linear should remain (LinearTensorLinear uses internal
    # nn.Linear core modules, so we only check user-facing layers).
    top_level_linears = [
        m for m in model.modules()
        if isinstance(m, nn.Linear) and not any(isinstance(p, (TensorLinear, LinearTensorLinear)) for p in model.modules())
    ]
    # Simpler/robust: check that every direct child of the model or its submodules
    # that was originally nn.Linear is now a TT layer.
    found_tt = any(isinstance(m, LayerCls) for m in model.modules())
    assert found_tt, f"No {LayerCls.__name__} layers found after compression"
    # The original linear layer modules (layers[i], proj) should now be TT layers.
    for child_name, child in model.named_children():
        if hasattr(child, '__len__'):  # ModuleList
            for sub in child:
                assert isinstance(sub, LayerCls), f"{child_name} child is {type(sub).__name__}, not TT"
        else:
            assert isinstance(child, LayerCls), f"{child_name} is {type(child).__name__}, not TT"


def test_compress_achieves_target_ratio():
    model = _make_toy_model(dims=(576, 576, 576), seed=1)
    orig_params = sum(p.numel() for p in model.parameters())
    compress_model_inplace(model, target_ratio=1.43, layer_type="tensor", init_method="svd", verbose=False)
    new_params = sum(p.numel() for p in model.parameters())
    actual_ratio = orig_params / max(new_params, 1)
    # Allow generous slack (rank search is integer-granular and we keep biases)
    assert 1.1 < actual_ratio < 1.8, f"compression ratio {actual_ratio:.2f} not near 1.43"


def _max_linear_IRQ_rank(model):
    """Return upper bound on effective rank we'd search for these layers."""
    in_features = model.layers[0].in_features
    out_features = model.layers[0].out_features
    return min(in_features, out_features)


@pytest.mark.parametrize("init_method", ["svd", "tt_svd"])
def test_compressed_output_close_to_original(init_method):
    """End-to-end: SVD/TT-SVD init should dramatically outperform random init.

    The original notebook's random-init compression produced garbage (rel err
    ~10+). Faithful initialisation (svd / tt_svd) preserves the pretrained
    weights up to SVD truncation, so should be dramatically better. We don't
    require < 0.5 absolute error because 1.43x compression of a *random* weight
    matrix (no low-rank structure) is genuinely lossy; what matters is the gap
    vs random init.
    """
    x = torch.randn(4, 576)

    # faithful (svd / tt_svd) model
    model_faithful = _make_toy_model(dims=(576, 576, 576), seed=1)
    model_faithful.eval()
    with torch.no_grad():
        y_before = model_faithful(x).detach().clone()
    compress_model_inplace(model_faithful, target_ratio=1.43, layer_type="tensor", init_method=init_method, verbose=False)
    model_faithful.eval()
    with torch.no_grad():
        y_after_faithful = model_faithful(x)
    rel_err_faithful = (torch.norm(y_after_faithful - y_before) / torch.norm(y_before)).item()

    # random-init model (original notebook behaviour): should be much worse
    model_random = _make_toy_model(dims=(576, 576, 576), seed=1)
    model_random.eval()
    y_before_random = model_random(x).detach().clone()
    compress_model_inplace(model_random, target_ratio=1.43, layer_type="tensor", init_method="random", verbose=False)
    model_random.eval()
    with torch.no_grad():
        y_after_random = model_random(x)
    rel_err_random = (torch.norm(y_after_random - y_before_random) / torch.norm(y_before_random)).item()

    # Faithful init must be dramatically better than random (which destroys
    # the pretrained weights entirely).
    assert rel_err_faithful < rel_err_random / 5, (
        f"[{init_method}] faithful ({rel_err_faithful:.3f}) should be >>5x better "
        f"than random ({rel_err_random:.3f})"
    )
    assert rel_err_faithful < 1.0, f"[{init_method}] faithful rel error too high: {rel_err_faithful}"
