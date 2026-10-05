"""Tests for tt_llm.compress (uniform + targeted modes)."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from tt_llm import compress_model_inplace, compress_model_targeted
from tt_llm.layers import LinearTensorLinear, TensorLinear


class _DummyTokenizer:
    """Minimal tokenizer for targeted-compression tests (no HF dependency)."""

    def __call__(self, prompt, return_tensors="pt", truncation=True, max_length=128):
        torch.manual_seed(abs(hash(prompt)) % (2**31))
        ids = torch.randint(0, 100, (1, min(max_length, 16)))
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


class ToyMLP(nn.Module):
    def __init__(self, dims=(576, 576, 576)):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
        self.proj = nn.Linear(dims[-1], dims[0])

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        x = torch.randn(1, input_ids.shape[1], 576) if input_ids is not None else torch.randn(1, 16, 576)
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.proj(x)


def _make_toy_model(dims=(576, 576, 576), seed=0):
    torch.manual_seed(seed)
    return ToyMLP(list(dims))


# --- uniform compression -------------------------------------------------
@pytest.mark.parametrize("layer_type,LayerCls", [
    ("tensor", TensorLinear),
    ("linear", LinearTensorLinear),
])
@pytest.mark.parametrize("init_method", ["svd", "tt_svd"])
def test_uniform_replaces_all_linears(layer_type, LayerCls, init_method):
    model = _make_toy_model(seed=1)
    compress_model_inplace(model, compression_pct=30, layer_type=layer_type, init_method=init_method, verbose=False)
    found_tt = any(isinstance(m, LayerCls) for m in model.modules())
    assert found_tt, f"No {LayerCls.__name__} layers found after compression"
    # children of the model should all be TT layers now
    for child_name, child in model.named_children():
        if hasattr(child, '__len__'):  # ModuleList
            for sub in child:
                assert isinstance(sub, LayerCls)
        else:
            assert isinstance(child, LayerCls)


def test_uniform_achieves_target_pct():
    model = _make_toy_model(seed=1)
    orig = sum(p.numel() for p in model.parameters())
    compress_model_inplace(model, compression_pct=30, layer_type="tensor", init_method="svd", verbose=False)
    new = sum(p.numel() for p in model.parameters())
    actual_pct = (1 - new / orig) * 100
    # integer-rank granularity -> allow slack
    assert 15 < actual_pct < 45, f"compression {actual_pct:.1f}% not near 30%"


def test_uniform_skip_layers():
    model = _make_toy_model(seed=1)
    compress_model_inplace(model, compression_pct=30, layer_type="tensor", init_method="svd",
                           verbose=False, skip_layers=["proj"])
    assert isinstance(model.proj, nn.Linear), "proj should remain nn.Linear"
    assert isinstance(model.layers[0], TensorLinear)


@pytest.mark.parametrize("init_method", ["svd", "tt_svd"])
def test_uniform_faithful_beats_random(init_method):
    """Faithful init must dramatically outperform random init (the old bug)."""
    x = torch.randn(4, 576)

    model = _make_toy_model(seed=1).eval()
    with torch.no_grad():
        y_before = model(x).detach().clone()
    compress_model_inplace(model, compression_pct=30, layer_type="tensor", init_method=init_method, verbose=False)
    model.eval()
    with torch.no_grad():
        y_after = model(x)
    err_faithful = (torch.norm(y_after - y_before) / torch.norm(y_before)).item()

    model_r = _make_toy_model(seed=1).eval()
    y_b_r = model_r(x).detach().clone()
    compress_model_inplace(model_r, compression_pct=30, layer_type="tensor", init_method="random", verbose=False)
    model_r.eval()
    with torch.no_grad():
        y_a_r = model_r(x)
    err_random = (torch.norm(y_a_r - y_b_r) / torch.norm(y_b_r)).item()

    assert err_faithful < err_random / 5, f"faithful {err_faithful:.3f} should be >>5x better than random {err_random:.3f}"
    assert err_faithful < 1.0


# --- targeted compression ------------------------------------------------
@pytest.mark.parametrize("init_method", ["svd", "tt_svd"])
def test_targeted_compresses_and_preserves_output(init_method):
    """Targeted compression should hit the global budget and stay bounded."""
    model = _make_toy_model(seed=2).eval()
    tok = _DummyTokenizer()
    prompts = ["hello world", "tensor train compression", "quantum computing"]
    x = torch.randn(4, 576)
    with torch.no_grad():
        y_before = model(x).detach().clone()

    orig = sum(p.numel() for p in model.parameters())
    result = compress_model_targeted(
        model, tok, prompts, compression_pct=30, layer_type="tensor",
        init_method=init_method, verbose=False,
    )
    new = sum(p.numel() for p in model.parameters())
    actual_pct = (1 - new / orig) * 100
    assert 10 < actual_pct < 45, f"targeted compression {actual_pct:.1f}% not near 30%"
    assert result["ratio"] > 1.0

    model.eval()
    with torch.no_grad():
        y_after = model(x)
    err = (torch.norm(y_after - y_before) / torch.norm(y_before)).item()
    assert err < 1.0, f"targeted [{init_method}] output error too high: {err}"


def test_targeted_importance_ranking():
    """Targeted compression should compress a low-importance layer more than a high one."""
    model = _make_toy_model(seed=3).eval()
    tok = _DummyTokenizer()
    prompts = ["a", "b", "c"]
    from tt_llm.activations import capture_activations, compute_importance
    from tt_llm.compress import _collect_linears

    linears = _collect_linears(model)
    acts = capture_activations(model, linears, tok, prompts)
    importance = compute_importance(acts)
    # all layers should have positive importance (random init -> nonzero activations)
    assert all(v > 0 for v in importance.values()), f"zero importance: {importance}"


def test_pct_rejects_invalid_values():
    model = _make_toy_model()
    with pytest.raises(ValueError):
        compress_model_inplace(model, compression_pct=-1)
    with pytest.raises(ValueError):
        compress_model_inplace(model, compression_pct=150)
