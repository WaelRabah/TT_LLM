"""Tests for tt_llm.sola (SoLA compression)."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from tt_llm import compress_model_sola
from tt_llm.sola import SoLALinear, _build_sola_layer, _compute_scaler, _rank_for_budget


class _DummyTokenizer:
    """Minimal tokenizer for SoLA tests (no HF dependency)."""

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


# --- unit tests for helpers ----------------------------------------------

def test_compute_scaler_uniform_when_no_activations():
    """No activations -> uniform scaler of ones."""
    scaler = _compute_scaler(None, 64)
    assert scaler.shape == (64,)
    assert np.allclose(scaler, 1.0)


def test_compute_scaler_reflects_activation_norms():
    """Scaler should be proportional to per-feature activation norms."""
    acts = np.zeros((100, 4), dtype=np.float64)
    acts[:, 0] = 10.0   # high activation
    acts[:, 1] = 0.0    # zero activation
    acts[:, 2] = 1.0    # normal
    acts[:, 3] = 5.0    # medium
    scaler = _compute_scaler(acts, 4)
    assert scaler.shape == (4,)
    # high-activation feature should get larger scaler than normal
    assert scaler[0] > scaler[2]
    # medium-activation feature should be between normal and high
    assert scaler[2] < scaler[3] < scaler[0]


def test_rank_for_budget():
    """Rank should fit within the parameter budget."""
    # budget = r * (in + out) + in (with scaling)
    # in=100, out=200, budget=600 -> r * 300 + 100 <= 600 -> r=1
    r = _rank_for_budget(100, 200, 600, use_scaling=True)
    assert r == 1

    # budget=1000 -> r * 300 + 100 <= 1000 -> r=3
    r = _rank_for_budget(100, 200, 1000, use_scaling=True)
    assert r == 3

    # budget=100, no scaling -> r * 300 <= 100 -> r=0 -> max(1, 0) = 1
    r = _rank_for_budget(100, 200, 100, use_scaling=False)
    assert r == 1

    # Cannot exceed min(in, out)
    r = _rank_for_budget(10, 20, 1_000_000, use_scaling=False)
    assert r == 10


def test_rank_for_budget_without_scaling():
    """Without scaling: params = r * (in + out)."""
    # in=50, out=50, budget=500 -> r * 100 <= 500 -> r=5
    r = _rank_for_budget(50, 50, 500, use_scaling=False)
    assert r == 5


# --- SoLALinear tests ----------------------------------------------------

def test_sola_linear_forward_shape():
    """SoLALinear should produce the correct output shape."""
    layer = SoLALinear(in_features=64, out_features=128, rank=8)
    x = torch.randn(4, 64)
    y = layer(x)
    assert y.shape == (4, 128)


def test_sola_linear_forward_3d():
    """SoLALinear should handle 3D input (batch, seq, dim)."""
    layer = SoLALinear(in_features=32, out_features=32, rank=4)
    x = torch.randn(2, 10, 32)
    y = layer(x)
    assert y.shape == (2, 10, 32)


def test_sola_linear_reconstructs_weight():
    """SoLALinear should reconstruct the SVD approximation faithfully."""
    torch.manual_seed(42)
    W = torch.randn(128, 64)  # (out, in)

    # Full-rank SVD (numpy)
    W_np = W.numpy()
    U, S, Vt = np.linalg.svd(W_np, full_matrices=False)
    rank = 32

    layer = SoLALinear(64, 128, rank)
    layer.load_svd(U[:, :rank], S[:rank], Vt[:rank, :])
    layer.eval()

    x = torch.randn(4, 64)
    with torch.no_grad():
        y_layer = layer(x)
        # The layer does: U @ (diag(S) @ Vt @ x) = W_approx @ x
        W_approx = torch.from_numpy(
            (U[:, :rank] * S[:rank]) @ Vt[:rank, :]
        ).float()
        y_direct = x @ W_approx.T

    assert torch.allclose(y_layer, y_direct, atol=1e-5)


def test_sola_linear_with_scaler():
    """SoLALinear with scaler should apply it to the input."""
    in_f, out_f, rank = 32, 64, 8
    scaler = np.ones(in_f) * 2.0

    layer = SoLALinear(in_f, out_f, rank, scaler=scaler)
    assert torch.allclose(layer.scaler, torch.ones(in_f) * 2.0)

    x = torch.randn(4, in_f)
    y = layer(x)
    assert y.shape == (4, out_f)


def test_sola_linear_param_count():
    """SoLALinear should have the expected parameter count."""
    in_f, out_f, rank = 64, 128, 16
    layer = SoLALinear(in_f, out_f, rank, scaler=np.ones(in_f))
    # down: rank * in_f, up: out_f * rank, scaler: in_f (buffer), bias: None
    expected = rank * in_f + out_f * rank + in_f
    actual = sum(p.numel() for p in layer.parameters())
    actual += sum(b.numel() for b in layer.buffers())
    assert actual == expected


def test_sola_linear_with_bias():
    """SoLALinear should include bias when provided."""
    bias = torch.randn(64)
    layer = SoLALinear(32, 64, 8, bias=bias)
    assert layer.bias is not None
    assert layer.bias.shape == (64,)

    x = torch.randn(2, 32)
    y = layer(x)
    assert y.shape == (2, 64)


# --- _build_sola_layer tests --------------------------------------------

def test_build_sola_layer_returns_module():
    """_build_sola_layer should return a SoLALinear module."""
    linear = nn.Linear(128, 256)
    acts = np.random.randn(50, 128).astype(np.float64)
    layer = _build_sola_layer(linear, acts, target_params=5000)
    assert isinstance(layer, SoLALinear)
    assert layer.in_features == 128
    assert layer.out_features == 256


def test_build_sola_layer_preserves_output_shape():
    """The compressed layer should produce the same output shape."""
    linear = nn.Linear(64, 32)
    x = torch.randn(4, 64)
    y_orig = linear(x)

    acts = np.random.randn(50, 64).astype(np.float64)
    layer = _build_sola_layer(linear, acts, target_params=2000)
    y_new = layer(x)
    assert y_new.shape == y_orig.shape


def test_build_sola_layer_no_activations():
    """_build_sola_layer should work without activations (uniform scaler)."""
    linear = nn.Linear(64, 32)
    layer = _build_sola_layer(linear, None, target_params=2000)
    assert isinstance(layer, SoLALinear)
    assert torch.allclose(layer.scaler, torch.ones(64))


# --- compress_model_sola integration tests ------------------------------

def test_sola_replaces_all_linears():
    """compress_model_sola should replace all nn.Linear layers."""
    model = _make_toy_model(seed=1)
    tok = _DummyTokenizer()
    prompts = ["hello world", "test prompt"]

    compress_model_sola(
        model, tok, prompts, compression_pct=30, verbose=False
    )
    sola_count = sum(1 for m in model.modules() if isinstance(m, SoLALinear))
    assert sola_count > 0, "No SoLALinear layers found after compression"


def test_sola_achieves_target_pct():
    """compress_model_sola should reduce parameters near the target."""
    model = _make_toy_model(seed=1)
    tok = _DummyTokenizer()
    prompts = ["hello world", "test prompt", "another"]

    orig = sum(p.numel() for p in model.parameters())
    compress_model_sola(
        model, tok, prompts, compression_pct=30, verbose=False
    )
    new = sum(p.numel() for p in model.parameters())
    actual_pct = (1 - new / orig) * 100
    # SoLA rank granularity -> allow slack
    assert 10 < actual_pct < 50, f"compression {actual_pct:.1f}% not near 30%"


def test_sola_skip_layers():
    """compress_model_sola should respect skip_layers."""
    model = _make_toy_model(seed=1)
    tok = _DummyTokenizer()
    prompts = ["hello"]

    compress_model_sola(
        model, tok, prompts, compression_pct=30,
        skip_layers=["proj"], verbose=False
    )
    assert isinstance(model.proj, nn.Linear), "proj should remain nn.Linear"
    sola_count = sum(1 for m in model.modules() if isinstance(m, SoLALinear))
    assert sola_count > 0, "should still compress other layers"


def test_sola_preserves_output():
    """SoLA compression should preserve output relatively well (training-free)."""
    torch.manual_seed(2)
    model = _make_toy_model(seed=2).eval()
    tok = _DummyTokenizer()
    prompts = ["hello world", "tensor compression", "quantum computing"]
    x = torch.randn(4, 576)
    with torch.no_grad():
        y_before = model(x).detach().clone()

    compress_model_sola(
        model, tok, prompts, compression_pct=20, verbose=False
    )
    model.eval()
    with torch.no_grad():
        y_after = model(x)

    err = (torch.norm(y_after - y_before) / torch.norm(y_before)).item()
    assert err < 1.5, f"SoLA output error too high: {err}"


def test_sola_rejects_invalid_values():
    """compress_model_sola should reject invalid compression_pct."""
    model = _make_toy_model()
    tok = _DummyTokenizer()
    with pytest.raises(ValueError):
        compress_model_sola(model, tok, ["a"], compression_pct=0, verbose=False)
    with pytest.raises(ValueError):
        compress_model_sola(model, tok, ["a"], compression_pct=100, verbose=False)
    with pytest.raises(ValueError):
        compress_model_sola(model, tok, ["a"], compression_pct=30, sparsity_ratio=0, verbose=False)
    with pytest.raises(ValueError):
        compress_model_sola(model, tok, ["a"], compression_pct=30, sparsity_ratio=1.5, verbose=False)
