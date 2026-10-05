# TT_LLM

Tensor-Train (TT) compression for transformer linear layers.

Replaces every `nn.Linear` in a HuggingFace causal LM with a TT-decomposed layer
whose cores are initialised from the pretrained weights via **SVD** or **TT-SVD**
(rather than random init, which destroys the pretrained knowledge).

## Structure

```
tt_llm/
├── layers.py           # TensorLinear & LinearTensorLinear (TT-decomposed Linear)
├── decompositions.py   # factorize_dim, tt_svd, svd, reconstruct_matrix
├── activations.py      # forward-hook activation capture + importance scoring
└── compress.py         # compress_model_inplace (uniform) / compress_model_targeted
tests/                  # pytest: decompositions, layers, compression
TT_LLM.ipynb            # Colab notebook: load model, compress, compare init methods
```

## Install

```bash
pip install -e .            # core (torch + numpy)
pip install -e .[models]    # + transformers/accelerate/huggingface_hub for the notebook
pip install -e .[test]      # + pytest
```

## Usage

### Uniform compression

```python
from tt_llm import compress_model_inplace

# Cut parameters by 30% (every nn.Linear compressed by the same amount)
compress_model_inplace(
    model,
    compression_pct=10,     # 30 = remove 30% of params (keep 70%)
    layer_type="tensor",    # "tensor" (TensorLinear) or "linear" (LinearTensorLinear)
    init_method="svd",       # "svd" | "tt_svd" | "random"
)
```

### Targeted (activation-aware) compression

```python
from tt_llm import compress_model_targeted

# Compress unimportant layers more aggressively than important ones
compress_model_targeted(
    model, tokenizer,
    calibration_prompts=["prompt1", "prompt2", ...],  # sample inputs
    compression_pct=30,
    layer_type="tensor",
    init_method="svd",
    importance_cutoff=0.1,  # only compress the 10% least important layers
)
```

The targeted mode runs the calibration prompts through the model with forward
hooks on every `nn.Linear`, scores each layer's importance by the Frobenius norm
of its input activations, and allocates the global parameter budget proportional
to importance — important layers keep more parameters, unimportant ones are
compressed more aggressively.

When `importance_cutoff` is set (e.g. `0.1`), only the bottom ``cutoff`` fraction
of layers by importance are compressed; the rest are left untouched.

### Initialisation methods

| `init_method` | Description                                                                 |
|---------------|-----------------------------------------------------------------------------|
| `svd`         | Global best-rank-`r` SVD of `W`, formatted as TT cores. Optimal Frobenius. |
| `tt_svd`      | Canonical TT-SVD with per-unfolding truncation.                              |
| `random`      | Random TT cores (ablation; reproduces the original notebook's behaviour).  |

## Tests

```bash
pytest tests/
```
