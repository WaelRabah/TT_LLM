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
└── compress.py         # compress_model_inplace (recursive Linear -> TT)
tests/                  # pytest: decompositions, layers, compression
TT_LLM.ipynb            # Colab notebook: load model, compress, compare inits
```

## Install

```bash
pip install -e .            # core (torch + numpy)
pip install -e .[models]    # + transformers/accelerate/huggingface_hub for the notebook
pip install -e .[test]      # + pytest
```

## Usage

```python
from tt_llm import compress_model_inplace

# model is a HuggingFace AutoModelForCausalLM (or any nn.Module with nn.Linear)
compress_model_inplace(
    model,
    target_ratio=1.42857,   # ~30% parameter reduction
    layer_type="tensor",   # "tensor" (TensorLinear) or "linear" (LinearTensorLinear)
    init_method="svd",      # "svd" | "tt_svd" | "random"
)
```

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
