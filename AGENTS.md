# AGENTS.md

## Commands

- **Run tests**: `.venv/bin/python -m pytest tests/` (or `pytest tests/` if installed)
- **Lint / typecheck**: none configured
- **Install (dev)**: `python3 -m venv .venv && .venv/bin/pip install -e .[test]`
  - Core deps: `torch`, `numpy` (CPU torch is fine: `pip install torch --index-url https://download.pytorch.org/whl/cpu`)
- **Validate notebook JSON**: `.venv/bin/python -c "import json; json.load(open('TT_LLM.ipynb'))"`

## Architecture

- `tt_llm/layers.py`: `TensorLinear` (raw TT-core params) and `LinearTensorLinear`
  (TT cores stored as `nn.Linear` modules). Both implement the same TT-matrix
  contraction via `torch.tensordot`.
- `tt_llm/decompositions.py`: `factorize_dim` (balanced integer factorisation),
  `tt_svd` (canonical TT-SVD), `svd` (global rank-r SVD formatted as TT),
  `reconstruct_matrix` (contract cores back to 2D).
- `tt_llm/compress.py`: `compress_model_inplace` recursively swaps `nn.Linear`
  for TT layers, searching ranks to hit `target_ratio`.

## Key facts (learned the hard way)

- A globally rank-`r` matrix can have TT-ranks >> `r` (unfolding ranks exceed
  the matrix rank). So `tt_svd(W, max_rank=r)` can over-truncate.
- `svd()` leaves TT-ranks unbounded by default; pass `tt_max_rank` to cap them
  for a parameter budget.
- `factorize_dim` must search ALL divisors (not a narrow window) or it
  degenerates to `1` for dims like 1536.
- TT cores have layout `[r_{k-1}, i_k, o_k, r_k]`; `reconstruct_matrix` contracts
  to interleaved `(i_1, o_1, ..., i_d, o_d)` then transposes to `[out, in]`.
