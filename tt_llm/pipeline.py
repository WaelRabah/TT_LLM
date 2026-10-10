"""End-to-end pipeline: compress → SFT → KD → evaluate.

Usage from Colab notebook:

    from tt_llm.pipeline import run_pipeline
    results = run_pipeline(
        method="tt_llm",       # or "sola"
        compression_pct=15,
        sft_epochs=1,
        kd_epochs=1,
        eval_limit=500,
    )

All logic lives here so the notebook rarely needs updating — just pull
the repo and change parameters.
"""

from __future__ import annotations

import gc
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "HuggingFaceTB/SmolLM2-360M"


def _get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _get_dtype():
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


def _count_params(model):
    return sum(p.numel() for p in model.parameters())


def load_model(device=None, dtype=None):
    """Load the baseline SmolLM2-360M model + tokenizer."""
    device = device or _get_device()
    dtype = dtype or _get_dtype()
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype, device_map=device)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def compress_model(
    model,
    tokenizer,
    method: str = "tt_llm",
    compression_pct: float = 15,
    max_length: int = 128,
    verbose: bool = True,
):
    """Compress model in-place using TT-LLM or SoLA."""
    from .calibration import CALIBRATION_PROMPTS

    device = _get_device()
    if method == "tt_llm":
        from .compress import compress_model_targeted
        compress_model_targeted(
            model, tokenizer, CALIBRATION_PROMPTS,
            compression_pct=compression_pct,
            importance_cutoff=None,
            layer_type="linear",
            init_method="svd",
            max_length=max_length,
            device=device,
            verbose=verbose,
        )
    elif method == "sola":
        from .sola import compress_model_sola
        compress_model_sola(
            model, tokenizer, CALIBRATION_PROMPTS,
            compression_pct=compression_pct,
            sparsity_ratio=0.5,
            max_length=max_length,
            device=device,
            verbose=verbose,
        )
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'tt_llm' or 'sola'.")
    model.eval()


def run_sft(
    model,
    tokenizer,
    output_dir: str = "./checkpoints/sft",
    num_epochs: int = 1,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 2e-5,
):
    """Run SFT on Dolly-15K."""
    from .data import DollySFTDataset
    from .training import TrainConfig, train_sft

    device = _get_device()
    train_ds = DollySFTDataset(tokenizer, split="train", max_length=512)
    val_ds = DollySFTDataset(tokenizer, split="val", max_length=512)
    print(f"SFT data: {len(train_ds)} train, {len(val_ds)} val")

    config = TrainConfig(
        output_dir=output_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        bf16=(device == "cuda"),
    )
    return train_sft(model, tokenizer, train_ds, val_ds, config=config, device=device)


def run_kd(
    student,
    teacher,
    tokenizer,
    output_dir: str = "./checkpoints/kd",
    num_epochs: int = 1,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 1e-5,
    temperature: float = 2.0,
    alpha: float = 0.5,
):
    """Run KD with teacher model on Dolly-15K."""
    from .data import DollySFTDataset
    from .training import TrainConfig, train_kd

    device = _get_device()
    train_ds = DollySFTDataset(tokenizer, split="train", max_length=512)
    val_ds = DollySFTDataset(tokenizer, split="val", max_length=512)

    config = TrainConfig(
        output_dir=output_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        kd_temperature=temperature,
        kd_alpha=alpha,
        bf16=(device == "cuda"),
    )
    return train_kd(student, teacher, tokenizer, train_ds, val_ds, config=config, device=device)


def run_eval(
    model,
    tokenizer,
    limit: int | None = 500,
    batch_size: int = 8,
):
    """Run benchmark evaluation."""
    from .eval import run_benchmarks, print_results
    device = _get_device()
    results = run_benchmarks(
        model, tokenizer, device=device, batch_size=batch_size, limit=limit,
    )
    print_results(results, "Evaluation Results")
    return results


def run_pipeline(
    method: str = "tt_llm",
    compression_pct: float = 15,
    sft_epochs: int = 1,
    kd_epochs: int = 1,
    eval_limit: int | None = 500,
    skip_baseline_eval: bool = False,
):
    """Full pipeline: load → compress → SFT → KD → evaluate.

    Returns a dict with all results.
    """
    device = _get_device()
    print(f"Device: {device}")
    print(f"Method: {method}, compression: {compression_pct}%")

    # 1. Load teacher (baseline)
    print("\n=== Loading teacher model ===")
    teacher, tok = load_model()
    baseline_params = _count_params(teacher)
    print(f"Teacher params: {baseline_params:,}")

    # 2. Evaluate baseline (optional)
    baseline_results = None
    if not skip_baseline_eval:
        print("\n=== Evaluating baseline ===")
        baseline_results = run_eval(teacher, tok, limit=eval_limit)

    # 3. Load fresh student and compress
    print(f"\n=== Compressing with {method.upper()} ({compression_pct}%) ===")
    student, _ = load_model()
    t0 = time.time()
    compress_model(student, tok, method=method, compression_pct=compression_pct)
    print(f"Compression time: {time.time()-t0:.0f}s")
    compressed_params = _count_params(student)
    print(f"Student params: {compressed_params:,} "
          f"({compressed_params/baseline_params*100:.1f}% of original)")

    # 4. Quick generation test
    _test_generation(student, tok, device)

    # 5. Free teacher before SFT (reloaded for KD later)
    print("\n=== Freeing teacher VRAM for SFT ===")
    del teacher
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 6. SFT
    print("\n=== Stage 1: SFT ===")
    sft_metrics = run_sft(student, tok)
    _test_generation(student, tok, device)

    # 7. Reload teacher for KD
    print("\n=== Reloading teacher for KD ===")
    teacher, _ = load_model()

    # 8. KD
    print("\n=== Stage 2: KD ===")
    kd_metrics = run_kd(student, teacher, tok)
    _test_generation(student, tok, device)

    # 9. Free teacher before final eval
    del teacher
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 10. Final evaluation
    print("\n=== Final Evaluation ===")
    final_results = run_eval(student, tok, limit=eval_limit)

    # 11. Comparison
    print("\n=== Summary ===")
    print(f"  Params: {baseline_params:,} → {compressed_params:,}")
    if baseline_results and final_results:
        _print_comparison(baseline_results, final_results)

    # 12. Save final model
    student.save_pretrained("./checkpoints/final_compressed")
    tok.save_pretrained("./checkpoints/final_compressed")
    print("\nFinal model saved to ./checkpoints/final_compressed")

    return {
        "baseline_params": baseline_params,
        "compressed_params": compressed_params,
        "sft_metrics": sft_metrics,
        "kd_metrics": kd_metrics,
        "baseline_eval": baseline_results,
        "final_eval": final_results,
    }


def _test_generation(model, tok, device, prompt="Explain quantum computing in one sentence."):
    model.eval()
    inputs = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=50, do_sample=False)
    print(f"  Generation: {tok.decode(out[0], skip_special_tokens=True)[:150]}...")


def _print_comparison(baseline: dict, final: dict):
    print(f"\n  {'Task':25s}  {'Baseline':>10s}  {'Compressed':>10s}  {'Delta':>10s}")
    print(f"  {'-'*60}")
    all_tasks = sorted(set(list(baseline.keys()) + list(final.keys())))
    for task in all_tasks:
        b = baseline.get(task, -1)
        c = final.get(task, -1)
        delta = c - b if b >= 0 and c >= 0 else 0
        print(f"  {task:25s}  {b:10.4f}  {c:10.4f}  {delta:+10.4f}")
