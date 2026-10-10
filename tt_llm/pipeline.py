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


def _is_tpu():
    try:
        import torch_xla.core.xla_model as xm
        return len(xm.get_xla_supported_devices()) > 0
    except Exception:
        return False


def _get_device():
    if _is_tpu():
        import torch_xla.core.xla_model as xm
        return str(xm.xla_device())
    return "cuda" if torch.cuda.is_available() else "cpu"


def _get_dtype():
    if _is_tpu() or torch.cuda.is_available():
        return torch.bfloat16
    return torch.float32


def _empty_cache():
    if _is_tpu():
        import torch_xla.core.xla_model as xm
        xm.wait_device_ops()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def _count_params(model):
    return sum(p.numel() for p in model.parameters())


def load_model(device=None, dtype=None):
    """Load the baseline SmolLM2-360M model + tokenizer."""
    device = device or _get_device()
    dtype = dtype or _get_dtype()
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype)
    model.to(device)
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

    # 5. Eval after compression
    print("\n=== Evaluating compressed (pre-training) ===")
    post_compress_results = run_eval(student, tok, limit=eval_limit)

    # 6. Free teacher before SFT (reloaded for KD later)
    print("\n=== Freeing teacher VRAM for SFT ===")
    del teacher
    gc.collect()
    _empty_cache()

    # 7. SFT
    print("\n=== Stage 1: SFT ===")
    sft_metrics = run_sft(student, tok, num_epochs=sft_epochs)
    _test_generation(student, tok, device)

    # 8. Eval after SFT
    print("\n=== Evaluating after SFT ===")
    post_sft_results = run_eval(student, tok, limit=eval_limit)

    # 9. Reload teacher for KD
    print("\n=== Reloading teacher for KD ===")
    teacher, _ = load_model()

    # 10. KD
    print("\n=== Stage 2: KD ===")
    kd_metrics = run_kd(student, teacher, tok, num_epochs=kd_epochs)
    _test_generation(student, tok, device)

    # 11. Free teacher before final eval
    del teacher
    gc.collect()
    _empty_cache()

    # 12. Final evaluation
    print("\n=== Evaluating after KD ===")
    final_results = run_eval(student, tok, limit=eval_limit)

    # 13. Comparison
    print("\n=== Summary ===")
    print(f"  Params: {baseline_params:,} → {compressed_params:,}")
    _print_stages(baseline_results, post_compress_results,
                  post_sft_results, final_results)

    # 14. Save final model
    student.save_pretrained("./checkpoints/final_compressed")
    tok.save_pretrained("./checkpoints/final_compressed")
    print("\nFinal model saved to ./checkpoints/final_compressed")

    return {
        "baseline_params": baseline_params,
        "compressed_params": compressed_params,
        "sft_metrics": sft_metrics,
        "kd_metrics": kd_metrics,
        "baseline_eval": baseline_results,
        "post_compress_eval": post_compress_results,
        "post_sft_eval": post_sft_results,
        "final_eval": final_results,
    }


def _test_generation(model, tok, device, prompt="Explain quantum computing in one sentence."):
    model.eval()
    inputs = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=50, do_sample=False,
            repetition_penalty=1.2,
        )
    if "xla" in str(device):
        import torch_xla.core.xla_model as xm
        xm.mark_step()
    print(f"  Generation: {tok.decode(out[0], skip_special_tokens=True)[:200]}...")


def _print_stages(baseline, post_compress, post_sft, post_kd):
    """Print eval results across all 4 stages side by side."""
    stages = [
        ("Baseline", baseline or {}),
        ("Compressed", post_compress or {}),
        ("After SFT", post_sft or {}),
        ("After KD", post_kd or {}),
    ]
    all_tasks = sorted(set().union(*[s.keys() for _, s in stages]))

    print(f"\n  {'Task':25s}  {'Baseline':>10s}  {'Compressed':>10s}  {'After SFT':>10s}  {'After KD':>10s}")
    print(f"  {'-'*70}")
    for task in all_tasks:
        vals = [s.get(task, -1) for _, s in stages]
        row = f"  {task:25s}"
        for v in vals:
            row += f"  {'FAILED':>10s}" if v < 0 else f"  {v:10.4f}"
        print(row)
