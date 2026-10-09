"""Evaluation harness wrapper using lm-evaluation-harness.

Runs 5 standard benchmarks on a given model:
  - MMLU (5 subjects subset for speed)
  - HellaSwag
  - ARC-Easy
  - PIQA
  - WinoGrande

Usage::

    from tt_llm.eval import run_benchmarks

    results = run_benchmarks(model, tokenizer, device="cuda")
    for task, score in results.items():
        print(f"  {task}: {score:.4f}")
"""

from __future__ import annotations

import torch

BENCHMARK_TASKS = [
    "hellaswag",
    "arc_easy",
    "piqa",
    "winogrande",
    "mmlu_abstract_algebra",
    "mmlu_anatomy",
    "mmlu_astronomy",
    "mmlu_college_biology",
    "mmlu_college_chemistry",
]

FEW_SHOT = {
    "hellaswag": 0,
    "arc_easy": 0,
    "piqa": 0,
    "winogrande": 0,
    "mmlu_abstract_algebra": 5,
    "mmlu_anatomy": 5,
    "mmlu_astronomy": 5,
    "mmlu_college_biology": 5,
    "mmlu_college_chemistry": 5,
}

TASK_DISPLAY = {
    "hellaswag": "HellaSwag",
    "arc_easy": "ARC-Easy",
    "piqa": "PIQA",
    "winogrande": "WinoGrande",
    "mmlu_abstract_algebra": "MMLU (subset avg)",
    "mmlu_anatomy": "MMLU (subset avg)",
    "mmlu_astronomy": "MMLU (subset avg)",
    "mmlu_college_biology": "MMLU (subset avg)",
    "mmlu_college_chemistry": "MMLU (subset avg)",
}


def run_benchmarks(
    model: torch.nn.Module,
    tokenizer,
    device: str = "cuda",
    batch_size: int = 8,
    limit: int | None = None,
    tasks: list[str] | None = None,
) -> dict[str, float]:
    """Run benchmark evaluation on a model.

    Parameters
    ----------
    model : a HuggingFace ``AutoModelForCausalLM`` (possibly TT-compressed).
    tokenizer : the matching tokenizer.
    device : ``"cuda"`` or ``"cpu"``.
    batch_size : eval batch size.
    limit : cap examples per task (useful for quick runs; ``None`` = full).
    tasks : override the default task list.

    Returns
    -------
    dict mapping task name to accuracy (0–1).
    """
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        raise ImportError(
            "lm-eval not installed. Install with: pip install lm-eval"
        )

    model.to(device)
    model.eval()

    task_list = tasks or BENCHMARK_TASKS

    hf_lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        device=device,
    )

    results_dict = {}

    for task_name in task_list:
        num_fewshot = FEW_SHOT.get(task_name, 0)
        print(f"Running {TASK_DISPLAY.get(task_name, task_name)} "
              f"(fewshot={num_fewshot}", end="")
        if limit:
            print(f", limit={limit}", end="")
        print(")...", flush=True)

        try:
            results = lm_eval.simple_evaluate(
                model=hf_lm,
                tasks=[task_name],
                num_fewshot=num_fewshot,
                limit=limit,
            )
            metrics = results["results"][task_name]
            acc = metrics.get("acc,none", metrics.get("acc_norm,none", 0.0))
            results_dict[task_name] = acc
            print(f"  {TASK_DISPLAY.get(task_name, task_name)}: {acc:.4f}")
        except Exception as e:
            print(f"  {task_name}: FAILED ({e})")
            results_dict[task_name] = -1.0

    mmlu_scores = [
        results_dict[t] for t in task_list
        if t.startswith("mmlu_") and results_dict.get(t, -1) >= 0
    ]
    if mmlu_scores:
        results_dict["mmlu_subset_avg"] = sum(mmlu_scores) / len(mmlu_scores)

    return results_dict


def print_results(results: dict[str, float], title: str = ""):
    """Pretty-print benchmark results."""
    if title:
        print(f"\n{'='*50}")
        print(f"  {title}")
        print(f"{'='*50}")

    mmlu_avg = results.pop("mmlu_subset_avg", None)

    for task, score in results.items():
        display = TASK_DISPLAY.get(task, task)
        status = f"{score:.4f}" if score >= 0 else "FAILED"
        print(f"  {display:25s} {status}")

    if mmlu_avg is not None:
        print(f"  {'MMLU (subset avg)':25s} {mmlu_avg:.4f}")

    print(f"{'='*50}")
