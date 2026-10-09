"""Training loops for TT-compressed models.

Two stages:
1. **SFT** (Supervised Fine-Tuning) — fine-tune the compressed model on
   Dolly-15K with cross-entropy loss on response tokens.
2. **KD** (Knowledge Distillation) — fine-tune the compressed student with
   the uncompressed teacher's soft logits via KL divergence.

Both stages use bf16 autocast, gradient accumulation, cosine LR schedule,
and Weights & Biases-compatible logging.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


@dataclass
class TrainConfig:
    """Hyperparameters for SFT and KD training."""
    output_dir: str = "./checkpoints"
    num_epochs: int = 1
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    log_every: int = 10
    save_every: int = 0  # 0 = save only at end
    bf16: bool = True
    seed: int = 42

    # KD-specific
    kd_temperature: float = 2.0
    kd_alpha: float = 0.5  # weight on KD loss; (1-alpha) on SFT loss

    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)


def _get_cosine_schedule(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_total_steps: int,
):
    """Cosine LR schedule with linear warmup."""
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_total_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def train_sft(
    model: torch.nn.Module,
    tokenizer,
    train_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset | None = None,
    config: TrainConfig | None = None,
    device: str = "cuda",
) -> dict:
    """Stage 1: Supervised Fine-Tuning.

    Trains the model with cross-entropy loss on response tokens (instruction
    tokens are masked with -100). Uses bf16 autocast and gradient accumulation.

    Returns a dict with training metrics.
    """
    from .data import make_collate_fn

    if config is None:
        config = TrainConfig()

    torch.manual_seed(config.seed)
    model.to(device)
    model.train()

    pad_id = tokenizer.pad_token_id
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=make_collate_fn(pad_id),
        num_workers=2,
        pin_memory=True,
    )

    num_steps_per_epoch = math.ceil(
        len(train_loader) / config.gradient_accumulation_steps
    )
    num_total_steps = num_steps_per_epoch * config.num_epochs
    num_warmup = max(1, int(num_total_steps * config.warmup_ratio))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = _get_cosine_schedule(optimizer, num_warmup, num_total_steps)
    scaler = torch.amp.GradScaler("cuda") if config.bf16 else None

    metrics = {"train_loss": [], "val_loss": [], "lr": []}
    global_step = 0
    running_loss = 0.0

    print(f"SFT Training: {len(train_dataset)} examples, "
          f"{num_total_steps} steps, "
          f"eff_batch={config.effective_batch_size()}")

    for epoch in range(config.num_epochs):
        model.train()
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            if config.bf16:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    outputs = model(**batch)
                    loss = outputs.loss / config.gradient_accumulation_steps
            else:
                outputs = model(**batch)
                loss = outputs.loss / config.gradient_accumulation_steps

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running_loss += loss.item()
            metrics["train_loss"].append(loss.item() * config.gradient_accumulation_steps)
            metrics["lr"].append(scheduler.get_last_lr()[0])

            if (step + 1) % config.gradient_accumulation_steps == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % config.log_every == 0:
                    avg_loss = running_loss / config.log_every / config.gradient_accumulation_steps
                    print(f"  Epoch {epoch+1}/{config.num_epochs} "
                          f"Step {global_step}/{num_total_steps} "
                          f"loss={avg_loss:.4f} "
                          f"lr={scheduler.get_last_lr()[0]:.2e}")
                    running_loss = 0.0

                if config.save_every > 0 and global_step % config.save_every == 0:
                    _save_checkpoint(model, tokenizer, config.output_dir, f"step_{global_step}")

        if val_dataset is not None:
            val_loss = evaluate_loss(model, val_dataset, tokenizer, device, config)
            metrics["val_loss"].append(val_loss)
            print(f"  Epoch {epoch+1} validation loss: {val_loss:.4f}")

    _save_checkpoint(model, tokenizer, config.output_dir, "final")
    print(f"SFT done. Checkpoint saved to {config.output_dir}")
    return metrics


def train_kd(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    tokenizer,
    train_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset | None = None,
    config: TrainConfig | None = None,
    device: str = "cuda",
) -> dict:
    """Stage 2: Knowledge Distillation.

    Trains the student (compressed) model using a combination of:
    - KL divergence between teacher and student logits (soft targets)
    - Cross-entropy on response tokens (hard targets)

    ``L = alpha * T^2 * KL(student || teacher) + (1 - alpha) * CE``

    The teacher is kept in eval mode with no gradients.
    """
    from .data import make_collate_fn

    if config is None:
        config = TrainConfig()

    torch.manual_seed(config.seed)
    student.to(device)
    teacher.to(device)
    student.train()
    teacher.eval()

    for p in teacher.parameters():
        p.requires_grad = False

    pad_id = tokenizer.pad_token_id
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=make_collate_fn(pad_id),
        num_workers=2,
        pin_memory=True,
    )

    num_steps_per_epoch = math.ceil(
        len(train_loader) / config.gradient_accumulation_steps
    )
    num_total_steps = num_steps_per_epoch * config.num_epochs
    num_warmup = max(1, int(num_total_steps * config.warmup_ratio))

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = _get_cosine_schedule(optimizer, num_warmup, num_total_steps)
    scaler = torch.amp.GradScaler("cuda") if config.bf16 else None

    T = config.kd_temperature
    alpha = config.kd_alpha

    metrics = {"train_loss": [], "train_kd_loss": [], "train_ce_loss": [],
               "val_loss": [], "lr": []}
    global_step = 0
    running_loss = 0.0

    print(f"KD Training: {len(train_dataset)} examples, "
          f"{num_total_steps} steps, "
          f"T={T}, alpha={alpha}")

    for epoch in range(config.num_epochs):
        student.train()
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")

            if config.bf16:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    student_outputs = student(**batch)
                    with torch.no_grad():
                        teacher_logits = teacher(**batch).logits

                    student_logits = student_outputs.logits

                    ce_loss = F.cross_entropy(
                        student_logits.view(-1, student_logits.size(-1)),
                        labels.view(-1),
                        ignore_index=-100,
                    )

                    kd_loss = _kl_div_loss(
                        student_logits, teacher_logits, labels, T,
                    )

                    loss = (alpha * T * T * kd_loss + (1 - alpha) * ce_loss)
                    loss = loss / config.gradient_accumulation_steps
            else:
                student_outputs = student(**batch)
                with torch.no_grad():
                    teacher_logits = teacher(**batch).logits

                student_logits = student_outputs.logits
                ce_loss = F.cross_entropy(
                    student_logits.view(-1, student_logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                )
                kd_loss = _kl_div_loss(
                    student_logits, teacher_logits, labels, T,
                )
                loss = (alpha * T * T * kd_loss + (1 - alpha) * ce_loss)
                loss = loss / config.gradient_accumulation_steps

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running_loss += loss.item()
            metrics["train_loss"].append(loss.item() * config.gradient_accumulation_steps)
            metrics["train_kd_loss"].append(kd_loss.item())
            metrics["train_ce_loss"].append(ce_loss.item())
            metrics["lr"].append(scheduler.get_last_lr()[0])

            if (step + 1) % config.gradient_accumulation_steps == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), config.max_grad_norm)
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % config.log_every == 0:
                    avg_loss = running_loss / config.log_every / config.gradient_accumulation_steps
                    print(f"  Epoch {epoch+1}/{config.num_epochs} "
                          f"Step {global_step}/{num_total_steps} "
                          f"loss={avg_loss:.4f} "
                          f"(kd={kd_loss.item():.4f} ce={ce_loss.item():.4f}) "
                          f"lr={scheduler.get_last_lr()[0]:.2e}")
                    running_loss = 0.0

                if config.save_every > 0 and global_step % config.save_every == 0:
                    _save_checkpoint(student, tokenizer, config.output_dir, f"kd_step_{global_step}")

        if val_dataset is not None:
            val_loss = evaluate_loss(student, val_dataset, tokenizer, device, config)
            metrics["val_loss"].append(val_loss)
            print(f"  Epoch {epoch+1} validation loss: {val_loss:.4f}")

    _save_checkpoint(student, tokenizer, config.output_dir, "kd_final")
    print(f"KD done. Checkpoint saved to {config.output_dir}")
    return metrics


def _kl_div_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """KL(teacher || student) on non-masked positions.

    Both logits are scaled by ``1/temperature``. Masked positions
    (label == -100) are excluded.
    """
    mask = (labels != -100)
    if not mask.any():
        return torch.tensor(0.0, device=student_logits.device)

    s_log = F.log_softmax(student_logits[mask] / temperature, dim=-1)
    t_soft = F.softmax(teacher_logits[mask] / temperature, dim=-1)
    return F.kl_div(s_log, t_soft, reduction="batchmean")


def evaluate_loss(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    tokenizer,
    device: str,
    config: TrainConfig,
) -> float:
    """Compute average cross-entropy loss on a dataset (no gradient)."""
    from .data import make_collate_fn

    model.eval()
    pad_id = tokenizer.pad_token_id
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=make_collate_fn(pad_id),
    )
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            if config.bf16:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(**batch).loss
            else:
                loss = model(**batch).loss
            total_loss += loss.item()
            num_batches += 1
    model.train()
    return total_loss / max(num_batches, 1)


def _save_checkpoint(model, tokenizer, output_dir: str, tag: str):
    """Save model and tokenizer to ``output_dir/tag``."""
    path = os.path.join(output_dir, tag)
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"  Saved checkpoint: {path}")
