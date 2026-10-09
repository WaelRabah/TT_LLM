"""Data loading and preprocessing for SFT and KD training.

Uses the Databricks Dolly-15K instruction-following dataset.
Formats examples with an Alpaca-style prompt template, tokenizes with
the model's tokenizer, and masks instruction tokens (loss computed on
response tokens only).
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "{context_section}"
    "### Response:\n"
)


def format_dolly_example(example: dict) -> tuple[str, str]:
    """Format a Dolly-15K example into (prompt, response) text.

    Returns the instruction+context as the prompt and the response as the target.
    """
    instruction = example["instruction"].strip()
    context = example.get("context", "").strip()
    response = example["response"].strip()

    context_section = f"### Context:\n{context}\n\n" if context else ""
    prompt = PROMPT_TEMPLATE.format(
        instruction=instruction,
        context_section=context_section,
    )
    return prompt, response


def load_dolly_dataset(
    split: str = "train",
    max_examples: int | None = None,
):
    """Load the Databricks Dolly-15K dataset from HuggingFace.

    Parameters
    ----------
    split : ``"train"`` or ``"val"`` — the dataset is split 95/5.
    max_examples : cap the number of examples (useful for quick Colab runs).
    """
    from datasets import load_dataset

    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    ds = ds.train_test_split(test_size=0.05, seed=42)
    ds = ds[split]
    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))
    return ds


class DollySFTDataset(Dataset):
    """Tokenized Dolly-15K dataset for supervised fine-tuning.

    Instruction tokens are masked with ``-100`` so that loss is computed
    only on the response tokens.
    """

    def __init__(
        self,
        tokenizer,
        split: str = "train",
        max_length: int = 512,
        max_examples: int | None = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        ds = load_dolly_dataset(split=split, max_examples=max_examples)
        eos_token = tokenizer.eos_token or "<|endoftext|>"
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            tokenizer.pad_token = eos_token
            pad_id = tokenizer.pad_token_id

        for row in ds:
            prompt, response = format_dolly_example(row)
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            response_ids = tokenizer(response + eos_token, add_special_tokens=False)["input_ids"]

            input_ids = prompt_ids + response_ids
            labels = [-100] * len(prompt_ids) + response_ids[:]

            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                labels = labels[:max_length]

            self.examples.append({
                "input_ids": input_ids,
                "labels": labels,
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch: list[dict], pad_id: int) -> dict:
    """Collate variable-length sequences into a padded batch.

    Returns ``input_ids``, ``attention_mask``, and ``labels`` tensors.
    Padding is right-sided; label padding is ``-100``.
    """
    max_len = max(len(ex["input_ids"]) for ex in batch)
    input_ids = []
    attention_mask = []
    labels = []
    for ex in batch:
        pad_len = max_len - len(ex["input_ids"])
        input_ids.append(ex["input_ids"] + [pad_id] * pad_len)
        attention_mask.append([1] * len(ex["input_ids"]) + [0] * pad_len)
        labels.append(ex["labels"] + [-100] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def make_collate_fn(pad_id: int):
    """Return a collate_fn closure with the given pad token id."""
    def _collate(batch):
        return collate_fn(batch, pad_id)
    return _collate
