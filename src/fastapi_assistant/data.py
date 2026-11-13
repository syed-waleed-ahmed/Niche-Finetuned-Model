"""Dataset loading and tokenization for supervised fine-tuning.

The JSONL dataset uses the classic instruction-tuning schema::

    {"instruction": "...", "input": "", "output": "..."}

Tokenization applies **completion-only masking**: prompt tokens are labeled
``-100`` so the loss is computed only over the assistant's answer, which is the
correct objective for instruction tuning. Sequences are left unpadded here and
padded per-batch by the data collator (see :mod:`fastapi_assistant.training`).
"""

from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizerBase

from .config import Settings
from .prompts import build_messages, encode_chat

# Loss ignore index used by PyTorch's cross-entropy and HF collators.
IGNORE_INDEX = -100

# Training only needs these fields; eval rows also carry id/category/must_include.
SFT_FIELDS = ("instruction", "input", "output")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_raw_dataset(settings: Settings) -> DatasetDict:
    """Load the train/eval JSONL splits as a :class:`~datasets.DatasetDict`.

    Rows are projected to the SFT fields so the eval split's benchmark metadata
    does not leak into tokenization or batching.
    """

    def _split(path: Path) -> Dataset:
        rows = [{k: str(row.get(k) or "") for k in SFT_FIELDS} for row in read_jsonl(path)]
        return Dataset.from_list(rows)

    return DatasetDict(train=_split(settings.train_path), eval=_split(settings.eval_path))


def build_tokenize_fn(tokenizer: PreTrainedTokenizerBase, settings: Settings):
    """Return a per-example tokenization function with completion-only masking."""
    max_len = settings.max_seq_length

    def _tokenize(example: dict) -> dict:
        question = example["instruction"]
        context = example.get("input") or None
        answer = example["output"]

        messages = build_messages(question, context)

        # Prompt (system + user + assistant generation header). This is a strict
        # prefix of the full sequence, so its length tells us how many leading
        # tokens to mask.
        prompt_ids = encode_chat(tokenizer, messages, add_generation_prompt=True)
        full_ids = encode_chat(
            tokenizer,
            [*messages, {"role": "assistant", "content": answer}],
            add_generation_prompt=False,
        )

        full_ids = full_ids[:max_len]
        labels = list(full_ids)
        prompt_len = min(len(prompt_ids), len(full_ids))
        for i in range(prompt_len):
            labels[i] = IGNORE_INDEX

        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
        }

    return _tokenize


def tokenize_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset: DatasetDict,
    settings: Settings,
) -> DatasetDict:
    """Tokenize every split, dropping the original text columns."""
    tokenize_fn = build_tokenize_fn(tokenizer, settings)
    return dataset.map(
        tokenize_fn,
        batched=False,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing",
    )
