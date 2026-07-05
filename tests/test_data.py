"""Tests for dataset tokenization and completion-only masking.

Uses a minimal fake tokenizer so the logic can be verified without loading a real
model. The fake mirrors how real chat templates render an assistant turn: the
generation prompt tokens are a prefix of the full completed sequence.
"""

from __future__ import annotations

from fastapi_assistant.config import Settings
from fastapi_assistant.data import IGNORE_INDEX, build_tokenize_fn


class FakeTokenizer:
    """Deterministic chat-template tokenizer stand-in.

    Each character contributes one token id. An assistant turn (or a generation
    prompt) is preceded by a sentinel header token (0), so the ``add_generation_prompt``
    output is a strict prefix of the full sequence.
    """

    HEADER = 0

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=True):
        ids: list[int] = []
        for message in messages:
            if message["role"] == "assistant":
                ids.append(self.HEADER)
            ids.extend(ord(ch) for ch in message["content"])
        if add_generation_prompt:
            ids.append(self.HEADER)
        return ids


def test_completion_only_masking():
    settings = Settings(_env_file=None, max_seq_length=512)
    tokenize = build_tokenize_fn(FakeTokenizer(), settings)

    result = tokenize({"instruction": "hi", "input": "", "output": "yo"})

    input_ids = result["input_ids"]
    labels = result["labels"]

    assert len(input_ids) == len(labels) == len(result["attention_mask"])

    # Only the answer content ("yo") is supervised. The assistant header token is
    # part of the generation prompt (add_generation_prompt=True), so it is masked.
    answer_tokens = [ord("y"), ord("o")]
    prompt_len = len(input_ids) - len(answer_tokens)

    assert labels[:prompt_len] == [IGNORE_INDEX] * prompt_len
    assert labels[prompt_len:] == answer_tokens
    # The generation-prompt (answer) region is a strict suffix of the full sequence.
    assert input_ids[prompt_len:] == answer_tokens
    # The token immediately before the answer is the masked assistant header.
    assert input_ids[prompt_len - 1] == FakeTokenizer.HEADER


def test_truncation_respects_max_seq_length():
    settings = Settings(_env_file=None, max_seq_length=16)
    tokenize = build_tokenize_fn(FakeTokenizer(), settings)

    # The system prompt alone exceeds 16 chars/tokens, so the result is truncated.
    result = tokenize({"instruction": "abcdefghij", "input": "", "output": "klmnop"})

    assert len(result["input_ids"]) == 16
    assert len(result["labels"]) == 16
    assert len(result["attention_mask"]) == 16
