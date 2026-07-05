"""Tests for shared prompt construction."""

from __future__ import annotations

from fastapi_assistant.prompts import SYSTEM_PROMPT, build_messages, encode_chat


def test_build_messages_basic():
    messages = build_messages("How do I define a GET endpoint?")
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == SYSTEM_PROMPT
    assert messages[1]["content"] == "How do I define a GET endpoint?"


def test_build_messages_with_context():
    messages = build_messages("Fix this", context="@app.get()")
    assert "Context:" in messages[1]["content"]
    assert "@app.get()" in messages[1]["content"]


def test_build_messages_strips_whitespace():
    messages = build_messages("  spaced  ")
    assert messages[1]["content"] == "spaced"


class _ListTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=True):
        return [1, 2, 3]


class _DictTokenizer:
    """Mimics transformers >= 5, which returns a BatchEncoding (dict)."""

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=True):
        return {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}


def test_encode_chat_normalizes_list_return():
    assert encode_chat(_ListTokenizer(), [], add_generation_prompt=True) == [1, 2, 3]


def test_encode_chat_normalizes_dict_return():
    assert encode_chat(_DictTokenizer(), [], add_generation_prompt=True) == [1, 2, 3]
