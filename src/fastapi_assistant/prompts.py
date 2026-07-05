"""Shared prompt construction.

Training and inference MUST build prompts identically, otherwise the model sees a
different format at serving time than it was trained on. Keeping this logic in one
module guarantees they stay aligned. We rely on the tokenizer's native chat
template (``apply_chat_template``) so the base model's instruction-following
alignment is preserved instead of inventing a bespoke prompt format.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid importing transformers at module load
    from transformers import PreTrainedTokenizerBase

SYSTEM_PROMPT = (
    "You are a senior Python engineer specialized in the FastAPI web framework. "
    "Answer questions concisely and correctly, and include minimal runnable code "
    "examples where they help. If a question is outside FastAPI, say so briefly."
)


def build_messages(question: str, context: str | None = None) -> list[dict[str, str]]:
    """Build the chat-format message list shared by training and inference.

    Args:
        question: The user's FastAPI question (maps to the dataset ``instruction``).
        context: Optional extra context (maps to the dataset ``input``).

    Returns:
        A list of ``{"role": ..., "content": ...}`` messages ready for
        ``tokenizer.apply_chat_template``.
    """
    user_content = question.strip()
    if context:
        user_content = f"{user_content}\n\nContext:\n{context.strip()}"

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def encode_chat(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> list[int]:
    """Tokenize a chat message list into a flat list of token ids.

    Normalizes across transformers versions: newer releases return a
    ``BatchEncoding`` from ``apply_chat_template(tokenize=True)`` (a ``UserDict``,
    which is a ``Mapping`` but *not* a ``dict`` subclass), while older ones return a
    plain ``list[int]``. Returning a plain list keeps the downstream masking logic
    simple and version-independent.
    """
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        tokenize=True,
    )
    ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
    # Collapse a possible batch dimension ([[...]] -> [...]) for a single conversation.
    if ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return list(ids)
