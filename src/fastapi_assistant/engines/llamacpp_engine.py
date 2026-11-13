"""llama.cpp backend: a merged, quantized GGUF model on CPU.

This is the production default. A 1B-parameter model quantized to Q4_K_M is about
0.7 GB and generates several times faster than fp32 torch on the same CPU, which
is what makes free-tier CPU instances viable. The chat template embedded in the
GGUF (written by llama.cpp's converter from the tokenizer) is applied by
``create_chat_completion``, so the prompt format matches training.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any, cast

from ..config import Settings
from ..hub import resolve_gguf_path
from ..prompts import STOP_SEQUENCES
from .base import (
    BaseEngine,
    Deadline,
    FinishReason,
    GenerationParams,
    Message,
    PromptTooLongError,
    StreamEvent,
)

if TYPE_CHECKING:
    from llama_cpp import Llama

log = logging.getLogger(__name__)

# Upper bound on the tokens a chat template adds around one message (role markers,
# separators, end-of-turn). Over-counting is deliberate: the estimate must never be
# lower than the real prompt, or an over-long prompt would slip past ``fit``.
_TEMPLATE_TOKENS_PER_MESSAGE = 8
_TEMPLATE_TOKENS_FIXED = 8  # BOS plus the assistant generation prompt


class LlamaCppEngine(BaseEngine):
    backend = "llamacpp"

    def __init__(self, settings: Settings) -> None:
        model_id = settings.gguf_repo_id or str(settings.gguf_path or "gguf")
        super().__init__(model_id)
        self._settings = settings
        self._llm: Llama | None = None
        # A llama.cpp context is single-threaded from the caller's perspective.
        self._lock = threading.Lock()

    def _load_impl(self) -> None:
        from llama_cpp import Llama

        s = self._settings
        path = resolve_gguf_path(s)
        threads = s.llamacpp_n_threads or max(1, (os.cpu_count() or 2))
        log.info("Loading GGUF %s (n_ctx=%d, threads=%d)", path, s.llamacpp_n_ctx, threads)
        self._llm = Llama(
            model_path=str(path),
            n_ctx=s.llamacpp_n_ctx,
            n_threads=threads,
            chat_format=s.llamacpp_chat_format or None,  # "" -> use the GGUF template
            verbose=False,
        )
        self._uses_adapter = s.finetuned_weights

    def _model(self) -> Llama:
        self.load()
        assert self._llm is not None  # set by a successful load
        return self._llm

    def context_window(self) -> int:
        return self._settings.llamacpp_n_ctx

    def count_prompt_tokens(self, messages: Sequence[Message]) -> int:
        # Tokenizing reads only the vocabulary, so it does not take the generation
        # lock and never waits behind a running request.
        llm = self._model()
        content = sum(
            len(llm.tokenize(m["content"].encode("utf-8"), add_bos=False, special=False))
            for m in messages
        )
        return content + _TEMPLATE_TOKENS_PER_MESSAGE * len(messages) + _TEMPLATE_TOKENS_FIXED

    def stream(
        self,
        messages: Sequence[Message],
        params: GenerationParams,
        cancel: threading.Event | None = None,
    ) -> Iterator[StreamEvent]:
        llm = self._model()
        deadline = Deadline(params.timeout_s, cancel)
        finish: FinishReason | None = None
        completion_tokens = 0
        prompt_tokens = 0
        with self._lock:
            chunks: Any = llm.create_chat_completion(
                messages=cast(Any, list(messages)),
                max_tokens=params.max_new_tokens,
                temperature=params.temperature if params.do_sample else 0.0,
                top_p=params.top_p,
                repeat_penalty=params.repetition_penalty,
                # Added to the template's own end-of-sequence stop; llama.cpp holds
                # back text that could be the start of a stop sequence while streaming.
                stop=list(STOP_SEQUENCES),
                stream=True,
            )
            try:
                for chunk in _context_errors(chunks, self._settings.llamacpp_n_ctx):
                    if deadline.expired():
                        finish = deadline.reason
                        break
                    choice = chunk["choices"][0]
                    delta = choice.get("delta", {}).get("content")
                    if delta:
                        completion_tokens += 1  # llama.cpp streams one token per chunk
                        yield StreamEvent(delta=delta)
                    if choice.get("finish_reason"):
                        finish = "length" if choice["finish_reason"] == "length" else "stop"
            finally:
                close = getattr(chunks, "close", None)
                if close is not None:
                    close()
            # n_tokens is the context length after generation: prompt + completion.
            prompt_tokens = max(0, int(getattr(llm, "n_tokens", 0)) - completion_tokens)
        yield StreamEvent(
            finish_reason=finish or "stop",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def close(self) -> None:
        llm, self._llm = self._llm, None
        if llm is not None and hasattr(llm, "close"):
            llm.close()


def _context_errors(chunks: Iterator[Any], n_ctx: int) -> Iterator[Any]:
    """Re-raise llama.cpp's context overflow as :class:`PromptTooLongError`.

    ``fit`` normally rejects these before admission; this is the backstop if a chat
    template ever adds more tokens than the estimate allows for.
    """
    try:
        yield from chunks
    except ValueError as exc:
        # llama-cpp-python: "Requested tokens (N) exceed context window of M"
        match = re.search(r"\((\d+)\) exceed context window", str(exc))
        if match:
            raise PromptTooLongError(int(match.group(1)), n_ctx) from exc
        raise
