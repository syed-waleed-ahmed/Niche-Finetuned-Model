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
import threading
from collections.abc import Iterator, Sequence

from ..config import Settings
from ..hub import resolve_gguf_path
from .base import BaseEngine, Deadline, GenerationParams, Message, StreamEvent

log = logging.getLogger(__name__)


class LlamaCppEngine(BaseEngine):
    backend = "llamacpp"

    def __init__(self, settings: Settings) -> None:
        model_id = settings.gguf_repo_id or str(settings.gguf_path or "gguf")
        super().__init__(model_id)
        self._settings = settings
        self._llm = None
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

    def stream(
        self,
        messages: Sequence[Message],
        params: GenerationParams,
        cancel: threading.Event | None = None,
    ) -> Iterator[StreamEvent]:
        self.load()
        deadline = Deadline(params.timeout_s, cancel)
        finish = None
        completion_tokens = 0
        prompt_tokens = 0
        with self._lock:
            chunks = self._llm.create_chat_completion(
                messages=list(messages),
                max_tokens=params.max_new_tokens,
                temperature=params.temperature if params.do_sample else 0.0,
                top_p=params.top_p,
                repeat_penalty=params.repetition_penalty,
                stream=True,
            )
            try:
                for chunk in chunks:
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
            prompt_tokens = max(0, int(getattr(self._llm, "n_tokens", 0)) - completion_tokens)
        yield StreamEvent(
            finish_reason=finish or "stop",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def close(self) -> None:
        llm, self._llm = self._llm, None
        if llm is not None and hasattr(llm, "close"):
            llm.close()
