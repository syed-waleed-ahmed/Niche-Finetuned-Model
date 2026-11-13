"""Engine contract shared by every inference backend.

An engine owns model weights for the lifetime of the process and turns a list of
chat messages into text. The HTTP layer depends only on this contract, so the
backend (torch + LoRA, llama.cpp GGUF, or a remote OpenAI-compatible server) is a
configuration choice, not a code change.

Engines are synchronous and are always driven from worker threads. Concurrency is
bounded one level up by :class:`~fastapi_assistant.admission.AdmissionController`;
engines that cannot run requests in parallel also hold an internal lock.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

log = logging.getLogger(__name__)

Message = dict[str, str]
FinishReason = Literal["stop", "length", "timeout", "cancelled"]
LoadState = Literal["idle", "loading", "ready", "failed"]


# A prompt must leave at least this many tokens for the answer, or it is rejected.
MIN_COMPLETION_TOKENS = 16


class ModelNotReadyError(RuntimeError):
    """Raised when the model could not be loaded."""


class PromptTooLongError(ValueError):
    """The prompt does not fit in the model's context window (HTTP 400)."""

    def __init__(self, prompt_tokens: int, context_window: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.context_window = context_window
        super().__init__(
            f"The prompt is {prompt_tokens} tokens; the model's context window is "
            f"{context_window} tokens, including at least {MIN_COMPLETION_TOKENS} for the answer"
        )


@dataclass(frozen=True)
class GenerationParams:
    """Sampling and budget parameters for one generation."""

    max_new_tokens: int = 256
    temperature: float = 0.4
    top_p: float = 0.9
    repetition_penalty: float = 1.05
    timeout_s: float | None = None

    @property
    def do_sample(self) -> bool:
        return self.temperature > 0.0


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: FinishReason = "stop"


@dataclass
class StreamEvent:
    """A streamed text delta. The final event carries the finish reason and usage."""

    delta: str = ""
    finish_reason: FinishReason | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def is_final(self) -> bool:
        return self.finish_reason is not None


@runtime_checkable
class Engine(Protocol):
    backend: str
    model_id: str

    @property
    def is_loaded(self) -> bool: ...

    @property
    def uses_adapter(self) -> bool: ...

    @property
    def load_state(self) -> LoadState: ...

    @property
    def load_error(self) -> str | None: ...

    def load(self) -> None: ...

    def start_background_load(self) -> None: ...

    def fit(self, messages: Sequence[Message], params: GenerationParams) -> GenerationParams: ...

    def generate(
        self, messages: Sequence[Message], params: GenerationParams
    ) -> GenerationResult: ...

    def stream(
        self,
        messages: Sequence[Message],
        params: GenerationParams,
        cancel: threading.Event | None = None,
    ) -> Iterator[StreamEvent]: ...

    def close(self) -> None: ...


class Deadline:
    """Wall-clock budget for one generation, plus cooperative cancellation."""

    def __init__(self, timeout_s: float | None, cancel: threading.Event | None = None) -> None:
        self._deadline = time.monotonic() + timeout_s if timeout_s else None
        self._cancel = cancel
        self.reason: FinishReason | None = None

    def expired(self) -> bool:
        if self._cancel is not None and self._cancel.is_set():
            self.reason = "cancelled"
            return True
        if self._deadline is not None and time.monotonic() > self._deadline:
            self.reason = "timeout"
            return True
        return False


class BaseEngine:
    """Load-once lifecycle shared by all engines.

    Subclasses implement :meth:`_load_impl` and :meth:`stream`; :meth:`generate`
    defaults to collecting the stream.
    """

    backend = "base"

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self._uses_adapter = False
        self._state: LoadState = "idle"
        self._error: str | None = None
        self._load_lock = threading.Lock()

    # -- lifecycle -------------------------------------------------------
    @property
    def is_loaded(self) -> bool:
        return self._state == "ready"

    @property
    def uses_adapter(self) -> bool:
        return self._uses_adapter

    @property
    def load_state(self) -> LoadState:
        return self._state

    @property
    def load_error(self) -> str | None:
        return self._error

    def load(self) -> None:
        """Load weights exactly once (thread-safe, idempotent).

        A failed load is retried on the next call, so a transient download error
        does not wedge the process.
        """
        if self._state == "ready":
            return
        with self._load_lock:
            if self._state == "ready":  # re-check under lock
                return
            self._state = "loading"
            started = time.perf_counter()
            try:
                self._load_impl()
            except Exception as exc:
                self._state = "failed"
                self._error = str(exc) or exc.__class__.__name__
                log.exception("Failed to load model (%s backend)", self.backend)
                raise ModelNotReadyError(self._error) from exc
            self._state = "ready"
            self._error = None
            log.info(
                "Model ready (backend=%s, model=%s, adapter=%s) in %.1fs",
                self.backend,
                self.model_id,
                self._uses_adapter,
                time.perf_counter() - started,
            )

    def start_background_load(self) -> None:
        """Begin loading in a daemon thread unless already loading or loaded."""
        if self._state in ("ready", "loading"):
            return

        def _run() -> None:
            try:
                self.load()
            except ModelNotReadyError:
                pass  # recorded in load_state / load_error

        threading.Thread(target=_run, name="model-load", daemon=True).start()

    def _load_impl(self) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def close(self) -> None:
        """Release resources. Default: nothing to do."""

    # -- context budget ----------------------------------------------------
    def context_window(self) -> int | None:
        """Total tokens (prompt + completion) the model accepts, or None if unknown."""
        return None

    def count_prompt_tokens(self, messages: Sequence[Message]) -> int | None:
        """Prompt length in tokens, never under-counted; None if it cannot be measured."""
        return None

    def fit(self, messages: Sequence[Message], params: GenerationParams) -> GenerationParams:
        """Check that ``messages`` fit the context window; clamp the answer budget.

        Called before a request is admitted, so an oversized prompt is a 400 rather
        than a failure in the middle of generation (or of an SSE stream). Engines that
        cannot measure their context (e.g. a remote server) accept everything here.
        """
        window = self.context_window()
        if window is None:
            return params
        prompt_tokens = self.count_prompt_tokens(messages)
        if prompt_tokens is None:
            return params
        room = window - prompt_tokens
        if room < MIN_COMPLETION_TOKENS:
            raise PromptTooLongError(prompt_tokens, window)
        if params.max_new_tokens > room:
            return dataclasses.replace(params, max_new_tokens=room)
        return params

    # -- inference ---------------------------------------------------------
    def stream(
        self,
        messages: Sequence[Message],
        params: GenerationParams,
        cancel: threading.Event | None = None,
    ) -> Iterator[StreamEvent]:  # pragma: no cover - abstract
        raise NotImplementedError

    def generate(self, messages: Sequence[Message], params: GenerationParams) -> GenerationResult:
        parts: list[str] = []
        final = StreamEvent(finish_reason="stop")
        for event in self.stream(messages, params):
            if event.delta:
                parts.append(event.delta)
            if event.is_final:
                final = event
        return GenerationResult(
            text="".join(parts).strip(),
            prompt_tokens=final.prompt_tokens,
            completion_tokens=final.completion_tokens,
            finish_reason=final.finish_reason or "stop",
        )
