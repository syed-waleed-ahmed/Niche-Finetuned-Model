"""Request orchestration: retrieval -> admission -> generation -> accounting.

Routes stay thin and declarative; everything that touches the model, the queue,
metrics, or persistence lives here so the native and OpenAI-compatible APIs share
one code path.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import AsyncIterator, Sequence

from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

from .admission import AdmissionController, QueueFullError, QueueTimeoutError, Slot
from .config import Settings
from .engines.base import Engine, GenerationParams, GenerationResult, Message, StreamEvent
from .metrics import Metrics
from .retrieval import Passage, Retriever, format_context
from .storage import Store, UsageEvent

log = logging.getLogger(__name__)


def _close_quietly(gen) -> None:
    try:
        gen.close()
    except Exception:  # pragma: no cover - best effort cleanup
        log.debug("Error while closing generation stream", exc_info=True)


class AssistantService:
    def __init__(
        self,
        settings: Settings,
        engine: Engine,
        admission: AdmissionController,
        metrics: Metrics,
        store: Store | None = None,
        retriever: Retriever | None = None,
    ) -> None:
        self.settings = settings
        self.engine = engine
        self.admission = admission
        self.metrics = metrics
        self.store = store
        self.retriever = retriever
        metrics.queue_depth.set_function(lambda: admission.waiting)
        metrics.inflight.set_function(lambda: admission.inflight)
        metrics.model_loaded.set_function(lambda: 1.0 if engine.is_loaded else 0.0)
        metrics.build_info.info(
            {
                "backend": engine.backend,
                "model": engine.model_id,
                "model_version": settings.model_version,
            }
        )

    # -- inputs ----------------------------------------------------------------
    def params(
        self,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> GenerationParams:
        s = self.settings
        return GenerationParams(
            max_new_tokens=max_new_tokens or s.model_max_new_tokens,
            temperature=s.model_temperature if temperature is None else temperature,
            top_p=top_p or s.model_top_p,
            repetition_penalty=s.model_repetition_penalty,
            timeout_s=s.generation_timeout_s,
        )

    async def retrieve(self, query: str, enabled: bool | None = None) -> list[Passage]:
        """Fetch supporting documentation. Failures degrade to no context."""
        use = self.settings.rag_enabled if enabled is None else enabled
        if not use or self.retriever is None:
            return []
        started = time.perf_counter()
        try:
            return await run_in_threadpool(self.retriever.retrieve, query, self.settings.rag_top_k)
        except Exception:
            log.exception("Retrieval failed; answering without context")
            return []
        finally:
            self.metrics.retrieval_seconds.observe(time.perf_counter() - started)

    def context(self, passages: list[Passage]) -> str | None:
        if not passages:
            return None
        return format_context(passages, self.settings.rag_max_context_chars)

    # -- context budget ---------------------------------------------------------
    async def fit(self, messages: Sequence[Message], params: GenerationParams) -> GenerationParams:
        """Reject a prompt that cannot fit the context window; clamp the answer budget.

        Runs before admission, so an oversized prompt never takes a queue slot and a
        streaming client gets a 400 instead of an error event after a 200.
        """
        return await run_in_threadpool(self.engine.fit, messages, params)

    # -- admission ---------------------------------------------------------------
    async def admit(self) -> Slot:
        started = time.perf_counter()
        try:
            slot = await self.admission.acquire()
        except QueueFullError:
            self.metrics.rejections.labels(reason="queue_full").inc()
            raise
        except QueueTimeoutError:
            self.metrics.rejections.labels(reason="queue_timeout").inc()
            raise
        self.metrics.queue_wait_seconds.observe(time.perf_counter() - started)
        return slot

    # -- generation --------------------------------------------------------------
    def _observe(
        self,
        result_tokens: int,
        prompt_tokens: int,
        finish: str,
        seconds: float,
        endpoint: str,
        stream: bool,
    ) -> None:
        m = self.metrics
        m.generation_seconds.labels(endpoint=endpoint, stream=str(stream).lower()).observe(seconds)
        m.prompt_tokens.inc(prompt_tokens)
        m.completion_tokens.inc(result_tokens)
        m.finish_reasons.labels(reason=finish).inc()
        if seconds > 0 and result_tokens > 0:
            m.tokens_per_second.observe(result_tokens / seconds)

    async def complete(
        self, messages: Sequence[Message], params: GenerationParams, endpoint: str
    ) -> GenerationResult:
        slot = await self.admit()
        started = time.perf_counter()
        try:
            result = await run_in_threadpool(self.engine.generate, messages, params)
        except Exception:
            self.metrics.generation_errors.inc()
            raise
        finally:
            slot.release()
        self._observe(
            result.completion_tokens,
            result.prompt_tokens,
            result.finish_reason,
            time.perf_counter() - started,
            endpoint,
            stream=False,
        )
        return result

    async def stream(
        self,
        messages: Sequence[Message],
        params: GenerationParams,
        slot: Slot,
        endpoint: str,
    ) -> AsyncIterator[StreamEvent]:
        """Yield stream events while holding ``slot``; always releases it.

        If the client disconnects, the engine is told to stop via ``cancel`` and the
        underlying generator is closed off the event loop.
        """
        cancel = threading.Event()
        gen = self.engine.stream(messages, params, cancel)
        started = time.perf_counter()
        first_token = True
        try:
            async for event in iterate_in_threadpool(gen):
                if first_token and event.delta:
                    first_token = False
                    self.metrics.time_to_first_token_seconds.labels(endpoint=endpoint).observe(
                        time.perf_counter() - started
                    )
                if event.is_final:
                    self._observe(
                        event.completion_tokens,
                        event.prompt_tokens,
                        event.finish_reason or "stop",
                        time.perf_counter() - started,
                        endpoint,
                        stream=True,
                    )
                yield event
        except Exception:
            self.metrics.generation_errors.inc()
            raise
        finally:
            cancel.set()
            threading.Thread(target=_close_quietly, args=(gen,), daemon=True).start()
            slot.release()

    # -- accounting --------------------------------------------------------------
    def record_usage(self, event: UsageEvent) -> None:
        """Persist a usage event (blocking; run as a background task)."""
        if self.store is None:
            return
        if not self.settings.store_transcripts:
            event.question = None
            event.answer = None
        try:
            self.store.record_usage(event)
        except Exception:
            log.exception("Failed to record usage", extra={"request_id": event.request_id})
