"""FastAPI application factory and HTTP routes.

The app is built by :func:`create_app` so that tests can inject a fake engine and
store. ASGI servers can use the factory directly
(``uvicorn --factory fastapi_assistant.api:create_app``); ``fastapi_assistant.api:app``
is also available and is built on first access.

Routes:

* ``GET /health`` (liveness), ``GET /ready`` (readiness), ``GET /metrics`` (Prometheus)
* ``POST /generate`` — native API, JSON or Server-Sent Events
* ``GET /v1/models``, ``POST /v1/chat/completions`` — OpenAI-compatible
* ``POST /v1/feedback`` — rate an answer by request id
"""

from __future__ import annotations

import hmac
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from functools import partial
from typing import Any, Literal

from fastapi import APIRouter, BackgroundTasks, Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import __version__
from .admission import AdmissionController, AdmissionRejected, QueueFullError
from .auth import Authenticator, AuthError, extract_api_key
from .config import Settings, get_settings
from .engines import Engine, ModelNotReadyError, create_engine
from .engines.base import PromptTooLongError
from .limits import BodySizeLimitMiddleware
from .metrics import Metrics
from .prompts import build_messages, normalize_chat
from .retrieval import Passage, Retriever, create_retriever
from .schemas import (
    ChatChoice,
    ChatChoiceMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    FeedbackRequest,
    FeedbackResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    ModelCard,
    ModelList,
    Source,
    Usage,
)
from .service import AssistantService
from .storage import Feedback, Principal, Store, UsageEvent, create_store

log = logging.getLogger(__name__)

PUBLIC_MODEL_ID = "fastapi-assistant"
# Probes stay reachable without the edge secret so platforms can health-check.
_OPEN_PATHS = frozenset({"/health", "/ready", "/metrics"})
# The API returns JSON and SSE only; these stop a browser from sniffing a response
# into something executable or framing the interactive docs.
# Streams report an overflow found mid-generation (a rare backstop; `fit` rejects
# over-long prompts with a precise 400 before admission) with fixed text, so no
# exception text ever reaches a client.
_PROMPT_TOO_LONG_IN_STREAM = "The prompt does not fit the model's context window"
_SECURITY_HEADERS = {
    "x-content-type-options": "nosniff",
    "x-frame-options": "DENY",
    "referrer-policy": "no-referrer",
}


async def _client_gone(request: Request) -> None:
    """Return once the client has disconnected.

    Waits on the connection's receive channel, as Starlette's streaming responses do.
    ``request.is_disconnected()`` is not used: it polls without waiting, and behind the
    request-context middleware that poll never sees the disconnect message.
    """
    while True:
        message = await request.receive()
        if message["type"] == "http.disconnect":
            return


def _sse(data: Any, event: str | None = None) -> str:
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {payload}\n\n"


def _openai_finish(reason: str) -> Literal["stop", "length"]:
    return "length" if reason in ("length", "timeout", "cancelled") else "stop"


def _sources(passages: list[Passage]) -> list[Source]:
    return [Source(url=p.url, heading=p.heading, score=round(p.score, 4)) for p in passages]


def _init_sentry(settings: Settings) -> None:
    # Anything that is not a DSN URL (unset, "", or a "disabled" placeholder kept in
    # Secret Manager so the Cloud Run spec stays valid) leaves Sentry off.
    if not (settings.sentry_dsn or "").startswith(("https://", "http://")):
        return
    import sentry_sdk

    sentry_sdk.init(
        dsn=settings.sentry_dsn,
        environment=settings.sentry_environment,
        release=f"fastapi-assistant@{__version__}+{settings.model_version}",
        traces_sample_rate=settings.sentry_traces_sample_rate,
        send_default_pii=False,
    )


def create_app(
    settings: Settings | None = None,
    engine: Engine | None = None,
    store: Store | None = None,
    retriever: Retriever | None = None,
) -> FastAPI:
    """Build and configure a FastAPI application instance."""
    settings = settings or get_settings()
    engine = engine or create_engine(settings)
    store = store if store is not None else create_store(settings)
    metrics = Metrics()
    admission = AdmissionController(
        settings.max_concurrency, settings.max_queue, settings.queue_timeout_s
    )
    authenticator = Authenticator(
        settings.api_key, store, settings.key_cache_ttl_s, allow_anonymous=settings.public_demo
    )
    if settings.public_demo:
        log.info("Public demo mode: requests without an API key are accepted as anonymous.")
    if not authenticator.enabled:
        log.warning(
            "Authentication is DISABLED: neither API_KEY nor DATABASE_URL is set, so every "
            "caller is accepted. Fine for local development; never for a public origin."
        )
    service = AssistantService(settings, engine, admission, metrics, store, retriever)
    _init_sentry(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if settings.warmup_on_startup:
            # Load in the background so the server accepts probes immediately;
            # /ready reports 503 until the weights are in memory.
            engine.start_background_load()
        # Load the index whenever one is available: RAG_ENABLED only sets the
        # per-request default, and clients can opt in with use_retrieval=true.
        index_available = (
            settings.rag_backend == "pgvector"
            and bool(settings.database_url)
            or (
                settings.rag_backend == "local" and (settings.rag_index_dir / "meta.json").is_file()
            )
        )
        if service.retriever is None and (settings.rag_enabled or index_available):
            try:
                service.retriever = await run_in_threadpool(
                    create_retriever, settings.model_copy(update={"rag_enabled": True})
                )
            except Exception:
                log.exception("Retrieval disabled: failed to initialize the index")
        yield
        log.info("Shutting down FastAPI assistant")
        engine.close()
        if store is not None:
            store.close()

    app = FastAPI(
        title="FastAPI Niche Assistant",
        version=__version__,
        summary="Fine-tuned small language model specialized in FastAPI.",
        description=(
            "A LoRA fine-tuned model that answers FastAPI questions, with an "
            "OpenAI-compatible API, streaming, retrieval over the official docs, "
            "admission control, and per-user API keys."
        ),
        contact={"name": "Syed Waleed Ahmed"},
        license_info={"name": "MIT", "url": "https://opensource.org/licenses/MIT"},
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.engine = engine
    app.state.service = service
    app.state.metrics = metrics
    app.state.store = store

    if settings.cors_allow_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_allow_origins,
            # Auth is header-based (API keys), so cookies are never needed.
            allow_credentials=False,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["*"],
            expose_headers=[
                "X-Request-ID",
                "X-Process-Time-Ms",
                "Retry-After",
                # Set by the edge Worker and passed through with the origin's CORS headers.
                "X-Cache",
                "X-Served-By",
                "X-RateLimit-Limit",
                "X-RateLimit-Remaining",
                "X-RateLimit-Reset",
            ],
        )
    # Wraps ``receive``, so the cap applies before any handler reads or parses the
    # body (and so before authentication). The ``request_context`` middleware below
    # is registered later and runs outside it, so the origin-secret check comes first.
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.max_request_bytes)

    # -- middleware ---------------------------------------------------------------
    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        request.state.request_id = request_id
        start = time.perf_counter()

        if settings.origin_shared_secret and request.url.path not in _OPEN_PATHS:
            supplied = request.headers.get("x-origin-secret", "")
            if not hmac.compare_digest(supplied, settings.origin_shared_secret):
                return JSONResponse(
                    {"detail": "Direct origin access is not allowed"},
                    status_code=403,
                    headers={"x-request-id": request_id, **_SECURITY_HEADERS},
                )

        try:
            response = await call_next(request)
        except Exception:
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            log.exception(
                "Unhandled request error",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "duration_ms": duration_ms,
                },
            )
            raise
        elapsed = time.perf_counter() - start
        duration_ms = round(elapsed * 1000, 2)
        route = request.scope.get("route")
        route_label = getattr(route, "path", "unmatched")
        metrics.requests.labels(
            route=route_label, method=request.method, status=str(response.status_code)
        ).inc()
        metrics.request_seconds.labels(route=route_label).observe(elapsed)
        response.headers["x-request-id"] = request_id
        response.headers["x-process-time-ms"] = f"{duration_ms:.2f}"
        for name, value in _SECURITY_HEADERS.items():
            response.headers.setdefault(name, value)
        log.info(
            "request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
            },
        )
        return response

    # -- error mapping -------------------------------------------------------------
    def _error(
        request: Request,
        status: int,
        message: str,
        err_type: str,
        headers=None,
        code: str | int | None = None,
    ):
        if request.url.path.startswith("/v1/"):
            body: dict[str, Any] = {
                "error": {"message": message, "type": err_type, "code": code or status}
            }
        else:
            body = {"detail": message}
        return JSONResponse(body, status_code=status, headers=headers)

    def _error_type(status: int) -> str:
        if status >= 500:
            return "server_error"
        return {
            401: "authentication_error",
            403: "permission_error",
            404: "not_found",
            429: "rate_limit_error",
        }.get(status, "invalid_request_error")

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        return _error(
            request,
            exc.status_code,
            str(exc.detail),
            _error_type(exc.status_code),
            getattr(exc, "headers", None),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        if request.url.path.startswith("/v1/"):
            first = exc.errors()[0] if exc.errors() else {}
            loc = ".".join(str(p) for p in first.get("loc", ()))
            return _error(
                request,
                400,
                f"{loc}: {first.get('msg', 'invalid request')}",
                "invalid_request_error",
            )
        return JSONResponse({"detail": jsonable_errors(exc)}, status_code=422)

    @app.exception_handler(AdmissionRejected)
    async def admission_error(request: Request, exc: AdmissionRejected):
        status = 429 if isinstance(exc, QueueFullError) else 503
        return _error(
            request, status, str(exc), "overloaded", {"Retry-After": str(exc.retry_after_s)}
        )

    @app.exception_handler(ModelNotReadyError)
    async def not_ready_error(request: Request, exc: ModelNotReadyError):
        return _error(request, 503, "Model not ready", "model_not_ready", {"Retry-After": "30"})

    @app.exception_handler(PromptTooLongError)
    async def prompt_too_long_error(request: Request, exc: PromptTooLongError):
        metrics.prompt_rejections.inc()
        return _error(
            request, 400, str(exc), "invalid_request_error", code="context_length_exceeded"
        )

    # -- dependencies --------------------------------------------------------------
    async def require_principal(
        request: Request,
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
        authorization: str | None = Header(default=None),
    ) -> Principal:
        key = extract_api_key(x_api_key, authorization)
        try:
            if store is None:  # static key only: no I/O, skip the threadpool hop
                principal = authenticator.authenticate(key)
            else:
                principal = await run_in_threadpool(authenticator.authenticate, key)
        except AuthError as exc:
            raise HTTPException(
                status_code=401, detail=str(exc), headers={"WWW-Authenticate": "Bearer"}
            ) from exc
        request.state.principal = principal
        return principal

    def _health(status: str) -> HealthResponse:
        return HealthResponse(
            status=status,
            model_loaded=engine.is_loaded,
            uses_adapter=engine.uses_adapter,
            backend=engine.backend,
            model_version=settings.model_version,
            load_state=engine.load_state,
            access="public" if settings.public_demo or not authenticator.enabled else "key",
        )

    # -- ops routes ------------------------------------------------------------------
    ops = APIRouter(tags=["ops"])

    @ops.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """Liveness probe: the process is up. Never loads the model."""
        return _health("ok")

    @ops.get("/ready", response_model=HealthResponse, responses={503: {"description": "Not ready"}})
    def ready() -> HealthResponse:
        """Readiness probe: 200 once the model is loaded; starts loading if idle.

        Non-blocking, so platform probes never time out while weights download.
        """
        if engine.is_loaded:
            return _health("ready")
        engine.start_background_load()
        # The load error itself is logged server-side, never returned to clients.
        raise HTTPException(status_code=503, detail=f"Model not ready: {engine.load_state}")

    @ops.get("/metrics", include_in_schema=False)
    def metrics_endpoint(authorization: str | None = Header(default=None)) -> Response:
        if not settings.enable_metrics:
            raise HTTPException(status_code=404, detail="Not Found")
        if settings.metrics_token:
            token = extract_api_key(None, authorization) or ""
            if not hmac.compare_digest(token, settings.metrics_token):
                raise HTTPException(status_code=401, detail="Invalid metrics token")
        body, content_type = metrics.render()
        return Response(content=body, media_type=content_type)

    # -- native API ------------------------------------------------------------------
    inference = APIRouter(tags=["inference"])

    @inference.post(
        "/generate",
        response_model=GenerateResponse,
        responses={
            200: {
                "content": {"text/event-stream": {}},
                "description": "JSON, or SSE if stream=true",
            },
            401: {"description": "Missing or invalid API key"},
            429: {"description": "Queue full; see Retry-After"},
            503: {"description": "Model not ready or queue timeout"},
        },
    )
    async def generate(
        payload: GenerateRequest,
        request: Request,
        background: BackgroundTasks,
        principal: Principal = Depends(require_principal),
    ):
        """Answer a FastAPI question."""
        request_id = request.state.request_id
        started = time.perf_counter()
        passages = await service.retrieve(payload.question, payload.use_retrieval)
        messages = build_messages(payload.question, service.context(passages))
        params = await service.fit(
            messages, service.params(payload.max_new_tokens, payload.temperature, payload.top_p)
        )

        if payload.stream:
            slot = await service.admit()

            async def events():
                parts: list[str] = []
                try:
                    async for event in service.stream(messages, params, slot, "/generate"):
                        if event.delta:
                            parts.append(event.delta)
                            yield _sse({"delta": event.delta}, event="token")
                        if event.is_final:
                            usage = Usage(
                                prompt_tokens=event.prompt_tokens,
                                completion_tokens=event.completion_tokens,
                                total_tokens=event.prompt_tokens + event.completion_tokens,
                            )
                            yield _sse(
                                {
                                    "finish_reason": event.finish_reason,
                                    "usage": usage.model_dump(),
                                    "sources": [s.model_dump() for s in _sources(passages)],
                                    "request_id": request_id,
                                    "model_version": settings.model_version,
                                },
                                event="done",
                            )
                            await run_in_threadpool(
                                service.record_usage,
                                UsageEvent(
                                    request_id=request_id,
                                    endpoint="/generate",
                                    status_code=200,
                                    latency_ms=(time.perf_counter() - started) * 1000,
                                    model_version=settings.model_version,
                                    key_id=principal.key_id,
                                    prompt_tokens=event.prompt_tokens,
                                    completion_tokens=event.completion_tokens,
                                    finish_reason=event.finish_reason,
                                    question=payload.question,
                                    answer="".join(parts),
                                ),
                            )
                except Exception as exc:
                    log.exception("Streaming generation failed", extra={"request_id": request_id})
                    if isinstance(exc, ModelNotReadyError):
                        message = "Model not ready"
                    elif isinstance(exc, PromptTooLongError):
                        message = _PROMPT_TOO_LONG_IN_STREAM
                    else:
                        message = "Generation failed"
                    yield _sse({"message": message}, event="error")

            return StreamingResponse(
                events(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                background=BackgroundTask(slot.release),
            )

        try:
            result = await service.complete(
                messages, params, "/generate", client_gone=partial(_client_gone, request)
            )
        except (AdmissionRejected, ModelNotReadyError, PromptTooLongError):
            raise
        except Exception as exc:
            log.exception("Generation failed", extra={"request_id": request_id})
            raise HTTPException(status_code=500, detail="Generation failed") from exc

        background.add_task(
            service.record_usage,
            UsageEvent(
                request_id=request_id,
                endpoint="/generate",
                status_code=200,
                latency_ms=(time.perf_counter() - started) * 1000,
                model_version=settings.model_version,
                key_id=principal.key_id,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                finish_reason=result.finish_reason,
                question=payload.question,
                answer=result.text,
            ),
        )
        return GenerateResponse(
            question=payload.question,
            answer=result.text,
            model=engine.model_id,
            model_version=settings.model_version,
            uses_adapter=engine.uses_adapter,
            finish_reason=result.finish_reason,
            usage=Usage(
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                total_tokens=result.prompt_tokens + result.completion_tokens,
            ),
            sources=_sources(passages),
            request_id=request_id,
        )

    # -- OpenAI-compatible API -------------------------------------------------------
    openai = APIRouter(prefix="/v1", tags=["openai-compatible"])

    @openai.get("/models", response_model=ModelList)
    async def list_models(principal: Principal = Depends(require_principal)) -> ModelList:
        return ModelList(data=[ModelCard(id=PUBLIC_MODEL_ID)])

    @openai.post(
        "/chat/completions",
        response_model=ChatCompletionResponse,
        responses={200: {"content": {"text/event-stream": {}}}},
    )
    async def chat_completions(
        payload: ChatCompletionRequest,
        request: Request,
        background: BackgroundTasks,
        principal: Principal = Depends(require_principal),
    ):
        """OpenAI Chat Completions compatible endpoint (supports ``stream``)."""
        request_id = request.state.request_id
        started = time.perf_counter()
        raw = [m.model_dump() for m in payload.messages]
        if sum(len(m["content"]) for m in raw) > settings.max_input_chars:
            raise HTTPException(status_code=400, detail="Conversation is too long")
        last_user = next((m["content"] for m in reversed(raw) if m["role"] == "user"), "")
        passages = await service.retrieve(last_user) if last_user else []
        try:
            messages = normalize_chat(
                raw, max_turns=settings.max_chat_turns, context=service.context(passages)
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail="The conversation must end with a user message"
            ) from exc
        params = await service.fit(
            messages,
            service.params(
                payload.max_completion_tokens or payload.max_tokens,
                payload.temperature,
                payload.top_p,
            ),
        )
        completion_id = f"chatcmpl-{request_id}"
        created = int(time.time())

        if payload.stream:
            slot = await service.admit()

            def chunk(delta: dict, finish: str | None = None) -> str:
                return _sse(
                    {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": PUBLIC_MODEL_ID,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                    }
                )

            async def events():
                parts: list[str] = []
                yield chunk({"role": "assistant"})
                try:
                    async for event in service.stream(
                        messages, params, slot, "/v1/chat/completions"
                    ):
                        if event.delta:
                            parts.append(event.delta)
                            yield chunk({"content": event.delta})
                        if event.is_final:
                            yield chunk({}, _openai_finish(event.finish_reason or "stop"))
                            await run_in_threadpool(
                                service.record_usage,
                                UsageEvent(
                                    request_id=request_id,
                                    endpoint="/v1/chat/completions",
                                    status_code=200,
                                    latency_ms=(time.perf_counter() - started) * 1000,
                                    model_version=settings.model_version,
                                    key_id=principal.key_id,
                                    prompt_tokens=event.prompt_tokens,
                                    completion_tokens=event.completion_tokens,
                                    finish_reason=event.finish_reason,
                                    question=last_user,
                                    answer="".join(parts),
                                ),
                            )
                except PromptTooLongError:
                    yield _sse(
                        {
                            "error": {
                                "message": _PROMPT_TOO_LONG_IN_STREAM,
                                "type": "invalid_request_error",
                                "code": "context_length_exceeded",
                            }
                        }
                    )
                except Exception:
                    log.exception("Streaming generation failed", extra={"request_id": request_id})
                    yield _sse({"error": {"message": "Generation failed", "type": "server_error"}})
                yield _sse("[DONE]")

            return StreamingResponse(
                events(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                background=BackgroundTask(slot.release),
            )

        try:
            result = await service.complete(
                messages,
                params,
                "/v1/chat/completions",
                client_gone=partial(_client_gone, request),
            )
        except (AdmissionRejected, ModelNotReadyError, PromptTooLongError):
            raise
        except Exception as exc:
            log.exception("Generation failed", extra={"request_id": request_id})
            raise HTTPException(status_code=500, detail="Generation failed") from exc

        background.add_task(
            service.record_usage,
            UsageEvent(
                request_id=request_id,
                endpoint="/v1/chat/completions",
                status_code=200,
                latency_ms=(time.perf_counter() - started) * 1000,
                model_version=settings.model_version,
                key_id=principal.key_id,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                finish_reason=result.finish_reason,
                question=last_user,
                answer=result.text,
            ),
        )
        return ChatCompletionResponse(
            id=completion_id,
            created=created,
            model=PUBLIC_MODEL_ID,
            choices=[
                ChatChoice(
                    message=ChatChoiceMessage(content=result.text),
                    finish_reason=_openai_finish(result.finish_reason),
                )
            ],
            usage=Usage(
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                total_tokens=result.prompt_tokens + result.completion_tokens,
            ),
        )

    @openai.post("/feedback", response_model=FeedbackResponse)
    async def feedback(
        payload: FeedbackRequest, principal: Principal = Depends(require_principal)
    ) -> FeedbackResponse:
        """Rate an answer by the ``X-Request-ID`` it was served with."""
        record = Feedback(
            request_id=payload.request_id,
            rating=payload.rating,
            key_id=principal.key_id,
            comment=payload.comment,
        )
        if store is None:
            # Without a database the rating is only logged. The comment is free text a
            # user typed, so only its length goes to the logs.
            log.info(
                "feedback rating=%s comment_chars=%d",
                payload.rating,
                len(payload.comment or ""),
                extra={"request_id": payload.request_id},
            )
        else:
            await run_in_threadpool(store.record_feedback, record)
        return FeedbackResponse()

    app.include_router(ops)
    app.include_router(inference)
    app.include_router(openai)
    return app


def jsonable_errors(exc: RequestValidationError) -> list[dict[str, Any]]:
    from fastapi.encoders import jsonable_encoder

    return jsonable_encoder(exc.errors())


app: FastAPI  # ``fastapi_assistant.api:app``, built on first access below


def __getattr__(name: str):
    # ``fastapi_assistant.api:app`` for ASGI servers, built lazily so importing this
    # module (e.g. in tests) never constructs an engine from the environment.
    if name == "app":
        global app
        app = create_app()
        return app
    raise AttributeError(name)
