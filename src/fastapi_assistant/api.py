"""FastAPI application factory and HTTP routes.

The app is built by :func:`create_app` so that tests can inject a fake engine and
disable model warmup. A module-level ``app`` is exposed for ASGI servers
(``uvicorn fastapi_assistant.api:app``).
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from . import __version__
from .config import Settings, get_settings
from .inference import AssistantEngine, ModelNotReadyError
from .schemas import GenerateRequest, GenerateResponse, HealthResponse

log = logging.getLogger(__name__)


def get_engine(request: Request) -> AssistantEngine:
    """FastAPI dependency: resolve the engine from application state."""
    return request.app.state.engine


def create_app(
    settings: Settings | None = None,
    engine: AssistantEngine | None = None,
) -> FastAPI:
    """Build and configure a FastAPI application instance."""
    settings = settings or get_settings()
    engine = engine or AssistantEngine(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if settings.warmup_on_startup:
            try:
                engine.load()
            except ModelNotReadyError:
                # Don't crash startup; /ready will report the failure and retry.
                log.warning("Model warmup failed at startup; will retry on first /ready or /generate")
        yield
        log.info("Shutting down FastAPI assistant")

    app = FastAPI(
        title="FastAPI Niche Assistant",
        version=__version__,
        summary="Fine-tuned TinyLlama assistant specialized in FastAPI.",
        description=(
            "A LoRA fine-tuned TinyLlama model that answers FastAPI questions, "
            "served with health/readiness probes, structured logging, and optional "
            "API-key auth."
        ),
        contact={"name": "Syed Waleed Ahmed"},
        license_info={"name": "MIT", "url": "https://opensource.org/licenses/MIT"},
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.engine = engine

    if settings.cors_allow_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_allow_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        start = time.perf_counter()
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
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        response.headers["x-request-id"] = request_id
        response.headers["x-process-time-ms"] = f"{duration_ms:.2f}"
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

    def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
        """Reject requests without a valid API key when one is configured."""
        if settings.api_key and x_api_key != settings.api_key:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    def health(engine: AssistantEngine = Depends(get_engine)) -> HealthResponse:
        """Liveness probe: the process is up. Does not force a model load."""
        return HealthResponse(
            status="ok",
            model_loaded=engine.is_loaded,
            uses_adapter=engine.uses_adapter,
        )

    @app.get("/ready", response_model=HealthResponse, tags=["ops"])
    def ready(engine: AssistantEngine = Depends(get_engine)) -> HealthResponse:
        """Readiness probe: ensures the model can be loaded before serving traffic."""
        try:
            engine.load()
        except ModelNotReadyError as exc:
            raise HTTPException(status_code=503, detail=f"Model not ready: {exc}") from exc
        return HealthResponse(
            status="ready",
            model_loaded=engine.is_loaded,
            uses_adapter=engine.uses_adapter,
        )

    @app.post(
        "/generate",
        response_model=GenerateResponse,
        tags=["inference"],
        dependencies=[Depends(require_api_key)],
    )
    def generate(
        payload: GenerateRequest,
        engine: AssistantEngine = Depends(get_engine),
    ) -> GenerateResponse:
        """Generate an answer to a FastAPI question."""
        try:
            answer = engine.generate(
                payload.question,
                max_new_tokens=payload.max_new_tokens,
                temperature=payload.temperature,
                top_p=payload.top_p,
            )
        except ModelNotReadyError as exc:
            raise HTTPException(status_code=503, detail=f"Model not ready: {exc}") from exc
        except Exception as exc:  # pragma: no cover - defensive guard
            log.exception("Generation failed")
            raise HTTPException(status_code=500, detail="Generation failed") from exc

        return GenerateResponse(
            question=payload.question,
            answer=answer,
            model=settings.base_model_name,
            uses_adapter=engine.uses_adapter,
        )

    return app


# ASGI entrypoint for `uvicorn fastapi_assistant.api:app`.
app = create_app()
