from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .inference import generate_answer, load_finetuned_model


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_loaded = False
    logger.info("Warming model for startup")
    load_finetuned_model()
    app.state.model_loaded = True
    try:
        yield
    finally:
        logger.info("Shutting down FastAPI assistant")


app = FastAPI(
    title="FastAPI Niche Assistant",
    version="1.0.0",
    description="A production-ready FastAPI service for a fine-tuned TinyLlama assistant specialized in FastAPI questions.",
    summary="Fine-tuned FastAPI assistant",
    lifespan=lifespan,
    contact={"name": "Syed Waleed Ahmed"},
    license_info={"name": "Proprietary / Internal Use"},
)


class GenerateRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    max_new_tokens: int = Field(default=256, ge=1, le=512)


class GenerateResponse(BaseModel):
    question: str
    answer: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=bool(getattr(app.state, "model_loaded", False)))


@app.get("/ready", response_model=HealthResponse)
def ready() -> HealthResponse:
    model_loaded = bool(getattr(app.state, "model_loaded", False))
    try:
        load_finetuned_model()
    except Exception as exc:  # pragma: no cover - readiness should surface startup failures clearly
        app.state.model_loaded = False
        raise HTTPException(status_code=503, detail=f"Model is not ready: {exc}") from exc

    app.state.model_loaded = True
    return HealthResponse(status="ready", model_loaded=model_loaded)


@app.post("/generate", response_model=GenerateResponse)
def generate(payload: GenerateRequest) -> GenerateResponse:
    answer = generate_answer(payload.question, max_new_tokens=payload.max_new_tokens)
    return GenerateResponse(question=payload.question, answer=answer)
