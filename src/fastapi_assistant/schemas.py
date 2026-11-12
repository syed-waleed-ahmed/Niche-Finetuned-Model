"""Pydantic request/response models for the HTTP API.

These form the public contract of the service and drive automatic validation and
OpenAPI documentation. ``/generate`` is the native API; ``/v1/*`` mirrors the
OpenAI Chat Completions API so existing SDKs and tools work unchanged.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Model(BaseModel):
    model_config = ConfigDict(protected_namespaces=())


# -- native API ---------------------------------------------------------------
class GenerateRequest(_Model):
    """Input payload for POST /generate."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="A FastAPI-related question.",
        examples=["How do I define a POST endpoint in FastAPI?"],
    )
    max_new_tokens: int | None = Field(
        default=None,
        ge=1,
        le=1024,
        description="Override the server default for maximum generated tokens.",
    )
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. 0 disables sampling (greedy decoding).",
    )
    top_p: float | None = Field(
        default=None,
        gt=0.0,
        le=1.0,
        description="Nucleus sampling probability mass.",
    )
    stream: bool = Field(
        default=False,
        description="Stream the answer as Server-Sent Events (event: token / done / error).",
    )
    use_retrieval: bool | None = Field(
        default=None,
        description="Ground the answer in retrieved documentation (defaults to RAG_ENABLED).",
    )


class Usage(_Model):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class Source(_Model):
    url: str
    heading: str
    score: float


class GenerateResponse(_Model):
    """Response payload for POST /generate."""

    question: str
    answer: str
    model: str = Field(description="Model identifier used for generation.")
    model_version: str = Field(description="Deployed release label.")
    uses_adapter: bool = Field(
        description="Whether fine-tuned weights (LoRA adapter or merged GGUF) are in use."
    )
    finish_reason: str = Field(description="stop, length, or timeout.")
    usage: Usage
    sources: list[Source] = Field(default_factory=list)
    request_id: str


class HealthResponse(_Model):
    """Response payload for /health and /ready."""

    status: str
    model_loaded: bool
    uses_adapter: bool
    backend: str = ""
    model_version: str = ""
    load_state: str = ""


class FeedbackRequest(_Model):
    request_id: str = Field(..., min_length=1, max_length=128)
    rating: Literal[-1, 1] = Field(description="1 for a good answer, -1 for a bad one.")
    comment: str | None = Field(default=None, max_length=2000)


class FeedbackResponse(_Model):
    status: Literal["recorded"] = "recorded"


# -- OpenAI-compatible API ------------------------------------------------------
class ChatMessage(_Model):
    role: Literal["system", "user", "assistant"]
    content: str = Field(max_length=8000)


class ChatCompletionRequest(_Model):
    model: str | None = Field(default=None, description="Ignored; the service serves one model.")
    messages: list[ChatMessage] = Field(..., min_length=1, max_length=100)
    max_tokens: int | None = Field(default=None, ge=1, le=1024)
    max_completion_tokens: int | None = Field(default=None, ge=1, le=1024)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    stream: bool = False
    n: int = Field(default=1, ge=1, le=1, description="Only n=1 is supported.")
    user: str | None = None


class ChatChoiceMessage(_Model):
    role: Literal["assistant"] = "assistant"
    content: str


class ChatChoice(_Model):
    index: int = 0
    message: ChatChoiceMessage
    finish_reason: Literal["stop", "length"]


class ChatCompletionResponse(_Model):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: Usage


class ModelCard(_Model):
    id: str
    object: Literal["model"] = "model"
    created: int = 0
    owned_by: str = "fastapi-assistant"


class ModelList(_Model):
    object: Literal["list"] = "list"
    data: list[ModelCard]
