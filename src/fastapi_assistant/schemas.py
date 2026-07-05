"""Pydantic request/response models for the HTTP API.

These form the public contract of the service and drive automatic validation and
OpenAPI documentation.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class GenerateRequest(BaseModel):
    """Input payload for POST /generate."""

    model_config = ConfigDict(protected_namespaces=())

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


class GenerateResponse(BaseModel):
    """Response payload for POST /generate."""

    model_config = ConfigDict(protected_namespaces=())

    question: str
    answer: str
    model: str = Field(description="Base model identifier used for generation.")
    uses_adapter: bool = Field(
        description="Whether a fine-tuned LoRA adapter is applied on top of the base model."
    )


class HealthResponse(BaseModel):
    """Response payload for /health and /ready."""

    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    uses_adapter: bool
