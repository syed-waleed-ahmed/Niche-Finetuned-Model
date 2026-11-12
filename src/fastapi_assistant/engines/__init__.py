"""Inference backends behind one :class:`Engine` contract."""

from __future__ import annotations

from ..config import Settings
from .base import (
    BaseEngine,
    Engine,
    GenerationParams,
    GenerationResult,
    Message,
    ModelNotReadyError,
    StreamEvent,
)

__all__ = [
    "BaseEngine",
    "Engine",
    "GenerationParams",
    "GenerationResult",
    "Message",
    "ModelNotReadyError",
    "StreamEvent",
    "create_engine",
]


def create_engine(settings: Settings) -> Engine:
    """Build the engine selected by ``ENGINE_BACKEND``.

    Backends are imported lazily so that, for example, the llama.cpp image does
    not need torch installed.
    """
    if settings.engine_backend == "llamacpp":
        from .llamacpp_engine import LlamaCppEngine

        return LlamaCppEngine(settings)
    if settings.engine_backend == "remote":
        from .remote_engine import RemoteEngine

        return RemoteEngine(settings)
    from .transformers_engine import TransformersEngine

    return TransformersEngine(settings)
