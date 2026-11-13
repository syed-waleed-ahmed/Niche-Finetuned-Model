"""Backward-compatible import location for the torch engine.

New code should use :func:`fastapi_assistant.engines.create_engine`, which selects
the backend from configuration. ``AssistantEngine`` remains as an alias of the
transformers + LoRA engine.
"""

from __future__ import annotations

from .engines.base import GenerationParams, ModelNotReadyError
from .engines.transformers_engine import TransformersEngine as AssistantEngine

__all__ = ["AssistantEngine", "GenerationParams", "ModelNotReadyError"]
