"""Remote backend: delegate generation to an OpenAI-compatible server.

This is the "offload inference" scaling step: vLLM, TGI, or ``llama.cpp`` server
provide continuous batching and a paged KV cache, and this service becomes a thin
gateway that keeps the same HTTP contract, auth, metrics, and admission control.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator, Sequence

import httpx

from ..config import Settings
from .base import BaseEngine, Deadline, GenerationParams, GenerationResult, Message, StreamEvent


def _finish(value: str | None) -> str:
    return "length" if value == "length" else "stop"


class RemoteEngine(BaseEngine):
    backend = "remote"

    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None) -> None:
        if not settings.remote_base_url:
            raise ValueError("REMOTE_BASE_URL is required for the remote backend")
        super().__init__(settings.remote_model or settings.base_model_name)
        self._settings = settings
        headers = (
            {"Authorization": f"Bearer {settings.remote_api_key}"}
            if settings.remote_api_key
            else {}
        )
        self._client = httpx.Client(
            base_url=settings.remote_base_url.rstrip("/"),
            headers=headers,
            timeout=httpx.Timeout(settings.remote_timeout_s, connect=10.0),
            transport=transport,
        )

    def _load_impl(self) -> None:
        # "Loading" a remote model means verifying the upstream is reachable.
        resp = self._client.get("/v1/models")
        resp.raise_for_status()
        self._uses_adapter = self._settings.finetuned_weights

    def _payload(self, messages: Sequence[Message], params: GenerationParams, stream: bool) -> dict:
        return {
            "model": self.model_id,
            "messages": list(messages),
            "max_tokens": params.max_new_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
            # Not part of the OpenAI spec; accepted by vLLM and llama.cpp server.
            "repetition_penalty": params.repetition_penalty,
            "stream": stream,
        }

    def generate(self, messages: Sequence[Message], params: GenerationParams) -> GenerationResult:
        self.load()
        resp = self._client.post(
            "/v1/chat/completions",
            json=self._payload(messages, params, stream=False),
            timeout=params.timeout_s or self._settings.remote_timeout_s,
        )
        resp.raise_for_status()
        body = resp.json()
        choice = body["choices"][0]
        usage = body.get("usage") or {}
        return GenerationResult(
            text=(choice["message"].get("content") or "").strip(),
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            finish_reason=_finish(choice.get("finish_reason")),
        )

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
        with self._client.stream(
            "POST", "/v1/chat/completions", json=self._payload(messages, params, stream=True)
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if deadline.expired():
                    finish = deadline.reason
                    break
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if not chunk.get("choices"):
                    continue
                choice = chunk["choices"][0]
                delta = (choice.get("delta") or {}).get("content")
                if delta:
                    completion_tokens += 1
                    yield StreamEvent(delta=delta)
                if choice.get("finish_reason"):
                    finish = _finish(choice["finish_reason"])
        yield StreamEvent(finish_reason=finish or "stop", completion_tokens=completion_tokens)

    def close(self) -> None:
        self._client.close()
