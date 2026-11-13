"""Prometheus metrics.

Each app instance owns its own registry, so tests (which build many apps) never
collide on duplicate metric names. Labels are bounded: routes use the path
template (``/v1/chat/completions``), never the raw URL.
"""

from __future__ import annotations

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
)

_LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60, 120)
_TPS_BUCKETS = (1, 2, 5, 10, 15, 20, 30, 50, 75, 100, 200)


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry(auto_describe=True)
        r = self.registry
        self.build_info = Info("assistant_build", "Service build and model identity", registry=r)
        self.requests = Counter(
            "assistant_http_requests_total",
            "HTTP requests by route and status code",
            ["route", "method", "status"],
            registry=r,
        )
        self.request_seconds = Histogram(
            "assistant_http_request_duration_seconds",
            "HTTP request latency (time to response headers for streams)",
            ["route"],
            buckets=_LATENCY_BUCKETS,
            registry=r,
        )
        self.generation_seconds = Histogram(
            "assistant_generation_duration_seconds",
            "Model generation latency",
            ["endpoint", "stream"],
            buckets=_LATENCY_BUCKETS,
            registry=r,
        )
        self.tokens_per_second = Histogram(
            "assistant_generation_tokens_per_second",
            "Completion tokens per second per generation",
            buckets=_TPS_BUCKETS,
            registry=r,
        )
        self.prompt_tokens = Counter(
            "assistant_prompt_tokens_total", "Prompt tokens processed", registry=r
        )
        self.completion_tokens = Counter(
            "assistant_completion_tokens_total", "Completion tokens generated", registry=r
        )
        self.finish_reasons = Counter(
            "assistant_generation_finish_total",
            "Generations by finish reason",
            ["reason"],
            registry=r,
        )
        self.generation_errors = Counter(
            "assistant_generation_errors_total", "Generations that raised", registry=r
        )
        self.queue_depth = Gauge(
            "assistant_queue_depth", "Requests waiting for a generation slot", registry=r
        )
        self.inflight = Gauge(
            "assistant_inflight_generations", "Generations currently running", registry=r
        )
        self.rejections = Counter(
            "assistant_admission_rejections_total",
            "Requests rejected by admission control",
            ["reason"],
            registry=r,
        )
        self.model_loaded = Gauge(
            "assistant_model_loaded", "1 when the model is loaded and serving", registry=r
        )
        self.retrieval_seconds = Histogram(
            "assistant_retrieval_duration_seconds",
            "Retrieval (embed + search) latency",
            buckets=_LATENCY_BUCKETS,
            registry=r,
        )

    def render(self) -> tuple[bytes, str]:
        return generate_latest(self.registry), CONTENT_TYPE_LATEST
