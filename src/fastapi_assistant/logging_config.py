"""Logging configuration.

Provides plain human-readable logs for local development and structured JSON logs
for production log aggregators (set ``LOG_JSON=true``). Request-scoped fields such
as ``request_id`` are attached via ``logging`` ``extra=`` and surfaced by the JSON
formatter.
"""

from __future__ import annotations

import json
import logging
import sys
import time

_EXTRA_FIELDS = ("request_id", "method", "path", "status_code", "duration_ms")


class JsonFormatter(logging.Formatter):
    """Render each log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003 - stdlib name
        payload: dict[str, object] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for field in _EXTRA_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str = "INFO", json_logs: bool = False) -> None:
    """Configure the root logger. Safe to call multiple times (idempotent handlers)."""
    root = logging.getLogger()
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    if json_logs:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-8s %(name)s :: %(message)s")
        )
    root.addHandler(handler)
    root.setLevel(level.upper())

    # Let our access logging replace uvicorn's default access logs.
    logging.getLogger("uvicorn.access").propagate = False
