"""Run the HTTP service with the configured ASGI server.

Two servers are supported (``HTTP_SERVER``):

* ``uvicorn`` (the default): HTTP/1.1 only. Fine for local use and for hosts that
  pass a client's disconnect on as a closed connection.
* ``hypercorn``: HTTP/1.1 and cleartext HTTP/2 (h2c) on the same port. Cloud Run needs
  it. Over HTTP/1.1, Cloud Run does not tell the container that a client has gone
  away, so an abandoned generation would run to its end and keep the generation slot
  busy. Over HTTP/2 (``ports: [{name: h2c}]`` in the service spec) the front end
  resets the request's stream, the app receives ``http.disconnect``, and the
  generation is cancelled. Probes and tools that speak HTTP/1.1 keep working.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from .config import Settings

if TYPE_CHECKING:
    from hypercorn.config import Config

# How long in-flight requests get to finish after SIGTERM. Cloud Run sends SIGTERM and
# kills the container 10 seconds later, so a longer value only matters elsewhere.
GRACEFUL_SHUTDOWN_S = 30
# How long an idle connection is kept. The proxy in front (Cloud Run's front end, or a
# load balancer) reuses connections, so it should be the side that closes an idle one:
# a server that closes first can do so just as the proxy sends the next request.
KEEP_ALIVE_S = 620


def hypercorn_config(settings: Settings) -> Config:
    """The Hypercorn configuration used in production (and by the HTTP/2 tests)."""
    from hypercorn.config import Config

    config = Config()
    config.bind = [f"{settings.api_host}:{settings.api_port}"]
    config.graceful_timeout = GRACEFUL_SHUTDOWN_S
    config.keep_alive_timeout = KEEP_ALIVE_S
    # The app writes its own structured access log (see logging_config). Server errors
    # go through the root logger, so they share its format (JSON in the image).
    config.accesslog = None
    config.errorlog = logging.getLogger("hypercorn.error")
    config.include_server_header = False
    return config


async def serve_hypercorn(
    app: Any,
    config: Config,
    shutdown_trigger: Callable[..., Awaitable[None]] | None = None,
) -> None:
    """Serve ``app`` until ``shutdown_trigger`` returns (or SIGINT/SIGTERM without one)."""
    from hypercorn.asyncio import serve

    await serve(app, config, shutdown_trigger=shutdown_trigger)


def run(settings: Settings) -> None:
    """Start the configured server and block until it stops."""
    if settings.http_server == "hypercorn":
        from .api import create_app

        asyncio.run(serve_hypercorn(create_app(), hypercorn_config(settings)))
        return

    import uvicorn

    uvicorn.run(
        "fastapi_assistant.api:create_app",
        factory=True,
        host=settings.api_host,
        port=settings.api_port,
        log_config=None,
        proxy_headers=True,
        forwarded_allow_ips="*",
        timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_S,
    )
