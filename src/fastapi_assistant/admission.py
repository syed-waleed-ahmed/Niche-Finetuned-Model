"""Admission control: bounded concurrency plus a bounded wait queue.

A replica can only run ``max_concurrency`` generations at once (one, for a CPU
model). Without a bound, extra requests pile up behind the engine lock until the
client or load balancer times out, which hides overload and wastes work. Instead:

* up to ``max_concurrency`` requests run;
* up to ``max_queue`` more wait, each for at most ``queue_timeout_s``;
* anything beyond that is rejected immediately with 429 and a ``Retry-After``
  estimate, so the edge can fail over to another replica or the client can back off.
"""

from __future__ import annotations

import asyncio
import math
import time


class AdmissionRejected(Exception):
    """Base class for requests the controller refused to run."""

    def __init__(self, message: str, retry_after_s: int) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


class QueueFullError(AdmissionRejected):
    """The wait queue is full; retry later or elsewhere (HTTP 429)."""


class QueueTimeoutError(AdmissionRejected):
    """The request waited too long for a free slot (HTTP 503)."""


class Slot:
    """A held execution slot. ``release`` is idempotent."""

    def __init__(self, controller: AdmissionController) -> None:
        self._controller = controller
        self._started = time.monotonic()
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._controller._release(time.monotonic() - self._started)


class AdmissionController:
    def __init__(self, max_concurrency: int, max_queue: int, queue_timeout_s: float) -> None:
        self.max_concurrency = max_concurrency
        self.max_queue = max_queue
        self.queue_timeout_s = queue_timeout_s
        self._sem = asyncio.Semaphore(max_concurrency)
        self._waiting = 0
        self._inflight = 0
        # Exponentially weighted mean service time, used for Retry-After hints.
        self._avg_service_s = 10.0

    @property
    def waiting(self) -> int:
        return self._waiting

    @property
    def inflight(self) -> int:
        return self._inflight

    def retry_after(self) -> int:
        """Rough seconds until a new request could start."""
        backlog = (self._waiting + self._inflight) / self.max_concurrency
        return max(1, math.ceil(backlog * self._avg_service_s))

    async def acquire(self) -> Slot:
        if self._sem.locked() and self._waiting >= self.max_queue:
            raise QueueFullError("Server is at capacity", self.retry_after())
        self._waiting += 1
        try:
            await asyncio.wait_for(self._sem.acquire(), timeout=self.queue_timeout_s)
        except asyncio.TimeoutError as exc:
            raise QueueTimeoutError("Timed out waiting for capacity", self.retry_after()) from exc
        finally:
            self._waiting -= 1
        self._inflight += 1
        return Slot(self)

    def _release(self, service_s: float) -> None:
        self._inflight -= 1
        self._avg_service_s = 0.8 * self._avg_service_s + 0.2 * service_s
        self._sem.release()
