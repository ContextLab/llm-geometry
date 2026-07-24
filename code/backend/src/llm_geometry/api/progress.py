"""Server-Sent Events progress stream for precompute jobs.

Polls the job registry ~4×/second and emits ``progress`` events on change, then a
terminal ``done`` or ``error`` event. Comfortably exceeds the ≥1 Hz progress budget
(FR-009, SC-003) without any cross-thread plumbing.
"""

from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator

from sse_starlette.sse import EventSourceResponse

from ..errors import NotFoundError
from ..jobs.registry import registry

_POLL_SECONDS = 0.25


async def job_event_response(job_id: str) -> EventSourceResponse:
    if registry.get(job_id) is None:
        raise NotFoundError(f"job {job_id} not found")

    async def event_generator() -> AsyncIterator[dict]:
        last_version = -1
        while True:
            job = registry.get(job_id)
            if job is None:
                yield {
                    "event": "error",
                    "data": json.dumps(
                        {"type": "NotFoundError", "message": "job no longer exists"}
                    ),
                }
                return
            if job.version != last_version:
                last_version = job.version
                if job.status == "done":
                    yield {"event": "done", "data": json.dumps({"cache_key": job.cache_key})}
                    return
                if job.status == "error":
                    yield {
                        "event": "error",
                        "data": json.dumps(
                            job.error or {"type": "ComputeError", "message": "failed"}
                        ),
                    }
                    return
                yield {
                    "event": "progress",
                    "data": json.dumps({"progress": job.progress, "message": job.message}),
                }
            await asyncio.sleep(_POLL_SECONDS)

    return EventSourceResponse(event_generator())
