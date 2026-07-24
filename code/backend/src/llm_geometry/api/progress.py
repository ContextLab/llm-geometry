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
                    # Feature 002 (additive): a job's result payload (e.g. the minted
                    # checkpoint_id / weights_token) rides along on the done event.
                    done_data: dict = {"cache_key": job.cache_key}
                    if job.result:
                        # Same numeric encoding as the HTTP path (6 significant digits)
                        # so the two delivery routes agree on e.g. loss_before.
                        for k, v in job.result.items():
                            done_data[k] = float(f"{v:.6g}") if isinstance(v, float) else v
                    yield {"event": "done", "data": json.dumps(done_data)}
                    return
                if job.status == "error":
                    yield {
                        "event": "error",
                        "data": json.dumps(
                            job.error or {"type": "ComputeError", "message": "failed"}
                        ),
                    }
                    return
                progress_data: dict = {"progress": job.progress, "message": job.message}
                if job.phase:  # optional phase label (feature 002, contracts/api.md)
                    progress_data["phase"] = job.phase
                yield {
                    "event": "progress",
                    "data": json.dumps(progress_data),
                }
            await asyncio.sleep(_POLL_SECONDS)

    return EventSourceResponse(event_generator())
