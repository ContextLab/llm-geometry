"""Precompute job registry: progress tracking + single-flight dedup.

A job tracks the status/progress of one precompute, keyed by the cache key it
produces. ``get_or_create`` returns the existing active job for a key instead of
starting a second one, so concurrent identical requests share one computation
(FR-008). State is thread-safe (computations run in worker threads) and pollable, so
the SSE endpoint can stream progress without cross-thread plumbing (FR-009).
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class Job:
    job_id: str
    cache_key: str
    status: str = "queued"  # queued | running | done | error
    progress: float = 0.0
    message: str = ""
    error: Optional[dict[str, Any]] = None
    version: int = 0  # bumps on every change; lets SSE detect updates
    # Feature 002 (additive, backward-compatible — contracts/api.md): an optional
    # phase label carried on SSE progress events ("train" | "finetune" | …) and an
    # optional result payload merged into the terminal `done` event data (e.g.
    # {"checkpoint_id": …} for /api/geo/train).
    phase: Optional[str] = None
    result: Optional[dict[str, Any]] = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "cache_key": self.cache_key,
            "status": self.status,
            "progress": round(self.progress, 4),
            "message": self.message,
            "error": self.error,
            "version": self.version,
            "phase": self.phase,
            "result": self.result,
        }


class JobRegistry:
    def __init__(self) -> None:
        self._by_id: dict[str, Job] = {}
        self._active_by_key: dict[str, str] = {}  # cache_key -> active job_id
        self._lock = threading.Lock()

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._by_id.get(job_id)

    def get_or_create(self, cache_key: str, phase: str | None = None) -> tuple[Job, bool]:
        """Return ``(job, created)``. If an active job already exists for this cache
        key, return it with ``created=False`` (single-flight). ``phase`` labels a
        newly created job's SSE progress events (feature 002; optional)."""
        with self._lock:
            existing_id = self._active_by_key.get(cache_key)
            if existing_id is not None:
                job = self._by_id.get(existing_id)
                if job is not None and job.status in ("queued", "running"):
                    return job, False
            job = Job(job_id=uuid.uuid4().hex, cache_key=cache_key, phase=phase)
            self._by_id[job.job_id] = job
            self._active_by_key[cache_key] = job.job_id
            return job, True

    def update(
        self, job_id: str, progress: float | None = None, message: str | None = None
    ) -> None:
        with self._lock:
            job = self._by_id.get(job_id)
            if job is None:
                return
            job.status = "running"
            if progress is not None:
                job.progress = max(0.0, min(1.0, float(progress)))
            if message is not None:
                job.message = message
            job.version += 1

    def finish(self, job_id: str, result: dict[str, Any] | None = None) -> None:
        """Mark done; ``result`` (optional) is merged into the SSE ``done`` event data
        (feature 002 — e.g. the minted checkpoint_id / weights_token)."""
        with self._lock:
            job = self._by_id.get(job_id)
            if job is None:
                return
            job.status = "done"
            job.progress = 1.0
            if result is not None:
                job.result = dict(result)
            job.version += 1
            if self._active_by_key.get(job.cache_key) == job_id:
                del self._active_by_key[job.cache_key]

    def fail(self, job_id: str, error: dict[str, Any]) -> None:
        with self._lock:
            job = self._by_id.get(job_id)
            if job is None:
                return
            job.status = "error"
            job.error = error
            job.version += 1
            if self._active_by_key.get(job.cache_key) == job_id:
                del self._active_by_key[job.cache_key]


# Process-wide registry used by the API and precompute orchestrator.
registry = JobRegistry()
