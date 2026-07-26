"""Shared HTTP routes: health, the curated model catalog, and the job/SSE plumbing.

Feature 004 removed the three embedding-geometry views (vector field, Sankey,
manifold) and every endpoint that existed only to serve them. What remains here is
the machinery BOTH explorer tabs still depend on:

* ``/api/health``                — schema version handshake
* ``/api/models``, ``/api/models/resolve`` — the Architecture Explorer's curated menu
  and its pre-flight capability check (a rejected model never becomes active)
* ``/api/jobs/{id}``, ``/api/jobs/{id}/events`` — job status + Server-Sent progress,
  used by the Geometry Lab's training and fine-tuning runs

The per-artifact compute endpoints and the precompute pipeline they drove are gone;
``/api/geo/*`` and ``/api/arch/*`` own their own compute paths.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from ..config import SCHEMA_VERSION
from ..errors import NotFoundError
from ..jobs.registry import registry
from ..models.loader import load_model, resolve_model
from ..models.registry import curated_models
from . import progress as progress_mod

router = APIRouter()


class ResolveBody(BaseModel):
    model_id: str


@router.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "schema_version": SCHEMA_VERSION}


@router.get("/models")
def list_models() -> dict[str, Any]:
    models: list[dict[str, Any]] = []
    for entry in curated_models():
        try:
            models.append(resolve_model(entry["model_id"]))
        except Exception as exc:  # keep listing even if one model can't be resolved
            models.append({**entry, "status": "unsupported", "reason": str(exc)})
    return {"models": models}


@router.post("/models/resolve")
def resolve(body: ResolveBody) -> dict[str, Any]:
    # raises UnsupportedModelError -> 422 envelope; never a fallback (FR-003)
    return resolve_model(body.model_id)


@router.get("/jobs/{job_id}")
def job_status(job_id: str) -> dict[str, Any]:
    job = registry.get(job_id)
    if job is None:
        raise NotFoundError(f"job {job_id} not found")
    return job.snapshot()


@router.get("/jobs/{job_id}/events")
async def job_events(job_id: str):
    return await progress_mod.job_event_response(job_id)


@router.get("/tokenize")
def tokenize_route(model_id: str, text: str = "") -> dict[str, Any]:
    """Token ids + decoded strings — the Architecture Explorer's token strip.

    The static build answers this from the model's real tokenizer files via
    transformers.js, so both runtimes show the same segmentation for a prompt even
    when only one of them can trace it.
    """
    lm = load_model(model_id)
    ids = list(lm.tokenizer(text)["input_ids"]) if text else []
    return {
        "model_id": lm.model_id,
        "tokens": [{"token": int(t), "token_str": lm.tokenizer.decode([int(t)])} for t in ids],
    }
