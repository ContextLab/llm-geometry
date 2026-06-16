"""HTTP routes implementing contracts/api.md.

Read endpoints get-or-compute synchronously (so a single curl returns data and a
cache hit is fast); ``/api/precompute`` + the SSE stream give the non-blocking,
progress-driven path the frontend uses. Both share one cache and one single-flight
lock per key.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from fastapi import APIRouter, Response
from pydantic import BaseModel, Field

from ..config import (
    DEFAULT_2D_METHOD,
    DEFAULT_3D_METHOD,
    DEFAULT_GRID_N,
    DEFAULT_SEED,
    SCHEMA_VERSION,
)
from ..errors import NotFoundError
from ..jobs.registry import registry
from ..compute.context import tokenize_strings
from ..models.loader import load_model, resolve_model
from ..models.registry import curated_models
from ..precompute import get_or_compute_sync, request_precompute
from . import progress as progress_mod

router = APIRouter()


def _jsonable(value: Any) -> Any:
    """Coerce numpy types to plain JSON-serializable Python values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


class ResolveBody(BaseModel):
    model_id: str


class PrecomputeBody(BaseModel):
    artifact_type: str
    model_id: str
    params: dict[str, Any] = Field(default_factory=dict)
    inputs: dict[str, Any] = Field(default_factory=dict)


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


@router.post("/precompute")
def precompute(body: PrecomputeBody, response: Response) -> dict[str, Any]:
    result = request_precompute(body.artifact_type, body.model_id, body.params, body.inputs)
    response.status_code = 200 if result["ready"] else 202
    return result


@router.get("/jobs/{job_id}")
def job_status(job_id: str) -> dict[str, Any]:
    job = registry.get(job_id)
    if job is None:
        raise NotFoundError(f"job {job_id} not found")
    return job.snapshot()


@router.get("/jobs/{job_id}/events")
async def job_events(job_id: str):
    return await progress_mod.job_event_response(job_id)


@router.get("/distribution")
def distribution(
    model_id: str,
    prefix_text: str = "",
    temperature: float = 1.0,
    top_k: int | None = None,
    response_text: str = "",
    response_step: int = 0,
) -> dict[str, Any]:
    payload = get_or_compute_sync(
        "distribution",
        model_id,
        {"temperature": temperature, "top_k": top_k},
        {"prefix_text": prefix_text, "response_text": response_text, "response_step": response_step},
    )
    meta = dict(payload["meta"])
    resp: dict[str, Any] = {**meta}
    if not top_k:
        resp["probs"] = payload["arrays"]["probs"].tolist()
    return _jsonable(resp)


@router.get("/embeddings")
def embeddings(
    model_id: str,
    layer: int = 0,
    source: str = "static",
    format: str = "meta",
    reference_set_size: int | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {"layer": layer, "source": source}
    if reference_set_size is not None:
        params["reference_set_size"] = reference_set_size
    payload = get_or_compute_sync("embeddings", model_id, params)
    meta = dict(payload["meta"])
    resp: dict[str, Any] = {**meta, "token_ids": payload["arrays"]["token_ids"].tolist()}
    if format == "full":
        resp["vectors"] = payload["arrays"]["vectors"].tolist()
    return _jsonable(resp)


@router.get("/reduction/2d")
def reduction_2d(
    model_id: str,
    method: str = DEFAULT_2D_METHOD,
    seed: int = DEFAULT_SEED,
    grid_n: int = DEFAULT_GRID_N,
    with_grid: bool = False,
    source: str = "static",
    layer: int = 0,
    reference_set_size: int | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {"method": method, "seed": seed, "grid_n": grid_n,
                              "with_grid": with_grid, "source": source, "layer": layer}
    if reference_set_size is not None:
        params["reference_set_size"] = reference_set_size
    payload = get_or_compute_sync("reduction_2d", model_id, params)
    meta = dict(payload["meta"])
    arrays = payload["arrays"]
    resp: dict[str, Any] = {
        **meta,
        "coords": arrays["coords"].tolist(),
        "token_ids": arrays["token_ids"].tolist(),
    }
    if "grid_vertices" in arrays:
        resp["grid"] = {
            "n": meta.get("grid_n"),
            "vertices": arrays["grid_vertices"].tolist(),
            "reference_token_ids": arrays["grid_reference_token_ids"].tolist(),
        }
    return _jsonable(resp)


@router.get("/reduction/3d")
def reduction_3d(
    model_id: str,
    method: str = DEFAULT_3D_METHOD,
    seed: int = DEFAULT_SEED,
    source: str = "static",
    layer: int = 0,
    reference_set_size: int | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {"method": method, "seed": seed, "source": source, "layer": layer}
    if reference_set_size is not None:
        params["reference_set_size"] = reference_set_size
    payload = get_or_compute_sync("reduction_3d", model_id, params)
    meta = dict(payload["meta"])
    arrays = payload["arrays"]
    return _jsonable({
        **meta,
        "coords": arrays["coords"].tolist(),
        "token_ids": arrays["token_ids"].tolist(),
    })


@router.get("/token_cloud")
def token_cloud_route(
    model_id: str,
    seed: int = DEFAULT_SEED,
    spread_mu: float = 0.65,
) -> dict[str, Any]:
    """Full-vocabulary 2D cloud (a dot per token) — the shared layout the vector field's
    arrows sit in. Cached once per model; the browser fetches it once and reuses it."""
    payload = get_or_compute_sync("token_cloud", model_id, {"seed": seed, "spread_mu": spread_mu})
    meta = dict(payload["meta"])
    a = payload["arrays"]
    return _jsonable({
        **meta,  # includes token_strs (printable decoded strings, aligned with token_ids)
        "coords": a["warped"].tolist(),
        "token_ids": a["token_ids"].tolist(),
    })


@router.get("/vector_field")
def vector_field_route(
    model_id: str,
    prefix_text: str = "",
    temperature: float = 1.0,
    layer_from: int = 0,
    layer_to: int | None = None,
    grid_n: int = DEFAULT_GRID_N,
    fanout: int = 4,
    seed: int = DEFAULT_SEED,
    reference_set_size: int | None = None,
    response_text: str = "",
    response_step: int = 0,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "temperature": temperature, "layer_from": layer_from,
        "layer_to": layer_from if layer_to is None else layer_to,
        "grid_n": grid_n, "fanout": fanout, "seed": seed,
    }
    if reference_set_size is not None:
        params["reference_set_size"] = reference_set_size
    payload = get_or_compute_sync(
        "vector_field", model_id, params,
        {"prefix_text": prefix_text, "response_text": response_text, "response_step": response_step},
    )
    meta = dict(payload["meta"])
    a = payload["arrays"]
    resp: dict[str, Any] = {
        **meta,
        "starts": a["starts"].tolist(), "ends": a["ends"].tolist(), "probs": a["probs"].tolist(),
        "start_tokens": a["start_tokens"].tolist(), "end_tokens": a["end_tokens"].tolist(),
    }
    if "trajectory" in a:
        resp["trajectory"] = a["trajectory"].tolist()
        resp["trajectory_probs"] = a["trajectory_probs"].tolist()
    return _jsonable(resp)


@router.get("/sankey")
def sankey_route(
    model_id: str,
    prefix_text: str = "",
    temperature: float = 1.0,
    n_particles: int = 24,
    n_steps: int = 8,
    seed: int = DEFAULT_SEED,
    response_text: str = "",
    response_step: int = 0,
) -> dict[str, Any]:
    payload = get_or_compute_sync(
        "sankey", model_id,
        {"temperature": temperature, "n_particles": n_particles, "n_steps": n_steps, "seed": seed},
        {"prefix_text": prefix_text, "response_text": response_text, "response_step": response_step},
    )
    return _jsonable(dict(payload["meta"]))  # nodes / links / token_strs live in meta


@router.get("/manifold")
def manifold_route(
    model_id: str,
    prefix_text: str = "",
    temperature: float = 1.0,
    seed: int = DEFAULT_SEED,
    reference_set_size: int | None = None,
    response_text: str = "",
    response_step: int = 0,
) -> dict[str, Any]:
    params: dict[str, Any] = {"temperature": temperature, "seed": seed}
    if reference_set_size is not None:
        params["reference_set_size"] = reference_set_size
    payload = get_or_compute_sync(
        "manifold", model_id, params,
        {"prefix_text": prefix_text, "response_text": response_text, "response_step": response_step},
    )
    meta = dict(payload["meta"])
    a = payload["arrays"]
    return _jsonable({
        **meta,
        "vertices": a["vertices"].tolist(), "faces": a["faces"].tolist(), "warp": a["warp"].tolist(),
        "token_points": a["token_points"].tolist(), "token_emis": a["token_emis"].tolist(),
        "token_ids": a["token_ids"].tolist(),
        "traj_points": a["traj_points"].tolist(),  # response trajectory on the radius-2 sphere
    })


@router.get("/tokenize")
def tokenize_route(model_id: str, text: str = "") -> dict[str, Any]:
    """Token ids + strings for a text — lets the UI animate over response tokens."""
    lm = load_model(model_id)
    return {"model_id": lm.model_id, "tokens": tokenize_strings(lm, text)}
