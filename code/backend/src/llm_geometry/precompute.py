"""Precompute orchestrator.

Maps an ``(artifact_type, model_id, params, inputs)`` request to a deterministic
cache key + a compute closure, then either returns the cached artifact or computes it
once (single-flight) and stores it. Reductions transparently obtain their source
embeddings through the same cached path, so the whole dependency chain is cached and
regenerable (FR-004..FR-008). A non-blocking variant runs the work in a thread and
reports progress through the job registry (FR-009).
"""

from __future__ import annotations

import threading
from typing import Any, Callable


from .cache.keys import make_cache_key
from .cache.store import CacheStore
from .config import (
    DEFAULT_2D_METHOD,
    DEFAULT_3D_METHOD,
    DEFAULT_GRID_N,
    DEFAULT_RBF_WIDTH,
    DEFAULT_REFERENCE_SET_SIZE,
    DEFAULT_SEED,
)
from .errors import ComputeError, InvalidParamError, LLMGeometryError
from .jobs.registry import registry
from .models.loader import resolve_model

ProgressCb = Callable[[float, str], None]

ARTIFACT_TYPES = (
    "distribution",
    "embeddings",
    "reduction_2d",
    "reduction_3d",
    "token_cloud",
    "vector_field",
    "vector_field_animation",
    "sankey",
    "manifold",
    "manifold_animation",
)

_store = CacheStore()
_key_locks: dict[str, threading.Lock] = {}
_key_locks_guard = threading.Lock()


def _lock_for(key: str) -> threading.Lock:
    with _key_locks_guard:
        lock = _key_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _key_locks[key] = lock
        return lock


def _subprogress(cb: ProgressCb | None, lo: float, hi: float, label: str) -> ProgressCb | None:
    if cb is None:
        return None

    def inner(p: float, m: str) -> None:
        cb(lo + (hi - lo) * p, f"{label}: {m}")

    return inner


def _plan(
    artifact_type: str,
    model_id: str,
    params: dict[str, Any] | None,
    inputs: dict[str, Any] | None,
) -> tuple[str, dict[str, Any], Callable[[ProgressCb | None], dict[str, Any]]]:
    """Return ``(cache_key, spec, compute)`` for a request. Validates the model
    (raising UnsupportedModelError) and pins its revision into the key."""
    ref = resolve_model(model_id)  # raises UnsupportedModelError if bad; pins revision
    mid, revision = ref["model_id"], ref["revision"]
    params = dict(params or {})
    inputs = dict(inputs or {})

    if artifact_type == "distribution":
        temperature = float(params.get("temperature", 1.0))
        if temperature < 0:
            raise InvalidParamError(f"temperature must be >= 0, got {temperature}")
        top_k = params.get("top_k")
        top_k = int(top_k) if top_k else None
        prefix_text = inputs.get("prefix_text", "") or ""
        response_text = inputs.get("response_text", "") or ""
        response_step = int(inputs.get("response_step", 0) or 0)
        norm_params = {"temperature": temperature, "top_k": top_k}
        norm_inputs = {
            "prefix_text": prefix_text,
            "response_text": response_text,
            "response_step": response_step,
        }
        key, spec = make_cache_key(
            model_id=mid,
            revision=revision,
            artifact_type="distribution",
            inputs=norm_inputs,
            params=norm_params,
        )

        def compute(cb: ProgressCb | None) -> dict[str, Any]:
            from .compute.distributions import next_token_distribution

            if cb:
                cb(0.1, "running forward pass")
            res = next_token_distribution(
                mid,
                prefix_text=prefix_text,
                temperature=temperature,
                top_k=top_k,
                response_text=response_text,
                response_step=response_step,
            )
            if cb:
                cb(1.0, "done")
            return res

        return key, spec, compute

    if artifact_type == "embeddings":
        source = params.get("source", "static")
        layer = int(params.get("layer", 0)) if source == "contextual" else 0
        rss = params.get("reference_set_size", DEFAULT_REFERENCE_SET_SIZE)
        rss = int(rss) if rss is not None else None
        norm_params = {"layer": layer, "source": source, "reference_set_size": rss}
        key, spec = make_cache_key(
            model_id=mid,
            revision=revision,
            artifact_type="embeddings",
            params=norm_params,
        )

        def compute(cb: ProgressCb | None) -> dict[str, Any]:
            from .compute.embeddings import per_layer_embeddings

            return per_layer_embeddings(
                mid, layer=layer, source=source, reference_set_size=rss, progress_cb=cb
            )

        return key, spec, compute

    if artifact_type in ("reduction_2d", "reduction_3d"):
        source = params.get("source", "static")
        layer = int(params.get("layer", 0)) if source == "contextual" else 0
        rss = params.get("reference_set_size", DEFAULT_REFERENCE_SET_SIZE)
        rss = int(rss) if rss is not None else None
        seed = int(params.get("seed", DEFAULT_SEED))
        emb_params = {"source": source, "layer": layer, "reference_set_size": rss}

        if artifact_type == "reduction_2d":
            method = params.get("method", DEFAULT_2D_METHOD)
            grid_n = int(params.get("grid_n", DEFAULT_GRID_N))
            with_grid = bool(params.get("with_grid", False))
            norm_params = {
                "method": method,
                "seed": seed,
                "with_grid": with_grid,
                "grid_n": grid_n if with_grid else 0,
                **emb_params,
            }
            key, spec = make_cache_key(
                model_id=mid,
                revision=revision,
                artifact_type="reduction_2d",
                params=norm_params,
            )

            def compute(cb: ProgressCb | None) -> dict[str, Any]:
                from .reduce.grid import build_grid
                from .reduce.twod import reduce_2d

                emb = get_or_compute_sync(
                    "embeddings", mid, emb_params, None, _subprogress(cb, 0.0, 0.6, "embeddings")
                )
                vectors = emb["arrays"]["vectors"]
                token_ids = emb["arrays"]["token_ids"]
                if cb:
                    cb(0.7, f"reducing to 2D ({method})")
                coords = reduce_2d(vectors, method=method, seed=seed)
                meta = {
                    "model_id": mid,
                    "revision": revision,
                    "method": method,
                    "seed": seed,
                    "dims": "2d",
                    "count": int(coords.shape[0]),
                }
                arrays: dict[str, Any] = {"coords": coords, "token_ids": token_ids}
                if with_grid:
                    if cb:
                        cb(0.9, "building reference grid")
                    vertices, ref_ids = build_grid(coords, token_ids, n=grid_n)
                    arrays["grid_vertices"] = vertices
                    arrays["grid_reference_token_ids"] = ref_ids
                    meta["grid_n"] = grid_n
                if cb:
                    cb(1.0, "done")
                return {"meta": meta, "arrays": arrays}

            return key, spec, compute

        # reduction_3d
        method = params.get("method", DEFAULT_3D_METHOD)
        norm_params = {"method": method, "seed": seed, **emb_params}
        key, spec = make_cache_key(
            model_id=mid,
            revision=revision,
            artifact_type="reduction_3d",
            params=norm_params,
        )

        def compute(cb: ProgressCb | None) -> dict[str, Any]:
            from .reduce.sphere import reduce_3d_sphere

            emb = get_or_compute_sync(
                "embeddings", mid, emb_params, None, _subprogress(cb, 0.0, 0.6, "embeddings")
            )
            vectors = emb["arrays"]["vectors"]
            token_ids = emb["arrays"]["token_ids"]
            if cb:
                cb(0.7, f"reducing to 3D sphere ({method})")
            coords = reduce_3d_sphere(vectors, method=method, seed=seed)
            meta = {
                "model_id": mid,
                "revision": revision,
                "method": method,
                "seed": seed,
                "dims": "3d_sphere",
                "count": int(coords.shape[0]),
            }
            if cb:
                cb(1.0, "done")
            return {"meta": meta, "arrays": {"coords": coords, "token_ids": token_ids}}

        return key, spec, compute

    if artifact_type == "token_cloud":
        seed = int(params.get("seed", DEFAULT_SEED))
        spread_mu = float(params.get("spread_mu", 0.65))
        norm_params = {"seed": seed, "spread_mu": spread_mu}
        key, spec = make_cache_key(
            model_id=mid,
            revision=revision,
            artifact_type="token_cloud",
            params=norm_params,
        )

        def compute(cb: ProgressCb | None) -> dict[str, Any]:
            from .compute.token_cloud import token_cloud

            return token_cloud(mid, seed=seed, spread_mu=spread_mu, progress_cb=cb)

        return key, spec, compute

    if artifact_type == "vector_field":
        temperature = float(params.get("temperature", 1.0))
        if temperature < 0:
            raise InvalidParamError(f"temperature must be >= 0, got {temperature}")
        layer_from = int(params.get("layer_from", params.get("layer", 0)))
        layer_to = int(params.get("layer_to", layer_from))
        grid_n = int(params.get("grid_n", DEFAULT_GRID_N))
        fanout = int(params.get("fanout", 4))
        spread_mu = float(params.get("spread_mu", 0.65))
        rss = params.get("reference_set_size", DEFAULT_REFERENCE_SET_SIZE)
        rss = int(rss) if rss is not None else None
        seed = int(params.get("seed", DEFAULT_SEED))
        prefix_text = inputs.get("prefix_text", "") or ""
        response_text = inputs.get("response_text", "") or ""
        response_step = int(inputs.get("response_step", 0) or 0)
        norm_params = {
            "temperature": temperature,
            "layer_from": layer_from,
            "layer_to": layer_to,
            "grid_n": grid_n,
            "fanout": fanout,
            "spread_mu": spread_mu,
            "reference_set_size": rss,
            "seed": seed,
        }
        key, spec = make_cache_key(
            model_id=mid,
            revision=revision,
            artifact_type="vector_field",
            inputs={
                "prefix_text": prefix_text,
                "response_text": response_text,
                "response_step": response_step,
            },
            params=norm_params,
        )

        def compute(cb: ProgressCb | None) -> dict[str, Any]:
            from .compute.vector_field import vector_field

            return vector_field(
                mid,
                prefix_text=prefix_text,
                temperature=temperature,
                layer_from=layer_from,
                layer_to=layer_to,
                grid_n=grid_n,
                reference_set_size=rss,
                seed=seed,
                fanout=fanout,
                spread_mu=spread_mu,
                response_text=response_text,
                response_step=response_step,
                progress_cb=cb,
            )

        return key, spec, compute

    if artifact_type == "vector_field_animation":
        temperature = float(params.get("temperature", 1.0))
        if temperature < 0:
            raise InvalidParamError(f"temperature must be >= 0, got {temperature}")
        layer_to = int(params.get("layer_to", 0))
        grid_n = int(params.get("grid_n", DEFAULT_GRID_N))
        rss = params.get("reference_set_size", 576)
        rss = int(rss) if rss is not None else None
        seed = int(params.get("seed", DEFAULT_SEED))
        prefix_text = inputs.get("prefix_text", "") or ""
        response_text = inputs.get("response_text", "") or ""
        norm_params = {
            "temperature": temperature,
            "layer_to": layer_to,
            "grid_n": grid_n,
            "reference_set_size": rss,
            "seed": seed,
        }
        key, spec = make_cache_key(
            model_id=mid,
            revision=revision,
            artifact_type="vector_field_animation",
            inputs={"prefix_text": prefix_text, "response_text": response_text},
            params=norm_params,
        )

        def compute(cb: ProgressCb | None) -> dict[str, Any]:
            from .compute.vector_field import vector_field_animation

            return vector_field_animation(
                mid,
                prefix_text=prefix_text,
                temperature=temperature,
                layer_to=layer_to,
                reference_set_size=rss,
                seed=seed,
                grid_n=grid_n,
                response_text=response_text,
                progress_cb=cb,
            )

        return key, spec, compute

    if artifact_type == "sankey":
        temperature = float(params.get("temperature", 1.0))
        if temperature < 0:
            raise InvalidParamError(f"temperature must be >= 0, got {temperature}")
        n_particles = int(params.get("n_particles", 800))
        n_steps = int(params.get("n_steps", 8))
        seed = int(params.get("seed", DEFAULT_SEED))
        prefix_text = inputs.get("prefix_text", "") or ""
        # The swarm is prompt-conditioned and RESPONSE-INDEPENDENT, so the response is NOT part of
        # the cache key — editing it reuses this cached swarm and only refetches the cheap overlay.
        norm_params = {
            "temperature": temperature,
            "n_particles": n_particles,
            "n_steps": n_steps,
            "seed": seed,
        }
        key, spec = make_cache_key(
            model_id=mid,
            revision=revision,
            artifact_type="sankey",
            inputs={"prefix_text": prefix_text},
            params=norm_params,
        )

        def compute(cb: ProgressCb | None) -> dict[str, Any]:
            from .compute.sankey import sankey

            return sankey(
                mid,
                prefix_text=prefix_text,
                temperature=temperature,
                n_particles=n_particles,
                n_steps=n_steps,
                seed=seed,
                progress_cb=cb,
            )

        return key, spec, compute

    if artifact_type == "manifold":
        temperature = float(params.get("temperature", 1.0))
        if temperature < 0:
            raise InvalidParamError(f"temperature must be >= 0, got {temperature}")
        rss = params.get("reference_set_size", None)  # None = every vocab token
        rss = int(rss) if rss is not None else None
        seed = int(params.get("seed", DEFAULT_SEED))
        width = float(params.get("width", DEFAULT_RBF_WIDTH))
        prefix_text = inputs.get("prefix_text", "") or ""
        response_text = inputs.get("response_text", "") or ""
        response_step = int(inputs.get("response_step", 0) or 0)
        norm_params = {
            "temperature": temperature,
            "reference_set_size": rss,
            "seed": seed,
            "width": width,
        }
        key, spec = make_cache_key(
            model_id=mid,
            revision=revision,
            artifact_type="manifold",
            inputs={
                "prefix_text": prefix_text,
                "response_text": response_text,
                "response_step": response_step,
            },
            params=norm_params,
        )

        def compute(cb: ProgressCb | None) -> dict[str, Any]:
            from .compute.manifold import manifold

            return manifold(
                mid,
                prefix_text=prefix_text,
                temperature=temperature,
                reference_set_size=rss,
                seed=seed,
                width=width,
                response_text=response_text,
                response_step=response_step,
                progress_cb=cb,
            )

        return key, spec, compute

    if artifact_type == "manifold_animation":
        temperature = float(params.get("temperature", 1.0))
        if temperature < 0:
            raise InvalidParamError(f"temperature must be >= 0, got {temperature}")
        rss = params.get("reference_set_size", None)
        rss = int(rss) if rss is not None else None
        seed = int(params.get("seed", DEFAULT_SEED))
        width = float(params.get("width", DEFAULT_RBF_WIDTH))
        prefix_text = inputs.get("prefix_text", "") or ""
        response_text = inputs.get("response_text", "") or ""
        norm_params = {
            "temperature": temperature,
            "reference_set_size": rss,
            "seed": seed,
            "width": width,
        }
        key, spec = make_cache_key(
            model_id=mid,
            revision=revision,
            artifact_type="manifold_animation",
            inputs={"prefix_text": prefix_text, "response_text": response_text},
            params=norm_params,
        )

        def compute(cb: ProgressCb | None) -> dict[str, Any]:
            from .compute.manifold import manifold_animation

            return manifold_animation(
                mid,
                prefix_text=prefix_text,
                temperature=temperature,
                reference_set_size=rss,
                seed=seed,
                width=width,
                response_text=response_text,
                progress_cb=cb,
            )

        return key, spec, compute

    raise InvalidParamError(
        f"unknown artifact_type {artifact_type!r}; expected one of {ARTIFACT_TYPES}"
    )


def cache_key_for(artifact_type: str, model_id: str, params=None, inputs=None) -> str:
    key, _, _ = _plan(artifact_type, model_id, params, inputs)
    return key


def get_or_compute_sync(
    artifact_type: str,
    model_id: str,
    params: dict[str, Any] | None = None,
    inputs: dict[str, Any] | None = None,
    progress_cb: ProgressCb | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Return the cached artifact, computing+storing it once if missing. Single-flight
    per cache key so concurrent identical requests never duplicate work (FR-008)."""
    key, spec, compute = _plan(artifact_type, model_id, params, inputs)
    if not force:
        hit = _store.get(key)
        if hit is not None:
            return hit
    with _lock_for(key):
        if not force:
            hit = _store.get(key)
            if hit is not None:
                return hit
        try:
            result = compute(progress_cb)
        except LLMGeometryError:
            raise
        except Exception as exc:  # surface the real failure (no fabricated result)
            raise ComputeError(
                f"failed to compute {artifact_type} for '{model_id}': {exc}"
            ) from exc
        _store.put(key, spec, result["meta"], result["arrays"])
        cached = _store.get(key)
        if cached is None:  # write succeeded but read-back failed -> integrity problem
            raise ComputeError(f"artifact {key} failed integrity verification after write")
        return cached


def request_precompute(
    artifact_type: str,
    model_id: str,
    params: dict[str, Any] | None = None,
    inputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Non-blocking: return a cache hit, or start (or attach to) a background job."""
    key, _, _ = _plan(artifact_type, model_id, params, inputs)
    if _store.has(key):
        return {"cache_key": key, "status": "complete", "ready": True, "job_id": None}

    job, created = registry.get_or_create(key)
    if created:
        thread = threading.Thread(
            target=_run_job,
            args=(job.job_id, artifact_type, model_id, params, inputs),
            daemon=True,
        )
        thread.start()
    return {"cache_key": key, "job_id": job.job_id, "status": job.status, "ready": False}


def _run_job(job_id, artifact_type, model_id, params, inputs) -> None:
    def cb(progress: float, message: str) -> None:
        registry.update(job_id, progress=progress, message=message)

    try:
        get_or_compute_sync(artifact_type, model_id, params, inputs, progress_cb=cb)
        registry.finish(job_id)
    except LLMGeometryError as exc:
        registry.fail(job_id, {"type": exc.error_type, "message": exc.message})
    except Exception as exc:
        registry.fail(job_id, {"type": "ComputeError", "message": str(exc)})


def get_store() -> CacheStore:
    return _store
