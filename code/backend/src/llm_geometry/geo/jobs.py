"""Geometry Lab job orchestration: train + fine-tune on the shared job registry.

Keeps `/api/geo/*` orchestration out of the core `precompute.py` (which is owned by
the 001 machinery): POST /api/geo/train and POST /api/geo/finetune are idempotent,
single-flight, and cache-hit friendly, with progress streamed through the existing
`/api/jobs/{id}/events` SSE endpoint (phase-labeled per the feature-002 contract).

Also home to weight-set minting for POST /api/geo/weights, including the per-matrix
source sidecar that lets GET /api/geo/weights report `"learned" | "edited" |
"preset:<name>"` for any matrix of any minted token.
"""

from __future__ import annotations

import threading
from typing import Any

from ..cache.store import CacheStore
from ..config import SCHEMA_VERSION
from ..errors import InvalidParamError, LLMGeometryError
from ..jobs.registry import registry
from ..torchstate import TORCH_GLOBAL_LOCK
from .config import N_LAYERS, SEED
from .finetune import finetune, finetune_cache_key
from .scratch import corpus_stats, scratch_cache_key, train_scratch
from .tokenizer import get_tokenizer
from .train import canonical_cache_key, resolve_weight_set, train_canonical
from .weights import (
    _ARTIFACT_PREFIX as _WEIGHTS_PREFIX,  # stable on-disk prefix ("geo-weights")
    EDITABLE_MATRICES,
    build_weight_set,
    save_weight_set,
    weights_token,
)

_SOURCES_PREFIX = "geo-weights-sources"

# cache_key -> most recent job_id minted by *this* module (lets GET /api/geo/spec
# report a "training" status without creating a job as a side effect).
_jobs_by_key: dict[str, str] = {}
_jobs_lock = threading.Lock()


# -- canonical training ------------------------------------------------------------------


def training_job_id(seed: int = SEED) -> str | None:
    """The id of an in-flight canonical training job, if any (for GET /api/geo/spec)."""
    key, _ = canonical_cache_key(seed)
    with _jobs_lock:
        job_id = _jobs_by_key.get(key)
    if job_id is None:
        return None
    job = registry.get(job_id)
    if job is not None and job.status in ("queued", "running"):
        return job_id
    return None


def request_train(seed: int = SEED) -> dict[str, Any]:
    """Idempotent, single-flight POST /api/geo/train.

    Cache hit -> ``{"checkpoint_id", "status": "complete", "ready": True}`` (200).
    Otherwise start (or attach to) a background training job ->
    ``{"job_id", "ready": False}`` (202); progress via the jobs SSE with
    ``phase: "train"`` and a ``done`` event carrying ``checkpoint_id``.
    """
    store = CacheStore()
    key, _ = canonical_cache_key(seed)
    if store.get(key) is not None:
        meta = train_canonical(seed=seed, store=store)  # < 100 ms cached fetch
        return {"checkpoint_id": meta["checkpoint_id"], "status": "complete", "ready": True}
    job, created = registry.get_or_create(key, phase="train")
    with _jobs_lock:
        _jobs_by_key[key] = job.job_id
    if created:
        threading.Thread(target=_run_train, args=(job.job_id, seed), daemon=True).start()
    return {"job_id": job.job_id, "ready": False}


def _run_train(job_id: str, seed: int) -> None:
    def cb(progress: float, message: str) -> None:
        registry.update(job_id, progress=progress, message=message)

    try:
        # deterministic_torch flips process-global torch state (determinism flag +
        # RNG); serialize with arch tracing/generation on the shared lock.
        with TORCH_GLOBAL_LOCK:
            meta = train_canonical(progress_cb=cb, seed=seed)
        registry.finish(job_id, result={"checkpoint_id": meta["checkpoint_id"]})
    except Exception as exc:  # noqa: BLE001 — surfaced verbatim via the job error event
        # Contract: training failures surface as TrainingFailedError (500 via SSE).
        registry.fail(job_id, {"type": "TrainingFailedError", "message": str(exc)})
    finally:
        with _jobs_lock:  # prune the spec-reporting map once the job is terminal
            if _jobs_by_key.get(canonical_cache_key(seed)[0]) == job_id:
                _jobs_by_key.pop(canonical_cache_key(seed)[0], None)


# -- fine-tuning -------------------------------------------------------------------------


def request_finetune(
    *,
    text: str,
    base: str = "learned",
    steps: int,
    lr: float,
    seed: int = SEED,
) -> dict[str, Any]:
    """Idempotent, single-flight POST /api/geo/finetune (source already resolved to text).

    Content-hash cache hit -> ``{"weights_token", "loss_before", "loss_after",
    "ready": True}`` (200). Otherwise a background job -> ``{"job_id",
    "ready": False}`` (202); SSE ``phase: "finetune"``, ``done`` data =
    ``{"weights_token", "loss_before", "loss_after"}``. Never mutates the canonical.
    """
    if not text.strip():
        raise InvalidParamError("fine-tuning text is empty")
    if len(get_tokenizer().encode_stream(text)) < 2:
        raise InvalidParamError(
            "fine-tuning text is too short after tokenization (need at least 2 tokens)"
        )
    store = CacheStore()
    base_token = weights_token(resolve_weight_set(base, store=store))
    key, _ = finetune_cache_key(base_token, text, steps, lr, seed)
    entry = store.get(key)
    if entry is not None:
        meta = entry["meta"]
        return {
            "weights_token": meta["weights_token"],
            "loss_before": meta["loss_before"],
            "loss_after": meta["loss_after"],
            "ready": True,
        }
    job, created = registry.get_or_create(key, phase="finetune")
    if created:
        threading.Thread(
            target=_run_finetune, args=(job.job_id, text, base, steps, lr, seed), daemon=True
        ).start()
    return {"job_id": job.job_id, "ready": False}


def _run_finetune(job_id: str, text: str, base: str, steps: int, lr: float, seed: int) -> None:
    def cb(progress: float, message: str) -> None:
        registry.update(job_id, progress=progress, message=message)

    try:
        # Same global-torch-state serialization as canonical training.
        with TORCH_GLOBAL_LOCK:
            result = finetune(base=base, text=text, steps=steps, lr=lr, seed=seed, progress_cb=cb)
        registry.finish(
            job_id,
            result={
                "weights_token": result["weights_token"],
                "loss_before": result["loss_before"],
                "loss_after": result["loss_after"],
            },
        )
    except LLMGeometryError as exc:
        registry.fail(job_id, {"type": exc.error_type, "message": exc.message})
    except Exception as exc:  # noqa: BLE001 — surfaced verbatim via the job error event
        registry.fail(job_id, {"type": "TrainingFailedError", "message": str(exc)})


# -- from-scratch training ---------------------------------------------------------------


def request_train_scratch(
    *,
    text: str,
    epochs: int,
    seed: int = SEED,
) -> dict[str, Any]:
    """Idempotent, single-flight POST /api/geo/train_scratch (source already text).

    Content-hash cache hit -> the trained model's metadata + ``ready: True`` (200).
    Otherwise a background job -> ``{"job_id", "ready": False}`` (202); SSE
    ``phase: "train_scratch"``, ``done`` data = the same metadata. The canonical
    checkpoint is never touched — this mints an entirely separate model.
    """
    stats = corpus_stats(text)
    if stats["n_distinct"] < stats["vocab_words_required"]:
        raise InvalidParamError(
            f"This text has only {stats['n_distinct']} distinct word types, and the "
            f"model's vocabulary is {stats['vocab_words_required']} words wide — "
            "training it would leave most of the vocabulary undefined. Paste more text "
            "(a few pages of prose), or point at a larger HuggingFace dataset."
        )
    key, _ = scratch_cache_key(text, epochs, seed)
    entry = CacheStore().get(key)
    if entry is not None:
        return {**entry["meta"], "ready": True}
    job, created = registry.get_or_create(key, phase="train_scratch")
    if created:
        threading.Thread(
            target=_run_train_scratch, args=(job.job_id, text, epochs, seed), daemon=True
        ).start()
    return {"job_id": job.job_id, "ready": False}


def _run_train_scratch(job_id: str, text: str, epochs: int, seed: int) -> None:
    def cb(progress: float, message: str) -> None:
        registry.update(job_id, progress=progress, message=message)

    try:
        with TORCH_GLOBAL_LOCK:
            result = train_scratch(text=text, epochs=epochs, seed=seed, progress_cb=cb)
        registry.finish(
            job_id,
            result={
                "weights_token": result["weights_token"],
                "vocab_size": result["vocab_size"],
                "final_loss": result["final_loss"],
                "n_tokens": result["n_tokens"],
                "n_distinct": result["n_distinct"],
                "epochs": result["epochs"],
            },
        )
    except LLMGeometryError as exc:
        registry.fail(job_id, {"type": exc.error_type, "message": exc.message})
    except Exception as exc:  # noqa: BLE001 — surfaced verbatim via the job error event
        registry.fail(job_id, {"type": "TrainingFailedError", "message": str(exc)})


# -- weight-set minting + per-matrix source tracking -------------------------------------


def mint_weight_set(base: str, edits: list[dict[str, Any]]) -> dict[str, Any]:
    """POST /api/geo/weights: apply ``edits`` to ``base``; mint a content-hash token.

    Alongside the weight artifact, a small sidecar records the per-matrix source map
    (inherited from ``base``'s own sidecar, overlaid with these edits) so
    GET /api/geo/weights can report exact `"preset:<name>" / "edited"` provenance.
    """
    store = CacheStore()
    base_ws = resolve_weight_set(base, store=store)
    ws, summaries = build_weight_set(base_ws, edits)
    token = save_weight_set(ws, source="edited", store=store)

    sources = dict(_source_map(base, store))
    for s in summaries:
        sources[_matrix_key(s["layer"], s["matrix"])] = s["source"]
    store.put(
        f"{_SOURCES_PREFIX}-{token}",
        {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": _SOURCES_PREFIX,
            "weights_token": token,
        },
        {"weights_token": token, "sources": sources},
        {},
    )
    return {"weights_token": token, "edited": summaries}


def _matrix_key(layer: int | None, matrix: str) -> str:
    return "embedding" if matrix == "embedding" else f"layers.{int(layer)}.{matrix}"


def _all_matrix_keys() -> list[str]:
    keys = ["embedding"]
    for layer in range(N_LAYERS):
        keys.extend(f"layers.{layer}.{m}" for m in EDITABLE_MATRICES if m != "embedding")
    return keys


def _source_map(token_or_learned: str, store: CacheStore) -> dict[str, str]:
    if token_or_learned == "learned":
        return {}
    entry = store.get(f"{_SOURCES_PREFIX}-{token_or_learned}")
    if entry is not None:
        return dict(entry["meta"].get("sources", {}))
    # No sidecar (fine-tuned base, or evicted): every matrix inherited the base's
    # set-level provenance — seed the map from it so edits ON TOP of a fine-tune
    # don't mislabel untouched-but-fine-tuned matrices as "learned".
    artifact = store.get(f"{_WEIGHTS_PREFIX}-{token_or_learned}")
    if artifact is not None and artifact["meta"].get("source") != "learned":
        return {k: "edited" for k in _all_matrix_keys()}
    return {}


def matrix_source(token: str | None, layer: int | None, matrix: str, store: CacheStore) -> str:
    """Contract source for one matrix: "learned" | "edited" | "preset:<name>"."""
    if token is None or token == "learned":
        return "learned"
    entry = store.get(f"{_SOURCES_PREFIX}-{token}")
    if entry is not None:
        # Matrices never touched along the edit chain keep their canonical provenance.
        return entry["meta"].get("sources", {}).get(_matrix_key(layer, matrix), "learned")
    # No sidecar (e.g. a fine-tuned token, or an evicted sidecar): fall back to the
    # set-level source stored with the weight artifact itself — real recorded data,
    # never a guess. Anything non-canonical maps onto the contract's closed enum as
    # "edited" (fine-tuning edits every matrix).
    artifact = store.get(f"{_WEIGHTS_PREFIX}-{token}")
    if artifact is not None and artifact["meta"].get("source") == "learned":
        return "learned"
    return "edited"
