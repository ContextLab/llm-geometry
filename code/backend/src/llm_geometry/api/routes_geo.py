"""Geometry Lab routes (`/api/geo/*`).

Implements the frozen contract in specs/002-interactive-model-explorer/contracts/api.md
exactly: paths, params, response field names, and status codes. Heavy work (training,
fine-tuning) is orchestrated through `llm_geometry.geo.jobs` on the shared job
registry, so progress streams through the existing `/api/jobs/{id}/events` SSE with
phase labels. Every numeric array in a response is plain nested lists rounded to
6 significant digits (`_jsonable`).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from fastapi import APIRouter, Request, Response
from pydantic import BaseModel, Field

from ..cache.store import CacheStore
from ..errors import InvalidParamError
from ..geo import jobs as geo_jobs
from ..geo.config import (
    CONTEXT_WINDOW,
    CORPUS_ID,
    D_MODEL,
    FINETUNE_DEFAULT_LR,
    FINETUNE_DEFAULT_MAX_SAMPLES,
    FINETUNE_DEFAULT_STEPS,
    FINETUNE_MAX_STEPS,
    MLP_HIDDEN,
    N_HEADS,
    N_LAYERS,
    SEED,
    SPECIAL_TOKENS,
    VOCAB_SIZE,
)
from ..geo.fields import force_field, next_next_field
from ..geo.finetune import load_text_from_hf
from ..geo.model import model_from_weight_set
from ..geo.tokenizer import get_tokenizer
from ..geo.train import canonical_cache_key, resolve_weight_set
from ..geo.weights import EDITABLE_MATRICES
from .encoding import jsonable_6sig

router = APIRouter(prefix="/geo", tags=["geo"])


# -- contract-wide array/number encoding (shared helper; red-team round 2 NIT-6) ---------

_jsonable = jsonable_6sig


# -- GET /api/geo/spec -------------------------------------------------------------------


@router.get("/spec")
def spec() -> dict[str, Any]:
    store = CacheStore()
    key, _ = canonical_cache_key()
    entry = store.get(key)
    if entry is not None:
        meta = entry["meta"]
        checkpoint = {
            "status": "ready",
            "checkpoint_id": meta["checkpoint_id"],
            "final_loss": meta["final_loss"],
            "coverage_uniformity": meta["coverage_uniformity"],
            "field_directional_entropy": meta["field_directional_entropy"],
            "job_id": None,
        }
    else:
        job_id = geo_jobs.training_job_id()
        checkpoint = {
            "status": "training" if job_id is not None else "missing",
            "checkpoint_id": None,
            "final_loss": None,
            "coverage_uniformity": None,
            "field_directional_entropy": None,
            "job_id": job_id,
        }
    return _jsonable(
        {
            "model": {
                "d_model": D_MODEL,
                "n_layers": N_LAYERS,
                "n_heads": N_HEADS,
                "mlp_hidden": MLP_HIDDEN,
                "vocab_size": VOCAB_SIZE,
                "context_window": CONTEXT_WINDOW,
                "tied_unembedding": True,
                "corpus": CORPUS_ID,
                "seed": SEED,
            },
            "special_tokens": dict(SPECIAL_TOKENS),
            "checkpoint": checkpoint,
        }
    )


# -- POST /api/geo/train -----------------------------------------------------------------


class TrainBody(BaseModel):
    seed: int = SEED


@router.post("/train")
def train(response: Response, body: TrainBody | None = None) -> dict[str, Any]:
    result = geo_jobs.request_train(seed=body.seed if body is not None else SEED)
    response.status_code = 200 if result["ready"] else 202
    return result


# -- GET /api/geo/tokenize ---------------------------------------------------------------


@router.get("/tokenize")
def tokenize(text: str = "") -> dict[str, Any]:
    enc = get_tokenizer().encode(text)
    return {"tokens": enc.tokens(), "n_unk": enc.n_unk, "truncated": enc.truncated}


# -- GET /api/geo/trace ------------------------------------------------------------------


@router.get("/trace")
def trace(prompt: str = "", weights_token: str | None = None) -> dict[str, Any]:
    tok = get_tokenizer()
    enc = tok.encode(prompt)
    if not enc.ids:
        raise InvalidParamError("prompt is empty after tokenization")
    model = model_from_weight_set(resolve_weight_set(weights_token or "learned"))
    tr = model.forward_trace(enc.ids)
    probs = np.asarray(tr["probs"], dtype=np.float64)
    top = [int(i) for i in np.argsort(-probs)[:10]]
    next_id = top[0]
    return _jsonable(
        {
            "tokens": enc.tokens(),
            "embeddings": tr["embeddings"],
            "layers": tr["layers"],
            "probs": tr["probs"],
            "logits_topk": {
                "ids": top,
                "texts": [tok.id_to_text[i] for i in top],
                "probs": [float(probs[i]) for i in top],
            },
            "next_token": {"id": next_id, "text": tok.id_to_text[next_id]},
        }
    )


# -- GET /api/geo/vector_field -----------------------------------------------------------


@router.get("/vector_field")
def vector_field(
    mode: str = "next_next",
    layer: str = "full",
    prompt: str = "",
    weights_token: str | None = None,
    temperature: float = 0.0,
    top_m: int = 1,
    antisymmetrize: bool = False,
) -> dict[str, Any]:
    if mode not in ("next_next", "force"):
        raise InvalidParamError(f'mode must be "next_next" or "force", got {mode!r}')
    prompt_ids = get_tokenizer().encode(prompt).ids
    model = model_from_weight_set(resolve_weight_set(weights_token or "learned"))
    if mode == "next_next":
        field = next_next_field(
            model, prompt_ids, layer=layer, temperature=temperature, top_m=top_m
        )
    else:  # force mode; layer="full" -> 400 (InvalidParamError raised by force_field)
        field = force_field(model, prompt_ids, layer=layer, antisymmetrize=antisymmetrize)
    return _jsonable(field)


# -- GET /api/geo/weights ----------------------------------------------------------------


@router.get("/weights")
def weights_get(
    matrix: str,
    layer: int | None = None,
    weights_token: str | None = None,
) -> dict[str, Any]:
    if matrix not in EDITABLE_MATRICES:
        raise InvalidParamError(f"matrix must be one of {EDITABLE_MATRICES}, got {matrix!r}")
    store = CacheStore()
    ws = resolve_weight_set(weights_token or "learned", store=store)
    if matrix == "embedding":  # contract: embedding ignores `layer`
        name, src_layer = "embedding", None
    else:
        if layer is None or not 0 <= layer < N_LAYERS:
            raise InvalidParamError(
                f"layer must be an int in 0..{N_LAYERS - 1} for {matrix}, got {layer!r}"
            )
        name, src_layer = f"layers.{layer}.{matrix}", layer
    values = np.asarray(ws[name], dtype=np.float32)
    return _jsonable(
        {
            "values": values,
            "shape": list(values.shape),
            "source": geo_jobs.matrix_source(weights_token, src_layer, matrix, store),
        }
    )


# -- POST /api/geo/weights ---------------------------------------------------------------


class WeightsBody(BaseModel):
    base: str = "learned"
    # Loosely typed on purpose: edit validation lives in geo.weights.build_weight_set so
    # malformed edits produce the contract's InvalidWeightEditError envelope (422),
    # never a raw pydantic validation response.
    edits: list[dict[str, Any]] = Field(default_factory=list)


@router.post("/weights")
def weights_post(body: WeightsBody) -> dict[str, Any]:
    return _jsonable(geo_jobs.mint_weight_set(body.base, body.edits))


# -- POST /api/geo/finetune --------------------------------------------------------------


def _as_int(value: Any, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        raise InvalidParamError(f"{name} must be an integer, got {value!r}")


def _as_float(value: Any, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        raise InvalidParamError(f"{name} must be a number, got {value!r}")


@router.post("/finetune")
async def finetune_route(request: Request, response: Response) -> dict[str, Any]:
    """JSON body {text|hf_dataset|…} or multipart with a .txt/.md `file` field."""
    content_type = (request.headers.get("content-type") or "").lower()
    file_text: str | None = None
    if content_type.startswith("multipart/"):
        form = await request.form()
        upload = form.get("file")
        if upload is not None:
            if isinstance(upload, str):
                raise InvalidParamError("the `file` field must be an uploaded file")
            filename = (upload.filename or "").lower()
            if not filename.endswith((".txt", ".md")):
                raise InvalidParamError(
                    f"uploaded file must be .txt or .md, got {upload.filename!r}"
                )
            try:
                file_text = (await upload.read()).decode("utf-8")
            except UnicodeDecodeError as exc:
                raise InvalidParamError(f"uploaded file is not valid UTF-8 text: {exc}")
        payload: dict[str, Any] = {
            k: form.get(k)
            for k in ("text", "hf_dataset", "hf_split", "max_samples", "steps", "lr", "base")
            if form.get(k) is not None
        }
    else:
        try:
            payload = await request.json()
        except Exception:
            raise InvalidParamError("request body must be JSON or multipart form data")
        if not isinstance(payload, dict):
            raise InvalidParamError("JSON body must be an object")

    text = payload.get("text")
    hf_dataset = payload.get("hf_dataset")
    provided = [s for s in (text, file_text, hf_dataset) if s is not None]
    if len(provided) != 1:
        raise InvalidParamError(
            "exactly one of text / file / hf_dataset must be provided, "
            f"got {len(provided)} sources"
        )

    steps = _as_int(payload.get("steps", FINETUNE_DEFAULT_STEPS), "steps")
    if not 1 <= steps <= FINETUNE_MAX_STEPS:
        raise InvalidParamError(f"steps must be in 1..{FINETUNE_MAX_STEPS}, got {steps}")
    lr = _as_float(payload.get("lr", FINETUNE_DEFAULT_LR), "lr")
    if lr <= 0:
        raise InvalidParamError(f"lr must be > 0, got {lr}")
    base = str(payload.get("base", "learned"))

    if hf_dataset is not None:
        # Resolved synchronously so an unusable dataset id is a 422 here (contract),
        # not a late job error event.
        resolved_text = load_text_from_hf(
            str(hf_dataset),
            split=str(payload.get("hf_split", "train")),
            max_samples=_as_int(
                payload.get("max_samples", FINETUNE_DEFAULT_MAX_SAMPLES), "max_samples"
            ),
        )
    else:
        resolved_text = str(text) if text is not None else file_text
    assert resolved_text is not None

    result = geo_jobs.request_finetune(text=resolved_text, base=base, steps=steps, lr=lr)
    response.status_code = 200 if result.get("ready") else 202
    return _jsonable(result)
