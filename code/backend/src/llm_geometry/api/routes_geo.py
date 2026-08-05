"""Geometry Lab routes (`/api/geo/*`).

Implements the frozen contract in specs/002-interactive-model-explorer/contracts/api.md
exactly: paths, params, response field names, and status codes. Heavy work (training,
fine-tuning) is orchestrated through `llm_geometry.geo.jobs` on the shared job
registry, so progress streams through the existing `/api/jobs/{id}/events` SSE with
phase labels. Every numeric array in a response is plain nested lists rounded to
6 significant digits (`_jsonable`).
"""

from __future__ import annotations

import re
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
    MAX_SEED,
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
from ..geo.bundle import export_bundle, import_bundle
from ..geo.scratch import (
    SCRATCH_DEFAULT_EPOCHS,
    SCRATCH_DEFAULT_MAX_SAMPLES,
    SCRATCH_MAX_EPOCHS,
    corpus_stats,
)
from ..geo.tokenizer import tokenizer_for
from ..geo.train import canonical_cache_key, resolve_weight_set
from ..geo.weights import EDITABLE_MATRICES
from .encoding import jsonable_6sig
from .params import as_float, as_int

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
    # Loosely typed on purpose, like WeightsBody: pydantic's lax mode would coerce "7"
    # and true to integers before the route ever sees them, and the contract's own
    # refusal (`_as_seed`) says which value was wrong and why.
    seed: Any = SEED


@router.post("/train")
def train(response: Response, body: TrainBody | None = None) -> dict[str, Any]:
    # Bounded, not just typed: an unbounded seed was accepted, used, and echoed back as a
    # number the browser cannot read exactly (`_as_seed`).
    result = geo_jobs.request_train(seed=_as_seed(body.seed) if body is not None else SEED)
    response.status_code = 200 if result["ready"] else 202
    return result


# -- GET /api/geo/tokenize ---------------------------------------------------------------


@router.get("/tokenize")
def tokenize(text: str = "", weights_token: str | None = None) -> dict[str, Any]:
    # A model trained from scratch carries its own vocabulary, so the ids (and which
    # words are <unk>) depend on WHICH model is active — additive `weights_token`.
    enc = tokenizer_for(weights_token).encode(text)
    return {"tokens": enc.tokens(), "n_unk": enc.n_unk, "truncated": enc.truncated}


# -- GET /api/geo/trace ------------------------------------------------------------------


@router.get("/trace")
def trace(prompt: str = "", weights_token: str | None = None) -> dict[str, Any]:
    tok = tokenizer_for(weights_token)
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
    prompt_ids = tokenizer_for(weights_token).encode(prompt).ids
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
    if name not in ws:
        # A weight set can be incomplete only if it was written by a build that did not
        # validate on the way in (see geo/bundle.import_bundle). `ws[name]` then raised
        # a bare KeyError, which the error middleware rendered as
        # `500 {"message": "'layers.0.W_V'"}` — an opaque string the UI showed verbatim.
        raise InvalidParamError(
            f"the weight set for weights_token {weights_token or 'learned'!r} is "
            f"incomplete: it has no {name!r}. It cannot be read or run; re-import the "
            "model file (a current build refuses incomplete files on load)."
        )
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
    """A JSON integer, or a typed refusal — never a coercion.

    THE implementation, shared with ``routes_lex``, lives in :mod:`api.params`; the rule
    and the reasons are documented there. It used to be a copy in each module, and the
    copies drifted (``_as_float`` here still called ``float(value)`` long after the lex
    one stopped, so ``lr: "٠.٥"`` trained at 0.5 on this tab and 400'd on the other).

    Multipart form fields arrive as strings (there is no other wire encoding for them), so
    a decimal string is parsed there and only there — see :func:`_form_int`.
    """
    return as_int(value, name)


def _form_int(value: Any, name: str) -> int:
    """Like :func:`_as_int`, but a multipart form's string field is parsed first.

    Multipart carries no types: `steps=100` arrives as the STRING "100" and always did.
    Refusing strings there would break the file-upload form the JSON path shares a body
    parser with, so the string is parsed strictly (base 10, ASCII digits only, no
    ``"0x10"``, no ``"٧"``) and then held to exactly the same rule.
    """
    if isinstance(value, str):
        text = value.strip()
        if not re.fullmatch(r"[+-]?[0-9]+", text):
            raise InvalidParamError(f"{name} must be an integer, got {value!r}")
        return int(text)
    return _as_int(value, name)


def _as_float(value: Any, name: str) -> float:
    """A FINITE JSON number, or a typed refusal — :func:`_as_int`'s rule, one type down.

    THE implementation is :func:`api.params.as_float`, shared with ``routes_lex``. This
    was the drifted copy: a bare ``float(value)`` inside a ``try``, which read Python's
    idea of a numeric string (``"٧"``, ``"７"``, ``"७"``, ``"٠.٥"``, ``"1_000"``, ``"1e3"``
    — all ``NaN`` to JavaScript's ``Number``) and answered 202, and leaked ``10**400`` as
    an untyped 500. See :mod:`api.params` for the measurements.
    """
    return as_float(value, name)


def _as_seed(value: Any, name: str = "seed") -> int:
    """An integer seed inside the range JavaScript can read back exactly.

    Unbounded, ``POST /api/geo/train`` answered 202 for ``9007199254740993`` and echoed it
    into a response the browser reads as ...992 — a run reported under a seed nobody asked
    for. The same bound ``POST /api/lex/train`` enforces.
    """
    seed = _as_int(value, name)
    if abs(seed) > MAX_SEED:
        raise InvalidParamError(
            f"{name} must lie in [-{MAX_SEED}, {MAX_SEED}]: outside that range JavaScript "
            f"cannot read the number back exactly, so the run would be reported under a "
            f"different seed than it used (got {seed})"
        )
    return seed


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

    # Multipart fields are strings on the wire; JSON fields are typed. Same rule either
    # way, one parser per encoding (see `_form_int`).
    as_int = _form_int if content_type.startswith("multipart/") else _as_int
    steps = as_int(payload.get("steps", FINETUNE_DEFAULT_STEPS), "steps")
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
            max_samples=as_int(
                payload.get("max_samples", FINETUNE_DEFAULT_MAX_SAMPLES), "max_samples"
            ),
        )
    else:
        resolved_text = str(text) if text is not None else file_text
    assert resolved_text is not None

    result = geo_jobs.request_finetune(text=resolved_text, base=base, steps=steps, lr=lr)
    response.status_code = 200 if result.get("ready") else 202
    return _jsonable(result)


# -- POST /api/geo/train_scratch ----------------------------------------------------------


@router.post("/train_scratch")
async def train_scratch_route(request: Request, response: Response) -> dict[str, Any]:
    """Train a BRAND NEW model on the user's own corpus (feature 004, FR-420).

    Same three sources as /finetune (JSON `text`, multipart `file`, or `hf_dataset`),
    but this builds a fresh vocabulary from that text and freshly initialized weights
    instead of continuing from the canonical checkpoint. The canonical checkpoint is
    never touched.
    """
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
            for k in ("text", "hf_dataset", "hf_split", "max_samples", "epochs")
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

    as_int = _form_int if content_type.startswith("multipart/") else _as_int
    epochs = as_int(payload.get("epochs", SCRATCH_DEFAULT_EPOCHS), "epochs")
    if not 1 <= epochs <= SCRATCH_MAX_EPOCHS:
        raise InvalidParamError(f"epochs must be in 1..{SCRATCH_MAX_EPOCHS}, got {epochs}")

    if hf_dataset is not None:
        # Resolved synchronously so an unusable dataset id is a 422 here, not a late
        # job error event (same rule the fine-tune route follows).
        resolved_text = load_text_from_hf(
            str(hf_dataset),
            split=str(payload.get("hf_split", "train")),
            max_samples=as_int(
                payload.get("max_samples", SCRATCH_DEFAULT_MAX_SAMPLES), "max_samples"
            ),
        )
    else:
        resolved_text = str(text) if text is not None else file_text
    assert resolved_text is not None

    result = geo_jobs.request_train_scratch(text=resolved_text, epochs=epochs)
    response.status_code = 200 if result.get("ready") else 202
    return _jsonable(result)


# -- GET /api/geo/corpus_stats ------------------------------------------------------------


@router.get("/corpus_stats")
def corpus_stats_route(text: str = "") -> dict[str, Any]:
    """Token / distinct-type counts for a candidate corpus.

    Lets the UI show whether a paste is big enough to train on BEFORE the user waits
    for a training run that would be refused.
    """
    return _jsonable(corpus_stats(text))


# -- GET/POST /api/geo/model (portable save + load) ---------------------------------------


@router.get("/model")
def export_model_route(weights_token: str = "learned") -> dict[str, Any]:
    """The active model as one portable, self-describing bundle (weights + vocabulary)."""
    return _jsonable(export_bundle(weights_token))


@router.post("/model")
async def import_model_route(request: Request) -> dict[str, Any]:
    """Load a bundle saved by GET /api/geo/model; returns its weights_token."""
    try:
        payload = await request.json()
    except Exception:
        raise InvalidParamError("request body must be a JSON model bundle")
    return _jsonable(import_bundle(payload))
