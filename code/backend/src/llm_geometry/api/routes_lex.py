"""Lexicon Lab routes (`/api/lex/*`) — feature 006.

The contract for these endpoints is `specs/006-lexicon-lab-tiny/contracts/api-lex.md`.
They are *additive*: the frozen feature-002 contract
(`specs/002-interactive-model-explorer/contracts/api.md`) is untouched, and everything
here reuses its machinery unchanged — the same error envelope, the same
`/api/jobs/{id}/events` SSE (with `phase: "lex_train"`), the same `CacheStore`, and the
same "every float rounded to 6 significant digits" array encoding.

Three things live here that could look like they belong elsewhere, and do not:

* **The model store.** A trained Lexicon model is `(weights, vocabulary, config)` — none
  of the three means anything without the other two, because the vocabulary *is* the
  independent variable of this tab. `llm_geometry.geo.weights` stores a fixed-shape
  model, so it cannot hold these; a `lex-model-<token>` artifact does.
* **Job orchestration.** `geo/jobs.py` is the Geometry Lab's; this is the same pattern
  (single-flight on a content-hash cache key, background thread, progress through the
  shared registry), applied to a different computation.
* **The portable bundle.** Same shape and the same strictness as `geo/bundle.py`: a
  declared token that does not match a re-hash of the file's own weights is rejected
  rather than loaded, because silently attaching the wrong vocabulary to a set of
  weights would make every word in the UI confidently wrong.

Nothing in this module fabricates a number. Coverage is measured against real text, the
spectrum is computed from real weights, and generation runs a real forward pass.
"""

from __future__ import annotations

import base64
import hashlib
import threading
from functools import lru_cache
from typing import Any

import numpy as np
from fastapi import APIRouter, Request, Response

from ..cache.keys import _canonical, make_cache_key
from ..cache.store import CacheStore
from ..config import SCHEMA_VERSION
from ..errors import ComputeError, InvalidParamError, LLMGeometryError, NotFoundError
from ..jobs.registry import registry
from ..lex.config import (
    BUDGET_SOURCES,
    CORPUS_BYTES,
    CORPUS_GUTENBERG_ID,
    CORPUS_SHA256,
    CORPUS_TITLE,
    CORPUS_YEAR,
    CTX_CHOICES,
    D_MODEL_CHOICES,
    DEFAULT_BATCH,
    DEFAULT_BUDGET,
    DEFAULT_BUDGET_SOURCE,
    DEFAULT_CTX,
    DEFAULT_D_MODEL,
    DEFAULT_DROPOUT,
    DEFAULT_LR,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_N_HEADS,
    DEFAULT_N_LAYERS,
    DEFAULT_SAMPLE_EVERY,
    DEFAULT_SEED,
    DEFAULT_STEPS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIED,
    DEFAULT_WEIGHT_DECAY,
    GENERATION_BANNED_IDS,
    GRAD_CLIP_NORM,
    LAYER_NORM_EPS,
    MAX_NEW_TOKENS,
    MAX_STEPS,
    MLP_RATIO,
    N_HEAD_CHOICES,
    N_LAYER_CHOICES,
    ONECYCLE_DIV_FACTOR,
    ONECYCLE_FINAL_DIV_FACTOR,
    ONECYCLE_PCT_START,
    PCA_COMPONENTS,
    SPECIAL_TOKENS,
    SPECTRUM_DISPLAY_K,
    VAL_FRACTION,
    param_count,
)
from ..lex.corpus import load_corpus_text
from ..lex.dolch import DOLCH_ORDER, dolch_sizes
from ..lex.spectrum import (
    compare_to_baseline,
    model_matrix,
    random_baseline_spectrum,
    spectrum,
)
from ..lex.vocab import LexVocab, build_vocab, tokenize
from ..torchstate import TORCH_GLOBAL_LOCK
from .encoding import jsonable_6sig

router = APIRouter(prefix="/lex", tags=["lex"])

_jsonable = jsonable_6sig

#: Cache prefix for a stored Lexicon model. Bumping this invalidates every stored model.
_MODEL_PREFIX = "lex-model"
_TRAIN_ARTIFACT = "lex-train"

BUNDLE_FORMAT = "llm-geometry/lex-model"
BUNDLE_VERSION = 1

#: Words returned as examples of what a budget cannot say. A sample, labelled as one.
_OOV_SAMPLE_SIZE = 24


# -- corpus + vocabulary helpers ----------------------------------------------------------


@lru_cache(maxsize=1)
def _default_corpus() -> str:
    """The shipped corpus body, digest-verified on first use (never re-read per request)."""
    return load_corpus_text()


@lru_cache(maxsize=64)
def _corpus_stats(text: str) -> dict[str, int]:
    tokens = tokenize(text)
    lines = [line for line in text.splitlines() if line.strip()]
    return {
        "n_tokens": len(tokens),
        "n_distinct": len(set(tokens)),
        "n_lines": len(lines),
        "n_chars": len(text),
    }


def _resolve_budget(source: str, budget: str, size: int | None, corpus_text: str) -> LexVocab:
    """Validate a (source, budget, size) triple and build the vocabulary it names."""
    if source not in BUDGET_SOURCES:
        raise InvalidParamError(f"source must be one of {list(BUDGET_SOURCES)}, got {source!r}")
    if budget not in DOLCH_ORDER:
        raise InvalidParamError(f"budget must be one of {DOLCH_ORDER}, got {budget!r}")
    if size is not None:
        if source != "frequency":
            raise InvalidParamError(
                'size applies only to source="frequency"; a Dolch budget IS its list, '
                "so its size is measured from the data, not chosen"
            )
        if size < 1:
            raise InvalidParamError(f"size must be at least 1, got {size}")
    vocab = build_vocab(source, budget, corpus_text, size=size)
    if vocab.budget_size < 1:
        raise InvalidParamError(
            "the resolved budget is empty — this corpus has no word tokens to draw one from"
        )
    return vocab


def _budget_payload(vocab: LexVocab, corpus_text: str) -> dict[str, Any]:
    cov = vocab.coverage(corpus_text)
    return {
        "source": vocab.source,
        "budget": vocab.budget_name,
        "size": vocab.budget_size,
        "rows": vocab.rows,
        "coverage": cov.as_dict(),
    }


def _oov_sample(vocab: LexVocab, corpus_text: str) -> list[dict[str, Any]]:
    """The most frequent out-of-budget types: what this budget literally cannot say."""
    from collections import Counter

    in_budget = set(vocab.words)
    counts = Counter(t for t in tokenize(corpus_text) if t not in in_budget)
    return [
        {"word": word, "count": count}
        for word, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:_OOV_SAMPLE_SIZE]
    ]


# -- the model store ----------------------------------------------------------------------


def _model_token(
    weights: dict[str, np.ndarray], config: dict[str, Any], words: tuple[str, ...]
) -> str:
    """Content hash over weights + shape + vocabulary.

    The vocabulary is inside the hash on purpose: two models with byte-identical weights
    but different word lists are different models, and giving them one token would let a
    cache hit serve the wrong labels.
    """
    h = hashlib.sha256()
    h.update(_canonical(config).encode("utf-8"))
    h.update(_canonical(list(words)).encode("utf-8"))
    _hash_weights(h, weights)
    return h.hexdigest()[:32]


def _hash_weights(h: "hashlib._Hash", weights: dict[str, np.ndarray]) -> None:
    """Feed ``[name, repr(shape), float32-LE bytes]`` for every tensor, name-sorted."""
    for name in sorted(weights):
        arr = np.ascontiguousarray(np.asarray(weights[name], dtype=np.float32))
        h.update(name.encode("utf-8"))
        h.update(repr(arr.shape).encode("utf-8"))
        h.update(arr.tobytes())


def _weights_token(weights: dict[str, np.ndarray]) -> str:
    """Content hash over the WEIGHTS ALONE.

    ``_model_token`` proves the weights, the config and the word list belong together;
    this one names the weights by themselves, which is what the browser's Weight Lab mints
    for an edited weight set and what tells you two files hold the same weights under
    different vocabularies. Same construction as ``geo/weights.py::weights_token``.
    """
    h = hashlib.sha256()
    _hash_weights(h, weights)
    return h.hexdigest()[:32]


def _vocab_digest(words: tuple[str, ...] | list[str], source: str, budget: str) -> str:
    """SHA-256 of the canonical vocabulary block.

    The weights hash cannot cover a word list, so the word list carries its own digest.
    Without it a file with genuine weights and a fabricated vocabulary loads cleanly and
    mislabels every token on screen.
    """
    canonical = _canonical(
        {
            "budget": budget,
            "source": source,
            "specials": list(SPECIAL_TOKENS),
            "words": list(words),
        }
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _save_model(
    *,
    weights: dict[str, np.ndarray],
    config: dict[str, Any],
    vocab: LexVocab,
    metrics: dict[str, Any],
    store: CacheStore | None = None,
) -> str:
    store = store or CacheStore()
    token = _model_token(weights, config, vocab.words)
    key = f"{_MODEL_PREFIX}-{token}"
    if store.get(key) is None:  # identical content already stored -> dedup
        store.put(
            key,
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": _MODEL_PREFIX,
                "model_token": token,
            },
            {
                "model_token": token,
                "config": config,
                "vocab": {
                    "source": vocab.source,
                    "budget": vocab.budget_name,
                    "words": list(vocab.words),
                },
                "metrics": metrics,
            },
            {name: np.asarray(a, np.float32) for name, a in weights.items()},
        )
    return token


def _load_model(
    token: str, store: CacheStore | None = None
) -> tuple[Any, LexVocab, dict[str, Any]]:
    """Rebuild ``(LexModel, LexVocab, meta)`` for a stored token, or 404."""
    from ..lex.model import LexConfig, model_from_weight_dict

    store = store or CacheStore()
    entry = store.get(f"{_MODEL_PREFIX}-{token}")
    if entry is None:
        raise NotFoundError(
            f"no Lexicon model with token {token!r}. Models live in the server's cache; "
            "train one (POST /api/lex/train) or load a saved bundle (POST /api/lex/model)."
        )
    meta = entry["meta"]
    cfg = LexConfig.from_dict(dict(meta["config"]))
    vocab = LexVocab(
        words=tuple(meta["vocab"]["words"]),
        source=meta["vocab"]["source"],
        budget_name=meta["vocab"]["budget"],
    )
    weights = {name: np.asarray(arr, np.float32) for name, arr in entry["arrays"].items()}
    model = model_from_weight_dict(cfg, weights)
    return model, vocab, meta


# -- parameter coercion (same envelope as every other failure) -----------------------------


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


def _as_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in ("true", "false", "1", "0"):
        return value.lower() in ("true", "1")
    raise InvalidParamError(f"{name} must be a boolean, got {value!r}")


def _one_of(value: Any, choices: tuple[int, ...], name: str) -> int:
    number = _as_int(value, name)
    if number not in choices:
        raise InvalidParamError(f"{name} must be one of {list(choices)}, got {number}")
    return number


def _model_config_from(payload: dict[str, Any], vocab_rows: int) -> dict[str, Any]:
    """Validate the shape controls (FR-611) and return a `LexConfig`-shaped dict."""
    d_model = _one_of(payload.get("d_model", DEFAULT_D_MODEL), D_MODEL_CHOICES, "d_model")
    n_layers = _one_of(payload.get("n_layers", DEFAULT_N_LAYERS), N_LAYER_CHOICES, "n_layers")
    n_heads = _one_of(payload.get("n_heads", DEFAULT_N_HEADS), N_HEAD_CHOICES, "n_heads")
    ctx = _one_of(payload.get("ctx", DEFAULT_CTX), CTX_CHOICES, "ctx")
    if d_model % n_heads != 0:
        raise InvalidParamError(
            f"d_model must be divisible by n_heads so every head gets an equal slice: "
            f"d_model={d_model}, n_heads={n_heads}"
        )
    dropout = _as_float(payload.get("dropout", DEFAULT_DROPOUT), "dropout")
    if not 0.0 <= dropout < 1.0:
        raise InvalidParamError(f"dropout must be in [0, 1), got {dropout}")
    return {
        "vocab_rows": vocab_rows,
        "d_model": d_model,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "ctx": ctx,
        "tied": _as_bool(payload.get("tied", DEFAULT_TIED), "tied"),
        "dropout": dropout,
    }


async def _json_body(request: Request) -> dict[str, Any]:
    try:
        payload = await request.json()
    except Exception:
        raise InvalidParamError("request body must be JSON")
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise InvalidParamError("JSON body must be an object")
    return payload


async def _text_source(payload: dict[str, Any]) -> str:
    """The corpus this request trains/measures against: default, paste, or HF dataset.

    Reuses feature 004's HuggingFace loader (FR-608) and resolves it *synchronously* so
    an unusable dataset id is a 422 on this request, not a late job error event — the
    same rule `/api/geo/finetune` follows.
    """
    text = payload.get("text")
    hf_dataset = payload.get("hf_dataset")
    if text is not None and hf_dataset is not None:
        raise InvalidParamError("provide at most one of text / hf_dataset, not both")
    if hf_dataset is not None:
        from ..geo.finetune import load_text_from_hf

        return load_text_from_hf(
            str(hf_dataset),
            split=str(payload.get("hf_split", "train")),
            max_samples=_as_int(payload.get("max_samples", 2000), "max_samples"),
        )
    if text is not None:
        resolved = str(text)
        if not tokenize(resolved):
            raise InvalidParamError("the supplied text has no word tokens to train on")
        return resolved
    return _default_corpus()


# -- GET /api/lex/spec ---------------------------------------------------------------------


@router.get("/spec")
def spec() -> dict[str, Any]:
    """Everything the tab needs before the user touches a control.

    Budget sizes are **measured** from `dolch.py`, never quoted (FR-602): the reason the
    largest is 314 and not the widely-cited 315 is that `Santa Claus` cannot be matched by
    a word tokenizer and was dropped, and a number the UI quotes must be the number the
    data actually has.
    """
    corpus = _default_corpus()
    stats = _corpus_stats(corpus)
    sizes = dolch_sizes()
    return _jsonable(
        {
            "corpus": {
                "title": CORPUS_TITLE,
                "year": CORPUS_YEAR,
                "gutenberg_id": CORPUS_GUTENBERG_ID,
                "sha256": CORPUS_SHA256,
                "bytes": CORPUS_BYTES,
                **stats,
            },
            "budget_sources": list(BUDGET_SOURCES),
            "budgets": [
                {"name": name, "size": sizes[name], "rows": sizes[name] + len(SPECIAL_TOKENS)}
                for name in DOLCH_ORDER
            ],
            "special_tokens": {name: index for index, name in enumerate(SPECIAL_TOKENS)},
            "generation_banned_ids": list(GENERATION_BANNED_IDS),
            "model": {
                "d_model_choices": list(D_MODEL_CHOICES),
                "n_layer_choices": list(N_LAYER_CHOICES),
                "n_head_choices": list(N_HEAD_CHOICES),
                "ctx_choices": list(CTX_CHOICES),
                "mlp_ratio": MLP_RATIO,
                "layer_norm_eps": LAYER_NORM_EPS,
                "defaults": {
                    "d_model": DEFAULT_D_MODEL,
                    "n_layers": DEFAULT_N_LAYERS,
                    "n_heads": DEFAULT_N_HEADS,
                    "ctx": DEFAULT_CTX,
                    "tied": DEFAULT_TIED,
                    "dropout": DEFAULT_DROPOUT,
                    "budget_source": DEFAULT_BUDGET_SOURCE,
                    "budget": DEFAULT_BUDGET,
                },
            },
            "training": {
                "max_steps": MAX_STEPS,
                "grad_clip_norm": GRAD_CLIP_NORM,
                "val_fraction": VAL_FRACTION,
                "onecycle": {
                    "pct_start": ONECYCLE_PCT_START,
                    "div_factor": ONECYCLE_DIV_FACTOR,
                    "final_div_factor": ONECYCLE_FINAL_DIV_FACTOR,
                },
                "defaults": {
                    "steps": DEFAULT_STEPS,
                    "lr": DEFAULT_LR,
                    "batch_size": DEFAULT_BATCH,
                    "weight_decay": DEFAULT_WEIGHT_DECAY,
                    "seed": DEFAULT_SEED,
                    "sample_every": DEFAULT_SAMPLE_EVERY,
                },
            },
            "generation": {
                "max_new_tokens_limit": MAX_NEW_TOKENS,
                "defaults": {
                    "temperature": DEFAULT_TEMPERATURE,
                    "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
                    "seed": DEFAULT_SEED,
                },
            },
            "spectrum": {"pca_components": PCA_COMPONENTS, "display_k": SPECTRUM_DISPLAY_K},
        }
    )


# -- GET /api/lex/budgets ------------------------------------------------------------------


@router.get("/budgets")
def budgets(
    source: str = DEFAULT_BUDGET_SOURCE,
    d_model: int = DEFAULT_D_MODEL,
    n_layers: int = DEFAULT_N_LAYERS,
    n_heads: int = DEFAULT_N_HEADS,
    ctx: int = DEFAULT_CTX,
    tied: bool = DEFAULT_TIED,
) -> dict[str, Any]:
    """Every budget at once, measured against the shipped corpus (US-1, FR-606, FR-612).

    The parameter count travels with each row because it is the other half of the
    trade-off the user is making: a bigger budget covers more of the corpus *and* costs
    more embedding rows.
    """
    corpus = _default_corpus()
    config = _model_config_from(
        {
            "d_model": d_model,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "ctx": ctx,
            "tied": tied,
        },
        vocab_rows=0,  # filled per budget below
    )
    rows = []
    for name in DOLCH_ORDER:
        vocab = _resolve_budget(source, name, None, corpus)
        payload = _budget_payload(vocab, corpus)
        payload["param_count"] = param_count(
            vocab_rows=vocab.rows,
            d_model=config["d_model"],
            n_layers=config["n_layers"],
            ctx=config["ctx"],
            tied=config["tied"],
        )
        rows.append(payload)
    return _jsonable(
        {
            "source": source,
            "corpus": {"title": CORPUS_TITLE, **_corpus_stats(corpus)},
            "model": {k: v for k, v in config.items() if k != "vocab_rows"},
            "budgets": rows,
        }
    )


# -- POST /api/lex/coverage ----------------------------------------------------------------


@router.post("/coverage")
async def coverage(request: Request) -> dict[str, Any]:
    """One budget measured against one corpus (SC-603, US-3, US-5).

    POST rather than GET because the corpus may be a whole pasted book. With no `text`
    and no `hf_dataset` this measures the shipped corpus, which is what the UI shows by
    default.
    """
    payload = await _json_body(request)
    corpus = await _text_source(payload)
    source = str(payload.get("source", DEFAULT_BUDGET_SOURCE))
    budget = str(payload.get("budget", DEFAULT_BUDGET))
    size = payload.get("size")
    vocab = _resolve_budget(
        source, budget, _as_int(size, "size") if size is not None else None, corpus
    )
    return _jsonable(
        {
            **_budget_payload(vocab, corpus),
            "corpus": _corpus_stats(corpus),
            "oov_sample": _oov_sample(vocab, corpus),
            "words": list(vocab.words),
        }
    )


# -- POST /api/lex/train -------------------------------------------------------------------


def _train_cache_key(spec_dict: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    return make_cache_key(
        model_id="lex",
        revision=None,
        artifact_type=_TRAIN_ARTIFACT,
        inputs={"corpus_sha256": spec_dict.pop("corpus_sha256")},
        params=spec_dict,
        seed=spec_dict.get("seed"),
    )


@router.post("/train")
async def train(request: Request, response: Response) -> dict[str, Any]:
    """Train from scratch, or fine-tune an existing model on new text (US-2, FR-619).

    Idempotent and single-flight on a content-hash key, exactly like `/api/geo/train`:
    a repeat of the same request returns the cached model (200), a new one starts a
    background job (202) whose progress streams through `/api/jobs/{id}/events` with
    `phase: "lex_train"`.

    With `base` set, the *existing model's* vocabulary is used and travels with the
    result — feature 004's issue #6 was exactly the opposite mistake, and repeating it
    would silently re-tokenize a fine-tune against a vocabulary the weights never saw.
    """
    payload = await _json_body(request)
    text = await _text_source(payload)

    steps = _as_int(payload.get("steps", DEFAULT_STEPS), "steps")
    if not 1 <= steps <= MAX_STEPS:
        raise InvalidParamError(f"steps must be in 1..{MAX_STEPS}, got {steps}")
    lr = _as_float(payload.get("lr", DEFAULT_LR), "lr")
    if lr <= 0:
        raise InvalidParamError(f"lr must be > 0, got {lr}")
    batch_size = _as_int(payload.get("batch_size", DEFAULT_BATCH), "batch_size")
    if batch_size < 1:
        raise InvalidParamError(f"batch_size must be at least 1, got {batch_size}")
    weight_decay = _as_float(payload.get("weight_decay", DEFAULT_WEIGHT_DECAY), "weight_decay")
    if weight_decay < 0:
        raise InvalidParamError(f"weight_decay must be >= 0, got {weight_decay}")
    seed = _as_int(payload.get("seed", DEFAULT_SEED), "seed")
    sample_every = _as_int(payload.get("sample_every", DEFAULT_SAMPLE_EVERY), "sample_every")
    if sample_every < 1:
        raise InvalidParamError(f"sample_every must be at least 1, got {sample_every}")

    base = payload.get("base")
    if base is not None:
        base = str(base)
        _, base_vocab, base_meta = _load_model(base)  # 404 here, not inside the job
        config = dict(base_meta["config"])
        vocab = base_vocab
        for control in (
            "d_model",
            "n_layers",
            "n_heads",
            "ctx",
            "tied",
            "source",
            "budget",
            "size",
        ):
            if control in payload:
                raise InvalidParamError(
                    f"{control} cannot be set when fine-tuning from `base`: a fine-tune keeps "
                    "the base model's shape and vocabulary, which is the whole point of it"
                )
    else:
        source = str(payload.get("source", DEFAULT_BUDGET_SOURCE))
        budget = str(payload.get("budget", DEFAULT_BUDGET))
        size = payload.get("size")
        vocab = _resolve_budget(
            source, budget, _as_int(size, "size") if size is not None else None, text
        )
        config = _model_config_from(payload, vocab_rows=vocab.rows)

    key, _ = _train_cache_key(
        {
            "corpus_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "config": config,
            "vocab": list(vocab.words),
            "vocab_source": vocab.source,
            "base": base,
            "steps": steps,
            "lr": lr,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "seed": seed,
        }
    )
    store = CacheStore()
    entry = store.get(key)
    if entry is not None:
        response.status_code = 200
        return _jsonable({**entry["meta"]["result"], "ready": True})

    job, created = registry.get_or_create(key, phase="lex_train")
    if created:
        threading.Thread(
            target=_run_train,
            args=(
                job.job_id,
                key,
                text,
                config,
                vocab,
                base,
                steps,
                lr,
                batch_size,
                weight_decay,
                seed,
                sample_every,
            ),
            daemon=True,
        ).start()
    response.status_code = 202
    return {"job_id": job.job_id, "ready": False}


def _run_train(
    job_id: str,
    key: str,
    text: str,
    config: dict[str, Any],
    vocab: LexVocab,
    base: str | None,
    steps: int,
    lr: float,
    batch_size: int,
    weight_decay: float,
    seed: int,
    sample_every: int,
) -> None:
    from ..lex.generate import generate_text
    from ..lex.model import LexConfig, LexModel
    from ..lex.train import TrainProgress, token_stream, train_lex

    try:
        with TORCH_GLOBAL_LOCK:  # training flips process-global torch RNG state
            cfg = LexConfig.from_dict(config)
            model = _load_model(base)[0] if base is not None else LexModel(cfg, seed=seed)
            stream = token_stream(text, vocab)
            if stream.size <= cfg.ctx:
                raise InvalidParamError(
                    f"this corpus is {stream.size} tokens long and the model's context is "
                    f"{cfg.ctx} — there is not a single full training window in it"
                )

            # FR-618. `last_sample` persists between ticks on purpose: a sample that
            # only rode on the one tick that produced it would be overwritten within
            # milliseconds and the ~4 Hz SSE poller would essentially never show one.
            state = {"sample": "", "at": 0}

            def on_progress(progress: TrainProgress) -> None:
                # Interval measured against the last sample rather than `step %
                # sample_every`, so it fires whatever cadence `progress_every` happens
                # to impose on the callback.
                if (
                    progress.step - state["at"] >= sample_every
                    or progress.step == progress.total_steps
                ):
                    state["sample"] = " ".join(
                        generate_text(
                            model, vocab, "", max_new_tokens=12, temperature=0.9, seed=seed
                        ).split()
                    )
                    state["at"] = progress.step
                message = (
                    f"step {progress.step}/{progress.total_steps} · "
                    f"loss {progress.loss:.3f} · lr {progress.lr:.2e}"
                )
                if state["sample"]:
                    message += f" · {state['sample']}"
                registry.update(
                    job_id, progress=progress.step / progress.total_steps, message=message
                )

            result = train_lex(
                stream=stream,
                model=model,
                steps=steps,
                lr=lr,
                batch_size=batch_size,
                weight_decay=weight_decay,
                seed=seed,
                progress_cb=on_progress,
                progress_every=max(1, steps // 100),
            )
            sample = generate_text(model, vocab, "", max_new_tokens=24, temperature=0.9, seed=seed)
            token = _save_model(
                weights=model.weight_dict(),
                config=cfg.as_dict(),
                vocab=vocab,
                metrics={
                    "first_loss": result.first_loss,
                    "final_loss": result.final_loss,
                    "val_loss": result.val_loss,
                    "steps": result.steps,
                    "seed": result.seed,
                    "elapsed_s": result.elapsed_s,
                    "base": base,
                },
            )
        payload = _jsonable(
            {
                "model_token": token,
                "first_loss": result.first_loss,
                "final_loss": result.final_loss,
                "val_loss": result.val_loss,
                "steps": result.steps,
                "seed": result.seed,
                "elapsed_s": result.elapsed_s,
                "n_tokens": int(stream.size),
                "vocab_size": vocab.budget_size,
                "vocab_rows": vocab.rows,
                "param_count": model.n_parameters(),
                "sample": sample,
                "history": result.history,
            }
        )
        CacheStore().put(
            key,
            {"schema_version": SCHEMA_VERSION, "artifact_type": _TRAIN_ARTIFACT},
            {"result": payload},
            {},
        )
        registry.finish(job_id, result={k: v for k, v in payload.items() if k != "history"})
    except LLMGeometryError as exc:
        registry.fail(job_id, {"type": exc.error_type, "message": exc.message})
    except Exception as exc:  # noqa: BLE001 — surfaced verbatim via the job error event
        registry.fail(job_id, {"type": "TrainingFailedError", "message": str(exc)})


# -- GET /api/lex/spectrum -----------------------------------------------------------------


@router.get("/spectrum")
def spectrum_route(
    model_token: str,
    matrix: str = "embedding",
    baseline: bool = True,
    baseline_seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """The geometry of a trained model's embedding (FR-620..FR-623).

    `baseline=true` (the default) also computes the spectrum of an **untrained** model at
    the same shape. That is not decoration: effective rank climbs with `|V|` for random
    matrices too, so a trained curve without the random control cannot distinguish
    learning from arithmetic (FR-622, SC-604).
    """
    if matrix not in ("embedding", "readout"):
        raise InvalidParamError(f'matrix must be "embedding" or "readout", got {matrix!r}')
    model, vocab, meta = _load_model(model_token)
    tied = bool(meta["config"]["tied"])
    if matrix == "readout" and tied:
        raise InvalidParamError(
            "this model is tied: its readout IS its embedding, so it has exactly one "
            'spectrum. Request matrix="embedding" and label it tied.'
        )

    trained = spectrum(model_matrix(model, matrix))
    payload: dict[str, Any] = {
        "model_token": model_token,
        "matrix": matrix,
        "tied": tied,
        "projection": "pca",  # FR-623: these coordinates are a PROJECTION, and say so
        "display_k": SPECTRUM_DISPLAY_K,
        "tokens": list(vocab.itos),
        "spectrum": trained.as_dict(),
    }
    if baseline:
        from ..lex.model import LexConfig

        cfg = LexConfig.from_dict(dict(meta["config"]))
        untrained = random_baseline_spectrum(
            cfg, seed=baseline_seed, which="embedding" if tied else matrix
        )
        payload["baseline"] = untrained.summary()
        payload["comparison"] = compare_to_baseline(trained, untrained)
    return _jsonable(payload)


# -- POST /api/lex/generate ----------------------------------------------------------------


@router.post("/generate")
async def generate(request: Request) -> dict[str, Any]:
    """Generate text from a trained model (FR-605, SC-602).

    Every generated word is in-budget *by construction* — the vocabulary IS the budget
    and `<unk>`/`<bos>`/`<pad>` are masked — so `out_of_budget` below is always empty.
    It is returned anyway, and checked here rather than assumed, because that guarantee
    is the tab's central claim and a claim nobody verifies is a claim nobody can trust.
    """
    payload = await _json_body(request)
    token = payload.get("model_token")
    if not token:
        raise InvalidParamError("model_token is required")
    model, vocab, meta = _load_model(str(token))

    temperature = _as_float(payload.get("temperature", DEFAULT_TEMPERATURE), "temperature")
    if temperature < 0:
        raise InvalidParamError(f"temperature must be >= 0, got {temperature}")
    max_new_tokens = _as_int(
        payload.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS), "max_new_tokens"
    )
    if not 1 <= max_new_tokens <= MAX_NEW_TOKENS:
        raise InvalidParamError(
            f"max_new_tokens must be in 1..{MAX_NEW_TOKENS}, got {max_new_tokens}"
        )
    seed = _as_int(payload.get("seed", DEFAULT_SEED), "seed")
    prompt = str(payload.get("prompt", ""))
    stop_at_eos = _as_bool(payload.get("stop_at_eos", False), "stop_at_eos")

    from ..lex.generate import generate_text

    with TORCH_GLOBAL_LOCK:
        text = generate_text(
            model,
            vocab,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            seed=seed,
            stop_at_eos=stop_at_eos,
        )

    prompt_tokens = tokenize(prompt)
    in_budget = set(vocab.words)
    produced = tokenize(text)
    out_of_budget = sorted({w for w in produced if w not in in_budget})
    if out_of_budget:
        # Unreachable by construction; if it ever happens the guarantee is broken and the
        # right answer is a loud 500, never a quietly filtered string.
        raise ComputeError(
            "generated text contains out-of-budget words "
            f"{out_of_budget[:10]} — the in-budget-by-construction guarantee is broken"
        )
    return _jsonable(
        {
            "model_token": token,
            "prompt": prompt,
            "text": text,
            "words": produced,
            "n_words": len(produced),
            "out_of_budget": out_of_budget,
            "prompt_tokens": [
                {"text": word, "id": vocab.stoi.get(word, 0), "unk": word not in in_budget}
                for word in prompt_tokens
            ],
            "temperature": temperature,
            "seed": seed,
            "vocab_size": vocab.budget_size,
            "final_loss": meta["metrics"].get("final_loss"),
        }
    )


# -- GET/POST /api/lex/model (portable save + load) -----------------------------------------


def _b64(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(np.asarray(array, dtype="<f4")).tobytes()).decode(
        "ascii"
    )


def _unb64(data: str, shape: list[int]) -> np.ndarray:
    try:
        raw = base64.b64decode(data, validate=True)
    except Exception as exc:
        raise InvalidParamError(f"weight payload is not valid base64: {exc}")
    expected = int(np.prod(shape)) * 4
    if len(raw) != expected:
        raise InvalidParamError(
            f"weight payload is {len(raw)} bytes but shape {shape} needs {expected}"
        )
    return np.frombuffer(raw, dtype="<f4").reshape(shape).astype(np.float32)


@router.get("/model")
def export_model(model_token: str) -> dict[str, Any]:
    """The whole model as one portable, self-describing bundle (US-8, SC-607).

    A `model_token` is a content hash that only means something to the cache that minted
    it. The bundle is the form that survives a reload, another machine, and next week:
    weights, the vocabulary that gives its ids meaning, and the config that gives its
    weights a shape.
    """
    model, vocab, meta = _load_model(model_token)
    weights = model.weight_dict()
    return {
        "format": BUNDLE_FORMAT,
        "version": BUNDLE_VERSION,
        "model_token": model_token,
        # The two half-digests. `model_token` proves the halves belong together; these say
        # WHICH half is wrong, and let a weight set be recognised across vocabularies. All
        # three are mandatory on load, in this build and in the browser's.
        "weights_token": _weights_token(weights),
        "vocab_sha256": _vocab_digest(vocab.words, vocab.source, vocab.budget_name),
        "config": dict(meta["config"]),
        "vocab": {
            "source": vocab.source,
            "budget": vocab.budget_name,
            "words": list(vocab.words),
            "specials": list(SPECIAL_TOKENS),
        },
        "metrics": dict(meta.get("metrics", {})),
        "weights": {
            name: {"shape": list(array.shape), "data": _b64(array)}
            for name, array in weights.items()
        },
    }


@router.post("/model")
async def import_model(request: Request) -> dict[str, Any]:
    """Load a bundle written by GET /api/lex/model; returns its `model_token`.

    Validation is strict and loud on purpose. `tied` travels in the bundle and is checked
    against the weight names it implies, because the source project's `probe.py` dropped
    `tie` on reload and silently reloaded a tied checkpoint as an untied model — the
    reloaded thing was then not the model that was saved.
    """
    payload = await _json_body(request)
    if payload.get("format") != BUNDLE_FORMAT:
        raise InvalidParamError(
            f"not a Lexicon Lab model bundle: format is {payload.get('format')!r}, "
            f"expected {BUNDLE_FORMAT!r}"
        )
    if _as_int(payload.get("version", 0), "version") != BUNDLE_VERSION:
        raise InvalidParamError(
            f"bundle version {payload.get('version')!r} is not supported "
            f"(this build reads version {BUNDLE_VERSION})"
        )
    for field in ("config", "vocab", "weights"):
        if not isinstance(payload.get(field), dict):
            raise InvalidParamError(f"bundle is missing its {field!r} object")

    from ..lex.model import LexConfig, model_from_weight_dict

    config = LexConfig.from_dict(dict(payload["config"]))
    words = payload["vocab"].get("words")
    if not isinstance(words, list) or not all(isinstance(w, str) for w in words):
        raise InvalidParamError("bundle vocab.words must be a list of strings")
    vocab = LexVocab(
        words=tuple(words),
        source=str(payload["vocab"].get("source", "dolch")),
        budget_name=str(payload["vocab"].get("budget", "custom")),
    )
    if vocab.rows != config.vocab_rows:
        raise InvalidParamError(
            f"bundle vocabulary has {vocab.rows} rows but its config declares "
            f"{config.vocab_rows}; the weights and the word list do not describe one model"
        )

    weights: dict[str, np.ndarray] = {}
    for name, entry in payload["weights"].items():
        if not isinstance(entry, dict) or "shape" not in entry or "data" not in entry:
            raise InvalidParamError(f"weight {name!r} must be an object with shape and data")
        shape = [_as_int(v, f"{name}.shape") for v in entry["shape"]]
        weights[str(name)] = _unb64(str(entry["data"]), shape)

    # `load_weight_dict` rejects a missing/extra `head_w`, which is exactly the tied-flag
    # check: a tied bundle carrying a readout, or an untied one missing it, cannot load.
    model = model_from_weight_dict(config, weights)

    # Integrity is MANDATORY, not opt-in. Treating a missing digest as "nothing to check"
    # is exactly the hole feature 004 shipped in the Geometry Lab: a tampered file loaded
    # cleanly the moment you DELETED the field rather than edited it.
    def _declared(field: str, hex_len: int) -> str:
        value = payload.get(field)
        if (
            not isinstance(value, str)
            or len(value) != hex_len
            or any(c not in "0123456789abcdef" for c in value)
        ):
            raise InvalidParamError(
                f"bundle has no usable {field!r}, so its contents cannot be verified — "
                "refusing to load it. Re-export the model to get a valid file."
            )
        return value

    weight_dict = model.weight_dict()
    recomputed = _model_token(weight_dict, config.as_dict(), vocab.words)
    declared = _declared("model_token", 32)
    if declared != recomputed:
        raise InvalidParamError(
            f"bundle declares model_token {declared!r} but its own contents hash to "
            f"{recomputed!r}; refusing to load a file whose weights and label disagree"
        )
    declared_weights = _declared("weights_token", 32)
    actual_weights = _weights_token(weight_dict)
    if declared_weights != actual_weights:
        raise InvalidParamError(
            f"this model file is corrupt: its weights hash to {actual_weights} but it "
            f"declares {declared_weights}. Loading it would pair the wrong vocabulary "
            "with these weights, so it is refused."
        )
    declared_vocab = _declared("vocab_sha256", 64)
    actual_vocab = _vocab_digest(vocab.words, vocab.source, vocab.budget_name)
    if declared_vocab != actual_vocab:
        raise InvalidParamError(
            f"this model file is corrupt: its vocabulary hashes to {actual_vocab[:16]}… "
            f"but it declares {declared_vocab[:16]}…. Loading it would label every token "
            "with the wrong word, so it is refused."
        )
    metrics = payload.get("metrics")
    token = _save_model(
        weights=model.weight_dict(),
        config=config.as_dict(),
        vocab=vocab,
        metrics=dict(metrics) if isinstance(metrics, dict) else {},
    )
    return _jsonable(
        {
            "model_token": token,
            "config": config.as_dict(),
            "vocab_size": vocab.budget_size,
            "vocab_rows": vocab.rows,
            "param_count": model.n_parameters(),
        }
    )
