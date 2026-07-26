"""Train a brand-new GeoTransformer from scratch on the user's own text.

This is NOT the fine-tune path. Fine-tuning starts from the canonical checkpoint and
keeps the shipped ~1000-word vocabulary, so text about anything the corpus doesn't
cover mostly turns into ``<unk>``. Training from scratch builds a **fresh vocabulary
from the supplied text** and freshly initialized embeddings, so the resulting model is
genuinely a different model — its token ids mean different words.

Because the ids mean different words, the vocabulary is stored WITH the weights (see
``weights.save_weight_set(vocab_json=…)``) and every read path that turns ids back
into text resolves the tokenizer from the model's token rather than a global.

The vocabulary size is fixed (``VOCAB_SIZE``) because it is a model dimension, so the
text must contain at least ``VOCAB_WORDS`` distinct token types. Short text is refused
with a plain-language error naming the shortfall rather than silently padding.
"""

from __future__ import annotations

import hashlib
from typing import Any, Callable

import numpy as np

from ..cache.keys import make_cache_key
from ..cache.store import CacheStore
from ..errors import InvalidParamError
from .config import (
    CONTEXT_WINDOW,
    EOS_ID,
    SEED,
    VOCAB_SIZE,
    VOCAB_WORDS,
)
from .finetune import load_text_from_hf
from .tokenizer import GeoTokenizer, split_words
from .train import train_geo_model
from .weights import save_weight_set

ProgressCb = Callable[[float, str], None]

_ARTIFACT_TYPE = "geo-scratch"

#: Epoch budget for a from-scratch run. Lower than the canonical checkpoint's because
#: user corpora are typically much smaller and this runs interactively.
SCRATCH_DEFAULT_EPOCHS = 12
SCRATCH_MAX_EPOCHS = 60
#: Records pulled when the corpus comes from a HuggingFace dataset.
SCRATCH_DEFAULT_MAX_SAMPLES = 2000


def scratch_cache_key(text: str, epochs: int, seed: int) -> tuple[str, dict[str, Any]]:
    text_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return make_cache_key(
        model_id="geo-transformer",
        revision=None,
        artifact_type=_ARTIFACT_TYPE,
        inputs={"text_sha256": text_sha},
        params={"epochs": int(epochs)},
        seed=seed,
    )


def corpus_stats(text: str) -> dict[str, int]:
    """Token count and distinct-type count for a candidate corpus.

    The UI shows these BEFORE training so the vocabulary requirement is visible up
    front instead of arriving as a failure after a long wait.
    """
    words = split_words(text)
    return {
        "n_tokens": len(words),
        "n_distinct": len(set(words)),
        "vocab_words_required": VOCAB_WORDS,
    }


def train_scratch(
    *,
    text: str | None = None,
    hf_dataset: str | None = None,
    hf_split: str = "train",
    max_samples: int = SCRATCH_DEFAULT_MAX_SAMPLES,
    epochs: int = SCRATCH_DEFAULT_EPOCHS,
    seed: int = SEED,
    progress_cb: ProgressCb | None = None,
    store: CacheStore | None = None,
) -> dict[str, Any]:
    """Train a new model on ``text`` (or a streamed HF dataset).

    Returns ``{"weights_token", "vocab_size", "final_loss", "n_tokens", "n_distinct",
    "epochs", "cached"}``. The canonical checkpoint is never touched.
    """
    if (text is None) == (hf_dataset is None):
        raise InvalidParamError("exactly one of text/hf_dataset must be provided")
    epochs = int(epochs)
    if not 1 <= epochs <= SCRATCH_MAX_EPOCHS:
        raise InvalidParamError(f"epochs must be in 1..{SCRATCH_MAX_EPOCHS}, got {epochs!r}")

    if hf_dataset is not None:
        if progress_cb is not None:
            progress_cb(0.0, f"streaming {hf_dataset} ({hf_split})")
        text = load_text_from_hf(hf_dataset, split=hf_split, max_samples=max_samples)
    assert text is not None
    if not text.strip():
        raise InvalidParamError("training text is empty")

    stats = corpus_stats(text)
    if stats["n_distinct"] < VOCAB_WORDS:
        raise InvalidParamError(
            f"This text has only {stats['n_distinct']} distinct word types, and the "
            f"model's vocabulary is {VOCAB_WORDS} words wide — training it would leave "
            "most of the vocabulary undefined. Paste more text (a few pages of prose), "
            "or point at a larger HuggingFace dataset."
        )

    store = store or CacheStore()
    key, _spec = scratch_cache_key(text, epochs, seed)
    entry = store.get(key)
    if entry is not None:
        return {**entry["meta"], "cached": True}

    if progress_cb is not None:
        progress_cb(0.02, f"building a {VOCAB_SIZE}-token vocabulary from your text")
    tokenizer = GeoTokenizer.from_corpus_text(text)
    ids = tokenizer.encode_stream(text)
    if len(ids) < CONTEXT_WINDOW:
        raise InvalidParamError(
            f"training text is too short after tokenization ({len(ids)} tokens; "
            f"at least {CONTEXT_WINDOW} are needed for one training window)"
        )
    stream = np.asarray(ids + [EOS_ID], dtype=np.int64)

    def training_progress(frac: float, message: str) -> None:
        if progress_cb is not None:
            progress_cb(0.05 + 0.93 * frac, message)

    ws, final_loss = train_geo_model(
        seed=seed,
        epochs=epochs,
        progress_cb=training_progress,
        corpus_stream=stream,
    )

    vocab_json = tokenizer.to_json()
    token = save_weight_set(ws, source="scratch", store=store, vocab_json=vocab_json)
    meta = {
        "weights_token": token,
        "vocab_size": VOCAB_SIZE,
        "final_loss": float(final_loss),
        "n_tokens": int(stats["n_tokens"]),
        "n_distinct": int(stats["n_distinct"]),
        "epochs": epochs,
        "seed": int(seed),
    }
    store.put(key, _spec, meta, {"embedding": np.asarray(ws["embedding"], dtype=np.float32)})
    if progress_cb is not None:
        progress_cb(1.0, f"trained · final loss {final_loss:.2f}")
    return {**meta, "cached": False}
