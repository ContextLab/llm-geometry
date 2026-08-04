"""Save and load a Geometry Lab model as a single self-describing file.

A ``weights_token`` is a content hash that only means something to the cache that
minted it — reload the page next week, or open the site on another machine, and an
edited or scratch-trained model is gone. A bundle is the portable form: weights AND
the vocabulary that gives its token ids meaning, plus enough provenance to know what
you are looking at.

Validation is strict and loud. A bundle whose declared ``weights_token`` does not
match a re-hash of its own weights is rejected rather than loaded, because a silent
mismatch would attach the wrong vocabulary to a set of weights and every label in the
UI would then be quietly wrong.
"""

from __future__ import annotations

import base64
import hashlib
from typing import Any

import numpy as np

from ..cache.store import CacheStore
from ..errors import InvalidParamError
from .config import CONTEXT_WINDOW, D_MODEL, MLP_HIDDEN, N_HEADS, N_LAYERS, VOCAB_SIZE
from .tokenizer import GeoTokenizer
from .train import resolve_weight_set
from .weights import (
    load_weight_set_vocab,
    save_weight_set,
    validate_weight_set,
    weight_set_owns_vocab,
    weights_token,
)

BUNDLE_FORMAT = "llm-geometry/geo-model"
BUNDLE_VERSION = 2  # v1 had no vocabulary integrity check (see import_bundle)

_EXPECTED_CONFIG = {
    "d_model": D_MODEL,
    "n_layers": N_LAYERS,
    "n_heads": N_HEADS,
    "mlp_hidden": MLP_HIDDEN,
    "vocab_size": VOCAB_SIZE,
    "context_window": CONTEXT_WINDOW,
}


def vocab_digest(vocab_json: str) -> str:
    """SHA-256 of the vocabulary JSON.

    The weights hash cannot cover the vocabulary, so without this a file could carry
    intact weights and a FABRICATED word list and load cleanly — every label in the UI
    would then be confidently wrong, which is exactly what this module exists to stop.
    """
    return hashlib.sha256(vocab_json.encode("utf-8")).hexdigest()


def _b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(np.asarray(arr, dtype="<f4")).tobytes()).decode(
        "ascii"
    )


def _unb64(data: str, shape: list[int]) -> np.ndarray:
    raw = base64.b64decode(data, validate=True)
    expected = int(np.prod(shape)) * 4
    if len(raw) != expected:
        raise InvalidParamError(
            f"weight payload is {len(raw)} bytes but shape {shape} needs {expected}"
        )
    return np.frombuffer(raw, dtype="<f4").reshape(shape).astype(np.float32)


def export_bundle(token: str, store: CacheStore | None = None) -> dict[str, Any]:
    """Build the portable bundle for ``token`` ("learned" resolves the canonical one)."""
    store = store or CacheStore()
    ws = resolve_weight_set(token, store=store)
    real_token = weights_token(ws)
    vocab_json = None if token == "learned" else load_weight_set_vocab(token, store=store)
    if vocab_json is None:
        # Falling back to the canonical vocabulary is RIGHT for anything descended from
        # the shipped checkpoint and CATASTROPHIC for a model whose ids mean its own
        # words: the file would carry these weights under Alice in Wonderland's word
        # list, with `vocab_sha256` computed over that list, so it would verify and no
        # reader could ever detect it. Refuse instead — the same guard the TS engine's
        # `exportBundle` already had.
        if token != "learned" and weight_set_owns_vocab(token, store=store):
            raise InvalidParamError(
                f"weights_token {token!r} has no vocabulary stored beside it, and its "
                "ids mean its own words rather than the shipped model's — saving it "
                "now would pair these weights with the wrong word list. Load the model "
                "file again (or retrain) so its vocabulary is present."
            )
        from .tokenizer import get_tokenizer

        vocab_json = get_tokenizer().to_json()
    return {
        "format": BUNDLE_FORMAT,
        "version": BUNDLE_VERSION,
        "weights_token": real_token,
        "config": dict(_EXPECTED_CONFIG),
        "vocab": vocab_json,
        "vocab_sha256": vocab_digest(vocab_json),
        "weights": {
            name: {"shape": list(np.asarray(arr).shape), "data": _b64(arr)}
            for name, arr in sorted(ws.items())
        },
    }


def import_bundle(bundle: Any, store: CacheStore | None = None) -> dict[str, Any]:
    """Validate a bundle and store its model; return ``{"weights_token", "vocab_size"}``."""
    if not isinstance(bundle, dict):
        raise InvalidParamError("a model file must be a JSON object")
    if bundle.get("format") != BUNDLE_FORMAT:
        raise InvalidParamError(
            f"not a Geometry Lab model file (format={bundle.get('format')!r}, "
            f"expected {BUNDLE_FORMAT!r})"
        )
    if bundle.get("version") != BUNDLE_VERSION:
        raise InvalidParamError(
            f"model file version {bundle.get('version')!r} is not supported "
            f"(this build reads version {BUNDLE_VERSION}). Version 1 files carried no "
            "vocabulary integrity check; re-export the model to get a v2 file."
        )

    config = bundle.get("config")
    if not isinstance(config, dict):
        raise InvalidParamError("model file is missing its `config` block")
    for field, expected in _EXPECTED_CONFIG.items():
        if config.get(field) != expected:
            raise InvalidParamError(
                f"model file was built for {field}={config.get(field)!r}, but this "
                f"build's GeoTransformer is {field}={expected} — they are different "
                "architectures, so the weights cannot be loaded"
            )

    raw_weights = bundle.get("weights")
    if not isinstance(raw_weights, dict) or not raw_weights:
        raise InvalidParamError("model file carries no weights")
    ws: dict[str, np.ndarray] = {}
    for name, payload in raw_weights.items():
        if not isinstance(payload, dict) or "shape" not in payload or "data" not in payload:
            raise InvalidParamError(f"weight {name!r} is malformed (need shape + data)")
        try:
            ws[name] = _unb64(str(payload["data"]), [int(d) for d in payload["shape"]])
        except InvalidParamError:
            raise
        except Exception as exc:
            raise InvalidParamError(f"weight {name!r} could not be decoded: {exc}")

    # Completeness and shapes, BEFORE the hash check — exactly what the TS engine's
    # `validateWeightSet` already did. Hashing says only "these bytes are the bytes
    # this file declares"; it says nothing about whether they form a model. A file
    # carrying one tensor, with every digest honestly recomputed over that one tensor,
    # used to load with a 200 and then surface as a 500 whose whole message was the
    # bare string 'layers.0.W_V'.
    validate_weight_set(ws, context="model file")

    # Integrity is MANDATORY, not opt-in. Treating a missing `weights_token` as "nothing
    # to check" let a file with tampered weights load silently just by deleting a field.
    declared = bundle.get("weights_token")
    if not isinstance(declared, str) or not declared:
        raise InvalidParamError(
            "model file has no `weights_token`, so its contents cannot be verified — "
            "refusing to load it. Re-export the model to get a valid file."
        )
    actual = weights_token(ws)
    if declared != actual:
        raise InvalidParamError(
            "this model file is corrupt: its weights hash to "
            f"{actual} but it declares {declared}. Loading it would pair the wrong "
            "vocabulary with these weights, so it is refused."
        )

    vocab_json = bundle.get("vocab")
    if not isinstance(vocab_json, str):
        raise InvalidParamError("model file is missing its `vocab` block")
    # The weights hash says nothing about the vocabulary, so the vocabulary carries its
    # own digest. Without it a file with genuine weights and an invented word list
    # loaded cleanly and mislabelled every token on screen.
    declared_vocab = bundle.get("vocab_sha256")
    if not isinstance(declared_vocab, str) or not declared_vocab:
        raise InvalidParamError(
            "model file has no `vocab_sha256`, so its vocabulary cannot be verified — "
            "refusing to load it. Re-export the model to get a valid file."
        )
    actual_vocab = vocab_digest(vocab_json)
    if declared_vocab != actual_vocab:
        raise InvalidParamError(
            "this model file is corrupt: its vocabulary hashes to "
            f"{actual_vocab[:16]}… but it declares {declared_vocab[:16]}…. Loading it "
            "would label every token with the wrong word, so it is refused."
        )
    tokenizer = GeoTokenizer.from_json(vocab_json)  # raises on a malformed vocabulary

    # An imported model always owns its word list: the file carried one, and that is
    # what the ids in these weights mean.
    token = save_weight_set(
        ws, source="imported", store=store, vocab_json=vocab_json, owns_vocab=True
    )
    return {"weights_token": token, "vocab_size": len(tokenizer.id_to_text)}
