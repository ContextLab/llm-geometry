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
from typing import Any

import numpy as np

from ..cache.store import CacheStore
from ..errors import InvalidParamError
from .config import CONTEXT_WINDOW, D_MODEL, MLP_HIDDEN, N_HEADS, N_LAYERS, VOCAB_SIZE
from .tokenizer import GeoTokenizer
from .train import resolve_weight_set
from .weights import load_weight_set_vocab, save_weight_set, weights_token

BUNDLE_FORMAT = "llm-geometry/geo-model"
BUNDLE_VERSION = 1

_EXPECTED_CONFIG = {
    "d_model": D_MODEL,
    "n_layers": N_LAYERS,
    "n_heads": N_HEADS,
    "mlp_hidden": MLP_HIDDEN,
    "vocab_size": VOCAB_SIZE,
    "context_window": CONTEXT_WINDOW,
}


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
        from .tokenizer import get_tokenizer

        vocab_json = get_tokenizer().to_json()
    return {
        "format": BUNDLE_FORMAT,
        "version": BUNDLE_VERSION,
        "weights_token": real_token,
        "config": dict(_EXPECTED_CONFIG),
        "vocab": vocab_json,
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
            f"(this build reads version {BUNDLE_VERSION})"
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

    declared = bundle.get("weights_token")
    actual = weights_token(ws)
    if declared is not None and declared != actual:
        raise InvalidParamError(
            "this model file is corrupt: its weights hash to "
            f"{actual} but it declares {declared}. Loading it would pair the wrong "
            "vocabulary with these weights, so it is refused."
        )

    vocab_json = bundle.get("vocab")
    if not isinstance(vocab_json, str):
        raise InvalidParamError("model file is missing its `vocab` block")
    tokenizer = GeoTokenizer.from_json(vocab_json)  # raises on a malformed vocabulary

    token = save_weight_set(ws, source="imported", store=store, vocab_json=vocab_json)
    return {"weights_token": token, "vocab_size": len(tokenizer.id_to_text)}
