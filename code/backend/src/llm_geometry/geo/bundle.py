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
    own_vocab_json,
    save_weight_set,
    validate_weight_set,
    weight_set_owns_vocab,
    weights_token,
)

BUNDLE_FORMAT = "llm-geometry/geo-model"
# v1 had no vocabulary integrity check; v2 named a model by a hash of its WEIGHTS ALONE.
# `weights_token` now hashes the word list too (see `weights.weights_token`), which is a
# change to what the `weights_token` FIELD IN THE FILE means, so the file format moved with
# it. Leaving it at 2 made every pre-change file with its own word list fail the re-hash and
# be reported as "this model file is corrupt" — an accusation against an intact file, and
# against the exact file the cache's schema-bump message tells the user to open.
BUNDLE_VERSION = 3
# The version whose `weights_token` covers the weights only. Such a file is READ, not
# refused: see the migration in `import_bundle`.
LEGACY_WEIGHTS_ONLY_VERSION = 2
SUPPORTED_VERSIONS = (LEGACY_WEIGHTS_ONLY_VERSION, BUNDLE_VERSION)

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
    # The file's identity covers its word list (`weights_token`), so this re-hash must
    # reproduce the token the store minted. If it does not, the store and the file
    # disagree about which model this is — say so rather than writing the file.
    real_token = weights_token(ws, own_vocab_json(vocab_json))
    if token != "learned" and real_token != token:
        raise InvalidParamError(
            f"weights_token {token!r} does not match a re-hash of its own weights and "
            f"vocabulary ({real_token!r}) — the stored model is inconsistent, so saving "
            "it would produce a file that names the wrong model. Retrain or reload it."
        )
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
    version = bundle.get("version")
    if version not in SUPPORTED_VERSIONS:
        raise InvalidParamError(
            f"model file version {version!r} is not supported (this build reads versions "
            f"{LEGACY_WEIGHTS_ONLY_VERSION} and {BUNDLE_VERSION}). Version 1 files carried "
            f"no vocabulary integrity check; re-export the model to get a v{BUNDLE_VERSION} "
            "file."
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
    # Hash and store the CANONICAL serialization of what the file carried, not the file's
    # own bytes: `vocab_sha256` is about the bytes on disk, but the model's identity must
    # not depend on whether a writer emitted its keys in a different order. The browser
    # engine hashes `canonicalVocabJson(tokenizer.words)` for exactly the same reason.
    canonical_vocab = tokenizer.to_json()

    # Integrity is MANDATORY, not opt-in. Treating a missing `weights_token` as "nothing
    # to check" let a file with tampered weights load silently just by deleting a field.
    #
    # The hash covers the WORD LIST as well as the weights (see `weights.weights_token`),
    # so this check now also catches the attack the vocabulary digest cannot: swapping the
    # word list AND recomputing `vocab_sha256` over the substitute leaves a file that is
    # internally consistent but no longer hashes to the model it names. It runs after the
    # vocabulary is validated because the vocabulary is one of its inputs.
    declared = bundle.get("weights_token")
    if not isinstance(declared, str) or not declared:
        raise InvalidParamError(
            "model file has no `weights_token`, so its contents cannot be verified — "
            "refusing to load it. Re-export the model to get a valid file."
        )
    owned = own_vocab_json(canonical_vocab)
    actual = weights_token(ws, owned)
    legacy = weights_token(ws, None)
    if version == LEGACY_WEIGHTS_ONLY_VERSION:
        # MIGRATION, not a refusal. A v2 file names itself by a hash of its weights alone,
        # so it is checked against that hash and the current identity is re-derived from
        # the (weights, word list) pair it carries. Refusing instead would strand an intact
        # file — the user's only copy — and would buy nothing: the binding a v3 token gives
        # is absent from EVERY v2 file, including the ones that load today only because
        # their word list happens to be the shipped one and so takes no part in either
        # hash. What this format cannot prove — that these words are the words these
        # weights were trained with — it never could; that is what the bump records, not
        # something introduced by reading it.
        if declared != legacy:
            raise InvalidParamError(
                "this model file is corrupt: its weights hash to "
                f"{legacy} but it declares {declared}. Loading it would pair the wrong "
                "vocabulary with these weights, so it is refused."
            )
    elif declared != actual:
        # Deliberately NOT special-cased when ``declared == legacy``: a file carrying
        # weights, an own word list and a weights-only token is what a version-2 writer
        # produced AND what swapping a version-3 file's word list produces. The two are
        # indistinguishable, so a file that DECLARES version 3 is held to version 3.
        raise InvalidParamError(
            "this model file is corrupt: its weights and vocabulary hash to "
            f"{actual} but it declares {declared}. Loading it would pair the wrong "
            "vocabulary with these weights, so it is refused."
        )

    # An imported model owns the word list the file carried — unless that list IS the
    # shipped one, in which case there is nothing of its own to own and the model keeps
    # the canonical token it would have had anyway (`save_weight_set` normalizes this).
    token = save_weight_set(
        ws,
        source="imported",
        store=store,
        vocab_json=canonical_vocab,
        owns_vocab=owned is not None,
    )
    return {"weights_token": token, "vocab_size": len(tokenizer.id_to_text)}
