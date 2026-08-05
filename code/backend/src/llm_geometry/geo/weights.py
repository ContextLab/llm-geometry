"""Weight presets, weight-set editing, content-hash tokens, and persistence.

A *weight set* is the complete parameter dict of a GeoTransformer (see
``GeoTransformer.weight_names()``). Edits target the contract's addressable matrices
(W_Q/W_K/W_V/W_O per layer, each 3×3, and the 1003×3 embedding); everything else is
carried through from the base set unchanged.

``weights_token`` is a content hash over the *full resulting weight set* (float32
bytes, name-sorted): stateless, deduplicating, identical edits from an identical base
always mint the identical token. Weight sets persist through the shared
integrity-checked ``CacheStore`` so tokens stay valid across workers and restarts
(LRU-evicted with the rest of the cache; FR-104).

Preset semantics (3×3 unless noted):
  identity        — I₃. Embedding (1003×3): row *i* = eᵢ mod 3 (already unit-norm).
  toeplitz_fuzzy  — fuzzy-diagonal Toeplitz T[i,j] = exp(−(i−j)²/(2σ²)), σ=0.75.
                    Embedding: same formula, rows unit-normalized.
  random          — seeded 𝒩(0, 1/√cols). Embedding: seeded random unit vectors.
  random_autocorr — seeded white noise smoothed with a Gaussian filter (neighboring
                    entries correlated), rescaled to the `random` std.
                    Embedding: rows unit-normalized (a smooth path on the sphere).
  zero            — all zeros. Rejected for the embedding: zero rows cannot satisfy
                    the unit-norm constraint (InvalidWeightEditError, by design).
  learned         — the matrix from the canonical learned checkpoint.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter

from ..cache.store import CacheStore
from ..config import SCHEMA_VERSION
from ..errors import InvalidParamError, InvalidWeightEditError, NotFoundError
from .config import CONTEXT_WINDOW, D_MODEL, MAX_SEED, MLP_HIDDEN, N_LAYERS, VOCAB_SIZE

PRESETS = ("identity", "toeplitz_fuzzy", "random", "random_autocorr", "zero", "learned")
EDITABLE_MATRICES = ("W_Q", "W_K", "W_V", "W_O", "embedding")
_TOEPLITZ_SIGMA = 0.75
_ARTIFACT_PREFIX = "geo-weights"


def _unit_rows(mat: np.ndarray, context: str) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    if np.any(norms < 1e-8):
        raise InvalidWeightEditError(
            f"{context}: {int(np.sum(norms < 1e-8))} embedding row(s) have (near-)zero "
            "norm and cannot be unit-normalized"
        )
    return (mat / norms).astype(np.float32)


def _matrix_shape(name: str) -> tuple[int, int]:
    return (VOCAB_SIZE, D_MODEL) if name == "embedding" else (D_MODEL, D_MODEL)


def preset_matrix(preset: str, matrix: str, seed: int = 0) -> np.ndarray:
    """Build one preset matrix for ``matrix`` (a name from EDITABLE_MATRICES)."""
    if preset not in PRESETS:
        raise InvalidWeightEditError(f"Unknown preset {preset!r}; expected one of {PRESETS}")
    if matrix not in EDITABLE_MATRICES:
        raise InvalidWeightEditError(
            f"Unknown matrix {matrix!r}; expected one of {EDITABLE_MATRICES}"
        )
    rows, cols = _matrix_shape(matrix)
    is_embedding = matrix == "embedding"

    if preset == "identity":
        out = np.tile(np.eye(cols, dtype=np.float32), (rows // cols + 1, 1))[:rows]
        return out.astype(np.float32)
    if preset == "toeplitz_fuzzy":
        # Row i peaks at column i (mod cols): a plain fuzzy diagonal for square
        # matrices, a cyclically-repeated one for the tall embedding (matching the
        # identity preset's row-cycling semantics — otherwise rows far from the
        # 3-wide diagonal would be all-zero and violate the unit-norm constraint).
        i = (np.arange(rows) % cols)[:, None]
        j = np.arange(cols)[None, :]
        out = np.exp(-((i - j) ** 2) / (2.0 * _TOEPLITZ_SIGMA**2)).astype(np.float32)
        return _unit_rows(out, "toeplitz_fuzzy embedding") if is_embedding else out
    if preset == "random":
        rng = np.random.default_rng(seed)
        out = (rng.standard_normal((rows, cols)) / np.sqrt(cols)).astype(np.float32)
        return _unit_rows(out, "random embedding") if is_embedding else out
    if preset == "random_autocorr":
        rng = np.random.default_rng(seed)
        noise = rng.standard_normal((rows, cols))
        sigma = (8.0, 0.0) if is_embedding else (0.8, 0.8)  # embedding: correlate down rows
        smooth = gaussian_filter(noise, sigma=sigma, mode="wrap")
        std = smooth.std()
        if std < 1e-12:
            raise InvalidWeightEditError("random_autocorr produced a degenerate (constant) matrix")
        out = (smooth / std / np.sqrt(cols)).astype(np.float32)
        return _unit_rows(out, "random_autocorr embedding") if is_embedding else out
    if preset == "zero":
        if is_embedding:
            raise InvalidWeightEditError(
                "preset 'zero' is invalid for the embedding: zero rows cannot satisfy "
                "the unit-norm constraint"
            )
        return np.zeros((rows, cols), dtype=np.float32)
    # preset == "learned": the canonical checkpoint's matrix.
    from .train import load_canonical_weight_set  # local import to avoid a cycle

    ws = load_canonical_weight_set()
    key = "embedding" if is_embedding else None
    if key is not None:
        return np.asarray(ws["embedding"], dtype=np.float32).copy()
    raise InvalidWeightEditError(
        "preset 'learned' for a layer matrix requires the layer; use build_weight_set"
    )


def _learned_embedding() -> np.ndarray:
    """The canonical embedding, verbatim (already unit-norm — do NOT re-normalize)."""
    from .train import load_canonical_weight_set  # local import to avoid a cycle

    ws = load_canonical_weight_set()
    return np.asarray(ws["embedding"], dtype=np.float32).copy()


def _learned_layer_matrix(layer: int, matrix: str) -> np.ndarray:
    from .train import load_canonical_weight_set  # local import to avoid a cycle

    ws = load_canonical_weight_set()
    return np.asarray(ws[f"layers.{layer}.{matrix}"], dtype=np.float32).copy()


def validate_values(matrix: str, values: object) -> np.ndarray:
    try:
        arr = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise InvalidWeightEditError(f"values for {matrix!r} are not a numeric matrix: {exc}")
    expected = _matrix_shape(matrix)
    if arr.shape != expected:
        raise InvalidWeightEditError(
            f"values for {matrix!r} have shape {tuple(arr.shape)}, expected {expected}"
        )
    if not np.all(np.isfinite(arr)):
        raise InvalidWeightEditError(f"values for {matrix!r} contain non-finite entries")
    return arr


def _edit_seed(value: object, n: int) -> int:
    """An edit's seed as an integer in ``0..MAX_SEED``, or a typed refusal.

    ``int(value or 0)`` accepted everything JSON can express and quietly rewrote it:
    ``1.5`` selected the seed-1 preset, ``"7"`` and ``True`` selected other people's
    matrices, ``-1`` reached ``np.random.default_rng`` and came back as an untyped 500,
    and ``Infinity`` raised ``OverflowError`` — also a 500. A seed picks WHICH matrix you
    get, so a coerced one is a different matrix than the one requested, reported as though
    it were the one requested. Same rule as ``api.routes_geo._as_int``.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidWeightEditError(f"edit {n}: seed must be an integer, got {value!r}")
    if isinstance(value, float):
        if not np.isfinite(value):
            raise InvalidWeightEditError(f"edit {n}: seed must be a finite integer, got {value!r}")
        if not float(value).is_integer():
            raise InvalidWeightEditError(
                f"edit {n}: seed must be an integer, got {value!r} — it is not rounded or "
                "truncated, because a seed that is not the seed you asked for silently "
                "selects a different matrix"
            )
        value = int(value)
    seed = int(value)
    if not 0 <= seed <= MAX_SEED:
        raise InvalidWeightEditError(f"edit {n}: seed must be in 0..{MAX_SEED}, got {seed}")
    return seed


def build_weight_set(
    base: dict[str, np.ndarray], edits: list[dict]
) -> tuple[dict[str, np.ndarray], list[dict]]:
    """Apply ``edits`` to a copy of ``base``; return (new weight set, edit summaries).

    Each edit is ``{"layer": int, "matrix": str, "preset": str|None,
    "values": nested lists|None, "seed": int}`` with exactly one of preset/values
    (contract POST /api/geo/weights). The embedding's unit-norm invariant is enforced
    after every embedding edit.
    """
    ws = {name: np.asarray(arr, dtype=np.float32).copy() for name, arr in base.items()}
    summaries: list[dict] = []
    for n, edit in enumerate(edits):
        matrix = edit.get("matrix")
        if matrix not in EDITABLE_MATRICES:
            raise InvalidWeightEditError(
                f"edit {n}: unknown matrix {matrix!r}; expected one of {EDITABLE_MATRICES}"
            )
        preset = edit.get("preset")
        values = edit.get("values")
        if (preset is None) == (values is None):
            raise InvalidWeightEditError(
                f"edit {n} ({matrix}): exactly one of preset/values must be given"
            )
        seed = _edit_seed(edit.get("seed", 0), n)
        layer = edit.get("layer")
        if matrix != "embedding":
            if (
                layer is None
                or isinstance(layer, bool)
                or not (isinstance(layer, int) and 0 <= layer < N_LAYERS)
            ):
                raise InvalidWeightEditError(
                    f"edit {n} ({matrix}): layer must be an int in 0..{N_LAYERS - 1}, "
                    f"got {layer!r}"
                )

        if values is not None:
            arr = validate_values(matrix, values)
            source = "edited"
        elif preset == "learned":
            # The embedding used to be excluded here, so "learned" fell through to
            # preset_matrix + _unit_rows and came back ~1e-6 off the canonical rows.
            # The content hash then never matched the checkpoint, and the UI was stuck
            # reporting "hand-edited weights" forever for a model that WAS the shipped
            # one. Restore the canonical array verbatim instead.
            arr = (
                _learned_embedding()
                if matrix == "embedding"
                else _learned_layer_matrix(int(layer), matrix)
            )
            source = "preset:learned"
        else:
            arr = preset_matrix(str(preset), matrix, seed=seed)
            source = f"preset:{preset}"

        if matrix == "embedding":
            # Already unit-norm when it came straight from the checkpoint; re-normalizing
            # in float32 is what introduced the drift.
            if source != "preset:learned":
                arr = _unit_rows(arr, f"edit {n} (embedding)")
            ws["embedding"] = arr
        else:
            ws[f"layers.{int(layer)}.{matrix}"] = arr
        summaries.append({"layer": layer, "matrix": matrix, "source": source})
    return ws, summaries


# -- content-hash tokens + persistence -------------------------------------------------


#: Domain separator between the weight bytes and the vocabulary bytes in a token.
#: A literal that cannot occur in a tensor name, so no weight set can be confused with
#: a weight set plus a word list.
_VOCAB_HASH_TAG = b"\x00geo-vocab-v1\x00"


def weights_token(ws: dict[str, np.ndarray], vocab_json: str | None = None) -> str:
    """Content hash over the full model: name-sorted float32 bytes, then its vocabulary.

    ``vocab_json`` is the *canonical* tokenizer serialization (``GeoTokenizer.to_json``)
    for a model whose token ids mean words of its OWN, and ``None`` for one that reads
    under the shipped vocabulary. Passing ``None`` reproduces the original
    weights-only hash byte for byte, so the canonical checkpoint's id never moves.

    **The word list is part of the model's identity, not metadata beside it.** Two weight
    sets with identical numbers and different vocabularies are two different models: the
    same id means `qalokemu` in one and `the` in the other. While the hash covered only
    the weights, they collided — and because the store deduplicates on that hash and wrote
    the metadata first-write-wins, whichever vocabulary arrived first was the one BOTH
    models were saved under, with every digest recomputed over it and therefore verifying.
    Loading a pre-fix model file and then training from scratch was enough to reach it: the
    scratch run's own 1,000 words were discarded and its saved file read `[',', '"',
    'the', '.']`. Hashing the vocabulary makes the collision impossible rather than
    policing it afterwards, and it makes the two stacks agree, which a caching policy
    could not (Python kept the first vocabulary, the TS engine the last).
    """
    h = hashlib.sha256()
    for name in sorted(ws):
        arr = np.ascontiguousarray(np.asarray(ws[name], dtype=np.float32))
        h.update(name.encode("utf-8"))
        h.update(repr(arr.shape).encode("utf-8"))
        h.update(arr.tobytes())
    if vocab_json is not None:
        h.update(_VOCAB_HASH_TAG)
        h.update(vocab_json.encode("utf-8"))
    return h.hexdigest()[:32]


def own_vocab_json(vocab_json: str | None) -> str | None:
    """The vocabulary bytes that take part in a content hash, or None.

    ``vocab_json`` must already be the CANONICAL serialization (``GeoTokenizer.to_json``):
    a model's identity cannot depend on the key order a writer happened to emit, and the
    browser engine hashes ``canonicalVocabJson(words)``.

    A word list identical to the shipped one is NOT an own vocabulary: such a model reads
    under the canonical tokenizer whatever it was trained on, there is nothing to
    substitute, and treating it as owned would give the canonical checkpoint two different
    tokens depending on whether it arrived through the checkpoint or through a file.
    """
    if vocab_json is None:
        return None
    from .tokenizer import get_tokenizer  # local import to avoid a cycle

    return None if vocab_json == get_tokenizer().to_json() else vocab_json


def _artifact_key(token: str) -> str:
    return f"{_ARTIFACT_PREFIX}-{token}"


def save_weight_set(
    ws: dict[str, np.ndarray],
    source: str,
    store: CacheStore | None = None,
    vocab_json: str | None = None,
    owns_vocab: bool | None = None,
) -> str:
    """Persist a weight set under its content-hash token; return the token.

    ``vocab_json`` travels with any model whose token ids mean words of its OWN:
    trained from scratch on a user's text, imported from such a file, or DERIVED from
    one of those by fine-tuning or a weight edit. Omitted (None) ⇒ the canonical
    vocabulary, which is the right answer only for sets descended from the canonical
    checkpoint.

    ``owns_vocab`` records that claim independently of the payload (it defaults to
    ``vocab_json is not None``). The two are written together, so a set that says it
    owns a word list but carries none is a corrupted entry — and ``export_bundle``
    refuses it rather than silently substituting the shipped vocabulary, which is the
    exact substitution the three digests exist to prevent.

    The token covers the vocabulary (see :func:`weights_token`), so the store's dedup can
    no longer hand one model's weights another model's words: different word lists mean
    different keys. The reconciliation below is the belt to that braces — it fires only
    on a self-contradictory write (``owns_vocab=True`` with no payload, which hashes like
    an unowned set) and raises instead of silently keeping whichever entry got there
    first, which is how the substitution used to happen.
    """
    store = store or CacheStore()
    owns = bool(vocab_json is not None if owns_vocab is None else owns_vocab)
    hashed_vocab = own_vocab_json(vocab_json) if owns else None
    if owns and vocab_json is not None and hashed_vocab is None:
        # The word list IS the shipped one, so this model does not own a vocabulary in any
        # sense that matters: it reads under the canonical tokenizer either way, and
        # claiming otherwise would give the canonical checkpoint two tokens depending on
        # whether it arrived as the checkpoint or as a file.
        owns = False
    # Only a genuinely-own word list is stored; `owns_vocab=True` with nothing to store
    # is the corrupted shape `export_bundle` refuses, and it is preserved as written.
    vocab_json = hashed_vocab
    token = weights_token(ws, hashed_vocab)
    key = _artifact_key(token)
    existing = store.get(key)
    if existing is not None:
        stored = existing["meta"]
        stored_vocab = stored.get("vocab") if isinstance(stored.get("vocab"), str) else None
        stored_owns = bool(stored.get("owns_vocab", stored_vocab is not None))
        if (stored_vocab, stored_owns) != (vocab_json, owns):
            raise InvalidParamError(
                f"weights_token {token!r} is already stored with a different vocabulary "
                f"claim (stored owns_vocab={stored_owns}, writing owns_vocab={owns}) — "
                "refusing to overwrite or to reuse it, because a content hash that carried "
                "someone else's word list would mislabel every token in this model."
            )
        return token
    spec = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": _ARTIFACT_PREFIX,
        "weights_token": token,
    }
    meta: dict[str, Any] = {
        "weights_token": token,
        "source": source,
        "names": sorted(ws),
        "owns_vocab": owns,
    }
    if vocab_json is not None:
        meta["vocab"] = vocab_json
    store.put(key, spec, meta, {name: np.asarray(a, np.float32) for name, a in ws.items()})
    return token


def _missing_entry_error(token: str, store: CacheStore) -> NotFoundError:
    """The refusal for a token this store cannot serve, saying WHICH kind of gone it is.

    A schema bump orphans every ``geo-weights`` entry an older build wrote, and the store
    reports that as a plain miss. Reporting it to the user as "evicted" would be a guess,
    and the remedy differs: an evicted token comes back by re-submitting the edit, while a
    v14 entry never comes back at all, because the identity it was keyed under no longer
    names it (see ``config.SCHEMA_VERSION``).
    """
    stale = store.stale_schema_version(_artifact_key(token))
    if stale is not None:
        return NotFoundError(
            f"weights_token {token!r} was stored by an earlier build (cache schema "
            f"v{stale}; this build reads v{SCHEMA_VERSION}) and is not loaded. A model's "
            "identity now covers the words its ids mean as well as its weights, and a "
            f"v{stale} entry's word list cannot be checked against the token naming it — "
            "it may belong to a different model, which is exactly how a model came to be "
            "labelled with another model's words. Nothing was deleted, and a SAVED MODEL "
            "FILE still loads — including one written before this change, which this build "
            "reads as a version-2 file and re-identifies: open it again, or train the "
            "model again."
        )
    return NotFoundError(
        f"weights_token {token!r} is unknown (never minted here, or evicted); "
        "re-submit the edit to mint it again"
    )


def weight_set_entry(token: str, store: CacheStore | None = None) -> dict[str, Any]:
    """The stored artifact for ``token``, or raise — the ONE place a token is resolved.

    Every read of a persisted model goes through here so a store miss cannot mean
    different things on different routes. It used to: ``load_weight_set`` raised, while
    the vocabulary lookup answered ``None`` — indistinguishable from "this model reads
    under the shipped word list" — so ``GET /api/geo/tokenize`` labelled an unknown
    model's ids with Alice in Wonderland's words and returned 200 while
    ``GET /api/geo/trace`` 404'd on the same token.
    """
    store = store or CacheStore()
    entry = store.get(_artifact_key(token))
    if entry is None:
        raise _missing_entry_error(token, store)
    return entry


def load_weight_set_vocab(token: str, store: CacheStore | None = None) -> str | None:
    """The vocabulary JSON stored alongside ``token``, or None for the canonical one.

    ``None`` means exactly one thing — "this model's ids mean the shipped words". A token
    that is not stored raises (:func:`weight_set_entry`) rather than sharing that answer.
    """
    entry = weight_set_entry(token, store=store)
    vocab = entry["meta"].get("vocab")
    return vocab if isinstance(vocab, str) else None


def weight_set_owns_vocab(token: str, store: CacheStore | None = None) -> bool:
    """True iff ``token``'s ids mean its own words rather than the canonical ones."""
    meta = weight_set_entry(token, store=store)["meta"]
    # Entries written before `owns_vocab` existed recorded ownership only implicitly,
    # by carrying a vocabulary; read them the same way rather than guessing.
    return bool(meta.get("owns_vocab", isinstance(meta.get("vocab"), str)))


def inherited_vocab(base: str, store: CacheStore | None = None) -> tuple[str | None, bool]:
    """The (vocabulary JSON, owns_vocab) a set DERIVED from ``base`` must carry.

    Fine-tuning and weight editing produce a new model whose ids still mean the base
    model's words. Dropping that word list on the way through is not a cosmetic loss:
    the derived set then reads under the shipped Alice-in-Wonderland vocabulary and a
    saved file pairs your weights with those words under a `vocab_sha256` computed
    over them, so the file verifies and every label on screen is wrong.
    """
    if base == "learned":
        return None, False
    return load_weight_set_vocab(base, store=store), weight_set_owns_vocab(base, store=store)


# -- completeness / shape validation ------------------------------------------------------

#: Every tensor a GeoTransformer weight set must contain, and its exact shape. Kept as
#: plain data (rather than reached through ``GeoTransformer.weight_names()``) so file
#: validation costs no torch module construction. ``test_weight_shapes_match_the_model``
#: pins it against the real module.
WEIGHT_SHAPES: dict[str, tuple[int, ...]] = {
    "embedding": (VOCAB_SIZE, D_MODEL),
    "pos_embedding": (CONTEXT_WINDOW, D_MODEL),
    **{
        f"layers.{i}.{name}": shape
        for i in range(N_LAYERS)
        for name, shape in (
            ("W_Q", (D_MODEL, D_MODEL)),
            ("W_K", (D_MODEL, D_MODEL)),
            ("W_V", (D_MODEL, D_MODEL)),
            ("W_O", (D_MODEL, D_MODEL)),
            ("W_in", (D_MODEL, MLP_HIDDEN)),
            ("b_in", (MLP_HIDDEN,)),
            ("W_out", (MLP_HIDDEN, D_MODEL)),
            ("b_out", (D_MODEL,)),
        )
    },
}


def validate_weight_set(ws: dict[str, np.ndarray], context: str = "weight set") -> None:
    """Raise unless ``ws`` is a COMPLETE, correctly-shaped GeoTransformer weight set.

    The mirror of ``lib/geoEngine/model.validateWeightSet``. Without it a model file
    carrying one tensor (with every digest honestly recomputed over that one tensor)
    loaded with a 200 and only fell over later, once as an opaque 500 whose message was
    the bare string ``'layers.0.W_V'``.
    """
    missing = sorted(n for n in WEIGHT_SHAPES if n not in ws)
    extra = sorted(n for n in ws if n not in WEIGHT_SHAPES)
    if missing or extra:
        raise InvalidParamError(
            f"{context} is incomplete (missing: {missing or 'none'}, "
            f"unexpected: {extra or 'none'}) — a GeoTransformer needs all "
            f"{len(WEIGHT_SHAPES)} tensors, so it cannot be run"
        )
    for name, shape in WEIGHT_SHAPES.items():
        actual = tuple(np.asarray(ws[name]).shape)
        if actual != shape:
            raise InvalidParamError(
                f"{context}: weight {name!r} has shape {actual}, expected {shape}"
            )


def load_weight_set(token: str, store: CacheStore | None = None) -> dict[str, np.ndarray]:
    """Load a persisted weight set by token; raise NotFoundError if absent/evicted."""
    entry = weight_set_entry(token, store=store)
    return {name: np.asarray(arr, dtype=np.float32) for name, arr in entry["arrays"].items()}
