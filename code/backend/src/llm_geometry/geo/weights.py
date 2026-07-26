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
from ..errors import InvalidWeightEditError, NotFoundError
from .config import D_MODEL, N_LAYERS, VOCAB_SIZE

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
        seed = int(edit.get("seed", 0) or 0)
        layer = edit.get("layer")
        if matrix != "embedding":
            if layer is None or not (isinstance(layer, int) and 0 <= layer < N_LAYERS):
                raise InvalidWeightEditError(
                    f"edit {n} ({matrix}): layer must be an int in 0..{N_LAYERS - 1}, "
                    f"got {layer!r}"
                )

        if values is not None:
            arr = validate_values(matrix, values)
            source = "edited"
        elif preset == "learned" and matrix != "embedding":
            arr = _learned_layer_matrix(int(layer), matrix)
            source = "preset:learned"
        else:
            arr = preset_matrix(str(preset), matrix, seed=seed)
            source = f"preset:{preset}"

        if matrix == "embedding":
            arr = _unit_rows(arr, f"edit {n} (embedding)")
            ws["embedding"] = arr
        else:
            ws[f"layers.{int(layer)}.{matrix}"] = arr
        summaries.append({"layer": layer, "matrix": matrix, "source": source})
    return ws, summaries


# -- content-hash tokens + persistence -------------------------------------------------


def weights_token(ws: dict[str, np.ndarray]) -> str:
    """Content hash over the full weight set (name-sorted float32 bytes)."""
    h = hashlib.sha256()
    for name in sorted(ws):
        arr = np.ascontiguousarray(np.asarray(ws[name], dtype=np.float32))
        h.update(name.encode("utf-8"))
        h.update(repr(arr.shape).encode("utf-8"))
        h.update(arr.tobytes())
    return h.hexdigest()[:32]


def _artifact_key(token: str) -> str:
    return f"{_ARTIFACT_PREFIX}-{token}"


def save_weight_set(
    ws: dict[str, np.ndarray],
    source: str,
    store: CacheStore | None = None,
    vocab_json: str | None = None,
) -> str:
    """Persist a weight set under its content-hash token; return the token.

    ``vocab_json`` travels with models trained from scratch on a user's own text: their
    token ids mean different words than the canonical checkpoint's, so the vocabulary
    is part of the model, not a global. Omitted (None) ⇒ the canonical vocabulary.
    """
    store = store or CacheStore()
    token = weights_token(ws)
    key = _artifact_key(token)
    if store.get(key) is None:  # dedup: identical content already stored
        spec = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": _ARTIFACT_PREFIX,
            "weights_token": token,
        }
        meta: dict[str, Any] = {"weights_token": token, "source": source, "names": sorted(ws)}
        if vocab_json is not None:
            meta["vocab"] = vocab_json
        store.put(key, spec, meta, {name: np.asarray(a, np.float32) for name, a in ws.items()})
    return token


def load_weight_set_vocab(token: str, store: CacheStore | None = None) -> str | None:
    """The vocabulary JSON stored alongside ``token``, or None for the canonical one."""
    store = store or CacheStore()
    entry = store.get(_artifact_key(token))
    if entry is None:
        return None
    vocab = entry["meta"].get("vocab")
    return vocab if isinstance(vocab, str) else None


def load_weight_set(token: str, store: CacheStore | None = None) -> dict[str, np.ndarray]:
    """Load a persisted weight set by token; raise NotFoundError if absent/evicted."""
    store = store or CacheStore()
    entry = store.get(_artifact_key(token))
    if entry is None:
        raise NotFoundError(
            f"weights_token {token!r} is unknown (never minted here, or evicted); "
            "re-submit the edit to mint it again"
        )
    return {name: np.asarray(arr, dtype=np.float32) for name, arr in entry["arrays"].items()}
