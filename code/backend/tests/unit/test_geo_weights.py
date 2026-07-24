"""Unit tests for Geometry Lab weight presets, editing, and content-hash tokens.

Everything is real (FR-109): the ``learned`` preset resolves against the actual
canonical checkpoint (training it once if the shared cache is cold — the result is
cached for every later test and app run).
"""

from __future__ import annotations

import numpy as np
import pytest

from llm_geometry.cache.store import CacheStore
from llm_geometry.errors import InvalidWeightEditError, NotFoundError
from llm_geometry.geo.config import D_MODEL, N_LAYERS, VOCAB_SIZE
from llm_geometry.geo.model import GeoTransformer
from llm_geometry.geo.weights import (
    EDITABLE_MATRICES,
    PRESETS,
    build_weight_set,
    load_weight_set,
    preset_matrix,
    save_weight_set,
    weights_token,
)


@pytest.fixture(scope="module")
def base_ws():
    return GeoTransformer(seed=0).get_weight_set()


# -- presets ---------------------------------------------------------------------------


@pytest.mark.parametrize("preset", [p for p in PRESETS if p != "learned"])
@pytest.mark.parametrize("matrix", ["W_Q", "embedding"])
def test_every_preset_shape(preset, matrix):
    if preset == "zero" and matrix == "embedding":
        with pytest.raises(InvalidWeightEditError):
            preset_matrix(preset, matrix)
        return
    out = preset_matrix(preset, matrix, seed=0)
    expected = (VOCAB_SIZE, D_MODEL) if matrix == "embedding" else (D_MODEL, D_MODEL)
    assert out.shape == expected
    assert out.dtype == np.float32
    assert np.all(np.isfinite(out))
    if matrix == "embedding":  # unit-norm rows enforced for every embedding preset
        assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)


def test_learned_preset_shapes():
    from llm_geometry.geo.train import train_canonical

    train_canonical()  # real training on cold cache; instant cache hit afterwards
    emb = preset_matrix("learned", "embedding")
    assert emb.shape == (VOCAB_SIZE, D_MODEL)
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-3)
    ws, edited = build_weight_set(
        GeoTransformer(seed=0).get_weight_set(),
        [{"layer": 1, "matrix": "W_V", "preset": "learned"}],
    )
    assert ws["layers.1.W_V"].shape == (D_MODEL, D_MODEL)
    assert edited == [{"layer": 1, "matrix": "W_V", "source": "preset:learned"}]


def test_toeplitz_fuzzy_structure():
    t = preset_matrix("toeplitz_fuzzy", "W_Q")
    assert np.allclose(np.diag(t), 1.0)
    assert np.allclose(t, t.T)  # symmetric in |i-j|
    assert t[0, 0] > t[0, 1] > t[0, 2] > 0  # fuzzy decay off the diagonal
    assert np.isclose(t[0, 1], np.exp(-1 / (2 * 0.75**2)), atol=1e-6)
    assert np.isclose(t[0, 1], t[1, 2])  # Toeplitz: constant along diagonals


def test_random_presets_are_seeded():
    a = preset_matrix("random", "W_Q", seed=7)
    b = preset_matrix("random", "W_Q", seed=7)
    c = preset_matrix("random", "W_Q", seed=8)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    auto = preset_matrix("random_autocorr", "embedding", seed=3)
    # Autocorrelation down the vocab axis: neighboring rows are much more aligned
    # than random unit vectors (whose expected dot product is 0).
    neighbor_dot = float(np.mean(np.sum(auto[:-1] * auto[1:], axis=1)))
    assert neighbor_dot > 0.5


def test_identity_embedding_rows_cycle():
    emb = preset_matrix("identity", "embedding")
    assert np.array_equal(emb[:3], np.eye(3, dtype=np.float32))
    assert np.array_equal(emb[3], emb[0])


# -- build_weight_set + tokens ---------------------------------------------------------


def test_content_hash_stability(base_ws):
    edits = [
        {"layer": 0, "matrix": "W_V", "preset": "identity"},
        {"layer": 2, "matrix": "W_Q", "preset": "random", "seed": 5},
    ]
    ws1, _ = build_weight_set(base_ws, edits)
    ws2, _ = build_weight_set(base_ws, edits)
    assert weights_token(ws1) == weights_token(ws2)  # same edits -> same token
    assert weights_token(ws1) != weights_token(base_ws)  # edit changes the token
    ws3, _ = build_weight_set(base_ws, [{"layer": 0, "matrix": "W_V", "preset": "zero"}])
    assert weights_token(ws3) != weights_token(ws1)


def test_build_does_not_mutate_base(base_ws):
    before = weights_token(base_ws)
    build_weight_set(base_ws, [{"layer": 0, "matrix": "W_Q", "preset": "zero"}])
    assert weights_token(base_ws) == before


def test_explicit_values_and_embedding_normalization(base_ws):
    vals = np.full((VOCAB_SIZE, D_MODEL), 2.0)  # rows of norm 2√3 -> renormalized
    ws, edited = build_weight_set(
        base_ws, [{"layer": 0, "matrix": "embedding", "values": vals.tolist()}]
    )
    assert edited[0]["source"] == "edited"
    assert np.allclose(np.linalg.norm(ws["embedding"], axis=1), 1.0, atol=1e-5)


@pytest.mark.parametrize(
    "edit",
    [
        {"layer": 0, "matrix": "W_X", "preset": "identity"},  # bad matrix name
        {"layer": 0, "matrix": "W_Q"},  # neither preset nor values
        {"layer": 0, "matrix": "W_Q", "preset": "identity", "values": [[1, 0, 0]] * 3},  # both
        {"layer": 0, "matrix": "W_Q", "preset": "diagonalize"},  # unknown preset
        {"layer": 0, "matrix": "W_Q", "values": [[1, 0], [0, 1]]},  # bad shape
        {"layer": 0, "matrix": "W_Q", "values": [[float("nan")] * 3] * 3},  # non-finite
        {"layer": N_LAYERS, "matrix": "W_Q", "preset": "identity"},  # layer out of range
        {"layer": None, "matrix": "W_Q", "preset": "identity"},  # layer missing
        {"layer": 0, "matrix": "embedding", "preset": "zero"},  # zero embedding
        {"layer": 0, "matrix": "embedding", "values": np.zeros((VOCAB_SIZE, D_MODEL)).tolist()},
    ],
)
def test_invalid_edits_raise(base_ws, edit):
    with pytest.raises(InvalidWeightEditError):
        build_weight_set(base_ws, [edit])


def test_editable_matrix_list_matches_contract():
    assert EDITABLE_MATRICES == ("W_Q", "W_K", "W_V", "W_O", "embedding")
    assert PRESETS == ("identity", "toeplitz_fuzzy", "random", "random_autocorr", "zero", "learned")


# -- persistence -----------------------------------------------------------------------


def test_save_load_round_trip(base_ws, tmp_path):
    store = CacheStore(tmp_path)
    token = save_weight_set(base_ws, source="test", store=store)
    assert token == weights_token(base_ws)
    loaded = load_weight_set(token, store=store)
    assert sorted(loaded) == sorted(base_ws)
    for name in base_ws:
        assert np.array_equal(loaded[name], np.asarray(base_ws[name], np.float32))
    assert weights_token(loaded) == token  # token survives the round trip
    with pytest.raises(NotFoundError):
        load_weight_set("0" * 32, store=store)
