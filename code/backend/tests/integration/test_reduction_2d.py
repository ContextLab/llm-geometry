"""US2 — 2D reduction + reference grid (T023). Uses tiny-gpt2."""

import numpy as np

from llm_geometry import precompute
from llm_geometry.precompute import get_or_compute_sync

MODEL = "sshleifer/tiny-gpt2"
REF = 150


def test_pca_2d_with_grid():
    payload = get_or_compute_sync(
        "reduction_2d", MODEL,
        {"method": "pca", "with_grid": True, "grid_n": 12, "reference_set_size": REF},
    )
    coords = payload["arrays"]["coords"]
    assert coords.shape == (REF, 2)

    vertices = payload["arrays"]["grid_vertices"]
    refs = payload["arrays"]["grid_reference_token_ids"]
    assert vertices.shape == (144, 2)  # 12 x 12
    assert refs.shape == (144,)
    # Every grid reference point is a real token from the reference set (FR-011).
    token_ids = set(payload["arrays"]["token_ids"].tolist())
    assert set(refs.tolist()).issubset(token_ids)


def test_reduction_2d_reproducible_with_seed():
    params = {"method": "pca", "reference_set_size": REF, "seed": 0}
    key = precompute.cache_key_for("reduction_2d", MODEL, params)
    precompute.get_store().delete(key)
    first = get_or_compute_sync("reduction_2d", MODEL, params)["arrays"]["coords"].copy()
    precompute.get_store().delete(key)
    second = get_or_compute_sync("reduction_2d", MODEL, params)["arrays"]["coords"]
    assert np.array_equal(first, second)  # FR-013
