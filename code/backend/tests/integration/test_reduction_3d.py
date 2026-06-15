"""US2 — 3D spherical reduction (T024). Uses tiny-gpt2."""

import numpy as np

from llm_geometry import precompute
from llm_geometry.precompute import get_or_compute_sync

MODEL = "sshleifer/tiny-gpt2"
REF = 150


def test_sphere_points_have_unit_norm():
    coords = get_or_compute_sync(
        "reduction_3d", MODEL, {"method": "pca3", "reference_set_size": REF, "seed": 0}
    )["arrays"]["coords"]
    assert coords.shape == (REF, 3)
    norms = np.linalg.norm(coords, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)  # FR-012: tokens on the unit sphere


def test_mds_3d_reduction_real():
    """The metric-MDS option (FR-012) really runs and lands on the unit sphere."""
    params = {"method": "mds", "reference_set_size": 80, "seed": 0}
    key = precompute.cache_key_for("reduction_3d", MODEL, params)
    precompute.get_store().delete(key)
    coords = get_or_compute_sync("reduction_3d", MODEL, params)["arrays"]["coords"]
    assert coords.shape == (80, 3)
    norms = np.linalg.norm(coords, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_reduction_3d_reproducible_with_seed():
    params = {"method": "pca3", "reference_set_size": REF, "seed": 0}
    key = precompute.cache_key_for("reduction_3d", MODEL, params)
    precompute.get_store().delete(key)
    first = get_or_compute_sync("reduction_3d", MODEL, params)["arrays"]["coords"].copy()
    precompute.get_store().delete(key)
    second = get_or_compute_sync("reduction_3d", MODEL, params)["arrays"]["coords"]
    assert np.array_equal(first, second)  # FR-013
