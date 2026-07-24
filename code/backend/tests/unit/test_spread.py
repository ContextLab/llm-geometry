"""Unit tests for density-flattening + the warp_like displacement mapping."""

import numpy as np

from llm_geometry.reduce.spread import flatten_density, warp_like


def test_flatten_density_spreads_a_clump():
    rng = np.random.default_rng(0)
    # a tight clump plus a couple of far outliers -> low coverage of the bounding box
    coords = np.vstack([rng.standard_normal((400, 2)) * 0.05, np.array([[5.0, 5.0], [-5.0, -5.0]])])
    out = flatten_density(coords, mu=0.85, seed=0)
    assert out.shape == coords.shape
    # the spread layout should occupy a larger fraction of its bounding box than the clump
    def coverage(p):
        lo, hi = p.min(0), p.max(0)
        cells = 12
        ix = np.clip(((p - lo) / np.where(hi > lo, hi - lo, 1) * cells).astype(int), 0, cells - 1)
        return len(set(map(tuple, ix))) / (cells * cells)
    assert coverage(out) > coverage(coords)


def test_flatten_density_is_deterministic():
    rng = np.random.default_rng(1)
    coords = rng.standard_normal((200, 2))
    a = flatten_density(coords, mu=0.8, seed=3)
    b = flatten_density(coords, mu=0.8, seed=3)
    assert np.array_equal(a, b)


def test_warp_like_maps_src_points_onto_dst():
    rng = np.random.default_rng(2)
    src = rng.standard_normal((300, 2))
    dst = src * 2.0 + np.array([1.0, -1.0])  # a known affine displacement field
    # points that ARE cloud points should land (near) their dst image
    mapped = warp_like(src[:10], src, dst)
    assert mapped.shape == (10, 2)
    assert np.allclose(mapped, dst[:10], atol=1e-3)


def test_warp_like_handles_single_point_and_empty():
    src = np.random.default_rng(3).standard_normal((50, 2))
    dst = src + 1.0
    one = warp_like(src[0], src, dst)
    assert one.shape == (1, 2)
    assert np.allclose(one, dst[0], atol=1e-3)
    empty = warp_like(np.empty((0, 2)), src, dst)
    assert empty.shape == (0, 2)
