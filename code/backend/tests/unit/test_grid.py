"""Unit — reference grid vertex count + nearest-token math (T039)."""

import numpy as np

from llm_geometry.reduce.grid import build_grid


def test_grid_vertex_count_and_membership():
    coords = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    token_ids = np.array([10, 11, 12, 13])
    vertices, refs = build_grid(coords, token_ids, n=3)
    assert vertices.shape == (9, 2)
    assert refs.shape == (9,)
    assert set(refs.tolist()).issubset({10, 11, 12, 13})


def test_corner_vertex_maps_to_nearest_token():
    coords = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    token_ids = np.array([10, 11, 12, 13])
    vertices, refs = build_grid(coords, token_ids, n=2)
    # n=2 grid corners coincide with the four points; first vertex is (xmin, ymin)=(0,0) -> token 10.
    assert vertices[0].tolist() == [0.0, 0.0]
    assert int(refs[0]) == 10
