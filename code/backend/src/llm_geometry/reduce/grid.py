"""n×n reference grid over the 2D space.

Lays a regular grid over the bounds of the 2D coordinates and snaps each vertex to
its nearest token — the grid "reference points" the vector-field view draws from
(FR-011). These are distinct from the embedding "reference set" (see
compute/embeddings.py).
"""

from __future__ import annotations

import numpy as np

from ..errors import InvalidParamError


def build_grid(coords: np.ndarray, token_ids: np.ndarray, n: int = 25) -> tuple[np.ndarray, np.ndarray]:
    coords = np.asarray(coords, dtype=np.float64)
    token_ids = np.asarray(token_ids)
    if n < 2:
        raise InvalidParamError(f"grid n must be >= 2, got {n}")
    if coords.shape[0] < 1:
        raise InvalidParamError("need at least one reduced point to build a grid")

    xmin, ymin = coords.min(axis=0)
    xmax, ymax = coords.max(axis=0)
    xs = np.linspace(xmin, xmax, n)
    ys = np.linspace(ymin, ymax, n)
    grid_x, grid_y = np.meshgrid(xs, ys)
    vertices = np.column_stack([grid_x.ravel(), grid_y.ravel()])  # [n*n, 2]

    from scipy.spatial import cKDTree

    tree = cKDTree(coords)
    _, nearest = tree.query(vertices, k=1)
    reference_token_ids = token_ids[nearest]

    return vertices.astype(np.float32), np.asarray(reference_token_ids, dtype=np.int64)
