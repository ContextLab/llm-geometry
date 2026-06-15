"""Density flattening (ContextLab/mapper-style).

After a 2D projection, token embeddings tend to clump (a few outliers dominate the
variance). Following mapper's "density flattening via approximate optimal transport
(mu)", we spread the points toward even spatial coverage / a grid-like layout using a
sliced-optimal-transport flow toward a regular grid target, then blend back by ``mu``
(0 = original projection, 1 = fully gridded). Deterministic given the seed.
"""

from __future__ import annotations

import numpy as np


def _grid_target(n: int, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    g = int(np.ceil(np.sqrt(n)))
    xs = np.linspace(lo[0], hi[0], g)
    ys = np.linspace(lo[1], hi[1], g)
    gx, gy = np.meshgrid(xs, ys)
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    return grid[:n]


def flatten_density(coords: np.ndarray, mu: float = 0.85, iters: int = 120, seed: int = 0) -> np.ndarray:
    """Spread ``coords`` toward an even grid-like layout via sliced OT; blend by ``mu``."""
    coords = np.asarray(coords, dtype=np.float64)
    n = coords.shape[0]
    if n < 3 or mu <= 0:
        return coords.astype(np.float32)

    lo = coords.min(0)
    hi = coords.max(0)
    span = np.where(hi > lo, hi - lo, 1.0)
    lo = lo - 0.02 * span
    hi = hi + 0.02 * span
    target = _grid_target(n, lo, hi)

    rng = np.random.default_rng(seed)
    pts = coords.copy()
    lr = 0.5
    for _ in range(iters):
        theta = rng.uniform(0.0, np.pi)
        d = np.array([np.cos(theta), np.sin(theta)])
        proj = pts @ d
        tproj = np.sort(target @ d)
        order = np.argsort(proj)
        desired = np.empty(n)
        desired[order] = tproj  # rank-match the projection to the grid's projection
        pts += lr * np.outer(desired - proj, d)

    out = coords + float(mu) * (pts - coords)
    return out.astype(np.float32)
