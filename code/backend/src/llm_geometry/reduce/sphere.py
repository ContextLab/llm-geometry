"""3D spherical reduction: tokens on the unit sphere.

Reduce embeddings to 3D, then L2-normalize each point onto the unit sphere so the
manifold view can place tokens on a sphere (FR-012). ``pca3`` is the default (fast,
deterministic); ``mds`` (metric MDS) is a real option. Output rows all have unit norm.
"""

from __future__ import annotations

import numpy as np

from ..errors import InvalidParamError


def reduce_3d_sphere(vectors: np.ndarray, method: str = "pca3", seed: int = 0) -> np.ndarray:
    X = np.asarray(vectors, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 1:
        raise InvalidParamError("vectors must be a 2D array with at least one row")

    if method == "pca3":
        from sklearn.decomposition import PCA

        n_components = min(3, X.shape[0], X.shape[1])
        coords = PCA(n_components=n_components, random_state=seed).fit_transform(X)
    elif method == "mds":
        from sklearn.manifold import MDS

        coords = MDS(
            n_components=3,
            random_state=seed,
            n_init=1,
            normalized_stress="auto",
        ).fit_transform(X)
    else:
        raise InvalidParamError(f"3D method must be 'pca3' or 'mds', got {method!r}")

    # Pad to 3 columns if the reduction produced fewer, then project onto the unit sphere.
    if coords.shape[1] < 3:
        pad = np.zeros((coords.shape[0], 3 - coords.shape[1]), dtype=coords.dtype)
        coords = np.hstack([coords, pad])

    norms = np.linalg.norm(coords, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = coords / norms
    return unit.astype(np.float32)
