"""2D dimensionality reduction (PCA default, UMAP optional).

PCA is the default because it is deterministic and fast, which keeps cache results
reproducible (FR-013, SC-002). UMAP is a real, selectable option with a fixed seed.
"""

from __future__ import annotations

import numpy as np

from ..errors import InvalidParamError


def _pad_to_2d(coords: np.ndarray) -> np.ndarray:
    if coords.shape[1] >= 2:
        return coords[:, :2]
    pad = np.zeros((coords.shape[0], 2 - coords.shape[1]), dtype=coords.dtype)
    return np.hstack([coords, pad])


def reduce_2d(vectors: np.ndarray, method: str = "pca", seed: int = 0) -> np.ndarray:
    X = np.asarray(vectors, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 1:
        raise InvalidParamError("vectors must be a 2D array with at least one row")

    if method == "pca":
        from sklearn.decomposition import PCA

        n_components = min(2, X.shape[0], X.shape[1])
        if n_components < 1:
            raise InvalidParamError("not enough data to reduce")
        coords = PCA(n_components=n_components, random_state=seed).fit_transform(X)
        coords = _pad_to_2d(coords)
    elif method == "umap":
        import umap

        n_neighbors = min(15, max(2, X.shape[0] - 1))
        reducer = umap.UMAP(n_components=2, random_state=seed, n_neighbors=n_neighbors)
        coords = reducer.fit_transform(X)
    else:
        raise InvalidParamError(f"2D method must be 'pca' or 'umap', got {method!r}")

    return np.asarray(coords, dtype=np.float32)
