"""Visualization 3 — reachable "thoughts" as a warped manifold.

Reduce embeddings to 3D and place tokens as unit directions on a sphere. Compute the
next-token emission distribution from the context, then warp a unit sphere mesh
outward toward likely tokens using a normalized RBF (neighbors dragged along), so the
surface bulges toward reachable next tokens. (project_description.md §3)

Note: this uses RBF displacement, a smooth, order-invariant v1 of the warp. Full
Open3D ARAP deformation and the order-invariant multi-token combination are flagged in
project_description.md as open research and are deferred.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ..config import DEFAULT_REFERENCE_SET_SIZE, DEFAULT_SEED
from ..errors import InvalidParamError
from ..models.loader import load_model
from .distributions import next_token_distribution
from .embeddings import reference_token_ids

ProgressCb = Callable[[float, str], None]


def _uv_sphere(n_lat: int = 36, n_lon: int = 72) -> tuple[np.ndarray, np.ndarray]:
    lats = np.linspace(0.0, np.pi, n_lat)
    lons = np.linspace(0.0, 2.0 * np.pi, n_lon)
    verts = []
    for la in lats:
        for lo in lons:
            verts.append((np.sin(la) * np.cos(lo), np.cos(la), np.sin(la) * np.sin(lo)))
    faces = []
    for i in range(n_lat - 1):
        for j in range(n_lon - 1):
            a = i * n_lon + j
            b = a + 1
            c = a + n_lon
            d = c + 1
            faces.append((a, b, d))
            faces.append((a, d, c))
    return np.array(verts, dtype=np.float64), np.array(faces, dtype=np.int64)


def manifold(
    model_id: str,
    prefix_text: str = "",
    temperature: float = 1.0,
    reference_set_size: int | None = DEFAULT_REFERENCE_SET_SIZE,
    seed: int = DEFAULT_SEED,
    width: float = 0.2,
    progress_cb: ProgressCb | None = None,
) -> dict[str, Any]:
    if temperature < 0:
        raise InvalidParamError(f"temperature must be >= 0, got {temperature}")

    lm = load_model(model_id)
    token_ids = reference_token_ids(lm, reference_set_size)
    matrix = lm.model.get_input_embeddings().weight.detach().cpu().numpy().astype(np.float64)
    emb = matrix[token_ids]

    # Reuse the tested 3D spherical reducer (handles <3 hidden dims by padding, then
    # projects onto the unit sphere) -> token unit directions.
    from ..reduce.sphere import reduce_3d_sphere

    dirs = reduce_3d_sphere(emb, method="pca3", seed=seed).astype(np.float64)

    if progress_cb:
        progress_cb(0.4, "computing emission distribution")
    probs_full = next_token_distribution(model_id, prefix_text=prefix_text, temperature=temperature)["arrays"]["probs"]
    emis = probs_full[token_ids].astype(np.float64)
    if emis.max() > 0:
        emis = emis / emis.max()

    if progress_cb:
        progress_cb(0.6, "warping sphere (RBF)")
    verts, faces = _uv_sphere()
    from scipy.spatial.distance import cdist

    top = np.argsort(-emis)[:200]  # warp toward the strongest emitters (tractable)
    distances = cdist(verts, dirs[top])
    weights = np.exp(-(distances**2) / max(width, 1e-3))
    warp = (weights * emis[top]).sum(axis=1)
    if warp.max() > 0:
        warp = warp / warp.max()
    displaced = verts * (1.0 + warp[:, None])  # bulge outward up to ~2x radius

    if progress_cb:
        progress_cb(1.0, "done")

    order = np.argsort(-emis)[:25]
    meta = {
        "model_id": lm.model_id, "revision": lm.revision, "prefix_text": prefix_text or "",
        "temperature": float(temperature), "n_vertices": int(verts.shape[0]),
        "n_faces": int(faces.shape[0]),
        "top_tokens": [
            {"token_str": lm.tokenizer.decode([int(token_ids[i])]), "prob": float(emis[i])}
            for i in order
        ],
    }
    arrays = {
        "vertices": displaced.astype(np.float32),
        "faces": faces.astype(np.int64),
        "warp": warp.astype(np.float32),
        "token_points": (dirs * 2.0).astype(np.float32),
        "token_emis": emis.astype(np.float32),
    }
    return {"meta": meta, "arrays": arrays}
