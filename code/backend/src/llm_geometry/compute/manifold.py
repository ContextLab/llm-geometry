"""Visualization 3 — reachable "thoughts" as a warped manifold. (project_description.md §3)

Reduce embeddings to 3D unit directions (tokens sit on a radius-2 sphere). Start from a
unit sphere mesh and, for each likely next token, move the nearest sphere vertex TOWARD
that token's radius-2 coordinate (distance ∝ emission probability), dragging neighbors
along via a normalized RBF, then apply Open3D `deform_as_rigid_as_possible` (ARAP).
Looping over the top emitting tokens produces the final reachable-thoughts manifold.

Follows the `normed_rbf` / `warp_mesh` recipe in project_description.md. The ARAP step
constrains the warped (bump) region and lets the rest deform rigidly. NOTE: the
order-invariant combination of per-token warps is flagged as open research in the
description; here warps are applied sequentially over the top tokens.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from scipy.spatial.distance import cdist

from ..config import DEFAULT_REFERENCE_SET_SIZE, DEFAULT_SEED
from ..errors import ComputeError, InvalidParamError
from ..models.loader import load_model
from .distributions import next_token_distribution
from .embeddings import reference_token_ids

ProgressCb = Callable[[float, str], None]


def _normed_rbf(x: np.ndarray, center: np.ndarray, width: float, exponent: int = 2) -> np.ndarray:
    vals = np.exp(-np.power(cdist(x, np.atleast_2d(center)), exponent) / width)
    vals -= vals.min()
    top = vals.max()
    if top > 0:
        vals /= top
    return vals  # [V, 1] in [0, 1]


def _warp_mesh(o3d, mesh, target: np.ndarray, p: float, width: float, exponent: int = 2):
    """Move the closest vertex toward ``target`` by proportion ``p`` (RBF-weighted over
    neighbors), then ARAP-deform. Mirrors project_description.md's warp_mesh."""
    verts = np.asarray(mesh.vertices)
    closest = int(np.argmin(cdist(verts, np.atleast_2d(target))))
    weights = p * _normed_rbf(verts, verts[closest], width, exponent)  # [V,1]
    tgt = np.tile(np.asarray(target, dtype=float), (verts.shape[0], 1))
    w = np.tile(weights, (1, verts.shape[1]))
    warped = tgt * w + verts * (1.0 - w)

    # Constrain the meaningfully-warped (bump) region; ARAP rigidly deforms the rest.
    moved = np.where(weights[:, 0] > 0.02)[0]
    if moved.size == 0:
        return mesh
    ids = o3d.utility.IntVector([int(i) for i in moved])
    positions = o3d.utility.Vector3dVector(warped[moved])
    deformed = mesh.deform_as_rigid_as_possible(ids, positions, max_iter=3)
    deformed.compute_vertex_normals()
    return deformed


def manifold(
    model_id: str,
    prefix_text: str = "",
    temperature: float = 1.0,
    reference_set_size: int | None = DEFAULT_REFERENCE_SET_SIZE,
    seed: int = DEFAULT_SEED,
    width: float = 0.3,
    warp_top: int = 24,
    response_text: str = "",
    response_step: int = 0,
    progress_cb: ProgressCb | None = None,
) -> dict[str, Any]:
    if temperature < 0:
        raise InvalidParamError(f"temperature must be >= 0, got {temperature}")
    try:
        import open3d as o3d
    except Exception as exc:  # pragma: no cover - dependency must be installed
        raise ComputeError(f"open3d is required for the manifold visualization: {exc}") from exc

    lm = load_model(model_id)
    token_ids = reference_token_ids(lm, reference_set_size)
    matrix = lm.model.get_input_embeddings().weight.detach().float().cpu().numpy().astype(np.float64)
    emb = matrix[token_ids]

    from ..reduce.sphere import reduce_3d_sphere

    dirs = reduce_3d_sphere(emb, method="pca3", seed=seed).astype(np.float64)  # unit token dirs

    if progress_cb:
        progress_cb(0.35, "computing emission distribution")
    probs_full = next_token_distribution(
        model_id, prefix_text=prefix_text, temperature=temperature,
        response_text=response_text, response_step=response_step,
    )["arrays"]["probs"]
    emis = probs_full[token_ids].astype(np.float64)
    if emis.max() > 0:
        emis = emis / emis.max()

    if progress_cb:
        progress_cb(0.45, "building sphere mesh")
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=24)
    mesh.compute_vertex_normals()
    orig = np.asarray(mesh.vertices).copy()

    order = np.argsort(-emis)[:max(1, int(warp_top))]
    for rank, ti in enumerate(order):
        p = float(emis[ti]) * 0.9
        if p <= 1e-3:
            continue
        target = dirs[ti] * 2.0  # the token's coordinate on the radius-2 sphere
        mesh = _warp_mesh(o3d, mesh, target, p, width)
        if progress_cb:
            progress_cb(0.45 + 0.5 * (rank + 1) / len(order), f"warping toward token {rank + 1}/{len(order)}")

    final_verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    warp = np.linalg.norm(final_verts - orig, axis=1)  # displacement magnitude per vertex
    if warp.max() > 0:
        warp = warp / warp.max()

    if progress_cb:
        progress_cb(1.0, "done")

    top_list = np.argsort(-emis)[:25]
    meta = {
        "model_id": lm.model_id, "revision": lm.revision, "prefix_text": prefix_text or "",
        "temperature": float(temperature), "n_vertices": int(final_verts.shape[0]),
        "n_faces": int(faces.shape[0]), "warp_top": int(len(order)),
        "top_tokens": [
            {"token_str": lm.tokenizer.decode([int(token_ids[i])]), "prob": float(emis[i])}
            for i in top_list
        ],
        "token_strs": [lm.tokenizer.decode([int(t)]) for t in token_ids],  # aligned with token_points
    }
    arrays = {
        "vertices": final_verts.astype(np.float32),
        "faces": faces.astype(np.int64),
        "warp": warp.astype(np.float32),
        "token_points": (dirs * 2.0).astype(np.float32),
        "token_emis": emis.astype(np.float32),
    }
    return {"meta": meta, "arrays": arrays}
