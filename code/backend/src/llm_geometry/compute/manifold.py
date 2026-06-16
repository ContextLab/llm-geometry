"""Visualization 3 — reachable "thoughts" as a warped manifold. (project_description.md §3)

Reduce embeddings to 3D unit directions (tokens sit on a radius-2 sphere). Start from a
unit sphere mesh and raise a smooth outward bump in the direction of each likely next
token (height ∝ emission probability) via a radial RBF lift, then apply Open3D
`deform_as_rigid_as_possible` (ARAP). The per-token RBF caps are SUMMED into one lift
field (commutative → order-invariant, the open question flagged in the description) and
the total is capped, so the surface is a set of gentle rounded domes rather than a sharp
teardrop spiking toward a single token. Lifting each vertex along its own radial direction
(not toward one shared point) is what keeps the bumps rounded.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from scipy.spatial.distance import cdist

from ..config import DEFAULT_SEED
from ..errors import ComputeError, InvalidParamError
from ..models.loader import load_model
from .distributions import next_token_distribution
from .printable import printable_reference_ids, printable_tokens

ProgressCb = Callable[[float, str], None]


def _warp_sphere(o3d, dirs: np.ndarray, emis: np.ndarray, width: float, warp_top: int):
    """Build the radius-1 sphere, raise a smooth (capped, order-invariant) outward-lift field
    toward the top emitting tokens, ARAP-smooth, and return (vertices, faces, normalized warp)."""
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=30)
    mesh.compute_vertex_normals()
    verts = np.asarray(mesh.vertices)
    orig = verts.copy()
    vdir = verts / (np.linalg.norm(verts, axis=1, keepdims=True) + 1e-9)
    order = np.argsort(-emis)[: max(1, int(warp_top))]
    lift = np.zeros(verts.shape[0])
    for ti in order:
        p = float(np.sqrt(max(0.0, emis[ti]))) * 0.75
        if p <= 1e-3:
            continue
        d = cdist(verts, np.atleast_2d(dirs[ti]))[:, 0]
        lift += p * np.exp(-(d * d) / width)
    cap = 0.9
    lift = cap * np.tanh(lift / cap)
    warped = verts + vdir * lift[:, None]
    moved = np.where(lift > 0.02)[0]
    if moved.size:
        ids = o3d.utility.IntVector([int(i) for i in moved])
        positions = o3d.utility.Vector3dVector(warped[moved])
        mesh = mesh.deform_as_rigid_as_possible(ids, positions, max_iter=5)
        mesh.compute_vertex_normals()
    final_verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    warp = np.linalg.norm(final_verts - orig, axis=1)
    if warp.max() > 0:
        warp = warp / warp.max()
    return final_verts, faces, warp


def manifold(
    model_id: str,
    prefix_text: str = "",
    temperature: float = 1.0,
    reference_set_size: int | None = None,  # None = a dot for EVERY vocab token
    seed: int = DEFAULT_SEED,
    width: float = 0.3,  # RBF cap width on the UNIT sphere (small = localized rounded domes)
    warp_top: int = 48,
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
    # Only PRINTABLE tokens become markers (no special/byte-fragment noise); ship strings.
    all_ids, all_strs = printable_tokens(lm)
    if reference_set_size is None or int(reference_set_size) >= all_ids.shape[0]:
        token_ids, token_strs = all_ids, all_strs
    else:
        token_ids = printable_reference_ids(lm, reference_set_size)
        id2s = dict(zip(all_ids.tolist(), all_strs))
        token_strs = [id2s[int(t)] for t in token_ids]
    matrix = lm.model.get_input_embeddings().weight.detach().float().cpu().numpy().astype(np.float64)
    emb = matrix[token_ids]

    # Response trajectory: reduce the response tokens TOGETHER with the reference tokens so
    # they sit in the same sphere frame, then split out their radius-2 positions (drawn as a
    # line on the sphere; the frontend reveals it up to response_step as ▶ Play advances).
    printable_set = set(int(i) for i in all_ids.tolist())
    resp_ids = [int(t) for t in (lm.tokenizer(response_text)["input_ids"] if response_text else []) if int(t) in printable_set]

    from ..reduce.sphere import reduce_3d_sphere

    n_ref = int(token_ids.shape[0])
    if resp_ids:
        dirs_all = reduce_3d_sphere(np.vstack([emb, matrix[resp_ids]]), method="pca3", seed=seed).astype(np.float64)
        dirs, traj_dirs = dirs_all[:n_ref], dirs_all[n_ref:]
    else:
        dirs = reduce_3d_sphere(emb, method="pca3", seed=seed).astype(np.float64)
        traj_dirs = np.empty((0, 3), dtype=np.float64)

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
        progress_cb(0.5, "warping the sphere toward likely tokens")
    final_verts, faces, warp = _warp_sphere(o3d, dirs, emis, width, warp_top)

    if progress_cb:
        progress_cb(1.0, "done")

    order = np.argsort(-emis)[:max(1, int(warp_top))]
    top_list = np.argsort(-emis)[:25]
    meta = {
        "model_id": lm.model_id, "revision": lm.revision, "prefix_text": prefix_text or "",
        "temperature": float(temperature), "n_vertices": int(final_verts.shape[0]),
        "n_faces": int(faces.shape[0]), "warp_top": int(len(order)),
        "top_tokens": [
            {"token_str": token_strs[int(i)], "prob": float(emis[i])}
            for i in top_list
        ],
        "token_strs": token_strs,  # real decoded strings (printable), aligned with token markers
    }
    arrays = {
        "vertices": final_verts.astype(np.float32),
        "faces": faces.astype(np.int64),
        "warp": warp.astype(np.float32),
        "token_points": (dirs * 2.0).astype(np.float32),
        "token_emis": emis.astype(np.float32),
        "token_ids": token_ids.astype(np.int64),
        "traj_points": (traj_dirs * 2.0).astype(np.float32),  # response tokens on the sphere
    }
    if resp_ids:
        meta["trajectory_token_strs"] = [lm.tokenizer.decode([t]) for t in resp_ids]
        meta["trajectory_emis"] = [float(probs_full[t]) for t in resp_ids]
    return {"meta": meta, "arrays": arrays}


def manifold_animation(
    model_id: str,
    prefix_text: str = "",
    temperature: float = 1.0,
    reference_set_size: int | None = None,
    seed: int = DEFAULT_SEED,
    width: float = 0.3,
    warp_top: int = 48,
    response_text: str = "",
    progress_cb: ProgressCb | None = None,
) -> dict[str, Any]:
    """All KEY FRAMES of the manifold animation (one per response step). The sphere geometry
    (token positions, the trajectory line) is computed ONCE; only the warped mesh + per-token
    emission change per frame, so the frontend can morph the surface SMOOTHLY between frames
    and lay the trajectory dots down one at a time."""
    try:
        import open3d as o3d
    except Exception as exc:  # pragma: no cover
        raise ComputeError(f"open3d is required for the manifold visualization: {exc}") from exc

    lm = load_model(model_id)
    all_ids, all_strs = printable_tokens(lm)
    if reference_set_size is None or int(reference_set_size) >= all_ids.shape[0]:
        token_ids, token_strs = all_ids, all_strs
    else:
        token_ids = printable_reference_ids(lm, reference_set_size)
        id2s = dict(zip(all_ids.tolist(), all_strs))
        token_strs = [id2s[int(t)] for t in token_ids]
    matrix = lm.model.get_input_embeddings().weight.detach().float().cpu().numpy().astype(np.float64)
    emb = matrix[token_ids]
    printable_set = set(int(i) for i in all_ids.tolist())
    resp_ids = [int(t) for t in (lm.tokenizer(response_text)["input_ids"] if response_text else []) if int(t) in printable_set]

    from ..reduce.sphere import reduce_3d_sphere

    n_ref = int(token_ids.shape[0])
    if resp_ids:
        dirs_all = reduce_3d_sphere(np.vstack([emb, matrix[resp_ids]]), method="pca3", seed=seed).astype(np.float64)
        dirs, traj_dirs = dirs_all[:n_ref], dirs_all[n_ref:]
    else:
        dirs = reduce_3d_sphere(emb, method="pca3", seed=seed).astype(np.float64)
        traj_dirs = np.empty((0, 3), dtype=np.float64)

    n_frames = len(resp_ids) + 1
    verts_per, warp_per, emis_per, faces = [], [], [], None
    for s in range(n_frames):
        if progress_cb:
            progress_cb(0.1 + 0.85 * s / n_frames, f"key frame {s + 1}/{n_frames}")
        probs = next_token_distribution(
            model_id, prefix_text=prefix_text, temperature=temperature,
            response_text=response_text, response_step=s,
        )["arrays"]["probs"]
        emis = probs[token_ids].astype(np.float64)
        if emis.max() > 0:
            emis = emis / emis.max()
        fv, faces, warp = _warp_sphere(o3d, dirs, emis, width, warp_top)
        verts_per.append(fv.astype(np.float32))
        warp_per.append(warp.astype(np.float32))
        emis_per.append(emis.astype(np.float32))

    meta = {
        "model_id": lm.model_id, "revision": lm.revision, "n_frames": int(n_frames),
        "n_vertices": int(verts_per[0].shape[0]), "prefix_text": prefix_text or "",
        "token_strs": token_strs,
        "trajectory_token_strs": [lm.tokenizer.decode([t]) for t in resp_ids],
    }
    arrays = {
        "faces": faces.astype(np.int64),                       # static
        "token_points": (dirs * 2.0).astype(np.float32),       # static
        "traj_points": (traj_dirs * 2.0).astype(np.float32),   # static
        "vertices": np.stack(verts_per),                       # (F, V, 3) — morph these
        "warp": np.stack(warp_per),                            # (F, V)
        "token_emis": np.stack(emis_per),                      # (F, R) — marker alpha/size
    }
    if progress_cb:
        progress_cb(1.0, "done")
    return {"meta": meta, "arrays": arrays}
