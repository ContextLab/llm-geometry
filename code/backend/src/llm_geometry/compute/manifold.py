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

from ..config import DEFAULT_RBF_WIDTH, DEFAULT_SEED
from ..errors import ComputeError, InvalidParamError
from ..models.loader import load_model
from .distributions import next_token_distribution
from .printable import printable_reference_ids, printable_tokens

ProgressCb = Callable[[float, str], None]


def _reference_with_top_emitters(
    lm: Any,
    all_ids: np.ndarray,
    all_strs: list[str],
    reference_set_size: int | None,
    probs_list: list[np.ndarray],
    top_k: int = 64,
) -> tuple[np.ndarray, list[str]]:
    """Reference markers = the evenly-spaced background subset UNION the top printable
    tokens of every supplied next-token distribution.

    Without the union the warp targets whatever happens to fall in the arbitrary
    evenly-spaced subset: with a 2000-marker subset over a ~150k vocab the true top
    token (e.g. " Paris" at 30%) was usually absent, so the surface bulged toward
    ~0.1%-probability tokens instead (red-team CRITICAL finding). The union guarantees
    every token the model actually reaches for is a marker the surface can bulge toward.
    """
    if reference_set_size is None or int(reference_set_size) >= all_ids.shape[0]:
        return all_ids, all_strs
    base = printable_reference_ids(lm, reference_set_size)
    tops = [all_ids[np.argsort(-probs[all_ids])[: int(top_k)]] for probs in probs_list]
    token_ids = np.unique(np.concatenate([base, *tops]))
    id2s = dict(zip(all_ids.tolist(), all_strs))
    token_strs = [id2s[int(t)] for t in token_ids]
    return token_ids, token_strs


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


def _surface_field(lm, prefix_list, src_token_ids, printable_ids):
    """For each source token, the model's most-likely PRINTABLE next token given the prompt +
    that token, and its probability — a flow field of "from here, the model goes there"."""
    import torch

    vocab = lm.model.get_output_embeddings().weight.shape[0]
    pmask = torch.zeros(int(vocab), dtype=torch.bool, device=lm.device)
    pmask[torch.as_tensor(sorted(printable_ids), device=lm.device, dtype=torch.long)] = True
    neg = torch.finfo(torch.float32).min
    tgt, prob = [], []
    bs = 64
    for s in range(0, len(src_token_ids), bs):
        chunk = src_token_ids[s : s + bs]
        ids = torch.tensor(
            [list(prefix_list) + [int(tk)] for tk in chunk], dtype=torch.long, device=lm.device
        )
        with torch.no_grad():
            logits = lm.model(input_ids=ids).logits[:, -1, :].float()
        p = torch.softmax(logits.masked_fill(~pmask.unsqueeze(0), neg), dim=-1)
        pv, pi = p.max(dim=-1)
        tgt.extend(int(x) for x in pi.cpu().numpy())
        prob.extend(float(x) for x in pv.cpu().numpy())
    return tgt, prob


def manifold(
    model_id: str,
    prefix_text: str = "",
    temperature: float = 1.0,
    reference_set_size: int | None = None,  # None = a dot for EVERY vocab token
    seed: int = DEFAULT_SEED,
    width: float = DEFAULT_RBF_WIDTH,  # RBF cap width on the UNIT sphere (small = localized domes)
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

    # Emission distribution FIRST: both the reference-set union (the markers must
    # include the tokens the model actually predicts) and the surface flow field need it.
    if progress_cb:
        progress_cb(0.2, "computing emission distribution")
    probs_full = next_token_distribution(
        model_id,
        prefix_text=prefix_text,
        temperature=temperature,
        response_text=response_text,
        response_step=response_step,
    )["arrays"]["probs"]

    token_ids, token_strs = _reference_with_top_emitters(
        lm, all_ids, all_strs, reference_set_size, [probs_full]
    )
    matrix = (
        lm.model.get_input_embeddings().weight.detach().float().cpu().numpy().astype(np.float64)
    )
    emb = matrix[token_ids]

    printable_set = set(int(i) for i in all_ids.tolist())
    resp_ids = [
        int(t)
        for t in (lm.tokenizer(response_text)["input_ids"] if response_text else [])
        if int(t) in printable_set
    ]

    # TRUE emission probabilities are what we report (tooltips/captions must never show
    # a subset-max-normalized "100%"); the geometry alone uses a normalized copy so the
    # warp amplitude stays visually strong regardless of the distribution's peak.
    emis = probs_full[token_ids].astype(np.float64)
    emis_geom = emis / emis.max() if emis.max() > 0 else emis

    # Surface flow field: for the top emitting tokens, the model's most-likely NEXT token —
    # "given an embedding here, where on the manifold does the model go next?"
    from .context import prefix_ids as _prefix_ids

    n_ref = int(token_ids.shape[0])
    surface_k = min(40, n_ref)
    src_order = np.argsort(-emis)[:surface_k]
    src_token_ids = [int(token_ids[i]) for i in src_order]
    if progress_cb:
        progress_cb(0.45, "tracing the surface flow field")
    tgt_ids, tgt_probs = _surface_field(
        lm, _prefix_ids(lm, prefix_text), src_token_ids, printable_set
    )
    uniq_tgt = sorted(set(tgt_ids))
    tgt_row = {tk: i for i, tk in enumerate(uniq_tgt)}

    # Reduce reference tokens + response tokens + surface targets TOGETHER so they share one
    # sphere frame, then split out each group's radius-2 positions.
    from ..reduce.sphere import reduce_3d_sphere

    stack = [emb]
    if resp_ids:
        stack.append(matrix[resp_ids])
    if uniq_tgt:
        stack.append(matrix[uniq_tgt])
    dirs_all = reduce_3d_sphere(np.vstack(stack), method="pca3", seed=seed).astype(np.float64)
    dirs = dirs_all[:n_ref]
    off = n_ref
    if resp_ids:
        traj_dirs = dirs_all[off : off + len(resp_ids)]
        off += len(resp_ids)
    else:
        traj_dirs = np.empty((0, 3), dtype=np.float64)
    tgt_dirs = (
        dirs_all[off : off + len(uniq_tgt)] if uniq_tgt else np.empty((0, 3), dtype=np.float64)
    )
    surf_src = dirs[src_order] * 2.0
    surf_dst = (
        (np.array([tgt_dirs[tgt_row[tk]] for tk in tgt_ids]) * 2.0)
        if uniq_tgt
        else np.empty((0, 3))
    )

    if progress_cb:
        progress_cb(0.6, "warping the sphere toward likely tokens")
    final_verts, faces, warp = _warp_sphere(o3d, dirs, emis_geom, width, warp_top)

    if progress_cb:
        progress_cb(1.0, "done")

    order = np.argsort(-emis)[: max(1, int(warp_top))]
    top_list = np.argsort(-emis)[:25]
    meta = {
        "model_id": lm.model_id,
        "revision": lm.revision,
        "prefix_text": prefix_text or "",
        "temperature": float(temperature),
        "n_vertices": int(final_verts.shape[0]),
        "n_faces": int(faces.shape[0]),
        "warp_top": int(len(order)),
        "top_tokens": [{"token_str": token_strs[int(i)], "prob": float(emis[i])} for i in top_list],
        "token_strs": token_strs,  # real decoded strings (printable), aligned with token markers
        "surface_src_strs": [token_strs[int(i)] for i in src_order],
        "surface_dst_strs": [lm.tokenizer.decode([t]) for t in tgt_ids],
        "surface_probs": [round(float(p), 5) for p in tgt_probs],
    }
    arrays = {
        "vertices": final_verts.astype(np.float32),
        "faces": faces.astype(np.int64),
        "warp": warp.astype(np.float32),
        "token_points": (dirs * 2.0).astype(np.float32),
        "token_emis": emis.astype(np.float32),
        "token_ids": token_ids.astype(np.int64),
        "traj_points": (traj_dirs * 2.0).astype(np.float32),  # response tokens on the sphere
        # surface flow field: from a likely token (src) to its predicted next token (dst)
        "surface_src": surf_src.astype(np.float32),
        "surface_dst": surf_dst.astype(np.float32),
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
    width: float = DEFAULT_RBF_WIDTH,
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
    matrix = (
        lm.model.get_input_embeddings().weight.detach().float().cpu().numpy().astype(np.float64)
    )
    printable_set = set(int(i) for i in all_ids.tolist())
    resp_ids = [
        int(t)
        for t in (lm.tokenizer(response_text)["input_ids"] if response_text else [])
        if int(t) in printable_set
    ]

    from ..reduce.sphere import reduce_3d_sphere
    from .context import prefix_ids as _prefix_ids

    n_frames = len(resp_ids) + 1
    prefix = _prefix_ids(lm, prefix_text)

    # Pass 0 — every frame's next-token distribution up front: the reference set must be
    # FIXED across frames (the mesh morphs between identical marker sets), and it must
    # contain the top emitters of EVERY frame so each frame's bulges target real tokens.
    probs_per: list[np.ndarray] = []
    for s in range(n_frames):
        if progress_cb:
            progress_cb(0.02 + 0.1 * s / n_frames, f"frame {s + 1}/{n_frames}: distribution")
        probs_per.append(
            next_token_distribution(
                model_id,
                prefix_text=prefix_text,
                temperature=temperature,
                response_text=response_text,
                response_step=s,
            )["arrays"]["probs"]
        )
    token_ids, token_strs = _reference_with_top_emitters(
        lm, all_ids, all_strs, reference_set_size, probs_per
    )
    emb = matrix[token_ids]
    n_ref = int(token_ids.shape[0])
    surface_k = min(40, n_ref)

    # Pass 1 — per-frame emission + surface flow field: as the context unfolds, the top emitting
    # tokens and where each one would lead NEXT both change, so we recompute them per key frame.
    # Every predicted target is collected so it can be placed in the SAME sphere frame below.
    # `emis_per` keeps TRUE probabilities (reported); the warp normalizes per frame itself.
    emis_per, src_order_per, tgt_ids_per, tgt_prob_per = [], [], [], []
    for s in range(n_frames):
        if progress_cb:
            progress_cb(
                0.12 + 0.38 * s / n_frames, f"frame {s + 1}/{n_frames}: emission + flow field"
            )
        emis = probs_per[s][token_ids].astype(np.float64)
        emis_per.append(emis)
        order = np.argsort(-emis)[:surface_k]
        src_order_per.append(order)
        ctx = list(prefix) + resp_ids[:s]
        tids, tprobs = _surface_field(lm, ctx, [int(token_ids[i]) for i in order], printable_set)
        tgt_ids_per.append(tids)
        tgt_prob_per.append(tprobs)

    uniq_tgt = sorted(set(t for fr in tgt_ids_per for t in fr))
    tgt_row = {tk: i for i, tk in enumerate(uniq_tgt)}

    # One reduction: reference markers + response tokens + every surface target share a frame.
    stack = [emb]
    if resp_ids:
        stack.append(matrix[resp_ids])
    if uniq_tgt:
        stack.append(matrix[uniq_tgt])
    dirs_all = reduce_3d_sphere(np.vstack(stack), method="pca3", seed=seed).astype(np.float64)
    dirs = dirs_all[:n_ref]
    off = n_ref
    if resp_ids:
        traj_dirs = dirs_all[off : off + len(resp_ids)]
        off += len(resp_ids)
    else:
        traj_dirs = np.empty((0, 3), dtype=np.float64)
    tgt_dirs = (
        dirs_all[off : off + len(uniq_tgt)] if uniq_tgt else np.empty((0, 3), dtype=np.float64)
    )

    # Pass 2 — warp the sphere per frame + assemble that frame's surface arrows.
    verts_per, warp_per, faces = [], [], None
    surf_src_per, surf_dst_per, surf_srcstr_per, surf_dststr_per, surf_prob_per = [], [], [], [], []
    for s in range(n_frames):
        if progress_cb:
            progress_cb(0.5 + 0.45 * s / n_frames, f"key frame {s + 1}/{n_frames}: warping")
        eg = emis_per[s]
        eg = eg / eg.max() if eg.max() > 0 else eg  # geometry-only normalization
        fv, faces, warp = _warp_sphere(o3d, dirs, eg, width, warp_top)
        verts_per.append(fv.astype(np.float32))
        warp_per.append(warp.astype(np.float32))
        order = src_order_per[s]
        tids = tgt_ids_per[s]
        surf_src_per.append((dirs[order] * 2.0).astype(np.float32))
        surf_dst_per.append(
            (np.array([tgt_dirs[tgt_row[t]] for t in tids]) * 2.0).astype(np.float32)
        )
        surf_srcstr_per.append([token_strs[int(i)] for i in order])
        surf_dststr_per.append([lm.tokenizer.decode([t]) for t in tids])
        surf_prob_per.append([round(float(p), 5) for p in tgt_prob_per[s]])

    meta = {
        "model_id": lm.model_id,
        "revision": lm.revision,
        "n_frames": int(n_frames),
        "n_vertices": int(verts_per[0].shape[0]),
        "prefix_text": prefix_text or "",
        "token_strs": token_strs,
        "trajectory_token_strs": [lm.tokenizer.decode([t]) for t in resp_ids],
        "surface_src_strs": surf_srcstr_per,  # (F)(K) per-frame source token strings
        "surface_dst_strs": surf_dststr_per,  # (F)(K) per-frame predicted-next token strings
        "surface_probs": surf_prob_per,  # (F)(K)
    }
    arrays = {
        "faces": faces.astype(np.int64),  # static
        "token_points": (dirs * 2.0).astype(np.float32),  # static
        "traj_points": (traj_dirs * 2.0).astype(np.float32),  # static
        "vertices": np.stack(verts_per),  # (F, V, 3) — morph these
        "warp": np.stack(warp_per),  # (F, V)
        "token_emis": np.stack(emis_per).astype(np.float32),  # (F, R) — marker alpha/size
        "surface_src": np.stack(surf_src_per),  # (F, K, 3)
        "surface_dst": np.stack(surf_dst_per),  # (F, K, 3)
    }
    if progress_cb:
        progress_cb(1.0, "done")
    return {"meta": meta, "arrays": arrays}
