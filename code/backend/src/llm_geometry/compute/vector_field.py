"""Visualization 1 — transformer layers as a vector field (quiver).

An n×n grid is laid over the 2D-reduced (PCA) input-embedding space; each grid vertex
snaps to its nearest token (a "reference point"). Conditioned on the current context,
we predict each reference token's most-likely next token and draw an arrow from the
reference token's location to the predicted token's location, projected into the SAME
PCA space. Arrow opacity encodes the prediction probability. (project_description.md §1)
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch

from ..config import DEFAULT_GRID_N, DEFAULT_REFERENCE_SET_SIZE, DEFAULT_SEED, EMBED_BATCH_SIZE
from ..errors import InvalidParamError
from ..models.loader import load_model
from .embeddings import reference_token_ids

ProgressCb = Callable[[float, str], None]


def _context_ids(lm, prefix_text: str) -> list[int]:
    if prefix_text:
        return list(lm.tokenizer(prefix_text)["input_ids"])
    start = lm.tokenizer.bos_token_id or lm.tokenizer.eos_token_id or 0
    return [int(start)]


def _predict_next(lm, base_ids, tokens, temperature, progress_cb):
    n = len(tokens)
    nxt = np.empty(n, dtype=np.int64)
    prob = np.empty(n, dtype=np.float32)
    t = max(float(temperature), 1e-6)
    bs = max(1, int(EMBED_BATCH_SIZE))
    for s in range(0, n, bs):
        chunk = [int(x) for x in tokens[s : s + bs]]
        seqs = [list(base_ids) + [tok] for tok in chunk]
        ids = torch.tensor(seqs, dtype=torch.long, device=lm.device)
        with torch.no_grad():
            logits = lm.model(input_ids=ids).logits[:, -1, :].float()
        probs = torch.softmax(logits / t, dim=-1)
        top = torch.argmax(probs, dim=-1)
        nxt[s : s + len(chunk)] = top.cpu().numpy()
        prob[s : s + len(chunk)] = probs[torch.arange(len(chunk)), top].cpu().numpy()
        if progress_cb:
            progress_cb(min(1.0, (s + len(chunk)) / n), f"predicting {s + len(chunk)}/{n}")
    return nxt, prob


def vector_field(
    model_id: str,
    prefix_text: str = "",
    temperature: float = 1.0,
    grid_n: int = DEFAULT_GRID_N,
    reference_set_size: int | None = DEFAULT_REFERENCE_SET_SIZE,
    seed: int = DEFAULT_SEED,
    progress_cb: ProgressCb | None = None,
) -> dict[str, Any]:
    if grid_n < 2:
        raise InvalidParamError(f"grid_n must be >= 2, got {grid_n}")
    if temperature < 0:
        raise InvalidParamError(f"temperature must be >= 0, got {temperature}")

    lm = load_model(model_id)
    token_ids = reference_token_ids(lm, reference_set_size)
    matrix = lm.model.get_input_embeddings().weight.detach().cpu().numpy().astype(np.float64)
    emb = matrix[token_ids]

    from sklearn.decomposition import PCA

    pca = PCA(n_components=2, random_state=seed).fit(emb)
    coords = pca.transform(emb)  # [N, 2]

    if progress_cb:
        progress_cb(0.15, "laying grid + reference points")
    from scipy.spatial import cKDTree

    xmin, ymin = coords.min(0)
    xmax, ymax = coords.max(0)
    xs = np.linspace(xmin, xmax, grid_n)
    ys = np.linspace(ymin, ymax, grid_n)
    gx, gy = np.meshgrid(xs, ys)
    vertices = np.column_stack([gx.ravel(), gy.ravel()])
    _, ref_idx = cKDTree(coords).query(vertices, k=1)  # nearest reference index per vertex
    uniq = np.unique(ref_idx)

    base = _context_ids(lm, prefix_text)
    nxt_u, prob_u = _predict_next(
        lm, base, token_ids[uniq], temperature,
        (lambda p, m: progress_cb(0.25 + 0.65 * p, m)) if progress_cb else None,
    )
    pos = {int(u): i for i, u in enumerate(uniq)}
    nxt_coords = pca.transform(matrix[nxt_u])  # predicted tokens in the same space

    u_for_vertex = np.array([pos[int(r)] for r in ref_idx])
    starts = coords[ref_idx].astype(np.float32)
    ends = nxt_coords[u_for_vertex].astype(np.float32)
    probs = prob_u[u_for_vertex].astype(np.float32)
    start_tokens = token_ids[ref_idx].astype(np.int64)
    end_tokens = nxt_u[u_for_vertex].astype(np.int64)

    if progress_cb:
        progress_cb(1.0, "done")

    meta = {
        "model_id": lm.model_id, "revision": lm.revision, "grid_n": grid_n,
        "temperature": float(temperature), "prefix_text": prefix_text or "",
        "count": int(starts.shape[0]),
        "start_token_strs": [lm.tokenizer.decode([int(t)]) for t in start_tokens],
        "end_token_strs": [lm.tokenizer.decode([int(t)]) for t in end_tokens],
    }
    arrays = {
        "starts": starts, "ends": ends, "probs": probs,
        "start_tokens": start_tokens, "end_tokens": end_tokens,
    }
    return {"meta": meta, "arrays": arrays}
