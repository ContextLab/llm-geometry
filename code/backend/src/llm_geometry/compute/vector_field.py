"""Visualization 1 — transformer layers as a vector field (quiver). (project_description.md §1)

An n×n grid over the 2D-reduced embedding space snaps each vertex to its nearest token
(a "reference point"); conditioned on the context, each reference token's top next
tokens are predicted and drawn as arrows from the reference token's position to each
predicted token's position.

- LAYER(S): one or a RANGE of layers can be shown. All requested layers are projected
  into ONE shared PCA space (so they're comparable); arrows are tagged with their layer
  and color-coded, showing how the field evolves across depth.
- TEMPERATURE fan-out: temperature > 0 draws the top-`fanout` arrows per reference point
  (opacity ∝ probability); temperature 0 draws the single argmax arrow.
- RESPONSE trajectory + animation: an optional response is traced through the space; the
  effective context is prefix + response[:response_step], so stepping `response_step`
  animates how the field changes with each subsequent token.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
import torch

from ..config import DEFAULT_GRID_N, DEFAULT_REFERENCE_SET_SIZE, DEFAULT_SEED, EMBED_BATCH_SIZE
from ..errors import InvalidParamError
from ..models.loader import load_model
from .context import effective_context_ids, prefix_ids
from .embeddings import reference_token_ids

ProgressCb = Callable[[float, str], None]
MAX_LAYERS_SHOWN = 6


def _normalize_layers(layers: Sequence[int], num_layers: int) -> list[int]:
    ls = sorted({max(0, min(int(l), num_layers)) for l in layers})
    if not ls:
        ls = [0]
    if len(ls) > MAX_LAYERS_SHOWN:  # evenly sample to keep the overlay legible
        idx = np.unique(np.linspace(0, len(ls) - 1, MAX_LAYERS_SHOWN).astype(int))
        ls = [ls[i] for i in idx]
    return ls


def _embed_layers_and_topk(lm, base_ids, tokens, layers, temperature, fan, progress_cb, lo, hi):
    n = len(tokens)
    embs = {L: np.empty((n, lm.hidden_size), dtype=np.float32) for L in layers}
    top_tokens = np.empty((n, fan), dtype=np.int64)
    top_probs = np.empty((n, fan), dtype=np.float32)
    t = max(float(temperature), 1e-6)
    bs = max(1, int(EMBED_BATCH_SIZE))
    for s in range(0, n, bs):
        chunk = [int(x) for x in tokens[s : s + bs]]
        ids = torch.tensor([list(base_ids) + [tok] for tok in chunk], dtype=torch.long, device=lm.device)
        with torch.no_grad():
            out = lm.model(input_ids=ids, output_hidden_states=True)
        for L in layers:
            embs[L][s : s + len(chunk)] = out.hidden_states[L][:, -1, :].float().cpu().numpy()
        probs = torch.softmax(out.logits[:, -1, :].float() / t, dim=-1)
        tp, ti = torch.topk(probs, k=fan, dim=-1)
        top_tokens[s : s + len(chunk)] = ti.cpu().numpy()
        top_probs[s : s + len(chunk)] = tp.cpu().numpy()
        if progress_cb:
            progress_cb(lo + (hi - lo) * min(1.0, (s + len(chunk)) / n), f"reference points {s + len(chunk)}/{n}")
    return embs, top_tokens, top_probs


def _embed_layers_only(lm, base_ids, tokens, layers, progress_cb, lo, hi):
    n = len(tokens)
    embs = {L: np.empty((n, lm.hidden_size), dtype=np.float32) for L in layers}
    bs = max(1, int(EMBED_BATCH_SIZE))
    for s in range(0, n, bs):
        chunk = [int(x) for x in tokens[s : s + bs]]
        ids = torch.tensor([list(base_ids) + [tok] for tok in chunk], dtype=torch.long, device=lm.device)
        with torch.no_grad():
            out = lm.model(input_ids=ids, output_hidden_states=True)
        for L in layers:
            embs[L][s : s + len(chunk)] = out.hidden_states[L][:, -1, :].float().cpu().numpy()
        if progress_cb:
            progress_cb(lo + (hi - lo) * min(1.0, (s + len(chunk)) / n), f"predicted tokens {s + len(chunk)}/{n}")
    return embs


def _trajectory(lm, base_ids, response_text, layer, pca):
    resp = list(lm.tokenizer(response_text)["input_ids"])
    if not resp:
        return None
    ids = torch.tensor([list(base_ids) + resp], dtype=torch.long, device=lm.device)
    with torch.no_grad():
        out = lm.model(input_ids=ids, output_hidden_states=True)
    hidden = out.hidden_states[layer][0]
    logits = out.logits[0]
    base_len = len(base_ids)
    embs, probs, strs = [], [], []
    for j, tok in enumerate(resp):
        pos = base_len + j
        embs.append(hidden[pos].float().cpu().numpy())
        probs.append(float(torch.softmax(logits[pos - 1].float(), dim=-1)[tok].item()))
        strs.append(lm.tokenizer.decode([int(tok)]))
    coords = pca.transform(np.asarray(embs, dtype=np.float64)).astype(np.float32)
    return coords, np.asarray(probs, dtype=np.float32), strs


def vector_field(
    model_id: str,
    prefix_text: str = "",
    temperature: float = 1.0,
    layers: Sequence[int] = (0,),
    grid_n: int = DEFAULT_GRID_N,
    reference_set_size: int | None = DEFAULT_REFERENCE_SET_SIZE,
    seed: int = DEFAULT_SEED,
    fanout: int = 4,
    response_text: str = "",
    response_step: int = 0,
    progress_cb: ProgressCb | None = None,
) -> dict[str, Any]:
    if grid_n < 2:
        raise InvalidParamError(f"grid_n must be >= 2, got {grid_n}")
    if temperature < 0:
        raise InvalidParamError(f"temperature must be >= 0, got {temperature}")

    lm = load_model(model_id)
    layer_list = _normalize_layers(layers, lm.num_layers)
    fan = max(1, int(fanout)) if temperature > 0 else 1
    base = effective_context_ids(lm, prefix_text, response_text, response_step)
    token_ids = reference_token_ids(lm, reference_set_size)

    start_embs, top_tokens, top_probs = _embed_layers_and_topk(
        lm, base, token_ids, layer_list, temperature, fan, progress_cb, 0.05, 0.5
    )

    by_token = {L: {int(token_ids[i]): start_embs[L][i] for i in range(len(token_ids))} for L in layer_list}
    predicted = np.unique(top_tokens)
    missing = np.array([p for p in predicted if int(p) not in by_token[layer_list[0]]], dtype=np.int64)
    if missing.size:
        miss = _embed_layers_only(lm, base, missing, layer_list, progress_cb, 0.5, 0.75)
        for L in layer_list:
            for i, p in enumerate(missing):
                by_token[L][int(p)] = miss[L][i]

    if progress_cb:
        progress_cb(0.82, "projecting (shared PCA) + laying grid")
    from sklearn.decomposition import PCA

    stacked = np.vstack([start_embs[L] for L in layer_list]).astype(np.float64)
    pca = PCA(n_components=2, random_state=seed).fit(stacked)
    coords = {L: pca.transform(start_embs[L].astype(np.float64)) for L in layer_list}
    pred_coords = {
        L: {tok: pca.transform(emb.astype(np.float64).reshape(1, -1))[0] for tok, emb in by_token[L].items()}
        for L in layer_list
    }

    from scipy.spatial import cKDTree

    primary = layer_list[0]
    pc = coords[primary]
    xmin, ymin = pc.min(0)
    xmax, ymax = pc.max(0)
    gx, gy = np.meshgrid(np.linspace(xmin, xmax, grid_n), np.linspace(ymin, ymax, grid_n))
    vertices = np.column_stack([gx.ravel(), gy.ravel()])
    _, ref_idx = cKDTree(pc).query(vertices, k=1)
    unique_ref = np.unique(ref_idx)

    starts, ends, probs, s_tokens, e_tokens, arrow_layers = [], [], [], [], [], []
    for r in unique_ref:
        for L in layer_list:
            for f in range(fan):
                p_tok = int(top_tokens[r, f])
                starts.append(coords[L][r])
                ends.append(pred_coords[L][p_tok])
                probs.append(float(top_probs[r, f]))
                s_tokens.append(int(token_ids[r]))
                e_tokens.append(p_tok)
                arrow_layers.append(int(L))

    meta = {
        "model_id": lm.model_id, "revision": lm.revision, "grid_n": grid_n,
        "layers": layer_list, "layer": primary, "num_layers": lm.num_layers,
        "temperature": float(temperature), "fanout": fan, "prefix_text": prefix_text or "",
        "count": len(starts), "reference_points": int(unique_ref.shape[0]),
        "response_step": int(response_step),
        "start_token_strs": [lm.tokenizer.decode([t]) for t in s_tokens],
        "end_token_strs": [lm.tokenizer.decode([t]) for t in e_tokens],
    }
    arrays = {
        "starts": np.asarray(starts, dtype=np.float32),
        "ends": np.asarray(ends, dtype=np.float32),
        "probs": np.asarray(probs, dtype=np.float32),
        "start_tokens": np.asarray(s_tokens, dtype=np.int64),
        "end_tokens": np.asarray(e_tokens, dtype=np.int64),
        "arrow_layers": np.asarray(arrow_layers, dtype=np.int64),
    }

    if response_text:
        if progress_cb:
            progress_cb(0.92, "tracing response trajectory")
        traj = _trajectory(lm, prefix_ids(lm, prefix_text), response_text, primary, pca)
        if traj is not None:
            tcoords, tprobs, tstrs = traj
            arrays["trajectory"] = tcoords
            arrays["trajectory_probs"] = tprobs
            meta["trajectory_token_strs"] = tstrs

    if progress_cb:
        progress_cb(1.0, "done")
    return {"meta": meta, "arrays": arrays}
