"""Visualization 1 — transformer layers as a vector field (quiver). (project_description.md §1)

Each grid vertex snaps to its nearest token (a "reference point"). For a layer range
n..m, the arrow START is the reference token's embedding at layer n and the arrow END is
its most-likely next token's embedding at layer m (the doc's "input at the first layer,
response is from the last layer"). Both layers are projected into ONE shared PCA space,
then spread toward an even grid-like layout (mapper-style density flattening).

- LAYER RANGE (from n, to m): start = layer n, end = layer m.
- TEMPERATURE fan-out: temperature > 0 draws the top-`fanout` arrows per reference point.
- RESPONSE trajectory + animation: effective context = prefix + response[:response_step].
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch

from ..config import DEFAULT_GRID_N, DEFAULT_REFERENCE_SET_SIZE, DEFAULT_SEED, EMBED_BATCH_SIZE
from ..errors import InvalidParamError
from ..models.loader import load_model
from .context import effective_context_ids, prefix_ids
from .embeddings import reference_token_ids

ProgressCb = Callable[[float, str], None]


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


def _embed_layer_only(lm, base_ids, tokens, layer, progress_cb, lo, hi):
    n = len(tokens)
    embs = np.empty((n, lm.hidden_size), dtype=np.float32)
    bs = max(1, int(EMBED_BATCH_SIZE))
    for s in range(0, n, bs):
        chunk = [int(x) for x in tokens[s : s + bs]]
        ids = torch.tensor([list(base_ids) + [tok] for tok in chunk], dtype=torch.long, device=lm.device)
        with torch.no_grad():
            out = lm.model(input_ids=ids, output_hidden_states=True)
        embs[s : s + len(chunk)] = out.hidden_states[layer][:, -1, :].float().cpu().numpy()
        if progress_cb:
            progress_cb(lo + (hi - lo) * min(1.0, (s + len(chunk)) / n), f"predicted tokens {s + len(chunk)}/{n}")
    return embs


def _trajectory_embeddings(lm, base_ids, response_text, layer):
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
    return np.asarray(embs, dtype=np.float64), np.asarray(probs, dtype=np.float32), strs


def vector_field(
    model_id: str,
    prefix_text: str = "",
    temperature: float = 1.0,
    layer_from: int = 0,
    layer_to: int | None = None,
    grid_n: int = DEFAULT_GRID_N,
    reference_set_size: int | None = DEFAULT_REFERENCE_SET_SIZE,
    seed: int = DEFAULT_SEED,
    fanout: int = 4,
    spread_mu: float = 0.85,
    response_text: str = "",
    response_step: int = 0,
    progress_cb: ProgressCb | None = None,
) -> dict[str, Any]:
    if grid_n < 2:
        raise InvalidParamError(f"grid_n must be >= 2, got {grid_n}")
    if temperature < 0:
        raise InvalidParamError(f"temperature must be >= 0, got {temperature}")

    lm = load_model(model_id)
    n_from = max(0, min(int(layer_from), lm.num_layers))
    n_to = n_from if layer_to is None else max(0, min(int(layer_to), lm.num_layers))
    fan = max(1, int(fanout)) if temperature > 0 else 1
    base = effective_context_ids(lm, prefix_text, response_text, response_step)
    token_ids = reference_token_ids(lm, reference_set_size)

    ref_layers = sorted({n_from, n_to})
    ref_embs, top_tokens, top_probs = _embed_layers_and_topk(
        lm, base, token_ids, ref_layers, temperature, fan, progress_cb, 0.05, 0.5
    )
    emb_from = ref_embs[n_from]
    emb_to = ref_embs[n_to]

    # Embeddings of predicted tokens at the TO layer (end positions).
    to_by_token = {int(token_ids[i]): emb_to[i] for i in range(len(token_ids))}
    predicted = np.unique(top_tokens)
    missing = np.array([p for p in predicted if int(p) not in to_by_token], dtype=np.int64)
    if missing.size:
        miss = _embed_layer_only(lm, base, missing, n_to, progress_cb, 0.5, 0.78)
        for i, p in enumerate(missing):
            to_by_token[int(p)] = miss[i]

    if progress_cb:
        progress_cb(0.82, "projecting (shared PCA) + spreading")
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2, random_state=seed).fit(np.vstack([emb_from, emb_to]).astype(np.float64))
    start_coords = pca.transform(emb_from.astype(np.float64))
    pred_list = [int(p) for p in predicted]
    pred_raw = pca.transform(np.vstack([to_by_token[p] for p in pred_list]).astype(np.float64))
    pred_row = {p: i for i, p in enumerate(pred_list)}

    traj = _trajectory_embeddings(lm, prefix_ids(lm, prefix_text), response_text, n_to) if response_text else None
    traj_raw = pca.transform(traj[0]) if traj is not None else np.empty((0, 2))

    # Spread the whole field toward an even grid-like layout (mapper density flattening).
    from ..reduce.spread import flatten_density

    n_ref = start_coords.shape[0]
    n_pred = pred_raw.shape[0]
    combined = np.vstack([start_coords, pred_raw, traj_raw])
    flat = flatten_density(combined, mu=spread_mu, seed=seed)
    flat_start = flat[:n_ref]
    flat_pred = flat[n_ref : n_ref + n_pred]
    flat_traj = flat[n_ref + n_pred :]

    from scipy.spatial import cKDTree

    xmin, ymin = flat_start.min(0)
    xmax, ymax = flat_start.max(0)
    gx, gy = np.meshgrid(np.linspace(xmin, xmax, grid_n), np.linspace(ymin, ymax, grid_n))
    vertices = np.column_stack([gx.ravel(), gy.ravel()])
    _, ref_idx = cKDTree(flat_start).query(vertices, k=1)
    unique_ref = np.unique(ref_idx)

    starts, ends, probs, s_tokens, e_tokens = [], [], [], [], []
    for r in unique_ref:
        for f in range(fan):
            p_tok = int(top_tokens[r, f])
            starts.append(flat_start[r])
            ends.append(flat_pred[pred_row[p_tok]])
            probs.append(float(top_probs[r, f]))
            s_tokens.append(int(token_ids[r]))
            e_tokens.append(p_tok)

    meta = {
        "model_id": lm.model_id, "revision": lm.revision, "grid_n": grid_n,
        "layer_from": n_from, "layer_to": n_to, "num_layers": lm.num_layers,
        "temperature": float(temperature), "fanout": fan, "prefix_text": prefix_text or "",
        "count": len(starts), "reference_points": int(unique_ref.shape[0]),
        "response_step": int(response_step), "spread_mu": float(spread_mu),
        "start_token_strs": [lm.tokenizer.decode([t]) for t in s_tokens],
        "end_token_strs": [lm.tokenizer.decode([t]) for t in e_tokens],
    }
    arrays = {
        "starts": np.asarray(starts, dtype=np.float32),
        "ends": np.asarray(ends, dtype=np.float32),
        "probs": np.asarray(probs, dtype=np.float32),
        "start_tokens": np.asarray(s_tokens, dtype=np.int64),
        "end_tokens": np.asarray(e_tokens, dtype=np.int64),
    }
    if traj is not None:
        arrays["trajectory"] = flat_traj.astype(np.float32)
        arrays["trajectory_probs"] = traj[1]
        meta["trajectory_token_strs"] = traj[2]

    if progress_cb:
        progress_cb(1.0, "done")
    return {"meta": meta, "arrays": arrays}
