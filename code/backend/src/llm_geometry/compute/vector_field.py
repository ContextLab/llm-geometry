"""Visualization 1 — transformer layers as a vector field (quiver). (project_description.md §1)

For an n×n grid over the 2D-reduced (PCA) embedding space at a chosen LAYER, each grid
vertex snaps to its nearest token (a "reference point"). Conditioned on the context,
each reference token's top next tokens are predicted; an arrow is drawn from the
reference token's layer-L position to each predicted token's layer-L position.

- LAYER slider: positions use the contextual hidden state at ``layer`` (0 = input
  embedding, num_layers = final layer).
- TEMPERATURE fan-out: temperature > 0 draws multiple semi-transparent arrows per
  reference point (the top ``fanout`` next tokens, opacity ∝ probability); temperature
  = 0 draws the single argmax arrow.
- RESPONSE trajectory: an optional response string is traced as a path through the same
  space, each step colored by the model's probability of that token.
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


def _layer_embed_and_topk(lm, base_ids, tokens, layer, temperature, fan, progress_cb, lo, hi):
    """For each token t: forward(base+[t]); return its layer-L last-position hidden state
    and the top-``fan`` next tokens + probabilities."""
    n = len(tokens)
    embs = np.empty((n, lm.hidden_size), dtype=np.float32)
    top_tokens = np.empty((n, fan), dtype=np.int64)
    top_probs = np.empty((n, fan), dtype=np.float32)
    t = max(float(temperature), 1e-6)
    bs = max(1, int(EMBED_BATCH_SIZE))
    for s in range(0, n, bs):
        chunk = [int(x) for x in tokens[s : s + bs]]
        ids = torch.tensor([list(base_ids) + [tok] for tok in chunk], dtype=torch.long, device=lm.device)
        with torch.no_grad():
            out = lm.model(input_ids=ids, output_hidden_states=True)
        embs[s : s + len(chunk)] = out.hidden_states[layer][:, -1, :].float().cpu().numpy()
        probs = torch.softmax(out.logits[:, -1, :].float() / t, dim=-1)
        tp, ti = torch.topk(probs, k=fan, dim=-1)
        top_tokens[s : s + len(chunk)] = ti.cpu().numpy()
        top_probs[s : s + len(chunk)] = tp.cpu().numpy()
        if progress_cb:
            progress_cb(lo + (hi - lo) * min(1.0, (s + len(chunk)) / n), f"reference points {s + len(chunk)}/{n}")
    return embs, top_tokens, top_probs


def _layer_embed_only(lm, base_ids, tokens, layer, progress_cb, lo, hi):
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


def _trajectory(lm, base_ids, response_text, layer, pca):
    """Trace a response string through the same PCA space, colored by token probability."""
    resp = list(lm.tokenizer(response_text)["input_ids"])
    if not resp:
        return None
    full = list(base_ids) + resp
    ids = torch.tensor([full], dtype=torch.long, device=lm.device)
    with torch.no_grad():
        out = lm.model(input_ids=ids, output_hidden_states=True)
    hidden = out.hidden_states[layer][0]  # [seq, hidden]
    logits = out.logits[0]  # [seq, vocab]
    base_len = len(base_ids)
    embs, probs, strs = [], [], []
    for j, tok in enumerate(resp):
        pos = base_len + j
        embs.append(hidden[pos].float().cpu().numpy())
        pred_logits = logits[pos - 1].float()
        probs.append(float(torch.softmax(pred_logits, dim=-1)[tok].item()))
        strs.append(lm.tokenizer.decode([int(tok)]))
    coords = pca.transform(np.asarray(embs, dtype=np.float64)).astype(np.float32)
    return coords, np.asarray(probs, dtype=np.float32), strs


def vector_field(
    model_id: str,
    prefix_text: str = "",
    temperature: float = 1.0,
    layer: int = 0,
    grid_n: int = DEFAULT_GRID_N,
    reference_set_size: int | None = DEFAULT_REFERENCE_SET_SIZE,
    seed: int = DEFAULT_SEED,
    fanout: int = 4,
    response_text: str = "",
    progress_cb: ProgressCb | None = None,
) -> dict[str, Any]:
    if grid_n < 2:
        raise InvalidParamError(f"grid_n must be >= 2, got {grid_n}")
    if temperature < 0:
        raise InvalidParamError(f"temperature must be >= 0, got {temperature}")

    lm = load_model(model_id)
    if layer < 0 or layer > lm.num_layers:
        raise InvalidParamError(f"layer must be in 0..{lm.num_layers}, got {layer}")

    fan = max(1, int(fanout)) if temperature > 0 else 1
    base = _context_ids(lm, prefix_text)
    token_ids = reference_token_ids(lm, reference_set_size)

    start_emb, top_tokens, top_probs = _layer_embed_and_topk(
        lm, base, token_ids, layer, temperature, fan, progress_cb, 0.05, 0.55
    )

    # Embeddings of predicted tokens not already in the reference set (same layer/context).
    ref_emb_by_token = {int(token_ids[i]): start_emb[i] for i in range(len(token_ids))}
    predicted = np.unique(top_tokens)
    missing = np.array([p for p in predicted if int(p) not in ref_emb_by_token], dtype=np.int64)
    if missing.size:
        miss_emb = _layer_embed_only(lm, base, missing, layer, progress_cb, 0.55, 0.8)
        for i, p in enumerate(missing):
            ref_emb_by_token[int(p)] = miss_emb[i]

    if progress_cb:
        progress_cb(0.85, "projecting (PCA) + laying grid")
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2, random_state=seed).fit(start_emb.astype(np.float64))
    coords = pca.transform(start_emb.astype(np.float64))  # reference token coords [N,2]
    pred_coords = {tok: pca.transform(emb.astype(np.float64).reshape(1, -1))[0]
                   for tok, emb in ref_emb_by_token.items()}

    from scipy.spatial import cKDTree

    xmin, ymin = coords.min(0)
    xmax, ymax = coords.max(0)
    xs = np.linspace(xmin, xmax, grid_n)
    ys = np.linspace(ymin, ymax, grid_n)
    gx, gy = np.meshgrid(xs, ys)
    vertices = np.column_stack([gx.ravel(), gy.ravel()])
    _, ref_idx = cKDTree(coords).query(vertices, k=1)
    unique_ref = np.unique(ref_idx)  # reference points = distinct snapped tokens

    starts, ends, probs, s_tokens, e_tokens = [], [], [], [], []
    for r in unique_ref:
        s_coord = coords[r]
        for f in range(fan):
            p_tok = int(top_tokens[r, f])
            starts.append(s_coord)
            ends.append(pred_coords[p_tok])
            probs.append(float(top_probs[r, f]))
            s_tokens.append(int(token_ids[r]))
            e_tokens.append(p_tok)

    starts = np.asarray(starts, dtype=np.float32)
    ends = np.asarray(ends, dtype=np.float32)
    probs = np.asarray(probs, dtype=np.float32)

    meta = {
        "model_id": lm.model_id, "revision": lm.revision, "grid_n": grid_n, "layer": int(layer),
        "num_layers": lm.num_layers, "temperature": float(temperature), "fanout": fan,
        "prefix_text": prefix_text or "", "count": int(starts.shape[0]),
        "reference_points": int(unique_ref.shape[0]),
        "start_token_strs": [lm.tokenizer.decode([t]) for t in s_tokens],
        "end_token_strs": [lm.tokenizer.decode([t]) for t in e_tokens],
    }
    arrays = {
        "starts": starts, "ends": ends, "probs": probs,
        "start_tokens": np.asarray(s_tokens, dtype=np.int64),
        "end_tokens": np.asarray(e_tokens, dtype=np.int64),
    }

    if response_text:
        if progress_cb:
            progress_cb(0.92, "tracing response trajectory")
        traj = _trajectory(lm, base, response_text, layer, pca)
        if traj is not None:
            tcoords, tprobs, tstrs = traj
            arrays["trajectory"] = tcoords
            arrays["trajectory_probs"] = tprobs
            meta["trajectory_token_strs"] = tstrs

    if progress_cb:
        progress_cb(1.0, "done")
    return {"meta": meta, "arrays": arrays}
