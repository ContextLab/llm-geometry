"""Visualization 1 — transformer layers as a vector field (quiver). (project_description.md §1)

The positions are CONTEXTUAL prediction-layer embeddings — the hidden state a token has at
layer ``m`` *given the current context* (the representation just before the LM head turns it
into next-token probabilities), NOT the context-independent input embedding. That is the
whole phenomenon: change the prompt and a token's representation (and where the response
trajectory lands) moves.

A single PCA is fit on those contextual embeddings (reference tokens at layer ``from``, the
tokens they predict at layer ``to``, and the response trajectory at layer ``to``) and the
points are spread (mapper-style density flattening) so they fill the plane instead of
clumping — fitting the frame ON the contextual embeddings is what avoids the corner-cramming
you get if you project them through an unrelated (static-embedding) frame. A regular grid is
laid over the result; each grid arrow points from its nearest reference token (layer ``from``)
toward the token it predicts next (layer ``to``).

- LAYER RANGE: arrow start = reference token @ layer ``from``; end = predicted token @ layer ``to``.
- TEMPERATURE fan-out: temperature > 0 draws the top-`fanout` predicted arrows per grid cell.
- RESPONSE trajectory: response tokens' layer-``to`` hidden states (context-dependent), spread
  into the same frame; effective grid context = prefix + response[:response_step].
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch

from ..config import DEFAULT_GRID_N, DEFAULT_REFERENCE_SET_SIZE, DEFAULT_SEED, EMBED_BATCH_SIZE
from ..errors import InvalidParamError
from ..models.loader import load_model
from .context import effective_context_ids, prefix_ids
from .printable import printable_reference_ids, printable_tokens

ProgressCb = Callable[[float, str], None]


def _printable_mask(lm) -> torch.Tensor:
    vocab = lm.model.get_output_embeddings().weight.shape[0]
    mask = torch.zeros(int(vocab), dtype=torch.bool, device=lm.device)
    ids, _ = printable_tokens(lm)
    mask[torch.as_tensor(ids, device=lm.device, dtype=torch.long)] = True
    return mask


def _embed_layers_and_topk(lm, base_ids, tokens, layers, temperature, fan, pmask, progress_cb, lo, hi):
    """Per-token contextual hidden states at each requested layer + the top-`fan` PRINTABLE
    next tokens (the model's real prediction, conditioned on the context)."""
    n = len(tokens)
    embs = {L: np.empty((n, lm.hidden_size), dtype=np.float32) for L in layers}
    top_tokens = np.empty((n, fan), dtype=np.int64)
    top_probs = np.empty((n, fan), dtype=np.float32)
    t = max(float(temperature), 1e-6)
    neg = torch.finfo(torch.float32).min
    bs = max(1, int(EMBED_BATCH_SIZE))
    for s in range(0, n, bs):
        chunk = [int(x) for x in tokens[s : s + bs]]
        ids = torch.tensor([list(base_ids) + [tok] for tok in chunk], dtype=torch.long, device=lm.device)
        with torch.no_grad():
            out = lm.model(input_ids=ids, output_hidden_states=True)
        for L in layers:
            embs[L][s : s + len(chunk)] = out.hidden_states[L][:, -1, :].float().cpu().numpy()
        logits = (out.logits[:, -1, :].float() / t).masked_fill(~pmask.unsqueeze(0), neg)
        probs = torch.softmax(logits, dim=-1)
        tp, ti = torch.topk(probs, k=fan, dim=-1)
        top_tokens[s : s + len(chunk)] = ti.cpu().numpy()
        top_probs[s : s + len(chunk)] = tp.cpu().numpy()
        if progress_cb:
            progress_cb(lo + (hi - lo) * min(1.0, (s + len(chunk)) / n), f"reference points {s + len(chunk)}/{n}")
    return embs, top_tokens, top_probs


def _embed_layer_only(lm, base_ids, tokens, layer, progress_cb, lo, hi):
    """Contextual hidden state at ``layer`` for each token (fed after the context)."""
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
    """Each response token's CONTEXTUAL hidden state at ``layer`` (conditioned on the prefix),
    so a different prompt moves the trajectory."""
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
    spread_mu: float = 0.65,
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
    ref_ids = printable_reference_ids(lm, reference_set_size)
    pmask = _printable_mask(lm)

    ref_layers = sorted({n_from, n_to})
    ref_embs, top_tokens, top_probs = _embed_layers_and_topk(
        lm, base, ref_ids, ref_layers, temperature, fan, pmask, progress_cb, 0.05, 0.5
    )
    emb_from = ref_embs[n_from]
    emb_to = ref_embs[n_to]

    # Contextual layer-`to` embeddings of the predicted tokens (their representation when they
    # follow the same context); reuse the reference tokens where they coincide.
    to_by_token = {int(ref_ids[i]): emb_to[i] for i in range(len(ref_ids))}
    predicted = np.unique(top_tokens)
    missing = np.array([p for p in predicted if int(p) not in to_by_token], dtype=np.int64)
    if missing.size:
        miss = _embed_layer_only(lm, base, missing, n_to, progress_cb, 0.5, 0.74)
        for i, p in enumerate(missing):
            to_by_token[int(p)] = miss[i]
    pred_list = [int(p) for p in predicted]
    pred_row = {p: i for i, p in enumerate(pred_list)}
    pred_embs = np.vstack([to_by_token[p] for p in pred_list])

    traj = _trajectory_embeddings(lm, prefix_ids(lm, prefix_text), response_text, n_to) if response_text else None
    traj_embs = traj[0] if traj is not None else np.empty((0, lm.hidden_size), dtype=np.float64)

    if progress_cb:
        progress_cb(0.8, "reducing the contextual embeddings (PCA) + spreading")
    from sklearn.decomposition import PCA

    from ..reduce.spread import flatten_density

    # ONE frame fit on the CONTEXTUAL embeddings we actually plot — so the trajectory lands
    # in-frame and spreads, never crammed into a corner.
    fit_on = np.vstack([emb_from.astype(np.float64), pred_embs, traj_embs])
    pca = PCA(n_components=2, random_state=seed).fit(fit_on)
    for i in range(2):  # deterministic sign so the frame is stable run-to-run
        if pca.components_[i].sum() < 0:
            pca.components_[i] *= -1.0

    n_ref, n_pred = emb_from.shape[0], pred_embs.shape[0]
    raw = np.vstack([pca.transform(emb_from.astype(np.float64)), pca.transform(pred_embs),
                     pca.transform(traj_embs) if traj is not None else np.empty((0, 2))])
    flat = flatten_density(raw, mu=spread_mu, seed=seed)
    flat_start = flat[:n_ref]
    flat_pred = flat[n_ref : n_ref + n_pred]
    flat_traj = flat[n_ref + n_pred :]

    if progress_cb:
        progress_cb(0.9, "laying out the flow-field grid")
    from scipy.spatial import cKDTree

    lo = np.percentile(flat_start, 1, axis=0)
    hi = np.percentile(flat_start, 99, axis=0)
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], grid_n), np.linspace(lo[1], hi[1], grid_n))
    grid_vertices = np.column_stack([gx.ravel(), gy.ravel()])  # regular grid of fixed origins
    _, nn = cKDTree(flat_start).query(grid_vertices, k=1)
    cell = min((hi[0] - lo[0]) / (grid_n - 1), (hi[1] - lo[1]) / (grid_n - 1))
    arrow_len = 0.85 * float(cell)  # long bodies (most of a grid cell)

    starts, ends, probs, s_tokens, e_tokens = [], [], [], [], []
    for gi in range(grid_vertices.shape[0]):
        r = int(nn[gi])
        v = grid_vertices[gi]
        for f in range(fan):
            p_tok = int(top_tokens[r, f])
            d = flat_pred[pred_row[p_tok]] - flat_start[r]  # layer-from token → layer-to prediction
            norm = float(np.hypot(d[0], d[1]))
            u = d / norm if norm > 1e-9 else np.zeros(2)
            starts.append(v)
            ends.append(v + u * arrow_len)
            probs.append(float(top_probs[r, f]))
            s_tokens.append(int(ref_ids[r]))
            e_tokens.append(p_tok)

    meta = {
        "model_id": lm.model_id, "revision": lm.revision, "grid_n": grid_n,
        "layer_from": n_from, "layer_to": n_to, "num_layers": lm.num_layers,
        "temperature": float(temperature), "fanout": fan, "prefix_text": prefix_text or "",
        "count": len(starts), "reference_points": int(grid_vertices.shape[0]),
        "response_step": int(response_step), "spread_mu": float(spread_mu), "seed": int(seed),
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
