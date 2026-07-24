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


def _embed_layers_and_topk(
    lm, base_ids, tokens, layers, temperature, fan, pmask, progress_cb, lo, hi
):
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
        ids = torch.tensor(
            [list(base_ids) + [tok] for tok in chunk], dtype=torch.long, device=lm.device
        )
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
            progress_cb(
                lo + (hi - lo) * min(1.0, (s + len(chunk)) / n),
                f"reference points {s + len(chunk)}/{n}",
            )
    return embs, top_tokens, top_probs


def _embed_layer_only(lm, base_ids, tokens, layer, progress_cb, lo, hi):
    """Contextual hidden state at ``layer`` for each token (fed after the context)."""
    n = len(tokens)
    embs = np.empty((n, lm.hidden_size), dtype=np.float32)
    bs = max(1, int(EMBED_BATCH_SIZE))
    for s in range(0, n, bs):
        chunk = [int(x) for x in tokens[s : s + bs]]
        ids = torch.tensor(
            [list(base_ids) + [tok] for tok in chunk], dtype=torch.long, device=lm.device
        )
        with torch.no_grad():
            out = lm.model(input_ids=ids, output_hidden_states=True)
        embs[s : s + len(chunk)] = out.hidden_states[layer][:, -1, :].float().cpu().numpy()
        if progress_cb:
            progress_cb(
                lo + (hi - lo) * min(1.0, (s + len(chunk)) / n),
                f"predicted tokens {s + len(chunk)}/{n}",
            )
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

    # Position EVERYTHING in one layer's space (the readout layer `n_to`). Mixing layers
    # (start@n, end@m) makes early- and late-layer hidden states separate into different
    # regions, throwing the trajectory off the grid — so reference tokens, predicted tokens,
    # and the trajectory are all embedded at `n_to`.
    ref_embs, top_tokens, top_probs = _embed_layers_and_topk(
        lm, base, ref_ids, [n_to], temperature, fan, pmask, progress_cb, 0.05, 0.5
    )
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

    traj = (
        _trajectory_embeddings(lm, prefix_ids(lm, prefix_text), response_text, n_to)
        if response_text
        else None
    )
    traj_embs = traj[0] if traj is not None else np.empty((0, lm.hidden_size), dtype=np.float64)

    if progress_cb:
        progress_cb(0.8, "reducing the contextual embeddings (PCA) + spreading")
    from sklearn.decomposition import PCA

    from ..reduce.spread import flatten_density

    # ONE frame fit on the CONTEXTUAL layer-`to` embeddings we actually plot (reference
    # tokens, their predicted tokens, the trajectory) — so the trajectory lands in-frame and
    # spreads, never crammed into a corner or off the grid.
    fit_on = np.vstack([emb_to.astype(np.float64), pred_embs, traj_embs])
    pca = PCA(n_components=2, random_state=seed).fit(fit_on)
    for i in range(2):  # deterministic sign so the frame is stable run-to-run
        if pca.components_[i].sum() < 0:
            pca.components_[i] *= -1.0

    n_ref, n_pred = emb_to.shape[0], pred_embs.shape[0]
    raw = np.vstack(
        [
            pca.transform(emb_to.astype(np.float64)),
            pca.transform(pred_embs),
            pca.transform(traj_embs) if traj is not None else np.empty((0, 2)),
        ]
    )
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
        "model_id": lm.model_id,
        "revision": lm.revision,
        "grid_n": grid_n,
        "layer_from": n_from,
        "layer_to": n_to,
        "num_layers": lm.num_layers,
        "temperature": float(temperature),
        "fanout": fan,
        "prefix_text": prefix_text or "",
        "count": len(starts),
        "reference_points": int(grid_vertices.shape[0]),
        "response_step": int(response_step),
        "spread_mu": float(spread_mu),
        "seed": int(seed),
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


def vector_field_animation(
    model_id: str,
    prefix_text: str = "",
    temperature: float = 1.0,
    layer_to: int | None = None,
    reference_set_size: int | None = 576,
    seed: int = DEFAULT_SEED,
    grid_n: int = DEFAULT_GRID_N,
    response_text: str = "",
    progress_cb: ProgressCb | None = None,
) -> dict[str, Any]:
    """All KEY FRAMES of the response animation over ONE STATIC grid.

    A key frame is a response step (context = prefix + response[:s]). A single PCA frame is fit
    on the union of all frames' contextual embeddings, and a regular n×n grid of fixed vertices
    is laid over it ONCE. The grid vertices never move; what changes frame to frame is (a) which
    reference token is nearest each vertex — the token that location "refers to" right now — and
    (b) the arrow that vertex casts toward that token's predicted next token. So you watch the
    flow field re-organise as the context unfolds, while the lattice stays put for continuity.
    """
    lm = load_model(model_id)
    n_to = lm.num_layers if layer_to is None else max(0, min(int(layer_to), lm.num_layers))
    ref_ids = printable_reference_ids(lm, reference_set_size)
    pmask = _printable_mask(lm)
    prefix = prefix_ids(lm, prefix_text)
    printable_set = set(int(i) for i in printable_tokens(lm)[0].tolist())
    resp_ids = [
        int(t)
        for t in (lm.tokenizer(response_text)["input_ids"] if response_text else [])
        if int(t) in printable_set
    ]
    n_frames = len(resp_ids) + 1
    R = int(ref_ids.shape[0])

    # Per key frame: reference embeddings @ layer_to + the (printable) top-1 prediction + the
    # predicted tokens' embeddings @ layer_to (reuse the reference ones where they coincide).
    per_frame = []
    for s in range(n_frames):
        if progress_cb:
            progress_cb(0.05 + 0.7 * s / n_frames, f"key frame {s + 1}/{n_frames}")
        ctx = list(prefix) + resp_ids[:s]
        embs, top_tokens, top_probs = _embed_layers_and_topk(
            lm, ctx, ref_ids, [n_to], temperature, 1, pmask, None, 0, 0
        )
        emb_to = embs[n_to]
        by_tok = {int(ref_ids[i]): emb_to[i] for i in range(R)}
        predicted = np.unique(top_tokens)
        missing = np.array([p for p in predicted if int(p) not in by_tok], dtype=np.int64)
        if missing.size:
            miss = _embed_layer_only(lm, ctx, missing, n_to, None, 0, 0)
            for i, p in enumerate(missing):
                by_tok[int(p)] = miss[i]
        pred_emb = np.vstack(
            [by_tok[int(top_tokens[i, 0])] for i in range(R)]
        )  # (R, H) predicted-token emb per ref
        per_frame.append((emb_to, pred_emb, top_probs[:, 0], top_tokens[:, 0]))

    traj = _trajectory_embeddings(lm, prefix, response_text, n_to) if response_text else None

    if progress_cb:
        progress_cb(0.8, "projecting all key frames into one consistent frame")
    from sklearn.decomposition import PCA
    from scipy.spatial import cKDTree

    fit = [f[0] for f in per_frame] + [f[1] for f in per_frame]
    if traj is not None:
        fit.append(traj[0])
    pca = PCA(n_components=2, random_state=seed).fit(np.vstack(fit).astype(np.float64))
    for i in range(2):  # deterministic sign so the frame is stable
        if pca.components_[i].sum() < 0:
            pca.components_[i] *= -1.0

    # The reference CLOUD moves with the context; we use it only to decide which token is
    # nearest each (fixed) grid vertex per frame and which way that vertex's arrow points.
    ref_pos = [pca.transform(f[0].astype(np.float64)) for f in per_frame]  # list of (R, 2)
    pred_pos = [pca.transform(f[1].astype(np.float64)) for f in per_frame]  # list of (R, 2)

    # ONE static grid over the union extent — these vertices never move.
    allref = np.vstack(ref_pos)
    lo = np.percentile(allref, 1, axis=0)
    hi = np.percentile(allref, 99, axis=0)
    gn = max(2, int(grid_n))
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], gn), np.linspace(lo[1], hi[1], gn))
    grid = np.column_stack([gx.ravel(), gy.ravel()]).astype(np.float32)  # (G, 2) STATIC
    G = int(grid.shape[0])
    cell = min((hi[0] - lo[0]) / (gn - 1), (hi[1] - lo[1]) / (gn - 1))
    arrow_len = 0.85 * float(cell)

    from_tok = np.empty((n_frames, G), dtype=np.int64)
    to_tok = np.empty((n_frames, G), dtype=np.int64)
    dirs = np.empty((n_frames, G, 2), dtype=np.float32)
    gprob = np.empty((n_frames, G), dtype=np.float32)
    for s in range(n_frames):
        _, nn = cKDTree(ref_pos[s]).query(grid, k=1)  # nearest ref token at each vertex
        d = pred_pos[s][nn] - ref_pos[s][nn]  # that token's ref→prediction flow
        norm = np.hypot(d[:, 0], d[:, 1])
        u = np.where(norm[:, None] > 1e-9, d / np.maximum(norm[:, None], 1e-9), 0.0)
        dirs[s] = u.astype(np.float32)
        from_tok[s] = ref_ids[nn]
        to_tok[s] = per_frame[s][3][nn]
        gprob[s] = per_frame[s][2][nn]

    traj_pos = (
        pca.transform(traj[0]).astype(np.float32)
        if traj is not None
        else np.empty((0, 2), dtype=np.float32)
    )

    # token id → decoded string, for the union of every token that labels a vertex in any frame
    involved = set(int(t) for t in ref_ids.tolist()) | set(
        int(t) for t in np.unique(to_tok).tolist()
    )
    token_strs = {str(t): lm.tokenizer.decode([t]) for t in involved}

    meta = {
        "model_id": lm.model_id,
        "revision": lm.revision,
        "n_frames": int(n_frames),
        "layer_to": int(n_to),
        "num_layers": lm.num_layers,
        "grid_n": gn,
        "reference_points": G,
        "arrow_len": arrow_len,
        "temperature": float(temperature),
        "prefix_text": prefix_text or "",
        "token_strs": token_strs,  # id (as str) -> decoded token
        "trajectory_token_strs": (traj[2] if traj is not None else []),
    }
    arrays = {
        "grid": grid,  # (G, 2) static vertices
        "from_tokens": from_tok,  # (F, G) nearest ref token id per vertex, per frame
        "to_tokens": to_tok,  # (F, G) its predicted next token id
        "dirs": dirs,  # (F, G, 2) unit arrow direction per vertex
        "probs": gprob,  # (F, G) top-1 probability
        "trajectory": traj_pos,
        "trajectory_probs": (traj[1] if traj is not None else np.empty((0,), dtype=np.float32)),
    }
    if progress_cb:
        progress_cb(1.0, "done")
    return {"meta": meta, "arrays": arrays}
