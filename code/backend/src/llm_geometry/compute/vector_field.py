"""Visualization 1 — transformer layers as a vector field (quiver). (project_description.md §1)

A macOS "Drift"-style flow field. Every token sits at its FIXED position in the cached
``token_cloud`` layout (a static-embedding PCA, density-flattened so the points spread out).
A regular grid of fixed origins is laid over that layout; each origin's arrow points, at a
UNIFORM length, from the nearest reference token toward the token the model predicts comes
next — read out at layer ``layer_to`` (the final-layer logits, or a logit-lens readout of an
earlier layer). Because positions are the prompt-independent cloud layout, the origins are
stable and the arrows simply ROTATE as the prompt reshapes the prediction.

Only the grid + arrows + the response trajectory are drawn (not the whole cloud); the cloud
is used purely as the shared coordinate frame. All targets are PRINTABLE tokens.

- TEMPERATURE fan-out: temperature > 0 draws the top-`fanout` predicted arrows per origin.
- RESPONSE trajectory: each response token at its cloud position; effective grid context =
  prefix + response[:response_step].
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


def _predict_topk(lm, base_ids, ref_ids, n_to, temperature, fan, progress_cb, lo, hi):
    """Top-`fan` PRINTABLE next tokens for each reference token, read out at layer ``n_to``
    (the real logits when n_to is the final layer; a logit-lens projection of the layer's
    hidden state otherwise, so the layer slider changes the prediction)."""
    n = len(ref_ids)
    final = n_to >= lm.num_layers
    t = max(float(temperature), 1e-6)
    mask = _printable_mask(lm)
    neg = torch.finfo(torch.float32).min
    w_out = None if final else lm.model.get_output_embeddings().weight.detach().float()
    top_tokens = np.empty((n, fan), dtype=np.int64)
    top_probs = np.empty((n, fan), dtype=np.float32)
    bs = max(1, int(EMBED_BATCH_SIZE))
    for s in range(0, n, bs):
        chunk = [int(x) for x in ref_ids[s : s + bs]]
        ids = torch.tensor([list(base_ids) + [tok] for tok in chunk], dtype=torch.long, device=lm.device)
        with torch.no_grad():
            out = lm.model(input_ids=ids, output_hidden_states=not final)
            if final:
                logits = out.logits[:, -1, :].float()
            else:
                logits = out.hidden_states[n_to][:, -1, :].float() @ w_out.t()  # logit lens
            logits = (logits / t).masked_fill(~mask.unsqueeze(0), neg)
            probs = torch.softmax(logits, dim=-1)
            tp, ti = torch.topk(probs, k=fan, dim=-1)
        top_tokens[s : s + len(chunk)] = ti.cpu().numpy()
        top_probs[s : s + len(chunk)] = tp.cpu().numpy()
        if progress_cb:
            progress_cb(lo + (hi - lo) * min(1.0, (s + len(chunk)) / n), f"predicting at layer {n_to}: {s + len(chunk)}/{n}")
    return top_tokens, top_probs


def _trajectory(lm, base_ids, response_text, row_of, cloud_pos):
    """Each PRINTABLE response token at its cloud position, with its emission probability."""
    resp = list(lm.tokenizer(response_text)["input_ids"])
    if not resp:
        return None
    ids = torch.tensor([list(base_ids) + resp], dtype=torch.long, device=lm.device)
    with torch.no_grad():
        logits = lm.model(input_ids=ids).logits[0]
    base_len = len(base_ids)
    pos, probs, strs = [], [], []
    for j, tok in enumerate(resp):
        row = row_of.get(int(tok))
        if row is None:  # skip unprintable tokens (not in the cloud)
            continue
        p = base_len + j
        pos.append(cloud_pos[row])
        probs.append(float(torch.softmax(logits[p - 1].float(), dim=-1)[int(tok)].item()))
        strs.append(lm.tokenizer.decode([int(tok)]))
    if not pos:
        return None
    return np.asarray(pos, dtype=np.float32), np.asarray(probs, dtype=np.float32), strs


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

    # Fixed coordinate frame: the cached static-embedding cloud (a position for every
    # printable token). Computed once per model; here used only as a position lookup.
    from ..precompute import get_or_compute_sync

    cloud = get_or_compute_sync("token_cloud", model_id, {"seed": seed, "spread_mu": spread_mu})
    cloud_ids = cloud["arrays"]["token_ids"]
    cloud_pos = cloud["arrays"]["warped"].astype(np.float64)  # (Vp, 2) spread positions
    row_of = {int(t): r for r, t in enumerate(cloud_ids.tolist())}

    top_tokens, top_probs = _predict_topk(lm, base, ref_ids, n_to, temperature, fan, progress_cb, 0.1, 0.8)
    ref_pos = cloud_pos[[row_of[int(t)] for t in ref_ids]]  # reference token positions (cloud)

    if progress_cb:
        progress_cb(0.85, "laying out the flow-field grid")
    from scipy.spatial import cKDTree

    # Regular grid of FIXED origins over the (prompt-independent) cloud extent → stable
    # origins, so arrows rotate smoothly between prompts.
    lo = np.percentile(cloud_pos, 1, axis=0)
    hi = np.percentile(cloud_pos, 99, axis=0)
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], grid_n), np.linspace(lo[1], hi[1], grid_n))
    grid_vertices = np.column_stack([gx.ravel(), gy.ravel()])
    _, nn = cKDTree(ref_pos).query(grid_vertices, k=1)  # nearest reference token per origin
    cell = min((hi[0] - lo[0]) / (grid_n - 1), (hi[1] - lo[1]) / (grid_n - 1))
    arrow_len = 0.6 * float(cell)  # uniform length (a little under one grid cell)

    starts, ends, probs, s_tokens, e_tokens = [], [], [], [], []
    for gi in range(grid_vertices.shape[0]):
        r = int(nn[gi])
        v = grid_vertices[gi]
        for f in range(fan):
            p_tok = int(top_tokens[r, f])
            d = cloud_pos[row_of[p_tok]] - ref_pos[r]  # toward the predicted token's position
            norm = float(np.hypot(d[0], d[1]))
            u = d / norm if norm > 1e-9 else np.zeros(2)
            starts.append(v)
            ends.append(v + u * arrow_len)
            probs.append(float(top_probs[r, f]))
            s_tokens.append(int(ref_ids[r]))
            e_tokens.append(p_tok)

    traj = _trajectory(lm, prefix_ids(lm, prefix_text), response_text, row_of, cloud_pos) if response_text else None

    meta = {
        "model_id": lm.model_id, "revision": lm.revision, "grid_n": grid_n,
        "layer_from": n_from, "layer_to": n_to, "num_layers": lm.num_layers,
        "temperature": float(temperature), "fanout": fan, "prefix_text": prefix_text or "",
        "count": len(starts), "reference_points": int(grid_vertices.shape[0]),
        "response_step": int(response_step), "spread_mu": float(spread_mu),
        "seed": int(seed), "vocab_size": int(cloud["meta"]["vocab_size"]),
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
        arrays["trajectory"] = traj[0]
        arrays["trajectory_probs"] = traj[1]
        meta["trajectory_token_strs"] = traj[2]

    if progress_cb:
        progress_cb(1.0, "done")
    return {"meta": meta, "arrays": arrays}
