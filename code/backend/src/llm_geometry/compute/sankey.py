"""Visualization 2 — token sequences as a Sankey diagram via a particle swarm.

A swarm of up to a few thousand particles samples *actual responses to a prompt*: each
particle starts from the context (prompt) only and samples its own continuation, one token
at a time, conditioning on its own draws and stopping when it emits end-of-stream. Each step
does sub-batched forward passes (memory-safe at scale), records the combined next-token
distribution (mean over live particles), and the flows become a density of where the model's
responses actually go. If the user supplies a specific response, that exact path is computed
teacher-forced (P(tokenₖ | prompt + response[:k])) and HIGHLIGHTED over the swarm.
(project_description.md §2)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

import numpy as np
import torch

from ..config import DEFAULT_SEED
from ..errors import InvalidParamError
from ..models.loader import load_model
from .context import prefix_ids
from .printable import printable_tokens

ProgressCb = Callable[[float, str], None]

MAX_PARTICLES = 2000
FORWARD_BATCH = 128  # sub-batch size for the per-step forward pass


def _highlight_path(lm, base, response_text, t, n_steps):
    """Teacher-forced path for the user's response: the token ids and the model probability of
    each, P(tokenₖ | prompt + response[:k]), in one forward pass over prompt+response."""
    resp = [int(x) for x in lm.tokenizer(response_text)["input_ids"]][:n_steps]
    if not resp:
        return []
    ids = torch.tensor([list(base) + resp], dtype=torch.long, device=lm.device)
    with torch.no_grad():
        logits = lm.model(input_ids=ids).logits[0].float()
    base_len = len(base)
    out = []
    for k, tok in enumerate(resp):
        probs = torch.softmax(logits[base_len - 1 + k] / t, dim=-1)
        out.append({"pos": k, "token": int(tok), "prob": round(float(probs[tok]), 6)})
    return out


def sankey(
    model_id: str,
    prefix_text: str = "",
    temperature: float = 1.0,
    n_particles: int = 800,
    n_steps: int = 8,
    seed: int = DEFAULT_SEED,
    response_text: str = "",
    top_nodes: int = 18,
    progress_cb: ProgressCb | None = None,
) -> dict[str, Any]:
    if n_particles < 1 or n_steps < 1:
        raise InvalidParamError("n_particles and n_steps must be >= 1")
    if temperature < 0:
        raise InvalidParamError(f"temperature must be >= 0, got {temperature}")

    n_particles = min(int(n_particles), MAX_PARTICLES)
    lm = load_model(model_id)
    gen = torch.Generator(device=lm.device).manual_seed(int(seed))
    base = prefix_ids(lm, prefix_text)  # the swarm conditions on the PROMPT only
    eos = lm.tokenizer.eos_token_id
    t = max(float(temperature), 1e-6)
    printable_ids = set(int(i) for i in printable_tokens(lm)[0].tolist())  # show only named tokens

    seqs = [list(base) for _ in range(n_particles)]
    alive = [True] * n_particles
    prev_tok: list[int | None] = [None] * n_particles
    node_count: dict[tuple[int, int], int] = defaultdict(int)
    link_count: dict[tuple[int, int, int], int] = defaultdict(int)
    alive_at: dict[int, int] = {}  # live particles entering each step (denominator for node prob)
    per_position: list[dict] = []
    dist_tokens: set[int] = set()
    distribution_k = 12

    for step in range(n_steps):
        idxs = [i for i in range(n_particles) if alive[i]]
        if not idxs:
            break
        alive_at[step] = len(idxs)
        combined_sum: torch.Tensor | None = None
        draws: dict[int, int] = {}
        for cs in range(0, len(idxs), FORWARD_BATCH):
            chunk = idxs[cs : cs + FORWARD_BATCH]
            ids = torch.tensor([seqs[i] for i in chunk], dtype=torch.long, device=lm.device)
            with torch.no_grad():
                logits = lm.model(input_ids=ids).logits[:, -1, :].float()
            probs = torch.softmax(logits / t, dim=-1)
            chunk_sum = probs.sum(dim=0)
            combined_sum = chunk_sum if combined_sum is None else combined_sum + chunk_sum
            d = torch.multinomial(probs, num_samples=1, generator=gen).squeeze(-1)
            for j, i in enumerate(chunk):
                draws[i] = int(d[j].item())

        combined = combined_sum / len(idxs)  # type: ignore[operator]
        cand = min(distribution_k * 4, int(combined.shape[0]))
        dv, di = torch.topk(combined, k=cand)
        top: list[dict] = []
        for q in range(cand):
            tok = int(di[q])
            if tok not in printable_ids:
                continue
            top.append({"token": tok, "prob": round(float(dv[q]), 5)})
            if len(top) >= distribution_k:
                break
        per_position.append({"pos": step, "top": top})
        dist_tokens.update(e["token"] for e in top)

        for i in idxs:
            tok = draws[i]
            seqs[i].append(tok)
            node_count[(step, tok)] += 1
            if prev_tok[i] is not None:
                link_count[(step - 1, prev_tok[i], tok)] += 1
            prev_tok[i] = tok
            if eos is not None and tok == eos:
                alive[i] = False
        if progress_cb:
            progress_cb((step + 1) / n_steps, f"swarm step {step + 1}/{n_steps} · {len(idxs)} particles")

    # The user's specific response, teacher-forced, to overlay as a highlighted path.
    highlight = _highlight_path(lm, base, response_text, t, n_steps)
    hl_nodes = {(h["pos"], h["token"]) for h in highlight}
    hl_links = {(highlight[k - 1]["pos"], highlight[k - 1]["token"], highlight[k]["token"]) for k in range(1, len(highlight))}
    hl_prob = {(h["pos"], h["token"]): h["prob"] for h in highlight}

    # Cap each position to its top-K most-populated tokens (density at scale).
    by_pos: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for (pos, tok), c in node_count.items():
        by_pos[pos].append((tok, c))
    kept: set[tuple[int, int]] = set()
    for pos, lst in by_pos.items():
        lst.sort(key=lambda x: -x[1])
        cnt = 0
        for tok, _c in lst:  # keep the top-K PRINTABLE tokens per position
            if tok not in printable_ids:
                continue
            kept.add((pos, tok))
            cnt += 1
            if cnt >= max(1, int(top_nodes)):
                break

    # Force-keep the highlighted response path so the user's exact response always shows, even
    # where it's an unlikely continuation the swarm never drew. Use a count of just 1 so the node
    # gets a valid layout slot WITHOUT inflating its apparent probability — it's marked instead by
    # a gold outline + overlay dot on the frontend, and the tooltip carries the true (often tiny)
    # teacher-forced probability.
    for (pos, tok) in hl_nodes:
        kept.add((pos, tok))
        node_count[(pos, tok)] = max(node_count.get((pos, tok), 0), 1)
    for (pos, a, b) in hl_links:
        link_count[(pos, a, b)] = max(link_count.get((pos, a, b), 0), 1)

    # Keep only nodes REACHABLE from position 0 via kept links, so every flow starts at the
    # prompt and ends where its particles stop — none appears to begin mid-stream (the top-K
    # capping can otherwise orphan a later token whose predecessor was dropped). Highlight nodes
    # are roots too, so the response path is never severed.
    adj: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for (pos, a, b), _c in link_count.items():
        if (pos, a) in kept and (pos + 1, b) in kept:
            adj[(pos, a)].append((pos + 1, b))
    reachable = {(p, t) for (p, t) in kept if p == 0} | hl_nodes
    stack = list(reachable)
    while stack:
        node = stack.pop()
        for nxt in adj.get(node, []):
            if nxt not in reachable:
                reachable.add(nxt)
                stack.append(nxt)
    kept = reachable

    nc = dict(node_count)
    nodes = [
        {
            "pos": p, "token": tok, "count": c,
            "prob": round(hl_prob[(p, tok)] if (p, tok) in hl_prob else c / max(1, alive_at.get(p, n_particles)), 6),
            "highlight": (p, tok) in hl_nodes,
        }
        for (p, tok), c in sorted(nc.items())
        if (p, tok) in kept
    ]
    links = [
        {
            "pos": p, "source_token": a, "target_token": b, "value": c,
            "cond": round(c / max(1, nc.get((p, a), 1)), 6),  # empirical P(target | source)
            "highlight": (p, a, b) in hl_links,
        }
        for (p, a, b), c in sorted(link_count.items())
        if (p, a) in kept and (p + 1, b) in kept
    ]

    tokens_seen = {tok for (_, tok) in nc if (_, tok) in kept} | dist_tokens | {h["token"] for h in highlight}
    token_strs = {str(tok): lm.tokenizer.decode([tok]) for tok in tokens_seen}
    for h in highlight:
        h["token_str"] = token_strs[str(h["token"])]

    meta = {
        "model_id": lm.model_id, "revision": lm.revision, "prefix_text": prefix_text or "",
        "temperature": float(temperature), "n_particles": n_particles, "n_steps": n_steps,
        "token_strs": token_strs, "nodes": nodes, "links": links, "per_position": per_position,
        "highlight": highlight,  # the user's response path (teacher-forced model probabilities)
    }
    arrays = {"node_counts": np.array([n["count"] for n in nodes] or [0], dtype=np.int64)}
    return {"meta": meta, "arrays": arrays}
