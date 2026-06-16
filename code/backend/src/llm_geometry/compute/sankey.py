"""Visualization 2 — token sequences as a Sankey diagram via a particle swarm.

A swarm of up to a few thousand particles samples next tokens across positions. Each
step does sub-batched forward passes (memory-safe at scale), accumulates the combined
next-token distribution (mean over particles), samples each particle's next token, and
stops particles that emit end-of-stream. For display the per-position nodes are capped
to the top-K most-populated tokens (the flows become a density). (project_description.md §2)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

import numpy as np
import torch

from ..config import DEFAULT_SEED
from ..errors import InvalidParamError
from ..models.loader import load_model
from .context import effective_context_ids
from .printable import printable_tokens

ProgressCb = Callable[[float, str], None]

MAX_PARTICLES = 2000
FORWARD_BATCH = 128  # sub-batch size for the per-step forward pass


def sankey(
    model_id: str,
    prefix_text: str = "",
    temperature: float = 1.0,
    n_particles: int = 500,
    n_steps: int = 8,
    seed: int = DEFAULT_SEED,
    response_text: str = "",
    response_step: int = 0,
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
    base = effective_context_ids(lm, prefix_text, response_text, response_step)
    eos = lm.tokenizer.eos_token_id
    t = max(float(temperature), 1e-6)
    printable_ids = set(int(i) for i in printable_tokens(lm)[0].tolist())  # show only named tokens

    seqs = [list(base) for _ in range(n_particles)]
    alive = [True] * n_particles
    prev_tok: list[int | None] = [None] * n_particles
    node_count: dict[tuple[int, int], int] = defaultdict(int)
    link_count: dict[tuple[int, int, int], int] = defaultdict(int)
    per_position: list[dict] = []
    dist_tokens: set[int] = set()
    distribution_k = 12

    for step in range(n_steps):
        idxs = [i for i in range(n_particles) if alive[i]]
        if not idxs:
            break
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

    # Keep only nodes REACHABLE from position 0 via kept links, so every flow starts at the
    # prompt and ends where its particles stop — none appears to begin mid-stream (the top-K
    # capping can otherwise orphan a later token whose predecessor was dropped).
    adj: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for (pos, a, b), _c in link_count.items():
        if (pos, a) in kept and (pos + 1, b) in kept:
            adj[(pos, a)].append((pos + 1, b))
    reachable = {(p, t) for (p, t) in kept if p == 0}
    stack = list(reachable)
    while stack:
        node = stack.pop()
        for nxt in adj.get(node, []):
            if nxt not in reachable:
                reachable.add(nxt)
                stack.append(nxt)
    kept = reachable

    nodes = [
        {"pos": p, "token": tok, "count": c}
        for (p, tok), c in sorted(node_count.items())
        if (p, tok) in kept
    ]
    links = [
        {"pos": p, "source_token": a, "target_token": b, "value": c}
        for (p, a, b), c in sorted(link_count.items())
        if (p, a) in kept and (p + 1, b) in kept
    ]

    tokens_seen = {tok for (_, tok) in node_count if (_, tok) in kept} | dist_tokens
    token_strs = {str(tok): lm.tokenizer.decode([tok]) for tok in tokens_seen}

    meta = {
        "model_id": lm.model_id, "revision": lm.revision, "prefix_text": prefix_text or "",
        "temperature": float(temperature), "n_particles": n_particles, "n_steps": n_steps,
        "token_strs": token_strs, "nodes": nodes, "links": links, "per_position": per_position,
    }
    arrays = {"node_counts": np.array([n["count"] for n in nodes] or [0], dtype=np.int64)}
    return {"meta": meta, "arrays": arrays}
