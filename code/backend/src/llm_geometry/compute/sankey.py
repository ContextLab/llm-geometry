"""Visualization 2 — token sequences as a Sankey diagram via a particle swarm.

Start from the prompt; sample ``n_particles`` next tokens; advance each particle by
conditioning on its own draw; record per-position token occupancy (nodes) and the
position-to-position transitions (links). A particle stops once it emits end-of-stream.
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
from .context import effective_context_ids

ProgressCb = Callable[[float, str], None]


def sankey(
    model_id: str,
    prefix_text: str = "",
    temperature: float = 1.0,
    n_particles: int = 24,
    n_steps: int = 8,
    seed: int = DEFAULT_SEED,
    response_text: str = "",
    response_step: int = 0,
    progress_cb: ProgressCb | None = None,
) -> dict[str, Any]:
    if n_particles < 1 or n_steps < 1:
        raise InvalidParamError("n_particles and n_steps must be >= 1")
    if temperature < 0:
        raise InvalidParamError(f"temperature must be >= 0, got {temperature}")

    lm = load_model(model_id)
    gen = torch.Generator(device=lm.device).manual_seed(int(seed))
    base = effective_context_ids(lm, prefix_text, response_text, response_step)
    eos = lm.tokenizer.eos_token_id
    t = max(float(temperature), 1e-6)

    seqs = [list(base) for _ in range(n_particles)]
    alive = [True] * n_particles
    prev_tok = [None] * n_particles
    node_count: dict[tuple[int, int], int] = defaultdict(int)
    link_count: dict[tuple[int, int, int], int] = defaultdict(int)
    per_position: list[dict] = []  # combined next-token distribution per position (FR §2)
    dist_tokens: set[int] = set()
    distribution_k = 10

    for step in range(n_steps):
        idxs = [i for i in range(n_particles) if alive[i]]
        if not idxs:
            break
        ids = torch.tensor([seqs[i] for i in idxs], dtype=torch.long, device=lm.device)
        with torch.no_grad():
            logits = lm.model(input_ids=ids).logits[:, -1, :].float()
        probs = torch.softmax(logits / t, dim=-1)

        # Combine across particles -> the full distribution displayed at this position.
        combined = probs.mean(dim=0)
        kk = min(distribution_k, int(combined.shape[0]))
        dv, di = torch.topk(combined, k=kk)
        per_position.append({
            "pos": step,
            "top": [{"token": int(di[q]), "prob": round(float(dv[q]), 5)} for q in range(kk)],
        })
        dist_tokens.update(int(di[q]) for q in range(kk))

        draws = torch.multinomial(probs, num_samples=1, generator=gen).squeeze(-1)
        for j, i in enumerate(idxs):
            tok = int(draws[j].item())
            seqs[i].append(tok)
            node_count[(step, tok)] += 1
            if prev_tok[i] is not None:
                link_count[(step - 1, prev_tok[i], tok)] += 1
            prev_tok[i] = tok
            if eos is not None and tok == eos:
                alive[i] = False
        if progress_cb:
            progress_cb((step + 1) / n_steps, f"swarm step {step + 1}/{n_steps}")

    tokens_seen = {tok for (_, tok) in node_count} | dist_tokens
    token_strs = {str(tok): lm.tokenizer.decode([tok]) for tok in tokens_seen}
    nodes = [
        {"pos": p, "token": tok, "count": c}
        for (p, tok), c in sorted(node_count.items())
    ]
    links = [
        {"pos": p, "source_token": a, "target_token": b, "value": c}
        for (p, a, b), c in sorted(link_count.items())
    ]

    meta = {
        "model_id": lm.model_id, "revision": lm.revision, "prefix_text": prefix_text or "",
        "temperature": float(temperature), "n_particles": n_particles, "n_steps": n_steps,
        "token_strs": token_strs, "nodes": nodes, "links": links,
        "per_position": per_position,
    }
    # one real array so the npz payload is non-empty
    arrays = {"node_counts": np.array([n["count"] for n in nodes] or [0], dtype=np.int64)}
    return {"meta": meta, "arrays": arrays}
