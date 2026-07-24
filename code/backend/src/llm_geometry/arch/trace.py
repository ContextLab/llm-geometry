"""Per-layer, per-node activation traces from one real forward pass.

Runs the shared tracer (`tracing.py`) with ``output_attentions`` +
``output_hidden_states`` on a real prompt (rendered through the model's chat template
when it has one) and reports, per layer: the attention pattern per head (downsampled
to at most 64x64 when the sequence is long), the residual-stream L2 norm per
position, and a 3-D PCA of that layer's hidden states (a visualization aid, fit per
response). ``node_activations`` covers every graph node — the same traced-event
stream `graph.py` builds nodes from (completeness invariant, SC-102).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ..config import ARCH_ATTENTION_MAX_SIDE, ARCH_DEFAULT_MAX_CONTEXT
from ..errors import InvalidParamError
from ..models.loader import load_model
from .tracing import encode_prompt, encode_shape, run_traced_forward
from .weights import strided_mean_2d


def _pca3(x: np.ndarray) -> np.ndarray:
    """Deterministic 3-component PCA of a (T, H) matrix; zero-padded when T < 3."""
    xc = x.astype(np.float64) - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    k = min(3, vt.shape[0])
    comps = vt[:k]
    signs = np.sign(comps[np.arange(k), np.abs(comps).argmax(axis=1)])
    signs[signs == 0] = 1.0
    proj = xc @ (comps * signs[:, None]).T
    if proj.shape[1] < 3:
        proj = np.hstack([proj, np.zeros((proj.shape[0], 3 - proj.shape[1]))])
    return proj.astype(np.float32)


def trace_forward(
    model_id: str,
    prompt: str,
    system_prompt: str | None = None,
    max_context: int = ARCH_DEFAULT_MAX_CONTEXT,
) -> dict[str, Any]:
    """Trace one real forward pass, per the `/api/arch/trace` contract."""
    if not (prompt or "").strip():
        raise InvalidParamError("prompt must be a non-empty string")
    max_context = int(max_context)
    if max_context < 1:
        raise InvalidParamError(f"max_context must be >= 1, got {max_context}")

    lm = load_model(model_id)
    ids, chat_template_used = encode_prompt(lm, prompt, system_prompt)
    ids = ids[-max_context:]  # truncate LEFT: keep the most recent context
    seq_len = len(ids)
    input_ids = torch.tensor([ids], dtype=torch.long, device=lm.device)

    out, events = run_traced_forward(
        lm.model, input_ids, output_attentions=True, output_hidden_states=True
    )

    tokens = [{"id": int(i), "text": lm.tokenizer.decode([int(i)])} for i in ids]

    layers: list[dict[str, Any]] = []
    for k, attn in enumerate(out.attentions):
        heads = attn[0].detach().float().cpu().numpy()  # [heads, T, T]
        downsampled = seq_len > ARCH_ATTENTION_MAX_SIDE
        if downsampled:
            heads = np.stack(
                [
                    strided_mean_2d(h, ARCH_ATTENTION_MAX_SIDE, ARCH_ATTENTION_MAX_SIDE)
                    for h in heads
                ]
            )
        hidden = out.hidden_states[k + 1][0].detach().float().cpu().numpy()  # [T, H]
        layers.append(
            {
                "layer": k,
                "attention": heads.astype(float).tolist(),
                "attention_downsampled": downsampled,
                "hidden_norm": np.linalg.norm(hidden, axis=1).astype(float).tolist(),
                "hidden_pca3": _pca3(hidden).astype(float).tolist(),
            }
        )

    logits = out.logits[0, -1, :].detach().float()
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    top = np.argsort(-probs)[:10]
    logits_topk = {
        "ids": [int(i) for i in top],
        "texts": [lm.tokenizer.decode([int(i)]) for i in top],
        "probs": [float(probs[int(i)]) for i in top],
    }

    node_activations: list[dict[str, Any]] = []
    seen: set[str] = set()
    for ev in events:
        if ev.node_id in seen:
            continue
        seen.add(ev.node_id)
        node_activations.append(
            {
                "node_id": ev.node_id,
                "out_norm": float(ev.out_norm),
                "out_shape": encode_shape(ev.out_shapes[0], seq_len) if ev.out_shapes else [],
            }
        )

    return {
        "tokens": tokens,
        "chat_template_used": chat_template_used,
        "layers": layers,
        "logits_topk": logits_topk,
        "node_activations": node_activations,
    }
