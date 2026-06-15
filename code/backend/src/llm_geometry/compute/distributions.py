"""Real next-token probability distributions.

Runs a forward pass on the tokenized context, takes the final-position logits,
applies temperature, and softmaxes to an exact distribution over the vocabulary.
``temperature == 0`` is the deterministic argmax. These are genuine model outputs,
never estimates or placeholders (FR-002, Constitution I).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ..errors import InvalidParamError
from ..models.loader import LoadedModel, load_model
from .context import effective_context_ids


def next_token_distribution(
    model_id: str,
    prefix_text: str = "",
    temperature: float = 1.0,
    top_k: int | None = None,
    response_text: str = "",
    response_step: int = 0,
) -> dict[str, Any]:
    if temperature is None or temperature < 0:
        raise InvalidParamError(f"temperature must be >= 0, got {temperature!r}")

    lm = load_model(model_id)
    ids = effective_context_ids(lm, prefix_text, response_text, response_step)
    input_ids = torch.tensor([ids], dtype=torch.long).to(lm.device)

    with torch.no_grad():
        out = lm.model(input_ids=input_ids, output_hidden_states=False)
    logits = out.logits[0, -1, :].float()

    if temperature == 0:
        probs = torch.zeros_like(logits)
        probs[int(torch.argmax(logits))] = 1.0
    else:
        probs = torch.softmax(logits / float(temperature), dim=-1)

    probs_np = probs.detach().cpu().numpy().astype(np.float32)
    top_token = int(np.argmax(probs_np))

    meta: dict[str, Any] = {
        "model_id": lm.model_id,
        "revision": lm.revision,
        "temperature": float(temperature),
        "prefix_text": prefix_text or "",
        "vocab_size": int(probs_np.shape[0]),
        "top_token": top_token,
        "top_token_str": lm.tokenizer.decode([top_token]),
    }
    if top_k:
        k = max(1, int(top_k))
        idx = np.argsort(-probs_np)[:k]
        meta["top"] = [
            {
                "token_id": int(i),
                "token_str": lm.tokenizer.decode([int(i)]),
                "prob": float(probs_np[int(i)]),
            }
            for i in idx
        ]
        meta["tail_mass"] = float(max(0.0, 1.0 - float(np.sum(probs_np[idx]))))

    return {"meta": meta, "arrays": {"probs": probs_np}}
