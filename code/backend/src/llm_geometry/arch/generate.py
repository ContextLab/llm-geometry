"""Real autoregressive generation, per the `/api/arch/generate` contract.

A manual decode loop over genuine model logits: ``temperature == 0`` is greedy
argmax; otherwise the post-temperature softmax is sampled (optionally with a seeded
generator for reproducibility). Each generated token reports its probability under
the pre-sample distribution plus the top-5 alternatives. Single-turn; the model's
chat template is used when available.
"""

from __future__ import annotations

from typing import Any

import torch

from ..config import (
    ARCH_MAX_NEW_TOKENS,
    ARCH_REPETITION_PENALTY,
    ARCH_TOP_K,
    ARCH_TOP_P,
)
from ..errors import InvalidParamError
from ..models.loader import LoadedModel, load_model
from .tracing import _TRACE_LOCK, encode_prompt


def _sample_filtered(
    logits: torch.Tensor,
    temperature: float,
    seen_ids: list[int],
    generator: torch.Generator | None,
) -> int:
    """Draw a token from the temperature softmax restricted to top-k ∩ top-p, with a
    repetition penalty applied first.

    Sampling the FULL vocabulary (the previous behavior) draws from the long tail on
    every single step, which small models cannot survive — it was the main reason
    replies read as word salad. These are the standard decoding constraints; the
    transformers.js runtime mirrors them so both stacks decode the same way.

    Only the DRAW is filtered. Every probability this module reports still comes from
    the unfiltered distribution, so no number shown to the user changes meaning.
    """
    work = logits.clone()
    if ARCH_REPETITION_PENALTY != 1.0 and seen_ids:
        idx = torch.tensor(sorted(set(seen_ids)), dtype=torch.long, device=work.device)
        vals = work[idx]
        # HF's convention: divide positive logits, multiply negative ones — both push
        # an already-used token down regardless of its sign.
        work[idx] = torch.where(
            vals > 0, vals / ARCH_REPETITION_PENALTY, vals * ARCH_REPETITION_PENALTY
        )

    scaled = work / float(temperature)
    k = min(ARCH_TOP_K, int(scaled.shape[0]))
    top_vals, top_idx = torch.topk(scaled, k=k)
    probs = torch.softmax(top_vals, dim=-1)

    # Nucleus: smallest prefix whose cumulative mass reaches top_p (never empty).
    csum = torch.cumsum(probs, dim=-1)
    keep = int(torch.searchsorted(csum, torch.tensor(ARCH_TOP_P, device=csum.device)).item()) + 1
    keep = max(1, min(keep, k))
    probs = probs[:keep]
    probs = probs / probs.sum()

    choice = int(torch.multinomial(probs, 1, generator=generator).item())
    return int(top_idx[choice])


def _eos_ids(lm: LoadedModel) -> set[int]:
    ids: set[int] = set()
    if lm.tokenizer.eos_token_id is not None:
        ids.add(int(lm.tokenizer.eos_token_id))
    configured = getattr(getattr(lm.model, "generation_config", None), "eos_token_id", None)
    if isinstance(configured, int):
        ids.add(configured)
    elif isinstance(configured, (list, tuple)):
        ids.update(int(i) for i in configured)
    return ids


def generate(
    model_id: str,
    prompt: str,
    system_prompt: str | None = None,
    temperature: float = 0.8,
    max_new_tokens: int = 64,
    seed: int | None = None,
) -> dict[str, Any]:
    if not (prompt or "").strip():
        raise InvalidParamError("prompt must be a non-empty string")
    if temperature is None or temperature < 0:
        raise InvalidParamError(f"temperature must be >= 0, got {temperature!r}")
    max_new_tokens = int(max_new_tokens)
    if not 1 <= max_new_tokens <= ARCH_MAX_NEW_TOKENS:
        raise InvalidParamError(
            f"max_new_tokens must be in 1..{ARCH_MAX_NEW_TOKENS}, got {max_new_tokens}"
        )

    lm = load_model(model_id)
    ids, _ = encode_prompt(lm, prompt, system_prompt)
    eos = _eos_ids(lm)

    generator: torch.Generator | None = None
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))

    tokens: list[dict[str, Any]] = []
    finish_reason = "length"
    current = torch.tensor([ids], dtype=torch.long, device=lm.device)
    past = None
    # Serialize with tracing: a trace installs global forward hooks on this SAME cached
    # model instance, so an unlocked concurrent generate would fire the tracer's hooks
    # from another thread and corrupt its event stream.
    with _TRACE_LOCK, torch.no_grad():
        for _ in range(max_new_tokens):
            out = lm.model(input_ids=current, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[0, -1, :].float()

            if temperature == 0:
                next_id = int(torch.argmax(logits))
                probs = torch.zeros_like(logits)
                probs[next_id] = 1.0
            else:
                # Reported probability: the UNFILTERED temperature softmax, so `prob`
                # keeps the meaning it has always had even though the draw below is
                # restricted to the top-k ∩ top-p nucleus.
                probs = torch.softmax(logits / float(temperature), dim=-1)
                next_id = _sample_filtered(
                    logits, temperature, ids + [t["id"] for t in tokens], generator
                )

            k = min(5, int(logits.shape[0]))
            top_ids = [int(i) for i in torch.topk(logits, k=k).indices]
            # Alternatives always report the MODEL's distribution (plain softmax):
            # at temperature 0 the sampling distribution is one-hot, and showing
            # "Paris 0.0%" for a genuinely ~2% alternative misleads (red-team F4).
            # `prob` of the chosen token stays the sampling probability (1.0 greedy).
            model_probs = torch.softmax(logits, dim=-1)
            tokens.append(
                {
                    "id": next_id,
                    "text": lm.tokenizer.decode([next_id]),
                    "prob": float(probs[next_id]),
                    "topk": {
                        "ids": top_ids,
                        "texts": [lm.tokenizer.decode([i]) for i in top_ids],
                        "probs": [float(model_probs[i]) for i in top_ids],
                    },
                }
            )
            if next_id in eos:
                finish_reason = "eos"
                break
            current = torch.tensor([[next_id]], dtype=torch.long, device=lm.device)

    text = lm.tokenizer.decode([t["id"] for t in tokens], skip_special_tokens=True)
    return {"text": text, "tokens": tokens, "finish_reason": finish_reason}
