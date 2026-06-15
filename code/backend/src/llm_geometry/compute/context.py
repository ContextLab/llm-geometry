"""Shared context construction.

The "effective context" is the prompt/context prefix plus the first ``response_step``
tokens of an optional response string. Stepping ``response_step`` from 0..len(response)
and re-rendering a visualization animates "how it changes with each subsequent token"
(project_description.md §1/§3).
"""

from __future__ import annotations

from typing import Any


def _start_token(lm) -> int:
    start = lm.tokenizer.bos_token_id
    if start is None:
        start = lm.tokenizer.eos_token_id
    if start is None:
        start = 0
    return int(start)


def prefix_ids(lm, prefix_text: str) -> list[int]:
    if prefix_text:
        return list(lm.tokenizer(prefix_text)["input_ids"])
    return [_start_token(lm)]


def response_ids(lm, response_text: str) -> list[int]:
    if not response_text:
        return []
    return list(lm.tokenizer(response_text)["input_ids"])


def effective_context_ids(
    lm, prefix_text: str, response_text: str = "", response_step: int = 0
) -> list[int]:
    """prefix + the first ``response_step`` response tokens (clamped)."""
    ids = prefix_ids(lm, prefix_text)
    if response_text and response_step > 0:
        resp = response_ids(lm, response_text)
        ids = ids + resp[: max(0, int(response_step))]
    return ids


def tokenize_strings(lm, text: str) -> list[dict[str, Any]]:
    """Token ids + decoded strings for a piece of text (used by /api/tokenize)."""
    ids = list(lm.tokenizer(text)["input_ids"]) if text else []
    return [{"token": int(t), "token_str": lm.tokenizer.decode([int(t)])} for t in ids]
