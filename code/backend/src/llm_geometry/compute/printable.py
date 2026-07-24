"""Printable-token filtering.

Open-weights vocabularies contain many tokens with no meaningful glyph: special tokens,
and (for byte-level BPE) fragments that decode to the Unicode replacement char ``�``.
These render as noise — or as a useless ``token #id`` hover — so every visualization shows
only PRINTABLE tokens and ships their real decoded strings. Decoding the whole vocab is
cheap (~0.5s) and memoized per (model, revision).
"""

from __future__ import annotations

import numpy as np

from ..models.loader import LoadedModel

_cache: dict[tuple[str, str], tuple[np.ndarray, list[str]]] = {}


def _is_printable(s: str) -> bool:
    if not s or not s.strip():
        return False
    if "�" in s:  # the replacement char from incomplete byte-BPE fragments
        return False
    return s.isprintable()


def printable_tokens(lm: LoadedModel) -> tuple[np.ndarray, list[str]]:
    """All vocabulary token ids whose decoded string is printable, plus those strings
    (aligned). Memoized per model."""
    key = (lm.model_id, lm.revision)
    hit = _cache.get(key)
    if hit is not None:
        return hit
    strs = lm.tokenizer.batch_decode([[i] for i in range(lm.vocab_size)])
    keep = [i for i, s in enumerate(strs) if _is_printable(s)]
    result = (np.asarray(keep, dtype=np.int64), [strs[i] for i in keep])
    _cache[key] = result
    return result


def printable_reference_ids(lm: LoadedModel, n: int | None) -> np.ndarray:
    """``n`` printable token ids evenly spaced across the (printable) vocab, or all of
    them when ``n`` is None. Deterministic → reproducible cache."""
    ids, _ = printable_tokens(lm)
    if n is None or int(n) >= ids.shape[0]:
        return ids
    idx = np.unique(np.linspace(0, ids.shape[0] - 1, int(n)).astype(np.int64))
    return ids[idx]
