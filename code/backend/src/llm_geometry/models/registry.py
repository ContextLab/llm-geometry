"""Curated model menu + id normalization.

The curated list is a convenience starting point; arbitrary open-weights HuggingFace
ids are also accepted and validated by the loader (FR-001).
"""

from __future__ import annotations

from ..config import CURATED_MODELS
from ..errors import InvalidParamError


def curated_models() -> list[dict[str, str]]:
    return [
        {"model_id": mid, "display_name": name, "source": "curated"}
        for mid, name in CURATED_MODELS.items()
    ]


def curated_ids() -> set[str]:
    return set(CURATED_MODELS.keys())


def display_name(model_id: str) -> str:
    return CURATED_MODELS.get(model_id, model_id)


def normalize_model_id(model_id: str | None) -> str:
    mid = (model_id or "").strip()
    if not mid:
        raise InvalidParamError("model_id must be a non-empty string")
    return mid
