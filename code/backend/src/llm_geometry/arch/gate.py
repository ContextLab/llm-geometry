"""Pre-download model-size gate (FR-107).

Estimates a model's total parameter count from HuggingFace **hub metadata only** —
the safetensors header index via ``get_safetensors_metadata`` (a few small JSON/header
fetches, never the weights), falling back to a config-based architectural estimate —
and raises ``ModelTooLargeError`` (HTTP 422) when the count exceeds
``ARCH_MAX_PARAMS``, *before* any weight download begins.
"""

from __future__ import annotations

import logging
from typing import Any

from huggingface_hub import get_safetensors_metadata
from transformers import AutoConfig

from ..config import ARCH_MAX_PARAMS
from ..errors import ModelTooLargeError, UnsupportedModelError
from ..models.hub import hub_call
from ..models.registry import normalize_model_id

logger = logging.getLogger(__name__)


def estimate_params_from_config(config: Any) -> int:
    """Architectural parameter estimate from a model config (no weights needed).

    Standard decoder-only accounting: embeddings (doubled when untied), per-layer
    attention (GQA-aware) + MLP (gated for SiLU/GLU families) + two norms, plus the
    final norm. A coarse but reliable gate signal — within tens of percent for
    Llama/Qwen/GPT-family models.
    """
    cfg = getattr(config, "text_config", None) or config
    hidden = int(getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", 0) or 0)
    layers = int(getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer", 0) or 0)
    vocab = int(getattr(cfg, "vocab_size", 0) or 0)
    heads = int(getattr(cfg, "num_attention_heads", None) or getattr(cfg, "n_head", 1) or 1)
    kv_heads = int(getattr(cfg, "num_key_value_heads", None) or heads)
    head_dim = int(getattr(cfg, "head_dim", None) or (hidden // heads if heads else 0))
    intermediate = int(
        getattr(cfg, "intermediate_size", None) or getattr(cfg, "n_inner", None) or 4 * hidden
    )
    act = str(getattr(cfg, "hidden_act", "gelu")).lower()
    mlp_mats = 3 if ("silu" in act or "glu" in act or "swish" in act) else 2

    attn = hidden * heads * head_dim + 2 * hidden * kv_heads * head_dim + heads * head_dim * hidden
    per_layer = attn + mlp_mats * hidden * intermediate + 2 * hidden
    embed = vocab * hidden * (1 if bool(getattr(cfg, "tie_word_embeddings", False)) else 2)
    return int(embed + layers * per_layer + hidden)


def check_model_size(model_id: str) -> dict[str, Any]:
    """Gate a model on total parameters BEFORE download; raises when over the ceiling.

    Returns ``{"model_id", "total_params", "max_params", "source"}`` when the model
    passes. ``source`` records which metadata path produced the count.
    """
    mid = normalize_model_id(model_id)

    total: int | None = None
    source = ""
    try:
        meta = hub_call(get_safetensors_metadata, mid)
        counts = getattr(meta, "parameter_count", None) or {}
        if counts:
            total = int(sum(counts.values()))
            source = "safetensors_metadata"
    except Exception as exc:
        # No safetensors metadata (or hub hiccup) -> config estimate. Log rather than
        # swallow: FR-107 depends on this gate, so the degraded path must be visible.
        logger.warning("size gate: safetensors metadata unavailable for %s: %s", mid, exc)
        total = None

    if total is None:
        try:
            config = AutoConfig.from_pretrained(mid)
        except Exception as exc:  # missing, gated, offline, malformed
            raise UnsupportedModelError(
                f"Could not load configuration for model '{mid}': {exc}. It may not "
                "exist, may be gated/private, or may require authentication.",
                detail={"model_id": mid},
            ) from exc
        total = estimate_params_from_config(config)
        source = "config_estimate"

    # The config estimate can undercount (multimodal towers, exotic blocks); give it a
    # 20% safety margin so an undercounted giant can't slip under the ceiling (FR-107).
    effective_ceiling = (
        ARCH_MAX_PARAMS if source == "safetensors_metadata" else ARCH_MAX_PARAMS * 0.8
    )

    if total > effective_ceiling:
        raise ModelTooLargeError(
            f"Model '{mid}' has ~{total:,} parameters, over the Architecture Explorer "
            f"ceiling of {ARCH_MAX_PARAMS:,}. Choose a smaller open-weights model.",
            detail={
                "model_id": mid,
                "total_params": int(total),
                "max_params": int(ARCH_MAX_PARAMS),
                "source": source,
            },
        )
    return {
        "model_id": mid,
        "total_params": int(total),
        "max_params": int(ARCH_MAX_PARAMS),
        "source": source,
    }
