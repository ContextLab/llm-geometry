"""Open-weights model loading + capability detection.

``resolve_model`` is cheap (config only) and answers "can we use this model?" by
checking it is a causal LM and pinning its revision. ``load_model`` does the full
load and confirms with a real forward pass that the model exposes both token-level
logits and per-layer hidden states. Every load/download/OOM failure is wrapped in a
typed ``UnsupportedModelError`` — never a fallback model and never a fabricated
result (FR-002, FR-003, FR-021; spec edge cases: offline/partial download, OOM).
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import torch

# Import the transformers/hub symbols eagerly at module load (single-threaded at app
# startup). transformers uses a lazy importer that is NOT safe under concurrent
# first-import from multiple worker threads, which otherwise causes intermittent
# "cannot import name AutoConfig" errors when several requests arrive at once.
from huggingface_hub import model_info
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from ..errors import UnsupportedModelError
from .registry import curated_ids, display_name, normalize_model_id

_CAUSAL_SUFFIXES = ("LMHeadModel", "ForCausalLM")
_MAX_LOADED = 2


@dataclass
class LoadedModel:
    model_id: str
    revision: str
    model: Any
    tokenizer: Any
    num_layers: int
    hidden_size: int
    vocab_size: int
    device: str = "cpu"


def _capabilities_from_config(config: Any) -> dict[str, Any]:
    architectures = list(getattr(config, "architectures", None) or [])
    is_causal = any(a.endswith(_CAUSAL_SUFFIXES) for a in architectures)
    num_layers = getattr(config, "num_hidden_layers", None) or getattr(config, "n_layer", None)
    hidden_size = getattr(config, "hidden_size", None) or getattr(config, "n_embd", None)
    vocab_size = getattr(config, "vocab_size", None)
    return {
        "is_causal_lm": bool(is_causal),
        "num_layers": int(num_layers) if num_layers else None,
        "hidden_size": int(hidden_size) if hidden_size else None,
        "vocab_size": int(vocab_size) if vocab_size else None,
        "architectures": architectures,
    }


def _resolve_revision(model_id: str) -> str:
    """Best-effort pin of the model's commit sha for reproducibility (FR-013)."""
    try:
        return model_info(model_id).sha or "main"
    except Exception:
        return "main"


@lru_cache(maxsize=64)
def resolve_model(model_id: str) -> dict[str, Any]:
    """Validate a model (config only) and return a ModelReference dict.

    Memoized per process (revision/capabilities are stable for a session); raised
    UnsupportedModelErrors are not cached, so a transient failure can be retried.
    Raises UnsupportedModelError if the model is missing/gated or not a causal LM.
    """
    mid = normalize_model_id(model_id)

    try:
        config = AutoConfig.from_pretrained(mid)
    except Exception as exc:  # missing, gated, offline, malformed
        raise UnsupportedModelError(
            f"Could not load configuration for model '{mid}': {exc}. It may not "
            "exist, may be gated/private, or may require authentication.",
            detail={"model_id": mid},
        ) from exc

    caps = _capabilities_from_config(config)
    if not caps["is_causal_lm"]:
        raise UnsupportedModelError(
            f"Model '{mid}' is not a causal (decoder) language model exposing "
            f"token-level next-token probabilities (architectures={caps['architectures']}).",
            detail={"model_id": mid, "architectures": caps["architectures"]},
        )

    return {
        "model_id": mid,
        "revision": _resolve_revision(mid),
        "source": "curated" if mid in curated_ids() else "user",
        "display_name": display_name(mid),
        "status": "supported",
        "capabilities": {
            "num_layers": caps["num_layers"],
            "hidden_size": caps["hidden_size"],
            "vocab_size": caps["vocab_size"],
            "exposes_token_probs": True,
            "exposes_hidden_states": True,
        },
    }


_loaded: "OrderedDict[str, LoadedModel]" = OrderedDict()
_loaded_lock = threading.Lock()     # guards the _loaded dict (fast path)
_load_serialize = threading.Lock()  # serializes the heavy, non-thread-safe from_pretrained


def load_model(model_id: str) -> LoadedModel:
    """Load tokenizer + causal LM (CPU, eval, no-grad) and verify it really exposes
    logits + hidden states. Cached in-process (small LRU)."""
    mid = normalize_model_id(model_id)
    with _loaded_lock:
        if mid in _loaded:
            _loaded.move_to_end(mid)
            return _loaded[mid]

    # transformers' from_pretrained is NOT safe to call concurrently for the same
    # model — racing loads land in a half-initialized "meta tensor" state (RuntimeError
    # "Cannot copy out of meta tensor"). Serialize the heavy load; the cache check above
    # stays lock-free so a warmed model is still instant.
    with _load_serialize:
        with _loaded_lock:
            if mid in _loaded:  # another thread finished the load while we waited
                _loaded.move_to_end(mid)
                return _loaded[mid]

        ref = resolve_model(mid)  # validates + pins revision (raises if unsupported)
        revision = ref["revision"]

        try:
            tokenizer = AutoTokenizer.from_pretrained(mid)
            model = AutoModelForCausalLM.from_pretrained(mid, output_hidden_states=True)
        except Exception as exc:  # download failure/partial, gated, OOM, etc.
            raise UnsupportedModelError(
                f"Failed to load model '{mid}': {exc}. The download may have failed or "
                "been interrupted, the model may be gated, or there may be insufficient "
                "memory for it on this machine.",
                detail={"model_id": mid},
            ) from exc

        torch.set_grad_enabled(False)
        model.eval()
        device = "cpu"
        model.to(device)

        # Real forward-pass verification (Constitution I): must yield logits over the
        # vocabulary AND per-layer hidden states.
        try:
            enc = tokenizer("hello world", return_tensors="pt")
            out = model(**{k: v.to(device) for k, v in enc.items()}, output_hidden_states=True)
            logits = out.logits
            hidden = out.hidden_states
            if logits is None or hidden is None or len(hidden) < 2:
                raise ValueError("model did not return logits and per-layer hidden states")
        except Exception as exc:
            raise UnsupportedModelError(
                f"Model '{mid}' loaded but does not expose the required token-level "
                f"probabilities and per-layer hidden states: {exc}.",
                detail={"model_id": mid},
            ) from exc

        vocab_size = int(getattr(model.config, "vocab_size", logits.shape[-1]))
        hidden_size = int(hidden[0].shape[-1])
        num_layers = ref["capabilities"]["num_layers"] or (len(hidden) - 1)

        lm = LoadedModel(
            model_id=mid,
            revision=revision,
            model=model,
            tokenizer=tokenizer,
            num_layers=int(num_layers),
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            device=device,
        )
        with _loaded_lock:
            _loaded[mid] = lm
            _loaded.move_to_end(mid)
            while len(_loaded) > _MAX_LOADED:
                _loaded.popitem(last=False)
        return lm
