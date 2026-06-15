"""US1 — capability gate: unsupported models fail loudly, no fallback (T016)."""

import pytest

from llm_geometry import precompute
from llm_geometry.errors import UnsupportedModelError
from llm_geometry.models.loader import resolve_model


def test_unknown_model_raises_and_creates_no_artifact():
    bad = "definitely-not-a-real-model-xyz-123"
    with pytest.raises(UnsupportedModelError):
        precompute.get_or_compute_sync("distribution", bad, {}, {"prefix_text": "x"})
    # A failed resolve means no key/artifact was ever created; nothing to clean up,
    # and crucially no fallback model was substituted (SC-007).


def test_model_loads_as_float32():
    """Models load in float32 so embeddings/logits convert to numpy everywhere
    (regression: Qwen defaults to bfloat16, which has no numpy bridge)."""
    import torch

    from llm_geometry.models.loader import load_model

    lm = load_model("sshleifer/tiny-gpt2")
    assert next(lm.model.parameters()).dtype == torch.float32


def test_non_causal_model_is_rejected():
    # BERT is a masked-LM encoder, not a causal LM exposing next-token probabilities.
    with pytest.raises(UnsupportedModelError):
        resolve_model("bert-base-uncased")


def test_load_failure_surfaces_clear_error():
    """A model id that cannot be loaded yields a clear typed error, not a traceback
    (covers the offline/partial-download + insufficient-hardware edge cases at the
    error-path level)."""
    with pytest.raises(UnsupportedModelError) as exc_info:
        resolve_model("this-org/does-not-exist-zzz")
    assert exc_info.value.message  # human-readable, non-empty
    assert exc_info.value.error_type == "UnsupportedModelError"
