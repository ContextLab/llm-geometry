"""Capability gate: unsupported models fail loudly, no fallback (T016).

Feature 004 removed the precompute pipeline, so the "nothing is created for a bad
model, and no substitute is silently used" property is now asserted through the
surviving path the Architecture Explorer actually calls: POST /api/models/resolve.
"""

import pytest
from fastapi.testclient import TestClient

from llm_geometry.api.app import app
from llm_geometry.errors import UnsupportedModelError
from llm_geometry.models.loader import resolve_model

client = TestClient(app)


def test_unknown_model_is_rejected_with_no_fallback():
    bad = "definitely-not-a-real-model-xyz-123"
    resp = client.post("/api/models/resolve", json={"model_id": bad})
    assert resp.status_code == 422
    err = resp.json()["error"]
    assert err["type"] == "UnsupportedModelError"
    assert err["message"]  # human-readable
    # Crucially, no substitute model is handed back in place of the bad one (SC-007).
    assert "model_id" not in err


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
