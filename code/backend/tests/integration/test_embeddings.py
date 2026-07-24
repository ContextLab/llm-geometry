"""US1 — per-layer embeddings (T014). Uses tiny-gpt2 for speed."""

import pytest

from llm_geometry.errors import InvalidParamError
from llm_geometry.precompute import get_or_compute_sync

MODEL = "sshleifer/tiny-gpt2"
REF = 100


def test_static_embeddings_shape():
    payload = get_or_compute_sync("embeddings", MODEL, {"source": "static", "reference_set_size": REF})
    vectors = payload["arrays"]["vectors"]
    hidden = payload["meta"]["shape"][1]
    assert vectors.shape == (REF, hidden)
    assert payload["arrays"]["token_ids"].shape == (REF,)


def test_contextual_embeddings_shape():
    payload = get_or_compute_sync("embeddings", MODEL, {"source": "contextual", "layer": 1, "reference_set_size": REF})
    vectors = payload["arrays"]["vectors"]
    assert vectors.ndim == 2
    assert vectors.shape[0] == REF
    assert payload["meta"]["source"] == "contextual"


def test_layer_out_of_range_raises():
    with pytest.raises(InvalidParamError):
        get_or_compute_sync("embeddings", MODEL, {"source": "contextual", "layer": 999, "reference_set_size": 10})


def test_invalid_source_raises():
    with pytest.raises(InvalidParamError):
        get_or_compute_sync("embeddings", MODEL, {"source": "nonsense", "reference_set_size": 10})
