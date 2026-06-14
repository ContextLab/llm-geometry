"""US1 — real next-token distributions (T013). Uses distilgpt2 (a real model)."""

import numpy as np
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_geometry.errors import InvalidParamError
from llm_geometry.precompute import get_or_compute_sync

MODEL = "distilgpt2"


def test_distribution_sums_to_one_and_top_matches_direct_argmax():
    prefix = "The capital of France is"
    payload = get_or_compute_sync("distribution", MODEL, {"temperature": 1.0}, {"prefix_text": prefix})
    probs = payload["arrays"]["probs"]
    assert probs.ndim == 1
    assert abs(float(probs.sum()) - 1.0) < 1e-3
    assert float(probs.min()) >= 0.0

    # Independent forward pass: our top token must equal the real argmax.
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL)
    model.eval()
    ids = tok(prefix, return_tensors="pt")
    with torch.no_grad():
        logits = model(**ids).logits[0, -1, :]
    assert int(payload["meta"]["top_token"]) == int(torch.argmax(logits))


def test_temperature_zero_is_argmax_onehot():
    probs = get_or_compute_sync("distribution", MODEL, {"temperature": 0.0}, {"prefix_text": "Hello"})["arrays"]["probs"]
    assert int((probs > 0).sum()) == 1
    assert abs(float(probs.sum()) - 1.0) < 1e-6


def test_negative_temperature_raises_invalid_param():
    with pytest.raises(InvalidParamError):
        get_or_compute_sync("distribution", MODEL, {"temperature": -0.5}, {"prefix_text": "x"})


def test_higher_temperature_increases_entropy():
    """Temperature should flatten the distribution (sanity on the real math)."""
    low = get_or_compute_sync("distribution", MODEL, {"temperature": 0.5}, {"prefix_text": "Hello"})["arrays"]["probs"]
    high = get_or_compute_sync("distribution", MODEL, {"temperature": 2.0}, {"prefix_text": "Hello"})["arrays"]["probs"]

    def entropy(p):
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())

    assert entropy(high) > entropy(low)
