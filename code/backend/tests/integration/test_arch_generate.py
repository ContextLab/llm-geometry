"""`/api/arch/generate` — real autoregressive generation, reproducible when seeded."""

import pytest

from llm_geometry.arch.generate import _eos_ids, generate
from llm_geometry.errors import InvalidParamError
from llm_geometry.models.loader import load_model

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


def test_greedy_is_deterministic():
    a = generate(MODEL, "What is 2+2?", temperature=0, max_new_tokens=12)
    b = generate(MODEL, "What is 2+2?", temperature=0, max_new_tokens=12)
    assert [t["id"] for t in a["tokens"]] == [t["id"] for t in b["tokens"]]
    assert a["text"] == b["text"] and a["text"].strip()


def test_seeded_sampling_is_reproducible():
    a = generate(MODEL, "Tell me something.", temperature=0.8, max_new_tokens=12, seed=1234)
    b = generate(MODEL, "Tell me something.", temperature=0.8, max_new_tokens=12, seed=1234)
    assert [t["id"] for t in a["tokens"]] == [t["id"] for t in b["tokens"]]
    assert a["text"] == b["text"]


def test_unseeded_generation_respects_max_new_tokens():
    res = generate(MODEL, "Write a long story about the sea.", temperature=0.8, max_new_tokens=8)
    assert 1 <= len(res["tokens"]) <= 8
    assert res["finish_reason"] in ("eos", "length")


def test_finish_reason_eos_when_reply_ends():
    res = generate(MODEL, "Say only the word hello.", temperature=0, max_new_tokens=128)
    assert res["finish_reason"] == "eos"
    assert len(res["tokens"]) < 128
    assert res["tokens"][-1]["id"] in _eos_ids(load_model(MODEL))


def test_finish_reason_length_when_budget_exhausted():
    res = generate(MODEL, "Count upward from one forever.", temperature=0, max_new_tokens=3)
    assert res["finish_reason"] == "length"
    assert len(res["tokens"]) == 3


def test_token_probs_and_topk_shape():
    res = generate(MODEL, "What color is the sky?", temperature=0.7, max_new_tokens=6, seed=0)
    for tok in res["tokens"]:
        assert 0 < tok["prob"] <= 1  # probability under the pre-sample distribution
        topk = tok["topk"]
        assert len(topk["ids"]) == len(topk["texts"]) == len(topk["probs"]) == 5
        assert topk["probs"] == sorted(topk["probs"], reverse=True)


def test_invalid_params_rejected():
    with pytest.raises(InvalidParamError):
        generate(MODEL, "hi", temperature=-0.1)
    with pytest.raises(InvalidParamError):
        generate(MODEL, "hi", max_new_tokens=129)
    with pytest.raises(InvalidParamError):
        generate(MODEL, "   ")


def test_greedy_topk_reports_model_distribution_not_one_hot():
    """Red-team F4: at temperature 0 the top-k alternatives must carry the MODEL's
    probabilities (softmax of the logits), not the degenerate one-hot sampling probs."""
    out = generate(MODEL, "The capital of France is", temperature=0.0, max_new_tokens=1)
    tok = out["tokens"][0]
    assert tok["prob"] == 1.0  # sampling prob of the greedy choice stays 1.0
    probs = tok["topk"]["probs"]
    assert all(0.0 < p <= 1.0 for p in probs)
    assert sorted(probs, reverse=True) == probs
    # the alternatives are NOT all zero (the old one-hot bug)
    assert sum(probs[1:]) > 0.0
