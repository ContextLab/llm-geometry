"""FR-101 — per-node activation traces share the graph's node universe (SC-102).

One real forward pass with real attentions/hidden states: attention rows are
probability distributions, hidden-state summaries have the right shapes, and
``node_activations`` covers exactly the graph's node-id set (the completeness
invariant both sides build from the same tracer).
"""

import numpy as np
import pytest

from llm_geometry.arch.graph import build_graph
from llm_geometry.arch.trace import trace_forward
from llm_geometry.arch.tracing import encode_prompt
from llm_geometry.models.loader import load_model

MODELS = ["HuggingFaceTB/SmolLM2-135M-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"]
PROMPT = "The capital of France is Paris."
MAX_CONTEXT = 16


@pytest.fixture(scope="module", params=MODELS, ids=["smollm2-135m", "qwen2.5-0.5b"])
def traced(request):
    trace = trace_forward(request.param, PROMPT, max_context=MAX_CONTEXT)
    graph = build_graph(request.param)
    return trace, graph, request.param


def test_token_count_within_max_context(traced):
    trace, _, _ = traced
    assert 1 <= len(trace["tokens"]) <= MAX_CONTEXT
    assert all(isinstance(t["id"], int) and isinstance(t["text"], str) for t in trace["tokens"])


def test_truncates_left_keeping_recent_context(traced):
    trace, _, model_id = traced
    lm = load_model(model_id)
    full_ids, _ = encode_prompt(lm, PROMPT)
    assert [t["id"] for t in trace["tokens"]] == full_ids[-MAX_CONTEXT:]


def test_chat_template_used_for_instruct_models(traced):
    trace, _, _ = traced
    assert trace["chat_template_used"] is True


def test_attention_rows_are_distributions_per_head(traced):
    """Short prompt -> no downsampling, so raw rows must be row-stochastic."""
    trace, graph, _ = traced
    assert len(trace["layers"]) == graph["meta"]["n_layers"]
    n_tokens = len(trace["tokens"])
    for layer in trace["layers"]:
        assert layer["attention_downsampled"] is False
        att = np.array(layer["attention"])
        assert att.shape == (graph["meta"]["heads"], n_tokens, n_tokens)
        assert np.allclose(att.sum(axis=-1), 1.0, atol=1e-3)
        assert att.min() >= 0.0


def test_hidden_summaries_have_contract_shapes(traced):
    trace, _, _ = traced
    n_tokens = len(trace["tokens"])
    for layer in trace["layers"]:
        norms = np.array(layer["hidden_norm"])
        assert norms.shape == (n_tokens,) and (norms > 0).all()
        assert np.array(layer["hidden_pca3"]).shape == (n_tokens, 3)


def test_node_activations_cover_exactly_the_graph_nodes(traced):
    """The completeness invariant: same node-id universe as /api/arch/graph."""
    trace, graph, _ = traced
    graph_ids = {n["id"] for n in graph["nodes"]}
    activation_ids = [a["node_id"] for a in trace["node_activations"]]
    assert len(activation_ids) == len(set(activation_ids))  # one entry per node
    assert set(activation_ids) == graph_ids
    for act in trace["node_activations"]:
        assert np.isfinite(act["out_norm"]) and act["out_norm"] >= 0
        assert isinstance(act["out_shape"], list)


def test_logits_topk_are_real_probabilities(traced):
    trace, _, _ = traced
    topk = trace["logits_topk"]
    assert len(topk["ids"]) == len(topk["texts"]) == len(topk["probs"]) == 10
    assert all(0 < p <= 1 for p in topk["probs"])
    assert topk["probs"] == sorted(topk["probs"], reverse=True)


def test_gpt2_trace_covers_graph_nodes():
    """GPT-2 smoke check: trace and graph share one node-id universe (SC-102)."""
    trace = trace_forward("gpt2", PROMPT, max_context=MAX_CONTEXT)
    graph = build_graph("gpt2")
    assert trace["chat_template_used"] is False  # base gpt2 has no chat template
    activation_ids = [a["node_id"] for a in trace["node_activations"]]
    assert len(activation_ids) == len(set(activation_ids))  # one entry per node
    assert set(activation_ids) == {n["id"] for n in graph["nodes"]}


def test_trace_reports_left_truncation():
    """Red-team F9: a prompt over max_context must set the additive `truncated` flag."""
    from llm_geometry.arch.trace import trace_forward

    short = trace_forward(MODELS[0], "hello there", max_context=64)
    assert short["truncated"] is False
    long_prompt = " word" * 300
    long = trace_forward(MODELS[0], long_prompt, max_context=64)
    assert long["truncated"] is True
    assert len(long["tokens"]) == 64
