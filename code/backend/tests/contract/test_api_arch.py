"""`/api/arch/*` — contract tests against the real app with real models (no mocks).

Locks the frozen contract in specs/002-interactive-model-explorer/contracts/api.md:
graph/weights/trace/generate response shapes, the pre-download size gate, the 6
significant-digit array encoding, and the typed-error envelope for every failure.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from llm_geometry.api.app import app
from llm_geometry.arch.graph import build_graph
from llm_geometry.arch.tracing import encode_prompt
from llm_geometry.models.loader import load_model

client = TestClient(app)

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"  # small, locally cached (Batch 1)
BIASED_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # Llama-style WITH attention biases
TOO_BIG = "Qwen/Qwen2.5-7B-Instruct"  # ~7.6B params, over the 1.5B ceiling
PARAM = "model.layers.0.self_attn.q_proj.weight"
PROMPT = "The capital of France is Paris."
MAX_CONTEXT = 16
NODE_KINDS = {
    "embedding",
    "linear",
    "layernorm",
    "rmsnorm",
    "rope",
    "attention_softmax",
    "residual_add",
    "activation",
    "mlp",
    "lm_head",
    "other",
}


def sig6(x: float) -> float:
    """The contract's array encoding: 6 significant digits (``%.6g``)."""
    return float(f"{float(x):.6g}")


@pytest.fixture(scope="module")
def lm():
    return load_model(MODEL)


@pytest.fixture(scope="module")
def graph_body():
    resp = client.get("/api/arch/graph", params={"model_id": MODEL})
    assert resp.status_code == 200
    return resp.json()


@pytest.fixture(scope="module")
def trace_body():
    resp = client.get(
        "/api/arch/trace",
        params={"model_id": MODEL, "prompt": PROMPT, "max_context": MAX_CONTEXT},
    )
    assert resp.status_code == 200
    return resp.json()


# --- GET /api/arch/graph ------------------------------------------------------------


def test_graph_contract_shape(graph_body):
    assert graph_body["model_id"] == MODEL
    assert isinstance(graph_body["schema_version"], int)
    meta = graph_body["meta"]
    for key in ("n_layers", "hidden", "heads", "kv_heads", "vocab", "total_params"):
        assert isinstance(meta[key], int) and meta[key] >= 1
    assert meta["traced_seq_len"] >= 1
    assert graph_body["nodes"] and graph_body["edges"]
    node_ids = [n["id"] for n in graph_body["nodes"]]
    assert len(node_ids) == len(set(node_ids))
    for node in graph_body["nodes"]:
        assert node["kind"] in NODE_KINDS
        assert node["op"] in ("module", "functional")
        assert isinstance(node["label"], str) and node["label"]
        assert node["layer"] is None or isinstance(node["layer"], int)
        assert node["group"] in ("stem", "head") or node["group"].startswith("layer_")
        for p in node["params"]:
            assert p["name"] in ("weight", "bias")
            assert len(p["shape"]) == 2 and all(isinstance(d, int) and d >= 1 for d in p["shape"])
            assert p["param_path"].startswith(node["id"])
    ids = set(node_ids)
    for edge in graph_body["edges"]:
        assert edge["from"] in ids and edge["to"] in ids
        assert isinstance(edge["tensor_shape"], list)


def test_graph_node_ids_match_direct_build(graph_body):
    """The endpoint serves the same node universe a direct arch.graph build makes."""
    direct = build_graph(MODEL)
    assert {n["id"] for n in graph_body["nodes"]} == {n["id"] for n in direct["nodes"]}
    assert graph_body["schema_version"] == direct["schema_version"]
    assert graph_body["meta"] == direct["meta"]


def test_graph_tied_weights_aliased_in_response(graph_body):
    """Tied tensors appear once; aliases point at the canonical param (FR-102)."""
    params = {p["param_path"]: p for n in graph_body["nodes"] for p in n["params"]}
    assert params["lm_head.weight"]["tied_to"] == "model.embed_tokens.weight"
    assert params["model.embed_tokens.weight"]["tied_to"] is None


def test_graph_oversized_model_is_422_envelope():
    """The size gate fires BEFORE any load: 422 ModelTooLargeError, no download."""
    resp = client.get("/api/arch/graph", params={"model_id": TOO_BIG})
    assert resp.status_code == 422
    err = resp.json()["error"]
    assert err["type"] == "ModelTooLargeError"
    assert err["detail"]["total_params"] > err["detail"]["max_params"]


def test_graph_unknown_model_is_422_envelope():
    resp = client.get("/api/arch/graph", params={"model_id": "definitely-not-real-xyz-123"})
    assert resp.status_code == 422
    assert resp.json()["error"]["type"] == "UnsupportedModelError"


# --- GET /api/arch/weights ----------------------------------------------------------


def test_weights_exact_window_equals_state_dict_slice(lm):
    resp = client.get(
        "/api/arch/weights",
        params={"model_id": MODEL, "param": PARAM, "r0": 3, "r1": 11, "c0": 5, "c1": 13},
    )
    assert resp.status_code == 200
    body = resp.json()
    window = lm.model.state_dict()[PARAM][3:11, 5:13].detach().float().cpu().numpy()
    assert body["method"] == "exact" and body["downsampled"] is False
    assert body["grid_shape"] == [8, 8]
    assert [body["r0"], body["r1"], body["c0"], body["c1"]] == [3, 11, 5, 13]
    # real values, elementwise, after the contract's 6-significant-digit rounding
    assert body["values"] == [[sig6(v) for v in row] for row in window.tolist()]
    assert body["stats"]["min"] == sig6(float(window.min()))
    assert body["stats"]["max"] == sig6(float(window.max()))
    assert body["stats"]["mean"] == sig6(float(window.mean()))
    assert body["stats"]["std"] == sig6(float(window.std()))


def test_weights_large_window_downsamples_to_grid(lm):
    resp = client.get("/api/arch/weights", params={"model_id": MODEL, "param": PARAM})
    assert resp.status_code == 200
    body = resp.json()
    full = lm.model.state_dict()[PARAM].detach().float().cpu().numpy()
    assert body["shape"] == list(full.shape)
    assert body["downsampled"] is True and body["method"] == "strided_mean"
    gr, gc = body["grid_shape"]
    assert gr <= 64 and gc <= 64
    assert len(body["values"]) == gr and all(len(row) == gc for row in body["values"])
    grid = np.asarray(body["values"], dtype=np.float64)
    assert grid.min() >= sig6(float(full.min())) and grid.max() <= sig6(float(full.max()))


def test_weights_bias_served_as_single_column():
    """1-D params (biases) use C=1 per the contract."""
    bias_path = "model.layers.0.self_attn.q_proj.bias"
    resp = client.get(
        "/api/arch/weights",
        params={"model_id": BIASED_MODEL, "param": bias_path, "r0": 0, "r1": 8},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["shape"][1] == 1
    assert body["grid_shape"] == [8, 1]
    bias = load_model(BIASED_MODEL).model.state_dict()[bias_path][0:8]
    expected = bias.detach().float().cpu().numpy()
    assert body["values"] == [[sig6(v)] for v in expected.tolist()]


def test_weights_unknown_param_is_404_envelope():
    resp = client.get(
        "/api/arch/weights",
        params={"model_id": MODEL, "param": "model.layers.0.no_such_param.weight"},
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["type"] == "NotFoundError"


# --- GET /api/arch/trace ------------------------------------------------------------


def test_trace_contract_shape(trace_body, graph_body):
    n_tokens = len(trace_body["tokens"])
    assert 1 <= n_tokens <= MAX_CONTEXT
    assert all(
        isinstance(t["id"], int) and isinstance(t["text"], str) for t in trace_body["tokens"]
    )
    assert trace_body["chat_template_used"] is True
    meta = graph_body["meta"]
    assert len(trace_body["layers"]) == meta["n_layers"]
    for k, layer in enumerate(trace_body["layers"]):
        assert layer["layer"] == k
        assert layer["attention_downsampled"] is False  # short prompt: raw attention
        att = np.asarray(layer["attention"], dtype=np.float64)
        assert att.shape == (meta["heads"], n_tokens, n_tokens)
        assert att.min() >= 0.0
        assert len(layer["hidden_norm"]) == n_tokens
        assert np.asarray(layer["hidden_pca3"], dtype=np.float64).shape == (n_tokens, 3)
    topk = trace_body["logits_topk"]
    assert len(topk["ids"]) == len(topk["texts"]) == len(topk["probs"]) == 10
    assert all(0 < p <= 1 for p in topk["probs"])
    assert topk["probs"] == sorted(topk["probs"], reverse=True)


def test_trace_node_activations_cover_the_graph_nodes(trace_body, graph_body):
    """Completeness invariant shared with /api/arch/graph: same node-id universe."""
    ids = [a["node_id"] for a in trace_body["node_activations"]]
    assert len(ids) == len(set(ids))  # one entry per traced node
    assert set(ids) == {n["id"] for n in graph_body["nodes"]}
    for act in trace_body["node_activations"]:
        assert np.isfinite(act["out_norm"]) and act["out_norm"] >= 0
        assert isinstance(act["out_shape"], list)


def test_trace_max_context_truncates_left(trace_body, lm):
    """T <= max_context, keeping the MOST RECENT tokens (left truncation)."""
    full_ids, used_template = encode_prompt(lm, PROMPT)
    assert used_template is True
    assert len(full_ids) > MAX_CONTEXT  # the chat template makes truncation real
    assert [t["id"] for t in trace_body["tokens"]] == full_ids[-MAX_CONTEXT:]


def test_trace_empty_prompt_is_400_envelope():
    resp = client.get("/api/arch/trace", params={"model_id": MODEL, "prompt": "   "})
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "InvalidParamError"


def test_trace_floats_rounded_to_6_significant_digits(trace_body):
    layer0 = trace_body["layers"][0]
    norms = layer0["hidden_norm"]
    assert norms and all(v == sig6(v) for v in norms)
    flat_attention = np.asarray(layer0["attention"], dtype=np.float64).ravel()
    assert all(float(v) == sig6(v) for v in flat_attention)
    assert all(v == sig6(v) for row in layer0["hidden_pca3"] for v in row)


# --- POST /api/arch/generate --------------------------------------------------------


def _generate(payload):
    return client.post("/api/arch/generate", json={"model_id": MODEL, **payload})


def test_generate_temperature_zero_is_deterministic():
    payload = {"prompt": "What is 2+2?", "temperature": 0, "max_new_tokens": 8}
    a = _generate(payload)
    b = _generate(payload)
    assert a.status_code == b.status_code == 200
    a, b = a.json(), b.json()
    assert [t["id"] for t in a["tokens"]] == [t["id"] for t in b["tokens"]]
    assert a["text"] == b["text"]
    assert all(t["prob"] == 1.0 for t in a["tokens"])  # greedy: one-hot distribution


def test_generate_seeded_sampling_is_deterministic():
    payload = {"prompt": "Tell me something.", "temperature": 0.8, "max_new_tokens": 8, "seed": 42}
    a = _generate(payload).json()
    b = _generate(payload).json()
    assert [t["id"] for t in a["tokens"]] == [t["id"] for t in b["tokens"]]
    assert a["text"] == b["text"]


def test_generate_token_shape_probs_and_rounding():
    resp = _generate(
        {"prompt": "What color is the sky?", "temperature": 0.7, "max_new_tokens": 6, "seed": 0}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body["text"], str)
    assert body["finish_reason"] in ("eos", "length")
    assert 1 <= len(body["tokens"]) <= 6
    for tok in body["tokens"]:
        assert isinstance(tok["id"], int) and isinstance(tok["text"], str)
        assert 0 < tok["prob"] <= 1
        assert tok["prob"] == sig6(tok["prob"])  # 6-significant-digit spot check
        topk = tok["topk"]
        assert len(topk["ids"]) == len(topk["texts"]) == len(topk["probs"]) == 5
        assert all(0 < p <= 1 and p == sig6(p) for p in topk["probs"])
        assert topk["probs"] == sorted(topk["probs"], reverse=True)


def test_generate_finish_reason_length_when_budget_exhausted():
    body = _generate(
        {"prompt": "Count upward from one forever.", "temperature": 0, "max_new_tokens": 3}
    ).json()
    assert body["finish_reason"] == "length"
    assert len(body["tokens"]) == 3


def test_generate_max_new_tokens_over_cap_is_400_envelope():
    resp = _generate({"prompt": "hi", "max_new_tokens": 129})
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "InvalidParamError"


def test_generate_oversized_model_is_422_envelope():
    resp = client.post(
        "/api/arch/generate", json={"model_id": TOO_BIG, "prompt": "hi", "max_new_tokens": 4}
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["type"] == "ModelTooLargeError"
