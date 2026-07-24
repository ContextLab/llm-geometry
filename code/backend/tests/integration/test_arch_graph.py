"""SC-102 / FR-101 — traced architecture graphs for two real instruct models.

The graph must come from a traced real forward pass: parameterless/functional steps
(rope, attention softmax, residual adds, activations) appear as nodes, every named
parameter belongs to exactly one node (ties aliased via ``tied_to``), meta matches
the model's real config, and node ids are stable across rebuilds.
"""

from collections import Counter

import pytest

from llm_geometry.arch.graph import build_graph
from llm_geometry.models.loader import load_model

MODELS = ["HuggingFaceTB/SmolLM2-135M-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"]


@pytest.fixture(scope="module", params=MODELS, ids=["smollm2-135m", "qwen2.5-0.5b"])
def graph_lm(request):
    return build_graph(request.param), load_model(request.param)


def test_graph_has_contract_shape(graph_lm):
    graph, lm = graph_lm
    assert graph["model_id"] == lm.model_id
    assert isinstance(graph["schema_version"], int)
    assert graph["nodes"] and graph["edges"]
    node_ids = [n["id"] for n in graph["nodes"]]
    assert len(node_ids) == len(set(node_ids))  # unique, stable ids
    for node in graph["nodes"]:
        assert node["op"] in ("module", "functional")
        assert node["group"] == (
            f"layer_{node['layer']}" if node["layer"] is not None else node["group"]
        )
        assert node["group"] in ("stem", "head") or node["group"].startswith("layer_")
        for p in node["params"]:
            assert p["name"] and p["param_path"].startswith(node["id"])
            assert len(p["shape"]) >= 2 and all(d >= 1 for d in p["shape"])
    ids = set(node_ids)
    for edge in graph["edges"]:
        assert edge["from"] in ids and edge["to"] in ids and edge["from"] != edge["to"]
        assert isinstance(edge["tensor_shape"], list)


def test_functional_ops_are_traced_nodes(graph_lm):
    """FR-101: parameterless ops appear as graph nodes, one per real occurrence."""
    graph, _ = graph_lm
    n_layers = graph["meta"]["n_layers"]
    functional = Counter(n["kind"] for n in graph["nodes"] if n["op"] == "functional")
    assert functional["rope"] == n_layers
    assert functional["attention_softmax"] == n_layers
    assert functional["residual_add"] == 2 * n_layers
    assert any(n["kind"] == "activation" for n in graph["nodes"])
    assert sum(1 for n in graph["nodes"] if n["kind"] == "activation") == n_layers


def test_every_parameter_belongs_to_exactly_one_node(graph_lm):
    """Completeness: the traced node set covers the full parameter set (SC-102)."""
    graph, lm = graph_lm
    all_params = sorted(name for name, _ in lm.model.named_parameters(remove_duplicate=False))
    covered = sorted(p["param_path"] for n in graph["nodes"] for p in n["params"])
    assert covered == all_params  # everything covered, nothing twice


def test_tied_lm_head_aliased_once(graph_lm):
    """Tied tensors are represented once and aliased via ``tied_to`` (FR-102)."""
    graph, lm = graph_lm
    assert lm.model.config.tie_word_embeddings
    params = {p["param_path"]: p for n in graph["nodes"] for p in n["params"]}
    embed_path = next(
        p for p in params if p.endswith("embed_tokens.weight") or p.endswith("wte.weight")
    )
    assert params[embed_path]["tied_to"] is None  # canonical owner
    assert params["lm_head.weight"]["tied_to"] == embed_path


def test_meta_matches_real_config(graph_lm):
    graph, lm = graph_lm
    config = lm.model.config
    meta = graph["meta"]
    assert meta["n_layers"] == config.num_hidden_layers
    assert meta["hidden"] == config.hidden_size
    assert meta["heads"] == config.num_attention_heads
    assert meta["kv_heads"] == getattr(config, "num_key_value_heads", config.num_attention_heads)
    assert meta["vocab"] == config.vocab_size
    assert meta["total_params"] == sum(p.numel() for p in lm.model.parameters())
    assert meta["traced_seq_len"] >= 1


def test_node_ids_stable_across_builds(graph_lm):
    """Deterministic execution -> identical node ids on a fresh (uncached) rebuild."""
    graph, lm = graph_lm
    fresh = build_graph(lm.model_id, use_cache=False)
    assert [n["id"] for n in fresh["nodes"]] == [n["id"] for n in graph["nodes"]]
    assert [n["kind"] for n in fresh["nodes"]] == [n["kind"] for n in graph["nodes"]]
    cached = build_graph(lm.model_id)  # cache hit path
    assert [n["id"] for n in cached["nodes"]] == [n["id"] for n in graph["nodes"]]
