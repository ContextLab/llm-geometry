"""Architecture graph built from a traced real forward pass (FR-101).

Every step that transforms the hidden state appears as a node — including
parameterless/functional ops (rotary position embedding, attention softmax, residual
adds, activations) captured by `tracing.py` during one real forward pass on a short
real prompt. Edges come from actual tensor dataflow (tensor identity + storage
aliasing across views), with an execution-order fallback so the pipeline stays
connected where views/copies break identity.

Tied weights are detected via identical storage (``data_ptr``) and represented once:
the canonical `param_path` owns the tensor, aliases carry ``tied_to`` (FR-102).
The finished graph is cached via the integrity-checked store, keyed on
``(model_id, revision, graph schema)``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ..cache.keys import make_cache_key
from ..cache.store import CacheStore
from ..config import ARCH_GRAPH_SCHEMA_VERSION
from ..errors import ComputeError
from ..models.loader import LoadedModel, load_model
from .tracing import TraceEvent, encode_shape, layer_of, run_traced_forward

# A short real prompt: the graph shape is prompt-independent, the prompt just drives
# one genuine forward pass (never a fabricated tensor).
GRAPH_TRACE_TEXT = "The quick brown fox jumps over the lazy dog."
_GRAPH_TRACE_MAX_TOKENS = 12


def build_graph(
    model_id: str, use_cache: bool = True, store: CacheStore | None = None
) -> dict[str, Any]:
    """Build (or fetch from cache) the traced architecture graph for a model."""
    lm = load_model(model_id)
    store = store or CacheStore()
    key, spec = make_cache_key(
        model_id=lm.model_id,
        revision=lm.revision,
        artifact_type="arch_graph",
        params={"graph_schema": ARCH_GRAPH_SCHEMA_VERSION},
    )
    if use_cache:
        hit = store.get(key)
        if hit is not None and "graph" in hit["meta"]:
            return hit["meta"]["graph"]

    graph = _build(lm)
    store.put(
        key,
        spec,
        {"graph": graph},
        {"traced_seq_len": np.asarray([graph["meta"]["traced_seq_len"]], dtype=np.int64)},
    )
    return graph


def _tied_aliases(model: torch.nn.Module) -> dict[str, str | None]:
    """param_path -> canonical param_path sharing the same storage (or None)."""
    by_ptr: dict[int, list[str]] = {}
    for name, param in model.named_parameters(remove_duplicate=False):
        by_ptr.setdefault(param.data_ptr(), []).append(name)
    aliases: dict[str, str | None] = {}
    for names in by_ptr.values():
        canonical = names[0]  # first in registration order (e.g. embed before lm_head)
        for name in names:
            aliases[name] = None if name == canonical else canonical
    return aliases


def _node_params(
    node_id: str, module: torch.nn.Module, aliases: dict[str, str | None]
) -> list[dict[str, Any]]:
    params = []
    for pname, param in module.named_parameters(recurse=False):
        path = f"{node_id}.{pname}"
        shape = list(param.shape) if param.dim() >= 2 else [int(param.shape[0]), 1]
        params.append(
            {
                "name": pname,
                "shape": [int(d) for d in shape],
                "param_path": path,
                "tied_to": aliases.get(path),
            }
        )
    return params


def _build_edges(
    events: list[TraceEvent], first_by_node: dict[str, TraceEvent], seq_len: int
) -> list[dict[str, Any]]:
    """Edges from real tensor dataflow, with an execution-order fallback.

    A producer registry maps output tensor ids AND storage pointers to the node that
    made them (storage aliasing recovers flow through views like ``.transpose``); a
    consumer with no traceable input connects to the previous event so the pipeline
    stays a connected chain.
    """
    by_tid: dict[int, str] = {}
    by_storage: dict[int, str] = {}
    out_shape: dict[str, list[Any]] = {
        nid: encode_shape(ev.out_shapes[0], seq_len) if ev.out_shapes else []
        for nid, ev in first_by_node.items()
    }
    edges: dict[tuple[str, str], dict[str, Any]] = {}
    prev: str | None = None

    def add_edge(src: str, dst: str) -> None:
        if src != dst and (src, dst) not in edges:
            edges[(src, dst)] = {"from": src, "to": dst, "tensor_shape": out_shape.get(src, [])}

    for ev in events:
        found = False
        for tid in ev.input_tids:
            src = by_tid.get(tid)
            if src and src != ev.node_id:
                add_edge(src, ev.node_id)
                found = True
        if not found:
            # second chance: views (.transpose/.view) share storage with their base
            for storage_src in _storage_sources(ev, by_storage):
                if storage_src != ev.node_id:
                    add_edge(storage_src, ev.node_id)
                    found = True
        if not found and prev is not None:
            add_edge(prev, ev.node_id)
        for tid in ev.output_tids:
            by_tid[tid] = ev.node_id
        for ptr in ev.output_storage_ptrs:
            by_storage.setdefault(ptr, ev.node_id)
        prev = ev.node_id
    return list(edges.values())


def _storage_sources(ev: TraceEvent, by_storage: dict[int, str]) -> list[str]:
    return [by_storage[p] for p in ev.input_storage_ptrs if p in by_storage]


def _build(lm: LoadedModel) -> dict[str, Any]:
    ids = list(lm.tokenizer(GRAPH_TRACE_TEXT)["input_ids"])[:_GRAPH_TRACE_MAX_TOKENS]
    seq_len = len(ids)
    input_ids = torch.tensor([ids], dtype=torch.long, device=lm.device)

    _, events = run_traced_forward(lm.model, input_ids)
    if not events:
        raise ComputeError(f"Tracing '{lm.model_id}' produced no events — cannot build a graph.")

    # first event per node id, in execution order
    first_by_node: dict[str, TraceEvent] = {}
    for ev in events:
        first_by_node.setdefault(ev.node_id, ev)

    aliases = _tied_aliases(lm.model)
    modules_by_path = dict(lm.model.named_modules())

    layered_orders = [ev.order for ev in first_by_node.values() if layer_of(ev.node_id) is not None]
    first_layer_order = min(layered_orders) if layered_orders else 0

    nodes: list[dict[str, Any]] = []
    for nid, ev in first_by_node.items():
        layer = layer_of(nid)
        if layer is not None:
            group = f"layer_{layer}"
        else:
            group = "stem" if ev.order < first_layer_order else "head"
        params = (
            _node_params(nid, modules_by_path[nid], aliases)
            if ev.op == "module" and nid in modules_by_path
            else []
        )
        nodes.append(
            {
                "id": nid,
                "kind": ev.kind,
                "op": ev.op,
                "label": ev.label,
                "layer": layer,
                "group": group,
                "params": params,
            }
        )

    # Completeness (FR-101): every named parameter belongs to exactly one node.
    all_params = [name for name, _ in lm.model.named_parameters(remove_duplicate=False)]
    covered = [p["param_path"] for node in nodes for p in node["params"]]
    missing = sorted(set(all_params) - set(covered))
    if missing:
        raise ComputeError(
            f"Traced graph for '{lm.model_id}' is incomplete: {len(missing)} parameters "
            f"belong to modules that never executed (e.g. {missing[:5]})."
        )
    if len(covered) != len(set(covered)):
        raise ComputeError(f"Traced graph for '{lm.model_id}' assigned a parameter twice.")

    # Functional completeness (FR-101): parameter coverage alone can't catch a missed
    # softmax/residual — a degraded-but-plausible graph would render fine. For the
    # decoder-only models this feature targets, every decoder layer must expose the
    # attention softmax and both residual adds; fail loud rather than draw an
    # incomplete architecture.
    by_layer: dict[int, dict[str, int]] = {}
    for node in nodes:
        if node["layer"] is not None and node["op"] == "functional":
            counts = by_layer.setdefault(int(node["layer"]), {})
            counts[node["kind"]] = counts.get(node["kind"], 0) + 1
    n_layers_cfg = int(getattr(lm.model.config, "num_hidden_layers", lm.num_layers))
    bad = [
        k
        for k in range(n_layers_cfg)
        if by_layer.get(k, {}).get("attention_softmax", 0) < 1
        or by_layer.get(k, {}).get("residual_add", 0) < 2
    ]
    if bad:
        raise ComputeError(
            f"Traced graph for '{lm.model_id}' is missing functional steps "
            f"(attention_softmax/residual_add) in decoder layer(s) {bad[:5]} — "
            "the 1-to-1 visualization guarantee would be violated."
        )

    config = lm.model.config
    heads = int(getattr(config, "num_attention_heads", 0) or 0)
    meta = {
        "n_layers": int(getattr(config, "num_hidden_layers", lm.num_layers)),
        "hidden": int(getattr(config, "hidden_size", lm.hidden_size)),
        "heads": heads,
        "kv_heads": int(getattr(config, "num_key_value_heads", heads) or heads),
        "vocab": int(getattr(config, "vocab_size", lm.vocab_size)),
        "total_params": int(sum(p.numel() for p in lm.model.parameters())),
        "traced_seq_len": seq_len,
    }

    return {
        "model_id": lm.model_id,
        "schema_version": ARCH_GRAPH_SCHEMA_VERSION,
        "meta": meta,
        "nodes": nodes,
        "edges": _build_edges(events, first_by_node, seq_len),
    }
