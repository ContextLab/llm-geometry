"""Traced real forward passes — the shared machinery behind the architecture graph
(`graph.py`) and the per-node activation trace (`trace.py`).

One real forward pass is observed three ways at once (FR-101):

* **forward hooks** on every module maintain a "currently executing module" stack and
  record one event per *leaf* module (embeddings, linears, norms, activations, rotary
  tables, lm_head);
* a **``TorchFunctionMode``** intercepts functional ops that never appear in
  ``named_modules()`` — the attention softmax (eager ``F.softmax`` or fused
  ``F.scaled_dot_product_attention``) and the residual-stream adds — and attributes
  each to the innermost executing module;
* the model family's ``apply_rotary_pos_emb`` function is **wrapped in place** for the
  duration of the pass, so rotary position embedding shows up as a real traced step of
  each attention block.

Both callers therefore share the exact same node-id universe (the completeness
invariant tested for SC-102): graph nodes and ``node_activations`` are two views of
the same traced event stream, never two separate enumerations.
"""

from __future__ import annotations

import re
import sys
import threading
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from torch.overrides import TorchFunctionMode

_LAYER_RE = re.compile(r"(?:^|\.)(?:layers|h|blocks)\.(\d+)(?:\.|$)")

_ACTIVATION_TYPES = (nn.SiLU, nn.GELU, nn.ReLU, nn.Tanh, nn.Sigmoid, nn.LeakyReLU, nn.ELU, nn.Mish)
_ADD_NAMES = {"add", "add_", "__add__", "__iadd__", "__radd__"}
_SOFTMAX_NAMES = {"softmax", "scaled_dot_product_attention"}
_FUNC_ACT_NAMES = {"silu", "gelu", "relu", "tanh", "sigmoid", "leaky_relu", "mish"}

_LEAF_LABELS = {
    "q_proj": "query projection",
    "k_proj": "key projection",
    "v_proj": "value projection",
    "o_proj": "attention output projection",
    "gate_proj": "MLP gate projection",
    "up_proj": "MLP up projection",
    "down_proj": "MLP down projection",
    "input_layernorm": "pre-attention norm",
    "post_attention_layernorm": "pre-MLP norm",
    "embed_tokens": "token embedding",
    "lm_head": "LM head (unembedding)",
    "norm": "final norm",
    "act_fn": "MLP activation",
    "rotary_emb": "rotary position table (cos/sin)",
}

# Tracing mutates process-global state (hooks, a monkeypatched rope function, the
# model's attention-implementation flag) — serialize traced forwards.
_TRACE_LOCK = threading.Lock()


@dataclass
class TraceEvent:
    """One traced step of the forward pass (a module call or a functional op)."""

    order: int
    node_id: str
    kind: str
    op: str  # "module" | "functional"
    label: str
    owner: str  # dotted path of the (innermost) module that ran this step
    out_norm: float
    out_shapes: list[tuple[int, ...]]
    input_tids: list[int] = field(default_factory=list)
    input_storage_ptrs: list[int] = field(default_factory=list)
    output_tids: list[int] = field(default_factory=list)
    output_storage_ptrs: list[int] = field(default_factory=list)


def classify_module(module: nn.Module, lm_head_module: nn.Module | None) -> str:
    """Map a leaf module to the contract's node-kind enum."""
    if lm_head_module is not None and module is lm_head_module:
        return "lm_head"
    if isinstance(module, nn.Embedding):
        return "embedding"
    if isinstance(module, nn.Linear) or type(module).__name__ == "Conv1D":
        return "linear"
    if isinstance(module, nn.LayerNorm):
        return "layernorm"
    cls_name = type(module).__name__
    if "RMSNorm" in cls_name:
        return "rmsnorm"
    if "Rotary" in cls_name:
        return "rope"
    if isinstance(module, _ACTIVATION_TYPES) or "activations" in type(module).__module__:
        return "activation"
    return "other"


def layer_of(node_id: str) -> int | None:
    """Transformer block index parsed from a dotted node id (``model.layers.7.…``)."""
    m = _LAYER_RE.search(node_id)
    return int(m.group(1)) if m else None


def encode_shape(shape: tuple[int, ...], seq_len: int) -> list[Any]:
    """Contract shape encoding: drop the leading batch dim, sequence dims -> ``"T"``."""
    dims = list(shape)
    if dims and dims[0] == 1:
        dims = dims[1:]
    return ["T" if d == seq_len else int(d) for d in dims]


def encode_prompt(lm: Any, prompt: str, system_prompt: str | None = None) -> tuple[list[int], bool]:
    """Token ids for a single-turn prompt, via the model's chat template when available.

    Returns ``(ids, chat_template_used)``. Without a chat template the system prompt is
    prepended as plain text.
    """
    tok = lm.tokenizer
    if getattr(tok, "chat_template", None):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        ids = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        if hasattr(ids, "keys"):  # dict / BatchEncoding (transformers v5)
            ids = ids["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return [int(i) for i in ids], True
    text = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
    return [int(i) for i in tok(text)["input_ids"]], False


def _flatten_tensors(obj: Any) -> list[torch.Tensor]:
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, (tuple, list)):
        out: list[torch.Tensor] = []
        for item in obj:
            out.extend(_flatten_tensors(item))
        return out
    if isinstance(obj, dict):
        out = []
        for item in obj.values():
            out.extend(_flatten_tensors(item))
        return out
    return []


def _norm_of(tensors: list[torch.Tensor]) -> float:
    total = 0.0
    for t in tensors:
        if t.is_floating_point():
            total += float(t.detach().float().pow(2).sum())
    return float(total**0.5)


class _Tracer:
    """Collects `TraceEvent`s from one forward pass of a real model."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.hidden_size = int(getattr(model.config, "hidden_size", 0))
        self.lm_head_module = model.get_output_embeddings()
        self.events: list[TraceEvent] = []
        self.stack: list[str] = []
        self.keepalive: list[torch.Tensor] = []  # pin outputs so id()s stay unique
        self._counters: dict[tuple[str, str], int] = {}
        self._activation_paths: set[str] = set()
        self._hooks: list[Any] = []
        self._rope_patches: list[tuple[Any, Any]] = []
        self._prev_attn_impl: str | None = None

    # -- event recording ---------------------------------------------------------

    def _record(
        self,
        *,
        node_id: str,
        kind: str,
        op: str,
        label: str,
        owner: str,
        inputs: list[torch.Tensor],
        outputs: list[torch.Tensor],
    ) -> None:
        self.keepalive.extend(outputs)
        self.events.append(
            TraceEvent(
                order=len(self.events),
                node_id=node_id,
                kind=kind,
                op=op,
                label=label,
                owner=owner,
                out_norm=_norm_of(outputs),
                out_shapes=[tuple(t.shape) for t in outputs],
                input_tids=[id(t) for t in inputs],
                input_storage_ptrs=[t.untyped_storage().data_ptr() for t in inputs],
                output_tids=[id(t) for t in outputs],
                output_storage_ptrs=[t.untyped_storage().data_ptr() for t in outputs],
            )
        )

    def _functional(self, kind: str, label: str, inputs: list, outputs: list) -> None:
        owner = self.stack[-1] if self.stack else ""
        n = self._counters[(owner, kind)] = self._counters.get((owner, kind), 0) + 1
        suffix = kind if n == 1 else f"{kind}_{n}"
        if kind == "residual_add":  # always numbered: two per decoder layer
            suffix = f"{kind}_{n}"
        self._record(
            node_id=f"{owner}.{suffix}" if owner else suffix,
            kind=kind,
            op="functional",
            label=label if n == 1 else f"{label} #{n}",
            owner=owner,
            inputs=inputs,
            outputs=outputs,
        )

    # -- module hooks ------------------------------------------------------------

    def _install_hooks(self) -> None:
        for name, module in self.model.named_modules():
            is_leaf = next(module.children(), None) is None
            if is_leaf and classify_module(module, self.lm_head_module) == "activation":
                self._activation_paths.add(name)

            def pre(mod: nn.Module, args: Any, kwargs: Any, _name: str = name) -> None:
                self.stack.append(_name)

            def post(
                mod: nn.Module,
                args: Any,
                kwargs: Any,
                output: Any,
                _name: str = name,
                _leaf: bool = is_leaf,
            ) -> None:
                if _leaf:
                    kind = classify_module(mod, self.lm_head_module)
                    label = _LEAF_LABELS.get(_name.rsplit(".", 1)[-1], _name.rsplit(".", 1)[-1])
                    self._record(
                        node_id=_name,
                        kind=kind,
                        op="module",
                        label=label,
                        owner=_name,
                        inputs=_flatten_tensors(args) + _flatten_tensors(kwargs),
                        outputs=_flatten_tensors(output),
                    )
                if self.stack and self.stack[-1] == _name:
                    self.stack.pop()

            self._hooks.append(module.register_forward_pre_hook(pre, with_kwargs=True))
            self._hooks.append(module.register_forward_hook(post, with_kwargs=True))

    # -- rope monkeypatch --------------------------------------------------------

    def _patch_rope(self) -> None:
        modeling_modules = {
            sys.modules[type(m).__module__]
            for m in self.model.modules()
            if type(m).__module__ in sys.modules
        }
        for mod in modeling_modules:
            fn = getattr(mod, "apply_rotary_pos_emb", None)
            if fn is None or not callable(fn):
                continue

            def make_wrapper(orig: Any) -> Any:
                def wrapper(*args: Any, **kwargs: Any) -> Any:
                    out = orig(*args, **kwargs)
                    self._functional(
                        "rope",
                        "rotary position embedding",
                        _flatten_tensors(args) + _flatten_tensors(kwargs),
                        _flatten_tensors(out),
                    )
                    return out

                return wrapper

            mod.apply_rotary_pos_emb = make_wrapper(fn)
            self._rope_patches.append((mod, fn))

    # -- functional-op capture ----------------------------------------------------

    def _make_mode(self) -> TorchFunctionMode:
        tracer = self

        class _FunctionalCapture(TorchFunctionMode):
            def __torch_function__(
                self, func: Any, types: Any, args: tuple = (), kwargs: dict | None = None
            ) -> Any:
                kwargs = kwargs or {}
                out = func(*args, **kwargs)
                name = getattr(func, "__name__", "")
                if name in _SOFTMAX_NAMES:
                    owner = tracer.stack[-1] if tracer.stack else ""
                    if "attn" in owner or "attention" in owner:
                        tracer._functional(
                            "attention_softmax",
                            "attention softmax",
                            [a for a in args if isinstance(a, torch.Tensor)],
                            _flatten_tensors(out),
                        )
                elif name in _ADD_NAMES:
                    ts = [a for a in args if isinstance(a, torch.Tensor)]
                    owner = tracer.stack[-1] if tracer.stack else ""
                    if (
                        len(ts) == 2
                        and ts[0].shape == ts[1].shape
                        and ts[0].dim() == 3
                        and ts[0].shape[-1] == tracer.hidden_size
                        and _LAYER_RE.search(owner + ".")
                        and owner.rsplit(".", 2)[-2:-1] == ["layers"]
                    ):
                        tracer._functional(
                            "residual_add", "residual add", ts, _flatten_tensors(out)
                        )
                elif name in _FUNC_ACT_NAMES:
                    owner = tracer.stack[-1] if tracer.stack else ""
                    if owner not in tracer._activation_paths:
                        # functional activation not represented by a module (rare;
                        # most HF models use ACT2FN module instances)
                        tracer._functional(
                            f"activation_{name}",
                            f"{name} activation",
                            [a for a in args if isinstance(a, torch.Tensor)],
                            _flatten_tensors(out),
                        )
                return out

        return _FunctionalCapture()

    # -- lifecycle ----------------------------------------------------------------

    def __enter__(self) -> "_Tracer":
        # Force eager attention during the traced pass: it exposes the softmax to the
        # function mode and (unlike v5 SDPA) supports output_attentions=True, so the
        # graph and the trace see the same per-layer attention_softmax node.
        # Sentinel (not None): a config whose value is genuinely None must be restored
        # to None, not left permanently "eager" on the shared cached model.
        self._had_attn_impl = hasattr(self.model.config, "_attn_implementation")
        self._prev_attn_impl = getattr(self.model.config, "_attn_implementation", None)
        self.model.config._attn_implementation = "eager"
        self._install_hooks()
        self._patch_rope()
        return self

    def __exit__(self, *exc: Any) -> None:
        for handle in self._hooks:
            handle.remove()
        for mod, fn in self._rope_patches:
            mod.apply_rotary_pos_emb = fn
        if self._had_attn_impl:
            self.model.config._attn_implementation = self._prev_attn_impl
        else:
            try:
                delattr(self.model.config, "_attn_implementation")
            except AttributeError:
                pass
        self.stack.clear()


def run_traced_forward(
    model: nn.Module,
    input_ids: torch.Tensor,
    *,
    output_attentions: bool = False,
    output_hidden_states: bool = False,
) -> tuple[Any, list[TraceEvent]]:
    """Run one real forward pass and return ``(model_outputs, ordered trace events)``.

    Functional-op node ids are deterministic (per-owner occurrence counters over a
    deterministic execution order), so repeated traces of the same model yield the
    same node-id set — the invariant `graph.py` and `trace.py` both rely on.
    """
    with _TRACE_LOCK:
        tracer = _Tracer(model)
        with tracer, torch.no_grad(), tracer._make_mode():
            outputs = model(
                input_ids=input_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                use_cache=False,
            )
        return outputs, tracer.events
