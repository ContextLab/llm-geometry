# Fix: gpt2 arch-graph tracing (500 ComputeError)

Date: 2026-07-24 · Branch: 002-interactive-model-explorer

## Root cause (verified empirically on real gpt2, transformers 5.12.0)

A diagnostic trace of gpt2 before the fix showed 12/12 layers captured
`attention_softmax` but **0 `residual_add` events**. The residual-add rule in
`code/backend/src/llm_geometry/arch/tracing.py` required the owning module path's
second-to-last component to be literally `"layers"`
(`owner.rsplit(".", 2)[-2:-1] == ["layers"]`). GPT-2 blocks are named
`transformer.h.<k>`, so both per-block residual adds were dropped and the (correct,
unchanged) per-layer completeness gate in `arch/graph.py` raised ComputeError.
Softmax attribution and Conv1D→linear classification already worked for gpt2.

## Exact rule changes (tracing.py only; graph.py untouched)

1. Residual adds: replaced the two-part `"layers"`-only scope check with one
   end-anchored block-scope regex
   `_BLOCK_SCOPE_RE = r"(?:^|\.)(?:layers|h|blocks)\.\d+$"` — matches
   `model.layers.<k>` (Llama/Qwen) and `transformer.h.<k>` (GPT-2), never
   sub-modules. Shape/dim/hidden-size conditions unchanged.
2. Attention softmax: now attributed via `_is_attention_scope(owner, owner_cls)` —
   the owning module's **class name** (`GPT2Attention`, `LlamaAttention`, ...)
   with the old path-substring spelling kept as fallback. Tracer records a
   path→class map at hook install.
3. Added GPT-2 leaf labels (`wte`, `wpe`, `c_attn`, `c_proj`, `c_fc`, `ln_1`,
   `ln_2`, `ln_f`, `act`) to `_LEAF_LABELS` — cosmetic, gpt2-only.
4. No rope changes needed: GPT-2's modeling module has no `apply_rotary_pos_emb`,
   so no rope nodes are fabricated; the graph.py gate only requires
   `attention_softmax >= 1` and `residual_add >= 2` per layer (verified, stays as-is).

## Id-stability proof (SmolLM2 + Qwen)

Dumped `(model_id, node id, kind, op, label)` for fresh uncached
`build_graph` of `HuggingFaceTB/SmolLM2-135M-Instruct` and
`Qwen/Qwen2.5-0.5B-Instruct` before and after the change:
`diff ids_before.txt ids_after.txt` → **identical, all 764 rows** (same ids, kinds,
ops, labels, same order).

## gpt2 end-to-end (real model, no mocks)

- `build_graph("gpt2")`: passes both gates; 12 attention_softmax, 24 residual_add,
  0 rope nodes; 4x12 Conv1D linear nodes; every named param owned;
  `lm_head.weight.tied_to == "transformer.wte.weight"`.
- `trace_forward("gpt2", ...)`: `node_activations` covers exactly the graph's
  149 node ids; `chat_template_used == False`.

## New tests (appended)

- `tests/integration/test_arch_graph.py`: `test_gpt2_functional_steps_complete_per_layer`,
  `test_gpt2_has_no_rope_nodes`, `test_gpt2_conv1d_projections_are_linear_nodes`,
  `test_gpt2_every_parameter_owned_and_wte_lm_head_tied`.
- `tests/integration/test_arch_trace.py`: `test_gpt2_trace_covers_graph_nodes`.

ruff + black clean on all touched files.

## Test output tail

`python -m pytest tests/integration/test_arch_graph.py tests/integration/test_arch_trace.py
tests/integration/test_arch_weights.py tests/integration/test_arch_generate.py
tests/contract/test_api_arch.py -q`:

```
...............................................................          [100%]
63 passed, 1 warning in 11.59s
```

(Only warning: pre-existing StarletteDeprecationWarning from fastapi/testclient.)

API check: `GET /api/arch/graph?model_id=gpt2` via TestClient → **200**,
149 nodes / 172 edges (previously 500 ComputeError).
