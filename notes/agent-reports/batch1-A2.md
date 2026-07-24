# Batch-1 A2 report — Architecture Explorer backend (`llm_geometry.arch`)

Status: COMPLETE. 43/43 tests pass; ruff + black clean.

## Files created/edited (all under /Users/jmanning/llm-geometry/code/backend/)

- `src/llm_geometry/arch/tracing.py` — shared traced-forward machinery (graph + trace use it)
- `src/llm_geometry/arch/graph.py` — traced architecture graph, cached via cache/store.py
- `src/llm_geometry/arch/weights.py` — exact/strided-mean weight windows + `strided_mean_2d`
- `src/llm_geometry/arch/trace.py` — per-layer attention/hidden_norm/pca3 + node_activations
- `src/llm_geometry/arch/generate.py` — real autoregressive generation (greedy/seeded sampling)
- `src/llm_geometry/arch/gate.py` — pre-download size gate (`check_model_size`)
- `src/llm_geometry/arch/__init__.py` — public API exports
- `src/llm_geometry/config.py` — added ARCH_MAX_PARAMS=1.5e9, ARCH_GRAPH_SCHEMA_VERSION,
  ARCH_DEFAULT_MAX_CONTEXT=64, ARCH_ATTENTION_MAX_SIDE=64, ARCH_WEIGHTS_MAX_CELLS=4096,
  ARCH_MAX_NEW_TOKENS=128
- `src/llm_geometry/errors.py` — added `ModelTooLargeError` (422); coexists cleanly with
  A1's concurrently-added `InvalidWeightEditError`
- Tests: `tests/integration/test_arch_{graph,weights,trace,generate}.py`,
  `tests/unit/test_arch_gate.py`

## Tracing approach (hooks + TorchFunctionMode + targeted rope patch)

torch.fx/HFTracer rejected as brittle under transformers **v5.12** (installed). One real
forward pass is observed three ways:

1. **Forward pre/post hooks on every module** maintain a "currently executing module"
   stack; each *leaf* module (embedding, linears, norms, activations, rotary table,
   lm_head) records one node event with output norms/shapes/tensor-ids.
2. **`TorchFunctionMode`** intercepts functional ops and attributes them to the
   innermost executing module: `F.softmax`/`F.scaled_dot_product_attention` inside an
   attention module → `…self_attn.attention_softmax`; 3-D adds of hidden-size tensors
   at decoder-layer scope → `…layers.k.residual_add_{1,2}` (mask/bias adds excluded by
   shape+owner rules); functional activations captured with a dedup rule (Llama/Qwen
   use ACT2FN modules, caught by hooks).
3. **`apply_rotary_pos_emb` wrapped in place** in each modeling module during the pass →
   `…self_attn.rope` functional nodes (v5 note: sdpa can't return attentions, so the
   tracer forces eager attention for traced passes and restores after; a lock
   serializes tracing since hooks/patches are global).

Edges are real dataflow: producer registry over tensor `id()` **and** storage pointers
(recovers flow through `.view/.transpose`), execution-order fallback keeps the chain
connected. Residual skip-connections appear via true tensor identity. Ids are
deterministic (per-owner occurrence counters over deterministic execution order) —
graph and trace share the identical id universe (SC-102 invariant verified).

Tied weights: grouped by `data_ptr` over `named_parameters(remove_duplicate=False)`;
canonical = first registration (embed_tokens); `lm_head.weight` → `tied_to:
model.embed_tokens.weight` (both models tie). Coverage enforced at build time
(`ComputeError` if any param unowned) and asserted in tests.

Gate: `get_safetensors_metadata` parameter_count (hub metadata only, no weights) with
config-based architectural estimate fallback; Qwen2.5-7B rejected in ~0.6 s
(7,615,616,512 params, source `safetensors_metadata`), cache checked for absence of
large blobs.

## Per-model graph stats

| model | nodes | edges | module/functional | kinds |
|-|-|-|-|-|
| SmolLM2-135M-Instruct | 424 | 483 | 304/120 | activation 30, attention_softmax 30, embedding 1, linear 210, lm_head 1, residual_add 60, rmsnorm 61, rope 31 |
| Qwen2.5-0.5B-Instruct | 340 | 387 | 244/96 | activation 24, attention_softmax 24, embedding 1, linear 168, lm_head 1, residual_add 48, rmsnorm 49, rope 25 |

(rope = n_layers functional + 1 module rotary cos/sin table; rmsnorm = 2/layer + final;
meta matches real configs: 134,515,008 / 494,032,768 total params, kv_heads 3 / 2.)

## Pytest output (final run, after black reformat re-verification)

```
...........................................                              [100%]
43 passed in 7.94s
```

(Full sequence rerun after black touched config.py + test_arch_graph.py: ruff "All
checks passed!", black "14 files would be left unchanged", pytest 43 passed.)

## Notes / unresolved

- `attention_downsampled` semantics: strided-mean over rows breaks exact
  row-stochasticity when T > 64 (flagged in payload per contract); tests verify
  row-stochastic pre-downsample on short prompts.
- `out_shape`/edge `tensor_shape` "T" substitution is by dim==seq_len match; a dim
  coincidentally equal to T (e.g. 9-head model traced at T=9) would also read "T" —
  cosmetic only; graph-build prompt is 10 tokens for both models so not hit.
- Generation `prob` for temperature 0 follows the 001 `distributions.py` precedent
  (one-hot ⇒ prob 1.0); topk ids are ranked by logits so they stay informative.
- Untouched (per rules): routes files, geo/, precompute.py, frontend. No git commands.
