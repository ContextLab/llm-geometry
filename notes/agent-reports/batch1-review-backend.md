# Batch-1 backend adversarial review (issue #1, feature 002)

Reviewer stance: refute correctness / contract-conformance / test-honesty.
Scope: commits 995894d, 00d6be2, 92e5854 — geo/, arch/, models/loader.py, tests.
Date: 2026-07-24. Verdict: **ACCEPT-WITH-FIXES** (no BLOCKER; 2 MAJOR; 5 MINOR; 3 NIT).

## What I verified as CORRECT (attempted refutation, failed)

- **Force aggregate math** (`geo/fields.force_field`): recomputed
  `Σ_{j≤i} softmax(⟨Kz_j,Qz_i⟩)·Vz_j` independently from the trace; max abs err
  **2.98e-8** vs the returned `sequence_forces`. Uses the real (never antisymmetrized)
  W_V for aggregates, unscaled scores matching the model — exact.
- **normal_residual** = |⟨f_i, ẑ_i⟩| with ẑ_i the layer input hidden dir: max err 5.96e-8.
- **Antisymmetric per-point field tangency**: max |⟨Vz,z⟩| = **2.55e-8** (float32 machine
  precision) with `(W_V−W_Vᵀ)/2`; plain field radial = 0.71 (check is meaningful).
- **next_next semantics** ("if next token were v, where does the following token point"):
  for v∈{0,100,500,1002} independently appended v, argmaxed the layer-`full` logit-lens,
  confirmed arrow tip == emb[argmax] to 1e-5, weight 1.0. Origin=points[v]. Correct.
- **Logit-lens layer**: hidden_out[layer_idx] readout; `full`→last layer == `forward()`. Coherent.
- **Training determinism is REAL, not a cache artifact**: two independent non-cached
  `train_geo_model(epochs=2, seed=0)` runs → identical `weights_token` and identical loss.
- **No-mock policy honored**: every test uses the real corpus, real training, real HF
  models (SmolLM2-135M, Qwen2.5-0.5B), real TinyStories streaming. tracing.py's rope
  monkeypatch/hooks observe a REAL forward pass (not fabricated tensors) — within policy.
- **Contract conformance** at the compute layer: field names/shapes/status codes match
  api.md (force `layer="full"`→400; oversized model→422 ModelTooLargeError before download,
  source `safetensors_metadata`, <60s, no >50MB blobs; weights exact≤max_cells else strided
  mean with true-window stats; graph completeness = every named param in exactly one node,
  tied lm_head aliased). Fast unit suites re-run green here (36 passed).
- 92e5854 out-of-scope diffs (precompute.py/routes.py/vector_field.py) verified to be
  **black line-wrapping only** — no behavior change hidden under "conformance".

## MAJOR

1. **`deterministic_torch` leaks the process-global torch RNG** — `geo/train.py:63-72`.
   Calls `torch.manual_seed(seed)` on entry and in `finally` restores ONLY
   `use_deterministic_algorithms`, NOT the RNG state. Confirmed: global RNG state differs
   across the context; `torch.rand` under the same outer seed changes. Docstring
   ("restores the prior mode") is false. Cross-contamination is real: `arch/generate.py:76`
   samples with `torch.multinomial(probs, 1, generator=None)` → draws from the GLOBAL RNG,
   so an *unseeded* generation's output depends on whether a geo train/finetune ran earlier
   in the process; also makes test outcomes order-dependent. Fix: snapshot
   `torch.get_rng_state()` (and cuda) before, restore in `finally`; or drive training off a
   local `torch.Generator` instead of `torch.manual_seed`.

2. **Tracing lock doesn't cover non-trace forwards on the shared cached model** —
   `arch/tracing.py:60,226-259,375`. `_install_hooks` registers forward pre/post hooks and a
   `TorchFunctionMode` on the SHARED `LoadedModel.model` (cached in `models/loader._loaded`)
   under `_TRACE_LOCK`. But `arch/generate.py` (and loader's own verify-forward) call
   `lm.model(...)` WITHOUT that lock. A concurrent forward during a trace fires the tracer's
   hooks and mutates the shared `tracer.stack`/`events` from another thread → corrupted
   trace or spurious nodes. Latent only because the route/job layer doesn't exist yet, but
   the contract targets multi-worker serving. Fix: a per-model reentrant lock guarding ALL
   forwards, or acquire `_TRACE_LOCK` in generate/trace-adjacent paths.

## MINOR

3. **attn-impl restore guard conflates "absent" with None** — `arch/tracing.py:346-358`.
   `_prev = getattr(config,"_attn_implementation",None)`; `__exit__` restores only
   `if _prev is not None`. Raw AutoConfig has `_attn_implementation=None`; a model whose
   config value is genuinely None would be left permanently `"eager"` on the shared cached
   model. Does NOT manifest for SmolLM2/Qwen (loaded config = 'sdpa', restores OK — verified),
   but latent. Use a sentinel and always restore.

4. **HF load errors masked as user error** — `geo/finetune.py:54-59,81-82`. Every exception
   from `load_dataset`/streaming → `UnsupportedModelError` (422). Transient hub/network
   failures become "unusable id". Distinguish infra from bad-id (Constitution I: don't
   silently mislabel).

5. **Size-gate silently swallows metadata failure** — `arch/gate.py:65-66`.
   `except Exception: total=None` drops the safetensors error and falls back to a coarse
   config estimate only asserted "within 2x". A model lacking safetensors metadata whose
   estimate undercounts could slip past FR-107. Log the swallowed error; add a safety margin.

6. **Graph completeness checks params only, not functional nodes** — `arch/graph.py:188-198`.
   Coverage asserts every named PARAMETER is owned, but a missed softmax/rope/residual
   functional node (name-rule based, `tracing.py`) yields a degraded-but-plausible graph that
   still passes. Only 2 models are gated/tested; arbitrary HF models are unverified.

## NIT

7. Compute layer returns `points` as np.ndarray and full-precision lists; contract requires
   nested lists rounded to 6 sig figs — route must convert/round (fields.py:92-100,153-161).
8. `next_next` at temperature=0 ignores `top_m` (single argmax). Matches contract outcome
   (one-hot) and is documented; harmless divergence from the literal "otherwise" branch.
9. Behavioral thresholds `test_finish_reason_eos_when_reply_ends`,
   `test_short_training_materially_beats_uniform` are model/version-dependent — real but
   flake-prone across transformers upgrades.

## Verdict
**ACCEPT-WITH-FIXES.** Core science (force/next_next/tangency math), determinism, no-mock
discipline, and compute-layer contract conformance are sound and independently verified.
Fix the two global-state issues (#1 RNG restore, #2 lock coverage) before the route/serving
layer lands; the rest are hardening.
