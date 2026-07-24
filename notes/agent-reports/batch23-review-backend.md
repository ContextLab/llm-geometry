# Batch-2 route-layer red-team review (backend) — issue #1

Scope: api/routes_geo.py, api/routes_arch.py, geo/jobs.py, additive jobs/registry.py +
api/progress.py (diff 92e5854..93478d5). Stance: REFUTE. Authority: frozen contracts/api.md.
Method: code read + live curls against the running :8000 stack. EXCLUDED arch/tracing.py, arch/graph.py.

## Findings

1. [MAJOR] Framework validation errors bypass the frozen error envelope.
   app.py (api/app.py:33-44) registers handlers only for LLMGeometryError and bare
   Exception; FastAPI's RequestValidationError is handled by Starlette BEFORE reaching
   them. So malformed/missing typed params return `{"detail":[...]}` (HTTP 422), not the
   contract envelope `{"error":{type,message,detail}}`. Confirmed LIVE on both routers:
   GET /api/geo/weights (no `matrix`) -> 422 detail; /api/geo/vector_field?temperature=abc
   -> 422 detail; /api/arch/weights (no `param`) -> 422 detail; ...&r1=abc -> 422 detail.
   Contract header mandates "same error envelope" for every failure path. Fix: add
   `@app.exception_handler(RequestValidationError)` returning the envelope (400/422
   InvalidParamError). One handler covers all routes.

2. [MINOR] Provenance mislabel on compound finetune->edit chains. geo/jobs.py:198-213 +
   mint_weight_set:169-171. A fine-tuned token has no sources sidecar, so minting edits
   on top of it inherits `{}` (_source_map:189-195); a matrix that was fine-tuned but not
   re-edited is then absent from the new sidecar and matrix_source() returns "learned"
   (jobs.py:205 `.get(key,"learned")`) — wrong (it is edited). Reachable because POST
   /weights `base` accepts any weights_token. Fix: seed inherited sidecar from the base
   artifact's set-level source when no sidecar exists.

3. [MINOR] Result-field encoding drift between delivery paths. loss_before/loss_after are
   6-sig-rounded on the 200 cache-hit HTTP path (route _jsonable, routes_geo.py:330) but
   emitted FULL precision on the 202->SSE done event (progress.py:46 json.dumps of
   job.result). Verified live: done data `"loss_before":5.279822826385498`. Same logical
   fields, two encodings. Fix: round in finish()/progress or document scalars as exempt.

4. [MINOR] Background-train global-state race. geo/jobs.py:74/128 spawn daemon threads
   running deterministic_torch (train.py:63-80), which flips process-global
   use_deterministic_algorithms(True) and reseeds the global RNG for the DURATION of
   training. arch.generate/trace serialize on _TRACE_LOCK but do NOT share it with geo
   training, so a concurrent arch generate can hit a nondeterministic-op error or
   RNG-perturbed sampling. Real (multi-tab demo), unlikely CPU-only. Fix: run geo training
   under a shared global-torch lock, or in a subprocess.

5. [MINOR] _jobs_by_key (geo/jobs.py:37) never pruned on finish/fail — grows one stale
   entry per distinct seed. Guarded by a status check (training_job_id:51-53) so not a
   functional leak, just unbounded. Fix: drop the key in the job's terminal callback.

6. [NIT] Duplicated encoders _jsonable (routes_geo.py:55) vs _sig6 (routes_arch.py:29).
   Verified SEMANTICALLY IDENTICAL (both float(f"{x:.6g}"), bool-before-int, ndarray/
   dict/list recursion) — the hypothesized drift is REFUTED. But duplicated, and NEITHER
   guards NaN/Inf -> json.dumps would emit invalid JSON. Fix: share one helper; coerce
   non-finite.

7. [NIT] SSE done event carries extra `cache_key` beyond the contract's listed fields
   (progress.py:43). Backward-compatible; contract says done data "includes" those fields.

## Refuted attacks (held up under live testing)
- Single-flight train/finetune: VERIFIED — two concurrent POSTs returned identical
  job_id (2253305d...). registry.get_or_create is correctly locked.
- Job-failure leak: REFUTED — fail()/finish() (registry.py:99-111) both delete from
  _active_by_key; _run_* wrap in try/except -> registry.fail. No active-job leak.
- TrainingFailedError path: correct — _run_train.fail({"type":"TrainingFailedError",...})
  surfaces verbatim as the SSE error event.
- Cache corruption on daemon kill: REFUTED — store.put is atomic (tmp + os.replace,
  store.py:118-128).
- phase/result thread-safety: REFUTED — writer sets result BEFORE version bump under
  lock; unlocked reader gates on version; GIL-atomic ref assignment. Safe.
- Size gate: VERIFIED — Qwen2.5-7B -> 422 ModelTooLargeError from hub metadata, no
  download. Business-logic envelopes VERIFIED on all paths (404 unknown param, 400
  max_cells=0 / r0>r1 / empty prompt / force+full / top_m=0 / neg temperature).
- Contract shapes VERIFIED: /geo/spec, /tokenize, /weights, /vector_field, /arch/weights.

## Verdict: REQUEST CHANGES
Route layer is a faithful, well-tested implementation of the frozen contract; core
behavior, single-flight, and business-logic envelopes all conform live. The one MAJOR
(finding 1) is a genuine conformance gap — malformed requests escape the frozen envelope
on both routers — fixable with a single app-level handler. Findings 2-5 are real but
edge-case MINORs.
