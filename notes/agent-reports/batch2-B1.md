# Batch-2 B1 report — Geometry Lab HTTP routes (`/api/geo/*`)

Date: 2026-07-24 · Branch: 002-interactive-model-explorer · Status: COMPLETE, all tests green.

## Endpoints implemented (`code/backend/src/llm_geometry/api/routes_geo.py`)
- `GET /api/geo/spec` — frozen model constants + special tokens + checkpoint block
  (ready ⇒ checkpoint_id + final_loss/coverage_uniformity/field_directional_entropy;
  training ⇒ job_id; missing ⇒ all null).
- `POST /api/geo/train` — idempotent single-flight: cache hit ⇒ 200
  `{checkpoint_id, status:"complete", ready:true}`; else 202 `{job_id, ready:false}`.
- `GET /api/geo/tokenize` — `{tokens:[{id,text,unk}], n_unk, truncated}`.
- `GET /api/geo/trace` — full contract shape via `forward_trace` (tokens, embeddings,
  4 layer blocks, probs×1003, logits_topk×10, next_token); empty-after-tokenization ⇒ 400.
- `GET /api/geo/vector_field` — next_next (T=0 argmax / top_m weighted) + force
  (antisymmetrize ⇒ tangent_exact, sequence_forces); force+layer="full" ⇒ 400.
- `GET|POST /api/geo/weights` — presets/values via `build_weight_set`
  (InvalidWeightEditError ⇒ 422 envelope); embedding ignores `layer`.
- `POST /api/geo/finetune` — JSON or multipart (.txt/.md `file`); exactly one of
  {text,file,hf_dataset} else 400; unusable HF id resolved *synchronously* ⇒ 422;
  content-hash cache hit ⇒ 200 ready; else 202 job.
- Contract-wide `_jsonable`: one helper; every ndarray ⇒ nested lists, every float
  rounded to 6 significant digits.

## Orchestration design (`geo/jobs.py`, new)
- Train/finetune jobs run as daemon threads on the EXISTING `jobs.registry`
  (single-flight keyed by the canonical/finetune cache key); `precompute.py` untouched.
- `request_train`: cache hit ⇒ `train_canonical` fetch (<100 ms); miss ⇒ job with
  phase "train"; done result `{checkpoint_id}`. `training_job_id()` lets /spec report
  "training" without creating a job. Train failures ⇒ `TrainingFailedError` error events.
- `request_finetune`: source pre-resolved to text in the route; base_token via content
  hash; job phase "finetune"; done result `{weights_token, loss_before, loss_after}`.
- Weight minting writes a `geo-weights-sources-<token>` sidecar (per-matrix source map,
  inherited along edit chains) so GET /weights reports exact
  `preset:<name>/edited/learned`; sidecar-less tokens (finetuned/evicted) fall back to
  the artifact's stored set-level source, mapped onto the contract's closed enum.

## Deviations / resolved ambiguities
1. **Extended `jobs/registry.py` + `api/progress.py` (additive)** — the frozen contract
   requires phase-labeled SSE progress and result-bearing `done` events
   (checkpoint_id / weights_token), which the machinery could not emit. Added optional
   `Job.phase`/`Job.result` (+ snapshot fields, `get_or_create(phase=)`,
   `finish(result=)`); done data = `{cache_key, **result}`. Backward-compatible; the
   001 SSE test still passes. These files were outside my nominal list but not on the
   explicit do-not-touch list; flagged here for red-team review.
2. `TrainingFailedError` emitted as a job-error-event *type string* only (contract:
   "500 via job error events") — no new class in `errors.py`.
3. Trace/tokenize share right-side truncation (first 50 tokens) so both endpoints agree
   on the same T≤50 token list.
4. New dependency `python-multipart==0.0.32` (contract's multipart upload path):
   added to `pyproject.toml` + `requirements.lock`.
5. GET /weights source for finetuned tokens reports "edited" (closed contract enum;
   fine-tuning edits every matrix).

## Tests (`tests/contract/test_api_geo.py`, 16 tests, all real — no mocks)
Full missing→training→ready cycle (real ~15 s retrain; single-flight job reuse; SSE
phase "train" + epoch messages + done checkpoint_id; gate metrics; 200 cache hit),
tokenize unk/truncation, trace shape/causality/rounding + empty-prompt 400,
next_next T=0 (one arrow per vocab point, weight 1.0, valid origins), antisymmetrized
force (tangency ⟨z,f⟩≈0, sequence_forces), force+full 400, W_V=identity token changes
the field (force layer-0 vec≡z), weights preset/values round-trips + inheritance +
422 envelopes, finetune JSON 202→SSE done (new token, loss drop, canonical unchanged,
200 re-hit), multipart .md upload, source-count/extension/steps 400s, bad HF id 422.

## Test output tail
```
tests/contract/test_api_geo.py alone:      16 passed, 1 warning in 25.73s
full geo suite (contract+unit+integration): 70 passed, 1 warning in 77.00s (0:01:17)
tests/contract/test_sse.py (001 machinery): 1 passed (no SSE regression)
```
ruff + black clean on all touched files
(routes_geo.py, geo/jobs.py, jobs/registry.py, api/progress.py, test_api_geo.py).
