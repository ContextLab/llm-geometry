---
description: "Task list for Core Project Machinery implementation"
---

# Tasks: Core Project Machinery

**Input**: Design documents from `/specs/001-core-machinery/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/api.md, quickstart.md

**Tests**: REQUIRED. Constitution v2.0.0 Principle I ("Real verification, never mocks")
and the spec's quickstart mandate real-model tests; every test task below calls real
HuggingFace models (`sshleifer/tiny-gpt2` for speed, `distilgpt2`/`gpt2` for real-output
assertions). No mock-only tests.

**Organization**: Grouped by user story (US1 P1, US2 P2, US3 P3) for independent
implementation and testing.

## Format: `[ID] [P?] [Story] Description with file path`

- **[P]**: Can run in parallel (different files, no incomplete dependencies)
- **[Story]**: US1 / US2 / US3 (Setup, Foundational, Polish have no story label)

## Path Conventions

Web app per plan.md: backend = `code/backend/src/llm_geometry/`, tests =
`code/backend/tests/`; frontend = `code/frontend/src/`, tests = `code/frontend/tests/`;
derived cache = `data/processed/cache/` (git-ignored).

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Dual-stack project initialization

- [X] T001 Create the dual-stack directory tree per plan.md (`code/backend/src/llm_geometry/{models,compute,reduce,cache,jobs,api}/` with `__init__.py`, `code/backend/tests/{unit,integration,contract}/`, `code/frontend/src/{controls,lib/viz,preview,styles}/`, `code/frontend/tests/{unit,e2e}/`)
- [X] T002 [P] Backend deps: create `code/backend/pyproject.toml` (package `llm_geometry`, ruff/black config) and `code/backend/requirements.txt` pinning torch, transformers, huggingface_hub, numpy, scipy, scikit-learn, umap-learn, fastapi, uvicorn, pydantic, orjson, pytest, httpx, sse-starlette
- [X] T003 [P] Frontend scaffold: Svelte + TypeScript + Vite in `code/frontend/` (`package.json`, `vite.config.ts`, `svelte.config.js`, `tsconfig.json`) with deps d3, three, vitest, @playwright/test, eslint, prettier, svelte-check
- [X] T004 [P] Configure repo-root `Dockerfile` for dual stack (Python 3.11 + Node 20, install backend `requirements.txt` `-e .`, build frontend) and add `data/processed/cache/`, `code/frontend/node_modules/`, `code/frontend/dist/`, `**/__pycache__/`, `.venv/` to `.gitignore`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Shared infrastructure every user story depends on

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T005 [P] Config in `code/backend/src/llm_geometry/config.py` (cache dir default `data/processed/cache/`, default model `gpt2`, curated model list, fixed seeds, perf budgets, `SCHEMA_VERSION`)
- [X] T006 [P] Typed errors in `code/backend/src/llm_geometry/errors.py` (`UnsupportedModelError`, `InvalidParamError`, `ComputeError` with `type`/`message`/`detail`)
- [X] T007 Cache keys in `code/backend/src/llm_geometry/cache/keys.py` (deterministic hash over sorted `{schema_version, model_id, revision, artifact_type, inputs, params, seed}`)
- [X] T008 Cache store in `code/backend/src/llm_geometry/cache/store.py` (atomic write+rename, JSON sidecar with spec+checksum+`complete` flag, read→verify(schema+checksum+complete)→miss-on-mismatch, regenerate) — depends on T005, T006, T007
- [X] T009 Jobs registry in `code/backend/src/llm_geometry/jobs/registry.py` (status/progress 0..1/message, single-flight dedup keyed by cache_key, async progress channel for SSE) — depends on T006
- [X] T010 [P] Model registry in `code/backend/src/llm_geometry/models/registry.py` (curated menu + arbitrary HF id normalization) — depends on T005
- [X] T011 Model loader + capability detection in `code/backend/src/llm_geometry/models/loader.py` (load model/tokenizer, pin `revision`, detect token-probs + hidden-states, raise `UnsupportedModelError` with reason, no fallback; **wrap load/download failures — unresolvable id, network/partial download, OOM/insufficient-hardware — into a clear typed error rather than a raw traceback, covering those spec edge cases**) — depends on T006, T010
- [X] T012 FastAPI app skeleton in `code/backend/src/llm_geometry/api/app.py` (ASGI app, global error-envelope handler, dev CORS, static-mount hook, `GET /api/health`) — depends on T005, T006

**Checkpoint**: Foundation ready — user stories can begin

---

## Phase 3: User Story 1 - Model → cached geometric data, with progress (Priority: P1) 🎯 MVP

**Goal**: Load an open-weights model and compute + cache its real next-token
distributions and per-layer embeddings, once, with visible progress; cache hits are
instant and identical; unsupported models fail loudly.

**Independent Test**: Request data for an uncomputed model → progress animation →
real cached artifact; second request instant and identical; a bad model id → clear
error, no artifact, no fallback.

### Tests for User Story 1 (write first; must FAIL before implementation)

- [X] T013 [P] [US1] Integration test in `code/backend/tests/integration/test_distributions.py`: real `distilgpt2` next-token probs sum to 1.0 and top token equals a direct forward-pass argmax; `temperature=0` ⇒ argmax mass; negative temperature ⇒ `InvalidParamError`
- [X] T014 [P] [US1] Integration test in `code/backend/tests/integration/test_embeddings.py`: real `sshleifer/tiny-gpt2` static + contextual per-layer embeddings have shape `[N, hidden_size]`; layer out of range raises `InvalidParamError`
- [X] T015 [P] [US1] Integration test in `code/backend/tests/integration/test_cache.py`: determinism (compute→delete→recompute→array-identical, SC-002); a partially-written artifact (no `complete` flag) is treated as a miss; **a checksum-mismatch and a schema-version-mismatch artifact are also treated as misses and recomputed (FR-007)**; two concurrent identical precompute requests run the work once (single-flight, FR-008)
- [X] T016 [P] [US1] Integration test in `code/backend/tests/integration/test_capability.py`: a missing/non-open-weights id raises `UnsupportedModelError` and creates no cache artifact and selects no fallback (SC-007); **a model that fails to load (unresolvable/download failure) surfaces a clear error and no fallback (covers the offline/partial-download + insufficient-hardware edge cases at the error-path level)**
- [X] T017 [P] [US1] Contract test in `code/backend/tests/contract/test_api_us1.py`: `/api/models`, `POST /api/models/resolve`, `POST /api/precompute` (200 hit / 202 job), `GET /api/jobs/{id}`, `GET /api/distribution`, `GET /api/embeddings` return the shapes + status/error codes in contracts/api.md

### Implementation for User Story 1

- [X] T018 [US1] Next-token distributions in `code/backend/src/llm_geometry/compute/distributions.py` (logits→/temperature→softmax; `T=0` argmax; top_k+tail_mass option; validate `temperature>=0`) — depends on T011
- [X] T019 [US1] Per-layer embeddings in `code/backend/src/llm_geometry/compute/embeddings.py` (static input-embedding matrix + batched contextual hidden states via `output_hidden_states`; configurable **embedding reference set** — the tokens embeddings are computed for, distinct from the grid "reference points" in T027) — depends on T011
- [X] T020 [US1] Precompute orchestrator in `code/backend/src/llm_geometry/precompute.py` (map artifact_type→compute fn, run under a job with progress callbacks ≥1/s, write via cache store, return cache hit or job) — depends on T008, T009, T018, T019
- [X] T021 [US1] API routes in `code/backend/src/llm_geometry/api/routes.py` for `/api/models`, `/api/models/resolve`, `/api/precompute`, `/api/jobs/{id}`, `/api/distribution`, `/api/embeddings` — depends on T012, T020
- [X] T022 [US1] SSE progress stream in `code/backend/src/llm_geometry/api/progress.py` (`GET /api/jobs/{id}/events`, ≥1/s, terminal done/error events) wired into routes — depends on T009, T021

**Checkpoint**: US1 fully functional and independently testable (the MVP)

---

## Phase 4: User Story 2 - Embeddings → explorable 2D/3D coordinates (Priority: P2)

**Goal**: Reduce cached embeddings to a 2D set (PCA/UMAP) with an n×n nearest-token
reference grid and a 3D spherical set (points on the unit sphere); cached + reproducible.

**Independent Test**: For a model with cached embeddings, request reductions → 2D
coords + resolvable grid reference points and 3D on-sphere coords; identical on a
second run with the same params.

### Tests for User Story 2 (write first; must FAIL before implementation)

- [X] T023 [P] [US2] Integration test in `code/backend/tests/integration/test_reduction_2d.py`: PCA 2D from real `tiny-gpt2` embeddings + 25×25 grid whose `reference_token_ids` are the nearest reduced points; reproducible with fixed seed (FR-013)
- [X] T024 [P] [US2] Integration test in `code/backend/tests/integration/test_reduction_3d.py`: 3D spherical reduction where every coord has unit norm (‖·‖≈1) and is reproducible with fixed seed
- [X] T025 [P] [US2] Contract test in `code/backend/tests/contract/test_api_us2.py`: `GET /api/reduction/2d?with_grid=true` and `GET /api/reduction/3d` return the shapes in contracts/api.md

### Implementation for User Story 2

- [X] T026 [P] [US2] 2D reduction in `code/backend/src/llm_geometry/reduce/twod.py` (PCA default, UMAP option, fixed seed) — depends on T019
- [X] T027 [P] [US2] Reference grid in `code/backend/src/llm_geometry/reduce/grid.py` (n×n vertices over 2D bounds + nearest-token per vertex) — depends on T026
- [X] T028 [P] [US2] 3D spherical reduction in `code/backend/src/llm_geometry/reduce/sphere.py` (MDS/pca3 → L2-normalize to unit sphere, fixed seed) — depends on T019
- [X] T029 [US2] Register reduction artifact types in the precompute orchestrator and add `/api/reduction/2d` + `/api/reduction/3d` routes in `code/backend/src/llm_geometry/api/routes.py` — depends on T020, T021, T026, T027, T028

**Checkpoint**: US1 + US2 both independently functional

---

## Phase 5: User Story 3 - Shared web shell, controls, instant/progress UX (Priority: P3)

**Goal**: A Svelte shell with reusable controls (model, prompt prefix, temperature,
layer) wired through a data-access client to the cached API; instant on cache hits,
progress animation when computing; a minimal preview proves the round trip.

**Independent Test**: Load the shell, change each control → correct cached data
fetched + preview updates (instant when cached, progress when not); unsupported model
shows a clear error; screenshots confirm a clean, smooth UI.

### Tests for User Story 3

- [X] T030 [P] [US3] Vitest unit tests in `code/frontend/tests/unit/dataClient.test.ts` (typed API calls, precompute→job→ready flow, progress subscription, error surfacing — against a stub HTTP layer, real client logic). NOTE: the **real** client↔backend path is exercised by the Playwright e2e in T031 against real `gpt2`, so this unit test is not a mock-only substitute (Constitution I)
- [X] T031 [P] [US3] Playwright e2e in `code/frontend/tests/e2e/shell.spec.ts` (against the running backend with real `gpt2`): change model/prefix/temperature/layer; assert cached=instant, uncached=progress animation; unsupported model shows error; save screenshots to `code/frontend/tests/e2e/__screenshots__/`

### Implementation for User Story 3

- [X] T032 [P] [US3] Shared interaction stores in `code/frontend/src/lib/stores.ts` (model_id, prefix_text, temperature, layer; bounds validation)
- [X] T033 [US3] Data-access client in `code/frontend/src/lib/dataClient.ts` (typed calls to contracts/api.md, precompute + job poll, SSE progress subscription, loading/progress/error state) — depends on T032
- [X] T034 [P] [US3] Controls `ModelSelector.svelte`, `PromptPrefix.svelte`, `Temperature.svelte`, `LayerSlider.svelte` in `code/frontend/src/controls/` (bound to stores) — depends on T032
- [X] T035 [P] [US3] Design tokens + transition/animation styles in `code/frontend/src/styles/` and a `Progress.svelte` indicator component in `code/frontend/src/lib/`
- [X] T036 [US3] Minimal preview `code/frontend/src/preview/Preview.svelte` rendering a real next-token distribution and a 2D reduced scatter (minimal D3) wired to dataClient — depends on T033, T035
- [X] T037 [US3] App shell `code/frontend/src/App.svelte` + `code/frontend/src/main.ts` composing controls + preview + progress with layout/theming — depends on T034, T036
- [X] T038 [US3] Production static serving (FastAPI serves built bundle) in `code/backend/src/llm_geometry/api/app.py` and dev `/api` proxy in `code/frontend/vite.config.ts` — depends on T012, T037

**Checkpoint**: All three user stories independently functional end-to-end

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Docs, reproducibility, and performance verification (Constitution IV, V, III)

- [X] T039 [P] Unit tests in `code/backend/tests/unit/` for pure functions: `test_keys.py` (cache-key determinism/sensitivity) and `test_grid.py` (grid vertex count + nearest-token math)
- [X] T040 [P] Update `README.md` with dual-stack setup/run instructions (replace template placeholders relevant to running the app)
- [X] T041 [P] Update `CLAUDE.md` Commands + Repository layout for the new stack; verify `Dockerfile` / `requirements.txt` / `package.json` match actual imports (Constitution IV/V)
- [X] T042 Performance verification: measure backend cache-hit latency (<100 ms, SC-001), first-`gpt2` precompute time (≤180 s, SC-003), **and frame-rate smoothness of shell transitions toward the 60 fps target (SC-004) via a Playwright trace / `requestAnimationFrame` sampling in `code/frontend/tests/e2e/`**; record all results in `specs/001-core-machinery/quickstart.md` "Expected outcomes"
- [X] T043 Run full quickstart V1–V4 validation end-to-end (backend `pytest`, frontend `vitest` + Playwright with screenshots) and confirm green

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (P1)**: no deps — start immediately
- **Foundational (P2)**: depends on Setup — BLOCKS all user stories
- **US1 (P3)**: depends on Foundational — the MVP
- **US2 (P4)**: depends on Foundational; reductions consume US1's embeddings (T019) — start after T019 or run its tests against pre-seeded embeddings
- **US3 (P5)**: depends on Foundational + a running US1/US2 API for e2e; frontend unit tasks can start once contracts are fixed
- **Polish (P6)**: depends on all desired stories complete

### Within Each User Story

- Tests written first and FAIL before implementation
- compute/reduce libraries before precompute orchestration before routes before SSE
- frontend: stores → dataClient → controls/preview → shell → serving

### Parallel Opportunities

- Setup: T002, T003, T004 in parallel
- Foundational: T005, T006, T010 in parallel; then T007→T008, T009, T011, T012
- US1 tests T013–T017 in parallel; US2 tests T023–T025 in parallel
- US2 impl T026/T028 in parallel (T027 after T026)
- US3 controls T034 + styles T035 in parallel; stores T032 before dataClient T033

---

## Parallel Example: User Story 1

```bash
# Launch all US1 tests together (they must fail first):
Task: "Integration test distributions in code/backend/tests/integration/test_distributions.py"
Task: "Integration test embeddings in code/backend/tests/integration/test_embeddings.py"
Task: "Integration test cache in code/backend/tests/integration/test_cache.py"
Task: "Integration test capability in code/backend/tests/integration/test_capability.py"
Task: "Contract test US1 API in code/backend/tests/contract/test_api_us1.py"
```

---

## Implementation Strategy

### MVP First (User Story 1)

1. Phase 1 Setup → Phase 2 Foundational → Phase 3 US1
2. **STOP and VALIDATE**: compute+cache a real `gpt2` distribution with progress;
   confirm cache-hit identity and the capability gate
3. Demo the MVP (the engine works end-to-end at the API layer)

### Incremental Delivery

1. Setup + Foundational → foundation ready
2. US1 → real cached data + progress (MVP)
3. US2 → reductions + reference grid
4. US3 → web shell + controls + preview (the full machinery, visible)
5. Polish → docs, reproducibility, performance evidence

---

## Notes

- [P] = different files, no incomplete dependencies
- Every test calls REAL models (no mocks) per Constitution Principle I
- Commit after each task or logical group; keep docs co-updated (Principle IV)
- Cache artifacts under `data/processed/cache/` are git-ignored and regenerable
- US1 is the MVP and the minimum to prove the machinery
