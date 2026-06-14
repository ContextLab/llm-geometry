# Implementation Plan: Core Project Machinery

**Branch**: `001-core-machinery` | **Date**: 2026-06-14 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/001-core-machinery/spec.md`

## Summary

Build the shared foundation that all three `llm-geometry` visualizations will run
on: a Python backend that loads open-weights HuggingFace models, extracts genuine
token-level next-token distributions and per-layer embeddings, reduces those
embeddings to 2D (PCA/UMAP) and 3D spherical (MDS) coordinates, and precomputes-
and-caches every expensive result once (single-flight, integrity-checked,
regenerable); a FastAPI service that serves cached artifacts as JSON and streams
precompute progress; and a Svelte + TypeScript web shell with the reusable controls
(model selector, prompt/context prefix, temperature, layer selector), a cached-data
client, a polished progress experience, and a minimal data-preview that proves the
whole model → compute → cache → serve → display path end-to-end — without building
any of the three production visualizations.

## Technical Context

**Language/Version**: Python 3.11 (backend); TypeScript 5.x on Node 20 (frontend)

**Primary Dependencies**:
- Backend: PyTorch (CPU-default, CUDA-optional), HuggingFace `transformers` +
  `huggingface_hub` (model load, logits→probabilities, `output_hidden_states`→
  per-layer embeddings, capability/gated detection), `numpy`, `scipy` (cdist/RBF,
  spherical geometry), `scikit-learn` (PCA, metric MDS), `umap-learn` (UMAP),
  `fastapi` + `uvicorn` (API + SSE progress), `pydantic` v2 (contracts/validation),
  `orjson`/`numpy` (`.npz`) for artifact serialization.
- Frontend: Svelte + Vite, TypeScript, D3.js (2D substrate; used minimally in the
  preview), Three.js (3D substrate; used minimally in the preview). No heavyweight
  state library — Svelte stores suffice.

**Storage**: Filesystem cache of derived artifacts under `data/processed/cache/`
(regenerable, git-ignored). Model weights use the standard HuggingFace cache
(`~/.cache/huggingface`), never committed. No database.

**Testing**: `pytest` (backend) with **real model calls** — `sshleifer/tiny-gpt2`
for fast paths and `distilgpt2`/`gpt2` for at least one real-output assertion per
capability; `vitest` (frontend unit); Playwright (frontend e2e + screenshot
verification of the shell, controls, progress animation, and preview).

**Target Platform**: Local/self-hosted web app for development (Vite dev server +
Uvicorn in dev; FastAPI serves the built static bundle + JSON API in prod).
Reproducible via the repo `Dockerfile` and pinned manifests.

**Project Type**: Web application (Python backend + Svelte frontend).

**Performance Goals** (the feature's measurable budgets; see SC-001/003/004):
- Cached interaction: backend cache-hit response **< 100 ms**; full round-trip
  (control change → preview updated) **< 1 s**.
- Animations/transitions render at **60 fps (~16 ms/frame)**.
- First-time precompute for the default model on the reference machine completes
  within a **documented budget (target ≤ 180 s)** with progress updates **≥ 1/s**.

**Constraints**:
- Reference machine: Apple-Silicon-class laptop, 16 GB RAM, CPU inference (no GPU
  assumed). Larger models/GPU are supported but not required.
- Open-weights, probability-exposing models only; never fall back or fabricate.
- Reproducibility: identical (model, inputs, params, seed) → identical cached
  outputs; PCA/MDS deterministic, UMAP seeded.

**Scale/Scope**: Default model `gpt2` (50,257-token vocab, 12 layers + embedding).
Default reference token set = configurable; defaults to the full vocabulary via the
static input-embedding matrix for the base layer, with contextual per-layer
embeddings computed in batches and cached. Default 2D reference grid `n = 25`
(625 vertices). Single-user/local concurrency (single-flight dedup of identical
precompute jobs); not a multi-tenant service in this feature.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Evaluated against Constitution v2.0.0:

|Principle / Constraint|How this plan complies|
|-|-|
|**I. Accuracy, Verification & Integrity**|All tests call real models with real inputs (no mock-only). Cache integrity checks prevent serving partial/stale results as real. Unverified items flagged `TODO(verify:)`. No placeholder/"expected" outputs ever surfaced.|
|**II. Single Source of Truth**|One canonical module per concern; edits in place; no `_v2`/backup files. Every cache artifact is derived and regenerable from (model+params); cache is git-ignored.|
|**III. UX Is Paramount**|Concrete perf budgets above; precompute-and-cache; immediate progress UI (SSE) whenever a result isn't instant; 60 fps target enforced via Playwright/visual checks.|
|**IV. Documentation Stays Current**|This change set updates `README.md`, `CLAUDE.md`, `Dockerfile`, dependency manifests, and the SPECKIT agent-context block in the same work. Quickstart is runnable.|
|**V. Reproducibility & Open Release**|Containerized (`Dockerfile`) + pinned manifests (`requirements.txt`, `package.json`/lockfile); quickstart lets an independent party reproduce cached data for the default model with no undocumented steps.|
|**Real verification, never mocks**|`pytest` integration tests download and run real HF models; failure to load a real model fails loudly.|
|**Cached computation**|Core of the design (cache module + single-flight jobs).|
|**Open-weights models**|Enforced by capability detection (FR-002/003).|
|**Per-feature performance budgets**|Defined above with the strategy (threading/async, caching, progress UI) to meet them.|

**Result**: PASS — no violations. Complexity Tracking left empty.

## Project Structure

### Documentation (this feature)

```text
specs/001-core-machinery/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output (HTTP API contract)
│   └── api.md
├── checklists/
│   └── requirements.md  # (from /speckit-specify)
└── tasks.md             # Phase 2 output (/speckit-tasks)
```

### Source Code (repository root)

```text
code/
├── backend/
│   ├── pyproject.toml             # package metadata for `llm_geometry`
│   ├── requirements.txt           # pinned backend deps
│   ├── src/llm_geometry/
│   │   ├── __init__.py
│   │   ├── config.py              # paths, defaults, perf budgets, seeds
│   │   ├── models/                # model load + capability detection
│   │   │   ├── __init__.py
│   │   │   ├── registry.py        # curated menu + arbitrary HF id resolution
│   │   │   └── loader.py          # load model/tokenizer, capability checks
│   │   ├── compute/               # real model-derived quantities
│   │   │   ├── embeddings.py      # per-layer token embeddings (batched)
│   │   │   └── distributions.py   # next-token probability distributions
│   │   ├── reduce/                # dimensionality reduction + reference geometry
│   │   │   ├── twod.py            # PCA/UMAP 2D
│   │   │   ├── grid.py            # n×n grid + nearest-token reference points
│   │   │   └── sphere.py          # spherical MDS (3D on a sphere)
│   │   ├── cache/                 # keys, storage, integrity, single-flight
│   │   │   ├── keys.py            # deterministic cache keys (+ schema version)
│   │   │   └── store.py           # read/write/verify/regenerate artifacts
│   │   ├── jobs/                  # precompute job registry + progress
│   │   │   └── registry.py        # status, progress, single-flight dedup
│   │   └── api/                   # FastAPI app
│   │       ├── app.py             # ASGI app, static serving, CORS (dev)
│   │       ├── routes.py          # endpoints (see contracts/api.md)
│   │       └── progress.py        # SSE progress stream
│   └── tests/
│       ├── unit/                  # cache keys, grid, reductions (real arrays)
│       ├── integration/           # REAL model: load→compute→cache→reduce
│       └── contract/              # API endpoint contract tests (real app)
└── frontend/
    ├── package.json
    ├── vite.config.ts
    ├── src/
    │   ├── main.ts
    │   ├── App.svelte             # shared shell + layout + theming
    │   ├── controls/
    │   │   ├── ModelSelector.svelte
    │   │   ├── PromptPrefix.svelte
    │   │   ├── Temperature.svelte
    │   │   └── LayerSlider.svelte
    │   ├── lib/
    │   │   ├── dataClient.ts      # cached-data access + progress subscription
    │   │   ├── stores.ts          # shared interaction-parameter state
    │   │   └── viz/               # minimal D3/Three.js mounts for the preview
    │   ├── preview/
    │   │   └── Preview.svelte     # minimal verification surface
    │   └── styles/                # design tokens, animations
    └── tests/
        ├── unit/                  # vitest (dataClient, stores)
        └── e2e/                   # playwright (shell, controls, progress, preview)

data/processed/cache/             # derived artifacts (git-ignored, regenerable)
```

**Structure Decision**: Web-application layout placed under the repo's existing
`code/` home (honoring the CDL repo convention) with a Python package
`code/backend/src/llm_geometry` and a Svelte app in `code/frontend`. Derived cache
artifacts live in `data/processed/cache/` (the repo's processed-data home), are
git-ignored, and are regenerable from the model + parameters. Repo-root `Dockerfile`
and `README.md`/`CLAUDE.md` are updated to cover the new dual-stack environment.

## Complexity Tracking

> No constitution violations — no entries required.
