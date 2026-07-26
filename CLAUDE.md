# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

`llm-geometry` studies LLMs through **geometric visualizations** — interactive web pages
with smooth animations and responsive interactions that make a transformer's internals
explorable. The aesthetic bar is explicit: modern, striking, beautiful, clean, intuitive,
thoroughly documented. Anything expensive is **computed once and cached**, with an
animated progress indicator while it runs.

`project_description.md` is the ORIGINAL vision document (three embedding-geometry
visualizations: vector field, Sankey, manifold). Read it for the science and the
aesthetic intent, but note that **feature 004 deliberately narrowed the deliverable**:
those three views were removed in favor of the two explorer tabs below. Treat
`project_description.md` as history plus design taste, not as a spec of what ships.

### Hard technical constraints
- Token-level probability distributions are required, so models must be **open-weights**
  (HuggingFace), not closed APIs.
- The static GitHub Pages build has no Python at runtime, so anything it shows is either
  computed live in the browser, range-read from HuggingFace's CDN, or precomputed by the
  real backend at build time — **never fabricated, never silently degraded**.

## Current state

The app is a **three-tab explorer** — two visualizations plus a reference tab — deployed at
https://context-lab.com/llm-geometry/.

- **Architecture Explorer** (`llm_geometry/arch/` + `api/routes_arch.py` + `viz/arch/`) —
  traced-forward-pass graphs of real HF models (functional ops are first-class nodes;
  tied weights aliased; pre-download size gate), weight-window serving, live traces, and
  real generation. The model menu is **curated only** — the old "any open-weights HF id"
  input was removed in 004 because the static build needs a community ONNX export that
  most repos lack; expanding it is tracked in issue #4.
- **Geometry Lab** (`llm_geometry/geo/` + `api/routes_geo.py` + `viz/geo/`) — a
  from-scratch `d_model=3` GeoTransformer really trained on a committed public-domain
  corpus, with next-next + attention-force vector fields on a Three.js sphere, editable
  weights via content-hash `weights_token`s, real fine-tuning, from-scratch training on
  arbitrary text or a real HuggingFace dataset, and file save/load.
- **Info tab** (`llm_geometry`-free; `viz/info/InfoTab.svelte` + `lib/Explain.svelte`) — the
  reference surface added by feature 005: notation, the GeoTransformer's forward pass as
  equations, both field definitions, the what-is-real/where-it-runs table, known limits, and
  verified references. Both explorer tabs also carry always-visible orientation prose plus
  collapsible `Explain` deep-dives. **Every number in that prose is transcribed from a source
  constant** — `tests/e2e/docs.spec.ts` pins the ones cheapest to let rot, so changing a constant
  without changing the sentence fails CI.
- **Static build** (feature 003) — a TypeScript port of the GeoTransformer
  (`src/lib/geoEngine/`, golden-tested against the Python backend to ≤1e-5),
  transformers.js generation, and safetensors HTTP Range reads. `VITE_DATA_MODE=static`
  selects `src/lib/staticClient/`.

The frozen HTTP contract is `specs/002-interactive-model-explorer/contracts/api.md` —
**change it only in its own commit, with a note explaining why**. CI is
`.github/workflows/ci.yml`; the Pages deploy is `.github/workflows/pages.yml` (both run
real models — no mocks anywhere). CI pins **Python 3.10** (the lock file is frozen from a
3.10 venv). CI caches MUST use run-id keys with `restore-keys` prefixes, never fixed keys:
`actions/cache` entries are immutable, and a fixed key once froze an empty snapshot and
made every run start cold.

**Removed in feature 004** (`specs/004-two-tab-explorer/spec.md`): the vector-field,
Sankey, and manifold views, plus `compute/{vector_field,sankey,manifold,token_cloud,
distributions,printable,embeddings,context}.py`, `reduce/`, `precompute.py`, the shared
control sidebar, and their tests/presets. Feature 001's spec is marked superseded rather
than deleted so the history stays readable.

## How work happens here: Spec-Driven Development (Spec Kit)

This project is initialized with **Spec Kit** (`.specify/`, integration = `claude`). Features are built through a spec → plan → tasks → implement pipeline, exposed as `speckit-*` skills (and `/speckit.*` slash commands). Use them rather than free-styling large features:

| Stage | Skill | Purpose |
|-|-|-|
| Constitution | `speckit-constitution` | Fill the project principles (currently template placeholders) |
| Specify | `speckit-specify` | Create `specs/NNN-<slug>/spec.md` from a feature description |
| Clarify | `speckit-clarify` | Resolve underspecified areas before planning |
| Plan | `speckit-plan` | Produce `plan.md` (+ research/data-model/quickstart) |
| Tasks | `speckit-tasks` | Generate dependency-ordered `tasks.md` |
| Analyze | `speckit-analyze` | Cross-check spec/plan/tasks consistency |
| Implement | `speckit-implement` | Execute the tasks |

- New features live under `specs/NNN-<slug>/` (created by `.specify/scripts/bash/create-new-feature.sh`; the active feature is tracked in `.specify/feature.json`). The `specs/` directory does not exist until the first feature is specified.
- The `<!-- SPECKIT START/END -->` block at the bottom of this file is **auto-managed** by the `speckit-agent-context-update` extension (config: `.specify/extensions/agent-context/agent-context-config.yml`). It is rewritten to point at the active plan — leave its contents to the tooling; edit only the human-authored sections above it.

## Repository layout

- `code/backend/` — Python package `llm_geometry` (`src/llm_geometry/{models,cache,jobs,api,arch,geo}`) + `tests/{unit,integration,contract}`; `pyproject.toml`, `requirements.txt`, pinned `requirements.lock`. The geo training corpus is package data (`geo/data/`).
- `code/frontend/` — Svelte + TS + Vite app (`src/{viz,controls,lib,styles}`) + `tests/{unit,e2e}`. `src/viz/{arch,geo}/` are the two tabs; each owns its controls.
- `docs/screenshots/` — verification screenshots referenced from issue threads.
- `notes/` — session notes and agent red-team/fix reports (`notes/agent-reports/`).
- `scripts/dev.sh` — health-checked dev-stack launcher for both servers.
- `.cache/llm-geometry/` — derived artifacts (git-ignored, regenerable; `LLM_GEOMETRY_CACHE_DIR` overrides).

## Commands

```bash
# Backend (FastAPI + PyTorch/transformers)
cd code/backend && python -m venv .venv && . .venv/bin/activate && pip install -e ".[test]"
uvicorn llm_geometry.api.app:app --port 8000          # JSON API on :8000
pytest -q                                             # real-model tests (no mocks)

# Frontend (Svelte + Vite)
cd code/frontend && npm install
npm run dev                                           # http://localhost:5173 (proxies /api)
npm run check && npm run test                         # svelte-check + vitest
npx playwright install chromium && npm run test:e2e   # e2e + screenshots

# Reproducible cross-stack env (pins Python 3.11 + Node 20) — primary, no daemon needed
mamba env create -f environment.yml && conda activate llm-geometry
# Or a container (requires a running Docker daemon)
docker build -t llm-geometry . && docker run -it -p 8000:8000 llm-geometry

# Or start both dev servers with one health-checked command
sh scripts/dev.sh              # sh scripts/dev.sh stop
```

Note: `.cache/llm-geometry/` holds derived artifacts (checkpoints, traced graphs) — git-ignored and
regenerable from `(model + params)`; delete it freely to force a rebuild
(`LLM_GEOMETRY_CACHE_DIR` overrides the location).

Note: the `Dockerfile` builds the real dual stack (python:3.11-slim + Node 20,
`pip install -e code/backend[test]`, `npm run build`, uvicorn on :8000) — keep it
and the requirements files in sync with what the code actually imports.

<!-- SPECKIT START -->
Active feature: **005-explain-the-visualizations** — the Info tab and the in-tab
explanatory text, written for a mathematically sophisticated reader. If you change a
constant, an equation, or what a control does, change the sentence that documents it in
the same commit.
- Spec: `specs/005-explain-the-visualizations/spec.md`
- Previous: `specs/004-two-tab-explorer/spec.md` (removals, defect fixes, real
  from-scratch training) — still the spec for both tabs' behavior
- Frozen API contract (both tabs):
  `specs/002-interactive-model-explorer/contracts/api.md`
- Superseded: `specs/001-core-machinery/` (the three removed views)
<!-- SPECKIT END -->
