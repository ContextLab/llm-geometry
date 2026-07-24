# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

`llm-geometry` studies LLMs through **geometric visualizations**. The deliverable described in `project_description.md` is a set of **interactive web pages** with smooth animations and responsive interactions that make the geometry of a transformer's embedding space explorable. The aesthetic bar is explicit: modern, striking, beautiful, clean, intuitive, and thoroughly documented. Anything expensive is **precomputed once and cached**, with an animated progress indicator while it runs.

Read `project_description.md` in full before designing anything — it is the source of truth for the science and the UX, and the snippets below only summarize it.

### The three visualizations (core architecture)
1. **Transformer layers as vector fields** — Reduce embeddings to 2D (UMAP/PCA), lay an *n×n* grid over the space, snap each grid vertex to its nearest token ("reference points"), then draw a quiver arrow from each reference token to the next token it predicts. Interactions: hover to reveal tokens, a layer slider, an editable prompt/context prefix, response shown as a colored trajectory, a temperature slider (>0 fans out into multiple semi-transparent vectors estimated over ~100 reps), and a model selector (incl. arbitrary open-weights HF models).
2. **Token sequences as Sankey diagrams** — A particle swarm estimates the next-token distribution at each position. X = sequence position, Y = token ID. Start from a prompt, sample *n* particles from the position-0 distribution, advance each particle by conditioning on its own draw, combine per-particle distributions into the displayed distribution, and stop a particle once it emits end-of-stream. Same interaction palette (hover, context prefix, temperature, model selector).
3. **Reachable "thoughts" as a manifold** — Reduce embeddings to 3D spherical coordinates (spherical MDS). Place all tokens on a radius-2 sphere and morph a unit sphere toward them, with displacement proportional to emission probability and neighbors dragged along via **RBF interpolation + Open3D `deform_as_rigid_as_possible` (ARAP)**. The order-invariance of combining per-token warps is an **open research question** flagged in the description — do not assume it is solved.

### Hard technical constraints
- Token-level probability distributions are required, so the models must be **open-weights** (HuggingFace), not closed APIs.
- Reductions and per-grid/per-token computations are heavy → design around a **precompute-and-cache** pipeline from the start, not as an afterthought.

## Current state

The **core machinery** (feature `001-core-machinery`) is implemented and verified with
real models — the shared foundation all three visualizations will run on:
- `code/backend/` — Python package `llm_geometry`: open-weights model loading +
  capability detection (`models/`), real next-token distributions + per-layer embeddings
  (`compute/`), 2D/grid/3D-spherical reductions (`reduce/`), integrity-checked
  precompute-and-cache (`cache/`), single-flight job registry (`jobs/`), and a FastAPI
  service with SSE progress (`api/`). Real-model tests in `code/backend/tests/`.
- `code/frontend/` — Svelte + TypeScript + Vite shell: shared controls (model selector,
  prompt prefix, temperature, layer), a cached-data client (`lib/dataClient.ts`), and a
  minimal live preview. Vitest unit + Playwright e2e tests.
- **All three visualizations are implemented** on top of the machinery and selectable
  via the web app's view switcher: a **vector field** (D3 quiver of next-token arrows),
  a **Sankey** diagram (d3-sankey over a particle swarm), and a **manifold** (Three.js
  RBF-warped sphere). Backends live in `code/backend/src/llm_geometry/compute/`
  (`vector_field.py`, `sankey.py`, `manifold.py`); frontends in `code/frontend/src/viz/`.
- **Feature 002 (issue #1) adds two explorer tabs** on the same machinery:
  an **Architecture Explorer** (`llm_geometry/arch/` + `api/routes_arch.py` +
  `viz/arch/`) — traced-forward-pass graphs of real HF models (functional ops are
  first-class nodes; tied weights aliased; pre-download size gate), weight-window
  serving, live traces, and real generation — and a **Geometry Lab**
  (`llm_geometry/geo/` + `api/routes_geo.py` + `viz/geo/`) — a from-scratch
  `d_model=3` GeoTransformer really trained on a committed public-domain corpus,
  with next-next + attention-force vector fields on a Three.js sphere, editable
  weights via content-hash `weights_token`s, and real fine-tuning. The frozen HTTP
  contract is `specs/002-interactive-model-explorer/contracts/api.md`; CI lives in
  `.github/workflows/ci.yml` (all real models — no mocks anywhere).

Still template scaffolding (to be replaced/extended as the science lands):
- `paper/main.tex` is the boilerplate "Template paper" (sin/cos demo figure).
- `code/notebooks/demo.ipynb` only generates that trig demo figure.

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

- `code/backend/` — Python package `llm_geometry` (`src/llm_geometry/{models,compute,reduce,cache,jobs,api}`) + `tests/{unit,integration,contract}`; `pyproject.toml`, `requirements.txt`, pinned `requirements.lock`.
- `code/frontend/` — Svelte + TS + Vite app (`src/{controls,lib,preview,styles}`) + `tests/{unit,e2e}`.
- `code/notebooks/` — Jupyter notebooks, one per paper figure (`demo.ipynb` = Figure 1).
- `data/raw/`, `data/processed/` — inputs before/after processing; `data/processed/cache/` holds derived precompute artifacts (git-ignored, regenerable from model+params).
- `paper/` — LaTeX sources (`main.tex`, `supplement.tex`), `figs/` (final PDFs) and `figs/source/` (panel sources — `trig.pdf` links to `source/sin.pdf`/`cos.pdf` and is meant to be re-linked in Illustrator), `admin/` (cover letters, forms), and the `CDL-bibliography` git submodule.

## Commands

```bash
# One-time setup: pull the CDL-bibliography submodule
sh setup.sh

# Backend (FastAPI + PyTorch/transformers + reductions)
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

# Build the paper (run from paper/; needs a LaTeX toolchain)
cd paper && sh compile.sh      # main.{tex→pdf} + supplement, then cleans aux files
```

Note: `data/processed/cache/` holds derived precompute artifacts — git-ignored and
regenerable from `(model + params)`; delete it freely to force a rebuild.

Note: the existing `Dockerfile` pins an old scientific-Python stack (Python 3.7, conda). The visualization framework will need its own dependency setup (open-weights model inference, dimensionality reduction, Open3D, a web stack) — extend the environment deliberately and keep `Dockerfile`/requirements in sync with what the code actually imports.

<!-- SPECKIT START -->
Active feature: **001-core-machinery** — Core Project Machinery (the shared
backend+frontend foundation all three visualizations run on). For technologies,
project structure, and commands, read the current plan and its artifacts:
- Plan: `specs/001-core-machinery/plan.md`
- Spec: `specs/001-core-machinery/spec.md`
- Research / data model / API contract / quickstart:
  `specs/001-core-machinery/{research.md,data-model.md,contracts/api.md,quickstart.md}`
- Tasks: `specs/001-core-machinery/tasks.md`
<!-- SPECKIT END -->
