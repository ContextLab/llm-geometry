# llm-geometry

Understanding large language models through **interactive geometric visualizations** of
their embedding space. The project will ship three explorable web visualizations —
transformer layers as **vector fields**, token sequences as **Sankey diagrams**, and
reachable "thoughts" as a **manifold** — all built on a shared, precompute-and-cache
foundation. See [`project_description.md`](project_description.md) for the science and UX.

## Status

The **core machinery** (the shared backend + frontend foundation all three
visualizations run on) is implemented: open-weights model loading with capability
detection, real next-token distributions and per-layer embeddings, 2D (PCA/UMAP) and 3D
(spherical) reductions with an *n×n* reference grid, an integrity-checked
precompute-and-cache pipeline with single-flight dedup and SSE progress, a FastAPI
service, and a Svelte web shell with the shared controls and a live preview.

Spec, plan, and tasks live in [`specs/001-core-machinery/`](specs/001-core-machinery/).
The three production visualizations are separate, upcoming features.

## Quickstart

Requires Python ≥ 3.10 and Node ≥ 20 (or use the Docker image below). First run
downloads a small open-weights model from HuggingFace and caches it.

```bash
# Backend (FastAPI + PyTorch/transformers + reduction stack)
cd code/backend
python -m venv .venv && . .venv/bin/activate
pip install -e ".[test]"
uvicorn llm_geometry.api.app:app --port 8000     # JSON API on :8000

# Frontend (Svelte + Vite) — in a second terminal
cd code/frontend
npm install
npm run dev                                       # http://localhost:5173 (proxies /api)
```

Open <http://localhost:5173>, keep the default model (`gpt2`) or type any open-weights
HuggingFace id, edit the prompt, and adjust temperature / layer to watch the cached
data update.

### Reproducible environments

A `conda`/`mamba` environment pins **both** stacks (Python 3.11 + Node 20) in one
isolated env:

```bash
mamba env create -f environment.yml   # or: conda env create -f environment.yml
conda activate llm-geometry
uvicorn llm_geometry.api.app:app --port 8000          # backend
cd code/frontend && npm install && npm run dev        # frontend
```

A `Dockerfile` is also provided (same dual stack) for hosts with a running Docker
daemon:

```bash
docker build -t llm-geometry .
docker run -it -p 8000:8000 llm-geometry              # serves API + built UI on :8000
```

## Tests

All tests use **real** models (no mocks):

```bash
cd code/backend && . .venv/bin/activate && pytest -q          # 39 tests, real tiny-gpt2/distilgpt2
cd code/frontend && npm run test                              # vitest unit tests
cd code/frontend && npx playwright install chromium && npm run test:e2e   # e2e + screenshots
```

End-to-end validation steps are in
[`specs/001-core-machinery/quickstart.md`](specs/001-core-machinery/quickstart.md).

## Repository layout

```
code/
├── backend/                 # Python package `llm_geometry`
│   ├── src/llm_geometry/     #   models · compute · reduce · cache · jobs · api
│   └── tests/                #   unit · integration · contract (real-model)
├── frontend/                # Svelte + TypeScript + Vite app (shell, controls, preview)
└── notebooks/               # Jupyter notebooks (paper figures)
data/
├── raw/ · processed/        # inputs; processed/cache/ holds derived artifacts (git-ignored)
paper/                       # LaTeX sources, figures, bibliography submodule
specs/001-core-machinery/    # spec · plan · research · data-model · contracts · tasks
```

Derived precompute artifacts under `data/processed/cache/` are git-ignored and
regenerable from `(model + parameters)`.

## Building the paper

```bash
sh setup.sh                  # one-time: pull the CDL-bibliography submodule
cd paper && sh compile.sh    # build main.{tex→pdf} + supplement
```

## Development

This project uses **Spec-Driven Development** (Spec Kit). Features flow through
specify → clarify → plan → tasks → analyze → implement. See [`CLAUDE.md`](CLAUDE.md) and
the project constitution at
[`.specify/memory/constitution.md`](.specify/memory/constitution.md).
