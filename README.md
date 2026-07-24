# llm-geometry

Understanding large language models through **interactive geometric visualizations**.
Five explorable views over real open-weights models — no mocks, no canned data:

1. **Vector field** — an *n×n* grid of reference tokens in 2D-reduced embedding
   space, each casting arrows toward its likely next tokens (temperature fans the
   arrows out; a typed response traces an animated trajectory).
2. **Sankey** — a particle swarm samples next tokens position-by-position; flow width
   is the particle count, with a teacher-forced gold path for your own response.
3. **Manifold** — a unit sphere warped (RBF + ARAP) toward the true top next tokens,
   with an optional surface flow field showing where each likely token leads next.
4. **Architecture** — a real model (default SmolLM2-135M-Instruct; any open HF id,
   size-gated before download) traced live: **every op of the forward pass** — RoPE,
   attention softmax, residual adds included — is a clickable node. Re-trace your own
   prompt, ▶ play the trace through the diagram, zoom into any weight matrix's actual
   values, and generate replies with per-token probabilities.
5. **Geometry** — a from-scratch *GeoTransformer* (`d_model=3`, 4 layers, 1 head,
   1000-word vocab) really trained on Alice in Wonderland, its 3-D token embeddings
   living directly on a rendered sphere. Explore the *next-next-token* field or the
   attention-force field `Σ softmax(⟨Kz_j,Qz_i⟩)·Vz_j`
   ([arXiv:2607.13295](https://arxiv.org/abs/2607.13295)) per layer, edit
   W_Q/W_K/W_V/W_O or the embeddings (presets or cell-by-cell), and fine-tune on your
   own text/file/HF dataset with real SGD.

See [`project_description.md`](project_description.md) for the science and UX vision,
and [issue #1](https://github.com/ContextLab/llm-geometry/issues/1) for the build log
(design decisions, red-team rounds, screenshots).

## Quickstart

Requires Python ≥ 3.10 and Node ≥ 20 (or use conda/Docker below). First run downloads
a small open-weights model from HuggingFace and caches it; the Geometry tab trains its
tiny model once (~25 s) and caches the checkpoint.

```bash
# Backend (FastAPI + PyTorch/transformers + reduction stack)
cd code/backend
python -m venv .venv && . .venv/bin/activate
pip install -e ".[test]"
uvicorn llm_geometry.api.app:app --port 8000      # JSON API on :8000

# Frontend (Svelte + Vite) — in a second terminal
cd code/frontend
npm install
npm run dev                                       # http://localhost:5173 (proxies /api)
```

Or start both with one command (health-checked, stale-port guarded):

```bash
sh scripts/dev.sh          # start · sh scripts/dev.sh stop
```

### Reproducible environments

One `conda`/`mamba` env pins both stacks (Python 3.11 + Node 20):

```bash
mamba env create -f environment.yml && conda activate llm-geometry
uvicorn llm_geometry.api.app:app --port 8000          # backend
cd code/frontend && npm install && npm run dev        # frontend
```

Or Docker (serves the API + built UI on :8000):

```bash
docker build -t llm-geometry . && docker run -it -p 8000:8000 llm-geometry
```

## Tests

Everything runs against **real models** — real downloads, real training, real
browsers; the project forbids mocks:

```bash
cd code/backend && . .venv/bin/activate
ruff check src/ tests/ && black --check src/ && pytest -q      # 200+ real-model tests

cd ../frontend
npm run check && npm run test && npm run build                 # types · unit · build
npx playwright install chromium && npm run test:e2e            # e2e vs the live stack
```

The same gate runs in CI (`.github/workflows/ci.yml`) on every push plus a nightly
schedule, with HuggingFace model and artifact caches.

## Repository layout

```
code/
├── backend/                  # Python package `llm_geometry`
│   ├── src/llm_geometry/     #   models · compute · reduce · cache · jobs · api
│   │   ├── arch/             #   Architecture tab: traced graphs, weight windows, generation
│   │   └── geo/              #   Geometry tab: GeoTransformer, training, fields (+ corpus data)
│   └── tests/                #   unit · integration · contract (all real-model)
└── frontend/                 # Svelte + TypeScript + Vite app
    └── src/{viz,controls,lib}/   # five views · shared controls · typed API client
docs/screenshots/             # verification screenshots referenced from issue threads
notes/                        # session notes + agent red-team/fix reports
scripts/dev.sh                # dev-stack launcher (both servers, health-checked)
specs/                        # Spec Kit features: 001 core machinery · 002 explorer tabs
.cache/llm-geometry/          # derived precompute artifacts (git-ignored, regenerable)
```

The API contract for the explorer tabs is frozen at
[`specs/002-interactive-model-explorer/contracts/api.md`](specs/002-interactive-model-explorer/contracts/api.md).

## Development

This project uses **Spec-Driven Development** (Spec Kit): specify → clarify → plan →
tasks → analyze → implement. See [`CLAUDE.md`](CLAUDE.md) and the project constitution
at [`.specify/memory/constitution.md`](.specify/memory/constitution.md).
