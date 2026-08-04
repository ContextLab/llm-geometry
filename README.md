# llm-geometry

Understanding large language models through **interactive geometric visualizations**.
Live at **<https://context-lab.com/llm-geometry/>** (a static build that runs the models
in your browser); the full stack below adds live tracing against real PyTorch.

Four tabs — three explorable views over real models, plus a reference tab. No mocks, no
canned data:

1. **Architecture** — a real open-weights model traced live: **every op of the forward
   pass** — RoPE, attention softmax, residual adds included — is a clickable node.
   Re-trace your own prompt, ▶ play the trace through the diagram, see every attention
   head of a layer at once, zoom into any weight matrix's actual values, and generate
   replies with per-token probabilities. Models come from a curated list (see
   [issue #4](https://github.com/ContextLab/llm-geometry/issues/4) for expanding it).
2. **Geometry** — a from-scratch *GeoTransformer* (`d_model=3`, 4 layers, 1 head,
   1000-word vocab in 1003 rows) really trained on Alice in Wonderland, its 3-D token embeddings
   living directly on a rendered sphere. Explore the *next-next-token* field or the
   attention-force field `Σ softmax(⟨Kz_j,Qz_i⟩)·Vz_j`
   ([arXiv:2607.13295](https://arxiv.org/abs/2607.13295)) per layer, edit
   W_Q/W_K/W_V/W_O or the embeddings (presets or cell-by-cell), train a brand-new model
   from scratch on your own text or a real HuggingFace dataset, fine-tune with real SGD,
   and save/load the result as a file.
3. **Lexicon** — the model stays small and the **vocabulary budget** becomes the control.
   A word-level transformer trains from scratch *in your browser* (both builds — this tab
   never calls the backend) on *The Real Mother Goose* (1916), under either a prescribed
   Dolch budget or the corpus's own most frequent words at the same `|V|`, so coverage,
   held-out loss, the text it generates and its embedding spectrum all answer together.
   The same tab carries the **vacancy transform**: replace content stems with
   pronounceable nonces (or swap them for each other) at a rate `p` while every function
   word and inflection stays put, and measure what a word's *identity* is worth to a model
   that has never seen one — against what it is worth to a pretrained model, scored in the
   Architecture tab.
4. **Info** — the reference tab: notation, the GeoTransformer's forward pass as equations,
   both field definitions, a what-is-real / where-it-runs table, the known limits, and the
   references. Every number in it is transcribed from a source constant, and the e2e suite
   fails if one drifts.

See [`project_description.md`](project_description.md) for the science and UX vision,
and [issue #1](https://github.com/ContextLab/llm-geometry/issues/1) for the build log
(design decisions, red-team rounds, screenshots).

## Quickstart

Requires Python ≥ 3.10 and Node ≥ 20 (or use conda/Docker below). First run downloads
a small open-weights model from HuggingFace and caches it; the Geometry tab trains its
tiny model once (~25 s) and caches the checkpoint.

```bash
# Backend (FastAPI + PyTorch/transformers)
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
ruff check src/ tests/ && black --check src/ && pytest -q      # real-model tests, no mocks

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
│   ├── src/llm_geometry/     #   models · cache · jobs · api
│   │   ├── arch/             #   Architecture tab: traced graphs, weight windows, generation
│   │   ├── geo/              #   Geometry tab: GeoTransformer, training, fields (+ corpus data)
│   │   └── lex/              #   Lexicon tab: budgets, word tokenizer, vacancy transform (+ corpus)
│   └── tests/                #   unit · integration · contract (all real-model)
└── frontend/                 # Svelte + TypeScript + Vite app
    └── src/                  #   viz/{arch,geo,lex,info} — one directory per tab
                              #   lib/ — typed API client, static client, TS engines, components
docs/screenshots/             # verification screenshots referenced from issue threads
notes/                        # session notes + agent red-team/fix reports
scripts/dev.sh                # dev-stack launcher (both servers, health-checked)
specs/                        # Spec Kit features (001 is superseded; 007 is current)
.cache/llm-geometry/          # derived artifacts: checkpoints, graphs (git-ignored)
```

The HTTP contract for every tab is frozen at
[`specs/002-interactive-model-explorer/contracts/api.md`](specs/002-interactive-model-explorer/contracts/api.md);
later features may only add endpoints to it, in their own commit.

## Development

This project uses **Spec-Driven Development** (Spec Kit): specify → clarify → plan →
tasks → analyze → implement. See [`CLAUDE.md`](CLAUDE.md) and the project constitution
at [`.specify/memory/constitution.md`](.specify/memory/constitution.md).
