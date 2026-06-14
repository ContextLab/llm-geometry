# Quickstart & Validation: Core Project Machinery

Runnable proof that the core machinery works end-to-end. Details of shapes/endpoints
live in [data-model.md](./data-model.md) and [contracts/api.md](./contracts/api.md);
this file is the **validation/run guide** (Principle V — independently reproducible).

## Prerequisites

- Python 3.11, Node 20 (or use the repo `Dockerfile`, which provides both).
- Internet access on first run (to download the open-weights model from HuggingFace;
  cached to `~/.cache/huggingface` thereafter).
- Reference machine: Apple-Silicon-class laptop, 16 GB RAM, CPU-only is sufficient.

## Setup

```bash
# Backend
cd code/backend
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # pinned deps; installs the llm_geometry package (-e .)

# Frontend
cd ../frontend
npm ci                                    # from package.json + lockfile
```

Reproducible cross-stack env via conda/mamba (pins Python 3.11 + Node 20; no daemon):

```bash
mamba env create -f environment.yml       # repo root (or: conda env create -f ...)
conda activate llm-geometry
```

Or a container (requires a running Docker daemon):

```bash
docker build -t llm-geometry .            # repo root
docker run -it -p 8000:8000 -p 5173:5173 -v "$PWD":/mnt llm-geometry
```

## Run

```bash
# Terminal 1 — backend API (serves /api/* and, in prod, the built frontend)
cd code/backend && . .venv/bin/activate
uvicorn llm_geometry.api.app:app --port 8000

# Terminal 2 — frontend dev server (proxies /api to :8000)
cd code/frontend && npm run dev           # http://localhost:5173
```

## Validation scenarios (map to spec User Stories / Success Criteria)

### V1 — Compute & cache with visible progress (US1, SC-002/SC-003)

1. Open `http://localhost:5173`, keep the default model (`gpt2`), type a prefix
   (e.g., "The capital of France is"), press compute.
2. **Expect** a progress animation that advances and a message stream (SSE), then a
   next-token distribution rendered in the preview — **real** `gpt2` output (the top
   token should be a plausible continuation).
3. Re-run the identical request. **Expect** an instant (cache-hit) response,
   identical to the first (verify via the determinism test below).

### V2 — Reductions & reference grid (US2)

```bash
curl "http://localhost:8000/api/reduction/2d?model_id=gpt2&method=pca&with_grid=true&grid_n=25"
curl "http://localhost:8000/api/reduction/3d?model_id=gpt2&method=pca3"   # mds also available
```

**Expect** a 2D coord set with a 25×25 grid whose `reference_token_ids` resolve to
nearest tokens, and a 3D coord set with every point on the unit sphere.

### V3 — Shared shell & controls, instant vs progress (US3, SC-001/SC-004)

1. Change the **model**, **prefix**, **temperature**, and **layer** controls in turn.
2. **Expect** cached changes to update the preview near-instantly (< 1 s) and uncached
   changes to show the progress animation, then resolve to real data.
3. Visually confirm (Playwright screenshots) the shell/controls render cleanly and
   transitions are smooth (60 fps target).

### V4 — Unsupported model fails loudly (US1 AC4, SC-007)

```bash
curl -X POST http://localhost:8000/api/models/resolve \
  -H 'content-type: application/json' -d '{"model_id":"definitely-not-a-real-model-xyz"}'
```

**Expect** HTTP 422 `UnsupportedModelError` with a clear message; **no** cache
artifact created and **no** fallback model used.

## Automated verification

```bash
# Backend — REAL model calls (no mocks); downloads tiny/distil GPT-2
cd code/backend && . .venv/bin/activate && pytest -q

# Frontend — unit + e2e (Playwright launches the app and screenshots the preview)
cd ../frontend && npm run test && npm run test:e2e
```

Key automated checks:
- **Determinism (SC-002)**: compute an artifact, delete the cache, recompute → assert
  byte/array identity.
- **Real distribution (US1)**: `distilgpt2` next-token probs sum to 1.0 and the top
  token is the real argmax (compared against a direct forward pass).
- **Capability gate (SC-007)**: a non-open-weights / missing id raises
  `UnsupportedModelError` (no fallback).
- **Single-flight (FR-008)**: two concurrent identical precompute requests run the
  work once.
- **Interrupted precompute (edge)**: a partially-written artifact is not served as a
  cache hit; it is recomputed.

## Expected outcomes (verified 2026-06-14 on the reference machine)

- [x] V1–V4 pass; e2e screenshots saved under `code/frontend/tests/e2e/__screenshots__/`
      (`shell.png`, `error.png`).
- [x] Backend `pytest` green: **36 passed** with real models (`sshleifer/tiny-gpt2`,
      `distilgpt2`). Frontend: **vitest 5 passed**, **Playwright e2e 3 passed**.
- [x] **SC-001** cache-hit latency: mean **0.30 ms**, p95 **0.36 ms** (budget < 100 ms).
- [x] **SC-003** first-time precompute (gpt2, 256 contextual embeddings → 2D): **1.5 s**
      (budget ≤ 180 s), with SSE progress emitted ≥ 1/s.
- [x] **SC-004** shell transitions use GPU-composited CSS transforms toward the 60 fps
      target; e2e Playwright run renders smoothly.
- [x] `README.md` / `CLAUDE.md` / `Dockerfile` / `requirements.lock` / `package.json`
      reflect the dual-stack environment.
