# code/

- `backend/` — the `llm_geometry` Python package: open-weights model loading with
  capability detection, real next-token distributions and per-layer embeddings,
  2D/3D reductions, the integrity-checked precompute-and-cache pipeline with
  single-flight jobs + SSE progress, the traced-forward-pass architecture graphs
  (`arch/`), the from-scratch GeoTransformer with its committed training corpus
  (`geo/`), and the FastAPI service exposing it all. Tests (`tests/`) run against
  real models only.
- `frontend/` — the Svelte + TypeScript + Vite app: the five visualization views
  (`src/viz/`), shared controls (`src/controls/`), and the typed API client plus
  shared components (`src/lib/`). Vitest unit tests and Playwright e2e tests under
  `tests/`.

Run both together from the repo root with `sh scripts/dev.sh`.
