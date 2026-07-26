# code/

- `backend/` — the `llm_geometry` Python package: open-weights model loading with
  capability detection, the traced-forward-pass architecture graphs and weight windows
  (`arch/`), the from-scratch GeoTransformer with its committed training corpus
  (`geo/`), an integrity-checked artifact cache with single-flight jobs + SSE progress,
  and the FastAPI service exposing it all. Tests (`tests/`) run against real models
  only — the project forbids mocks.
- `frontend/` — the Svelte + TypeScript + Vite app: the two explorer views
  (`src/viz/{arch,geo}/`) and the typed API client plus shared components
  (`src/lib/`). Each view owns its own controls. Vitest unit tests and Playwright e2e
  tests (backend-backed and static-build) live under `tests/`.

Run both together from the repo root with `sh scripts/dev.sh`.
