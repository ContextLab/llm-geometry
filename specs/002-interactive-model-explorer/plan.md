# Implementation Plan: Interactive Model Explorer

**Branch**: `002-interactive-model-explorer` · **Date**: 2026-07-24 · **Spec**: `spec.md`

The full plan — including design decisions D1–D10, red-team resolutions, batch
structure, and testing strategy — is posted (red-teamed) on the tracking issue:
https://github.com/ContextLab/llm-geometry/issues/1#issuecomment-5071465097
This file records the technical context Spec Kit expects; the issue comment governs.

## Technical Context

**Language/Version**: Python 3.10/3.11 (backend), TypeScript 5 + Svelte 4 (frontend)
**Primary Dependencies**: FastAPI, PyTorch, transformers, numpy/scipy (backend);
D3, Three.js, Vite, Vitest, Playwright (frontend)
**Storage**: content-addressed disk cache (`cache/store.py`), `data/processed/cache/`
**Testing**: pytest (real models, no mocks), vitest, Playwright e2e
**Target Platform**: local web app (uvicorn :8000 + Vite :5173), CI on GitHub Actions
**Project Type**: web (backend + frontend)
**Performance Goals**: cache hits < 100 ms; 60 fps interactions; instanced quiver
rendering; debounced (400 ms) trace requests with aborts
**Constraints**: open-weights models only; every visual element ↔ real model
component (1-to-1); no mocks/simulations anywhere; precompute-and-cache with SSE
progress
**Scale/Scope**: 2 new tabs, ~8 new backend endpoints (frozen in
`contracts/api.md`), tiny trained-from-scratch model (d_model=3, 4 layers,
vocab 1003), default HF model SmolLM2-135M-Instruct with 1.5e9-param ceiling

## Structure

- Backend: `llm_geometry/geo/` (tiny model, tokenizer, training, fields),
  `llm_geometry/arch/` (graph tracer, weight tiles, trace, generate),
  `api/routes_geo.py`, `api/routes_arch.py`.
- Frontend: `viz/geo/` (Geometry Lab), `viz/arch/` (Architecture Explorer),
  shared `lib/MatrixHeatmap.svelte` + `lib/PipelineDiagram.svelte`, per-view stores.
- Tests: `tests/{unit,integration,contract}` additions; `tests/e2e/explorer.spec.ts`.
- CI: `.github/workflows/ci.yml` (PR gate) + nightly full suite.
