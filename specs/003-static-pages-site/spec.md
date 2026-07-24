# Feature Specification: Static GitHub Pages Build

**Feature Branch**: `003-static-pages-site`

**Created**: 2026-07-24

**Status**: In progress

**Input**: Issue #1's "ideally it could run entirely in a browser" + the owner's
request to host on GitHub Pages. Research grounding (all claims verified):
`notes/agent-reports/static-site-research.md`. Technical plan: `plan.md`.

The full app served from **https://context-lab.com/llm-geometry/** (org custom
domain; Vite base `/llm-geometry/`) with **no Python backend**. Same five tabs, same
no-fabrication rules: everything shown is real — computed live in the browser,
range-read from HuggingFace's CDN, or precomputed by the real backend at build time
and clearly labeled as such.

## User Stories

### US-1: Geometry Lab fully live in-browser (P1)

The tiny GeoTransformer runs natively in TypeScript: forward/trace/fields/weight
edits/fine-tuning all execute client-side (Float32Array math; fine-tune in a Web
Worker). The trained checkpoint (~31 KB JSON) + tokenizer vocab ship as static
assets. **Acceptance**: every Geometry-tab interaction from feature 002 works with
the backend absent; a golden-vector test suite proves the TS engine matches the
Python backend's trace/field/finetune outputs to ≤1e-5.

### US-2: Architecture Explorer live where the browser allows, honest where not (P1)

- Generation/chat: LIVE via transformers.js v4 (`device: "webgpu"`, `dtype:
  "q4f16"`; WASM+q8 fallback), per-token probabilities from real logits.
- Tokenization strip: LIVE via transformers.js AutoTokenizer (vendored
  tokenizer.json, no model download).
- Weight inspector: LIVE exact windows via safetensors HTTP **Range** reads from
  `huggingface.co/.../resolve/<pinned-commit>/` (CORS verified; bf16→f32 decode);
  overview tiles precomputed (uint8 `.bin` + manifest).
- Architecture graph: precomputed JSON per curated model (ONNX exports expose no
  hooks — the graph still comes from the real traced forward pass, at build time).
- Trace panel: ONNX exposes no attentions/hidden states, so per-prompt traces are
  precomputed for a labeled dropdown of example prompts; free-prompt tracing shows
  a clear "full stack only" affordance linking to the README.
**Acceptance**: chat produces real replies in-browser; a zoomed weight window's
values equal the safetensors bytes; the degradation ladder (webgpu → wasm →
precomputed-only) shows honest banners at each level.

### US-3: The three 001 views on precomputed presets (P2)

Vector field / Sankey / Manifold render from precomputed artifacts for the default
model and a labeled set of preset prompts/params. Controls that re-slice loaded data
(layer slider, playback, hover, surface toggle) stay fully interactive; free-text
inputs are disabled with an honest "run the full stack" note. **Acceptance**: each
tab renders its presets and animates; no control silently no-ops.

### US-4: One-command deploy (P1)

`.github/workflows/pages.yml`: precompute job (real backend, HF + artifact caches) →
`VITE_DATA_MODE=static` build with base `/llm-geometry/` → deploy-pages@v5. Pages is
already enabled with `build_type: workflow`. **Acceptance**: the workflow deploys
green from `main`; the live URL serves the app; a post-deploy smoke check passes.

## Requirements

- **FR-201**: A `staticClient` implements the same interface as `dataClient`
  (feature-002 contract shapes) backed by: the TS geo engine, transformers.js,
  safetensors range reads, and precomputed JSON/bin assets. Build-time flag
  `VITE_DATA_MODE` selects it; the backend client remains the dev default.
- **FR-202**: The TS geo engine MUST be golden-vector-tested against the Python
  backend (assets exported by the real backend at build time; tolerance 1e-5).
- **FR-203**: No fabricated data anywhere: precomputed artifacts are produced by the
  real backend; anything unavailable in static mode is visibly labeled, never faked.
- **FR-204**: Weight windows fetched from HF MUST pin a commit revision and decode
  bf16 correctly; failures surface designed error states (FR-107 carries over).
- **FR-205**: The static build MUST work at the `/llm-geometry/` base path (all
  asset/API URLs via `import.meta.env.BASE_URL`), with `404.html` fallback.
- **FR-206**: Playwright specs run against the built static site (`vite preview`)
  covering US-1/2/3 acceptance; they run in CI alongside the existing suites.

## Success Criteria

- **SC-201**: Golden-vector suite green (trace, both field modes, weight presets,
  fine-tune loss trajectory) at ≤1e-5.
- **SC-202**: Static e2e suite green in CI; live URL smoke check green post-deploy.
- **SC-203**: Lighthouse-style sanity: initial static page (no model download)
  interactive < 5 s on a cold cache; geo tab fully live with zero network beyond
  the static origin.
