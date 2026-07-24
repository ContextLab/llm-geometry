# Session 2026-06-15 — "a dot for every vocabulary token"

## Done (committed f57eb89 on `001-core-machinery`, pushed)
Addressed feedback: "for the vector field and manifold figures, there should be a dot
for *every* token in the vocabulary; these representations look way too sparse."

- **`token_cloud` artifact** (`compute/token_cloud.py`): full static-embedding matrix →
  one PCA → density-flattened spread, cached once per model. `GET /api/token_cloud`
  ships only warped coords + token_ids (~7.5MB for Qwen 151,936). Internal `raw`/`pca_*`
  arrays stay server-side.
- **`reduce.spread.warp_like`**: kNN inverse-distance displacement map. Places the
  vector field's contextual layer-n/m arrows into the cloud's spread layout without a
  per-request global spread.
- **vector_field**: projects arrows through the cloud PCA + `warp_like`; `spread_mu`
  0.85→0.65 (arrows stay locally coherent). Returns `seed`/`spread_mu`/`vocab_size`.
- **Frontend VectorField.svelte**: ~150k dots on a `<canvas>` under the SVG arrows
  (shared scale from cloud extent), arrowheads (SVG2 context-stroke), prob² emphasis,
  nearest-dot quadtree hover. Fetches cloud once via `client.getTokenCloud` (memoized)
  using the field's exact seed/spread_mu.
- **manifold**: already full-vocab (reference_set_size None); hover falls back to
  `token #id` when per-token strings too many; point size shrinks at >20k points.

## Verified (real Qwen2.5-0.5B, Playwright)
All three views render; layer range, temperature fan-out, response trajectory + ▶ Play
(0→5 step animation), hover tooltips ("cour → ting · 97%"). Manifold warp now clearly
visible (RBF width 0.8) bulging toward predicted tokens.
Backend **60 passed**, e2e **9 passed**, svelte-check clean, frontend unit **5 passed**.

## Run locally
- Backend: `cd code/backend && . .venv/bin/activate && python -m uvicorn llm_geometry.api.app:app --port 8000`
- Frontend: `cd code/frontend && npm run dev` → http://localhost:5173
- NOTE: a stale backend bound to `127.0.0.1:8000` once shadowed a new `0.0.0.0:8000`
  one for `localhost` requests — if routes look missing, `lsof -iTCP:8000` and kill duplicates.

## Perf notes
- token_cloud (Qwen 151k): ~4.8s first compute (cached after).
- per-step manifold recompute (warm): ~1.2s; 3D reduction is cheap.
- Possible future polish: ship manifold `token_points` once (like the cloud) so Play
  doesn't re-download ~6MB/step; precompute all response steps for smoother animation.
