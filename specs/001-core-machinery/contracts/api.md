# API Contract: Core Project Machinery

HTTP/JSON contract exposed by the FastAPI backend (`code/backend/src/llm_geometry/api`).
All responses are JSON unless noted. Errors use a single envelope. Cached reads aim
for **< 100 ms** on a hit (SC-001). This contract is the source of truth for the
`contract/` tests and the frontend `dataClient.ts`.

## Error envelope (all endpoints)

```json
{ "error": { "type": "UnsupportedModelError", "message": "…", "detail": {} } }
```

- HTTP 400 — invalid parameters (e.g., negative temperature, layer out of range).
- HTTP 404 — unknown job id.
- HTTP 422 — unsupported model (not open-weights / no token probs / gated / missing).
  `type` ∈ {`UnsupportedModelError`}. **Never** a fallback model or fabricated data
  (FR-003, FR-021).
- HTTP 500 — internal compute error; `message` carries the real failure (no stub).

## GET /api/health

→ `200 { "status": "ok", "schema_version": <int> }`

## GET /api/models

Curated menu + capability summary (FR-001).

→ `200 { "models": [ { "model_id": "gpt2", "display_name": "GPT-2 (124M)",
        "source": "curated", "status": "supported",
        "capabilities": { "num_layers": 12, "hidden_size": 768, "vocab_size": 50257,
                          "exposes_token_probs": true, "exposes_hidden_states": true } },
        … ] }`

## POST /api/models/resolve

Validate/resolve an arbitrary HF id and run capability detection (FR-001/FR-002/R3).

← `{ "model_id": "distilgpt2" }`
→ `200 { ModelReference }` (status `supported`, capabilities populated, `revision` pinned)
→ `422 { error.type: "UnsupportedModelError", message }` if not loadable / not
   open-weights / no token probs / gated.

## POST /api/precompute

Request precomputation of an artifact. Returns immediately with either a cache hit or
a job to subscribe to (FR-004/FR-008/FR-009).

← `{ "artifact_type": "embeddings|distribution|reduction_2d|reduction_3d|grid",
     "model_id": "gpt2", "params": { … }, "inputs": { … } }`
→ `200 { "cache_key": "…", "status": "complete", "ready": true }`        # cache hit
→ `202 { "cache_key": "…", "job_id": "…", "status": "running", "ready": false }`  # computing
   (identical in-flight key ⇒ same `job_id`, FR-008)

## GET /api/jobs/{job_id}

→ `200 { PrecomputeJob }` (status, progress 0..1, message, error?)
→ `404` unknown id.

## GET /api/jobs/{job_id}/events  (text/event-stream, SSE)

Streams progress events until terminal (R7, FR-009):

```
event: progress
data: { "progress": 0.42, "message": "embedding layer 6/12" }

event: done
data: { "cache_key": "…" }

event: error
data: { "type": "…", "message": "…" }
```

Updates emitted **≥ 1/second** during long precomputes (SC-003).

## GET /api/distribution

Next-token distribution for a context (cached; FR-006).

Query: `model_id`, `prefix_text` (may be empty), `temperature` (`>= 0`),
optional `top_k`.
→ `200 { "model_id", "revision", "temperature", "top_token",
         "probs": [...]  |  "top": [{"token_id","token_str","prob"}], "tail_mass" }`
→ `400` if `temperature < 0`. → `422` unsupported model.

## GET /api/embeddings

Per-layer embeddings or their metadata (cached; FR-002).

Query: `model_id`, `layer` (`0..num_layers`), `source` (`static|contextual`),
optional `format=meta|full`.
→ `200 { "model_id","revision","layer","source","shape":[N,d],
         "vectors": [[...]]? }`  (`vectors` omitted when `format=meta`)
→ `400` layer out of range. → `422` unsupported model.

## GET /api/reduction/2d

2D reduced coordinates + (optional) reference grid (cached; FR-010/FR-011).

Query: `model_id`, `method` (`pca|umap`, default `pca`), `seed` (default fixed),
`grid_n` (default 25), optional `with_grid=true`.
→ `200 { "model_id","revision","method","seed",
         "coords": [[x,y]…],            // length N
         "token_ids": [...],            // aligned with coords
         "grid": { "n": 25, "vertices": [[x,y]…], "reference_token_ids": [...] }? }`

## GET /api/reduction/3d

3D spherical coordinates (cached; FR-012).

Query: `model_id`, `method` (`mds|pca3`, default `mds`), `seed` (default fixed).
→ `200 { "model_id","revision","method","seed",
         "coords": [[x,y,z]…],          // each on the unit sphere
         "token_ids": [...] }`

## GET /api/token_cloud

Full-vocabulary 2D cloud — **a dot for every token** (static-embedding PCA → density-
flattened "spread" layout). Computed once per model (cached) and the shared coordinate
space the vector-field arrows are placed in. Multi-MB; the browser fetches it once.

Query: `model_id`, `seed` (default fixed), `spread_mu` (0..1 grid-flattening, default 0.65).
→ `200 { "model_id","revision","vocab_size","seed","spread_mu",
         "coords": [[x,y]…],            // length = vocab_size (the spread layout)
         "token_ids": [...] }`          // aligned with coords
(The internal `raw`/`pca_*` projection arrays used to place arrows are NOT shipped.)

## GET /api/vector_field

Visualization 1 (FR §1). A "Drift"-style flow field: a regular `grid_n`×`grid_n` grid of
FIXED origins over the `/api/token_cloud` layout (same `seed`/`spread_mu`). Each `start` is
a grid origin and `end = start + unit·L` (uniform length); the unit direction is the local
prediction flow — nearest reference token at layer `layer_from` → its predicted next token
at layer `layer_to` — so orientations rotate as the prompt reshapes the output.

Query: `model_id`, `prefix_text`, `temperature` (>0 fans out), `layer_from`, `layer_to`,
`grid_n`, `fanout`, `seed`, `reference_set_size`, `response_text`, `response_step`.
→ `200 { …,"layer_from","layer_to","num_layers","fanout","reference_points","vocab_size",
         "seed","spread_mu","response_step",
         "starts": [[x,y]…], "ends": [[x,y]…], "probs": [...],
         "start_token_strs": [...], "end_token_strs": [...],
         "trajectory": [[x,y]…]?, "trajectory_probs": [...]?, "trajectory_token_strs": [...]? }`

## GET /api/sankey

Visualization 2 (FR §2). Particle-swarm next-token flow (hundreds–thousands of particles).

Query: `model_id`, `prefix_text`, `temperature`, `n_particles`, `n_steps`, `seed`,
`response_text`, `response_step`.
→ `200 { "n_steps","nodes":[{pos,token,count}…],"links":[{pos,source_token,target_token,value}…],
         "token_strs": { "<id>": "str" }, "per_position": [{pos, top:[{token,prob}…]}…] }`

## GET /api/manifold

Visualization 3 (FR §3). Unit sphere warped (RBF + Open3D ARAP) toward likely next tokens;
**a dot for every token** on the radius-2 sphere.

Query: `model_id`, `prefix_text`, `temperature`, `seed`, `reference_set_size` (default = full
vocab), `response_text`, `response_step`.
→ `200 { …,"n_vertices","n_faces","top_tokens":[{token_str,prob}…],"token_strs":[...],
         "vertices": [[x,y,z]…], "faces": [[i,j,k]…], "warp": [...],
         "token_points": [[x,y,z]…], "token_emis": [...], "token_ids": [...] }`

## GET /api/tokenize

Token ids + strings for a text (lets the UI animate over response tokens).

Query: `model_id`, `text`.
→ `200 { "model_id", "tokens": [{ "token": id, "token_str": "str" }…] }`

## Static frontend

`GET /` and assets serve the built Svelte bundle in production. In dev the Vite
server proxies `/api/*` to Uvicorn.

## Determinism guarantee (cross-cutting)

For identical query/body (including `seed`), every cached endpoint returns results
identical to the first computation (SC-002); the response carries the pinned
`revision` so identity is reproducible across environments (FR-013).
