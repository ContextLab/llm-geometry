# Static-site (GitHub Pages) hosting research for llm-geometry demos

Date: 2026-07-24. All claims below verified against live endpoints, official docs, or
primary source code (URLs inline). Curl transcripts were run locally from this machine.

## Q1 — transformers.js (latest: v4.2.0, published 2026-04-22)

npm dist-tags (verified via `curl https://registry.npmjs.org/@huggingface/transformers`):
`{'next': '4.0.0-next.11', 'latest': '4.2.0'}` — the v4 line is current; repo is now a
monorepo (`packages/transformers/...` on `main` of huggingface/transformers.js).

### (a) Per-layer attentions / hidden states: NO for standard exports (verified on the actual ONNX graph)

I downloaded `onnx-community/SmolLM2-135M-Instruct-ONNX/onnx/model_q4f16.onnx` (117,266,133
bytes) and parsed it with the `onnx` Python package. The full graph I/O:

- **Inputs:** `input_ids`, `attention_mask`, `position_ids`, `past_key_values.{0..29}.{key,value}`
  (shape `[batch, 3, past_seq, 64]` — 30 layers, GQA with 3 KV heads, head_dim 64)
- **Outputs:** `logits [batch, seq, 49152]` + `present.{0..29}.{key,value}` — **and nothing else.**
  No `attentions`, no `hidden_states`, no `last_hidden_state`.

Note: the correct repo id is `onnx-community/SmolLM2-135M-Instruct-ONNX` (with the `-ONNX`
suffix). `onnx-community/SmolLM2-135M-Instruct` does not exist — the HF API returns a
misleading `{"error":"Invalid username or password."}` for nonexistent repos. Sibling repos
`...-ONNX-GQA` and `...-ONNX-MHA` also exist.

transformers.js source confirms attentions are only surfaced when the *export* contains
them. `packages/transformers/src/models/modeling_utils.js` (`getAttentions`, line ~1222)
just scans forward-output names for `cross_attentions*/encoder_attentions*/decoder_attentions*`.
`models/whisper/modeling_whisper.js` (~line 388) throws:
> "Model outputs must contain cross attentions to extract timestamps. This is most likely
> because the model was not exported with `output_attentions=True`."

`GenerationConfig` does define `output_attentions` / `output_hidden_states` flags
(`src/generation/configuration_utils.js` lines 307/315), but they only pick up outputs the
graph actually produces. So for stock onnx-community text-gen exports:

- **Attentions: not available.** Would require a custom Optimum re-export with attention
  outputs (as done for Whisper `_with_attentions` exports) hosted under our own HF repo.
- **Hidden states: not available** (not even `last_hidden_state`).
- **What IS available per layer:** the KV cache (`present.N.key/value`) — i.e. each layer's
  post-`k_proj`/`v_proj` states (3 heads × 64 dims per token). Real per-layer signal, but
  not the residual stream and not enough to reconstruct attention (queries are not output).
- Full `logits` for every position are available (call the model directly instead of
  `generate()`), so next-token distributions at every position are easy.

### (b) WebGPU vs WASM, quantization, memory

API (verified from https://huggingface.co/blog/transformersjs-v3, unchanged in v4):

```js
const generator = await pipeline("text-generation",
  "onnx-community/SmolLM2-135M-Instruct-ONNX",
  { dtype: "q4f16", device: "webgpu" });   // device: "webgpu" | "wasm"
```

`dtype` options: `"fp32" | "fp16" | "q8" | "int8" | "uint8" | "q4" | "bnb4" | "q4f16"`;
per-module dtype maps are supported. Available files for SmolLM2-135M-Instruct-ONNX
(exact sizes from the HF tree API):

| file | bytes |
|-|-|
| model.onnx (fp32) | 538,868,107 |
| model_fp16.onnx | 269,851,815 |
| model_q4.onnx | 180,581,125 |
| model_bnb4.onnx | 173,947,195 |
| model_int8/quantized.onnx | 135,658,354 |
| model_q4f16.onnx | 117,266,133 |

Memory back-of-envelope (estimate, not measured): weights buffer ≈ file size; KV cache is
fp32 in this export → 30 layers × 2 × 3 heads × 64 × 4 B ≈ 45 KB/token (~92 MB at 2048 ctx);
logits 49,152 × 4 B ≈ 197 KB per output position. Practical peak for q4f16 + WebGPU:
roughly 300–500 MB GPU+JS heap; fp32-on-WASM (539 MB download) should be avoided.
`q4f16`/`fp16` need the WebGPU `shader-f16` feature. WebGPU ships by default in Chrome 113+,
Edge, Safari 26+ (mid-2025), Firefox 141+ (Windows) / 145+ (Apple Silicon); keep
`device: "wasm"` + `dtype: "q8"` as fallback (gpuweb wiki: Implementation Status).

### (c) Tokenization-only: YES, cheap

`AutoTokenizer.from_pretrained` loads **only** `tokenizer.json` + `tokenizer_config.json`
(verified: `src/utils/model_registry/get_tokenizer_files.js` returns exactly
`['tokenizer.json', 'tokenizer_config.json']`). For SmolLM2 that is 3,522,656 B + 3,794 B
(~3.5 MB, gzip-compressible) — no ONNX/model download. Works fully offline from our own
Pages origin if we vendor the two files and point `env.localModelPath` at them.

## Q2 — HF Hub raw-weight access from browsers: YES (CORS + Range verified end-to-end)

`curl -sIL -H "Origin: https://contextlab.github.io" -H "Range: bytes=0-99"
https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main/model.safetensors`

Hop 1 (huggingface.co, HTTP 302):
```
access-control-allow-origin: https://contextlab.github.io
access-control-expose-headers: ...,ETag,Link,Accept-Ranges,Content-Range,X-Linked-Size,...
accept-ranges: bytes
vary: Origin, Accept
x-linked-size: 269060552
ratelimit-policy: "fixed window";"resolvers";q=3000;w=300   # 3000 req / 5 min
```
Hop 2 (us.aws.cdn.hf.co via CloudFront, HTTP **206 Partial Content**):
```
access-control-allow-origin: *
access-control-expose-headers: *
accept-ranges: bytes
content-range: bytes 0-99/269060552
```

Proven workflow (executed locally, exact bytes):
1. `Range: bytes=0-7` → little-endian u64 header length = **30528**.
2. `Range: bytes=8-30535` → JSON header: 272 tensors, `__metadata__ {'format':'pt'}`.
3. Header gives `model.layers.0.self_attn.q_proj.weight: dtype=BF16 shape=[576,576]
   data_offsets=[62818560, 63482112)`; fetched `Range: bytes=62849096-63512647`
   (8 + 30528 + offsets) → exactly **663,552 bytes** = 576×576×2 (BF16). 
4. Gotcha: SmolLM2 safetensors are **BF16** — browser needs a bf16→f32 conversion
   (`u16 << 16` into a Float32Array bit pattern); there is no `BFloat16Array` in JS.

So a static page can lazily stream exact weight windows for any open-weights model with
plain `fetch(url, {headers: {Range: ...}})`. `X-Linked-Size`/`ETag` are CORS-exposed for
integrity/versioning; pin a revision (`/resolve/<commit>/`) for reproducibility.

## Q3 — GitHub Pages deploy for a Vite SPA in `code/frontend/`

Official flow (Vite guide https://vite.dev/guide/static-deploy + actions repos):
`actions/checkout@v7` → `actions/setup-node@v6` → build → `actions/configure-pages@v6` →
`actions/upload-pages-artifact@v5` (`path: code/frontend/dist`) → `actions/deploy-pages@v5`,
with `permissions: {pages: write, id-token: write}`, `environment: github-pages`, and
Pages "Source: GitHub Actions" enabled in repo settings. For the subdirectory, set
`defaults.run.working-directory: code/frontend` (or `working-directory` per step).

- **Base path:** project page at `https://contextlab.github.io/llm-geometry/` requires
  `base: '/llm-geometry/'` in `vite.config.ts` (Vite docs: "set `base` to `'/<REPO>/'`").
  Use `import.meta.env.BASE_URL` for runtime asset URLs.
- **Limits** (docs.github.com "GitHub Pages limits" + upload-pages-artifact README):
  published site ≤ **1 GB** (official; 10 GB absolute cap where "Pages will not even
  attempt to deploy"), soft **100 GB/month** bandwidth, deploy timeout **10 min**,
  artifact tarball must contain no symlinks/hard links. The git 100 MB per-file limit only
  applies to files committed to the repo, not to Actions-built artifacts — but we stay far
  below both. Model weights come from the HF CDN, so Pages only serves code + small data.
- **Serving behavior** (verified live): Pages gzips text types —
  `pages.github.com/versions.json` → `content-type: application/json; charset=utf-8` +
  `content-encoding: gzip`; same for `.js`/`.css` on `huggingface.github.io`. `cache-control:
  max-age=600` is fixed (fine; content-hash filenames make it a non-issue). `accept-ranges:
  bytes` present. Expect `.bin` to be served as `application/octet-stream` *without* gzip
  (Fastly compresses only compressible types) — so ship float data as `.bin` (already
  compact) or as gzip-friendly JSON; verify once after first deploy. SPA deep links: either
  use hash routing or copy `index.html` → `404.html` in the build.

## Q4 — Precompute-and-ship sizes (computed)

| artifact | size |
|-|-|
| geo checkpoint (1003×3 + 4×3×3 + 50×3 = 3,195 floats) as JSON | 66 KB full-repr; **31 KB** at 5 decimals; ~13 KB as f32 .bin |
| arch graph (~424 nodes / 483 edges) JSON | ~90–150 KB |
| 64×64 tiles × 200 nodes | f32 .bin **3.3 MB**; uint8 (per-tile min/max scale) **0.82 MB**; JSON ~6.5 MB (avoid) |

Everything is orders of magnitude under the 1 GB site cap. Recommendation: JSON for
checkpoint+graph (gzipped by Pages), uint8-quantized `.bin` + tiny JSON manifest for tiles.

## Q5 — In-browser fine-tuning of the tiny model: trivially interactive

d_model=3, 4 layers, vocab 1003, 50-token window. Params: core ≈ 432 (4 layers ×
[QKVO 36 + MLP(4×) 72]) + embed/head ≈ 3,009 each. Forward ≈ 2·params·T + attention
quadratic ≈ **470 K FLOPs/window** (logit layer 2·50·3·1003 ≈ 301 K dominates); fwd+bwd ≈ 3×
forward ≈ 1.41 MFLOPs/SGD step → **100 steps ≈ 141 MFLOPs**. At a pessimistic 50 MFLOPS
scalar JS: 2.8 s; at typical Float32Array JS (200 MF–1 GF): **0.14–0.7 s**. Interactive even
without workers; a Web Worker keeps the UI clean anyway.

Libraries: scalar autograds (micrograd ports) build ~millions of graph nodes for a
1003-wide softmax × 50 positions — too slow. TF.js supports training but is a heavyweight
dep and its WebGL dispatch overhead dominates at these tensor sizes. **Recommendation:
hand-rolled tensor-level forward+backward in plain TypeScript with Float32Array
(~300 lines: matmul, layernorm, softmax-xent, Adam).** No library needed.

## Recommended architecture

**Live in-browser (static page, no backend):**
- transformers.js v4 (`@huggingface/transformers`), `device: "webgpu"`, `dtype: "q4f16"`
  (117 MB from HF CDN, cached by the browser/Cache API); fallback `wasm` + `q8` (136 MB).
- Full next-token distributions (call model forward directly for per-position logits) and
  per-layer KV-cache readouts for SmolLM2-135M.
- Tokenization-only mode: vendor `tokenizer.json`(+config) on our Pages origin (~3.5 MB) —
  instant, no model download.
- Exact weight-window inspector: range-reads against
  `huggingface.co/.../resolve/<pinned-commit>/model.safetensors` (CORS `*`, 206 verified);
  bf16→f32 decode in JS; cache the 30 KB header once.
- Tiny-model training/fine-tuning demo: hand-rolled TS autograd in a Web Worker
  (~1.4 MFLOPs/step; 100 steps « 1 s).

**Precomputed at build time (CI, Python backend as a build tool):**
- geo checkpoint JSON (~31 KB), arch graph JSON (~100 KB), 64×64 weight tiles as uint8
  `.bin` + manifest (~0.9 MB) — emitted into `code/frontend/public/` (under `base`).
- Anything needing per-layer **hidden states or attention matrices** (vector-field/manifold
  reductions): precompute with the existing `llm_geometry` backend — the browser ONNX
  export does not expose them.

**Degradation ladder:** WebGPU q4f16 → WASM q8 → tokenizer-only + precomputed
distributions (page stays fully functional from static JSON if the model can't load).

**Deploy:** single workflow — backend precompute job (optional, cached) → `npm run build`
with `base: '/llm-geometry/'` → configure-pages@v6 → upload-pages-artifact@v5 →
deploy-pages@v5.

### Key sources
- https://huggingface.co/onnx-community/SmolLM2-135M-Instruct-ONNX (graph parsed locally)
- https://huggingface.co/blog/transformersjs-v3 (device/dtype API)
- https://github.com/huggingface/transformers.js — `packages/transformers/src/{models/modeling_utils.js, models/whisper/modeling_whisper.js, generation/configuration_utils.js, tokenization_utils.js, utils/model_registry/get_tokenizer_files.js}`
- https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main/model.safetensors (curl CORS/Range transcripts above)
- https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits
- https://github.com/actions/upload-pages-artifact ; https://vite.dev/guide/static-deploy
- https://registry.npmjs.org/@huggingface/transformers (v4.2.0)
- https://github.com/gpuweb/gpuweb/wiki/Implementation-Status (WebGPU availability)
