# Batch-3 B4 — Geometry Lab view (feature 002, issue #1)

Touched ONLY `code/frontend/src/viz/geo/**`. `npm run check`: 501 files, 0 errors, 0 warnings.

## Components
- `GeometryLab.svelte` — orchestrator: spec→train→ready gate (missing ⇒ auto POST geoTrain + SSE epoch progress; training ⇒ attach to job_id), controls (mode/layer/temperature/top_m/antisymmetrize via geo* stores), prompt + strip, debounced(400 ms)+AbortController fetches for field/trace/tokenize (FR-108), inline designed errors + retry, spec chips (loss/coverage/entropy).
- `GeoScene.svelte` — Three.js (Manifold patterns): soft-shaded unit sphere (depthWrite on so the far hemisphere occludes), 1003 embeddings as shader Points w/ raycast hover, vocab field as ONE instanced arrow class (shaft+head InstancedMesh pair, 2 draw calls, cap 5015), sequence forces as a second amber/thicker instanced class, prompt path as geodesic polyline. Field changes lerp old→new over 300 ms in the rAF loop with preallocated Float32Arrays + scratch Matrix4/Vector3 — zero per-frame allocation; new arrows grow from their origin. sqrt(weight) color ramp keeps fan-outs readable.
- `TokenStrip / WeightLab / FinetunePanel / AttentionView.svelte` + `vocab.ts`.
- `vocab.ts`: the frozen contract has no vocab-listing endpoint, but hover needs id→text for all 1003 tokens. Table generated from the real deterministic tokenizer, VERIFIED at runtime by a real /api/geo/tokenize probe (40 words across ranks); on mismatch falls back to API-learned labels / "token #id".

## Verified live (real stack, throwaway Playwright in /private/tmp; PNGs inspected)
- Ready gate instant on cached checkpoint; sphere + next-next field render; strip shows 12 tokens.
- Force mode: amber per-position forces + green geodesic prompt path; badge "max normal residual 0.144" → antisymmetrize → "tangent: exact"; "full" layer disabled (never triggers the contract 400).
- W_V layer-0 identity preset: token minted, "edited weights active", source `preset:identity`, field visibly reshaped; reset-to-learned works.
- OOV prompt: "zorblatt/quantum" chips dashed-red `<unk>` + "9 tokens · 2 unknown"; T=1/top_m=3 fan-out renders weighted arrows.
- Fine-tune pasted text: real SGD job, SSE progress, "loss 5.55 → 3.62 on your text", predictions visibly shifted ("again" → 12%); page reload keeps edited badge (sessionStorage). Bad HF dataset id → designed inline 422 error.
- Attention heatmaps (causal, per-layer tabs) + top-10 bars all real.

## Flagged (not fixed — shared files)
- `dataClient.subscribeProgress` discards the SSE done-event payload, so a 202 fine-tune's minted weights_token is unreachable via SSE. Workaround: on done I re-POST the identical request → content-hash cache hit (200) returns token+losses. Consider passing done data to onDone.
- Dev stack: started by me via scripts/dev.sh and LEFT RUNNING for B3 (logs .devservers/).
