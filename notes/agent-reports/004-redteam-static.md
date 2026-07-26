# Red-team: the STATIC GitHub-Pages build (feature 004)

**Target:** `http://localhost:4173/llm-geometry/` — the `vite preview` of `code/frontend/dist`,
built with `VITE_DATA_MODE=static PAGES_BASE=/llm-geometry/`.
**Date:** 2026-07-26 · **Method:** Playwright (Chromium, headless), 14 scripted attack phases,
every screenshot visually inspected, network fully instrumented.
**Rule under test:** nothing may be fabricated, and nothing may silently no-op.

## Headline

The build is **substantially honest**. Zero requests to `localhost:8000` or any `/api/…` path
across all 14 runs. Every control I could find does something real. The two big claims I was
asked to verify hardest both hold: **HF-dataset fine-tuning genuinely works end to end**, and
**weight windows are genuinely read from HuggingFace by HTTP Range** — I independently
re-derived one displayed value from the real safetensors file and it matched to 7 digits.

The findings below are mostly copy that was not updated when the static build was introduced,
one third-party-CDN dependency, and one glyph that renders as tofu.

### Environment caveat — read before the model-list findings

Two things about the build under test are **not** what GitHub Pages gets:

1. `dist/static-data/index.json` has `"quick": true` and `arch_models: [gpt2]`. This dist was
   produced by `scripts/export_static_assets.py --quick` (test mode,
   `ARCH_MODELS_QUICK = ["gpt2"]`, `scripts/export_static_assets.py:49`). `.github/workflows/pages.yml:66`
   runs the exporter with **no** `--quick`, so the deployed site ships all four curated models.
   Everywhere I say "only GPT-2 was offered", that is the quick export, not a product bug.
2. **Another session was committing to this repo throughout the review.** It landed
   `b1ffba6 "Copy audit: stop claiming things that stopped being true (FR-414)"` and
   `8ea8e23 "Architecture: fix the eight defects the red team found"` while I worked, and it
   **committed my probe script** `code/frontend/redteam-static.tmp.mjs` (plus four
   `redteam-geo*.tmp.mjs`) into HEAD. I deleted mine as instructed, so `git status` now shows
   `D code/frontend/redteam-static.tmp.mjs` — **that deletion needs committing, and the four
   `redteam-geo*.tmp.mjs` files should be removed from the tree too.** All findings below were
   re-verified against HEAD `8ea8e23` after those commits landed; every one still reproduces.
3. **The dist was rebuilt out from under me twice mid-review** (00:04 and 00:16) by a concurrent
   process running plain `npm run build` — which produces a **backend-mode** bundle
   (`"/api/geo…"` strings present) at base `/`. Both times the site went blank white. I restored
   it with `VITE_DATA_MODE=static PAGES_BASE=/llm-geometry/ npx vite build` and re-ran. See
   INFO-1.

---

## Findings

### MEDIUM-1 — The Architecture header promises prompt tracing the static build cannot do

**Did:** Loaded the Architecture tab; read the header blurb; typed a free-form prompt.
**Saw:** `src/viz/arch/ArchitectureExplorer.svelte` header renders, verbatim and unconditionally:

> "A real open-weights model, **traced live** — every op in its forward pass is a clickable node.
> **Type a prompt to trace it**, ▶ play the trace through the diagram, click any block to inspect
> its actual weights, then generate a reply and hover each token for its probabilities."

Typing "Tell me about octopuses in the deep sea" does **not** trace it. It correctly shows the
designed static note instead ("Per-layer traces need the model's hidden states, which browser
ONNX exports don't expose…"). So the header sentence is false for this build, and the trace is
precomputed, not "live".
**Expected:** The same `{#if STATIC_MODE}` branch the sibling component already uses —
`src/viz/arch/ArchModelPicker.svelte:112-113` says "They run in your browser via their community
ONNX exports" in static vs "Each is traced live by the backend" otherwise. The pattern exists;
the header just wasn't updated.
**Mitigation in place:** a static-aware note directly under the prompt box does say
"the per-layer trace below is precomputed for the example prompts (per-layer tensors need the
full stack)". So a careful reader is not deceived — but the headline still overstates.
**Suspected:** `src/viz/arch/ArchitectureExplorer.svelte:271-274`. Screenshot:
`/tmp/redteam-static/01-arch-initial.png`, `/tmp/redteam-static/10-arch-freeform.png`.
**Re-verified against HEAD (8ea8e23) after the concurrent session's copy-audit commits** — still
present; only "inspect its actual weights" → "inspect it" changed.

### MEDIUM-2 — The empty-trace placeholder makes the same false promise

**Did:** Cleared the Architecture prompt box.
**Saw:** `"Type a prompt on the left — 400 ms after you stop, the model runs it and every tensor
lands here."`
**Expected:** In the static build the model does *not* run an arbitrary prompt; only the two
precomputed example prompts produce a trace. This string should be static-aware (or should point
at the example dropdown).
**Suspected:** `src/viz/arch/ArchTracePanel.svelte:204` (was `:185` when first observed; the file
grew under me). Screenshot: `/tmp/redteam-static/160-arch-empty.png`.

### MEDIUM-3 — Live generation silently depends on `cdn.jsdelivr.net`, and ships 23 MB of dead WASM

**Did:** Ran "Generate reply" with full network capture.
**Saw:** The ONNX runtime WASM is fetched from a third party:

```
GET https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0-dev.20260416-b7804b056c/dist/ort-wasm-simd-threaded.asyncify.wasm
GET https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0-dev.20260416-b7804b056c/dist/ort-wasm-simd-threaded.asyncify.mjs
```

Meanwhile Vite emits `dist/assets/ort-wasm-simd-threaded.asyncify-DMmc6YqF.wasm` — **23,567,050
bytes**, 30% of the whole 79 MB `dist/` — which the runtime never loads, because
`env.backends.onnx.wasm.wasmPaths` is never set and ORT's default is the jsdelivr URL (the
literal ``cdn.jsdelivr.net/npm/onnxruntime-web@${…}/dist/`` is in the shipped
`transformersRuntime` chunk).
**Expected:** Either point `wasmPaths` at the self-hosted copy (removing the third-party
dependency *and* making the 23 MB useful), or don't ship the 23 MB. Also: the version is a
`-dev.20260416` prerelease pinned by transformers.js — if jsdelivr is blocked or that version is
unpublished, generation dies with a load error and no other explanation.
**Suspected:** `src/lib/staticClient/transformersRuntime.ts` (no `env` configuration anywhere in
the module).

### MEDIUM-4 — Generation runs an **unpinned** third-party ONNX repo, and the `revision` argument is silently discarded

**Did:** Watched the model download.
**Saw:** `GET https://huggingface.co/onnx-community/gpt2-ONNX/resolve/**main**/onnx/model_quantized.onnx`
— `main`, not a commit. `src/lib/staticClient/arch.ts:356` calls `rt.generate(body, repo, m.revision)`,
but `transformersRuntime.ts:174` names the parameter `_revision` and never uses it; the
`pipeline(...)` call passes only `{dtype, device}`.
**Expected:** Everything else in this build is scrupulously pinned — tokenizers use
`AutoTokenizer.from_pretrained(modelId, { revision })` and the weight inspector reads
`…/gpt2/resolve/607a30d783dfa663caf39e06633721c8d4cfcd7e/model.safetensors`. The generation path
is the one place where the bytes executed in the user's browser can change under you without a
code change. A parameter that is passed and then thrown away is also exactly the "silent no-op"
the project rule forbids.
**Note:** nothing is *fabricated* here — the logits are real. This is a reproducibility hole, not
a lie.
**Suspected:** `src/lib/staticClient/transformersRuntime.ts:174` (`_revision`) and `:84-87`.

### MEDIUM-5 — Stale doc comment claims two limitations that no longer exist

**Did:** Audited every "this build can't" message.
**Saw:** `src/lib/staticClient/errors.ts:5-7`:

> "anything the static site cannot compute for real — arbitrary prompts on **the 001 views**,
> **HF-dataset fine-tunes**, non-curated models — is refused…"

Both named limitations are dead: HF-dataset fine-tunes **work** (verified below, real TinyStories
rows), and the 001 views were removed by feature 004.
**Expected:** The comment to match `src/lib/staticClient/geo.ts:214-216`, which correctly says
"HuggingFace datasets DO work here (feature 004)".
**Severity note:** this string is **not user-visible** — it is a source comment. It is here
because "a note claiming a limitation that no longer exists" was explicitly in scope, and because
it is the kind of comment a future maintainer will trust.
**Suspected:** `src/lib/staticClient/errors.ts:5-7`.

### LOW-1 — Save/Load buttons render a tofu box instead of an arrow

**Did:** 3× device-scale crop of the Geometry Lab's model I/O row.
**Saw:** `▯ Save model` / `▯ Load model`. Codepoints confirmed: **U+2B73** (⭳) and **U+2B71** (⭱).
Neither is covered by the app's mono font stack, so Chromium draws `.notdef`. Reproduces at 1440px
and at 390px.
**Expected:** A glyph that renders, or an inline SVG.
**Suspected:** `src/viz/geo/TrainPanel.svelte:279-280`. Screenshot:
`/tmp/redteam-static/150-saveload-buttons.png`.

### LOW-2 — The fine-tune loss chip says "on your text" even when the source was a HF dataset

**Did:** Fine-tuned on `roneneldan/TinyStories` via the HF-dataset tab.
**Saw:** `loss 4.95 → 4.72 **on your text**` — but no text of mine was involved. The
active-model chip right above it gets it right: `active model: fine-tuned on roneneldan/TinyStories`.
**Expected:** The chip to branch like `FinetunePanel.svelte:94-97` already does for the model note.
**Suspected:** `src/viz/geo/FinetunePanel.svelte:158` (`"on your text"` hardcoded in the template).

### LOW-3 — A 4096-cell weight window costs 128 HTTP requests

**Did:** Zoomed the inspector into an exact 64×64 window of `transformer.h.0.attn.c_attn.weight`.
**Saw:** 66 requests to `huggingface.co` + 66 signed-CDN redirects to `us.aws.cdn.hf.co` — one
`Range` request per matrix row (`bytes=273778127-273778382`, etc.), each incurring a redirect.
Took ~8 s.
**Expected:** Contiguous rows could be coalesced into one range read per window (or a small
number), cutting this to single digits. Not a correctness issue — the data is right — but it is
128 round trips per interaction on a public CDN.
**Suspected:** `src/lib/staticClient/safetensors.ts` `readWindow`.

### LOW-4 — A 401 reaches the browser console on a bad dataset id

**Did:** Fine-tuned with dataset id `definitely/not-a-real-dataset-xyzzy`.
**Saw:** Console: `[error] Failed to load resource: the server responded with a status of 401 ()`.
**Expected:** The *app* behaves perfectly — it surfaces
`"That dataset didn't work: HuggingFace dataset service: The dataset does not exist, or is not
accessible without authentication (private or gated)…"`. The console error is Chromium logging the
HTTP status itself and is not suppressible from JS. Recorded only because "any console error is a
finding"; I do not think it is actionable.

### INFO-1 — Default `npm run build` produces a backend-mode bundle at base `/`

**Did:** Observed the served site go blank white twice mid-review; diffed `dist/index.html`.
**Saw:** `<script src="/assets/index-_9b4WkWu.js">` (no `/llm-geometry/` prefix) → 404 → empty
`<body>`, and `grep -c '"/api/geo' dist/…js` = 1, i.e. the bundle would call a backend that does
not exist on Pages.
**Expected:** This is *correct by design* — `vite.config.ts:11` defaults `base` to `/` and
`pages.yml:72-73` sets both env vars explicitly, so CI is fine. Flagging only because the failure
mode is silent and total (white page, two 404s, no error text), and `package.json` has a
`preview:static` script that does it right while bare `build` is the obvious thing to type.

### INFO-2 — Local `public/static-data/` carries 12 MB of dead feature-003 preset assets

`dist/static-data/presets/{manifold,sankey,vector}` + `token_cloud.json` (9.0 MB alone) are for
the three views feature 004 removed; the current exporter never writes them. They are **not** in
the deployed build — `.gitignore:156` ignores `code/frontend/public/static-data/`, so CI starts
clean. Local-only cruft; delete `code/frontend/public/static-data/presets/` when convenient.

---

## Network: every host the page contacted (aggregate, 14 runs)

| Host | Requests | What for |
|-|-|-|
| `huggingface.co` | 77 | safetensors headers + per-row `Range` reads at the **pinned** gpt2 revision; tokenizer/config files; `onnx-community/gpt2-ONNX` model files |
| `us.aws.cdn.hf.co` | 67 | signed xet-bridge redirects for those same `Range` reads |
| `localhost:4173` | 57 | the app itself + `static-data/{index,arch/gpt2/*,geo/*}.json` |
| `datasets-server.huggingface.co` | 7 | `/splits` + `/rows` for `roneneldan/TinyStories` (fine-tune + scratch training) |
| `cdn.jsdelivr.net` | 2 | ONNX runtime WASM — see MEDIUM-3 |

**`localhost:8000`: 0. Same-origin `/api/…`: 0.** Verified by aggregating every recorded request
across all phases.

---

## Verified working (I did this and watched it happen)

**No hidden backend**
- Cold load, both tabs, every panel: only `localhost:4173` + `static-data/*.json`. No `/api/`, no `:8000`.

**Architecture Explorer**
- Real in-browser generation (transformers.js, gpt2 ONNX, wasm·q8 after WebGPU probe fails):
  26 s, coherent GPT-2 continuation — *"now the world's largest producer of food, with a total
  market value exceeding $1.5 billion (€2.3bn)."* The runtime badge honestly degrades from
  `in-browser · model loads on first use` → `in-browser · wasm · q8`, tooltip
  *"real logits, no server"*.
- Per-token hover shows real distributions: `p = 100.0% · top-5: "'s" 92.3% · " capital" 3.0% · " leader" 0.7% · "-" 0.3% · " number" 0.2%`.
- **Weight windows are real, verified independently.** The inspector showed
  `row 304 · col 926 · -0.0838719` for `transformer.h.0.attn.c_attn`. I fetched
  `https://huggingface.co/gpt2/resolve/607a30d783dfa663caf39e06633721c8d4cfcd7e/model.safetensors`
  myself, parsed the header (`h.0.attn.c_attn.weight F32 [768,2304] offsets [270682112,277760000]`),
  computed the byte offset and read it: **-0.08387192338705063**. Match.
- Over-budget windows honestly label themselves `50257 × 768 · overview (whole tensor,
  downsampled)`; exact windows say `rows 273–336 · cols 895–958 of 768 × 2304 · exact values`.
- Free-form prompt → the designed `StaticNotice`, not an error and not a blank panel. It names
  both available example prompts by label *and* by exact prompt text.
- Example-prompt dropdown works; the "Greeting (with system prompt)" preset populates the system
  prompt field and loads a different trace.
- Trace playback: ▶ animates the diagram (pixels change); scrub slider seeks; speed selector is
  real — 0.5× advanced 21 ops in 3 s, 4× advanced 148 ops in 3 s.
- **All 12 attention heads visible at once** (12 tiles rendered); clicking a tile enlarges it;
  the layer slider changes both the head grid and the residual chart (verified by pixel diff at
  L0/L5/L9/L11, not by DOM diff — they are canvases).
- "follow playhead" is honest in both states: unchecked, the layer label stays at `layer 0 / 11`
  while playing; checked, it walks to `layer 9 / 11`.
- Diagram: scroll-zoom works, drag-pan works, layer headers expand (17 → 29 nodes), Esc closes
  the inspector. Ops with no weights say so ("This op has no learned weights…"). Tied weights
  labelled `tied → transformer.wte.weight`.
- Empty prompt disables Generate rather than failing.

**Geometry Lab**
- Sphere renders; **spin is OFF by default** (`aria-pressed=false`, and two frames 2.5 s apart are
  byte-identical). Toggling it on animates.
- **Force-mode occlusion is correct from the back.** Rotated 180°: the orange aggregate-force
  arrows and green prompt-token dots that dominate the front view are hidden behind the sphere,
  with only a limb sliver visible. (`53-geo-force.png` vs `54-geo-force-back.png`.)
- Force-mode badges are precise and self-critical: *"per-token field: exactly tangent"* and
  *"radial pull projected out: 0.144 max"*, the latter with a tooltip explaining why
  antisymmetrizing W_V does not reduce it.
- **The cited paper is real.** `arXiv:2607.13295` → "On Transformer Dynamics" (Mohammad Javad
  Latifi Jebelli), abstract confirms attention-as-two-body-interaction on a Riemannian manifold.
  HTTP 200. Grounded in `specs/002-interactive-model-explorer/spec.md:54-55`. (The tooltip's
  shorthand "W_V·z" is a simplification of the spec's `Σ softmax(⟨K z_j, Q z_i⟩)·V z_j`; I could
  not verify the exact attribution from the abstract alone.)
- Every view control moves the render (pixel-verified): field mode, layer (full/0/1/2/3),
  temperature 0→1.2, arrows/point 1→4, prompt text.
- Hover a dot → `"the" · token 5`.
- Tokenizer honesty: `quantum chromodynamics zzzyx` → `3 tokens · 3 unknown` with visible `<unk>`
  chips; a 57-word prompt shows `⋯ truncated at 50 · 50 tokens · 3 unknown`.
- **Weight lab:** presets apply and mint a token (`active model: hand-edited weights · be87fda2`);
  the seed selector appears only for seeded presets with the honest tooltip *"the static demo ships
  real backend-computed matrices for these numpy seeds — other seeds need the full stack"*;
  cell editing works (clicked, editor pre-filled `0.00028039`, typed `3.75`, sphere moved, badge
  became `edited`); "reset to learned" restores `shipped checkpoint`; the embedding matrix is
  correctly read-only with a scrollable 1003-column ribbon.
- **Edits really reach the model.** Applying `W_V=zero` L0 shifted the next-token distribution
  (7.1% → 7.2%). Then `W_Q=zero` and `W_O=zero` on the same layer produced *identical* output —
  which is exactly right, since zeroing W_V already kills that layer's value path; a later
  `W_V=random` L1 moved it again (7.4%). Consistent physics, not a stub.
- **Fine-tune, pasted text:** 4.0 s, `loss 5.27 → 4.63`, sphere moved, chip
  `fine-tuned on your text · 592d8cdd`.
- **Fine-tune, HuggingFace dataset — WORKS, end to end.** Entered `roneneldan/TinyStories`; the
  page called the public dataset viewer:
  `GET /splits?dataset=roneneldan%2FTinyStories` then four
  `GET /rows?…&config=default&split=train&offset=0|100|200|300&length=100`. Live progress
  `step 30/60 · loss 4.75` → `step 60/60 · loss 4.80`, final `loss 4.95 → 4.72`, chip
  `fine-tuned on roneneldan/TinyStories · 4cfab858`. 14.8 s. **No note anywhere claims this is
  disabled** — the old refusal is gone from the UI.
- Bad dataset id → plain-language refusal naming the real cause (private/gated/misspelled).
- **Train from scratch:** the gate is honest — a short paste shows
  `11 tokens · 10 distinct words — needs 1,000 distinct to fill the vocabulary, so this is 990
  short` and the button is disabled. A real run on TinyStories took **221 s**, streamed
  `reading roneneldan/TinyStories · train · rows 1000–1100` → `building a vocabulary from your
  text` → `epoch 1/2 · loss 5.08` → `epoch 2/2 · loss 4.81`, and produced
  `trained a new model · final loss 4.80 · 392,054 tokens · 2 epochs — it is now the active model,
  with its own vocabulary`. The prediction panel changed completely (`. 12.2% / <unk> 8.6% /
  , 7.1%` vs the shipped model's `, 7.1% / the 5.0%`), confirming a genuinely different model.
- **Save → load in a FRESH browser:** `geotransformer-5f0e302a.llmgeo.json`, 31,169 bytes,
  `{format, version, weights_token, config, vocab, weights}`. A brand-new page (fresh profile)
  started at `shipped checkpoint`, then after loading showed
  `active model: loaded from model.json · 5f0e302a` and `loaded model.json · 1003-token vocabulary`.
- **Tampered files are refused — all six modes**, with messages that name the actual defect, and
  the active model stays the shipped checkpoint every time:

  | Tamper | Response |
  |-|-|
  | flipped bytes in `layers.0.W_Q` | *"this model file is corrupt: its weights hash to 59eb4b21… but it declares 5f0e302a…. Loading it would pair the wrong vocabulary with these weights, so it is refused."* |
  | `format` changed | *"not a Geometry Lab model file (format="something/else", expected "llm-geometry/geo-model")"* |
  | deleted `layers.2.W_V` | *"Weight set mismatch (missing: layers.2.W_V, extra: none)"* |
  | `version: 99` | *"model file version 99 is not supported (this build reads version 1)"* |
  | `shape` → `[4,4]` | *"weight "layers.0.W_Q" has 9 values but shape [4, 4] needs 16"* |
  | non-JSON | *"junk.json is not valid JSON."* |

- **Reload persistence is honest.** Edited a W_V cell (token `e552db16`), reloaded: the page came
  back with `hand-edited weights active`, the same token `e552db16`, and the edited cell still
  bright in the heatmap. It does **not** silently revert to the shipped checkpoint while claiming
  to be edited. (Backed by `sessionStorage` with a hash-checked restore that drops mismatches —
  `src/lib/staticClient/geo.ts:48-90`.)
- Export PNG produced a real 475,089-byte `geometry-sphere.png`.
- Attention layer tabs (0–3) change the heatmap.
- Empty fine-tune text → *"Paste some text to fine-tune on first."* (not a silent no-op).

**Claims audit — checked and TRUE in this build**
- Masthead badge: *"static demo — computations run in your browser"*, tooltip *"This is the static
  demo build: everything shown is computed live in your browser or precomputed by the real
  backend — nothing is faked. Click for instructions to run the full stack (Python backend +
  frontend) locally."* — accurate for everything I exercised.
- *"a new checkpoint is minted — the learned one is never touched"* — true; "reset to learned"
  always returns to `shipped checkpoint`.
- *"a fresh vocabulary + fresh weights — not a fine-tune of the shipped one"* — true; the scratch
  model has its own 1003-token vocabulary and a different distribution.
- *"Models live in this browser session — save one to keep it."* — true (sessionStorage).
- *"Read live from the Hugging Face dataset viewer — real rows, no download."* — true, verified
  against the network log.
- *"issue #4 tracks widening the list"* — issue #4 is OPEN, titled "Architecture Explorer: expand
  model support beyond the curated list", and its body matches.
- Geo header chips (`final loss 4.89`, `coverage 0.90`, `field entropy 2.81`,
  `corpus gutenberg-11-alice-in-wonderland`) exactly match `static-data/geo/spec.json`
  (`4.88521`, `0.90055`, `2.81215`) — displayed, not invented.
- The exporter's `ARCH_MODELS_FULL` matches the backend's `CURATED_MODELS` exactly, so the
  deployed menu will not diverge between builds.

**Links (both, unauthenticated `curl`)**

| href | Status |
|-|-|
| `https://github.com/ContextLab/llm-geometry#quickstart` | 200 (README has a `## Quickstart` heading — anchor resolves) |
| `https://github.com/ContextLab/llm-geometry/issues/4` | 200 |

**390 px viewport**
`documentElement.scrollWidth === clientWidth === 390` on both tabs, before and after switching —
**zero horizontal page overflow**. Zero controls positioned outside the viewport on either tab.
The one element wider than the viewport is the attention heatmap canvas (right=487), which lives
inside its own `overflow-x: auto` container — the page body does not scroll. Layout screenshots
`130-narrow-arch.png` / `131-narrow-geo.png` are clean.

**Console / errors**
Across all phases: **0 page errors, 0 unhandled rejections, 0 failed requests, 0 non-2xx
responses** — except the deliberate 401 from the bogus dataset id (LOW-4) and the benign
`GL Driver Message … GPU stall due to ReadPixels` performance warnings emitted by Chromium's
WebGL layer during screenshotting.

---

## Not covered / caveats

- Only **GPT-2** could be exercised in the Architecture tab (quick export). SmolLM2-135M/360M and
  Qwen2.5-0.5B ship graphs, tiles and traces on disk but are absent from `index.json`, so their
  live-chat and weight paths were untested here.
- WebGPU was unavailable in headless Chromium (`No available adapters.`), so only the
  `wasm · q8` rung of the device ladder was exercised. The badge reported the fallback honestly.
- I reviewed only; no source file was modified. I did rebuild `dist/` twice (a build artifact) to
  undo the concurrent clobbering described in INFO-1.
- Artifacts: `/tmp/redteam-static/` (screenshots, per-phase network logs, `all-strings.json`,
  `links.json`, saved + tampered model files). The probe script
  `code/frontend/redteam-static.tmp.mjs` was deleted.
