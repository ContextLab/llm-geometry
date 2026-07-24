# Red-team: Manifold tab (2026-07-24)

Adversarial audit of the **Manifold** view (`code/frontend/src/viz/Manifold.svelte`,
`code/frontend/src/controls/ManifoldControls.svelte`, backend
`code/backend/src/llm_geometry/compute/manifold.py`). Method: real Chromium
(Playwright, `--enable-unsafe-swiftshader`) driving http://localhost:5173; every state
screenshotted to `/private/tmp/redteam-manifold/NN-desc.png` and visually inspected;
backend claims cross-checked against live `:8000` API responses. Stack left running.

## Findings

### 1. CRITICAL — The manifold bulges toward the wrong tokens (scientific correctness)
With prompt "The capital of France is" (Qwen2.5-0.5B-Instruct, T=1.0) the true
next-token distribution (`/api/distribution`) is ` Paris` **30.2%**. The manifold
caption instead reads **"bulging toward: having, G, !, home, noted"** and its
`top_tokens` list ` having` with prob **1.0**. Root cause chain:
- Frontend hard-codes `MARKERS = 2000` (`Manifold.svelte:26`) with the comment
  "the warp only uses the top tokens anyway" — false.
- `printable_reference_ids(lm, 2000)` (`compute/printable.py:41-48`) picks 2000 tokens
  **evenly spaced across the ~150k vocab, independent of the distribution** — ` Paris`
  is simply absent from the marker set (verified: not in `token_strs`).
- `manifold.py:138-140`: `emis = probs_full[token_ids]; emis = emis / emis.max()` —
  emissions are max-normalized **within the arbitrary subset**, so a ≈0.1%-probability
  token renders as "emission 100.0%". `warp_top=48` then warps toward the top of this
  subset, and the surface flow field seeds from the same wrong tokens.
- Reproduces on distilgpt2: caption "bulging toward: celebrating, taking, closed, its, of".
Every semantic promise of the view ("Bulges = high emission probability") is untrue at
the default settings. Evidence: `01-default-full.png`, `20-distilgpt2.png`,
`/private/tmp/redteam-manifold/manifold.json` vs `/api/distribution`.
Fix direction: take the top-k of the *full* distribution (union with a background
reference set) and show true probabilities, not subset-max-normalized ones.

### 2. HIGH — WebGL context leak on view switching (known-bug pattern, unfixed here)
`teardown()` (`Manifold.svelte:494-504`) calls `renderer.dispose()` but **not
`renderer.forceContextLoss()`** — the exact fix the sibling Geometry tab already has
(`src/viz/geo/GeoScene.svelte:501`). Repro (`s4-ctxleak.js`): 20× Manifold↔Vector-field
switches → **5× "WARNING: Too many active WebGL contexts. Oldest context will be
lost."** The manifold still rendered afterwards (`18-after-20-switches-canvas.png`, no
error banner) only because Chrome happened to evict already-orphaned contexts; with a
live Geometry tab / export in flight, a live context can be evicted instead. Port the
GeoScene fix.

### 3. HIGH — Surface flow field renders as a detached "hairball"
With the toggle on (`09-surface-on.png`): amber geodesics ride the invisible radius-2
token sphere (×1.02) while the visible mesh is a warped ~unit sphere, so arcs loop
through empty space far off the surface; many arcs project as straight chords across
the view; and dozens of cone arrowheads pile into a single amorphous amber blob
(PCA3 maps many "next token" targets to nearly the same point). The promised reading
("from here, the model goes there" on the surface) is not readable; combined with
Finding 1 the sources aren't actually likely tokens either. Toggle off correctly
removes it without recompute (`11-surface-off.png`).

### 4. MEDIUM — Stale + phantom hover tooltips
Tooltips only re-raycast on `pointermove`; the camera moves without the pointer
(autoRotate always on, wheel zoom, pan), so a tooltip lingers arbitrarily long showing
a token no longer under the cursor — see `04-after-zoom-in.png` ("Comey emission
0.0%" mid-zoom) and `05-after-pan.png` ("histories emission 0.0%"). Worse, the
raycaster (threshold 0.1) hits markers with alpha 0.10 that are visually invisible, so
tooltips pop over apparently empty surface (`06-hover-tooltip.png`, "$output emission
0.0%"). Every sampled hover in three runs returned a 0.0%-emission token. Fix: re-cast
on camera change or hide on wheel/drag; skip markers below an emission floor.

### 5. MEDIUM — Every slider tick / keystroke pause is a fresh cached precompute
The reload `$effect` (`Manifold.svelte:508-522`, 350 ms debounce) includes `rbfWidth`
(step 0.01 → 56 distinct values) and prompt/response text; `width` is part of the cache
key, so sweeping the bump-width slider or typing slowly fires a chain of full
ARAP+PCA precomputes and permanently grows `data/processed/cache` (one artifact per
touched value). Single-flight protects concurrency, not volume. Consider computing the
RBF width client-side from one cached reduction, or debouncing on pointer-up.

### 6. LOW — Narrow viewport overflows horizontally
At 390 px width the page has **95 px of horizontal overflow** (measured
`scrollWidth - clientWidth`); control hint text and the tab row are clipped at the
right edge (`23-narrow-390.png`). The canvas itself renders fine (413×480, ~9.7k
non-blank samples).

### 7. LOW — Temperature appears to do nothing in this view
Caption/top tokens identical at T=1.0 and T=1.5 ("having, G, !, home, noted") —
expected mathematically (softmax temperature is rank-preserving and the subset-max
normalization cancels most of the sharpening), but the UI offers the slider with no
hint that only bump *heights* subtly change (`19-temp-1.5.png`). Document or drop.

## Verified working (with evidence)
- Default warped sphere renders lit + vertex-colored, no blank canvas (`01`, `02`; pixel
  stats ~11k non-blank samples, 43 distinct colors).
- Drag-rotate (`03`), scroll-zoom (`04`), right-drag pan (`05`) all move the camera.
- Hover tooltips show token + emission (`06`) — content correctness issues aside.
- RBF width visibly changes geometry: 0.08 tight/creased vs 0.50 blended ball (`07`,`08`).
- Surface toggle on/off without recompute (`09`,`11`).
- Response trajectory: geodesic + dots, white current-token marker; caption reports
  "8 key frames … Paris → , → the → capital → of → France → ." correctly (`12`).
- ▶ Play advances (2/7 mid-play, `13`), Pause holds (3/7 stable over 2 s, `14`), scrub
  to 1 and to max works (`15`,`16`), replay-from-end resets to 0.
- Survives 20 tab switches rendering correctly, no error banner (`17`,`18`).
- Model switch (distilgpt2) and temperature changes recompute and render (`19`,`20`).
- ↻ Recompute: `force=true` is threaded end-to-end (`api/routes.py:383-397`) and
  completed in 2.0 s (`21`).
- PNG export: 116,900-byte valid PNG (magic checked), opened and visually confirmed a
  correct render — note the background exports transparent (`export-manifold.png`).
- GIF export (with response animation): 1,693,568-byte `GIF89a` rendered in 15.8 s
  (`export-manifold.gif`, `22-after-exports.png`).
- Narrow viewport (390 px): canvas renders correctly (`23`) — but see Finding 6 for the
  95 px page overflow.
- No page JS errors in any run; only benign SwiftShader "GPU stall due to ReadPixels"
  performance warnings caused by the test harness's own pixel sampling.

## Verdict
Interaction machinery (orbit, animation, scrub, exports, recompute, model/temp
switching) is solid, and the view degrades gracefully. But the tab currently fails its
own scientific claim: at the default 2000-marker setting the bulges, emission
percentages, and flow field are driven by an essentially arbitrary vocab sample
(Finding 1) — fix before this view is shown to anyone as science. Also port the
one-line `forceContextLoss()` fix (Finding 2) and rework the flow-field/tooltip
presentation (Findings 3-4).

Repro scripts: `/private/tmp/redteam-manifold/s1…s7*.js` (run with
`NODE_PATH=code/frontend/node_modules node <script>`).
