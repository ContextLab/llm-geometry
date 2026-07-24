# Red-team: Geometry tab (2026-07-24)

Adversarial pass over the Geometry Lab (`/api/geo/*` + `src/viz/geo/*`), focused on
incorrect geometry/math as well as breakage. Method: source review, 42 numeric
API cross-checks (`/private/tmp/redteam-geo/mathcheck.py`, run against the live
backend), and 7 real-Chromium Playwright sessions (`s1`–`s7_*.cjs`, 55 screenshots,
every one visually inspected). Stack left running; cache untouched.

## Findings

1. **[LOW] Preset `learned` shows "edited weights active" for bit-identical weights.**
   Weight lab → preset `learned` → Apply mints token `be5359a1c66bda29c8c554269e589009`
   — exactly the canonical `checkpoint_id` from `/api/geo/spec` (content-hash dedup
   working as designed) — yet the badge flips to "edited weights active | preset:learned".
   The active weights ARE the learned checkpoint; the badge is misleading. Evidence:
   s3 log + `s3_preset_learned.png`. Cause: badge keys off token non-null
   (`WeightLab.svelte` head), not content.

2. **[LOW] Clearing a cell editor and committing silently writes 0.**
   `MatrixHeatmap.svelte` `commitEdit()`: `Number("") === 0` and `Number.isFinite(0)`
   passes, so an emptied editor + Enter (or click-away — `onblur={commitEdit}`) commits
   `0` and mints a token instead of canceling. Repro: s7 — W_V(0,0) `0.00028039 → 0`,
   token minted, no error. Expected: empty input cancels like Escape.

3. **[LOW] Horizontal overflow at narrow viewport.** At 420 px the tab strip, header
   text and canvas overflow the viewport (page scrolls sideways). `s6_narrow_420.png`.

4. **[INFO] Float32-overflow edit (`1e39`) passes the client check** (finite in
   float64) and is caught server-side as designed
   (`422 InvalidWeightEditError: non-finite entries`, surfaced in the panel; token
   stays null). Defense-in-depth works; a client-side float32 bound would be cosmetic.

5. **[INFO] Sequence forces are anchored at raw token-embedding points** while being
   computed from per-layer hidden states (pos-embedding + residual). Documented design
   choice in `fields.py`; not a math error, but the arrow base is not the state the
   force acts on.

## Math verified correct (42/42 API cross-checks)

- **force mode**: `vec == W_V·z` (max err 1.1e-6); antisymmetrize `== 0.5(W_V−W_Vᵀ)z`
  (8.5e-7) with *exact* tangency max `|⟨v,z⟩|` = 5.3e-7 and `tangent_exact` flag;
  `weight == |v|/max|v|`; points all unit-norm; `sequence_forces == attn @ v` from
  `/trace` (9.2e-7); `normal_residual == |⟨f, ẑ_in⟩|` (3.4e-7).
- **next_next**: T=0 ⇒ 1003 argmax arrows, weight 1; arrow("alice") target == trace
  argmax; prompt conditioning + layer selection change the field; `layer=3 == full`
  bit-exactly; T=1/top_m=5 ⇒ 5015 arrows, per-origin prob sums ≤ 1, T=2 flatter than T=1.
- **trace**: attention exactly causal (upper-tri 0) and row-stochastic; top-10 ids
  == argsort(probs); probs sum to 1. UI top-10 for "the queen of" matches curl
  **digit-for-digit** (`,=7.5% <unk>=5.5% the=5.3% …`).
- **weights**: identity-W_V force field == z (radial — confirmed numerically and
  visually, `s3_preset_identity.png`); zero-W_V kills vocab arrows *and* amber
  sequence forces (correct: v=0), green path remains; chained W_V+W_Q edits both
  stick; provenance sensible after fine-tune + edit (`W_V preset:identity`,
  untouched matrices `edited`).
- **Designed errors** all correct per frozen contract: force+`layer=full` → 400;
  bad mode/negative T/layer 99/bad matrix/missing layer → 400; embedding-zero
  preset → 422 (surfaced in panel, token unchanged); NaN values → 422; bad HF
  dataset → 422 (surfaced); no-source/two-sources/steps=0 → 400 (contract line:
  "400 no source / more than one source; 422 unusable dataset id" — matches).

## Interaction sweep (all pass; screenshots in /private/tmp/redteam-geo/)

Sphere + 1003 points render; orbit + zoom work (s1_02–05); hover tooltips give real
words — 5 spot-checked ids match `/api/geo/tokenize` exactly (alice=17, rabbit=114,
dark=907, game=331, he=44); green geodesic path + amber forces in force mode
(s1_08). Layer sweeps visibly change both modes (s2_*); temperature fan-out visible
(s2_nn_T1_m5); antisymmetrize badge flips residual↔"tangent: exact" and back; badge
residual 0.144 consistent with default prompt. All 6 presets change the field with
distinct signatures. Persistence: token survives reload with edited field intact
(s4_02); bogus injected token self-heals — exactly three 404s, token cleared,
learned field, zero error residue, no retry loop (s4_03/04). Fine-tune: paste
(5.26→4.79), .txt upload (5.49→4.90), HF `roneneldan/TinyStories` (4.95→4.82) all
real SGD with new tokens; 500-step run shows live SSE ("step 190/500 · loss 2.45"),
overfits repeated text 4.65→1.70; empty text = client-side error; steps slider
10..500. Attention: per-layer tabs; upper-triangle pixels exactly the zero color
rgb(27,34,53) at the canvas level. Edge: all-unknown prompt → 3 <unk> chips, field
still fine; 60 tokens → "⋯ truncated at 50" chip; empty prompt → designed hint;
rapid typing → no flicker/stale. Stability: 20 tab switches → no WebGL error,
13 867 lit pixel samples (context-loss fix holds). Console clean apart from
SwiftShader perf warnings and the browser's own logs of designed 4xx responses.

## Verdict

**PASS.** No HIGH or MEDIUM findings. Every geometric/mathematical claim the tab
makes was reproduced independently from the API to float32 precision, and the UI
numbers match the API exactly. Three LOW UX nits (misleading "edited" badge for
learned-content tokens, empty-cell commit-as-zero, narrow-viewport overflow) and
two INFO observations. Artifacts: scripts + 55 PNGs in `/private/tmp/redteam-geo/`.
