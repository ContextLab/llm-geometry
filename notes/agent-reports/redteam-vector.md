# Red-team report — Vector field tab (2026-07-24)

Adversarial UI/API audit of the **Vector field** view, driven through real Chromium
(Playwright scripts + screenshots in `/private/tmp/redteam-vector/`). Backend :8000 and
frontend :5173 were left running. No repo edits.

## Findings

### 1. MAJOR — ▶ Play / step scrub: the canvas never reaches the requested frame
The animation tween creeps asymptotically and effectively freezes, so the drawn
trajectory/arrows desync from the step label.
- Repro: response "Paris, the capital of France." → wait for the animation to load →
  scrub the step slider (or press ▶ Play). 6 s after scrubbing to 0 and then to 3 the
  canvas still renders frame 0 (0 trajectory dots, label "3/7"): `32-scrub-to-0.png`,
  `33-scrub-to-3.png`. During Play, at label "2/7" the canvas showed frame ≈6.9 (6 dots +
  partial segment): `15-play-mid.png`, `16-paused.png`, `17-step2.png`.
- Whether the at-rest view shows the full trajectory is a race: one run rested at 7 dots,
  another (identical steps, cache warm) rested at 0 dots (`13-anim-rest.png` vs
  `32-scrub-to-0.png` "at rest: dots 0, label 7/7").
- Root cause (code): `code/frontend/src/viz/VectorField.svelte:57-60` — the `$effect`
  calls `tweenTo()`, which reads `animTime` (`const from = animTime`, line 124). That
  makes `animTime` a tracked dependency of the effect, so every rAF write to `animTime`
  re-runs the effect, which restarts the tween with a fresh `t0`; progress per restart is
  ~e(1 frame/650 ms) ≈ 0.1 % of the remaining distance → it never completes. Self-heals
  only when `load()` sets `animTime` directly (e.g. after a temperature change refetch).
- Fix hint: wrap the call in `untrack(() => tweenTo(...))` (or read only `$responseStep`
  in the effect and snapshot `animTime` outside the tracked scope).

### 2. MINOR — Caption arrow count is wrong in drift mode
Caption reads "576 grid arrows · … · fan-out 2" while 1152 `<line>` arrows are drawn
(`data.reference_points` counts grid vertices, not arrows). At temp 0 the server returns
fanout 1 and caption/DOM agree at 576. `01-default.png`, API cross-check: starts=1152,
reference_points=576.

### 3. MINOR — High temperature washes the field out
Colour/opacity are normalised by the max prob (`rel = p/maxp`); at temp 2 one outlier
arrow stays bright and the other ~1151 render near-invisible — the field reads as
"empty" (`08-temp2.png`). Consider a robust (quantile) normalisation.

### 4. MINOR — Fan-out is a hard-coded 2
`FANOUT = 2` in VectorField.svelte regardless of temperature; project_description.md §1
describes temperature>0 fanning out into multiple semi-transparent vectors estimated
over ~100 reps. No UI control; temp only re-weights two arrows.

### 5. NIT — Invalid model: raw HF traceback + unlabelled stale field
Loading `no-such-org/definitely-not-a-model-xyz123` surfaces the full HF exception text
(incl. `hf auth login` advice) in both the model badge message and the viz error, and the
previous model's field/caption ("layer 6/6") stays rendered beneath the error with no
staleness cue (`22-invalid-model.png`). Recovery by re-selecting a valid model is clean.
Two 422 console errors logged (the failed fetches) — handled.

### 6. NIT — Long-prompt first compute is slow with no cancel
A ~1 kB prompt took 115 s to first paint (precompute job with progress bar; cached
afterwards). No cancel affordance (`11-long-prompt.png`).

## Verified correct (attacks that failed)
- Default load: 1152 DOM arrows exactly match `/api/vector_field` rows; hover tooltips
  (arrow: "tok → tok · layer · %", vertex: nearest/predicted token) match API values
  (`02-hover-arrow.png`, `03-hover-origin.png`); 576 dedup origins = unique starts.
- Layer slider 24→0→12→24: arrows genuinely change per layer; rapid 13-step scrub
  collapsed to ONE debounced request, no wedge/duplication, caption tracks the slider,
  and the restored layer-24 field matches baseline (`04-layer00.png`–`06-after-scrub.png`).
- Temperature: typing 9 clamps to 2.00; temp 0 → 576 deterministic arrows, probs all
  1.0, caption "fan-out 1" matches the server (`07-temp0.png`).
- Prompts: empty, emoji ("🦄🌈 …"), and 1000-char inputs all render; console clean
  (`09`–`11`).
- Tooltip at the right edge stays inside the viewport (`12-tooltip-edge.png`).
- Model switching: gpt2 / distilgpt2 load in ~4 s; layer slider re-ranges (12/6);
  switching models mid-compute settles on the last selection (no stale overwrite)
  (`20`, `21`, `24-model-race.png`).
- Changing temperature mid-Play refetches and self-heals the animation (`18`).
- Clearing the response returns cleanly to drift mode (`19-cleared.png`).
- Exports: SVG (1.1 MB, valid, own background rect), PDF (v1.3, 1 page), PNG (2×,
  visually correct: dots, arrowheads, colours) all download non-empty
  (`export.svg/pdf/png`).
- Recompute: forces a 6.2 s re-fetch with spinner; field intact afterwards
  (`25-post-recompute.png`).
- Scroll-zoom + drag-pan work; arrowheads/dots scale sensibly (`26-zoomed.png`,
  `27-panned.png`).
- Tab to Sankey and back: prompt/temp/layer preserved, field re-renders from cache,
  console clean (`28`, `29`).
- 700 px viewport: single-column layout below the 900 px breakpoint, no overlap or
  clipping (`30`, `31-narrow-700-scrolled.png`).

## Verdict
The drift-mode vector field is solid: numbers match the API, controls debounce and
recover correctly, errors are typed, exports and layout hold up. The **response
animation is functionally broken** (Finding 1): the canvas frame effectively never
follows ▶ Play or the step slider because the tween restarts itself every animation
frame. One-line fix (`untrack`) — should be fixed before demoing the trajectory feature.
Findings 2–4 are presentation-accuracy issues worth cleaning up alongside it.
