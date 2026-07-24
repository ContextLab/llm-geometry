# FIX-B report (2026-07-24) — arch F1/F2/F3/F5/F6/F9-UI + geo G1/G2

All 8 fixes done, verified live on the running stack. Screenshots: `/private/tmp/fixb/` (all visually inspected).

## Changes

1. **F1 [HIGH] false success on size-gated model** — `ArchitectureExplorer.svelte:39,59-63,77`: new `revertingTo` — the revert's (cache-hit) load no longer wipes `pickerError`; only a load of the id the user actually picked clears it. `ArchModelPicker.svelte:70-84,~140-148`: non-empty `externalError` overrides the stale ok badge/message with error styling ("checking" wins while the user interacts again; picker's own error outranks a stale external one).
   Verify (live): typed `Qwen/Qwen2.5-7B-Instruct` → ModelTooLargeError copy shown, still visible 2 s later, badge `error` (red, not green), model stayed gpt2; picking a valid model cleared it, badge back to `ok`. `03-toolarge-error.png`, `04-recovered.png`.
2. **F2 default model vanishes from dropdown** — `ArchModelPicker.svelte:25-30,96-98`: `extraOptions` unconditionally appends the default (`HuggingFaceTB/SmolLM2-135M-Instruct`) + active id, deduped against `/api/models`. Verify: default present exactly once before AND after switching to gpt2; recovery in fix 1 selected it *from the dropdown*. `02-gpt2-options.png`.
3. **F3 1-D params render as 4×3584 px hairline** — `ArchInspector.svelte:64-69`: C==1 overviews request `max_cells: 128` (server returns ≤128 downsampled rows; curl-verified). `MatrixHeatmap.svelte:47-55`: split `cellW`/`cellH`; single-column matrices widen to ≥24 px (draw/hit-test/editor updated). Zoom already pages exact 128-row windows. Verify: Qwen-0.5B q_proj bias [896×1] → 24×256 px downsampled strip; zoom → exact 128-row window. `05-qwen-bias.png` + unit test `components.test.ts:118`.
4. **F6 window-local hover indices** — `ArchInspector.svelte:150-168`: `idxLabels()` builds GLOBAL labels from `r0/r1/c0/c1` + `grid_shape` (ranges for downsampled cells), passed as `rowLabels`/`colLabels`; geo's label-passing 3×3 unchanged. Verify: zoomed window "rows 378–505" → tooltip "row 441 · col 0 · 15.625" (local would be ≤127). `06-bias-zoom-tooltip.png`.
5. **F5 raw `<|im_end|>` in chat reply** — `ArchChat.svelte:60-70,template,+css`: tokens matching `/^<\|[^|]+\|>$/` render as a subtle mono pill showing the bare name (tooltip kept). Verify: T=0 reply "The capital of France is Paris." + `im_end` pill, no raw `<|`. `10-special-pill.png`.
6. **F9-UI truncation chip** — `ArchTracePanel.svelte:130-136,+chip,+css`: defensive read of additive `trace.truncated`; amber "⋯ prompt truncated to the last 64 tokens" chip next to the chat-template chip; absent field = no-op (all other runs). Verified by injecting `truncated:true` via route interception. `09-truncated-chip.png`.
7. **G1 preset `learned` shows "edited weights active"** — `GeometryLab.svelte:372` passes `checkpointId` (spec's `checkpoint_id`); `WeightLab.svelte:44-51,91`: badge is content-based, and a minted token equal to the checkpoint hash is **normalized to null** (documented decision: one canonical "learned" representation for badge/sessionStorage/fetches; self-heal effect covers stale persisted tokens). Verify: preset learned → Apply → "learned checkpoint" badge, sessionStorage token null; identity still mints "edited weights active", reset works. `07-geo-learned.png`, `08-geo-identity-edited.png`.
8. **G2 emptied editor commits 0** — `MatrixHeatmap.svelte:152-166`: empty/whitespace (and the `null` that Svelte's number-input bind produces) now CANCELS like Escape. Unit test `components.test.ts:72` (Enter + blur cancel; real value still commits).

## Gates
- `npm run check`: 0 errors, 0 warnings (503 files).
- `npm run test`: 34/34 pass (incl. 2 new MatrixHeatmap tests).
- `npx playwright test tests/e2e/explorer.spec.ts`: 8/8 pass.
- Live console: only the browser's own log of the designed 422 (ModelTooLargeError).

Not touched: Manifold/VectorField/Sankey, controls/, App.svelte, app.css, dataClient.ts, backend. No git commands run.
