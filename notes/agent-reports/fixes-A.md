# Fix agent FIX-A — vector-field + Sankey redteam fixes (2026-07-24)

All 12 findings from `redteam-vector.md` / `redteam-sankey.md` fixed and verified live
against the running stack (:5173/:8000). Scripts + screenshots: `/private/tmp/fixa/`
(`vector.mjs`, `sankey.mjs`, logs `vector-rerun.log`, `sankey-run.log`). New shared pure
helpers in `code/frontend/src/lib/vizMath.ts`; new stores (`fanout`, `modelError`) in
`src/lib/stores.ts`. Unit tests: `tests/unit/vizMath.test.ts`.

Checks: `npm run check` 0 errors · `npm run test` 40/40 (5 files, incl. 6 new) ·
`npx playwright test tests/e2e/shell.spec.ts` 13/13 passed (incl. the unsupported-model
error-message test, kept regex-compatible; log `/private/tmp/fixa/e2e-shell.log`).

| # | Fix (file:line) | Verified |
|-|-|-|
| F1 MAJOR | `VectorField.svelte:57-65` — step effect now calls `untrack(() => tweenTo(...))` (svelte `untrack`, line 3); effect depends only on `$responseStep` + `anim`, so rAF writes to `animTime` no longer restart the tween | Live: rest=7/7→7 dots; scrub→0 ⇒ 0 dots; scrub→3 ⇒ 3 dots; Play sampled 14×, 0 desyncs (canvas tracks label within 1 tween frame); `f1-rest/scrub0/scrub3/play-mid/final.png` |
| F2 | `VectorField.svelte:428` — caption uses `data.starts.length` (actual arrows), not `reference_points` | Caption 1152 == 1152 DOM lines (fan-out 2); 1728 == 1728 (fan-out 3) |
| F3 | `vizMath.ts::robustMax` (95th-pct, 1e-6 floor) used at `VectorField.svelte:226` (drift) and `:344` (animation); color scale `.clamp(true)` | temp 2: 213/1152 arrows ≥0.7 opacity (was ~1/1152), whole field reads populated — `f3-temp2.png` inspected; unit test with 1151+1-outlier distribution |
| F4 | `stores.ts:26` `fanout` store (default 2) → slider 1–5 in `LayerSlider.svelte:23-36` (visible only when `$temperature > 0`, styled like the other controls) → `VectorField.svelte:44,105` passes `fanout` to `/api/vector_field` | Live: `fanout=3` request observed, 1728 arrows drawn after cold precompute (progress bar), control hidden at temp 0 — `f4-fanout3.png` |
| F5 | `ModelSelector.svelte:44-50,80` — concise first line ("Model not found or could not be loaded" / "Could not reach the backend server", e2e-regex compatible), raw detail behind `title`; sets `modelError` store. `VectorField.svelte:415-419` dims `.stage` + amber "showing previous model" caption note | Live: 38-char message, 464-char title detail, stage dimmed + note, clean recovery — `f5-invalid-model.png` inspected; unit test mounts ModelSelector |
| S1 | `vizMath.ts::capLinkWidth` (28px cap, ≥1px) at `Sankey.svelte:306`; hover width also capped (:322) | temp0/seqlen2/1000p: max stroke-width exactly 28 (was 102) — readable ribbon, `s1-temp0.png` inspected; unit test |
| S2 | `Sankey.svelte:38-45,101,120-132` — swarm + highlight each record their params key; `:209` highlight integrated (rows/axis/gold/caption) only when keys match; caption (:400) binds `data.*` meta; `:394-395` stale chart dimmed + "recomputing…" badge | Live: during cold seqlen-24 recompute the old chart stays captioned "up to 10 steps", dimmed + badge, axis stays +9 (no stretch), mismatched highlight not drawn; gold path returns on settle — `s2-before/transitional/settled.png`, transitional inspected |
| S3 | `Sankey.svelte:216-219` — response-only rows capped at 8; caption note "+N more response tokens not shown" (:400) | 15-word garbage response: rows 20→26 (=18+8), caption "+2 more" — `s3-garbage.png` |
| S4 | `Sankey.svelte:172-180` — ResizeObserver on the stage → `draw()` on >1px width change | viewBox 743→702 on viewport resize — `s4-resized.png` |
| S5 | `Sankey.svelte:151-168` — ⏸ freezes `reveal` (no reset); ▶ resumes from frozen offset; completion and new ⏹ button (`:384`, `stopAndRestore`) restore full reveal; fresh data resets reveal (:101-103) | Live: paused at 75/273 paths, frozen across 900ms, "▶ Resume" + ⏹ shown; resume completes to 273; ⏹ restores full — `s5-paused.png` inspected, `s5-completed.png` |
| S6 | `Sankey.svelte:123-127,137-140,401` — highlight-fetch failure sets `hlError` → inline "✦ response path unavailable — retry" note with working retry | Live (invalid model): note appears; recovery clears it and restores gold path — `s6-hlerror.png` |
| S7 | `Sankey.svelte:400` — explicit `{" · "}` separator (no more "steps·"); `vizMath.ts::plural` for "1 transition"/"1 token row" | Caption reads "2 token rows · 1 transition · … · up to 2 steps"; unit test |

Notes: fan-out is part of the vector-field cache key — cold (temp, fanout) combos show the
progress bar once, as expected. No files outside the FIX-A ownership list were touched
except the two append-only store additions and the new `vizMath.ts`/test files.
