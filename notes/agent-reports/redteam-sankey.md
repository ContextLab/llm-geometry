# Red-team: Sankey tab (2026-07-24)

Method: real Chromium (Playwright, SwiftShader) driving http://localhost:5173; every state screenshotted to `/private/tmp/redteam-sankey/NN-*.png` and visually inspected; cross-checked against `curl /api/sankey` / `/api/sankey_highlight`. Scripts: `/private/tmp/redteam-sankey/rt{1,2,3}-*.mjs`. Stack left running.

## Findings (severity-tagged)

1. **MEDIUM — giant-ribbon blob at low diversity.** Flow width cap is `rowH * 0.8` (`Sankey.svelte:246`), and `rowH ∝ 1/nRows`. At temp=0, seqlen=2, particles=1000 the swarm collapses to 2 rows → one ribbon measured **102px stroke-width**, an abstract purple blob burying the cells; unreadable as a flow. Evidence: `12-temp0.png`. Fix idea: absolute px cap.

2. **MEDIUM — stale-diagram / fresh-caption incoherence during recompute.** Caption binds live stores and the gold highlight refetches independently (~120ms), but the swarm replaces only when its (15–75s) compute lands. During that window the OLD diagram is shown with a caption describing the NEW params, and the new highlight is redrawn onto the stale chart — even **stretching its axis** (`maxPos = max(d.max_pos, hlPos)`, `Sankey.svelte:171`). Evidence: `10-steps2-loading.png` ("up to 2 steps" under a 10-column chart), `17-steps24-progress.png` (old *baker/cartographer*-prompt swarm shown under the new "capital of France" prompt, highlight stretched to +4), `27-gpt2-inflight.png` (GPT-2 badge over Qwen chart).

3. **LOW — garbage/long response degrades layout.** Response tokens outside the swarm each append a row (`Sankey.svelte:161`): a 24-token garbage response → 42 rows at ~12px pitch (1 measured label overlap), chart dominated by empty gold guide rows; multi-byte emoji splits into several identical-looking "�" rows. Evidence: `19-garbage-response.png`.

4. **LOW — no redraw on window resize.** No resize listener; after shrinking to 700px the SVG keeps viewBox `0 0 743 560` (uniform downscale, shrunken text) until an unrelated redraw re-lays out to `0 0 602 560`. Measured in rt3.

5. **LOW — "⏸ Pause" is actually "stop + reveal all"** (`togglePlay`, `Sankey.svelte:120`): clicking Pause mid-sweep jumps straight to the fully revealed diagram.

6. **LOW (code) — highlight errors silently swallowed.** `loadHighlight` catch → `highlight=[]` (`Sankey.svelte:101-105`); a failing `/api/sankey_highlight` silently drops the gold path with no user feedback.

7. **NIT — caption text:** missing space "up to 10 steps· ✦ …" (Svelte `{#if}` whitespace trim, `Sankey.svelte:327`); "1 transitions" pluralization (`12-temp0.png`).

## Verified correct

- **Render = API exactly:** 273 flow paths / 180 cells (18 rows × 10 cols) match `/api/sankey` nodes/links/token_order; caption counts match API on 2 further combos (`01-default.png`).
- **Hover:** flow hover traces a coherent trajectory + tooltip (per-step P, particle count, trajectory log-p); cell tooltip (share %, count); gold-node teacher-forced P; "Paris" vs " Paris" correctly distinct rows (`02`,`03`,`04`,`05`).
- **Controls:** slider ranges (50–2000, 2–24) can't reach invalid values; backend 400s are designed envelopes (`InvalidParamError` for n_particles/n_steps<1, temperature<0; n_particles clamped ≤2000).
- **Prompts:** empty, emoji/CJK, 400-char prompts all render sensible swarms (`14`,`15`,`16`); seqlen 24 → 25 readable columns (`18`).
- **Progress UX:** cold params show animated bar + "swarm step k/24 · 50 particles" (`17`); Recompute shows "recomputing…" then fresh data (`25`,`26`).
- **Play:** continuous reveal sweep, full restore after (`06`,`07`).
- **Exports:** SVG/PDF/PNG/GIF/MP4 all download, valid formats, PNG visually correct (computed styles inlined, dark bg kept).
- **Robustness:** tab thrash ×10 → intact + hover alive; model switch mid-compute → run-guard wins, orphaned GPT-2 job never clobbers Qwen render (checked +45s) (`22`,`28`). **Zero console errors/pageerrors across all phases.**

## Verdict

**Solid.** No rendering corruption, crashes, data mismatches, or console errors under adversarial use. Real gaps are transitional-state coherence (finding 2) and degenerate-layout extremes (findings 1, 3).
