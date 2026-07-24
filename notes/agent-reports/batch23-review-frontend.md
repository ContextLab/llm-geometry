svelte-check: 502 files, 0 errors. All verifications complete; dev stack confirmed restored (:8000 and :5173 both 200, healthy). Note: I am the read-only code-reviewer (Write/Edit blocked), so I cannot create `notes/agent-reports/batch23-review-frontend.md` — per protocol the report is returned inline below for the parent to save.

---

# Batch-2/3 view-layer red-team (feature 002, issue #1) — REFUTE stance
Live-verified against the running :5173/:8000 stack (Chromium). svelte-check 502/0/0.

**1. [HIGH] FinetunePanel.svelte:37,60,71-75** — `onDone` ignores the SSE done payload (the very thing 617df5c added) and re-calls `submitOnce()`, which re-reads `$geoWeightsToken` fresh (:38). WeightLab's Apply is NOT disabled during a fine-tune, so if the user edits weights mid-run, `base` changes T0→T1 → different content hash → no cache hit → 202 → `!cached.ready` throws "Fine-tune finished but the result was not cached." A real, expensive, *successful* fine-tune reports a false failure and the minted checkpoint (already in the payload) is dropped. Same failure if T0 is LRU-evicted by done-time. Fix: `onDone:(data)=>finish(data as GeoFinetuneResult)`; delete the re-submit.

**2. [MEDIUM] GeoScene.svelte:482-495** — `teardown()` calls `renderer.dispose()` but never `forceContextLoss()`, and there's no `webglcontextlost` handler. Verified: cycling Geometry↔Vector 18× spews "Too many active WebGL contexts. Oldest context will be lost." Past ~16 visits the browser drops the live sphere's context → blank canvas, no designed error (try/catch only guards construction). Fix: `renderer.forceContextLoss()` in teardown + a `webglcontextlost` listener that sets `webglError`.

**3. [MEDIUM] GeometryLab.svelte:190-207,339 (explorerStores.ts:66)** — a persisted `geoWeightsToken` the server evicted wedges the view. Verified (bogus token): phase reaches `ready` but field+trace+WeightLab render THREE stacked `NotFoundError` blocks, and the field "retry" (:339) re-sends the same dead token → permanent no-op loop. Only recovery is WeightLab "reset to learned" (works). Fix: on a NotFoundError naming weights_token, auto `geoWeightsToken.set(null)` so the view self-heals.

**4. [MEDIUM] archShared.ts:41 / dataClient.ts:527-551** — the crafted NetworkError copy ("Start it (sh scripts/dev.sh)…") is dead in the real proxy topology. Verified with backend killed: both tabs DO show designed cards + Retry (no blank/stack-trace — FR-107 core holds), but the text is bare "HTTP 500" (Vite turns connection-refused into a 500 body); NetworkError never fires. Fix: map 500/502/503 w/ empty/HTML body to the "backend unreachable" copy.

**5. [LOW] ArchitectureExplorer.svelte:74-77** — the graph `$effect` reads `graphLoading/graphError` that `loadGraph()` writes synchronously → effect re-runs → `loadGraph` called twice on mount; only the module `graphCache` (archShared.ts:12) prevents a duplicate HTTP. Fix: `untrack` or gate solely on `id!==graphForModel`.

**6. [LOW] ArchitectureExplorer.svelte:154-172** — the `onDiagClick` re-dispatch workaround is now DEAD after the 617df5c pan fix. Verified: one trusted node click = exactly 1 click event, inspector opens once (double-fire REFUTED; `target.contains(e.target)` guard short-circuits). Remove for clarity.

**Attacks that held up (REFUTED):** node-click double-fire (1 event, live); stale-response races — seq/ctl guards at ArchitectureExplorer:95-101, GeometryLab:143-149, WeightLab:54-63 all correct; model-switch mid-trace aborts cleanly (:45-53); playback-vs-new-trace resets + rAF cleanup (ArchTracePanel:47-58,84); rAF/ResizeObserver/EventSource all torn down (GeoScene:483-484, GeometryLab:72-79, FinetunePanel:27); no runaway `$effect` loops (ArchTracePanel:47-58 read/write of `layerSel` is gated by non-reactive `lastTrace`).

**Verdict: REQUEST CHANGES** — 1 HIGH + 3 MEDIUM. #1 is the blocker (turns a successful fine-tune into a user-facing error under normal interaction); #2/#3 degrade real sessions silently. B3's diagram-click workaround and the disposal/race machinery are otherwise sound.

---

Note on the backend-outage test (attack #3): I killed backend PID 58135, confirmed both tabs degrade to designed error cards (not blank panels), then restarted uvicorn and updated `.devservers/backend.pid`. Stack is healthy and left running.