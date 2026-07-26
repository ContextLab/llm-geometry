# Red-team review — Architecture Explorer (live backend)

**Date:** 2026-07-26
**Target:** `http://localhost:5173/` → Architecture tab, against the real FastAPI backend on `:8000`
**Method:** Playwright driving a real Chromium against real models (Qwen2.5-0.5B-Instruct,
SmolLM2-135M/360M-Instruct, gpt2). 10 scripted phases, screenshots inspected visually at
`/tmp/redteam-arch-shots/`. No mocks, no source modified.

Reviewed strings/behaviour in `code/frontend/src/viz/arch/{ArchitectureExplorer,ArchTracePanel,ArchChat,ArchInspector,ArchModelPicker}.svelte`,
`archShared.ts`, and `code/frontend/src/lib/PipelineDiagram.svelte`.

**Bottom line:** no CRITICAL defects. Playback, model switching, chat, the inspector, the
debounce, truncation disclosure and the 390 px layout all genuinely work — several of them
better than the copy claims. One HIGH-severity dataviz defect makes a whole panel
information-free, and there are three MEDIUM issues (two of them false/overstated claims).

---

## HIGH

### H1 — The "‖residual stream‖ per token" chart is a single spike and a flat line; it conveys nothing

**What I did.** Played the trace on the default prompt `What is the capital of France?`
(Qwen2.5-0.5B-Instruct), then measured every rendered bar's pixel height and `aria-label`.

**What I saw.** 36 bars; measured heights
`[60, 2, 2, 2, 2, 2, 2, … 2]` — **35 of 36 bars sit at the floor**. The labels explain why:

```
"<|im_start|>: norm 1725.35"   ← the attention-sink / BOS token
"system: norm 17.86"
"\n: norm 13.36"
"You: norm 13.29"
```

Visible in `1-b-playing.png` and `2-b-heads.png` as one blue bar at the far left followed by
what reads as a dashed horizontal rule. The panel is labelled
`‖RESIDUAL STREAM‖ PER TOKEN · LAYER 6 OUT`, i.e. it advertises a per-token comparison.

**What I expected.** A chart where the per-token norms are actually comparable. This is not a
rare edge case: every modern instruct model parks a huge-norm attention sink on the first
token, so **this panel is degenerate for essentially every prompt on every curated instruct
model** — it renders in every screenshot I took, at every layer, for all four models.

**Cause.** `code/frontend/src/viz/arch/ArchTracePanel.svelte:143-147` scales linearly against
`maxNorm` (the raw max), and `:317`
`style:height={`${Math.max(4, (v / maxNorm) * 100)}%`}`. Note the floor is **4 %**, not 4 px —
in the 68 px box (`:631-639`, minus 4 px padding) that is ~2.4 px, so the "minimum visible
height" is not visible. A log scale, a robust/percentile max, or excluding the sink token from
the scale would all fix it.

---

## MEDIUM

### M2 — "all N heads" is claimed but only a fraction of the heads are rendered in view

**What I did.** Read the label and measured the head-grid container and its tiles at 1440 px
and again at 390 px.

**What I saw.** Label text: `attention · layer 7 · all 14 heads`. Measured:

```
{"tiles":14,"scrollH":614,"clientH":208,"scrolls":true,"visible":4}
```

**4 of 14 tiles are visible**; the rest need an inner scroll with no scroll affordance drawn.
`2-b-heads.png` shows row 2 of tiles sliced through the middle. At 390 px
(`6-f-390-breakdown.png`) the label reads `ALL 9 HEADS` and **1 of 9** tiles is visible plus a
sliver of the second.

**What I expected.** Either all heads actually visible at once, or copy that does not say "all".
The source comment states the intent outright — `ArchTracePanel.svelte:287-288`: *"Every head of
the layer at once: comparing heads was impossible when only one could be shown at a time."*
The clipping defeats the stated purpose of the feature.

**Cause.** Label `ArchTracePanel.svelte:284`; grid `:561-567` (`max-height: 13rem; overflow-y: auto`)
against tiles sized by `maxCanvasPx={74}` at `:299`.

### M3 — During a model switch the whole tab shows the *previous* model's data attributed to the *new* model

**What I did.** Throttled `/api/arch/graph` to make the (normally 10–60 s) cold-load window
observable, then switched Qwen2.5-0.5B → SmolLM2-360M and sampled the DOM at +1.5 s, +4 s, +7 s.

**What I saw.** Identical at all three samples:

```
overlay:    "downloading + tracing HuggingFaceTB/SmolLM2-360M-Instruct…"
picker:     "SmolLM2 360M Instruct · 32 layers"
headerMeta: "24 layers | hidden 896 | 14 heads · 2 KV | vocab 151,936 | 494.0M params"   ← Qwen
bdSub:      "…tokenize → 340 traced ops → next-token distribution"                        ← Qwen
tokens:     "<|im_start|> 151644 system 8948 ↵ 198 You 2610 are 525 …"                    ← Qwen template
headsLabel: "attention · layer 0 · all 14 heads"     ← Qwen has 14 heads; SmolLM2-360M has 15
layerLabel: "layer 0 / 23"                           ← Qwen has 24 layers; SmolLM2-360M has 32
metaOpacity: "1"   bdOpacity: "1"                    ← neither is dimmed
```

Screenshot `7-stale-midload.png`: the dropdown says SmolLM2 360M, the overlay names SmolLM2
360M, and the top-right corner simultaneously says `494.0M params` with Qwen's chat-template
tokens below at full opacity. During the free-running (unthrottled) switch in phase 3 the same
stale meta was still present at +400 ms.

**What I expected.** Stale numbers cleared or visibly dimmed/labelled as belonging to the
outgoing model, the way the diagram itself is (`.diag.dim`, `ArchitectureExplorer.svelte:479-482`).

**Cause.** The header meta (`ArchitectureExplorer.svelte:267-279`) renders from `graph`, and
`ArchTracePanel` (`:375-383`) renders from `trace`; both are only cleared inside `loadGraph`'s
`.then()` (`:51-57`), i.e. *after* the new graph resolves. Nothing dims them meanwhile.

### M4 — Chat tooltip prints two different distributions side by side as if they were one

**What I did.** Generated at temperature 0, 0.8 and 1.6 and read the per-token tooltips.

**What I saw.** At temperature 0, for the token `The`:

```
p = 100.0% · top-5: "The" 86.8% · "Paris" 5.3% · "As" 3.1% · "I" 1.3% · "France" 0.7%
```

The same token is simultaneously "100.0 %" and "86.8 %". At temperature 0.8 it reads
`p = 94.5% · top-5: "The" 86.8% …`. The two numbers only ever agree at temperature exactly 1.0.

**What I expected.** Either one consistent distribution, or a label that says which is which.

**Cause.** The backend deliberately reports two things — `code/backend/src/llm_geometry/arch/generate.py:117-137`:
`prob` is the *temperature-adjusted sampling* probability (one-hot at T=0) while `topk.probs` is
the *model's plain softmax* (its own comment: *"Alternatives always report the MODEL's
distribution (plain softmax)"*). That is a defensible backend choice, but the frontend tooltip
that renders them — `ArchChat.svelte:85-91` — labels the first bare `p =` and gives the reader
no way to know they are measured differently. The user-facing string
`"hover a token for its probability + alternatives"` (`ArchChat.svelte:157-160`) reinforces the
wrong reading that the alternatives are drawn from the same distribution as `p`.

---

## LOW

### L5 — Inspector heatmap tooltip keeps pre-zoom coordinates (and value) over the post-zoom data

**What I did.** Hovered the lm_head overview map, clicked to zoom without moving the mouse, then
nudged the mouse 6 px.

**What I saw.**

```
overview,  meta: whole matrix 151936 × 896 · downsampled  | tooltip: rows 73594–75967 · cols 434–447 · -0.0015144
after zoom, mouse still: rows 75579–75642 · cols 416–479 of 151936 × 896 · exact values
                                                          | tooltip: rows 73594–75967 · cols 434–447 · -0.0015144   ← stale
after 6 px mousemove:    (same meta)                      | tooltip: row 75612 · col 449 · -0.0162354               ← correct
```

Screenshot `9-zoom-stale-tip.png` (and visible in `8-zoom-lm_head.png`). Since the cursor is by
definition stationary at the instant of the zoom click, the user is shown a wrong row/col range
*and* a wrong value for the cell under the pointer until they jiggle the mouse.

**What I expected.** The tooltip to re-derive or hide when the underlying window changes.

**Cause.** `ArchInspector.svelte:155-168` recomputes `rowLabels`/`colLabels` reactively, but the
global tooltip singleton is only refreshed by a `mousemove` on the canvas.

### L6 — Truncation is disclosed, but the chip sits *after* all 64 token chips

**What I did.** Pasted a 14× repeated sentence (>64 tokens).

**What I saw.** Truncation *is* honest and real — `{"ntok":64,"trunc":"⋯ prompt truncated to the
last 64 tokens","anyTrunc":true,"first":" dog 5562"}`, and the API really returns
`"truncated": true`. But the strip begins abruptly mid-sentence at `dog` with no leading marker,
and the `⋯ prompt truncated…` chip renders after all 64 chips (`6-c-long.png`). The `⋯` glyph
implies *leading* elision while sitting at the trailing end.

**Cause.** `ArchTracePanel.svelte:208-218` — the chip is emitted after the `{#each trace.tokens}`
block. Truncation is left-side (`arch/trace.py:55-56`), so the marker belongs at the head.

### L7 — The diagram's initial zoom makes every node illegible until you play or zoom

**What I saw.** On load the viewBox is `-20 -10 680 2658` in a 520 px-tall SVG (~0.2× scale), so
32-unit nodes render ~6 px tall and the 11 px labels ~2 px — see `1-a-initial.png`, where the
diagram is a column of unreadable slivers. The caption invites "click a block to inspect its
weights" against targets a few pixels tall. Playback fixes it (`PipelineDiagram.svelte:243-264`
zooms to `FOLLOW_H` on the first focus frame), but the resting state does not.

**Cause.** `PipelineDiagram.svelte:197-206` fits the *entire* model height on first layout.

### L8 — Caption overpromises slightly on parameterless blocks

**What I saw.** Caption: `click a block to inspect its weights`
(`ArchitectureExplorer.svelte:368-371`). Clicking `rope`, `attention_softmax`, `residual_add`,
`mlp.act_fn` opens the inspector with **no weights** — it shows the honest note *"This op has no
learned weights — it transforms activations on the fly during the forward pass, so there is no
matrix to plot."* The inspector behaves correctly and does not error; only the caption is
imprecise.

### L9 — Unverifiable claim (not falsified)

`"first load of a new model can take 10–60 s — the graph is built from a real traced forward
pass, then cached"` (`ArchitectureExplorer.svelte:353-356`). All four models were already cached
in this session (warm graph builds: 1.2 s / 1.3 s / 1.7 s / 18.5 s), so I could not test a cold
download. Flagging as unverified rather than false.

---

## Verified working (things I actually confirmed, not skipped)

**Playback** — pressed ▶ and sampled every 700 ms for 7 s:
- The diagram visibly moves: viewBox `y` changed on 9/10 samples (105 → 586 → 903 → …).
- The playhead advances (op 10 → 99 of 340) and reports real tensors:
  `op 99/340 · MLP down projection · model.layers.6.mlp.down_proj · ‖out‖ 37.04 · [T × 896]`.
- **0/10 samples had the highlighted node outside the SVG viewport, and 0/10 had it missing from
  the DOM** — the auto-expand of the focused layer group (`PipelineDiagram.svelte:215-232`) works;
  the highlight never hid inside a collapsed group.
- Pause freezes it exactly (op 100 → 100 after 2 s); the button label flips `▶` ⇄ `❚❚`.
- Scrubbing to 0 / 170 / 339 lands on `model.embed_tokens`, `model.layers.12.input_layernorm`,
  `lm_head` — all in view.
- Pressing ▶ at the end **restarts from the beginning** (op 340/340 → op 13/340).
- Speed selector is accurate: measured **14.0 / 56.0 / 7.0 ops/s at 1× / 4× / 0.5×**, matching
  `NODES_PER_SEC = 14`.

**Layer / head controls** — follow-playhead ON walks layers 0→7 during playback; unchecking it
and setting layer 3 **holds layer 3 for the whole run**; dragging the layer slider auto-unchecks
"follow playhead" as documented. Clicking head tiles 0/3/6 selects them, and the selected-tile
highlight always matched the enlarged heatmap caption (`head 6 · rows attend to columns`).
The enlarged heatmap uses real token labels on hover (`are · <|im_start|> · 0`).

**Model picker** — exactly 4 curated options, **no free-text model input exists** (`[]` inputs in
the picker). All four switch coherently: meta, op count, layer max, head count and template chip
all update together (Qwen 24L/14H/340 ops; gpt2 12L/12H/149 ops; SmolLM2-135M 30L/9H/424 ops;
SmolLM2-360M 32L/15H/452 ops). A rejected/failed pick is impossible to trigger from the curated
list. The `issue #4` link is live: **HTTP 200**, real issue titled *"Architecture Explorer:
expand model support beyond the curated list"*.

**Base-model note** — shown for gpt2 only (`This model has no chat template — it is a base
model…`) and absent for all three instruct models, driven by the real `chat_template_used` flag.
Correctly gated.

**Chat** — replies are coherent for the default model at all three temperatures:
- T=0: `"The capital of France is Paris."` (finish: `eos`, 8 tokens), every `p = 100.0%` (greedy).
- T=0.8 / T=1.6: fluent multi-sentence continuations; sampled probabilities drop as expected
  (first token `p` 94.5 % → 38.1 %).
- A system prompt genuinely reaches the model: with *"You are a pirate…"* the top-5 shifted to
  `"The" 49.9% · "Ah" 16.2% · "Oh" 11.2%`.
- **Zero probability anomalies** across all runs: every `p` in 0–100 %, every tooltip had exactly
  5 alternatives, no top-5 summed above 100 %. Special tokens render as `im_end` pills, not raw.

**Inspector** — clicked embedding, linear, rmsnorm, lm_head, and four parameterless ops.
Real numbers throughout, e.g. `q_proj`: `whole matrix 896 × 896 · downsampled (strided mean)`,
`min -1.227 · max 1.172 · μ -0.0000169 · σ 0.06674`. Zoom fetches an exact window
(`rows 75579–75642 · cols 416–479 of 151936 × 896 · exact values`) with pan buttons. Tied weights
are labelled (`weight [151936 × 896] tied → model.embed_tokens.weight`). **Esc closed the
inspector every single time** (9/9). Parameterless nodes behave sensibly — explainer note, no error.
Note the 136 M-param tensors take ~2–6 s to render, covered by a shimmer skeleton.

**Graph/trace integrity** — verified via the API that the "every op is a clickable node" claim is
literally true: **340 graph nodes ↔ 340 traced activations, a perfect bijection** (0 activations
missing from the graph, 0 graph nodes never activated).

**Diagram caption claims** — all verified: scroll-to-zoom changes the viewBox, drag pans it,
`▸/▾` headers expand (18 → 32 nodes) and collapse back (→ 18), clicking a collapsed placeholder
expands it, Esc closes the inspector.

**Error paths / edge cases** — empty prompt → clean empty state (*"Type a prompt on the left…"*)
and the Generate button correctly disables; whitespace-only → same; emoji-only → traces without
error; long prompt → truncation genuinely disclosed (see L6); 8 rapid keystrokes fired
**exactly 1** trace request (debounce + abort work); rapid tab switching and rapid model
switching both settle on the correct final state with no error and no orphaned data.

**390 px** — `docScrollW == innerW == 390`, **no horizontal overflow, zero offending elements**,
and every control reachable and full-width (model select, prompt, generate all 292 px wide;
play button and follow checkbox on-screen). See `6-g-390-top.png`.

**Console** — clean across all 10 phases: no errors, no unhandled rejections, no failed requests.
The only two console entries in the entire run were WebGL performance *warnings*
(`GPU stall due to ReadPixels`) emitted by the **Geometry** tab while I was rapid-switching tabs,
not by the Architecture tab.

---

## Suggested priority

1. **H1** — rescale the residual-stream chart (log scale / percentile max / exclude the sink token).
2. **M2** — show all heads without clipping, or stop saying "all N heads".
3. **M3** — clear or dim `graph`/`trace`-derived UI at the *start* of a model switch.
4. **M4** — qualify the `p =` label (e.g. `p (sampled @ T=0.8) = 94.5% · model top-5: …`).
5. L5–L8 as polish.
