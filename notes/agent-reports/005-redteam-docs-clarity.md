# Red team: does the Info tab + in-tab explainers actually fix "I can't tell what I'm looking at"?

Reviewed 2026-07-26, ~15:25 EDT. Snapshot caveat: the files were being edited *during* this
review (`InfoTab.svelte` mtime 15:24, one minute before my final re-read; `GeometryLab.svelte`
15:21; `ArchitectureExplorer.svelte` 15:12). All quotes below were re-verified against the
on-disk files at 15:25. Line numbers past `InfoTab.svelte:360` may drift by ±1.

Verification performed: read all 8 named files plus `GeoScene.svelte`, `ArchChat.svelte`,
`ArchModelPicker.svelte`, `TokenStrip.svelte`, `AttentionView.svelte`, `ExportBar.svelte`,
`StaticBadge/StaticNotice`, `staticUx.ts`, `stores.ts`, `dataClient.ts`, `app.css`,
`tests/e2e/docs.spec.ts`; ran `npm run check` (1119 files, **0 errors, 0 warnings**);
verified `arxiv.org/abs/2607.13295` resolves (Latifi Jebelli, *On Transformer Dynamics*,
math.CO, July 2026); checked backend `arch/trace.py` and the frozen contract
`specs/002-.../contracts/api.md` for the `truncated` field.

**Verdict.** The writing is unusually good — precise, self-skeptical, and in several places
(the tangency projection, the sampler-vs-displayed-probability distinction, the off-scale
striping) it does exactly what the colleague needed. But the fix does not land, for three
structural reasons and one honesty regression:

1. Everything that points a confused reader at the explanation is itself hidden — behind a
   collapsed `<details>`, or behind a tab labelled with one generic word, or (in the Geometry
   Lab) behind a multi-minute training gate that renders the explainers unmounted.
2. The two most-used controls in the app — the Geometry Lab's **prompt** and the **export
   bar** — are documented nowhere, and the prompt is what makes the next-next field
   *prompt-conditioned*, the single fact a researcher most needs and is never told.
3. The pictures encode things the prose does not: arrow lengths are silently renormalized
   per-payload and clipped, which is the same sin the residual-norm chart goes out of its way
   to avoid.
4. **New** text in the Architecture explainer asserts two capabilities the deployed static
   site does not have, contradicting a `{#if STATIC_MODE}` paragraph 20 lines above it.

---

## HIGH

### H1 — The Architecture explainer promises capabilities the deployed site does not have
`code/frontend/src/viz/arch/ArchitectureExplorer.svelte:340-350` (new in this change)

> **Model** — swaps the whole tab to a different real model. First load of a new one
> downloads and traces it (10–60 s), then it is cached.

> **Prompt** and **system prompt** — retraced 400 ms after you stop typing.

Neither is true at `https://context-lab.com/llm-geometry/`. In static mode the graph is a
precomputed JSON fetch (`src/lib/staticClient/arch.ts`), and an off-example prompt produces
the designed `StaticNotice` miss instead of a trace (`ArchitectureExplorer.svelte:193-197`
→ `ArchTracePanel` `staticNote`). Both bullets are rendered **unconditionally**, 20 lines
below a `{#if STATIC_MODE}` paragraph in the same `<header>` that says the opposite
(`:279-283`). The page contradicts itself within one screen, and it does so in the direction
of overclaiming — precisely what `CLAUDE.md`'s "never fabricated, never silently degraded"
rule exists to prevent. This is a regression introduced by the documentation change.

**Fix.** Branch both bullets:
```svelte
{#if STATIC_MODE}
  <li><b>Model</b> — picks another model from the curated catalog. Its graph and example
  traces were produced by the real backend at build time; the weights are read live from
  HuggingFace's CDN.</li>
  <li><b>Prompt</b> and <b>system prompt</b> — tokenized live in your browser and used for
  chat. The op-by-op trace exists only for the example prompts in the dropdown above.</li>
{:else}
  …existing two bullets…
{/if}
```

### H2 — The Geometry explainers are unmounted during the minutes a first-timer is most confused
`code/frontend/src/viz/geo/GeometryLab.svelte:349-363`

```svelte
{#if phase === "boot" || phase === "training"}   … Progress …
{:else if phase === "error"}                     … Try again …
{:else}
  <div class="explainers">   <!-- line 363 -->
```
Against the full stack the first open runs a real training job — this repo's own e2e allows
**220 s** for it (`tests/e2e/docs.spec.ts:71-74`). For that entire window the visitor sees a
progress bar reading "Training the tiny transformer (once — it's cached forever after)" and
*nothing else*: no equations, no "what is a GeoTransformer", no Info-tab pointer. The
`phase === "error"` branch is a hard dead end for the same reason. This is the exact moment
the colleague's confusion happens, and it is the one moment the new documentation is absent.

**Fix.** Hoist `<div class="explainers">` (or at minimum the "Full notation … Info tab"
line, `:469-472`) above the `{#if phase …}` block. Nothing in the explainers depends on
`spec`, `field`, or `trace`.

### H3 — Arrow lengths are silently renormalized and clipped, while the prose invites quantitative reading
`code/frontend/src/viz/geo/GeoScene.svelte:300-305, 315, 356`

```ts
function lengthScale(mags: number[]): number {
  const p90 = sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.9))] || 1;
  return 0.34 / Math.max(p90, 1e-6);
}
…
const len = m < 1e-9 ? 0 : Math.min(0.55, m * s);   // retarget()      — vocab arrows
const len = mag < 1e-9 ? 0 : Math.min(0.6, mag * s); // retargetForces() — amber arrows
```
`lengthScale` is recomputed on **every** field payload, **separately** for the two arrow
classes. Consequences no text states:

- Multiplying `W_V` by any positive constant yields a **pixel-identical** thin-arrow field.
  Yet `InfoTab.svelte:279-281` calls W_V edits "the largest visible effect of any edit" and
  `GeometryLab.svelte:443-446` says the thin arrows "literally *are* `W_V x`". They are
  `W_V x` rescaled by a hidden, data-dependent factor.
- Thin and amber arrows are on two independent scales, so their lengths are not comparable —
  and the text (`InfoTab.svelte:238`) introduces them as "two different things … drawn"
  without saying so.
- Arrows past the clamp are drawn saturated with **no marking**. The residual-norm chart
  three sections earlier stripes clipped bars and prints an "off-scale" count
  (`ArchTracePanel.svelte:385-391`) precisely because silent clipping is dishonest; the
  sphere does the silent version.
- The "radial pull projected out: X max" badge is in model units while nothing on the sphere
  is, so the number the docs correctly insist on showing cannot be checked against the
  picture it is supposed to falsify.

**Fix.** One sentence in the fields explainer and in `InfoTab`'s force section: "Arrow length
is *relative within one field*: each field is scaled so its 90th-percentile magnitude is a
fixed on-screen length, and the longest are clipped — so compare directions and relative
lengths, not absolute ones, and never across two renders." Then mark clipped arrows (e.g.
half-opacity head) and print `max_i‖F_i‖` next to the residual badge so the badge is
dimensionally meaningful.

### H4 — "the logit lens": the stated *reason* is wrong, and "exactly" skips the layer norm
`code/frontend/src/viz/info/InfoTab.svelte:210-214`; `GeometryLab.svelte:389-393`

> Because the unembedding is tied, reading out at an intermediate layer is exactly the
> **logit lens**

Tying is not why. The logit lens (nostalgebraist, 2020) is *the unembedding applied to
intermediate residual streams*; it is defined and routinely used for untied models. What
tying actually buys **this tab** is different and more interesting: it places the readout
direction for token *u* at the same point of `S²` as *u*'s embedding, which is the only
reason the answer can be drawn as an arrow *to another point on the same sphere*. Separately,
the canonical logit lens applies the model's **final layer norm** before unembedding; this
model has none, so "exactly" is true only because LN was removed — and a reader in this area
will raise the missing LN as their first objection. As written, the sentence hands them a
non sequitur plus an unqualified "exactly".

**Fix.**
> Reading out at an intermediate layer is the **logit lens**: `E z⁽ˡ⁾` asks what the model
> would predict if it stopped thinking after layer ℓ. (The usual final layer norm has nothing
> to apply here — this model has none.) Because the unembedding is *tied*, those readout
> directions are the embedding points themselves, which is what lets the answer be drawn as
> an arrow to another point on the same sphere. That is what the **layer** selector does in
> next-next mode.

### H5 — The attention-sink paragraph conflates two phenomena and misattributes both to instruct tuning
`InfoTab.svelte:129-135`; `ArchTracePanel.svelte:216-222`

> instruct-tuned models park an enormous-norm **attention sink** on the first token

Two errors a domain reader catches on sight:

(a) The chart plots `‖residual stream‖` — a **massive activation** (Sun et al., *Massive
Activations in Large Language Models*). The **attention sink** (Xiao et al., *StreamingLLM*)
is the *attention-mass* phenomenon: a large share of every row of `A` landing on position 0.
They co-occur and are causally linked, but the thing being drawn is the norm, not the sink.

(b) It is not a property of instruction tuning. Base decoder-only LMs show it too; it tracks
having a fixed first token, not SFT/RLHF. `gpt2` — a base model — is in this app's own
curated menu, so a reader can falsify the sentence in two clicks. That is corrosive on a page
whose entire pitch is "we do not overclaim".

**Fix.**
> Most decoder-only LMs park a huge-norm **massive activation** on the first token (and often
> on delimiters). It is the residual-stream face of the well-known **attention sink**, where a
> large share of every attention row lands on position 0 — which you can see in the head grid
> immediately to the left.

---

## MEDIUM

### M1 — The prompt, the Geometry Lab's most-used control, is documented nowhere; the next-next field is never said to be prompt-conditioned
`InfoTab.svelte:272-310` ("What you can change, and what it does"); `GeometryLab.svelte:442-468`

The Info table's Control column is: `W_V`/embedding, `W_Q`/`W_K`, `W_O`, `layer`,
`temperature`/`arrows per point`. The geo explainer's list is: W_V/embedding, W_Q/W_K, W_O,
Presets, Fine-tune/Train-from-scratch. **Neither mentions the prompt box** that sits directly
below the field controls (`GeometryLab.svelte:536-551`) — even though the prompt drives the
green path, every amber `F_i`, the whole attention panel, and the *entire* next-next field,
whose construction is literally "append `v` to **your prompt**" (`InfoTab.svelte:217-220`).

The consequence is the review's most important comprehension gap: a reader is never told that
the next-next "vector field" is **prompt-conditioned** — that it is not an autonomous field on
`S²`, that it changes completely when the prompt changes, and that the arrow tail `E[v]` is the
token's *embedding*, not the residual-stream state that produced the prediction. Calling it a
"vector field" without that caveat is exactly what will mislead a dynamical-systems-literate
reader.

**Fix.** Add a first table row and a first geo bullet:
> **prompt** — the field is *conditioned on it*. Each arrow answers "append this token **to
> this prompt**", so retyping the prompt redraws all 1003 arrows. It also sets the green path,
> the amber aggregate forces, and the attention map below. (The arrow's tail is the token's
> embedding `E[v]`, not the residual state `z` the model was actually in — the tail is where
> the token *lives*, not where the model *was*.)

### M2 — The colour key is printed in the wrong colours
`InfoTab.svelte:240-243` + `:559-562`; `GeometryLab.svelte:413-416`

```svelte
<b>thin arrows</b> — the per-point field … <b>amber arrows</b> — the aggregate force …
```
```css
.eq b { color: var(--accent-2); font-weight: 600; }   /* #b794f6 — violet */
```
So the phrase "**amber arrows**" is rendered **violet** — and `#b794f6` is exactly
`COL_HI` (`GeoScene.svelte:41`), the colour of the high-probability *next-next* arrows, i.e.
the other field. In `GeometryLab.svelte:413-416` the same two labels fall through
`Explain`'s `.body :global(b)` and render plain white. Meanwhile the caption 160 lines lower
does it correctly with `.force-key { color: #ffb454 }` (`GeometryLab.svelte:573, 787-789`).
A legend that names colours must use them; using the *other* field's colour is worse than
using none.

**Fix.** Replace the `<b>` legend labels with colour-matched spans
(`#ffb454` amber, the `COL_DIM→COL_HI` ramp for the vocab field, `var(--good)` for the green
path) and delete the blanket `.eq b { color: var(--accent-2) }`.

### M3 — New copy instructs a mouse-only interaction with no keyboard or touch path
`ArchitectureExplorer.svelte:277-278`; `GeometryLab.svelte:326`

> generate a reply and **hover** a token for its probability and the alternatives it beat
> Each dot is a token; **hover it** for its word.

Every one of these tooltips is `onmousemove`/`onmouseleave` on a `role="note"` element with
**no `tabindex`** (`ArchChat.svelte:159-164`; `ArchTracePanel.svelte:267-289, 399-407`;
`GeoScene.svelte:186-207`, `pointermove`). They are unreachable by keyboard and effectively
unreachable on touch. The colleague who opens the link on a phone is now *instructed* to
perform an interaction that device cannot do, to obtain information available nowhere else.

**Fix.** Add `tabindex="0"` + `onfocus`/`onblur` mirroring `showTip`/`hideTip` to the token
chips and reply tokens; show the sphere tip on `pointerdown` as well as `pointermove`; and
state the fallback ("the same numbers are in the top-10 panel").

### M4 — `display: block` on `<table>` strips the table semantics and traps the scroller from the keyboard
`InfoTab.svelte:604-611`

```css
.tbl { width: 100%; border-collapse: collapse; … display: block; overflow-x: auto; }
```
`display: block` on a `<table>` removes its implicit `table` role in WebKit/Blink, so the
`<th>` headers of "What you can change" and "What's real" stop associating with their cells
for assistive tech. And the resulting scroll container has no `tabindex="0"`, so a
keyboard-only user on a 390 px phone cannot scroll to the third column — the column the prose
now explicitly directs them to ("the **right-hand column** describes what you have",
`:359-362`).

**Fix.** Keep `display: table` and move the overflow to a wrapper:
`<div class="tblwrap" tabindex="0" role="region" aria-label="capability comparison">`.

### M5 — `.eq` blocks wrap instead of scrolling, so every equation garbles on a phone
`InfoTab.svelte:544-555`; `Explain.svelte:121-132`

Both set `overflow-x: auto` but never `white-space: nowrap`. Lines are `<br />`-separated, so
the browser reflows *within* a line and the horizontal scroller never engages. At 390 px the
content box is ≈290 px; at `0.82rem` mono that is ~37 characters, while
`z_i ← z_i + W_outᵀ gelu(W_inᵀ z_i + b_in) + b_out` is ~48 — it breaks mid-formula with no
hanging indent and reads as two equations. The generation equation is worse: its continuation
line is indented with 16 literal `&nbsp;` (`InfoTab.svelte:153`), which after wrapping lands
at an arbitrary position.

**Fix.** `white-space: nowrap` on `.eq` in both files (then `overflow-x: auto` actually
scrolls), `tabindex="0"` on the scroller, and replace the `&nbsp;` run with
`.eq .cont { display: inline-block; padding-left: 2ch }`.

### M6 — The reference page has no URL: it cannot be linked, bookmarked, or deep-linked
`src/lib/stores.ts` (bare `writable<View>("architecture")`, no hash sync);
`InfoTab.svelte:23-25, 39-43`

```ts
function jump(id: string): void {
  document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
}
```
The author cannot send the colleague `…/llm-geometry/#info` or `#geo`; a reload always lands
on Architecture; Back does nothing after a ToC jump; and because the ToC entries are
`<button>` + `scrollIntoView` rather than `<a href="#geo">`, **focus never moves to the
target**, so a screen-reader user is scrolled while their reading cursor stays behind. There
is no `prefers-reduced-motion` guard anywhere in `src/` for the smooth scroll (or for the
`breathe`/`slide`/`pulse`/`fade` keyframe animations).

**Fix.** Sync `view` ↔ `location.hash`; make the ToC pills real `<a href="#…">` (fragment
navigation gives focus movement, history, and shareable links for free); guard the smooth
behaviour behind `matchMedia("(prefers-reduced-motion: reduce)")`.

### M7 — Every pointer to the Info tab is inside a collapsed disclosure
`Explain.svelte:21` (`open = false` default); `ArchitectureExplorer.svelte:362-366`;
`GeometryLab.svelte:469-472`; `App.svelte:22-29, 56`

The "see the **Info tab**" links live in the *body* of an `Explain`, which is collapsed by
default; the e2e suite even pins that default (`docs.spec.ts:124`). The tab itself is labelled
with the single word "Info", third in the strip, and the blurb line under the strip describes
only the *active* tab. So the modelled user — lands on Architecture, is confused, does not
think to expand a grey disclosure row — receives **no signal at all** that a reference page
exists. The intervention is invisible to exactly the person it was written for.

**Fix (preserves the uncluttered goal).** Open the first explainer on each tab on first visit
and persist dismissal in `localStorage`; and/or add one always-visible line under the tab
strip — "New here? **Start with Info** →" — that self-hides once Info has been opened. Update
`docs.spec.ts:124` accordingly rather than keeping the assertion that guarantees the problem.

### M8 — The header chips are undecodable
`GeometryLab.svelte:331-345`

Renders `shipped checkpoint · corpus X`, `final loss 4.12`, `coverage 0.87`,
`field entropy 3.10`, and (when edited) an 8-hex chip. `InfoTab.svelte:326-327` states the
gate thresholds once, buried in the training paragraph — but never says *these chips are those
numbers*, never defines "coverage uniformity", and never gives the one datum that tells a
researcher whether the model learned anything: the uniform-baseline cross-entropy
`ln 1003 ≈ 6.91` to read `final loss` against. The 8-hex chip is never connected to the
content hash described at `InfoTab.svelte:313-316`.

**Fix.** `title=` on each chip, plus one Info sentence: "The chips under the Geometry heading
are the acceptance-gate numbers for the active checkpoint. Final loss is next-token
cross-entropy in nats — a uniform model over 1003 tokens scores ln 1003 ≈ 6.91. The
8-character chip is the first 8 hex digits of the active weights' content hash."

### M9 — Static dead end: the empty state points at a control that may not exist
`ArchTracePanel.svelte:246-255`; `ArchitectureExplorer.svelte:374, 162-164`

> Pick an example prompt on the left — its full forward pass, traced by the real backend,
> lands here.

The dropdown is gated on `{#if STATIC_MODE && tracePresets.length > 0}`, and the presets fetch
swallows failures into `tracePresets = []`. A CDN hiccup therefore leaves the reader staring
at an instruction about a control that is not on screen, with no error and no next step.

**Fix.** When `STATIC_MODE && tracePresets.length === 0`, render a `StaticNotice` naming the
failure with a retry and the README link, instead of the "pick an example" prose.

### M10 — The justification for the collapsed default is an unverified browser claim
`Explain.svelte:4-11`

> is found by the browser's in-page search even while collapsed (Chromium/Safari
> `hidden=until-found` semantics for `<details>` content). The default is collapsed …

Nothing tests this (find-in-page is not drivable from Playwright), and it is per-engine
behaviour the project does not control. It matters because it is the stated *reason* the
documentation ships hidden: if find-in-page does not reach the text in a reader's browser, the
collapsed default hides the docs with no compensating affordance. I cannot verify the engine
matrix from here and am not asserting which engines do or don't — the finding is that a
load-bearing behavioural claim is asserted with no evidence and no test.

**Fix.** Drop the claim to what is verifiable ("the content stays in the DOM while collapsed")
and carry the discoverability weight with M7's always-visible pointer instead.

---

## LOW

### L1 — "it **is** the representation" is one word too strong, and fights the rest of the page
`InfoTab.svelte:169-175`, `:65-68`

> What is on screen is not a view of the representation — it **is** the representation, at
> full rank, with nothing discarded.

What is on screen is a perspective rasterization of ℝ³ — a projection, just an
invertible-by-rotating one rather than a lossy statistical one. And it sits in tension with the
same page's own (correct) statements: the residual stream is **not** confined to the sphere —
that is the stated reason there is no layer norm (`:203-208`) — and the position embeddings
`p_i` are unconstrained 3-vectors (`:182-183`). "The whole embedding space is the sphere on
screen" (`:66`) elides that the representation space is ℝ³ and `S²` is only where the 1003
embedding **rows** are pinned.

**Fix.** "No dimensionality reduction was applied: the 1003 embedding rows are unit vectors in
ℝ³ and are drawn where they are. The only projection left is your screen's, and you can rotate
it away. The residual stream itself is *not* confined to the sphere — that radial excursion is
what the green path shows."

### L2 — `ArchTrace` is missing a contract field; the code compensates with a cast and a stale comment
`src/lib/dataClient.ts:351-357`; `ArchTracePanel.svelte:179-184`

`ArchTrace` has no `truncated`, although the frozen contract declares it
(`specs/002-interactive-model-explorer/contracts/api.md:248`) and the backend emits it
(`code/backend/src/llm_geometry/arch/trace.py:55,114`). The panel therefore does
`(trace as ArchTrace & { truncated?: boolean })?.truncated === true` under a comment claiming
the contract "is gaining" the field "so the UI works both before and after the backend field
lands". It landed. The risk is live: because the field is optional-by-cast, an export path that
dropped it would silently disable the "earlier tokens dropped ⋯" chip while
`InfoTab.svelte:138-143` promises that chip exists.

**Fix.** Add `truncated: boolean` to `ArchTrace`; delete the cast and the comment.

### L3 — `role="tablist"` on controls that are not tabs
`GeometryLab.svelte:479, 486`

```svelte
<div class="seg" data-testid="geo-mode" role="tablist">
  <button class:active={$geoFieldMode === "next_next"} …>next-next</button>
```
A `tablist` whose children have no `role="tab"`, no `aria-selected`, and no associated
`tabpanel`: a screen reader announces a tab list containing nothing it recognises, and the
active field/layer is not conveyed at all. `FinetunePanel.svelte:130-143`,
`TrainPanel.svelte:209-219`, `AttentionView.svelte:23-27` do set `role="tab"`/`aria-selected`
but still have no `tabpanel`, no `aria-controls`, and no arrow-key handling — a half-built
pattern is worse than none.

**Fix.** Use `role="radiogroup"` + `role="radio" aria-checked` for the segmented pickers, or
drop the roles and put `aria-pressed` on the buttons (they already behave as toggles).

### L4 — Explainer titles are invisible to heading navigation
`Explain.svelte:25-29`

The title is a bare `<span class="title">` in the `<summary>`, so the outline runs
`h2` ("Architecture Explorer") → `h3` ("Processing breakdown") with both explainers absent.
A screen-reader user skimming by heading — the standard way to skim — cannot find "How to read
the diagram" at all.

**Fix.** `<span class="title" role="heading" aria-level="3">{title}</span>`, or an actual
`<h3>` inside the `<summary>`.

### L5 — `summary` cannot wrap on a narrow phone
`Explain.svelte:41-52, 77-82`

`summary { display: flex; align-items: baseline; gap: .5rem }` with no `flex-wrap`; `.hint`
has `min-width: 0` but `.title` has no `flex-shrink: 0`. At 390 px, "What can I change, and
what happens" plus the hint "which matrices move which picture — and what barely moves it"
share ~290 px, so *both* shrink and *both* wrap, producing a ragged two-column block in which
the title no longer reads as a title.

**Fix.** `@media (max-width: 560px) { summary { flex-wrap: wrap } .title { flex: 0 0 auto }
.hint { flex: 1 1 100% } }`.

### L6 — The notation table is a layout table
`InfoTab.svelte:81-88`

Four columns alternating symbol/definition, no `<thead>`, no `<th>`. It is a glossary, not
tabular data; assistive tech reads "V, vocabulary size, T, number of tokens in the prompt" as
one row spanning two unrelated pairs.

**Fix.** `<dl>` with `display: grid; grid-template-columns: auto 1fr auto 1fr` — identical
visuals, correct semantics, and it collapses to two columns on a phone for free.

### L7 — Screen elements visible on first load that no text anywhere names
Each of these is on screen before any explainer is opened:

| element | where | status |
|-|-|-|
| model-picker status badge (`ready` / `resolving` / `error`) | `ArchModelPicker.svelte:99-101` | never described |
| Weight Lab `source` badge | `WeightLab.svelte:213` | never described |
| **Export bar** above the sphere (`Export  PNG`) | `ExportBar.svelte`; `GeometryLab.svelte:576` | never mentioned — and "export this figure" is the first thing a researcher wants |
| the **white** dot ending the green path (= last token) | `GeoScene.svelte:507` (`isLast ? 0xffffff`) | never described |
| token-dot colour `#6ea8fe` and the `#2a3a6e → #b794f6` arrow ramp | `GeoScene.svelte:39-42` | only "brighter is more probable"; a near-zero-weight arrow at `#2a3a6e` is essentially invisible against the `0x1d2745` sphere shell |
| `↵` substituted for newline in token chips | `ArchTracePanel.svelte:175-177` | never described |
| the per-chip token **id** under each arch token | `ArchTracePanel.svelte:281` | never described |
| the Geometry Lab boot/training gate | `GeometryLab.svelte:349-355` | Info's training section describes the checkpoint as already shipped and never mentions that opening the tab may train it |

**Fix.** A compact legend block in Info's geo section (swatch + meaning, one row each) and one
sentence per orphan control.

### L8 — `ArchitectureExplorer.svelte` contains literal NUL bytes, making it invisible to `grep`
`ArchitectureExplorer.svelte:228` — `const key = \`${m}\0${p}\0${sp}\`;` (verified at byte
offset ≈8415; pre-existing, present in `HEAD`).

Intentional as a collision-proof delimiter, but `grep`/`ripgrep` classify the file as
**binary** and skip it: a repo-wide `grep -rn '<h2'` silently omits this file's heading — it
did so during this review, and any CI lint or copy-audit implemented with grep will silently
pass over the entire Architecture tab.

**Fix.** `const key = JSON.stringify([m, p, sp]);` — same guarantee, plain text.

### L9 — Static-mode copy in the graph loader
`ArchitectureExplorer.svelte:438-443` always renders "downloading + tracing **{model}**… first
load of a new model can take 10–60 s — the graph is built from a real traced forward pass, then
cached". In static mode nothing is downloaded or traced; a precomputed JSON is fetched. Same
class as H1, lower impact because it is transient.

---

## What is genuinely good (do not regress it)

- `InfoTab.svelte:253-270` (the tangency subtlety) and the paired `geo-residual-badge`
  (`GeometryLab.svelte:529-531`) are a model of how to document a visualization: the claim,
  the reason it is subtle, the correction applied, and the number that could falsify it —
  including the admission that an earlier version was 59° wrong.
- `InfoTab.svelte:156-163` ("Only the draw is filtered") resolves the exact ambiguity that
  makes sampled-probability displays untrustworthy, and `ArchChat.svelte:88-98` implements it.
- The "What's real, and where it runs" table is the right instrument, and the
  `{#if STATIC_MODE}` sentence telling the reader *which column applies to them* is a nice
  touch — undermined only by H1 contradicting it two clicks away.
- `docs.spec.ts` pinning documented numbers against the live API (`/api/geo/spec`, real slider
  bounds) is the correct defence against doc rot and should be extended to cover whichever
  strings H1's fix introduces.
