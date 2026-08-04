# Red-team D — Info tab, app shell, documentation prose, accessibility

Agent D of the four-tab campaign (`notes/2026-08-04-redteam-brief.md`).
Slice: `viz/info/InfoTab.svelte`, `lib/Explain.svelte`, `App.svelte`, `lib/stores.ts`,
the orientation prose and `Explain` deep-dives in all four tabs, `tests/e2e/docs.spec.ts`,
`README.md`, `CLAUDE.md`, `specs/007-vacancy-transform-field/*.md`.

Both modes attacked: the deployed static build at https://context-lab.com/llm-geometry/
and the running local full stack (`:5173` → `:8000`). No server was started or stopped;
all browser work was done with my own Node scripts against the repo's installed
Playwright, in the scratchpad, since deleted.

**Counts:** 0 critical · 0 high · 6 medium · 8 low.

The headline: **the numeric claims are in excellent shape.** I traced 60+ numbers from
the Info tab and all four tabs' prose to the constant or measurement they came from and
found **zero drift** — including every number `docs.spec.ts` does *not* pin. The defects
are in the surrounding layer: the shell's history model, silent loss of in-progress work,
two stale top-level documents, and the accessibility of the primary controls.

---

## Findings

### F1. Browser Back does not move between tabs, and `stores.ts` documents that it does
**Severity:** medium
**Where:** `code/frontend/src/lib/stores.ts:22-51`
**Reproduce:**
```
node scratch/shell.mjs http://localhost:5173     # goto '/', click Geometry, Lexicon, Info, then goBack()
```
**Observed:**
```
after click geometry -> http://localhost:5173/#geometry
after click lexicon  -> http://localhost:5173/#lexicon
after click info     -> http://localhost:5173/#info
after goBack -> about:blank visible view: []
```
On the live site, from a fresh load followed by two tab clicks:
```
url after 2 clicks: https://context-lab.com/llm-geometry/#lexicon
after Back: https://context-lab.com/llm-geometry/#architecture
```
i.e. Back skipped the two tabs entirely and returned to the previous *page load*.

**Expected:** `stores.ts`'s own docstring states the problem it is solving as
*"you cannot send a colleague a link to the Info tab, a reload always lands on
Architecture, and **Back does nothing**"* and then claims
*"URL -> store, so **Back/Forward and a pasted link both work**."*
Pasted links and reload do work (verified). **Back/Forward for tab navigation does not**,
because store→URL uses `window.history.replaceState`, which never pushes an entry:
```ts
// Store -> URL. replaceState, not a hash assignment: tab switching should not fill
// the back stack with every click, but the address bar must stay copyable.
window.history.replaceState(null, "", target);
```
The inline comment is honest about the design choice; the docstring three lines above
advertises the opposite behaviour. `hashchange` only ever fires for entries someone
else created. For a user who opened the deployed link in a fresh tab — the common case —
Back after browsing the tabs **leaves the site**.

Either behaviour is defensible; the documentation claiming the one that does not happen
is not. (`docs.spec.ts:41-63` tests deep links and reload, never Back.)
**Would it have thrown?** No.

---

### F2. Switching tabs mid-training silently destroys an in-progress Lexicon Lab run
**Severity:** medium
**Where:** `code/frontend/src/App.svelte:96-106` (`{#if $view === …}` unmounts the tab)
+ `code/frontend/src/viz/lex/TrainPanel.svelte:123-124`
+ documented condition in `viz/info/InfoTab.svelte:1060-1065`
**Reproduce:** live site → Lexicon → **Train from scratch** → after ~12 s click **Info**,
wait 5 s, click **Lexicon**.
**Observed:**
```
t+0s:  step 15/400 · loss 5.229 · lr 2.16e-4 · 2.0s
t+10s: step 114/400 · loss 3.317 · lr 2.98e-3 · 12.0s
--- switching to Info mid-run ---
after return, step marker: ["Train from scratch"]
  +5s … +50s after return: - done? false
```
The run is gone. The panel is back to its idle state. No error, no toast, no
"your run was cancelled" — the button simply reappears. The cause is deliberate and
clean (`onDestroy(() => { worker?.terminate(); })`, so there is no leak), but the
consequence is invisible.

A *completed* model does survive the round trip — I verified the model token
`03769632905e` was unchanged after Info → Lexicon — so only in-flight work is lost.

**Expected:** the Info tab's Known limits say only:
> "A model trained in the Lexicon Lab lives in that tab and nowhere else. There is no
> account and no server-side checkpoint, so **closing the page** ends the model unless
> you save the `.llmlex.json` bundle"

*Closing the page* is not the loss condition a user will hit. **Switching tabs** is —
and the app actively invites it: `App.svelte:88-92` shows a "New here? Start with
**Info** →" pointer on the landing tab, and every `Explain` deep-dive in the Architecture
and Geometry tabs ends with a button that navigates to the Info tab
(`docs.spec.ts:181`, `:210` assert those buttons work). The documentation affordance
destroys the work.
**Would it have thrown?** No.

---

### F3. `README.md` still describes a two-view app; the Lexicon Lab, the vacancy transform and the Info tab are absent
**Severity:** medium
**Where:** `README.md:4-19`, `:93`, `:97`
**Reproduce:** read the file; compare with the four tabs that ship.
**Observed:**
```
Two explorable views over real open-weights models — no mocks, no canned data:
1. **Architecture** — …
2. **Geometry** — …
```
and, in the layout section:
```
    └── src/{viz,lib}/        #   the two explorer views · typed API client + components
specs/                        # Spec Kit features (001 is superseded; 004 is current)
```
**Expected:** four tabs ship — the Lexicon Lab (feature 006), the vacancy transform
(007, which spans *both* the Lexicon Lab and the Architecture Explorer) and the Info tab
(005) are the entirety of the last three features and appear nowhere in the README.
`code/backend/src/llm_geometry/lex/` and `code/frontend/src/viz/{lex,info}/` are missing
from the layout tree, and `.specify/feature.json` reads
`{"feature_directory":"specs/007-vacancy-transform-field"}`, not 004. The deployed URL
is not mentioned at all. This is the first document a visitor reads and it under-describes
the shipped app by half.
**Would it have thrown?** No.

---

### F4. `CLAUDE.md` says "three-tab explorer" and names 005 as the active feature
**Severity:** medium
**Where:** `CLAUDE.md:28`, `CLAUDE.md:125-136` (the auto-managed `<!-- SPECKIT -->` block)
**Observed:**
```
28: The app is a **three-tab explorer** — two visualizations plus a reference tab — deployed at
126: Active feature: **005-explain-the-visualizations** — the Info tab and the in-tab
```
against
```
$ cat .specify/feature.json
{"feature_directory":"specs/007-vacancy-transform-field"}
```
**Expected:** four tabs; active feature 007. Line 28 is human-authored and stale by two
features; the SPECKIT block is tool-managed and was not regenerated when 006 and 007
landed. `CLAUDE.md` is what every future agent reads first, so this drift compounds —
an agent told there are three tabs will not go looking for the Lexicon Lab.
**Would it have thrown?** No.

---

### F5. The Geometry Lab's two primary controls are `role="tablist"` with children that are not tabs
**Severity:** medium (issue #7 is known; this is the concrete blast radius)
**Where:** `code/frontend/src/viz/geo/GeometryLab.svelte:500` and `:507`
**Reproduce:**
```
node scratch/accname.mjs '#geometry' 120000     # CDP Accessibility.getFullAXTree + DOM walk
```
**Observed:**
```
TABLISTS: [{"tid":"geo-mode","kids":[{"tag":"BUTTON","role":null,"sel":null,"ti":null,"txt":"next-next"},
                                     {"tag":"BUTTON","role":null,"sel":null,"ti":null,"txt":"force"}]},
           {"tid":"geo-layer","kids":[{"tag":"BUTTON","role":null,"sel":null,"ti":null,"txt":"full"},
                                      {"tag":"BUTTON","role":null,"sel":null,"ti":null,"txt":"0"}, …]}]
```
Source:
```svelte
<div class="seg" data-testid="geo-mode" role="tablist">
  <button class:active={$geoFieldMode === "next_next"} onclick={() => setMode("next_next")} …>next-next</button>
```
**Expected:** ARIA 1.2 requires a `tablist`'s owned elements to be `tab`. These are plain
buttons, so nothing carries `aria-selected` and the only signal of which field mode or
which layer is active is the `.active` class — i.e. a background colour. A screen-reader
user is told "tab list" and then given N unlabelled-state buttons: **they cannot determine
whether they are looking at the next-next field or the force field, or which layer is
selected.** These are the two controls the whole tab is about.

By contrast `TrainPanel.svelte:212-213`, `FinetunePanel.svelte:140-141` and
`AttentionView.svelte:25` *do* set `role="tab"` + `aria-selected` — so the pattern is
known in this codebase and these two were missed. (Those three are still incomplete
patterns: no `role="tabpanel"`, no roving `tabindex`, no arrow-key navigation — a lesser
issue I am folding in here rather than filing separately.)
**Would it have thrown?** No.

---

### F6. The top-level four-tab strip exposes no selected state at all
**Severity:** medium
**Where:** `code/frontend/src/App.svelte:76-85`
**Reproduce:** `node scratch/shell.mjs http://localhost:5173`
**Observed:**
```
NAV: {"role":null,"aria":null,"buttons":[
  {"t":"Architecture","role":null,"sel":null,"cur":null,"dis":false,"cls":"s-XsEmFtvddWTw active"},
  {"t":"Geometry","role":null,"sel":null,"cur":null,"dis":false,"cls":"s-XsEmFtvddWTw"}, … ]}
```
**Expected:** the `<nav class="tabs">` has no `role`, no `aria-label`, and the buttons
carry neither `aria-current` nor `aria-selected` — only `class="… active"`, styled as a
background gradient. A screen-reader user hears four ordinary buttons and has no way to
know which view is showing. This is the app's only navigation. One `aria-current="page"`
would fix it; it is genuinely absent rather than misused, which is why it is not covered
by issue #7's "misused `tablist` roles" wording.
**Would it have thrown?** No.

---

### F7. The Info tab's table of contents is mouse-only: it scrolls the viewport without moving focus
**Severity:** low
**Where:** `code/frontend/src/viz/info/InfoTab.svelte:28-30`, `:47-51`
```ts
function jump(id: string): void {
  document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
}
```
**Reproduce:** live site `#info` → activate the **Known limits** pill → press Tab.
**Observed:**
```
after TOC jump -> focus: {"tag":"BUTTON","txt":"Known limits","y":12134}
next Tab lands on: { tag: 'BUTTON', txt: 'Source & references', y: 349 }
```
**Expected:** focus stays on the TOC button while the viewport is 12 000 px away, so the
next Tab yanks the reader straight back to the top of the page. For a keyboard-only or
screen-reader user the TOC does nothing useful: the reading position never moves, only
the sighted viewport does. WCAG 2.4.3. The target headings have `scroll-margin-top` but
no `tabindex="-1"` and are never focused.
**Would it have thrown?** No.

---

### F8. An unknown or mis-cased hash silently shows Architecture and leaves the wrong URL in the address bar forever
**Severity:** low
**Where:** `code/frontend/src/lib/stores.ts:26-31`, `:39-49`
**Reproduce:** `node scratch/hash.mjs` against the live site.
**Observed:**
```
cold #not-a-tab -> url https://context-lab.com/llm-geometry/#not-a-tab | shows arch: 1 | info: 0
after clicking active Architecture, url still: https://context-lab.com/llm-geometry/#not-a-tab
cold #Info (capital I) -> info: 0 arch: 1
```
**Expected:** silently falling back to Architecture is a reasonable choice for a garbage
hash (and `docs.spec.ts:51-58` deliberately pins it). Two consequences are not:

1. The URL is never corrected. Because the fallback sets the store to `"architecture"`
   and Svelte's `writable` skips a `set` to an equal value, clicking the Architecture tab
   does not fire the subscriber, so `#not-a-tab` survives every interaction. The comment
   at `stores.ts:37` promises "the address bar must stay copyable" — here what is copyable
   is a URL that lies about what the recipient will see.
2. Matching is case-sensitive (`isView(h)` against a lower-case list), so a user who types
   or a mail client that capitalises `#Info` gets Architecture with no explanation.

**Would it have thrown?** No.

---

### F9. The Architecture model picker `<select>` has no accessible name
**Severity:** low
**Where:** `code/frontend/src/viz/arch/ArchModelPicker.svelte:99-103`
**Reproduce:** `node scratch/accname.mjs '#architecture' 7000` (CDP AX tree).
**Observed:**
```
#architecture UNNAMED interactive AX nodes: 1 [{"role":"combobox","id":183}]
   combobox SELECT testid=arch-model-select cls=s-lPLSXBJ21ZZf ph=-
```
Source: the visible label is a sibling `<span class="label">Model …</span>` with no `for`
/ `id` pairing and no `aria-label` on the `<select>`.
**Expected:** a screen reader announces "combo box, Qwen2.5 0.5B Instruct (default)" with
no indication of what is being chosen. **This is the only unnamed interactive control on
any of the four tabs** — I ran the same AX sweep on `#geometry` and `#lexicon` and both
returned `UNNAMED interactive AX nodes: 0`. Everything else is named via placeholder or
wrapping `<label>`. One `aria-label="Model"` closes it.
**Would it have thrown?** No.

---

### F10. Two Geometry Lab controls are documented under "The Architecture Explorer"
**Severity:** low
**Where:** `code/frontend/src/viz/info/InfoTab.svelte:149-175` — the `<h4 class="sub">Smaller
things on screen, named</h4>` sits under `<h3 id="arch">The Architecture Explorer</h3>`
(line 118) and before `<h3 id="geo">` (line 243).
**Observed:** two of that list's five bullets describe the Geometry Lab:
> "The **Export** control above the Geometry sphere writes what you are looking at to a
> PNG, including the WebGL canvas."

> "In the weight lab, the `source` badge says where the displayed matrix came from —
> the trained checkpoint, or the preset/edit that replaced it."

**Expected:** `ExportBar` is imported only by `src/viz/geo/GeometryLab.svelte:27` and
mounted only at `:602`; `WeightLab.svelte` exists only in `src/viz/geo/`. Neither control
exists in the Architecture Explorer. A reader using the TOC to find out what the
Architecture tab shows is told about two controls that are not there, and a reader of the
Geometry section never learns about the Export button at all.
**Would it have thrown?** No.

---

### F11. Citation support: the 1936 Dolch article is credited with the graded sublists and the noun list, which I could not verify came from it
**Severity:** low
**Where:** `code/frontend/src/viz/info/InfoTab.svelte:548-551` and `:1106-1110`
**Observed:** the prose says
> "**Dolch** — the graded sight-word lists Edward William Dolch published in **1936**, a
> real pedagogical word list still in use. The five cumulative budgets are
> **40 / 92 / 133 / 220 / 314** words"

and the reference reads
> "E. W. Dolch, *A Basic Sight Vocabulary*, The Elementary School Journal 36(6):456–460,
> 1936, doi:10.1086/457353 — the source of the Lexicon Lab's prescribed word budgets."

**What I could verify:** the citation is well formed (vol. 36 no. 6, February 1936,
pp. 456–460 — confirmed against the publisher page and secondary sources) and it is
unambiguously the source of the **220-word service list**. `Santa Claus` being in Dolch's
noun list — which the "Why 314 and not 315" section rests on — is confirmed by the
Wikipedia article on the Dolch word list, which uses it as its example of a dated noun.

**What I could not verify:** that the *graded* sublists (pre-primer 40 / primer 92 /
first grade 133) and the separate 95-noun list appeared in that five-page 1936 article
rather than in Dolch's *Problems in Reading* (1948). The article is paywalled; Wikipedia
states the list "was first published in a journal article in 1936" and later "published in
his book *Problems in Reading* in 1948" but, in its own words as fetched,
"does not explicitly clarify whether the grade-level divisions … and the separate 95-word
noun list originated in the 1936 journal article or the 1948 book." Since the graded
breakdown is what the Lexicon Lab's five nested budgets actually are, and since a
five-page article is a tight fit for 220 service words + 95 nouns + a grade-by-grade
partition, the attribution is plausible but unestablished. **I don't know** which it is —
somebody with library access should check page 456–460 before the sentence stands.
**Would it have thrown?** No.

---

### F12. `specs/007` FR-706 names an `avoid` parameter the implementation deliberately refuses to have
**Severity:** low
**Where:** `specs/007-vacancy-transform-field/spec.md:56`
**Observed:**
> "**FR-706** A nonce never collides with a real type of the corpus (`avoid`, contract §5.2)."

against `code/backend/src/llm_geometry/lex/vacancy.py` (`build_vacancy_map` docstring):
> "**The domain is avoided implicitly; there is no caller-supplied `avoid` parameter.**
> With one, the map depends on what the caller remembered to pass … The map is now a pure
> function of `(domain, seed, match_prosody)`."

**Expected:** the requirement's parenthetical names the exact API the implementation
argues at length must not exist (and whose existence in the source project is called out
in the Info tab as a defect: *"it accepts an `avoid` parameter and never passes one"*).
The requirement's substance holds; only the name is wrong. Worth fixing in the spec so a
future reader does not add the parameter back to satisfy it.
**Would it have thrown?** No.

---

### F13. `lex/config.py` states the corpus at 19,050 tokens; the shipped tab reports 19,071
**Severity:** low
**Where:** `code/backend/src/llm_geometry/lex/config.py:62` (comment) vs the live tab
**Observed:** the config comment justifying `DEFAULT_CTX = 32` says
> "QUALITY. 19,050 tokens is a small corpus and a 64-token window spans several
> unrelated rhymes."

The Lexicon Lab prose says, and the deployed tab printed during a real run:
```
held-out 2.912 · 19,071 tokens (the budget's wor…
```
matching `16,000` word tokens + `3,071` `<eos>` (both of which I recomputed from the
shipped corpus: `total word tokens 16000`, `lines 215 / 3071`).
**Expected:** 19,071. **I don't know** what measurement produced 19,050 — a 21-token gap
suggests a different `<bos>`/`<eos>` convention or an earlier trim, not a typo. It is a
code comment rather than user-facing prose, hence low, but it is the kind of number the
project's own rule says must be transcribed from a source.
**Would it have thrown?** No.

---

### F14. `docs.spec.ts` does not pin the shell behaviours it depends on
**Severity:** low
**Where:** `code/frontend/tests/e2e/docs.spec.ts:41-63`
**Observed:** the URL test covers deep links, mid-session unknown hashes, cold reload, and
that clicking a tab updates the URL. It does not cover Back/Forward (F1), does not assert
the URL is *corrected* after an unknown hash (F8), and does not cover case sensitivity.
The `stores.ts` docstring's Back/Forward claim therefore has no test behind it and was
free to become false.
**Expected:** the file's own stated purpose is that "the documentation states NUMBERS, and
numbers rot" — the same argument applies to the behaviours the documentation states.
**Would it have thrown?** No.

---

## What I tried that came back clean

### Every numeric claim I could trace — 60+, zero drift

I extracted every number from the Info tab and, by rendering all four tabs and expanding
every `<details>`, from all four tabs' orientation prose and `Explain` deep-dives, then
traced each to the constant or measurement it came from. **Including the ones
`docs.spec.ts` does not pin.** All correct:

| Claim (prose) | Source | ✓ |
|-|-|-|
| trace prompt "The quick brown fox…", "capped at 12 tokens" | `arch/graph.py:32-33` `GRAPH_TRACE_TEXT`, `_GRAPH_TRACE_MAX_TOKENS = 12` | ✓ |
| "gated at **1.5B parameters**", "20% safety margin" | `config.py` `ARCH_MAX_PARAMS = 1_500_000_000`; `arch/gate.py:87-90` `ARCH_MAX_PARAMS * 0.8` | ✓ |
| "4096-cell budget", "at most 64×64 per head", "last **64** tokens", "128 new tokens" | `ARCH_WEIGHTS_MAX_CELLS=4096`, `ARCH_ATTENTION_MAX_SIDE=64`, `ARCH_DEFAULT_MAX_CONTEXT=64`, `ARCH_MAX_NEW_TOKENS=128` | ✓ |
| "top-k 50 ∩ top-p 0.9", "repetition penalty of 1.1" | `ARCH_TOP_K/TOP_P/REPETITION_PENALTY` | ✓ |
| "Retraced 400 ms after you stop typing" | `ArchitectureExplorer.svelte:205` `debounced(…, 400)` | ✓ |
| "4 layers, 1 head, no layer norm, MLP hidden width 12, context window 50, vocabulary 1003", `<unk>=0 <eos>=1 <pad>=2` | `geo/config.py` | ✓ |
| "30 epochs of Adam at lr 2e-2, batch 64, windows every 10 tokens", repulsion "weight 0.3" | `TRAIN_EPOCHS/LR/BATCH_SIZE/WINDOW_STRIDE`, `REPULSION_WEIGHT` | ✓ |
| "coverage uniformity ≥ 0.80", "entropy ≥ 2.0 nats", "maximum is ln 64 ≈ 4.16", "ln 1003 ≈ 6.91" | `MIN_COVERAGE_UNIFORMITY`, `MIN_FIELD_DIRECTIONAL_ENTROPY`, `SPHERE_BINS=64`; ln 1003 = 6.9107 | ✓ |
| "up to 500 steps, default 100, lr 1e-2"; "epochs slider runs 1–30 and starts at 12" | `FINETUNE_*`; `TrainPanel.svelte:256` `min=1 max=30`, `:29` `$state(12)` | ✓ |
| "at least 1000 distinct types or the run is refused" | `scratch.py:109` `if stats["n_distinct"] < VOCAB_WORDS` | ✓ |
| presets `identity, toeplitz_fuzzy, random, random_autocorr, zero, learned`; "the embedding takes every one except `zero`" | `weights.py:40` `PRESETS`; `:97-102` raises for `is_embedding` | ✓ |
| "format `llm-geometry/geo-model` v2 … a separate SHA-256 of the vocabulary" | `bundle.py:30-31` | ✓ |
| "up to **59°** out of the plane" | `fields.py` comment, verbatim | ✓ |
| "90th-percentile magnitude" | `GeoScene.svelte:303` `sorted[… * 0.9]` | ✓ |
| Dolch **40 / 92 / 133 / 220 / 314**, and they nest | `dolch_sizes()` = exactly that; I verified strict nesting for all five | ✓ |
| "`going` … a word that occurs **27** times in the shipped corpus" | recomputed on the committed corpus: `going count 27` | ✓ |
| "**70.7%** against **60.8%**", "**39.2%**", "**215 of 3,071** lines" | `/api/lex/coverage`: dolch 0.6079/0.3921/215/3071, frequency 0.7067 | ✓ |
| "**2,233** types … own **2,211** … **1,944** eligible … **1,680** stems … **8,202** of **16,000**" | `POST /api/lex/vacancy {p:1,seed:0}` returns exactly those six | ✓ |
| "**5.1%** of this corpus's tokens" | `stressFromTableBefore = 0.0514375` | ✓ |
| "hand table of **61** entries" | `len(STRESS_TABLE) == 61` | ✓ |
| "**137** closed-class words"; "the **11** suffixes ing · edly · … · s"; "longer than two characters" | `len(FUNCTION_WORDS)==137`; `SUFFIXES` is those 11 in that order; `len(stem) > 2` | ✓ |
| "swap loses **244 / 322 / 233** image slots at p = 0.25/0.5/0.75, and **0** at both endpoints" | recomputed the restricted map myself: `0/244/322/233/0` | ✓ |
| "±**0.2** nats"; "a pool below **700** preserved tokens" | `VACANCY_Q8_UNCERTAINTY_NATS = 0.2`, `VACANCY_MIN_POOLED_PRESERVED = 700` | ✓ |
| "**110,445** bytes and hashes to `d514f0fd2cd4…`"; "**86,408** bytes loaded here" | `CORPUS_BYTES = 110_445`, `CORPUS_SHA256` prefix; measured trimmed body = 86408 | ✓ |
| "min(|V|−1, d), currently **min(317, 64)**"; "|V| = 314 words in **318** rows, 2 heads of width **32**" | 314+4 = 318, 318−1 = 317; `DEFAULT_D_MODEL=64`, `DEFAULT_N_HEADS=2` | ✓ |
| "embedding is **57%** … only **42%** at L=2"; "at d=128 with 4 layers the blocks hold **95%**" | `param_count()`: 57.1 / 41.7 / 94.6 | ✓ |
| "95/5 train/validation"; "clipped at global L2 norm 1"; "weight decay 0.01"; "lr/25, first **30%**, … lr/25/1e+4" | `VAL_FRACTION`, `GRAD_CLIP_NORM`, `DEFAULT_WEIGHT_DECAY`, `ONECYCLE_*` | ✓ |
| "12d² + 13d per block" and its five-way breakdown | `param_count` docstring, verified against 3 configurations | ✓ |
| "**0.333** against a nonsense corpus's **0.346**" | `notes/agent-reports/006-source-eval.md:186` records `"meter_anapest": 0.34639…` from the source's own tracked nonsense corpus `data/demo_jabber.txt`. **This substantiates the number** — a prior red-team (`006-redteam-lexicon.md:328` L-4) marked it unsubstantiable after re-drawing its own random corpus instead of using the tracked one. No defect; noting so it is not re-flagged. | ✓ |

### Equations — all four blocks match the code they document

Checked `InfoTab.svelte:271-278` against `geo/model.py::_run` line by line under the
stated column-vector convention:
- `q_i = W_Q z_i` ← `q = h @ layer.W_Q.T` ✓
- `A_ij = softmax_j ⟨k_j, q_i⟩ over j ≤ i` ← `scores = q @ k.transpose(1,2) + causal`,
  unscaled, `softmax(dim=-1)` ✓ — and the prose's claim that there is *no* `1/√d` is
  correct and is the reason the trace and the force field are the same numbers.
- `z_i ← z_i + W_O Σ A_ij v_j` ← `attn_out = (attn @ v) @ layer.W_O.T` ✓
- `z_i ← z_i + W_outᵀ gelu(W_inᵀ z_i + b_in) + b_out` ← `gelu(h @ W_in + b_in) @ W_out
  + b_out`; with `W_in:(3,12)` and `W_out:(12,3)` the transposes are exactly right ✓
- `logits = E z` ← `readout: h @ E.T` ✓, `E` is `V × d` as the Notation table says ✓

Field 1 (`:313-314`) matches `fields.py::next_next_field` including the temperature-0
argmax special case and the "one arrow per point regardless of arrows/point" consequence.
Field 2 (`:341-342`) matches `force_field`: `vecs = points @ w_eff.T` is `W_V x`, the
aggregate is `attn @ v_proj`, and `antisymmetrize` substitutes `0.5*(w_v - w_v.T)` in the
per-point field **only**, exactly as claimed. The tangent projection
`F_i^∥ = F_i − ⟨F_i, ẑ_i⟩ ẑ_i` anchored at the **embedding** (not the residual stream) is
what the code does, and the "does not make the aggregate tangent" caveat is correct.

### The what's-real / where-it-runs table — every row I could test, verified on the live static build

- **"Architecture: op-by-op trace — precomputed … an arbitrary prompt cannot be traced —
  the tab says so rather than inventing tensors."** Typed `An arbitrary prompt nobody
  precomputed, zzyzx 4917.` on the live site. The tab said, verbatim: *"Per-layer traces
  need the model's hidden states, which browser ONNX exports don't expose — the static
  demo ships traces precomputed by the real backend for these example prompts only:
  "Capital of France" … Pick one of them, or run the full stack (see the README) to trace
  any prompt."* No fabricated tensors. ✓
- **"exact windows are HTTP range reads straight out of the safetensors file on
  HuggingFace's CDN."** Clicked into the `q_proj` heat map on the live site and captured:
  ```
  https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/7ae557604adf67be50417f59c2c2f167def9a775/model.safetensors  bytes=0-7
  …                                                                                                                    bytes=8-32287
  …                                                                                                                    bytes=300874160-300987183
  ```
  Header probe, header, then the window — pinned to a commit sha, not `main`. The
  inspector then read `rows 325–388 · cols 328–391 of 896 × 896 · exact values`, which is
  64 × 64 = 4096 cells, matching the documented budget. ✓
- **"The whole-matrix overview is the backend's own response, quantized to 8 bits at build
  time."** The overview rendered with **zero** network requests. ✓
- **"Lexicon Lab … never calls the backend."** Loaded and trained on the live static site
  with a request log: `API calls: []`. ✓
- **"The pretrained arm … only the quantities with a measured error bound … `nonce − swap`
  … refused by name."** Ran the scorer live (Qwen2.5-0.5B, q8/wasm, p=1, seed 0). It
  reported `the cost of wrong content · nll(swap) − nll(english) · 0.961 nats`,
  showed a **STATIC DEMO** refusal badge in place of `nll(nonce) − nll(swap)`, and printed
  `the same measurement, tiny model · 0 nats, exactly`. The `±` quantization term is
  rendered (`VacancyScorePanel.svelte:232-233`). ✓
- **"the corpus … re-hashes that body in your browser."** The tab printed the digest
  `d514f0fd2cd4…` matching `CORPUS_SHA256`. ✓

### Citations that *do* support their claims

- **arXiv:2005.10242 (Wang & Isola)** — cited as "the uniformity potential used as the
  spherical-spread term in training". `geo/train.py:119-124` implements
  `torch.log(torch.exp(-REPULSION_T * sq_dists).mean())` with `REPULSION_T = 2.0`, i.e.
  `log E[exp(−t‖eᵢ−eⱼ‖²)]` at t = 2 — **literally** the paper's uniformity loss at its own
  default temperature. Exact support, not a gesture.
- **arXiv:2607.13295 (Latifi Jebelli)** — cited for "reads attention as a two-body
  interaction law moving tokens as particles on a manifold". The abstract, fetched:
  *"the token dynamics of a transformer are modeled by a system of interacting particles
  on a Riemannian manifold 𝓜, the attention mechanism being encoded by a time-independent
  two-body interaction law."* Word-for-word support for the framing claim. I could **not**
  confirm from the abstract alone that the paper states `F_i = Σ_{j≤i} A_ij V z_j` in that
  exact form; the frozen API contract already carries that formula, so this is a
  can't-verify rather than a contradiction.

### Accessibility that is fine

- **Focus visibility.** My first automated pass flagged 65 elements as having
  `outline-style: none` while focused. This was a **false positive**: `PipelineDiagram.svelte`
  (`:466`, `:479`, `:503`) styles focus with `:focus-visible rect` rather than `outline`.
  I pixel-compared a focused vs unfocused `diagram-collapsed-layer_1` and the focused
  screenshot carries a clear dashed blue ring. Retracted.
- **Colour contrast.** Zero genuine failures. An automated pass flagged 3 items on Info and
  ~19 across the other tabs; all were artifacts of my compositor — semi-transparent
  `rgba()` layers (Info) and `background: var(--accent-grad)` gradients, which
  `getComputedStyle().backgroundColor` reports as transparent (the tab pills, `Generate
  reply`, `Train from scratch`, etc. are dark text on a *light* gradient). After adding
  alpha compositing the Info tab returned `CONTRAST FAILURES: 0`, and I confirmed the
  gradient cases visually in a screenshot.
- **Scroll containers are keyboard-reachable.** Every `.eq` and `.tblwrap` in the Info tab
  has `role="group" aria-label=… tabindex="0"` with a `:focus-visible` ring, and all four
  `.eq` blocks inside tab explainers (`GeometryLab.svelte:367,407`,
  `LexiconLab.svelte:513`, `ModelPanel.svelte:292`) do too. Walking the Info tab by
  keyboard reached all 39 stops with no trap and no off-screen focus.
- **`Explain` is sound.** Built on `<details>`, keyboard-operable, `summary:focus-visible`
  styled, and the title carries `role="heading" aria-level="3"` so the explainers appear in
  the heading outline.
- **No keyboard traps.** Tab order wraps cleanly to `BODY` on every tab (Architecture: 120
  stops; Info: 39). Every focusable element was on-screen and non-zero-sized.
- **Heat maps are named.** `canvas[role=grid]` carries `aria-label="matrix heatmap, 36 × 36"`.
  (There is no `aria-live` region, so arrowing a heat map announces nothing — but I could
  not find a keyboard cell-cursor to begin with, so this is issue #7's hover-only tooltip
  territory rather than a new finding.)

### Shell behaviours that work

- Deep links: `#architecture`, `#geometry`, `#lexicon`, `#info` all resolve on both the
  local stack and the live site.
- Reload preserves the tab (`#info` → reload → `#info`, Info still showing).
- Tab clicks update the address bar.
- A completed Lexicon Lab model survives a tab round trip (model token `03769632905e`
  unchanged).
- The Info tab's `STATIC_MODE` conditional is correct on the deployed site: it printed
  *"You are currently on the static build"* and not the full-stack sentence.
- The first-time "New here? Start with **Info** →" pointer appears, works, and retires
  itself; `localStorage` access is guarded on both read and write.
- Zero console errors and zero page errors on every run, on both stacks.

### Responsive

No page-level horizontal scroll at **320 / 390 / 768 px** on any of the four tabs —
`document.scrollWidth === clientWidth` in all twelve combinations. Elements that exceed
the viewport (Info's tables and `.eq` blocks, the Lexicon vacancy ribbon) are inside their
`overflow-x: auto` containers by design, and those containers are keyboard-focusable. The
one element that overflows *without* an obvious scroll parent is the
`matrix-heatmap` canvas (right edge 348 px at a 320 px viewport, 420 at 390, 804 at 768);
the document does not scroll, so it is clipped by an ancestor. **I did not determine**
whether the clipped portion is reachable — that canvas belongs to Agent A/B's slice and I
did not want to duplicate their work.
