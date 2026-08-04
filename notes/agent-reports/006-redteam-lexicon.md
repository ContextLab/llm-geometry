# Feature 006 red team — the Lexicon Lab

**Date**: 2026-08-04 · **Scope**: `src/viz/lex/*`, `src/viz/info/InfoTab.svelte#lex`,
`src/lib/lexEngine/*`, `src/lib/staticClient/lex.ts`, `llm_geometry/lex/*`,
`api/routes_lex.py`, `specs/006-lexicon-lab-tiny/*`.
**Method**: every checkable assertion in user-visible prose was traced to its
implementation and then **run** — the TS engine under `vite-node`, the Python package in
the backend venv, and the real tab in a live Chromium against `npm run dev`. Nothing below
is inferred from reading alone. Where I could not substantiate a claim I say so.

**No fixes were made.**

---

## CRITICAL

### C-1 · The browser and Python do NOT train on the same token stream — `<eos>` is missing in the browser

The UI states, in three places, that the two runtimes run *the same recipe* and attributes
the only divergence to floating point:

> "Browser and Python run the **same recipe**, held to ≤1e-5 … Whole-run training equality
> is **not** claimed: platform BLAS and RNG streams diverge."
> — `LexiconLab.svelte:400-403`, and again `TrainPanel.svelte`:
> "The browser and the Python backend run **the same recipe** … platform BLAS and RNG
> streams diverge, so the same seed gives the same shape of curve, not the same curve."

The data differs, not just the arithmetic.

Python (`lex/train.py:111-126`):

```python
def token_stream(text: str, vocab: LexVocab) -> np.ndarray:
    """Encode a corpus as one id stream, with `<eos>` closing every non-blank line."""
    ids: list[int] = []
    for line in text.splitlines():
        line_ids = vocab.encode(tokenize(line))
        if line_ids:
            ids.extend(line_ids)
            ids.append(EOS_ID)
```

Browser (`lexEngine/trainWorker.ts:98`):

```ts
  const tokens = vocab.encodeText(req.text);
```

…where `encodeText` is (`lexEngine/vocab.ts:141-143`):

```ts
  encodeText(text: string): number[] {
    return this.encode(tokenize(text));
  }
```

`grep -n "EOS" src/lib/lexEngine/train.ts src/lib/lexEngine/trainWorker.ts` returns
**nothing**. The browser stream contains no `<eos>` at all.

Measured on the shipped corpus, Dolch `full`:

| | stream length | `<eos>` count |
|-|-|-|
| Python `token_stream` | **19,071** | 3,071 |
| TS `encodeText` | **16,000** | 0 |

Consequences, all verified by running a real 200-step browser-engine training:

* `nTokens` reported by the training panel is 16,000; the backend would report 19,071.
* The trained browser model never learns `<eos>`, so the generated verse is one
  unbroken line. Measured: `final sample has newline? false` →
  `"as she old said i of and made i a of kind the the man away come must then a ate…"`.
* Every sentence about `<eos>` is therefore inert in the only place the tab actually
  trains: SamplePanel — "`<eos>` was not [masked], because that is how a line ends";
  `generate.ts:34` — "`<eos>` renders as a LINE BREAK, which is how a nursery-rhyme model
  produces verse rather than one long line."
* The comment justifying `DEFAULT_CTX = 32` (`lex/config.py:58-59`) says the measurement
  was made on "19,050 tokens" — a Python-stream figure — while `DEFAULT_BATCH`'s comment
  (`config.py:83`) says the timing was "measured in Node on the same engine". Those are
  not the same corpus.

The golden test cannot catch this: it pins forward/loss/spectrum on **fixed weights**, not
the data pipeline.

### C-2 · A model with non-finite weights defeats the `-inf` ban and emits `<unk>`; only a silent post-filter (which the prose says does not exist) hides it

`generate.ts:88-99`:

```ts
    for (const banned of GENERATION_BANNED_IDS) logits[banned] = -Infinity;
    let next: number;
    if (temperature === 0) {
      next = 0;
      let best = -Infinity;
      for (let i = 0; i < V; i++) {
        if (logits[i] > best) { best = logits[i]; next = i; }
      }
```

`NaN > -Infinity` is `false`, so with NaN logits the loop never fires and `next` stays at
its initialiser `0` — which is `UNK_ID`. The banned-id mask is bypassed entirely. The
`<unk>` ids then vanish here (`generate.ts:130`):

```ts
  const words = vocab.decode(out).filter((w) => !SPECIALS.has(w));
```

…which is exactly the post-filter the tab insists does not exist:

> "There is no filter to leak through." — `InfoTab.svelte:561`
> "there is nothing to filter and no filter to have a bug in" — `SamplePanel.svelte:174`
> "no trie, no post-filter" — `generate.ts:10`, `api-lex.md`

Above `temperature = 0` the same NaN state falls through the CDF loop to
`next = V - 1` (`generate.ts:112`) — the last budget word, emitted with zero probability
mass, forever.

**Reproduced live in the browser**, no dev tools, only clicks: open the tab, select the
`×2` preset on `embed`, apply it ~140 times (float32 overflows to `Infinity`), then
Generate.

| temperature | what the tab showed |
|-|-|
| 0.9 | `"wood wood wood … wood"` ×40, captioned "40 tokens · every one drawn from the 314-word budget", badge **in budget by construction**, no error |
| 0 | `"0 tokens · every one drawn from the 314-word budget"`, badges **in budget by construction** + **greedy · deterministic**, no error (40 `<unk>` ids were generated and stripped) |

The Python backend refuses in both cases (`lex/generate.py:97-106`):

```python
            if not torch.isfinite(logits).any():
                raise ComputeError("all logits are -inf; generation cannot continue")
            …
                if not torch.isfinite(probs).all():
                    raise ComputeError(
                        "the model produced non-finite probabilities; refusing to sample"
                    )
```

The TS engine has **no equivalent guard**. This is the "never fabricated, never silently
degraded" rule broken in the browser, and it is a full-stack/static divergence with no
counterpart in the UI text.

A second, one-click route to the same state: `LexWeightLab.onCellEdit` (line 169) admits
any value passing `Number.isFinite`, then writes it into a `Float32Array` — `1e40` is
finite in JS and becomes `Infinity` on store. Verified at engine level: a single cell edit
of `layers.0.fc1_w[0] = 1e40` yields `ids = 0,0,0,0,0,0,0,0,0,0` at `T = 0`.

Collateral, same state: the logit-lens panel prints `<unk> NaN% <bos> NaN% …` with no
error; the token cloud emits 198 console errors
(`<circle> attribute cx/cy/r: Expected length, "NaN"`) and draws nothing.

---

## HIGH

### H-1 · Five simultaneous false statements after one weight-lab click on an untrained model

Every "untrained" caption in the tab keys off `isTrained = trained !== null`
(`LexiconLab.svelte:188`), but `edited` can be non-null while `trained` is null — the
Weight Lab's base is `freshModel` when nothing has been trained
(`LexiconLab.svelte:185`). One click (`zero` preset on `embed`) reaches that state.

**Live browser evidence, after that single click:**

| panel | what it said | what was true |
|-|-|-|
| Spectrum | effective rank **0.00**, description: "…and **57.67** for an untrained model at this exact shape — which is what you are looking at: nothing has been trained yet, **so these two are the same model**" | 0.00 vs 57.67 — not the same model |
| Spectrum | "The bars, the baseline and both rank markers **coincide** because they are the same random initialization" | every bar is zero; nothing coincides |
| Forward pass | "a real trace of the **random-init** model at this shape" | it is the edited model |
| Token cloud | "this is the projection of a **random Gaussian matrix** — a featureless blob is the correct picture" | it is an all-zero matrix; explained variance PC1/2/3 = 0.0%/0.0%/0.0% |
| Save/load | "a save right now writes the **random initialization**" | it writes the edited weights |

This also answers audit item 3 directly: effective rank is never printed *without* a
ceiling and a baseline (`SpectrumPanel.svelte:71` gates on `embedding !== null && baseline
!== null`, verified — see appendix), but the sentence **next to** it can be false.

The degenerate spectrum (0.00 across every statistic, 0.0% explained variance) is shown
with no warning. `api-lex.md` specifies a `degenerate: <bool>` field for exactly this; the
TS panel does not surface any equivalent.

### H-2 · `SC-606` ("every constant quoted in the tab's prose is pinned by a test") is not met for the Info tab's Lexicon section

`CLAUDE.md` and `spec.md` name `tests/e2e/docs.spec.ts` as the mechanism. That file (205
lines) contains **zero** Lexicon assertions:

```
$ grep -n -i "lex\|dolch\|314\|70.7\|60.8\|39.2\|3,071\|Mother Goose" tests/e2e/docs.spec.ts
(no output)
```

Pinned elsewhere: the budget sizes 40/92/133/220/314 (`tests/unit/lexEngine.test.ts:132`)
and "314 not 315" in the UI (`tests/e2e/lexicon.spec.ts:49`).

**Unpinned, hard-coded in `InfoTab.svelte`**: `70.7%`, `60.8%`, `39.2%`, `215 of 3,071`,
`27 times`, `1936`, `0.333` / `0.346`. I verified each of these is currently correct (see
appendix) — the finding is that nothing stops them rotting, which is the precise failure
mode feature 005 built the pinning convention to prevent.

### H-3 · `SC-604`'s characterisation of the effective-rank increments is contradicted by the code that produces it

`spec.md:156-158`:

> "…while **effective** rank only approaches it asymptotically, **with decelerating
> increments (+29.3 on the first step, +8.4 on the last)**."

Recomputed with `lex/spectrum.py::random_baseline_spectrum(cfg, 0)` at `d = 128` over the
five Dolch budgets:

| budget | rows | effective rank | Δ |
|-|-|-|-|
| pre_primer | 44 | 36.21 | — |
| primer | 96 | 64.38 | **+28.18** |
| first | 137 | 79.11 | +14.73 |
| service | 224 | 95.73 | **+16.62** |
| full | 318 | 104.16 | +8.42 |

Two errors. The first increment is **+28.18**, not +29.3 (which is not the difference of
any pair in the spec's own list). And the increments are **not** decelerating: 16.62 > 14.73.
The same "decelerating increments" phrasing is repeated in `architecture.md` (final block
quote), `lexEngine/spectrum.ts:25-26` and `tests/unit/lexEngine.test.ts:369`.

The five values themselves (36.21, 64.38, 79.11, 95.73, 104.16) reproduce exactly, and the
algebraic-rank claim ("128 from the `first` budget onward") reproduces exactly
(`np.linalg.matrix_rank` of the centred matrix: 43, 95, 128, 128, 128). None of these
numbers appears in the UI, so this is a spec/comment defect, not a user-facing lie.

---

## MEDIUM

### M-1 · "At `d = 16` with the full budget the embedding is the majority of the model" is false at the defaults

`ModelPanel.svelte:276-280`. Recomputed with `paramCount` (the same closed form the panel
displays), `|V| = 314`, rows = 318, ctx = 32:

| shape | total | embedding share |
|-|-|-|
| d=16, L=1, tied | 8,912 | 57.1% |
| **d=16, L=2, tied (the tab's default `L`, default `tied`)** | **12,192** | **41.7%** |
| d=16, L=3, tied | 15,472 | 32.9% |
| d=16, L=4, tied | 18,752 | 27.1% |

A reader who follows the instruction literally — move `d_model` to 16, leave everything
else alone — sees 42%, a minority. The claim only holds at `L = 1`, or untied. The
companion clause ("at `d = 128` with four layers the blocks dominate") is correct: 94.6%.

### M-2 · The Info tab's "What's real, and where it runs" table has no Lexicon Lab row, and its own navigation omits the tab

Enumerated live from the rendered DOM — the capability table's rows are: Geometry Lab
fields/traces, Geometry Lab edits/fine-tune/training, Architecture chat, Architecture
trace, Architecture weights. `JSON.stringify(rows).toLowerCase().includes('lexicon')` →
`false`.

The table is introduced with "it is worth being precise about what that changes… the
right-hand column describes what you have", so a whole tab's absence reads as an omission
of exactly the thing the section promises. It is also a missed *positive*: the Lexicon Lab
is the only tab that is byte-identical in both modes (`staticClient/index.ts:77-79` — "the
Lexicon Lab computes in the browser in BOTH modes"), which the table would have said in
one line.

Related, same file:

* The lede (`InfoTab.svelte:33-40`) still reads "**Two views** of a transformer at two
  magnifications… Nothing on **either** tab is a schematic". There are three explorer tabs.
* "Which tab do I want?" (`h3#start`) has cards for the Architecture Explorer and the
  Geometry Lab only. No Lexicon card. (The TOC pill list *does* include it.)
* "Known limits" contains no Lexicon entry. Its browser-vs-Python bullet enumerates the
  *Geometry Lab's* recipe ("…clipping, **sphere projection**, and vocabulary construction
  are the same") while being rendered as a general statement, and now silently covers a
  second tab whose divergence is larger than it claims (see C-1).
* "Source & references" cites Project Gutenberg **#11** (Alice) as "the training corpus for
  the shipped checkpoint" and no longer disambiguates; there is **no** citation for
  PG #10607 (*The Real Mother Goose*) and none for Dolch (1936), although the lex prose
  asserts both as sourced facts. The first bullet also says "**Both tabs** run locally
  against real PyTorch with one command" — which is stale in count and, for the Lexicon
  Lab, false: the tab never calls `/api/lex/*` in either mode.

### M-3 · The corpus digest chip presents a verification that never happens in the browser

`LexiconLab.svelte:266`:

```svelte
<span class="chip mono" title="sha256 of the committed bytes, re-verified by the backend before any run">{corpus.shipped.sha256.slice(0, 12)}…</span>
```

`loadCorpus()` (lines 120-139) fetches `static-data/lex/corpus.json` and checks only that
`asset.text` is a non-empty string. It does not hash anything. The displayed digest is
`d514f0…`, the sha256 of the **untrimmed** 110,445-byte file — not of the 86,408-character
body that is actually loaded and trained on. The export *does* carry a `body_sha256`
(`03769632…`) which would be checkable; the browser ignores it.

The tab's own header comment says "an invented corpus would make every number below a
fiction" and "It deliberately has no fallback" — but substituting `corpus.json` wholesale
produces a page that displays a healthy-looking digest beside fabricated coverage numbers.
This is the same shape of hole the bundle loader was hardened against.

### M-4 · Stale hint: "two digests" where the body says three

`ModelFile.svelte:162`: `hint="two digests, both mandatory, both fatal on mismatch"`. The
panel note directly above it says "under **three** checksums" and the Explain body lists
three (`model_token`, `weights_token`, `vocab_sha256`).

### M-5 · Tooltip-only information unreachable by keyboard

Enumerated live: `title` attributes on non-focusable elements inside `[data-testid=lex-view]`.
The provenance claims that appear **nowhere else** in the DOM are both in this class:

* `PG #10607` chip → "committed whole, header and licence footer intact, and trimmed only
  when the text is used"
* sha chip → "sha256 of the committed bytes, re-verified by the backend before any run"

Also tooltip-only: the "follow playhead" explanation, and every Weight Lab preset
description (`<option title=…>`, which most assistive tech does not expose at all) — e.g.
"Re-draw from this tensor's own initializer at the seed below". "identity (square only)"
is the one whose caveat also appears as text.

Compare the same pattern in the rest of the tab, which does this correctly: the "in budget
by construction" badge's tooltip repeats mechanism that is also spelled out in the Explain
block, so nothing is lost.

---

## LOW

* **L-1** `lexEngine/index.ts:26` usage example: `defaultConfig(vocab.rows); // d=64, L=2, H=2, ctx=64, tied`. `DEFAULT_CTX` is now **32**. Stale after the ctx 64→32 change.
* **L-2** `corpus.json` exports `n_lines: 3075`; `Coverage.total_lines` (what the UI and the Info tab show) is `3071`. Two line definitions, one corpus. Only 3,071 is user-visible, so this is latent.
* **L-3** `TokenCloud.svelte:185-187`: the four reserved rows "are **trained like any other row**". `<pad>` is `ignore_index` in the loss (`train.py:254`) and never appears in either training stream; in the browser `<eos>` never appears either (C-1). Under `tied` they still receive gradient through the softmax denominator, so the sentence is not flatly wrong — but it is not true in the sense a reader will take it.
* **L-4** `InfoTab.svelte:601` quotes the source's meter score as "0.333 against a nonsense corpus's 0.346". I re-ran the source (`~/Desktop/tiny-models/tiny-seuss`): `Budget().meter_score("and I do not like green eggs and ham", "anapest")` = **0.3333** ✓. The 0.346 depends on an unrecorded random nonsense corpus; my own 200-line draw over the same budget scored **0.3768**. The substantive claim (a nonsense corpus scores at least as high) reproduces; the exact second decimal is not reproducible from anything in the repo. **I cannot substantiate `0.346` specifically.**

---

## Appendix — checked and found CORRECT (do not re-walk)

**Every number in the Info tab's Lexicon section**, recomputed from `llm_geometry.lex` on
the committed corpus (digest verified: `corpus_sha256() == CORPUS_SHA256`, 110,445 bytes on
disk):

| claim | source | measured |
|-|-|-|
| budgets 40 / 92 / 133 / 220 / 314 | `dolch.dolch_sizes()` | exact ✓ (TS `dolchSizes()` identical) |
| the Dolch lists **nest** | set containment across all four steps | 0 words ever leave ✓ |
| 60.8% Dolch coverage at \|V\|=314 | `coverage.token_coverage` | 0.607875 ✓ |
| 70.7% frequency coverage at \|V\|=314 | same | 0.706750 ✓ |
| `<unk>` rate 39.2% | `1 − coverage` | 0.392125 ✓ |
| 215 of 3,071 whole lines | `whole_lines_in_budget / total_lines` | 215 / 3071 ✓ |
| `going` occurs 27 times | token count | 27 (and `giving` occurs 1) ✓ |
| \|V\|=314 not 315 (`Santa Claus` dropped) | `dolch.py` | ✓ |
| live UI counters | rendered DOM | 9,726 of 16,000 · 60.8% · 39.2% · 215/3,071 · 1,919 oov types of 2,211 — all match Python exactly ✓ |
| parameter count 122,496 at defaults | `paramCount` + rendered panel | ✓, and the three-term breakdown sums correctly |

**Logit-lens `exact` vs `approximate` pills (audit item 5) — the stages marked exact really
are exact.** For `L ∈ {1,2,3}` the marked stages are the last layer's `mlp` and the
`readout`. The `readout` stage reads `acts.logits` directly (`trace.ts:303`); the last-layer
`mlp` stage re-derives it through `lensOf(la.hOut, true)`. Measured agreement between the
two: **max probability difference = 0 (bitwise), identical top-k id order**, at every depth
tested. Every earlier stage is marked approximate. The claimed 1e-9 pin between traced and
independently-run logits exists (`tests/unit/lexTrace.test.ts:49`, `:68`).

**PCA labelling (audit item 4).** The cloud is labelled a projection in all four places it
can be perceived: the `<h3>` carries a `PCA projection` tag chip; the panel hint reads "a
shadow of the embedding, not the embedding"; the SVG `aria-label` begins "PCA projection
of N embedding rows…"; the caption names PC1/PC2/PC3 and prints explained variance.
`grep` confirms no second cloud renderer exists.

**Effective rank is never rendered without both controls.** `SpectrumPanel` gates the whole
chart on `complete = embedding !== null && baseline !== null` (line 71), the `chart` snippet
takes `base` as a required argument, and `barCount` widens the drawn range so the ceiling
line can never fall off the chart. `grep -rn "effectiveRank" src/viz/lex/` finds no other
numeric render site. (The *sentence* beside it can lie — see H-1.)

**Bundle integrity (audit item 6).** 26 tamper attempts through `importLexBundle`; the only
loads were the ones that should load.

Refused ✓: `model_token` / `weights_token` / `vocab_sha256` deleted, `null`, or `""`;
version `2` or `"1"`; wrong `format` tag; any weight byte flipped; `embed` zeroed, filled
with NaN, filled with Infinity, or truncated by 4 bytes; a vocab word swapped, dropped, or
added; a duplicated vocab word; tampered `specials`; `vocab.source` outside the enum; a
tied bundle carrying `head_w`; `config.dropout`, `config.ctx` or `config.n_heads` altered;
a lying `shape` field with correct data.
Loaded (correctly) ✓: the unmodified bundle; `metrics` deleted or replaced with lies
(outside every digest, by design); an unknown extra top-level field.

The backend enforces the same rules with the same "missing == wrong" semantics
(`routes_lex.py:1026-1057`, `_declared()` rejects a non-string, wrong-length, or
non-hex value before any comparison).

**Out-of-scope items do not leak (audit item 8).** `grep -i "mint|vacancy|rhyme|meter|
prosody|fingerprint|prediction"` over `src/viz/lex/*.svelte` finds FR-624/625 only inside
the "**Not shipped:**" list, and FR-626 only as a negation. The word "mint" otherwise refers
to content-hash minting of weight sets. No T1–T6 prediction, no schematic, no borrowed
curve appears anywhere in the tab. The provenance prose is, if anything, more careful than
it needs to be — it states the source is "a proposal, not results" and that "nothing here
reproduces its curves" in both the tab and the Info tab.

**Accessibility (audit item 9) — the radiogroups are correct.** All ten radiogroups
inspected in the live DOM (budget source, budget size, d_model, layers, heads, context,
train mode, corpus source, playback speed, attention layer): every one has
`tabindex="-1"` on the group, `aria-labelledby` or `aria-label`, **exactly one**
`aria-checked="true"`, and **exactly one** `tabindex="0"` (correct roving tabindex). The
arrow-key handler is present on each group, skips `:not([disabled])`, and moves focus and
selection together. Disabled radios use the native `disabled` attribute on a `<button>`, so
they are correctly removed from the tab order. Scroll containers (`.eq`, `.chart-wrap`,
`.tblwrap`) carry `tabindex="0"` for WCAG 2.1.1.

**Other verified-correct claims:**

* "halving the embedding … leaves **effective rank unchanged**" (`LexWeightLab.svelte:308`) — true: scaling `A` scales `Ac`, so every `pᵢ` is invariant. Confirmed numerically.
* An edit whose content hash returns to the base's is reported as the base, not as an edit (`LexWeightLab.commit`, line 125-133).
* The untrained model's spectrum really does equal the drawn baseline when nothing has been edited: effective rank 57.6665 both ways (σ₁ agrees to 4e-10).
* `generate` refuses a vocabulary whose row count differs from the model's, in both runtimes.
* Fine-tuning takes its dimensions from `model.cfg` and its word list from the trained vocab (`TrainPanel.runFinetune`), so feature 004's issue #6 is structurally excluded.
* The tab refuses to run at all when `static-data/lex/corpus.json` is unavailable, with an explicit no-fallback message (verified by reading; the asset is present in `public/` and `dist/`).
