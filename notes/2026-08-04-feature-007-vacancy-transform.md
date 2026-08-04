# Feature 007 — the vacancy transform (session notes)

Started 2026-08-04. Branch `007-vacancy-transform`. Follows feature 006 (Lexicon Lab).

## What the user asked for

> take a look at ~/Desktop/TinyModelsDoc for what the tiny model is *trying* to build. the
> limited vocabulary and word-level tokenizers are one part, but the vacancy transform is
> important too. Can you add this to the demo? This needs to be done carefully!

Scope decision (asked, answered): **both arms** — the tiny arm in the Lexicon Lab AND the
pretrained arm in the Architecture Explorer, completing the doc's T4 2×2.

## Source material

`~/Desktop/TinyModelsDoc/` — `tiny_models.tex` (the proposal, 1055 lines) and
`tiny-seuss.zip`. The zip is the ORIGINAL bundle; `~/Desktop/tiny-models/tiny-seuss/` is the
AUDITED copy from the previous session and differs in three files. Where they disagree, the
audited copy is right:
- `split_suffix` gains an `exceptions` set (`brother` etc.) — without it `brother → broth+er`
- `main` reports `len(j.vacated)` not `len(j.map)` — the zip prints the wrong count
- `README.md` retracts the unverified TTR 0.098 baseline (measured 0.121) and withdraws the
  "exact prosody" claim

## Key artifacts

- `specs/007-vacancy-transform-field/spec.md` — FR-701…726, SC-701…710
- `specs/007-vacancy-transform-field/architecture.md` — **the normative TS↔Python contract**.
  Written before any code, deliberately: feature 006's one CRITICAL bug was a contract gap.

## The load-bearing idea

For a word-level model trained from scratch, a word's "location" is a row index — the model
never sees the letters. So with `consistent = true`, `revealAfter = 0`, and the vocabulary
*mapped* through the transform (order preserved), the vacancy transform is a **pure
relabelling** and training is **bit-identical**. That is contract §7.3, spec SC-703.

I checked the proof by hand against `token_stream`'s actual behaviour:
- `t ∈ V` → `map(t) ∈ V_p` at the same index ✓
- `t ∉ V` → `map(t) ∉ V_p`, since `map(t) = map(w)` with `w ∈ V` would force `t = w` ✓
- both depend on **injectivity over `corpus types ∪ V`** — which is why the contract makes
  injectivity a verified property with a re-mint loop, not an assumption
- case is safe because `tokenize` lowercases and `avoid`/`used` are lowercase
- all types sharing a stem are vacated together, so a budget word can never be vacated while a
  corpus type sharing its stem is not

**Framing risk.** "The model is exactly invariant" can read as trivial — *of course, you
relabelled the vocabulary*. That IS the point, and the docs must say so: the doc asks whether an
embedding is an independent carrier of content or a summary of contextual support; in the tiny
regime it is provably the latter. The tiny arm's job is to establish the **baseline of zero**,
so that the pretrained arm's delta has a scale. Do not dress the null up as a curve.

## Eleven departures from the source

Contract §9 lists them all. Four are corrections to bugs that break properties the source
*claims* for itself:
- the map is built lazily while rewriting, so `used` makes a nonce depend on `p` — breaks stability
- the give-up path is `syllable + str(len(used))` — order-dependent
- the seam fix draws from a shared RNG — order-dependent
- injectivity is assumed; `avoid` is accepted and never passed, so a nonce can merge with a real word

Plus one that only bites across our two stacks: `top64 / 2**64` is not exactly representable, so
Python and JS can disagree at the boundary. Contract §4 uses `(top64 >> 11) / 2**53`, which is.

## RESOLVED — the swap control is in (contract §8.3, FR-719a, SC-707a)

The pretrained arm's `ΔnllPreserved` has a confound the contract states (§8.3): the vacated
passage genuinely has higher entropy, so every prediction degrades, scaffolding included.

A **swap control** would make the number interpretable rather than merely caveated: replace each
vacated stem with a *real English word* (drawn from the corpus, frequency-matched) instead of a
nonce form. Same machinery, different minting strategy — `mint: "nonce" | "swap"`. The context
is then equally wrong semantically but the forms are all known, so
`nonce − swap` isolates *unknown form* from *wrong content*.

Added to the contract as §8.3 while the two modules were still being written, since it does not
change §§1-7 and the modules are implementing those. It only adds a minting strategy.

The decomposition the UI must report:
- `nll(swap) − nll(english)` — the cost of **wrong content**
- `nll(nonce) − nll(swap)` — the cost of **unknown form**

and never `nll(nonce) − nll(english)` alone, which conflates them. The residual — that nonce
forms fragment into more subword tokens — is not separable without a tokenizer-level control,
and the UI says so rather than pretending the remainder is pure location.

The correctness check for the control is elegant: `swap` must satisfy the invariance theorem
exactly as `nonce` does, because the tiny model is equally blind to both.

## Two defects found in SHIPPED code while building this

Neither is a feature-007 bug; both were found because 007 forced us to drive the real app.

**1. Silent vocabulary substitution in the Geometry Lab** (fixed, `d6e9d5d`). Static build only,
across a reload, for a model with its own vocabulary (scratch-trained or file-loaded). Train →
Save (your word list) → reload → Save → same weights under *Alice in Wonderland*'s word list.
No error, and unrejectable: `vocab_sha256` is computed over the list that was written, so the
file verifies on both sides. Exactly the corruption the three digests exist to prevent,
committed by the writer. The persisted `ExportedWeightSet` carried weights but not the
vocabulary, so `tokenizerFor()` fell back to the canonical tokenizer — right for edited and
fine-tuned sets, catastrophic for scratch and imported ones.

**2. q4f16 produces garbage on WebGPU** (FIXED — see below). The dtype the app tries FIRST. The
session builds successfully and then returns degenerate output: every row of the `[1,T,V]`
logits bit-identical, gpt2 greedy-generating `,,,,,,,`, SmolLM2's logits all exactly 0 so every
NLL is `ln(49152) = 10.80267`. The fallback in `transformersRuntime.ts` only fires on a thrown
exception and nothing throws, so **the deployed Architecture Explorer is already showing wrong
probabilities on any `shader-f16` machine** — reproduced in real Chrome 150, not just Playwright.
q8, fp32 and q4 are correct; fp16 fails identically, so the fp16 *activation* path is the cause.

Why it survived: **plain headless Chromium exposes no WebGPU adapter** (`requestAdapter()` →
null), so the entire e2e suite has only ever exercised wasm/q8. That coverage gap is part of the
fix.

**The fix** (its own commit, independent of 007). Re-verified first in a real browser on the
real Apple Metal-3 adapter, per repo, `maxAbsRowDiff` between the first and last logit row of one
teacher-forced pass + a greedy continuation:

| repo | webgpu/q4f16 | webgpu/q8 |
|-|-|-|
| gpt2-ONNX | 0.000, `,,,,,,,,,,` | 91.99, ` Berlin. The capital of the United States is Washington` |
| SmolLM2-135M-Instruct-ONNX | 0.000, all logits 0, empty | 36.97, ` Berlin.\n\nThe capital of Italy is Rome` |
| SmolLM2-360M-Instruct-ONNX | 0.000, all logits 0, empty | (correct) |
| Qwen2.5-0.5B-Instruct | 16.47 — NOT degenerate, but worse text | 20.27, ` Berlin. What is…` |

Three of the four curated models are destroyed by q4f16; Qwen survives it. `q4` is correct
(SmolLM2-135M: 35.66, ` Berlin. The capital of the United States is Washington`) but is **not**
the smaller download the earlier note assumed — in every curated repo `model_q4.onnx` is LARGER
than `model_quantized.onnx` (gpt2 498 vs 280 MB, SmolLM2-135M 181 vs 136, SmolLM2-360M 386 vs
363, Qwen2.5-0.5B 786 vs 512). So the ladder is now **webgpu/q8 → wasm/q8**: both rungs read the
same file, so a rejected rung costs no second download.

1. `staticClient/logitsSanity.ts` — ONE invariant, in the spirit of the Geometry Lab's training
   gates: a causal LM's output must depend on its input, so the L∞ gap between the first and last
   next-token distribution of a fixed 12-token probe must exceed 1e-3. All-identical rows and
   all-zero logits are the same failure, not two rules. Asserted on every session at load, before
   any number is shown; a rejected rung falls through and is NAMED in the badge (`· fallback`).
2. `RUNTIME_LADDER` + `FP16_ACTIVATION_DTYPES` in `runtimeTypes.ts`, unit-tested
   (`tests/unit/logitsSanity.test.ts`) so CI always checks that no fp16-activation dtype creeps
   back, even where it cannot run a GPU.
3. `tests/e2e/webgpu.spec.ts` + a `webgpu` Playwright project. **The flag that matters on macOS
   is `--use-angle=metal`**: `--enable-unsafe-webgpu` alone still hands back google/swiftshader
   (no `shader-f16`); with it, headless Chromium gets the real apple/metal-3 adapter. Verified to
   FAIL on the pre-fix code (badge `webgpu · q4f16`; 2 distinct top-5 lists across 64 generated
   positions) and PASS after (64/64 distinct).

**Named residual gap:** GitHub-hosted runners have no GPU, so this test SKIPS in CI with a loud
reason. The WebGPU path is verified only on a developer machine with a real GPU; CI verifies the
invariant, the ladder, and one real session through the same gate on the WASM rung.

Lesson worth keeping: both defects are invisible to unit tests and to any test that checks for
thrown errors. They produce *plausible* wrong answers. The only thing that caught them was
running the real thing and comparing against a known-good reference.

## The swap control in the Lexicon Lab, and how its constraint is surfaced

`mint = "swap"` is now live in `VacancyPanel` (it had been rendered disabled). Three decisions
worth keeping, because each replaces a tempting shortcut:

1. **The constraint is shown, never enforced by the UI.** `p` is not clamped, `swap` is not
   silently downgraded to `nonce`, and the typed error is not caught-and-replaced with a
   fallback. `LexiconLab` asks the engine (`buildVacancyMap` with the real `consistent`,
   `mapVocabWords` at the real `p`) and CARRIES the refusal up as a string; the panel prints it
   verbatim in a refusal card, with buttons for the two exits (`p = 1`, `p = 0`, switch to
   nonce, use the consistent condition). With no vocabulary, the budget counters, the trainer
   and the invariance check simply have nothing to report — which is the honest state.
2. **The theorem is COUNTED, not asserted.** Beside the mint control, every domain type is
   pushed through the real transform at the current `p` and the distinct images are counted:
   `|domain| − |images|` lost image slots. Measured on the shipped corpus, seed 0:
   **244 / 322 / 233 at p = .25/.5/.75, and 0 at both endpoints** (reproduced independently in
   Python while writing the docs). Under `nonce` it is 0 everywhere. So §5.2a happens in front
   of the reader.
3. **The `bijective` chip branches on `injectiveAtEveryP`.** Under swap it reads "injective at
   p = 0, 1" rather than a bare tick — the map property is real, but claiming it at every `p`
   would be false.

Two shipped-code defects fixed in passing: `LexiconLab` never passed `mint` into `vacParams` at
all (so the control could not have worked), and `staticClient/arch.ts` still documented a
"±0.1 nats" quantization uncertainty that its own constant had superseded with 0.2.

## Documentation (FR-724/725/726, ui.md §3)

Info tab gains `<h3 id="vacancy">`: the T4 2×2 with the vacancy cell marked, the transform's
definition (with `u` as an equation), nesting + stability and the four properties the source
implementation claims and breaks, the invariance theorem with §7.4's framing (the exact zero IS
the finding), the swap decomposition with "cost of unknown form" stated as an UPPER BOUND, the
stress table's real status, and a by-name list of what the static build refuses plus the WebGPU
/ CI coverage gap. `#real`, `#limits` and `#refs` updated (Gutenberg #12 — *Through the
Looking-Glass* — added; link checked).

**Every number is pinned** in `tests/e2e/docs.spec.ts` (5 new tests): the counts come from a live
`POST /api/lex/vacancy` at p = 1 (2,233 / 2,211 / 1,944 / 1,680 / 8,202 / 16,000 and the 5.1%
stress-table coverage), the swap collisions are read off the running panel, the stress-table size
is read off the panel's own honesty line, and the static-mode ±0.2 nats / 700-token floor are
regex'd out of `staticClient/arch.ts`. Nothing in the section is a number a human retyped.

Note for anyone extending this: the pretrained arm's measured deltas are deliberately NOT quoted
in the Info tab. They depend on model, passage set and dtype, and the only honest pin would be a
real Qwen run per CI job. The panel reports them from the run the reader triggers.

## Status

### Contract defects found by BUILDING it (13, not one of which I caught by re-reading)

The two stacks were written independently from the contract so they could disagree. Every
disagreement turned out to be a defect in the document, never in one implementation:

1. case — suffix sliced case-preserved, so `gums→flels` but `GUMS→FLESS` (one type, two surfaces)
2. injectivity checked only at `p=1` and over bare nonces — `hanged→waked` collides at `p=0.25`
3. `CODAS` documented as 49 entries; it has 46
4. `typesVacated` undefined between stems and types (1922 vs 1665)
5. salt thresholds ambiguous between attempt counter and absolute salt
6. domain readable as the *active* budget — would re-mint on every budget switch
7. `avoid` optional, so the map depended on caller memory (0 vs 1 re-mint rounds, different nonces)
8. `vacancyDomain` helper existed in Python only
9. `VacancyMap.map` vs `.mapping`
10. `consistent=false` prosody drawn from the mint key, not the stem
11. `corpusTypesVacated` from map membership vs measured from the texts (1337 vs 665)
12. condition B not applied to the per-occurrence path — `tak→tak`, a word silently surviving
13. `forbidden` stored in one stack, reconstructed in the other (drops superseded nonces)

Four of these (1, 2, 10, 12) would have broken the invariance theorem. Numbers 4, 11 and 13 were
found only because a stack was told to STOP and report rather than reconcile to the other — a
golden fixture built over a silent reconciliation would have cemented both stacks being
consistently wrong.

- [x] Spec + contract written and committed (`36b9f3d`)
- [ ] Python `lex/vacancy.py` + tests
- [ ] TS `lexEngine/vacancy.ts` + tests
- [ ] transformers.js alignment probe (contract §8.2 requires this be settled empirically first)
- [ ] Golden fixture + parity
- [ ] Lexicon Lab vacancy panel
- [ ] API routes + static client
- [ ] Pretrained arm
- [x] Docs (Info tab + in-tab prose) — `#vacancy` section, `#real`/`#limits`/`#refs` updated,
      the swap control enabled in the Lexicon Lab
- [ ] Full suite, deploy, live verification

## Standing constraints (from CLAUDE.md)

No mocks, ever. Real corpus, real models, real browsers. Re-run **all** checks after any fix.
Never transcribe a number from the source document as if it were ours — the doc's
`0.351 → 0.345` anapest and `1.224 → 1.211` syllables are its numbers on a corpus we do not
have. Measure ours.
