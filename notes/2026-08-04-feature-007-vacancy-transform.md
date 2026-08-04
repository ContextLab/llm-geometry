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

## Status

- [x] Spec + contract written and committed (`36b9f3d`)
- [ ] Python `lex/vacancy.py` + tests
- [ ] TS `lexEngine/vacancy.ts` + tests
- [ ] transformers.js alignment probe (contract §8.2 requires this be settled empirically first)
- [ ] Golden fixture + parity
- [ ] Lexicon Lab vacancy panel
- [ ] API routes + static client
- [ ] Pretrained arm
- [ ] Docs (Info tab + in-tab prose)
- [ ] Full suite, deploy, live verification

## Standing constraints (from CLAUDE.md)

No mocks, ever. Real corpus, real models, real browsers. Re-run **all** checks after any fix.
Never transcribe a number from the source document as if it were ours — the doc's
`0.351 → 0.345` anapest and `1.224 → 1.211` syllables are its numbers on a corpus we do not
have. Measure ours.
