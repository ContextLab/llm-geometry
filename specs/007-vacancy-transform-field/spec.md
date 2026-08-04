# Feature 007 — The vacancy transform: the field-without-location instrument

**Feature Branch:** `007-vacancy-transform`
**Created:** 2026-08-04
**Status:** In progress

**Contract:** `specs/007-vacancy-transform-field/architecture.md` (normative — read it first)
**Builds on:** `specs/006-lexicon-lab-tiny/spec.md` (the Lexicon Lab this extends)
**Frozen HTTP contract:** `specs/002-interactive-model-explorer/contracts/api.md` — additive
endpoints only; any change to an existing endpoint gets its own commit with a note.

## Why

Feature 006 shipped the closed lexicon — the graded word budgets and the word-level
tokenizer — which is one half of what `~/Desktop/TinyModelsDoc/tiny_models.tex` argues a tiny
model is *for*. The other half is the **vacancy transform** (the doc's §"The vacancy
transform"), the instrument that manufactures Carroll's condition — full syntactic scaffolding,
vacant lexical content — at a controlled rate on any corpus. 006 deferred it as FR-624 on the
grounds that it needed a parameter-matched control. It has one: under the conditions of
contract §7 the transform preserves the vocabulary *exactly*, so the control is the design.

The transform is what turns "field" and "location" from vocabulary into numbers. Without it the
Lexicon Lab shows a budget; with it, the lab shows what a word's *identity* is worth to a model
that has never seen one — which is zero, exactly — and the Architecture Explorer shows what it
is worth to a model that has.

## Scope

Two arms, both real, no mocks anywhere.

**Tiny arm — Lexicon Lab.** The transform, a `p`-sweep, the doc's three control conditions,
live retraining against the existing in-browser trainer, and the invariance result.

**Pretrained arm — Architecture Explorer.** A real HF model over a passage and its vacated
twin, scored on the preserved closed-class scaffolding only.

Out of scope, and stated so it is not mistaken for an omission: the doc's lattice/trie decode
mask (`eval/mask_decode.py`, its T1/T6 instrument), the concept battery (its resource F), the
minting staircase (T3), and corpus synthesis by rejection sampling (its resource A). Each is a
separate instrument; none is needed for T4.

## Functional requirements

### The transform (shared, both stacks)

- **FR-701** A vacancy module exists in both stacks implementing contract §§1–6 exactly:
  `llm_geometry/lex/vacancy.py` and `code/frontend/src/lib/lexEngine/vacancy.ts`.
- **FR-702** The transform rewrites raw text in place, replacing only `WORD_RE` matches and
  passing all other bytes — punctuation, whitespace, line breaks — through unchanged.
- **FR-703** Vacancy is decided by `u(stem) < p` with `u` derived as contract §4, so vacated
  sets **nest** in `p`.
- **FR-704** The nonce for a stem is fixed by `(seed, stem)` alone — **stable** across `p`,
  across document order, and across corpora.
- **FR-705** The map is injective, verified per build with re-minting on collision, and
  `bijective` is reported in the statistics.
- **FR-706** A nonce never collides with a real type of the corpus (`avoid`, contract §5.2).
- **FR-707** Inflectional suffixes and closed-class words are preserved; the eligibility rules
  of contract §2.2 are implemented identically in both stacks.
- **FR-708** The three control conditions are supported: `consistent = false` (inconsistent
  assignment), `matchProsody = false` (no prosody matching), `revealAfter > 0` (partial reveal).
- **FR-709** `vacancyStats` returns exactly the fields of contract §10, including
  `stressTableCoverage`, and **no prosody number from the source document is transcribed
  anywhere** — every number shown is measured on our corpus.

### Lexicon Lab (tiny arm)

- **FR-710** A vacancy panel exposes `p`, `seed`, and the three conditions, and shows the
  transform acting on the live corpus with minted forms visually distinguished from preserved
  and not-yet-vacated words — the doc's Figure 5, interactive.
- **FR-711** Raising `p` visibly demonstrates nesting and stability: a form minted at a lower
  `p` is still present, unchanged, at every higher `p`. The UI states this and the state is
  derived from the real map, not annotated by hand.
- **FR-712** Prosody preservation is reported as measured before/after statistics, always
  alongside `stressTableCoverage`, and the UI states that the stress table is rule-seeded and
  unverified (contract §6.1).
- **FR-713** The lab can train on the vacated corpus with the existing trainer, at any
  condition, reporting loss, held-out loss, and generated samples exactly as for the
  untransformed corpus.
- **FR-714** The invariance result is *demonstrated*, not asserted: the panel runs the mapped
  vocabulary at two values of `p` and shows the resulting losses are identical, with the
  identity checked in the UI rather than claimed in prose.
- **FR-715** The conditions that break invariance show what breaks: coverage collapse and
  `<unk>` rate for `consistent = false`, type splitting for `revealAfter > 0`.
- **FR-716** Vacancy composes with everything the lab already does — any budget, any budget
  source, pasted text, a HuggingFace dataset, fine-tuning, weight editing, save/load.

### Architecture Explorer (pretrained arm)

- **FR-717** A passage and its vacated twin are scored by a real curated HF model, reporting
  the fields of contract §8.1.
- **FR-718** Token→word alignment is verified by reconstruction; a mismatch raises rather than
  mis-attributing (contract §8.2).
- **FR-719** The entropy confound is stated in the UI, and the tiny arm's exact zero is shown
  next to the pretrained delta so the number is interpretable (contract §8.4).
- **FR-719a** A **swap control** exists (`mint: "nonce" | "swap"`, contract §8.3): the same
  transform drawing a real, frequency-rank-matched English word instead of a nonce form. The
  pretrained arm reports the decomposition `nll(swap) − nll(english)` (wrong content) and
  `nll(nonce) − nll(swap)` (unknown form), and never reports `nll(nonce) − nll(english)` alone
  as if it measured location. The residual tokenization component is stated, not hidden.
- **FR-720** Both stacks produce the same numbers for the same model and passage **at the same
  dtype**. They do not at the dtypes actually shipped, which is a measured fact, not an
  assumption (contract §8.3a): ONNX fp32 ≡ torch to 5.3e-4 nats, but q8 shifts absolute
  `nllPreserved` by −0.19 nats on gpt2 and +0.40 on SmolLM2-135M.
- **FR-720a** The static build reports a quantity **only** where a measured error bound exists
  for the dtype it actually ran. Pooled `nonce − english` and `swap − english` qualify under q8
  (|Δ| ≤ 0.054 nats). `nonce − swap` and every per-passage delta do **not** and are refused with
  a typed error naming the full stack. If the running dtype has no measured bound, the panel
  refuses — a stated ± that was never measured is a fabricated error bar and is worse than no
  number.

### API and static build

- **FR-721** New endpoints are **additive**: `POST /api/lex/vacancy`, `POST /api/lex/train`
  gains optional vacancy parameters, `POST /api/arch/vacancy-score`. No existing response field
  changes meaning.
- **FR-722** The static build serves the same capability: `staticClient` implements every new
  endpoint in-browser, or refuses loudly with a typed error naming the command that would fix
  it. Nothing is fabricated and nothing silently degrades.
- **FR-723** A golden fixture pins the transform across both stacks (contract §11).

### Documentation

- **FR-724** The Info tab gains a vacancy section: the 2×2, the definition of the transform,
  the nesting and stability properties, the invariance theorem and what it does and does not
  say, and the honest status of the stress table.
- **FR-725** Both tabs carry orientation prose and `Explain` deep-dives to feature 005's
  standard, and every number in that prose is transcribed from a source constant.
- **FR-726** The source document's provenance is stated: what we ported, what we corrected
  (contract §9), and that its reported prosody figures are its own, on a corpus we do not have.

## Success criteria

- **SC-701** Nesting holds: for `p < p'`, the set of vacated types at `p` is a subset of that
  at `p'`. Asserted on the real corpus across a `p` grid and two seeds, in both stacks.
- **SC-702** Stability holds: a stem's nonce is byte-identical at every `p` at which it is
  vacated, and independent of document order. Asserted on the real corpus.
- **SC-703** **The invariance theorem holds** (contract §7.3): with `consistent = true` and
  `revealAfter = 0`, the mapped-vocabulary token id stream is element-for-element identical to
  the untransformed stream, across all five Dolch budgets, a frequency budget,
  `p ∈ {0, 0.25, 0.5, 0.75, 1}`, `seed ∈ {0, 7}`, and both `matchProsody` settings. A real
  short training run at two values of `p` produces bit-identical losses.
- **SC-704** The map is injective on the real corpus at every `p` tested, with `remintRounds`
  reported.
- **SC-705** The control conditions measurably break invariance: `consistent = false` raises
  the `<unk>` rate and the held-out loss relative to `consistent = true` at the same `p`, by a
  margin recorded from a real run rather than assumed.
- **SC-706** TS↔Python parity: the golden fixture matches within `tolerance` for floats and
  exactly for strings and id streams.
- **SC-707** The pretrained arm produces a `ΔnllPreserved` whose sign and magnitude are
  reported from a real model run, next to the tiny arm's exact zero.
- **SC-707a** ~~The swap control satisfies the invariance theorem exactly as the nonce strategy
  does — the tiny model is equally blind to both.~~ **This claim was false and is retracted.**
  It cannot hold, and the reason is a theorem rather than a bug (contract §5.2a): a map that is
  stable in `p` and whose images are *domain types* is injective at every `p` only if it is the
  identity. `swap` draws its replacements from the domain by construction, so at intermediate
  `p` a swapped word can collide with a word not yet vacated — measured at 191 / 246 / 190
  colliding types for `p` = 0.25 / 0.5 / 0.75.

  The corrected criterion, and what is actually asserted: `swap` satisfies the invariance
  theorem at **`p ∈ {0, 1}`** — 48 of the 120 SC-703 cases — and the remaining 72 are
  **refused with a typed error citing §5.2a**, never silently computed. `nonce` remains
  120/120. The refusal is the deliverable: an instrument that declines the configurations it
  cannot support is sound, one that quietly returns a non-injective map is not.

  This costs the pretrained arm nothing, which is the point worth checking rather than
  assuming: it scores at full vacancy, where `swap` *is* a bijection of the domain.
- **SC-707c** The decomposition `nll(swap) − nll(english)` and `nll(nonce) − nll(swap)` is
  reported from a real model run **in the full stack**, where fp32 makes it measurable.
- **SC-707b** The measured 2×2 is reported: a word's form is worth **exactly 0** to the tiny
  model and **10–20 % of ~1.0 nats** to a pretrained one (contract §8.3a). The static build
  either states a measured uncertainty for the dtype it ran, or refuses — verified by driving
  the deployed static build and confirming it does one or the other, never a bare number.
- **SC-708** Every prosody statistic displayed is accompanied by `stressTableCoverage`, and no
  number from the source document appears as if it were ours.
- **SC-709** The full suite is green locally and in CI: backend `pytest` + `ruff` + `black`,
  frontend `vitest` + `svelte-check` (0 errors, 0 warnings) + e2e in both projects, and the
  Pages deploy.
- **SC-710** The deployed site is verified by *using* it: the vacancy panel is exercised on
  https://context-lab.com/llm-geometry/ with a real training run, and the console is clean.

## Risks

- **The theorem could be false in a way the tests do not reach.** Mitigated by asserting it as
  data (the golden fixture pins id-stream digests) as well as an assertion, and by testing at
  the boundaries — `p = 0`, `p = 1`, and a budget word absent from the corpus.
- **Token alignment in the pretrained arm** depends on what transformers.js actually exposes.
  Contract §8.2 requires this be determined empirically before implementation, and verified by
  reconstruction at run time.
- **A null result reads as a bug.** The tiny arm's headline is an exact zero. The UI must
  present it as the finding it is (contract §7.4), not hide it behind a curve that looks like
  it is measuring something.
