# Fix agent — VACANCY CORE (feature 007, the transform itself)

Date: 2026-08-04. Scope owned: `llm_geometry/lex/vacancy.py`, `lib/lexEngine/vacancy.ts`,
`specs/007-vacancy-transform-field/{architecture,spec}.md`, the `vacancy-*golden*.json`
fixtures and their exporters, and the tests for all of the above.

Answering `redteam-007-arch.md` F1 and `redteam-007-lex.md` F3 / F5 / F7.

All four tasks are fixed. Everything below was measured on the real corpus, the real
passages and real gpt2 at float32 — no mocks anywhere, and no test was weakened to go green.

---

## TASK 1 (high) — `swap` emitted forms that are not English words

### What was wrong

Two compounding causes, both confirmed:

**(a) the swap map was keyed by STEM and the SOURCE suffix was re-attached.** The pool is
drawn from domain *types*, many of them already inflected, so `jump` + `ed` drawing `went`
gave `wented`, `leap` + `ing` drawing `thy` gave `thying`, `huff` + `ed` drawing `sacks`
gave `sacksed`.

**(b) `stem_and_suffix` splits the CLOSED CLASS open.** This is worse than the red team's
"spurious splits" framing, and it is not confined to the swap arm. Measured:

```
FUNCTION_WORDS whose stem is eligible (closed-class leak):
['after', 'always', 'does', 'during', 'having', 'this', 'unless']
```

`after → aft + er`, `this → thi + s`, `does → doe + s`. None of those stems is a function
word, so the stem-level test passed all seven and **seven function words were being vacated
in both arms** — while §0 of the contract claims the closed-class scaffolding survives
character for character and §8 takes its whole measurement over the words that survive.
`after → kitser` is that leak, not just an ugly swap form.

### The fix

* **§2.2 now has a whole-word test.** New predicate `is_vacatable(word, keep)` / `isVacatable`
  in both stacks: `lower(word) ∉ keepSet` **first**, then the existing stem tests.
  `is_eligible(stem)` is kept, unchanged, as the stem-level predicate; every call site that
  judges a WORD or a TYPE now calls `is_vacatable`. (Keeping `isEligible`'s signature is
  deliberate — `viz/lex/VacancyPanel.svelte` calls it and still compiles.)
* **§8.3's swap is now a type → type map inside one suffix class.** The vacatable domain
  types are partitioned by the suffix §3 splits off (`'' s ed er ing 's es ly ies est` on the
  shipped corpus), each class is frequency-ranked by `(count desc, type asc)`, and each class
  is permuted onto itself with no fixed point: windowed draw (unchanged constants, `swap`
  tag) → deterministic outward scan → endgame exchange. **Nothing is assembled**, so the
  image IS a domain type: a real word, of the same inflectional class as the word it
  replaces. The vacancy decision is still `u(stem) < p`, so swap and nonce vacate exactly
  the same tokens — the decomposition depends on that and it is asserted.
* A suffix class with one member cannot be deranged; on a PASSAGE-sized domain that is the
  common case (five of the six shipped passages have one, `ing` in three), so such a class is
  merged into the bare class. That is the one place the inflection match bends and it is
  stated in §8.3; the image is still a real domain word. If even the bare class cannot be
  permuted, the engine raises (`ComputeError: … cannot be permuted …`).
* `VacancyMap` gained `mint` (so no consumer has to infer what the keys are) and `stems` (so
  §10's `stems*` counts stay stem counts under both strategies).
* Applying a map with params whose `mint` disagrees now raises in both stacks.

### Reproducing tests

| test | file |
|-|-|
| `test_every_swap_form_is_a_real_word_of_the_domain` | `code/backend/tests/unit/test_lex_vacancy.py` |
| `test_a_swap_replacement_keeps_the_inflection_of_the_word_it_replaces` | same |
| `test_the_suffix_splitter_breaks_function_words_open_and_test_1_stops_it` | same |
| `test_the_closed_class_survives_the_transform_under_both_mints` | same |
| `test_a_swap_class_that_cannot_be_permuted_is_refused` | same |
| `test_a_singleton_suffix_class_is_merged_into_the_bare_class` | same |
| `§8.3: every swap form is a real word of the domain` (3 cases) | `code/frontend/tests/unit/vacancy.test.ts` |
| `§2.2 test 1: the suffix splitter must not break the closed class open` (2 cases) | same |

Each fails on the pre-fix engine — the first four in substance (the measurements below), all
of them at import on HEAD, where `is_vacatable` / `swap_pools` / `MAX_SEED` do not exist.

### Before / after, six shipped Architecture passages, `p = 1, seed = 0`

Form realness (`scripts`-free probe over `variant_texts`, dictionary = `/usr/share/dict/words`):

```
BEFORE  vacatedWords 776  notADomainWord 165 (21.3%)  notInDict 195 (25.1%)  notInDict(inflection-expanded) 22
AFTER   vacatedWords 767  notADomainWord   0 ( 0.0%)  notInDict  85 (11.1%)  notInDict(inflection-expanded) 15
```

The residual 85 are real words the web2 dictionary lacks in that form or case (`Began`,
`Feet`, `box`, `heard`) plus *Mother Goose's own* nonsense words (`intery`, `cutery`,
`kyloe`, `lauk`, `splish`). That is now the honest statement of the guarantee, and it is in
§8.3: **the image is a type of the passage's own domain, so it is only as English as the
source text — but it can never be a form the source text did not contain.**

### The magnitude of the bias (gpt2, float32, backend, 6 default passages, `p = 1, seed = 0`)

```
BEFORE   wrong_content 0.7166 ± 0.0544   unknown_form 0.2726 ± 0.0415   total 0.9892 ± 0.0595   847 pairs
AFTER    wrong_content 0.6904 ± 0.0539   unknown_form 0.2872 ± 0.0450   total 0.9776 ± 0.0590   856 pairs
```

**`unknown_form` was understated by 0.0146 nats — 5.4 % of its own value — and
`wrong_content` overstated by 0.0262.** The direction is exactly the one the red team
predicted. The magnitude is a third of the sampling standard error, so no conclusion moves:
~70 % of the damage is wrong content, ~30 % unknown form, on this model and this passage set.

Corroborating, and a second win: the swap variant's token count falls from **2 858 to 2 766**
against english's 2 754, i.e. from +3.8 % to **+0.4 %** — the swap arm is now very nearly
tokenization-neutral, which is what "ordinarily tokenized" was supposed to mean.

Preserved-token count rises 847 → 856 because seven function words stopped being vacated.

### §10 numbers that moved (all re-derived, all re-pinned)

| statistic | before | after |
|-|-|-|
| `domainTypesEligible` | 1944 | **1940** |
| `corpusTypesEligible` | 1922 | **1918** |
| `stemsTotal` | 1680 | **1676** |
| `tokensVacated` at `p = 1` | 8202 | **8125** |
| `corpusTypesVacated` seed 0 | 0/461/954/1430/1922 | **0/461/951/1427/1918** |
| `corpusTypesVacated` seed 7 | 0/434/975/1440/1922 | **0/433/972/1436/1918** |
| `domainTypesVacated` seed 0 | 0/469/966/1448/1944 | **0/469/963/1445/1940** |
| `domainTypesVacated` seed 7 | 0/440/985/1455/1944 | **0/439/982/1451/1940** |
| `revealAfter = 2, p = 0.7`: texts / membership | 665 / 1337 | **663 / 1334** |
| swap prosody match, seed 0 | 1586 of 1680 stems (94.4 %) | **1918 of 1940 types (98.9 %)** |

Unmoved: `domainTypesTotal` 2233, `corpusTypesTotal` 2211, `tokensTotal` 16000, `bijective`,
`imageSize` 2233, seed 7's single re-mint (`hang → smeeg`, `wak` still in `forbidden`).

---

## TASK 2 (medium) — TS ↔ Python divergence for `seed > 2^53`

Independently reproduced, mechanism first:

```
python str(2**53+1)         = 9007199254740993
javascript String(2**53+1)  = 9007199254740992      (Number.isInteger(2**53+1) === true)
  u('little') python=0.8658968709774663 javascript=0.7766856381143197 equal=False
  u('moon')   python=0.7949469779125994 javascript=0.6415060224178482 equal=False
```

**Chosen fix: constrain the domain and enforce it identically — raise, never clamp.**
`MAX_SEED = 2**53 - 1 = 9007199254740991` in both stacks, checked in `VacancyParams` /
`vacancyParams` **and** inside `vacancy_u` / `vacancyU` (the latter is public and reachable
directly). `Number.isInteger` was not a sufficient guard: `9007199254740993` passes it,
having already been rounded on the way in. Clamping was rejected on the ground the red team's
F4 gives — a number used that is not the number asked for.

The bound is stated in contract §4 (with the measurement and the reason) and in the §9
departures table as row 13.

Tests: `test_seed_is_bounded_to_exactly_representable_integers`
(`code/backend/tests/unit/test_lex_vacancy.py`) and
`§4: the seed is bounded to the integers JavaScript represents exactly`
(`code/frontend/tests/unit/vacancy.test.ts`). Both pin the boundary in both directions:
`±(2^53 − 1)` accepted, `±2^53`, `2^53 + 1` and `12345678901234567890` refused.

The HTTP route needs no change: `routes_lex.py` builds a `VacancyParams`, so an out-of-range
seed now returns a typed `400 InvalidParamError` instead of `200`.

---

## TASK 3 (medium) — the "measured" collision count in the contract

Neither the document's 191 / 246 / 190 nor the red team's 244 / 322 / 233 could be a stable
claim, because **the sentence named no seed and no counting definition**. Both are now stated:

> **lost image slots** `= |domain| − |{T_p(t) : t ∈ domain}|` — how many of the domain's
> 2 233 slots the type map at `p` fails to reach. It is what `lex-vacancy-lost-slots`
> measures live in the panel.

Derived last, on the fixed swap, at the seeds named:

```
seed 0:  0 / 349 / 484 / 364 / 0   at p = 0 / 0.25 / 0.5 / 0.75 / 1
seed 7:  0 / 336 / 475 / 372 / 0
```

Computed independently in each stack and asserted equal to those literals:
`test_swap_collisions_at_intermediate_p_are_measured_under_one_definition` (Python) and
`§5.2a: swap's lost image slots, measured under one stated definition` (TypeScript). The
`0` at `p ∈ {0, 1}` half of the old sentence reproduced and is kept.

`architecture.md` §5.2a and `spec.md` SC-707a now carry the new numbers, the definition, the
seed, and a note recording that the old figure named neither and was read by no test.

---

## TASK 4 (low) — `domainTypesVacated` computed by two rules

Both stacks now use **map membership only**: the type is vacatable (§2.2) and `u(stem) < p`.
TypeScript's extra `surfaceForm(...) !== t` clause is gone. The two readings coincide (B and
B₁ both forbid an image equal to its own source), and that coincidence is now *asserted*
rather than relied on, so a condition-B regression fails the definition rather than silently
producing two numbers:

* `test_domain_types_vacated_is_map_membership_under_both_mints` (Python)
* `§10: domainTypesVacated is map membership, under both mints` (TypeScript)

Both check `stats.domainTypesVacated == byMembership == byImage` at every `p`, for both mints.
§10 of the contract states the rule.

---

## Verification

* `code/backend`: `pytest -q` — **471 passed**, zero failures (341 unit + contract, 121
  integration with real gpt2, plus the 9 new cases). `ruff check src/ tests/` clean, `black --check src/ tests/` clean.
* `code/frontend`: `npx vitest run` — **467 passed, 1 skipped, 20/20 files**. `npm run check`
  (svelte-check) — 0 errors, 0 warnings.
* The differential suite is the golden fixture, regenerated from the real Python backend
  (`scripts/export_vacancy_golden.py`, 10 maps / 25 cases) and re-run against the real
  TypeScript engine: 81/81 in `vacancyGolden.test.ts`, including the full 2 233-entry maps at
  both seeds, the sha256 of the whole 86 kB vacated corpus in every case, all 23 §10 fields,
  and the swap maps at both seeds × both prosody settings.
* **SC-703's invariance proof still holds**: `test_sc703_over_the_full_grid_for_both_mint_strategies`
  passes unchanged — nonce 120/120, swap 48 passed / 72 refused — and the fixture's `idStream`
  digests are still equal across every `p`. Nothing in the theorem moved.
* Fixtures regenerated from the real backend, never hand-authored:
  `code/frontend/tests/fixtures/vacancy-golden.json` and `vacancy-api-golden.json`.
  `arch-vacancy-passages.json` pins only the ENGLISH passages and is unaffected (verified).
* `scripts/export_static_assets.py --quick` runs clean end to end on the new engine.

Not run, per charter: `npm run test:e2e`.

---

## Files I do NOT own that now state a number or a claim that is no longer true

**Every one of these is a wrong number on screen or in a docstring today.** I did not edit
them. `tests/e2e/docs.spec.ts` reads several of them off the live API and asserts the Info
prose contains them, so items 1 and 2 will FAIL that suite until they are updated.

1. `code/frontend/src/viz/info/InfoTab.svelte:705-708` — "**2,233** types … of which
   **1,944** are eligible, sharing **1,680** distinct stems. At `p = 1` that rewrites
   **8,202** of the corpus's **16,000** word tokens" → 1,944 → **1,940**, 1,680 → **1,676**,
   8,202 → **8,125**. (2,233 / 2,211 / 16,000 are unchanged.)
2. `code/frontend/src/viz/info/InfoTab.svelte:815-817` — "swap loses **244 / 322 / 233** image
   slots at `p = 0.25 / 0.5 / 0.75`" → **349 / 484 / 364** (seed 0; the panel measures it live,
   and `docs.spec.ts` compares the two).
3. `code/frontend/src/viz/info/InfoTab.svelte:803-807` — "Swap draws its replacements from a
   finite pool — **1,944** eligible domain types against the map's **1,680** stems — so the
   tail of the canonical order draws from what is left and its frequency match degrades; and
   a source type carrying a suffix may receive an already inflected replacement." The second
   half is **no longer true and must be deleted**: nothing is assembled, so no doubly-inflected
   surface can exist. The pool is now per suffix class (1 280 / 216 / 106 / 92 / 81 / 67 / 53 /
   29 / 8 / 8 for `'' s ed er ing 's es ly ies est`).
4. `code/frontend/src/viz/info/InfoTab.svelte:823` — "1,680 stems cannot cover 8,202 vacated
   tokens" → **1,676** / **8,125**.
5. `code/frontend/src/viz/lex/LexiconLab.svelte:223-224` — same pair in a comment:
   "1 680 open-class stems against 8 202 vacated tokens" → **1 676** / **8 125**.
6. `code/frontend/src/viz/lex/VacancyPanel.svelte:354-356` — **a code defect, not prose**:
   `[...map.mapping.keys()] … .map((stem) => ({ stem, u: vacancyU(stem, params.seed), … }))`
   treats the map's keys as STEMS. Under `mint = "swap"` they are now TYPES, so a row for a
   type whose own stem differs (`flower`, whose stem is `flow`) would show the wrong `u` and
   the wrong "vacated at p" cells. Branch on `map.mint`, or filter with `map.stems`.
7. `code/frontend/src/viz/lex/VacancyPanel.svelte:668-690` — the swap explainer says swap
   "draws a real English word instead"; it should now also say the replacement carries the
   **same inflection** (same suffix class) and that a singleton class merges into the bare one.
8. `code/backend/src/llm_geometry/arch/vacancy_score.py:12-14` — "every vacated **stem**
   replaced by a REAL, frequency-rank-matched English word" → it is now every vacated **word**,
   replaced by a real frequency-rank-matched word **of the same inflectional class**. (The
   property the docstring claims is now true; only the mechanism sentence is stale.)
9. `code/frontend/src/viz/arch/VacancyScorePanel.svelte:23-27` — same "replaces each vacated
   stem" wording.
10. `code/frontend/src/lib/staticClient/arch.ts:157-170` — **the most serious of these.** The
    justification for `VACANCY_Q8_UNCERTAINTY` quotes a measurement made with the OLD swap:
    "identical tokenization (2754/2858/3810 tokens, 847 preserved) … swap − english: fp32
    0.7166 q8 0.6440 … nonce − english: fp32 0.9892 q8 0.8790". The fp32 side is now
    **0.6904** and **0.9776**, with **2754/2766/3792** tokens and **856** preserved. The q8
    side cannot be re-measured outside a browser, so the stated ±0.2 now rests on a
    comparison whose fp32 arm has moved. I could not fix this: it needs a real q8 browser run
    on the new engine, and the file is not mine. **Whoever owns it must re-run both stacks on
    the default set before the constant or its prose is trusted again.**
11. `code/frontend/tests/e2e/docs.spec.ts:325-329` — comments only (`// 1,944 eligible`,
    `// 1,680 distinct stems`, `// 8,202 rewritten`); the assertions read the live API and are
    correct as written.
12. `notes/2026-08-04-feature-007-vacancy-transform.md` — the session notes quote the old §10
    numbers throughout. History, not a claim about the shipped code; flagging it only so the
    next reader is not misled.

---

## Things I changed in files that are arguably not mine, and why

* `code/backend/tests/contract/test_api_lex.py` — two assertions of the §10 constants
  (1944/1922/1680/8202 → 1940/1918/1676/8125). The test is right; only the pinned numbers
  moved. Leaving it red was not an option and the alternative was to leave the suite failing.

## What I could not fix

* **The q8 half of `staticClient/arch.ts`'s uncertainty measurement** (item 10 above) — it
  requires a browser run I am not permitted to make here, on a file I do not own.
* **The swap map's key change is visible to `VacancyPanel`** (item 6). I could have kept a
  stem-keyed compatibility projection, but every option I could see either put a wrong number
  on that panel or introduced a second, derived map — exactly the §5.8 asymmetry class that
  produced these defects. Carrying `mint` and `stems` on the map and reporting the caller is
  the honest form.
* **`is_eligible`'s two readings.** I kept `is_eligible(stem)` with its old signature so the
  panel still compiles; that leaves two predicates in the API, and a future caller could pick
  the wrong one. §2.2 now says in bold which is which, and the golden fixture pins BOTH per
  stem (`eligible` and `vacatable`) so a stack that swaps them fails.
