# Fix 007 — round 7, arch slice (agent "ROUND 7 ARCH")

Date: 2026-08-04. Branch `main`. Charter: findings F2–F6 of
`notes/agent-reports/verify-007-round5-arch.md`, plus F6 of `verify-007-round5-geolex.md`
(`ONNX_REPOS[m.model_id]`). A sibling agent was editing the geo/lex slice throughout; every
mutation below was applied in place, restored from a `cp` backup and re-verified with
`shasum -a 1` — never `git checkout`. `npm run test:e2e` was not run. No dev stack started.

## Task 1 — the ASCII escape hatch reopened the fragment-rewrite defect (F2) — FIXED

**What was wrong.** Round 5 closed a false refusal (`legs--upon`) with

```python
if all(ch.isascii() and (ch.isalpha() or ch in WORD_RE_JOINERS) for ch in run):
    continue
```

— a test of *which characters the run is written in*. `don''t` passes that test, is still
`don` + `t` under `WORD_RE`, and vacated to `little''t` with HTTP 200: character for
character the `don’t` → `big’t` defect the refusal exists to prevent. Pre-round-5 it was
refused. `word-alphabet-cases.json` case 24 pinned the broken behaviour as correct
(`{"text": "don''t", "fragmented": []}`), i.e. the fixture was moved to match the code
instead of the code being checked against the requirement.

**The fix** (`arch/vacancy_score.py:420-431`, mirror `staticClient/byteSpans.ts:212-239`).
The exemption is now the *em-dash convention only*, and it is a WHOLE-MATCH test:

```python
pieces = [piece for piece in EM_DASH_RE.split(run) if piece]   # EM_DASH_RE = -{2,}
if pieces and all(WORD_RE.fullmatch(piece) for piece in pieces):
    continue
```

Two or more ASCII hyphens between letters are a dash *between* words (punctuation), so the
run is cut there and every piece must be matched whole by `WORD_RE`. That is the real
distinction the charter names: `legs--upon` and `don''t` are both written entirely in
`WORD_RE`'s alphabet; only the first has pieces `WORD_RE` matches whole.

Verified behaviour (real functions, both stacks, identical answers):

| input | before (round 5) | now |
|-|-|-|
| `legs--upon`, `ba--are`, `hea--art`, `Lady--loves`, `a---b`, `legs--upon's` | accepted | accepted |
| `don''t`, `don-'t`, `don'-t`, `don''''t` | **accepted (defect)** | refused |
| `café--x` | refused | refused |
| `don’t`, `co<SHY><SHY>operate`, `don’’t` | refused | refused |
| the six default passages | `[[], [], [], [], [], []]` | `[[], [], [], [], [], []]` |

**Fixture corrected, and why that is not weakening a test.** `don''t` now expects
`["don''t"]`; the `why` field records that it was pinned as non-fragmented on 2026-08-04 and
what it did (scored 200, vacated to `little''t`). Three cases added: `don-'t`, `café--x`,
`legs--upon's`. The refusal message also changed in both stacks — the old advice ("use …
straight apostrophes and hyphens") was advice `don''t` already followed; it now says ONE
apostrophe or hyphen between letters, names `legs--upon` as fine and `don''t` as not.

**Tests** (they distinguish the two properties on runs that differ in exactly one of them —
both groups assert `^[A-Za-z'-]+$` before asserting the verdict):
`test_the_em_dash_exemption_is_a_whole_match_test_not_an_alphabet_test` (Python),
`exempts the em dash by WHOLE MATCH, not by the alphabet the run is written in` (TS), plus
the shared case table in both suites.

**Mutation (the verifier's own, re-applied):** restore the round-5 ASCII hatch →
Python `2 failed, 5 passed` (the new test + the shared case table);
TS `Tests 2 failed | 40 passed (42)`. Killed in both stacks.

## Task 2 — the constants were a facade (F3) — FIXED

**What was wrong.** The Python "pin" asserted its own literals (`pytest.approx(0.2872, …)`)
and never read the TypeScript file; the TS test did `toContain(String(constant))` after
interpolating that constant. `unknownForm` 0.2872 → 0.4872 passed all 815 tests.

**The fix — three real links, no number typed twice.**

```
real gpt2 run → specs/007-vacancy-transform-field/fp32-reference.json → VACANCY_FP32_REFERENCE
```

* `scripts/measure_vacancy_fp32.py` (new, committed) runs `vacancy_score("gpt2",
  default_passages(), p=1, seed=0)` and writes the record. I ran it: 2754/2766/3792 tokens,
  856 preserved in every variant, `wrong_content 0.6904174945163003 ± 0.05389436515855508`,
  `unknown_form 0.28721050610900456 ± 0.04496363720801029`,
  `total 0.9776280006253049 ± 0.05903211085957414` — matching the verifier's independent
  re-derivation exactly.
* `test_the_fp32_arm_quoted_in_the_static_client` now asserts **both** arrows: the live gpt2
  run against the record (`abs=1e-9`), and `VACANCY_FP32_REFERENCE` **parsed out of
  `staticClient/arch.ts`** against the same record (`abs=5e-4`). It contains no measurement
  literal of its own.
* `archVacancy.test.ts` → `ships fp32 constants that equal the recorded run of the real
  model` asserts the constant against the record from the browser side.

**Mutations:** TS `unknownForm` 0.2872 → 0.4872 → frontend `Tests 1 failed | 34 passed`
*and* backend `FAILED …::test_the_fp32_arm_quoted_in_the_static_client` (a TypeScript-only
edit now fails a Python test — the cross-stack link is real). Editing the *record* instead
(0.28721 → 0.30721) fails against the live model:
`assert 0.28721050610900456 == 0.3072105061090046 ± 1.0e-09`.

## Task 3 — the interpolation slots were position-blind (F4) — FIXED

`puts each figure in the clause it belongs to, not merely somewhere in the message`
(`archVacancy.test.ts`) matches anchored patterns:
`configuration that ships — gpt2, float32, … it is <fp32.unknownForm> ± <se> nats over <n>
paired tokens` and `variant texts that no longer exist: there float32 read <pre.fp32> and q8
read <pre.q8>, a 14 % error`, plus two negative patterns (neither number may appear inside
the other's clause) and an assertion that the two values differ at all.

**Mutation (the verifier's own):** exchange `${VACANCY_FP32_REFERENCE.unknownForm}` ↔
`${VACANCY_PRE_REWRITE_Q8.unknownForm.fp32}` → `Tests 1 failed | 34 passed (35)`. Killed.

## Task 4 — `ONNX_REPOS[m.model_id]` truthiness (geolex F6) — FIXED, and swept

`ONNX_REPOS` is now a **frozen null-prototype** table read only through
`onnxRepo(modelId)` (`Object.hasOwn`) and `onnxRepoIds()`; both call sites
(`arch.ts:935`, `:1095` — vacancy scoring and live generation) go through it.

**Sweep of my files** for the same shape: the only other dynamic lookups are
`texts[name]` / `scored[name]` / `preservedIdx[name]` / `previews[name]` / `per[name]`,
all keyed by the frozen `VACANCY_VARIANTS` literals (never user input); `lexEngine/vacancy.ts`
already uses a null-prototype table (`METER_FEET`); the Python `arch/*` side has no
attacker-controlled dict/`getattr` lookup of this shape (`getattr` calls in `gate.py`/
`generate.py` read fixed config attribute names). No further instances found.

**Test:** `archStaticFacts.test.ts` → `returns undefined for inherited properties instead of
an Object` (8 prototype keys). **Mutation:** revert to a plain object literal + direct index
→ `Tests 1 failed | 3 passed`. Killed.

## Task 5 — the remaining literals (F5) and the unpinned floor (F6) — FIXED

* **Interpolated:** `−0.19`/`+0.40` (new `VACANCY_PRE_REWRITE_Q8.absoluteShiftNats`, rendered
  by `signedNats`), `0.65` (`study.perPassageWorstNats`), `0.28`
  (`study.unknownFormWorstPassageNats`), `5.3e-4` (new
  `VACANCY_ONNX_TORCH_AGREEMENT_NATS`, rendered with `.toExponential(1)` because
  `${5.3e-4}` stringifies as `0.00053`). **The three rendered messages are byte-identical to
  what shipped** — verified by printing them before and after.
* **The claim is now checked, not repeated:** `hand-types no decimal figure in any refusal it
  can emit` extracts every decimal from all three refusals and requires each to be derivable
  from a constant. **Mutation:** hand-type `0.66 nats` in place of the interpolation →
  `Tests 1 failed | 34 passed`. Killed.
* **The Info tab's numbers:** `archStaticFacts.test.ts` reads `InfoTab.svelte` and pins its
  `700` to `VACANCY_MIN_POOLED_PRESERVED`, and its `0.054 / 0.073 / 0.110` to
  `study.pooledBoundNats` and to the *differences* of `VACANCY_PRE_REWRITE_Q8` — so those
  three are checked as arithmetic on the constants, not as three more literals. I did **not**
  interpolate the §10 counts (`2,233 / 1,676 / 8,125`, `349 / 484 / 364`): `docs.spec.ts`
  already compares them with values read from the running app, which is a stronger check than
  interpolation. Nothing now claims that every sentence is interpolated.
* **The floor:** `VACANCY_MIN_POOLED_PRESERVED` is exported and the gate extracted into
  `assertPooledPreservedBoundable(n)` (the scorer calls it), so it can be driven without a
  browser. `refuses a pool smaller than the size the retained q8 bound was measured at`
  checks: one default passage's worth of preserved tokens **derived from the recorded run**
  (856/6 ≈ 143) must be refused; the boundary at 699/700; and the constant equals the
  "~700 preserved closed-class tokens per condition" of `architecture.md` §8.3a, read from
  §8.3a. **Mutation:** 700 → 7 → `Tests 2 failed | 37 passed` (this test and the Info tab
  pin). Killed at unit level, where round 5's mutant survived.

## Suites (final, tree restored)

| suite | result |
|-|-|
| backend `pytest -q` | `680 passed, 1 warning in 169.48s` (cold cache); re-run on the same warm cache: `1 failed, 679 passed` — see below |
| backend `ruff check .` | `All checks passed!` |
| backend `black --check .` | `88 files would be left unchanged` |
| frontend `npm run check` | `1186 FILES 0 ERRORS 0 WARNINGS` |
| frontend `npx vitest run` | `Test Files 35 passed (35) · Tests 851 passed | 1 skipped (852)` |
| frontend `npm run build` | `✓ built in 3.69s` |

### An error I hit and did not cause (reported, not dismissed)

The SECOND backend run, against the cache the first run had just filled, failed:

```
FAILED tests/integration/test_geo_finetune.py::test_finetune_on_text_mints_new_checkpoint_and_learns
>       assert result["cached"] is False
E       assert True is False
1 failed, 679 passed, 1 warning in 178.66s
```

It is a **warm-cache order dependency**, not a code defect and not mine: the same file
against a fresh `LLM_GEOMETRY_CACHE_DIR` is `5 passed in 37.95s`. The test asserts the
fine-tune was computed rather than served from cache, which is only true the first time that
`(model + params)` is seen. That matters beyond a local re-run — CI restores
`.cache/llm-geometry` from a previous run's key, so a cache carrying this checkpoint would
fail the same way. `llm_geometry/geo/*` is the sibling agent's slice this round, so I have
recorded it here rather than editing their files mid-flight.

Counts move between runs because the sibling agent was adding tests to the geo/lex slice in
the same tree. SC-703 and the differential suite were not touched: `vacancy`,
`vacancyGolden`, `staticVacancy` and `archVacancy` all pass, and no §10 number changed.

## Files changed

`arch/vacancy_score.py`, `staticClient/{arch,byteSpans}.ts`,
`tests/unit/test_arch_word_classes.py`, `tests/integration/test_arch_vacancy_score.py`,
`tests/unit/{archVacancy,wordClasses}.test.ts`, new `tests/unit/archStaticFacts.test.ts`,
new `scripts/measure_vacancy_fp32.py`, new
`specs/007-vacancy-transform-field/fp32-reference.json`, and the two spec documents
(`architecture.md` §8.2/§8.3a, `word-alphabet-cases.json`).

## What I did not verify

1. **The q8 arm.** Unchanged from every previous round: it needs a browser. No q8 number was
   computed or extrapolated here.
2. **Anything rendered.** No e2e, no browser. The Info tab assertions are made against the
   component source, not the DOM; `docs.spec.ts` remains the only check of the rendered page,
   and I did not run it. `docs.spec.ts` was not modified, and the constant it greps
   (`VACANCY_MIN_POOLED_PRESERVED = 700`) is still in `staticClient/arch.ts` under that name,
   now `export const` — the regex `/VACANCY_MIN_POOLED_PRESERVED = (\d+)/` still matches.
3. **Whether `don''t` occurs in text a user would paste.** As in round 5: I showed both
   stacks refuse it now; I did not survey corpora for it.
4. **CI's runtimes.** Local macOS only: Node v22.16.0, CPython 3.10 in the repo venv.
