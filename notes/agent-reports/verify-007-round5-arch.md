# Verify 007 — round 5, against `fix-007-arch-r5.md` and `fix-007-teeth.md` (agent "V6a")

Date: 2026-08-04. Branch `main` @ `9f0bc5e`. `npm run test:e2e` was not run; no MCP browser
tool was used. Every mutation was applied in place, restored from a `cp` backup in the session
scratchpad and re-verified by `shasum -a 1` against that backup (never `git checkout`). Final
state verified: all seven mutated files byte-identical to their backups, `git status --short`
carries only the sibling agent's untracked `notes/agent-reports/verify-007-round5-geolex.md`.

**The tree was not clean when I started.** `git status --short` at t0:

```
 M code/frontend/src/lib/geoEngine/index.ts
 M code/frontend/src/lib/staticClient/geo.ts
```

both being a live `>=` → `>` mutation of `FINETUNE_MAX_UNK_RATE` by a sibling agent. My first
baseline run (`npx vitest run` → `Test Files 34 passed (34) · Tests 815 passed | 1 skipped
(816)`) therefore ran **with that mutation live and green**, which independently re-confirms
round 4's F5 note that no TypeScript test pins the `>=` unk bound. Both files were back to
`HEAD` before any of my own mutations, and stayed so.

| # | charter item | verdict |
|-|-|-|
| 1 | stale "measured" prose, fourth attempt | **PARTIAL** — every shipped number is correct today and 0.28721 re-derived exactly; the *structure* does not hold: two mutants restore the identical defect with 815/815 green |
| 2 | the joiner class, third attempt | **PARTIAL** — skew genuinely closed (0 disagreements in 4 719 cross-stack cases); one NEW hole, pinned as correct |
| 3 | mutation teeth | **PARTIAL** — all three re-applied kills confirmed; three fresh mutants survive |
| 4 | "no production behaviour changed" in `c603c9a`/`55c7b5a`/`9f0bc5e` | **VERIFIED** |
| 5 | SC-703 + the TS↔Python differential suite | **VERIFIED**, re-derived |

Suites, with the tree restored: backend `pytest -q` → `624 passed, 1 warning in 167.75s`;
frontend `npx vitest run` → `Test Files 34 passed (34) · Tests 815 passed | 1 skipped (816)`.

---

## (a) Defects surviving

### F1. No TypeScript test pins the `>=` unk bound — still true, and now demonstrated by accident
**Severity:** low
**Where:** `code/frontend/src/lib/geoEngine/index.ts:596`,
`code/frontend/src/lib/staticClient/geo.ts:275`
**Reproduce:** `git diff` at session start (a sibling agent's in-flight mutation), then
`npx vitest run`.
**Observed:** `-    if (unkRate >= FINETUNE_MAX_UNK_RATE)` / `+    if (unkRate > FINETUNE_MAX_UNK_RATE)`
in both files, with `Test Files 34 passed (34) · Tests 815 passed | 1 skipped (816)`.
**Expected:** a stream that is exactly 90 % `<unk>` must be refused; the comment on the line
says so in both files. Round 4 flagged this as unpinned; it is unpinned.
**Would it have thrown?** No — a 90 %-`<unk>` fine-tune would report a clean loss drop.
(Not my lane to fix; recorded because I observed it.)

---

## (b) NEW defects round 5 introduced

### F2. The ASCII escape hatch reopens the fragment-rewrite defect: `don''t` scores 200 and swaps to `little''t`
**Severity:** high
**Where:** `code/backend/src/llm_geometry/arch/vacancy_score.py:410`
(`if all(ch.isascii() and (ch.isalpha() or ch in WORD_RE_JOINERS) for ch in run): continue`)
and its mirror `code/frontend/src/lib/staticClient/byteSpans.ts:221`
(`if (WORD_RE_ALPHABET.test(run)) continue;`). Pinned as *correct* by
`specs/007-vacancy-transform-field/word-alphabet-cases.json` case 24
(`{"text": "don''t", "fragmented": []}`).
**Reproduce:** a passage containing `don''t`, through the real transform on the real model.
**Observed** (verbatim, `variant_texts(p=1, seed=0)` on a passage containing `don''t`):
```
english :: … while they don''t wander far from home …
swap    :: … while they little''t farmer far from white …
nonce   :: … while they dud''t sherrder far from mout …
```
and end to end, `vacancy_score("gpt2", [passage], p=1.0, seed=0)`:
```
"don''t"      SCORED {'wrong_content': 0.9517…, 'unknown_form': 0.1156…, 'total': 1.0673…}
'legs--upon'  SCORED {'wrong_content': 0.5880…, 'unknown_form': 0.4235…, 'total': 1.0115…}
'don’t'       RAISED InvalidParamError passage 0: the vacancy transform's word alphabet is …
```
Both stacks agree (`fragmentedWords("the don''t brown fox")` → `[]` in TS as well), so this is
one shared hole, not a divergence.
**Expected:** refusal. This is character-for-character the `don’t` → `big’t` defect the joiner
class exists to close: one wordlike run, `WORD_RE` matching `don` and `t`, one half rewritten
and a character `WORD_RE` cannot see surviving between the halves. Pre-round-5 it *was*
refused — `71061fc:…/vacancy_score.py` had no exemption
(`if parts and (len(parts) > 1 or parts[0] != run): out.append(run)`), and `'` was already a
joiner in `WORD_JOINER_CHARS`. The exemption's own stated justification —
"each piece is then a whole ASCII word the transform vacates *as* a word, and no character
survives inside a rewritten fragment" (`fix-007-arch-r5.md`, and repeated verbatim in both
docstrings) — is a claim about *what happened to the pieces*, but the code tests *the run's
alphabet*. `t` is not vacated, so the property does not hold and nothing checks it.
**Would it have thrown?** **No.** HTTP 200 with a plausible three-term decomposition.

### F3. `VACANCY_FP32_REFERENCE` is not pinned to the measurement it claims to be pinned to
**Severity:** high
**Where:** `code/frontend/src/lib/staticClient/arch.ts:215-225`; the claimed pin is
`code/backend/tests/integration/test_arch_vacancy_score.py:300`.
**Reproduce:**
```
perl -pi -e 's/^  unknownForm: 0\.2872,$/  unknownForm: 0.4872,/' src/lib/staticClient/arch.ts
npx vitest run
```
**Observed:** `Test Files 34 passed (34) · Tests 815 passed | 1 skipped (816)`.
**Expected:** the constant's docstring says "the constants are pinned to a real run of the real
model by `test_the_fp32_arm_quoted_in_the_static_client` … and to the sentences by
`tests/unit/staticVacancy.test.ts`. **A number cannot move without both failing.**" It can. The
Python test asserts its *own* literal `pytest.approx(0.2872, abs=5e-4)` and never reads the
TypeScript file; the TypeScript test asserts
`expect(msg).toContain(String(VACANCY_FP32_REFERENCE.unknownForm))`, which is a tautology once
the message interpolates the constant. `grep -rn "0\.2872"` finds exactly two code sites, in
two languages, with nothing between them. No e2e assertion covers it either
(`grep -rn "FP32_REFERENCE\|0\.2872\|unknownForm" tests/e2e/` → no matches).
**Would it have thrown?** No. The failure mode was moved from prose into a constant, not removed.

### F4. The interpolation is position-blind: the fp32 and pre-rewrite figures can be exchanged with 815/815 green
**Severity:** high
**Where:** `code/frontend/src/lib/staticClient/arch.ts:530` and `:536`, guarded by
`tests/unit/archVacancy.test.ts:309-337`.
**Reproduce:** exchange the two interpolation slots —
`${VACANCY_FP32_REFERENCE.unknownForm}` ↔ `${VACANCY_PRE_REWRITE_Q8.unknownForm.fp32}` — and
run the full suite.
**Observed:** `Test Files 34 passed (34) · Tests 815 passed | 1 skipped (816)`, with the shipped
refusal now reading:
> "on the configuration that ships — gpt2, float32, the six default passages, p = 1, seed = 0 —
> it is **0.2726** ± 0.045 nats over 856 paired tokens … on variant texts that no longer exist:
> there float32 read **0.2872**"

i.e. the pre-rewrite value asserted of the shipped configuration and the shipped value labelled
as history — the exact defect of rounds 2, 3 and 5, restored, undetected. All four `toContain`
assertions still pass because all four numbers are still somewhere in the string, and the three
`toMatch` guards check phrases that are still present.
**Expected:** an assertion that pins each number to its *clause*, not to the message.
**Would it have thrown?** No.

### F5. "Two exported constants now hold the figures, and every sentence is interpolated from them" is false
**Severity:** medium
**Where:** `staticClient/arch.ts:493-497` (`VACANCY_ABSOLUTE_REFUSAL`: `−0.19`, `+0.40`),
`:505` (`VACANCY_PER_PASSAGE_REFUSAL`: `0.65 nats` — literal, while the `115 %` beside it is
interpolated), `:539` and `:546` (`0.28 nats on a single passage`, `5.3e-4 nats`),
`viz/info/InfoTab.svelte:971` (`700`), `:1170` (`0.054`, `0.073`, `0.110`), `:906`
(`349 / 484 / 364`, `2,233`, `1,676`, `8,125`).
**Observed:** each of these is a hand-typed literal in a shipped user-facing string.
**Expected:** the round-5 report states the structural rule without qualification. I traced
**every one** of them to `specs/007-vacancy-transform-field/architecture.md` §8.3a and to a live
computation, and **none is currently wrong** (see (d)) — but the mechanism that was introduced
precisely because literals rot covers only three of them.
**Would it have thrown?** No.

### F6. Lowering the quantization floor from 700 to 7 fails no unit test
**Severity:** medium
**Where:** `staticClient/arch.ts:265` (`VACANCY_MIN_POOLED_PRESERVED = 700`).
**Reproduce:** set it to `7`, `npx vitest run`.
**Observed:** `Test Files 34 passed (34) · Tests 815 passed | 1 skipped (816)`. The static build
would then print pooled q8 differences on samples two orders of magnitude below the size the
retained bound was taken at — the condition under which a single-passage delta was wrong by
115 % of its own value.
**Expected:** a unit gate. The only gate is `tests/e2e/docs.spec.ts:474-490`, which extracts the
constant from the source and compares it to the Info tab's hand-typed `700` — so e2e *would*
catch this one, and unit CI would not. Recorded as the difference between the two gates, not as
an unguarded constant.
**Would it have thrown?** No.

---

## (c) Claims I could not verify

1. **The q8 arm.** Unchanged from three previous agents: it needs a real browser running this
   build's static scorer. No q8 number was computed, guessed or extrapolated here.
2. **Anything rendered.** No e2e, no browser, no screenshot. Every static-mode finding above is
   verified by running the real exported functions under `vite-node`/`vitest`.
3. **CI's runtimes.** All runs were local macOS: Node `v22.16.0` (ICU Unicode 16.0) and
   CPython `3.10.12` (`unicodedata` 13.0 — which *is* the version CI pins). Node 20
   (Unicode 15.0) was not exercised; see (d) for why I believe the pin is defended against it.
4. **Whether `don''t` (F2) occurs in text a user would paste.** I showed the transform mangles
   it and the endpoint scores it; I did not survey real corpora for it. The four `--` instances
   the escape hatch was written for are real (`ba--are`, `hea--art`, `Lady--loves`,
   `legs--upon`); a doubled ASCII apostrophe is a typo, not a convention.
5. **A user-visible surface for F3/F4.** The mutated string is `VACANCY_UNKNOWN_FORM_REFUSAL`,
   which `VacancyScorePanel.svelte:270` renders in static mode; I did not observe the DOM.

---

## (d) What I confirmed genuinely fixed

* **0.28721 re-derived independently, on the real stack.** Not by running their test — by
  calling `vacancy_score("gpt2", default_passages(), p=1.0, seed=0)` myself:
  ```
  english 2754 856 4.10153864807423
  swap    2766 856 4.79195614259053
  nonce   3792 856 5.079166648699535
  wrong_content 0.6904174945163003 0.05389436515855508 856 quant= None
  unknown_form  0.28721050610900456 0.04496363720801029 856 quant= None
  total         0.9776280006253049 0.05903211085957414 856 quant= None
  ```
  Exactly the claimed `0.28721050610900456 ± 0.04496363720801029` over 856 pairs, and exactly
  the shipped `VACANCY_FP32_REFERENCE` (0.6904 / 0.2872 / 0.045 / 0.9776 / 856). The backend
  attaches no quantization term (`quant= None` on all three), as round 4 also found.
* **Every other shipped number traced to a source and checked.** `VACANCY_PRE_REWRITE_Q8`:
  0.7166 − 0.644 = 0.0726 ≈ the §8.3a "0.073"; 0.9892 − 0.879 = 0.1102 ≈ its "0.110";
  0.2726 = 0.9892 − 0.7166 and 0.235 = 0.879 − 0.644; `(0.2726 − 0.235)/0.2726 = 13.8 %` → the
  stated 14 %; `study.unknownFormRange "0.06–0.21"`, `pooledBoundNats 0.054`,
  `perPassageWorstPercent 115`, `−0.19`/`+0.40`, `0.65`, `0.28`, `5.3e-4` all verbatim in
  §8.3a. **No stale figure and no unqualified "measured" claim remains** in `arch.ts`,
  `vacancyVerdict.ts`, `VacancyScorePanel.svelte`, `InfoTab.svelte`, `vacancy_score.py` or
  `specs/002-…/contracts/api.md` (the contract quotes no vacancy number at all and was last
  touched in its own commit, `9aa2579`).
* **The Info tab's live-derived numbers are right.** `349 / 484 / 364` lost image slots under
  `swap` at `p = 0.25 / 0.5 / 0.75`, seed 0, and 0 at both endpoints — re-derived exactly:
  ```
  swap p= 0.25 types 2233 distinct 1884 lost 349
  swap p= 0.5  types 2233 distinct 1749 lost 484
  swap p= 0.75 types 2233 distinct 1869 lost 364
  swap p= 0.0 / 1.0 lost 0 ;  nonce lost 0 at every p
  ```
  (No test pins them — see F5.)
* **The three `word-classes.json` copies really are byte-identical**, and really are what both
  stacks read: `shasum -a 256` → `7bf140446a27f4079baf3c853198789032b9af7d8018f83ce5fbb83e63a8f3b0`
  for all three; `arch/word_classes.py:43` reads the package copy through
  `importlib.resources`, `staticClient/wordClasses.ts:28` imports the frontend copy.
* **The generator is reproducible.** Running `scripts/export_word_classes.mjs` with `ROOT`
  redirected into the scratchpad reproduced the committed file **byte for byte**
  (same sha256, `11137 bytes`, `Unicode 16.0 via node v22.16.0`). A regeneration on an older
  Node would be caught: `word-alphabet-cases.json` carries U+0890 (Cf from Unicode 14),
  U+2E5D (Pd from 14) and U+10D6E (Pd from 16, astral) as *must-refuse* cases in both suites,
  so a 13.0/15.0 table fails them. (`unicodeVersion` itself is unpinned —
  `expect(PINNED_UNICODE_VERSION).toBe(table.unicodeVersion)` is a tautology — but a *newer*
  pin moves both stacks together, so it is not the divergence class.)
* **The Unicode skew is closed, measured across the whole code space.** Table classes:
  joiner 227, letter 141 028, mark 2 501, identical when expanded independently in each
  language. Against Python 3.10's own tables: 0 characters Python calls a joiner that the
  table does not, 31 the table adds; 0 letters Python knows that the table does not, 9 787 the
  table adds. Against Node 22's `\p{…}`: 0 node-only joiners, 0 node-only letters, 0
  table-only letters.
* **Both stacks answer identically on 4 719 hostile inputs.** All 227 joiners × 4 shapes
  (`don<J>t`, `don<J><J>t`, `don<J> t`, `<J>don`), 1 500 sampled letters × 2, 400 sampled
  marks × 2, four astral letters, and nine ASCII-alphabet oddities, `fragmented_words` vs
  `fragmentedWords`: **`cases 4719 disagreements 0`**.
* **`Pc` is really in the class and the whole class really refuses.** Of the 227 joiners,
  exactly two are not refused in `don<J>t` — `U+0027` and `U+002D`, the two `WORD_RE` itself
  accepts, which is correct. `don‿t`, `don_t`, `don＿t`, `don﹍t`, `don࢐t` (U+0890),
  `don⹝t` (U+2E5D) and `don𐵮t` (U+10D6E) all refuse in both stacks.
* **The `--` false positive is really gone, and nothing else in the shipped corpora regressed.**
  `fragmented_words` over the whole of `real-mother-goose.txt` flags 7 runs / 5 distinct, all
  curly apostrophes (`Foundation’s`, `one—the`, `away—you`, `Gutenberg’s`, `state’s`); over
  `alice-in-wonderland.txt`, 778 / 224, all curly apostrophes (`I’m`, `don’t`, `I’ve` …). The
  six default arch passages flag **nothing**: `[[], [], [], [], [], []]`.
* **The teeth commits changed no production code.** `git show --stat`: `c603c9a` →
  `code/frontend/tests/unit/navGuard.test.ts | 157 ++++--` only; `55c7b5a` →
  `code/backend/tests/integration/test_geo_derived_vocab.py | 90 ++++` only; `9f0bc5e` →
  `notes/agent-reports/fix-007-teeth.md | 241 ++++` only. The six deleted lines in `c603c9a`
  are a comment and the old regex, replaced by a **strictly stronger** one that additionally
  requires the `$effect(() => {` wrapper. No `expect(` was removed or weakened.
* **SC-703 and the TS↔Python differential suite, re-derived.**
  `npx vitest run tests/unit/{vacancy,vacancyGolden,staticVacancy,archVacancy}.test.ts` →
  `Test Files 4 passed (4) · Tests 237 passed (237)`, and the §10 line prints, at seeds 0 and 7:
  `stemsTotal=1676`, `tokensVacated(p=1.00)=8125`, `domainTypes=2233/1940`, `bijective=true`,
  `imageSize=2233`. Identical to the recorded baseline.

---

## (e) Mutation testing — 12 mutants, 9 killed, 3 survived

Every mutant applied in place, restored from `cp`, re-verified by `shasum -a 1`.

**Re-applied from round 5's own list (all three kills CONFIRMED):**

| mutant | result |
|-|-|
| `QUANTIZATION_TERM` → `"…retained bound — re-measured on the shipped swap"` | `Tests 4 failed \| 811 passed \| 1 skipped (816)` — killed |
| `VacancyPanel.svelte:557` `$effect(() => {…})` → bare `{…}` | `navGuard.test.ts`: `Tests 3 failed \| 14 passed (17)` — killed |
| `lex/vacancy.py:659` `< 2` → `< 2 or suffix == "s"` | `3 failed, 90 passed`, incl. `test_the_merge_boundary_is_class_SIZE_and_nothing_else` — killed |

**Fresh mutants on things nobody had mutated:**

| # | mutation | result |
|-|-|-|
| M4 | `VACANCY_FP32_REFERENCE.unknownForm` 0.2872 → 0.4872 | **SURVIVED** `815 passed` → **F3** |
| M5 | exchange the fp32 / pre-rewrite interpolation slots in `VACANCY_UNKNOWN_FORM_REFUSAL` | **SURVIVED** `815 passed` → **F4** |
| M6 | `VACANCY_MIN_POOLED_PRESERVED` 700 → 7 | **SURVIVED** `815 passed` (e2e-only gate) → **F6** |
| M7 | `word_classes.py` loader: drop the `named` merge | killed — `3 failed, 4 passed` |
| M8 | `word_classes.py` `_in_class`: drop singleton ranges | killed — `6 failed, 1 passed` |
| M9 | `wordClasses.ts` `classRanges`: drop the `named` merge | killed — `7 failed, 80 passed` |
| M10 | `byteSpans.ts`: remove the ASCII escape hatch | killed — `wordClasses.test.ts 2 failed, 5 passed` |
| M11 | `is_vacatable` → drop test 1 (`word.lower() not in keep`) | killed — `7 failed, 86 passed` |
| M12 | merge boundary `< 2` → `< 3` | killed — `2 failed, 91 passed`, incl. both boundary tests |

The loader, `is_vacatable` and the merge boundary are well pinned. The three survivors are all
in the same place: **the numbers in the prose, and the assertions that claim to pin them.**

## Repo hygiene

Nothing in the repository was modified by me. All probes (`rederive.py`, `py_joiners.py`,
`ts_joiners.ts`, `battery.{py,ts}`, `probe_hatch.py`, `probe_score.py`, `corpus_scan.py`,
`slots*.py`, `gen.mjs`) and every backup lived in the session scratchpad. Ports 8000/5173/4173
were never bound; no dev stack was started (the two model runs were in-process, with
`LLM_GEOMETRY_CACHE_DIR` pointed at the scratchpad).
