# Fix 007 — Architecture Explorer, findings F2–F10

Date: 2026-08-04. Worktree branch off `main` @ `0ed5365`. Fix commit(s) below.
Source report: `notes/agent-reports/redteam-007-arch.md`.

**F1 was NOT in scope and was not touched.** `lex/vacancy.py`, `lexEngine/vacancy.ts`,
`specs/007-vacancy-transform-field/architecture.md` and the vacancy golden fixtures are
unmodified — `git diff --stat` confirms it.

Verification environment: backend run with the main checkout's `.venv` (Python 3.10.12)
and `PYTHONPATH` pointed at this worktree's `src`; frontend `npm install` in the worktree,
with `code/frontend/public/static-data` symlinked to the main checkout's exported assets
(that directory is git-ignored and generated, so the worktree had none). The symlink was
removed before committing.

## Totals

- Backend: `473 passed` (`pytest -q`, all suites, real models).
- Frontend: `472 passed | 1 skipped` (`npx vitest run`), `svelte-check` **0 errors**,
  `ruff check` clean, `black --check` clean.
- **e2e was NOT run** (explicit instruction). Three e2e assertions were *written* and are
  unverified — flagged per finding below.

---

## F2 (high) — ordinary short passages 500'd with an opaque internal error

**Fixed.**

**Before** (verbatim, `vacancy_score("gpt2", [t])` then `jsonable_6sig`):

```
'I like cats and dogs.' ENCODE-FAIL ValueError non-finite value in response payload
'the dog'               RAISE ComputeError no tokens to average — the passage has no scored positions
'The dog barked.'       RAISE ComputeError no tokens to average — the passage has no scored positions
'Hello world'           RAISE ComputeError passage 0 has no word that survives the transform…
'the the'               ENCODE-FAIL ValueError non-finite value in response payload
```

**After** — every one is `InvalidParamError`, `http_status == 400`:

```
'I like cats and dogs.' -> InvalidParamError 400 this request pooled 1 paired preserved token(s); a
                           difference and its standard error need at least 2 (the sample variance
                           divides by n − 1, so at n = 1 the standard error does not exist)…
'the dog'               -> InvalidParamError 400 passage 0 has 1 preserved word(s) but no scored token
                           belonging to one in the 'english' variant: position 0 has no prediction…
'Hello world'           -> InvalidParamError 400 passage 0 has no word that survives the transform…
```

Changes: `MIN_PAIRED_PRESERVED = 2` + a pooled-pairs check, a per-variant
"no scored preserved token" check, and `ComputeError` → `InvalidParamError` for
"no word survives". `vacancy_score` was also restructured so **all** parameter failures
happen before `load_model` (transform first, weights second).

**Two-stack convergence (asked for explicitly).** The two thresholds are deliberately
different and both now say so in their own docstrings:
`MIN_PAIRED_PRESERVED = 2` (Python) is a **sampling** floor — a sample variance divides
by `n − 1`; `VACANCY_MIN_POOLED_PRESERVED = 700` (`staticClient/arch.ts`) is a
**quantization** floor, and a float32 stack has no quantization error to bound. Both
stacks now refuse with a typed error naming the count they got and the count they need.
The 700 gate subsumes the sampling floor, which is why `pairedDifference`'s `se = NaN`
branch is unreachable in the static build — noted at the constant.

**Test:** `code/backend/tests/integration/test_arch_vacancy_score.py::`
`test_short_passages_get_a_typed_400_naming_the_cause` (5 params — the red team's exact
five inputs) and `test_a_result_that_is_returned_always_encodes`.
**Before:** 5 failed. **After:** all pass.

---

## F3 (high) — a negative "cost of unknown form" presented as a resolved result

**Fixed.**

Root cause confirmed: `resolved` tested `|nats| > 2·se`, which is the correct test for
noise and the wrong one for direction, so a large negative estimate was *promoted* into
the conclusion branch.

Decision logic moved out of the component into `code/frontend/src/viz/arch/vacancyVerdict.ts`
(pure, tested). `verdictKind` now returns `refused | identity | unresolved | backwards |
conclusion`; `formSharePercent` returns `null` unless both numbers are positive;
`upperBoundLabel` refuses to call a negative value an upper bound; `formCostsLessThanContent`
gates the closing sentence on the ordering it asserts. The panel gained a dedicated
"backwards" paragraph that draws no conclusion.

**Before** (same tests, run against a transcription of the pre-fix panel logic):

```
× classifies the observed -0.355 ± 0.136 as backwards  → expected 'conclusion' to be 'backwards'
× calls p = 0 an identity                              → expected 'unresolved' to be 'identity'
× does not print a negative percentage as a share      → expected '-69%' to be null
× does not call a negative value an upper bound        → expected 'upper bound — see below' to match
                                                          /cannot have a negative upper bound/
```

**Test:** `code/frontend/tests/unit/archVacancyPanel.test.ts`, describe
`F3 — a negative cost is never promoted to the conclusion` (6 cases; the payload is the
red team's `-0.3548 ± 0.1359` over 22 pairs with `wrong_content 0.871`, `total 0.5162`).
**After:** all pass.

---

## F4 (high, FR-720a) — the live site printed a fabricated error bar

**Fixed.** Treated first, as instructed.

`VacancyScorePanel.svelte`'s secondary rows now render **every** term through
`secondaryLine()`, which appends the measured `quantizationUncertaintyNats` whenever the
stack attached one, and appends nothing when it did not.

**Before:** `secondaryLine` (pre-fix transcription) produced
`nll(nonce) − nll(english) = 0.879 ± 0.074` — no quantization term.
**After:** `nll(nonce) − nll(english) = 0.879 ± 0.074 (sampling) · ± 0.2 (quantization, measured)`.

The test also pins the arithmetic that makes it a fabrication rather than a rounding:
`0.879 + 0.0741 < 0.9892` (the stated interval excludes the float32 truth) and
`0.879 + 0.2 > 0.9892` (the dropped term is the one that contains it).

**Test:** `code/frontend/tests/unit/archVacancyPanel.test.ts`, describe
`F4 — every measured term of an error bar is rendered` (3 cases).
**Before:** 1 failed with `expected 'nll(nonce) − nll(english) = 0.879 ± 0…' to contain
'± 0.074 (sampling)'`. **After:** all pass.
**Also added (NOT RUN):** `tests/e2e/static.spec.ts` now asserts
`arch-vac-total-err` contains `0.2 (quantization, measured)` on the deployed build.

---

## F5 (medium) — Chat tooltips read the live slider

**Fixed, with one honest gap.**

`tokenTip` moved to `viz/arch/archShared.ts` and now takes `temperature` as a **required
argument** — the signature makes the bug unrepresentable. `ArchChat.svelte` captures
`forTemperature` alongside `result`, renders `drawn at T=…` in the reply meta, and shows a
`arch-temp-stale` warning once the slider no longer matches the reply on screen. The
slider gained `data-testid="arch-temperature"`.

**Test:** `code/frontend/tests/unit/archVacancyPanel.test.ts`, describe
`F5 — the reply tooltip names the temperature the reply was drawn at` (3 cases).
**Before:** 3 failed (`TypeError: tokenTip is not a function` — the function did not exist
as a testable unit; the pre-fix version read `$archTemperature` inside the component).
**After:** all pass.

**Gap, stated plainly:** the unit tests prove the *function* names the temperature it is
given; they do **not** prove the *component* hands it the captured one rather than the
store. The test that proves the wiring is
`tests/e2e/explorer.spec.ts::"a reply's tooltip keeps the temperature it was drawn at"`,
which I wrote and **did not run** (running e2e was excluded from my charter). I could not
find a non-mock seam to drive `ArchChat` in jsdom — `client` is a module singleton, and
substituting it would be exactly the mock this repo forbids. Treat that e2e as unverified
until someone runs it.

---

## F6 (medium) — Latin-diacritic words

**Fixed — and the finding was worse than reported.**

The red team saw a 500. There is also a **silent** symptom, which I measured:

```
words:  ['The','dog','and','the','cat','sat','on','the','mat','with','a','caf','and','a','r','sum',…]
swap:   'The cat and the warm best on the think with a washé and a réwoodé on the yellow by the sat.'
result: OK [('wrong_content', 0.718…), ('unknown_form', -0.502…), ('total', 0.215…)]
```

`café` is the word `caf` to `WORD_RE`, so the transform rewrote a fragment and the endpoint
**returned** the score. That is the plausible-wrong-answer class, not a crash.

Both stacks now refuse up front, naming the word:
`fragmented_words` / `check_word_alphabet` (Python) and `fragmentedWords` /
`checkWordAlphabet` (`staticClient/byteSpans.ts`), typed `InvalidParamError` (400). The
detector flags a Unicode-letter run only when `WORD_RE` matches *part* of it — so
`don't`, `good-bye`, emoji and CJK are untouched (CJK/emoji contain no `WORD_RE` match at
all, are never vacated, and are byte-identical in all three variants). The shipped corpus
and all six default passages were checked: **zero** fragmented words, so the default run is
unaffected.

I did **not** widen `WORD_RE`: that is the shared transform and its normative contract,
which another agent owns. The refusal says so in its own message.

The false docstring is corrected in both stacks (`preserved_token_indices` at
`vacancy_score.py`, `preservedTokenIndices` at `byteSpans.ts`): the "cannot happen"
claim now states the precise condition under which it holds, and names the case where it
did not.

**Tests:** `tests/unit/test_arch_vacancy_align.py::test_a_diacritic_word_is_refused_by_name_rather_than_mangled`,
`tests/integration/test_arch_vacancy_score.py::test_a_diacritic_passage_is_refused_before_any_number_is_computed`,
and `tests/unit/archVacancy.test.ts` describe `the word alphabet (§8.2) — red team F6`.
**Before:** Python integration `DID NOT RAISE InvalidParamError`; TS
`TypeError: fragmentedWords is not a function` ×2 and `expected [Function] to throw error
matching /ASCII letters only/`. **After:** all pass.

---

## F7 (medium) — `p ∈ (0,1)` exposed, `p = 0` reported as a measurement

**Fixed, both halves.**

*Intermediate p.* Both stacks now check the map that was actually built
(`vmap.injective_at_every_p` / `vmap.injectiveAtEveryP`) and raise `InvalidParamError`
citing §5.2a — the same rule `lex/vacancy.py:map_vocab_words` applies, which the arch path
never called. Verified: `p ∈ {0.05, 0.5, 0.95}` refuse; `p ∈ {0, 1}` do not.

*p = 0.* The Lexicon Lab permits `p = 0`, so I kept it rather than refusing — it is a real
null control and removing it would remove the reader's ability to see the instrument read
zero. What was wrong was the *presentation*, so every difference now carries
`identity: true` + an `identityNote` from the stack that computed it, and the panel has a
dedicated branch: **"exactly nothing … but by construction, not by measurement"** instead
of `0.000 ± 0.000 nats (sampling, 20 paired tokens)`.

*Panel.* The `p` input stays (mirroring the Lexicon Lab, which shows the refusal rather
than moving the slider for you), with a new always-visible `arch-vac-pnote` explaining
what `p` is and why only 0 and 1 are defined, and "Set p = 1" / "Set p = 0" actions on the
refusal.

**Note on an existing test I changed:** `test_p_zero_is_the_identity_and_costs_nothing`
previously asserted only that the differences are 0. It still asserts that; I added
assertions that the identity is *labelled*. That is a stricter contract, not a weakened
one — but it is a test I edited, and you should look at it.

**Tests:** `tests/unit/test_arch_vacancy_align.py::test_intermediate_p_is_refused_exactly_as_the_lexicon_lab_refuses_it`,
`tests/integration/test_arch_vacancy_score.py::test_intermediate_p_is_refused_end_to_end`
and `::test_p_zero_is_the_identity_and_costs_nothing`,
`tests/unit/archVacancy.test.ts::"refuses intermediate p, exactly as the Lexicon Lab does (§5.2a)"`.
**Before:** `DID NOT RAISE InvalidParamError`; TS `expected [Function] to throw an error`;
`test_p_zero…` failed on the missing `identity` flag. **After:** all pass.
**Also added (NOT RUN):** `tests/e2e/archVacancy.spec.ts::"refuses intermediate p…"`.

---

## F8 (low) — an exact 1-D tile labelled `strided_mean`

**Fixed, and one trap avoided.** Carrying `tile.method` through naively would have made the
inspector caption a uint8 strip **"exact values"** — swapping one false statement for
another. So the static client carries `tile.downsampled` / `tile.method` *and* sets a new
static-only `quantized: "uint8"`, and `ArchInspector` keys its overview branch off
`quantized` (not `downsampled`) and captions
`overview (whole tensor, strided mean, 8-bit)` or `… full resolution, 8-bit`.

**Test:** `tests/unit/staticClient.test.ts::"does not describe an EXACT 1-D tile as a
strided mean (red team F8)"`, driven against the real exported `gpt2` tiles.
**Before:** `expected 'strided_mean' to be 'exact'`. **After:** passes.
`tests/e2e/static.spec.ts`'s caption expectation was updated to the new string (**NOT RUN**).

---

## F9 (low) — `nChars` code points vs UTF-16 units

**Fixed.** Unicode **code points** in both stacks. `staticClient/byteSpans.ts` gained
`nCharsOf(text) = [...text].length`, used by `transformersRuntime.ts` in place of
`text.length`; Python's `len(str)` already means that, and `_stats` now says so. Byte
coordinates remain the unit for *attribution* — the change is to a reporting field only.

**Test:** `tests/unit/archVacancy.test.ts::"counts Unicode code points, never UTF-16 units"`,
pinning the red team's probe at `77` (`.length`) vs `75` (`nCharsOf`).
**Before:** `TypeError: nCharsOf is not a function`. **After:** passes.

---

## F10 (low) — the size gate quoted a ceiling it did not apply

**Fixed.** `effective_ceiling_for(source)` and `too_large_error(mid, total, source)` are
now separate, testable functions; the message quotes the ceiling that was **applied** and,
on the config-estimate path, explains the 80 % margin. `check_model_size` returns
`effective_ceiling`, and the detail dict carries it. `ArchitectureExplorer.svelte`'s
"capped at 1.5B parameters" prose now also states the 1.2B config-estimate ceiling
(`docs.spec.ts`'s pinned "1.5B parameters" string is untouched).

Like the red team, **I did not trigger this against a live 1.2–1.5 B model with no
safetensors index** — I have not found one worth downloading. The test drives the real
error-construction function with the real constants:
`tests/unit/test_arch_gate.py::test_rejection_quotes_the_ceiling_it_actually_applied`.
**Before:** `ImportError: cannot import name 'effective_ceiling_for'` (the pre-fix module
had no seam; the pre-fix message interpolated `ARCH_MAX_PARAMS` unconditionally — quoted
in the red team report). **After:** passes.

---

## Documentation updated in the same commit

- `viz/info/InfoTab.svelte` "Known limits": the swap-injectivity bullet now says the
  Architecture Explorer's decomposition is refused at intermediate `p` too, and a new
  bullet states the ASCII word alphabet and what it does to `café` / `naïvely`.
- `ArchitectureExplorer.svelte` diagram Explain: the 1.2 B config-estimate ceiling.
- `VacancyScorePanel.svelte`: the new `p` note, the backwards/identity verdict prose.
- Docstrings: `vacancy_score.py` module header (word alphabet), `preserved_token_indices`
  (both stacks), `_stats` (`nChars` unit), `gate.py` module header + both new functions,
  `VACANCY_MIN_POOLED_PRESERVED` and `MIN_PAIRED_PRESERVED` (why they differ).
- `specs/007-vacancy-transform-field/ui.md` §2.1 already lists `p` as a control and is
  still accurate (the control remains; it refuses). No spec edit was needed, and
  `architecture.md` was not touched.

## What I could NOT verify

1. **All three new/edited e2e assertions are unrun** (F4 static row, F5 temperature
   staleness, F7 refusal), plus the two e2e strings I updated to match new captions
   (`static.spec.ts` inspector caption). Running e2e was outside my charter.
2. **F5's component wiring** has no non-mock unit-level test — see the gap under F5.
3. **F10 was never triggered against a real oversized config-estimate model**, exactly as
   the red team said. The claim rests on the extracted function plus its constants.
4. I did not re-run the WebGPU project (no adapter here), and did not exercise Qwen /
   SmolLM2 in-browser.
