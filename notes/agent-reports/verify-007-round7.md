# Verify 007 — round 7 (agent "VERIFICATION AGENT V8")

Date: 2026-08-04. Branch `main` @ `34027ae`. `npm run test:e2e` was **not** run; no MCP
browser tool was used; no dev stack was started. Every mutation was applied in place,
restored from a `cp` backup in the session scratchpad and re-verified with `shasum -a 1`
(never `git checkout`). Tree at the end: `git status --short` carries only
`M notes/2026-08-04-redteam-campaign.md`, which is **not mine** — it appeared mid-session
(a concurrent sibling agent's note edit, adding a "Rounds 5–8" pointer); the tree was clean
at t0 and every file I touched is byte-identical to `HEAD`.

| # | charter item | verdict |
|-|-|-|
| 1 | the word-alphabet rule, third attempt | **REFUTED** — the rule is sound over the joiner/apostrophe/hyphen space (0 skew in 8 860 cross-stack cases) but the alphabet has never covered **digits**: `covid19` scores 200 and vacates to `window19` |
| 2 | the fp32 pin, second attempt | **PARTIAL** — the three arrows are real and the parser is not vacuous, but the tolerance is 10× the quoted precision, and **every other** measurement constant on the panel is still the round-5 facade |
| 3 | inherited-key sweep, sixth attempt | **PARTIAL** — round 7's five kills confirmed; 6 more instances of the shape remain (none exploitable today) |
| 4 | `BUNDLE_VERSION` 2→3 | **REFUTED** — a one-field downgrade (`version: 3` → `2`) accepts a substituted vocabulary in **both stacks**; the v3 binding is opt-out |
| 5 | number parity | **PARTIAL** — true for `/api/geo/*` + `/api/lex/*` JSON bodies only; `routes_arch.py` never calls `params.py` and Pydantic coerces first |
| 6 | teeth | **PARTIAL** — 9 of 10 re-applied/fresh mutants killed; the one survivor is an equivalent mutant (proved), but the NEW parity test has no teeth against the shared rule it tests |
| 7 | SC-703 + differential suite + §10 | **VERIFIED**, re-derived |

Suites, tree restored: backend `pytest -q` → `680 passed, 1 warning in 187.60s` (as
4 failed + 676 passed under a live mutation; unmutated baseline is 680). Frontend
`npx vitest run` → `Test Files 35 passed (35) · Tests 851 passed | 1 skipped (852)`.

---

## (a) Defects surviving from earlier rounds

### F1. The "measured" constants are still a facade — round 7 pinned exactly one of them
**Severity:** high
**Where:** `code/frontend/src/lib/staticClient/arch.ts:274-303`
(`VACANCY_PRE_REWRITE_Q8`, `VACANCY_ONNX_TORCH_AGREEMENT_NATS`)
**Reproduce:** apply this diff and run the whole frontend suite plus the backend arch suites:

```
-  unknownForm: { fp32: 0.2726, q8: 0.235, errorPercent: 14 },
+  unknownForm: { fp32: 0.5726, q8: 0.135, errorPercent: 74 },
-    unknownFormRange: "0.06–0.21",
+    unknownFormRange: "0.96–0.91",
-    perPassageWorstNats: 0.65,
+    perPassageWorstNats: 0.95,
-    unknownFormWorstPassageNats: 0.28,
+    unknownFormWorstPassageNats: 0.88,
-export const VACANCY_ONNX_TORCH_AGREEMENT_NATS = 5.3e-4;
+export const VACANCY_ONNX_TORCH_AGREEMENT_NATS = 5.3e-2;
```

**Observed** (verbatim):

```
 Test Files  35 passed (35)
      Tests  851 passed | 1 skipped (852)
```

and, on the same tree, `pytest tests/integration/test_arch_vacancy_score.py
tests/unit/test_arch_word_classes.py -q` → `27 passed in 10.02s`.

Five fabricated user-facing measurements ship with every test green, including
`VACANCY_ONNX_TORCH_AGREEMENT_NATS` — the number that justifies "use the full stack" in
*every refusal on this panel* — moved by two orders of magnitude.
**Expected:** round 7's own report says this class was fixed: *"the pin asserted the
constant against itself"* and *"three real links, no number typed twice"*. That was done
for `VACANCY_FP32_REFERENCE` only. The mechanism to fix the rest already exists in this
build and was applied once: `archStaticFacts.test.ts` reads `VACANCY_MIN_POOLED_PRESERVED`
out of `architecture.md` §8.3a. Every one of the constants above is stated in
`specs/007-vacancy-transform-field/architecture.md` (lines 894, 899, 943, 971, 973) and
none is read from there.
The `absoluteShiftNats` pair is the sole exception — `archVacancy.test.ts:554`
`expect(...).toMatch(/−0\.19|\+0\.40/)` catches it (`1 failed | 850 passed`), by a literal
typed twice, which is the pattern round 7 set out to remove.
**Would it have thrown?** No. The values shipped today are correct (they match
`architecture.md`); the defect is that nothing holds them there.

### F2. The fp32 pin's tolerance is 10× the precision the sentences quote
**Severity:** low
**Where:** `code/backend/tests/integration/test_arch_vacancy_score.py:364`
(`pytest.approx(..., abs=5e-4)`) and `tests/unit/archVacancy.test.ts:391-393`
(`toBeCloseTo(..., 3)`)
**Reproduce:** `unknownForm: 0.2872` → `0.2877` in `staticClient/arch.ts`, then
`pytest tests/integration/test_arch_vacancy_score.py::test_the_fp32_arm_quoted_in_the_static_client -q`
and `npx vitest run tests/unit/archVacancy.test.ts tests/unit/archStaticFacts.test.ts`.
**Observed:** `1 passed in 6.83s` (backend, real gpt2) and `Tests 39 passed (39)`.
The record says `0.28721050610900456`; the browser then quotes **0.2877** as "the
configuration that ships".
**Expected:** the constants are quoted to four decimals, so the pin should be `5e-5`.
At `5e-4` the last quoted digit is unpinned in both directions.
**Would it have thrown?** No.

### F3. Six more inherited-key lookups (the shape, not the list)
**Severity:** medium (1), low (5)
**Where / claim** — traced to their key sources, none exploitable with today's data:

1. `code/frontend/src/viz/arch/ArchInspector.svelte:217` —
   `<p class="explain">{KIND_EXPLAINER[node.kind]}</p>`. `node.kind` is a raw string
   from parsed graph JSON (`/api/arch/graph`, or `arch/<slug>/graph.json` read by
   `staticClient/arch.ts:762`). No `Object.hasOwn`, no fallback. A graph naming a kind
   `constructor` prints `function Object() { [native code] }` into the inspector.
   **medium** — the only unguarded `Record` lookup left that is keyed by remote data.
2. `code/frontend/src/lib/geoEngine/model.ts:230` — `if (key in config && config[key] !== expected)`,
   with `config` from `JSON.parse` of a checkpoint. `in` walks `Object.prototype`; this is a
   validation-**skip** shape (the frozen-config check silently not applied). Same file:196
   `if (sd === null && "embedding" in obj)`. **medium-low.**
3. `code/frontend/src/lib/PipelineDiagram.svelte:48` — `if (group in expandOverrides) return !expandOverrides[group];`
   over a `$state<Record<string,boolean>>({})` accumulator keyed by `node.group`. **low.**
4. `code/frontend/src/lib/geoEngine/tokenizer.ts:119-120` — `sp[token] !== undefined` on the
   `specials` object of an uploaded file, bare index unlike its sibling loaders. **low.**
5. `code/frontend/src/lib/staticClient/geo.ts:77,86-88` — `JSON.parse(sessionStorage…)` used as a
   `{}` accumulator (`all[token] = …`); keys are content hashes, so unexploitable, but it is the
   one persistence path not on `Object.create(null)`. **low.**
6. `code/frontend/src/viz/lex/LexWeightLab.svelte:118,130,175,180,199,203,216,218` — plain-object
   `WeightSet`/`shapes` indexed by a `$state` name. **low.**

Python side: **no findings**. `geo/model.py:227`'s `getattr` is gated by
`name not in EDITABLE_MATRICES` plus a layer-range check; every other `getattr` reads a
literal attribute name off an HF config.
**Would it have thrown?** (1) and (3) render a wrong string; (2) and (4) skip a check. None throws.

### F4. `routes_arch.py` never reaches `params.py` — Pydantic coerces first
**Severity:** high
**Where:** `code/backend/src/llm_geometry/api/routes_arch.py` (`GenerateBody`,
`VacancyScoreBody:62-80` — `p: float = 1.0`, `seed: int = 0`, `temperature`,
`max_new_tokens`), versus `llm_geometry/api/params.py`
**Reproduce:** FastAPI `TestClient`, real SmolLM2-135M weights.
**Observed:**
- `POST /api/arch/generate {"temperature": NaN}` → `500 InternalError: probability tensor
  contains either 'inf', 'nan' or element < 0` — an **untyped 500**; `generate.py:92` guards
  `temperature < 0`, which `NaN` passes.
- `{"temperature": Infinity}` → `200` and real text (uniform sampling).
- `{"temperature": true}` → `200` (coerced to 1.0); `" 7 "`, `"+7"`, `"1_000"`, `"1e3"` → all `200`.
- `/api/arch/vacancy-score`: `p="1_000"` → `1000.0`; `p=NaN` → accepted; `seed=" 7 "` → `7`.
- `GET /api/geo/vector_field?temperature=NaN` → `500 InternalError: non-finite value in
  response payload`; `Infinity` → `200`.
**Expected:** `params.py`'s own docstring — *"``NaN`` is the worst of them because nothing
throws"* — states the rule these endpoints do not apply. Round 7's report says
"`routes_geo` and `routes_lex` both delegate"; it is silent on `routes_arch`, and the new
contract test only exercises `POST /api/geo/finetune` and `POST /api/lex/train`.
**Would it have thrown?** `NaN` throws (untyped 500); `Infinity`/`true`/`" 7 "` do not.

### F5. Static build and backend still read the same numeric parameter differently
**Severity:** medium
**Where:** `code/frontend/src/lib/staticClient/transformersRuntime.ts:227`
(`Math.trunc(body.max_new_tokens ?? 64)`), `staticClient/arch.ts:1038-1039`
(`p = body.p ?? 1.0`, `const seed = body.seed ?? 0`), `staticClient/geo.ts:348` →
`hfDatasets.ts:158` (`Math.max(1, Math.trunc(...))`)
**Observed:** `max_new_tokens: 7.5` → static runs 7 tokens, backend refuses ("fractional
part"); `"0x10"` → static runs 16, backend refuses; `"1_000"` → static NaN-refuses, backend
runs 1000. `arch.ts` `seed: "٧" | 7.5 | {}` all pass unvalidated; `p: "0.5" | " 0.5 " |
true | [] | [0.5]` are accepted as non-numbers. `/api/geo/train_scratch` `max_samples`:
backend refuses `"7"`, `7.5`, `true`; static reads `"7"`→7, `"0x10"`→16, `7.5`→7,
`"٧"`→NaN→silently 0 rows.
**Expected:** `Math.trunc(body.x ?? default)` is the exact pattern `lib/params.ts` was
written to kill; these call sites were not migrated.
**Would it have thrown?** No — the two builds return different numbers from one request.

### F6. Multipart `POST /api/geo/finetune` can never set `lr`
**Severity:** low (latent)
**Where:** `code/backend/src/llm_geometry/api/routes_geo.py:337` — the form value is
collected and then passed to `_as_float`, which refuses strings; there is a `_form_int`
but no `_form_float`.
**Observed:** form `lr='0.05'` → `400 InvalidParamError: lr must be a number, got '0.05'`.
Not user-visible today: `FinetunePanel.svelte:56` sends only `steps`/`base`.
**Would it have thrown?** No — a typed 400 for a legal request.

---

## (b) NEW defects round 7 introduced or left open in the code it touched

### F7. `version: 2` is an opt-out: a downgraded file loads with any vocabulary — the substitution catastrophe, restored
**Severity:** critical
**Where:** `code/backend/src/llm_geometry/geo/bundle.py:226-238` and its mirror
`code/frontend/src/lib/geoEngine/index.ts:892-905`
**Reproduce:** build one payload — real-shaped weights, a word list that is the shipped
list **permuted** (so every id means a different word), an honest `vocab_sha256` over it,
and `weights_token = weights_token(ws, None)` — and import it twice, changing only
`version`:

```python
for label, ver in (("v3 declared", BUNDLE_VERSION), ("v2 declared (downgrade)", LEGACY_WEIGHTS_ONLY_VERSION)):
    r = import_bundle(bundle(ver, weights_token(ws, None)), store=CacheStore(tmp/label.split()[0]))
```

**Observed** (verbatim):

```
v3 declared: REFUSED -> this model file is corrupt: its weights and vocabulary hash to 5d07ba67532c13542325ac27c5a5efaa but it declare
v2 declared (downgrade): ACCEPTED  owns_vocab=True
    file words[:6] = ['vanished', 'stick', 'left', 'grin', 'find', 'trembling']
    canonical[:6] = [',', '"', 'the', '.', 'and', 'to']
```

**Expected:** the identity change (`0d23123`) exists precisely to stop this, and round 7's
own test `test_a_weights_only_token_under_the_current_version_is_still_refused` asserts it
— for files that declare version 3. `weights_token(ws, None)` is a hash of data the file
already carries, so anyone holding the file can compute it; changing one integer field
turns the refusal into an acceptance. Before round 7, `SUPPORTED_VERSIONS` was `(2,)` with
`BUNDLE_VERSION = 2`, so this payload hit `declared != actual` and was refused. The
migration **re-opened** the defect the campaign's own brief names as shipped example
`d6e9d5b` ("the digests *verified*, because the writer computed them over the substituted
list"). The `owns_vocab=True` flag means `GeometryLab.verifyVocab`'s probe then agrees, so
the tab reports the vocabulary **verified**.
The code comment defends reading v2 with *"the binding a v3 token gives is absent from
EVERY v2 file"* — true, and exactly why "declares v2" cannot be trusted as evidence that a
file is old. A genuinely-old file has a word list that either **is** the shipped one (takes
no part in either hash — loads under v3 unchanged) or is its own; only the second case
needs the migration, and it is indistinguishable from the attack.
**Would it have thrown?** No — 200, `owns_vocab=True`, every label wrong.

### F8. The word alphabet has never seen a digit: `covid19` scores 200 and vacates to `window19`
**Severity:** high
**Where:** `code/backend/src/llm_geometry/arch/vacancy_score.py:176-214` (`wordlike_runs`
starts a run only at `is_letter`; digits are neither letters nor joiners) and its mirror
`code/frontend/src/lib/staticClient/byteSpans.ts:172-176` (`LETTER_RUN`/`WORDLIKE_RE`)
**Reproduce:**
```python
variant_texts("I read the covid19 report and the level3 alarm and the top10 list and a house2house call.",
              p=1.0, seed=0, match_prosody=True, keep=frozenset())
```
**Observed** (verbatim):
```
english  :: I read the covid19 report and the level3 alarm and the top10 list and a house2house call.
swap     :: I wood the window19 yellow and the covid3 table and the read10 thing and a use2use watch.
nonce    :: I hilk the kranking19 norbleid and the cheedleous3 teanle and the hor10 sorp and a keaf2keaf scoord.
```
`fragmented_words(...)` → `[]` in Python and `fragmentedWords(...)` → `[]` in TypeScript;
`check_word_alphabet(passage, 0)` returns without raising. End to end on the real model:
```
SCORED. differences:
   wrong_content 0.3818 nPairs 17
   unknown_form 0.0519 nPairs 17
   total 0.4337 nPairs 17
```
**Expected:** refusal. This is character-for-character the defect the refusal exists to
prevent and the one its own message names: *"'a café' vacates to 'a washé'"*. `covid19` →
`window19` is `café` → `washé` with a digit instead of an accent; `house2house` →
`use2use` is `don''t` → `little''t` with a digit instead of an apostrophe — one wordlike
thing a reader calls one word, `WORD_RE` matching a part, the part rewritten and a
character it cannot see surviving between two halves. Three consecutive attempts at this
rule (joiners → alphabet test → whole-match test) all reasoned about **letters and
joiners** and never about the third character class a written word contains.
`specs/007-vacancy-transform-field/word-alphabet-cases.json` has 30 cases and **not one
digit**, so the fixture again pins what the code does rather than what the requirement
demands. Users supply this text: `VacancyScoreBody` (`routes_arch.py:74-80`) takes
`passage`/`passages` freely and the panel lets a reader edit an excerpt.
**Would it have thrown?** No — 200 with three numbers.

### F9. `said--'tis` is falsely refused, and the refusal's own advice cannot be complied with
**Severity:** medium
**Where:** `arch/vacancy_score.py:427-429` (`EM_DASH_RE.split` + `WORD_RE.fullmatch`) and
`byteSpans.ts:232-233`
**Reproduce:** `fragmented_words("said--'tis true")`, `fragmented_words("the--'twas")`
**Observed:** `["said--'tis"]` and `["the--'twas"]` — refused, both stacks.
**Expected:** these are the same Gutenberg em-dash convention `legs--upon` is exempted for,
with an elision apostrophe opening the second word (`--'tis`, `--'twas`, `--'em`,
`--'twere` are ordinary in the period prose this panel's corpus is drawn from). The refusal
says: *"Use a passage written in the ASCII alphabet, with ONE straight apostrophe or hyphen
between letters — "don't" and "good-bye" are matched whole, and so is the Gutenberg dash
"legs--upon""* — which is precisely what the user typed. That is the "there was no way to
comply" defect round 5 was fixing, reproduced one corner over. Neither committed corpus
contains it (I scanned both: `ASCII-alphabet refusals: []`), so this is reachable only
through pasted text — which the endpoint accepts.
**Would it have thrown?** It raises a typed 400 — safe, but a false accusation.

### F10. `fragmented_words`' docstring states a behaviour the function does not have
**Severity:** low
**Where:** `code/backend/src/llm_geometry/arch/vacancy_score.py:420` — *"the whole-match
test refuses it, and refuses ``don-'t``, ``'tis`` and ``co<SHY>operate`` with it."*
**Observed:** `fragmented_words("'tis")` → `[]`. A leading apostrophe is not part of a
wordlike run at all (`wordlike_runs("'tis")` → `['tis']`), so `'tis` is accepted, and
correctly so — only the sentence is wrong.
**Would it have thrown?** No.

### F11. The new number-parity test cannot, by construction, detect a change to the rule it tests
**Severity:** medium (test structure)
**Where:** `code/backend/tests/contract/test_api_number_parity.py`
**Reproduce:** `params.py:60` `if not value.is_integer():` → `if False:` (i.e. `as_int`
silently truncates `7.5` → `7`), then run that file.
**Observed:** `50 passed, 1 warning in 3.18s`.
**Expected:** the whole point of extracting `params.py` was to make the rule single and
checkable. A parity test asserts only that two endpoints **agree**, and after the
extraction they share one implementation, so they agree on any rule at all — including a
truncating one. It is not vacuous overall (the older `test_api_geo_params.py` /
`test_api_lex_params.py` do kill this mutant: `4 failed, 676 passed` on the full suite),
but the file round 7 added as the fix's teeth has none against the fix itself.
**Would it have thrown?** No.

---

## (c) Unverifiable here

1. **The q8 arm.** Unchanged since round 3: it needs a real browser with a WebGPU/WASM ONNX
   session. No q8 number was computed or extrapolated. I did not verify any q8 constant
   against a measurement; I verified only that `architecture.md` §8.3a states the same
   figures.
2. **Anything rendered.** `npm run test:e2e` was out of charter, so `docs.spec.ts` and the
   new `tests/e2e/navGuard.spec.ts` were not executed. `navGuard.spec.ts` has still never
   been run by anyone — round 7 says so itself.
3. **`/api/lex/spectrum?baseline_seed`** — not exercised; it needs a valid `model_token`
   that was not available without a training run.
4. **Whether digit-bearing words appear in text a real user pastes.** I scanned both
   committed corpora (`real-mother-goose.txt`, `alice-in-wonderland.txt`): zero
   alphanumeric words in either, so the shipped default passages are unaffected. I did not
   survey any wider corpus.
5. **CI runtimes.** Local macOS only: Node v22.16.0, CPython 3.10.12 in the repo venv.

---

## (d) Confirmed fixed

| round-7 claim | evidence |
|-|-|
| `don''t` / `don-'t` / `don'-t` / `café--x` refused; `legs--upon`, `a---b`, `legs--upon's` scored | reproduced in **both** stacks, identical answers |
| the joiner skew is closed | **8 860** generated cases (all strings over `a b ' -` to length 6, plus 4 000 random strings over letters × the joiner class × digits × `é —`): `total 8860 skew 0` |
| `VACANCY_FP32_REFERENCE` is pinned to a real run | the parser is **not** vacuous: `assert body` on a missing constant, and `assert set(fields) >= {…}` on any field whose value the regex cannot read; a 4-space re-indent, a lost trailing comma or an expression instead of a literal all fail rather than pass silently. `0.2872 → 0.4872` fails both stacks (round 5's surviving mutant) |
| the record cannot be edited without the model | `pytest.approx(..., abs=1e-9)` against a live `vacancy_score("gpt2", default_passages(), p=1, seed=0)` |
| `ONNX_REPOS` / `importWeightSet` / `importBundle` / `validateWeightSet` / `lexEngine/bundle.ts` | re-read; all now `Object.hasOwn` or `Object.create(null)`; the sweep agent found no regression in them |
| the v2 migration has teeth | 3 mutants, 3 kills: migration disabled → `4 failed, 29 passed`; v2 branch accepts any token → `1 failed` (`test_a_pre_identity_file_with_tampered_weights_is_still_refused`); v3 also accepting `legacy` → `1 failed` (`test_a_weights_only_token_under_the_current_version_is_still_refused`) |
| the nav-guard registration is driven in all four panels | `let busy = $state(false)` → `let busy = false`, one panel at a time: `viz/geo/TrainPanel` `1 failed \| 19 passed`, `viz/lex/TrainPanel` `1 failed \| 19 passed`, `viz/geo/FinetunePanel` `1 failed \| 19 passed` (baseline `20 passed`) |
| `params.py`'s rule has teeth | 4 mutants: `as_float` accepting `bool` → `1 failed`; `as_float` accepting non-finite → `2 failed`; `as_float` parsing numeric strings → `5 failed`; `as_int` truncating → parity file green but full suite `4 failed, 676 passed` (see F11) |
| SC-703 + the differential suite | `npx vitest run tests/unit/{vacancy,vacancyGolden,staticVacancy,archVacancy}.test.ts` → `Test Files 4 passed (4) · Tests 240 passed (240)` (was 237 in round 5; +3 new) |

### §10, re-derived

At seeds 0 and 7, from the run above:

```
seed=0 p=1.00 domainTypes=2233/1940/1940 corpusTypes=2211/1918/1918 stemsTotal=1676
  stemsVacated=1676 tokensTotal=16000 tokensVacated=8125 bijective=true imageSize=2233
seed=7 p=1.00 domainTypes=2233/1940/1940 corpusTypes=2211/1918/1918 stemsTotal=1676
  stemsVacated=1676 tokensTotal=16000 tokensVacated=8125 bijective=true imageSize=2233
```

`stemsTotal=1676`, `tokensVacated(p=1.00)=8125`, `domainTypes=2233/1940`, `bijective=true`,
`imageSize=2233` — **identical** to the round-5 baseline. No §10 number moved.

### One mutant survived, and it is provably equivalent

`EM_DASH_RE` `-{2,}` → `-+` survives (`27 passed`). It is **not** a teeth failure: over the
same 8 860 cases the two regexes give identical answers (`differ under -+ : 0`), because a
run containing a single ASCII hyphen either fullmatches `WORD_RE` (caught by the earlier
branch) or contains a second adjacent joiner that fails the piece test under both patterns.
`-{3,}` and `[-']{2,}` are both killed (`2 failed, 25 passed`, by
`test_the_em_dash_exemption_is_a_whole_match_test_not_an_alphabet_test` and
`test_the_shared_case_table_holds_here_exactly`).
