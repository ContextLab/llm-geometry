# Fix 007 — round 7, geo/lex slice (F1, F2, F3, F4, F5, F9, F10)

Agent "ROUND 7 GEO/LEX". Date: 2026-08-04. Branch `main`, from `9f0bc5e`.
Charter: `notes/agent-reports/verify-007-round5-geolex.md`. F6 was assigned elsewhere and
is untouched here. Nothing under `arch/`, `lex/vacancy.py`, `lexEngine/vacancy.ts`,
`specs/007-*` or `InfoTab.svelte` was modified — a sibling agent is editing those, and
their in-flight changes were present in the working tree throughout (see "Working tree"
at the end).

Every mutation below used `cp`/`shasum` backups in the session scratchpad and was
restored the same way; `git checkout` was never used. `npm run test:e2e` was not run.

| Finding | Status | Killed by |
|-|-|-|
| F10 — false remedy + unbumped `BUNDLE_VERSION` | **fixed** | 5 backend + 4 TS cases |
| F1 — `owns_vocab=True` with no word list reads as canonical | **fixed** | 1 backend + 2 TS cases |
| F2 — `routes_geo._as_float` is a bare `float()` | **fixed** | 50 differential cases |
| F3 — static geo truncates/accepts what the backend refuses | **fixed** | 3 static + 2 engine cases |
| F4 — unk-rate boundary mutant survives 815 TS tests | **fixed** | mutant dies in both sites |
| F5 — `lexEngine/bundle.ts:515` untyped `TypeError` | **fixed** | 11 cases + 3 mirrors found |
| F9 — nav-guard tests rest on a false premise | **fixed** | mutant dies in all 4 panels |

---

## F10 — the file format moved when the identity did

`weights_token` began hashing the word list in `0d23123`. That changed what the
`weights_token` FIELD IN A SAVED FILE means, so the file format moved with it — and
`BUNDLE_VERSION` stayed at 2. Consequence: every file the previous build wrote for a model
with its own vocabulary failed the re-hash and came back as

> this model file is corrupt: its weights and vocabulary hash to … but it declares …

which is an accusation against an intact file, and against the exact file the v15 cache
refusal points the user at ("a SAVED MODEL FILE still loads: open it again").

**Fix.** `BUNDLE_VERSION = 3`, `LEGACY_WEIGHTS_ONLY_VERSION = 2`, and `import_bundle` /
`importBundle` read BOTH: a version-2 payload is checked against the weights-only hash its
own format put in it, and the current identity is re-derived from the (weights, word list)
pair it carries. Refusing instead would strand an intact file and buy nothing — the binding
a v3 token gives is absent from *every* v2 file, including the ones that load unchanged
today only because their word list is the shipped one and so takes no part in either hash.
A file that DECLARES version 3 is held to version 3.

**One thing I deliberately did NOT do.** My first cut added a branch saying "this declares
version 3 but names itself the way a version-2 file does — not corrupt, fix the `version`
field". It is **undecidable**: weights + an own word list + a weights-only token is what a
v2 writer produced AND what swapping a v3 file's word list produces. `geoEngine.test.ts`'s
existing substitution-attack case caught me writing it (it asserted `/corrupt/` and got the
new sentence), which is the test doing its job. The branch is gone and the ambiguity is
recorded in a comment and in
`test_a_weights_only_token_under_the_current_version_is_still_refused`.

The v15 message is now **true** rather than deleted — it says the file loads, and it does,
including a pre-change one. `test_the_schema_bump_message_names_a_recovery_that_actually_works`
asserts the promise and the behaviour in one case so neither can drift alone.

**Contract.** `version: 2` → `3` in `specs/002-interactive-model-explorer/contracts/api.md`,
in its own commit, with a note explaining why and what `POST /api/geo/model` now accepts.

Tests (`tests/integration/test_geo_derived_vocab.py`, all failing before / passing after):
`test_a_pre_identity_model_file_opens_instead_of_being_called_corrupt`,
`test_a_pre_identity_file_with_tampered_weights_is_still_refused`,
`test_a_version_this_build_cannot_read_names_the_versions_it_can`,
`test_a_weights_only_token_under_the_current_version_is_still_refused`,
`test_the_schema_bump_message_names_a_recovery_that_actually_works`.
TS mirrors in `geoDerivedVocab.test.ts` → `a model file written before the identity change
opens [round 5, F10]` (4 cases).

Mutation proof: with `SUPPORTED_VERSIONS = (BUNDLE_VERSION,)` and the migration branch
disabled — i.e. exactly the shipped behaviour — `4 failed, 1 passed`.

## F1 — the readers now refuse what the writer refuses

`tokenizer_for` / `tokenizerFor` returned the CANONICAL tokenizer for a set whose
`owns_vocab` is set and whose word list is gone, while `export_bundle` / `exportBundle`
refused the identical state calling the substitution catastrophic. Four routes answered
200 with Alice in Wonderland's words, and because the answer really was the canonical
tokenizer, `GeometryLab.verifyVocab`'s probe agreed with it and the tab reported the
vocabulary **verified**.

Both readers now raise the writer's sentence. Also fixed while here: the TS
`exportBundle` refusal threw `notFound` (404-shaped) where the backend raises
`InvalidParamError` and the **frozen contract says 400** — "Where a model's vocabulary
cannot be recovered, this endpoint returns `400 InvalidParamError`". One user action, two
statuses, depending on which build served it. Now `invalidParam` in both.

Tests: `test_a_model_that_claims_a_word_list_it_has_not_got_is_refused_by_every_reader`
(direct call + `/tokenize`, `/trace`, `/vector_field`, `/finetune`, all 400
`InvalidParamError`); TS `a model that claims a word list it has not got is refused by
readers too [round 5, F1]` (2 cases, one of which pins reader and writer to the same
error TYPE). Mutation: guard → `if False` ⇒ the backend case fails.

## F2 — one implementation, not two that agree by inspection

New `llm_geometry/api/params.py` holds `as_int` / `as_float` / `as_bool`; `routes_geo` and
`routes_lex` both delegate. The geo copy's bare `float(value)` is gone, and with it
`lr: "٠.٥"` → 202-and-trains-at-0.5, `"٧"`/`"７"`/`"७"`/`"1_000"`/`"1e3"` → 202, and
`10**400` → untyped 500.

New `tests/contract/test_api_number_parity.py` sends the **same 24 hostile values** to
`POST /api/geo/finetune` and `POST /api/lex/train` and requires the same status, the same
error type and the same message shape, plus two controls proving the gate refuses TYPES
rather than everything (`lr: -1.0` reaches "lr must be > 0" on both; `steps: 0.0` is read
as the integer 0 on both). 50 cases. Against the pre-fix `_as_float`: **10 failed**.

## F3 — the static build's parameters are the backend's

`staticClient/geo.ts` used `Math.trunc(body.steps ?? …)` and `!(lr > 0)`; `Infinity > 0`
is `true`. Now `asInt`/`asFloat`.

The TS rule is now also in ONE place: `src/lib/params.ts` (`makeNumberParams`), bound to
each error taxonomy by two-line modules `staticClient/params.ts` and `geoEngine/params.ts`.
`staticClient/lex.ts`'s private copies were removed in favour of it.

**Mirror found while fixing it:** `GeoEngine.finetune` — the reference implementation the
Python backend is golden-tested against — had a THIRD answer (`Math.trunc(body.steps)`,
`body.lr ?? default`, `Math.trunc(body.seed)`). Fixed. Tests: `the engine's fine-tune
parameters are the parameters you asked for [round 5, F3]` (2 cases) and `the public
build's geo parameters are the backend's geo parameters [round 5, F3/F4]` (2 cases).

## F4 — the unk boundary now has teeth in TypeScript

Added the 90-of-100 case (`rate === FINETUNE_MAX_UNK_RATE` exactly) plus the 89-of-100
control, at both TS sites. **Mutation `>=` → `>` in both `geoEngine/index.ts` and
`staticClient/geo.ts` simultaneously: `2 failed, 57 passed`** (it previously survived all
815 tests). Both files restored, `shasum`-verified.

## F5 — the unfixed mirror, and three more of its own

`lexEngine/bundle.ts` `readWeights` read `shapes[name]` off a plain object, so a weight
called `toString` got a builtin, passed "has no slot for", and died as
`TypeError: shape.reduce is not a function` — untyped, outside the `ApiError` surface the
file dialog prints. Now `Object.hasOwn`, a null-prototype accumulator, and `Object.hasOwn`
for the missing-tensor test too.

**Checked whether the original fix had other mirrors — it had three:**

1. `geoEngine/index.ts::importBundle` accumulated into `{}`, so `ws["__proto__"] = arr`
   set the object's PROTOTYPE instead of adding a key: the tensor vanished, `Object.keys`
   never reported it, `validateWeightSet` saw nothing extra, and the `weights_token`
   re-hash was computed over the tensors that survived — a file loading as though it held
   exactly the tensors its config implies while carrying one nobody looked at. Now
   `Object.create(null)`.
2. `geoEngine/index.ts::importWeightSet` — the same, for sessionStorage payloads.
3. `geoEngine/model.ts::validateWeightSet` used `!(n in ws)`, which walks
   `Object.prototype`. Now `Object.hasOwn`.

Fixing (2) surfaced a real defect the silent drop had been hiding: with the tensor no
longer vanishing, `weightsToken` threw `NotFoundError` **out of `importWeightSet`**, whose
contract is to refuse by returning `false` — and it is called from `restorePersistedSets`
during boot, so one bad sessionStorage entry would have taken the tab down. It now refuses
with a reason.

Tests: `lexBundle.test.ts` → `a weight named after a JavaScript builtin is refused, typed
[round 5, F5]` (10 inherited keys + a control), each asserting `type === "InvalidParamError"`
and the "has no slot for" sentence; `geoDerivedVocab.test.ts` → `a geo model file cannot
smuggle a tensor past the weight set [round 5, F5 mirror]` (2 cases). Note: the fixtures
inject the key as **JSON text**, because assigning `weights["__proto__"] = …` in JavaScript
sets the prototype and creates no key at all — an object-literal fixture cannot reach this
case, and my first draft silently didn't.

## F9 — all four panels driven, and the mutant killed

The premise "these three cannot be driven to `busy` in jsdom" was false. All four panels
set their flag before anything that can fail (`geo/TrainPanel.run` and
`geo/FinetunePanel.run` before their first `await`; `lex/TrainPanel.resetRun` before
`new Worker`), so a real click registers observably even where the work cannot finish.

`navGuard.test.ts` gains `every panel that owns destructible work is DRIVEN, and registers
it` — 3 new cases (VacancyPanel was already driven), each: mount the real component, type
into the real textarea, click the real Run button, assert the registry holds the id, assert
`view.set` is HELD and names the run, then unmount and assert the registry empties. The
`lex/TrainPanel` case swallows the environment's `ReferenceError: Worker is not defined`
for exactly one click and asserts it was that error — the failure is jsdom, and the point
is that the registration already exists when it happens. The source-text block is kept, now
described honestly as pinning the SHAPE rather than the behaviour.

**Mutation `let busy = $state(false)` → `let busy = false`, one panel at a time:**

| panel | before | after |
|-|-|-|
| `viz/geo/TrainPanel.svelte` | survived | `1 failed, 19 passed` |
| `viz/geo/FinetunePanel.svelte` | survived | `1 failed, 19 passed` |
| `viz/lex/TrainPanel.svelte` | survived (17 passed) | `1 failed, 19 passed` |
| `viz/lex/VacancyPanel.svelte` (`demoBusy`) | caught | `2 failed, 18 passed` |

All four restored from `cp` backups, `shasum`-verified.

**e2e gap closed on paper, not run.** `tests/e2e/navGuard.spec.ts` is new: a real 400-step
Lexicon Lab run in a real browser, a tab click held with the `alertdialog` asserted
VISIBLE, Stay letting the run finish, Discard leaving, browser Back held the same way, and
an idle tab never held. I did **not** run the e2e suite (charter). It typechecks under
`svelte-check`; it has never been executed, and I say so rather than claiming it passes.

---

## Suites

Run after the last change, on the whole tree:

```
backend   pytest -q                 680 passed
          ruff check src/ tests/    All checks passed!
          black --check src/ tests/ 88 files would be left unchanged
frontend  npm run check             1186 FILES 0 ERRORS 0 WARNINGS
          npx vitest run            35 files, 851 passed | 1 skipped (852)
          npm run build             ✓ built in 3.14s
```

Before this round: 680 → was 649 backend cases; 852 → was 816 frontend. No existing test
was weakened, skipped or deleted; the only existing assertion changed is the one that
caught my undecidable branch, and it was changed back to what it always said.
SC-703 and the TS↔Python golden/differential cases are inside those runs and passed.

## Working tree

A sibling agent's in-flight edits were present throughout and are **not** mine and **not**
committed by me: `arch/vacancy_score.py`, `staticClient/arch.ts`, `staticClient/byteSpans.ts`,
`archVacancy.test.ts`, `wordClasses.test.ts`, `test_arch_vacancy_score.py`,
`test_arch_word_classes.py`, `archStaticFacts.test.ts`, `specs/007-*`,
`scripts/measure_vacancy_fp32.py`, and the two arch report files. Every commit here staged
explicit paths. One caution for whoever runs formatters next: `black src/` is
repository-wide, so scope it (`black <your files>`) while another agent is editing.

The untracked `notes/agent-reports/verify-007-round5-{geolex,arch}.md` are their authors'
deliverables and were left uncommitted.

No secrets: every change is code, tests, or prose; the fixtures are generated nonsense
words, content hashes, and the committed public-domain corpora.
