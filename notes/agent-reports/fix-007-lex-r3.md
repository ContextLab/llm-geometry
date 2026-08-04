# Fix round 3 — Lexicon Lab, the static wire boundary, and the nav guard

Date: 2026-08-04. Branch `main`, on top of `33c5ce6`. Agent: FIX LEX ROUND 3.

Charter: `notes/agent-reports/verify-007-arch-lex.md` items 4, 5, 6 (F3b, F5, F11, F12) and
`notes/agent-reports/verify-007-docs-shell.md` V1, V6, V7, V9. Owned files only; the
Architecture tab, `staticClient/arch.ts`, `specs/007-*` and `InfoTab.svelte` were left to
their owner. No server was started, stopped or restarted; `npm run test:e2e` was not run.

Every fix below has a test that **fails before it and passes after it**, verified by
applying the pre-fix code as a mutation and re-running (the "teeth" columns). Every mutation
was reverted and the tree re-checked.

---

## TASK 1 — inherited object keys validated as legitimate values

`key in objectLiteral` walks the prototype chain, so `constructor`, `toString`, `valueOf`,
`hasOwnProperty`, `isPrototypeOf`, `propertyIsEnumerable`, `toLocaleString` and `__proto__`
are all `true` for any object literal. Two validators relied on it, both on **untrusted
input**, and a third and fourth were found by the sweep below.

### Fixed

| where | input it validates | before | after |
|-|-|-|-|
| `src/viz/lex/provenance.ts:188` | `metrics.provenance` from a loaded `.llmlex.json` | `"constructor"` returned `{provenance:"constructor", declared:true}` → every panel's `Record<Provenance,…>` lookup missed → a random-init model rendered with **no untrained warning anywhere** under a load line ending in "verified" | `Object.hasOwn` → falls to `unrecorded`, which is the state that exists for exactly this |
| `src/lib/staticClient/lex.ts:509` | `mint` off the wire | the six inherited keys passed, were cast `mint as MintStrategy`, and threw an **untyped** `Error` out of `buildVacancyMap`; the backend answers the same request with a typed 400 | `Object.hasOwn` → typed `InvalidParamError`, same as `"bogus"` |
| `src/lib/staticClient/lex.ts:1551-1552` | weight names in an imported bundle | a bundle carrying a tensor named `constructor` was **not** reported as extra: the file loaded as though it held exactly the tensors its config implies, with the surplus silently dropped | `Object.hasOwn` on both `missing` and `extra` |
| `src/lib/staticClient/safetensors.ts:165,172,79` | `dtype` and tensor `name` from a **remote** safetensors header | `dtype:"constructor"` passed the gate; `BYTES_PER["constructor"]` is `Object`, so `count * bpe` is `NaN`, the length check `byteLength < NaN` is false, and the decoder's `else` branch — which meant F16 — returned a `Float32Array` of real-looking numbers. `tensors[name]` had the same hole. Real unsupported dtypes (`I64`, `F64`) were always refused correctly; only the inherited keys slipped | `Object.hasOwn` on the index and the dtype table, and `decodeScalars` now refuses an unrecognised dtype instead of falling through to F16 |

### Tests (all four cover inherited keys **as a class**, not one string)

| test | file |
|-|-|
| `treats every inherited object key as an unrecorded provenance, not a declared one` (8 keys + the 6 real values + `"banana"`) | `tests/unit/lexProvenance.test.ts` |
| `keeps every caveat for a file whose provenance is an inherited object key` — the real tab, real corpus, a real saved bundle, reading the DOM | `tests/unit/lexProvenance.test.ts` |
| `rejects the inherited object key <k> as a mint` × 6 | `tests/unit/staticVacancy.test.ts` |
| `refuses a dtype that is an inherited object key instead of decoding it as F16` | `tests/unit/staticSafetensors.test.ts` |
| the extra-tensor loop over `constructor`/`toString`/`__proto__`/`surplus` | `tests/unit/staticLex.test.ts` |

**Teeth.** Restoring `declared in PROVENANCES` → 2 failed (`expected {provenance:'constructor'} to
deeply equal {provenance:'unrecorded'}`, and `expected null not to be null` for the live tab).
Restoring `mint in MINT_STRATEGIES` → 8 failed, including `expected Error: vacancy: unknown
mint strategy "co…` for each key. Removing the `decodeScalars` guard → 1 failed
(`constructor: expected [Function] to throw an error`).

### The full sweep — every `x in OBJ` used as a validity test in the frontend

`rg --pcre2` over `code/frontend/{src,tests}` for the `in` operator outside `for…in`, then
each hit classified by whether an **untrusted** string can reach it.

| site | verdict |
|-|-|
| `src/viz/lex/provenance.ts:177` | **exploitable — fixed** |
| `src/lib/staticClient/lex.ts:500` (`mint`) | **exploitable — fixed** |
| `src/lib/staticClient/lex.ts:1532,1533` (bundle weight names) | **exploitable — fixed** |
| `src/lib/staticClient/safetensors.ts:157` (`dtype`) + `header.tensors[name]` | **exploitable — fixed** |
| `src/lib/PipelineDiagram.svelte:48` — `group in expandOverrides` | latent, **not mine**. Keys are graph group names from `arch/graph.py` (`stem`, `layer_N`, `head`), so no prototype key occurs today; if one did, `!expandOverrides["constructor"]` is `false` and that group would silently refuse to collapse. Cosmetic, but the same construction. |
| `src/lib/lexEngine/bundle.ts:542` — `Object.keys(shapes).filter(n => !(n in wire))` | safe today: the probe keys come from the internal `shapes` map, so no `Object.prototype` name is ever probed. Fragile only if a weight is ever named after one. |
| `src/lib/geoEngine/model.ts:104` (`n in ws`), `:226` (`key in config`) | same shape as above, same verdict: probe keys are internal. |
| `src/lib/geoEngine/model.ts:192` `"embedding" in obj`; `tokenizer.ts:144,150` `"format"`/`"specials" in obj`; `staticClient/lex.ts:1568` `"shape"`/`"data" in entry`; `viz/lex/VacancyPanel.svelte:697` `"seed" in parsed` | safe: the probed literal is not an `Object.prototype` member. |
| `tests/unit/lexGolden.test.ts:305` `"head_w" in weights` | test assertion, same reasoning, safe. |

Python has no analogue — `x in dict` there has no prototype chain — so `lex/vacancy.py`'s
`mint not in MINT_STRATEGIES` is correct as written, and the backend answered all six keys
with a typed 400 throughout. The good pattern already in the tree is
`lexEngine/dolch.ts:86`'s `isDolchBudgetName`, which uses `Array.includes`.

---

## TASK 2 — the public path quoted pre-rewrite statistics

### The numbers, re-derived here, not copied

Through the **running backend** on the real committed corpus
(`POST /api/lex/vacancy {"p":1.0,"seed":0}`), verbatim:

```
{'stemsTotal': 1676, 'stemsVacated': 1676, 'tokensVacated': 8125,
 'tokensTotal': 16000, 'domainTypesEligible': 1940, 'corpusTypesVacated': 1918}
```

and independently through the **browser engine** in `staticVacancy.test.ts`
(`client().lexVacancy({p:1,seed:0})` over `static-data/lex/corpus.json`): the same
`stemsTotal 1676` / `tokensVacated 8125`. So `1676` / `8125` is confirmed on both stacks;
`1680` / `8202` is what `staticClient/lex.ts:514` and `api-lex.md:207` were still saying.

### The fix is one declaration, not a fifth transcription

New file `src/lib/lexEngine/vacancyRefusals.ts` — `SWAP_SUPPLY = {stems: 1676,
vacatedTokens: 8125}`, `SWAP_INCONSISTENT_REFUSAL` built from it, and
`noMappedVocabularyRefusal(mint, p)` for §5.2a. Imported by **both**
`lexEngine/vacancy.ts` (which now throws `vacancy: ${SWAP_INCONSISTENT_REFUSAL}`) and
`staticClient/lex.ts` (which serves it as an `InvalidParamError`). `api-lex.md:207` corrected
to `1 676` / `8 125`.

### …and pinned, so it cannot rot again

* `re-measures both counts through the engine and finds them equal to the constants` —
  the two numbers come out of the real corpus through the real engine on every run.
* `is the sentence the engine itself throws, so neither can drift alone` — the engine's
  thrown message must contain the shared constant, so a re-typed copy in either place fails.
* **The parity fixture now has reject cases.** `scripts/export_vacancy_api_golden.py` gained
  a `REJECTS` block; the regenerated `vacancy-api-golden.json` carries **9 real 400s
  transcribed from the live FastAPI route** (status, error type, message, detail).
  `test_api_lex.py::test_vacancy_matches_the_static_client_fixture` now asserts the live
  route still refuses each one identically; `staticVacancy.test.ts` replays each against the
  static client and compares the error **type** exactly and the **numeric literals in the
  message as a multiset**. The two stacks are not forced to share wording (`consistent=True`
  against `consistent = true` — each names its own language's literal); they are forced to
  quote the same corpus.

**Teeth.** Setting the constants back to `1680` / `8202`: 4 failed, headline
`swap-under-the-inconsistent-control: the numbers it quotes: expected [ '1680', '8.3',
'8202' ] to deeply equal [ '1676', '8.3', '8125' ]`.

### A cross-stack divergence the new reject cases found immediately

`swap-at-intermediate-p` (`{"p":0.5,"mint":"swap"}`) is a typed 400 from the backend and was
an **untyped** throw from the static client (`error type: expected undefined to be
'InvalidParamError'`): `mapVocabWords` raises a plain `Error`, which `toApiError` would have
turned into a `ComputeError` had it been caught at all. Fixed in `vacancyVocab` with the
engine's own condition (`!vmap.injectiveAtEveryP && 0 < p < 1`) and the engine's own
sentence, wrapped in the contract's envelope. It is now green in both stacks.

Item 5's other half — the static `mint` refusal — is covered by TASK 1.

---

## TASK 3 — the nav guard covered 3 panels; two real training runs still died in silence

* `src/viz/lex/VacancyPanel.svelte` now registers `"lex-vacancy-demo"` from an `$effect` on
  `demoBusy` — the single flag `runDemo`, `stopDemo` and the error branch already agree on —
  and releases it in `onDestroy`, exactly as `TrainPanel` does. The two `view.set("architecture")`
  buttons in that same file now raise the hold instead of destroying the demo.
* `src/lib/stores.ts` gained a `beforeunload` listener: it returns immediately while the
  registry is empty (so an idle reload is never interrupted) and cancels the event while
  work is registered. `popstate` fires only for a traversal that stays in the document, so
  reload, close, a typed URL and the Back that steps off a cold deep link were all
  uncovered — verified in the other agent's report as `after Back: url= about:blank … run
  gone` and `native dialogs fired during reload: []`.

Tests, `tests/unit/navGuard.test.ts`:

* `asks the browser to confirm a reload or close while work is registered` — dispatches a
  real cancelable `beforeunload` and asserts `defaultPrevented` is `false` while idle,
  `true` while a run is registered, and `false` again after it releases.
* the panel table grew a third column (each panel's own busy flag) and a fourth row for
  `viz/lex/VacancyPanel.svelte`, plus `covers every panel that holds abortable work of its
  own`, which states the registry-is-opt-in reasoning as an assertion.

**Teeth.** Deleting the `beforeunload` listener → `the run was discarded silently: expected
false to be true`. Deleting VacancyPanel's registration → `viz/lex/VacancyPanel.svelte does
not register on \`demoBusy\``.

**Still unregistered, and NOT mine to fix:** `viz/arch/ArchitectureExplorer.svelte:238`
(`traceCtl?.abort()`). Same structure, same silent loss; it belongs to the Architecture
agent. Reported, not touched.

---

## TASK 4 — the README's units

`README.md:17` now reads **"1000-word vocab in 1003 rows"**. `geo/config.py:22-23` declares
`VOCAB_WORDS = 1000  # word/punctuation types drawn from the corpus` and `VOCAB_SIZE = 1003
# VOCAB_WORDS + the three specials below`; the app's own prose keeps them apart
(`GeometryLab.svelte`: "1000-word vocab" and "Its 1003 token embeddings").

New test `tests/unit/readmeNumbers.test.ts` reads the **exported vocabulary the Pages build
serves** (`public/static-data/geo/vocab.json`: 1003 tokens, 3 specials → 1000 words) and
asserts the README says exactly that, and that the row count is never quoted as a word count.

**Teeth.** Putting `1003-word vocab` back → `expected '# llm-geometry…' to contain
'1000-word vocab in 1003 rows'`.

---

## TASK 5 — the Info-tab sentence, for its owner

The behaviour is now better than it was (the `beforeunload` above), but the sentence still
overclaims, because the browser's own dialog **cannot name the work**. `InfoTab.svelte`,
Known limits, the "A training run cannot survive leaving its tab" bullet — the exact clause
needing correction, verbatim:

> Now any navigation away from a running tab is <i>held</i> and names what it would destroy,
> including browser <b>Back</b>, and doing nothing keeps the run.

Suggested replacement (accurate against `lib/stores.ts` as of this commit):

> Now a tab switch — including a browser <b>Back</b> that stays on this page — is
> <i>held</i> and names what it would destroy, and doing nothing keeps the run. Leaving the
> page itself (a reload, a closed tab, or a Back that steps off the site) cannot be held
> that way: the browser's own confirmation is raised instead, which asks, but cannot say
> which run is at stake.

---

## Suite

Run after all fixes, from a clean tree:

| suite | result |
|-|-|
| `npx vitest run` (frontend, 29 files) | **591 passed, 1 skipped (592)** |
| `npm run check` (svelte-check) | **1173 files, 0 errors, 0 warnings** |
| `npm run build` | **built in 3.73s** |
| `ruff check src/ tests/` | **All checks passed!** |
| `black --check src/ tests/` | **81 files would be left unchanged** |
| `pytest -q` (backend, real models) | **2 failed, 502 passed** in 162.57s — both failures are another agent's in-flight work, see below |
| `pytest -q tests/contract/test_api_lex.py tests/unit` | **315 passed** — everything my changes touch |

### The two backend failures — reported, not dismissed, and not mine

```
FAILED tests/integration/test_geo_scratch.py::test_bundle_roundtrip_preserves_the_model_exactly
FAILED tests/integration/test_geo_scratch.py::test_integrity_checks_cannot_be_bypassed
E  AssertionError: assert 'vocab_sha256' in 'this model file is corrupt: its weights hash to
   9c8c2564… but it declares 3ccfceb8…. Loading it would pair the wrong vocabulary with these
   weights, so it is refused.'
```

They come from an **uncommitted, in-flight rewrite of `code/backend/src/llm_geometry/geo/`**
by the concurrent Geometry agent — `git diff` shows `bundle.py` moving the mandatory
`weights_token` integrity check out of `import_bundle` and re-hashing with
`own_vocab_json(vocab_json)` in `export_bundle`, while `test_geo_scratch.py` still asserts the
old message. Nothing in my change touches `geo/`, and the lex + unit suites (315 tests,
including the two new parity assertions) are green. Recorded here so it is not lost; the geo
agent owns the fix.

### Backend teeth

Mutating `lex/vacancy.py`'s refusal to `1680 open-class stems against 8202` →
`test_vacancy_matches_the_static_client_fixture` fails with *"swap-under-the-inconsistent-control:
the route no longer refuses this request the way the parity fixture records"*. Reverted; tree clean.

`black --check scripts/export_vacancy_api_golden.py` passes under the project's
`line-length = 100`; run without the backend's `pyproject.toml` it reports a reformat, which
is black's 88-column default and not a property of the file.

## Notes for whoever closes the campaign

* `code/frontend/tests/unit/zzverify.test.ts` is still untracked in the working tree. It is
  not mine (header: *"TEMPORARY verification probe — delete after the run"*); it was already
  flagged in `verify-007-arch-lex.md`.
* While I worked, `code/frontend/src/lib/lexEngine/vacancy.ts` was reverted underneath me
  once — a concurrent agent's mutation-test `git checkout` took my edits with it. That is why
  the shared constants live in a **new** file (`vacancyRefusals.ts`) and why the pin is a
  behavioural test on the engine's thrown message rather than a source-text assertion: the
  guarantee survives either copy being rewritten by hand.
* No secrets, no scratch files, no screenshots. Every mutation used for a teeth check was
  reverted and verified.
