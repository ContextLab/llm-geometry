# Verify 007 — round 5, against `fix-007-{geo,lex}-r5.md` and `fix-007-teeth.md`

Agent "VERIFICATION AGENT V6b". Date: 2026-08-04. Branch `main` @ `9f0bc5e`.
No file in the repository was modified by me. Every mutation was applied with `cp`
backups in the session scratchpad and restored by `cp`, verified by `shasum`;
`git checkout` was never used. Backend ran on **:8020** with `LLM_GEOMETRY_CACHE_DIR`
pointed at the scratchpad. `npm run test:e2e` was not run; no MCP browser tool was used.

Two sibling-agent edits appeared in `git status` while I worked
(`code/frontend/src/viz/arch/vacancyVerdict.ts`, `code/backend/src/llm_geometry/lex/vacancy.py`).
Neither is mine; I left both alone.

| # | Charter item | Verdict |
|-|-|-|
| 1 | the vocabulary identity, fifth attempt | **PARTIAL** — no new corrupt *file*; a fifth wrong-ANSWER door survives, with the same route asymmetry as F1 |
| 2 | the schema bump | **PARTIAL** — refusal/keep/explain all VERIFIED; the recovery the message names is **REFUTED**, and it is a sibling format that moved without a bump |
| 3 | `hasOwn` / inherited-key sweep, 4th attempt | **REFUTED** — 3 more found, one of them a wrong answer that does not throw |
| 4 | coercion | **REFUTED** — `routes_geo._as_float` still coerces strings incl. Unicode digits, and 500s; static geo truncates |
| 5 | mutation teeth | **PARTIAL** — 4 of 5 fresh mutants killed; the unk-rate boundary mutant survives the whole TS suite |
| 6 | nav guard e2e gap | **REFUTED** — the "cannot be driven in jsdom" premise is false; a second mutant survives |

---

## (a) Defects surviving

### F1. `owns_vocab = True` with no stored word list reads under Alice's words in FOUR routes while `export_bundle` refuses the identical state — the fifth door
**Severity:** high (contingent on reachability, see below)
**Where:** `code/backend/src/llm_geometry/geo/tokenizer.py:289-290` +
`geo/weights.py:421-429` (`load_weight_set_vocab` returns `None` for "no `vocab` key",
which `tokenizer_for` reads as "canonical"); guard that proves the state is known:
`geo/bundle.py:88-94`. Mirror: `src/lib/geoEngine/index.ts:331-338` (`tokenizerFor`) vs
the guard at `:691-698` (`exportBundle`).
**Reproduce:** `scratchpad/p1.py` — store an entry with `owns_vocab=True` and no `vocab`
payload (the shape `save_weight_set`'s own docstring at `weights.py:343-344` says is
"preserved as written"), then read it.
**Observed (verbatim, Python):**
```
  wrote entry 13c30ba17bd870f5776e59bb3f06cc36 owns_vocab=True, no vocab payload
  tokenizer_for(...).words[:4] = [',', '"', 'the', '.']   (canonical?  True )
  export_bundle REFUSED: InvalidParamError: weights_token '13c30ba1…' has no vocabulary stored beside it, and its ids mean its own words rath…
  weight_set_owns_vocab: True
  load_weight_set_vocab: None
```
and it **propagates through derivation** — `POST /api/geo/weights` on it succeeds:
```
  derived token: 8b1678f1ad7003da073fb06ca0e18b99
  derived owns_vocab: True  vocab: None
  derived tokenizer words[:4]: [',', '"', 'the', '.']
```
Same asymmetry in the browser engine (`scratchpad/t1.ts`, real `GeoEngine.fromAssets`
over the committed `static-data/geo` assets), after removing only the vocabulary entry:
```
   tokenizerFor -> words[:3] = [ ',', '"', 'the' ]  canonical? true
   tokenize() -> ["alice","said","the","queen"]
   exportBundle THREW: NotFoundError weights_token 'dc205ce0…' has no vocabulary in this session, and it…
```
**Expected:** `/tokenize`, `/trace`, `/vector_field` and `/finetune` must refuse exactly
what `export_bundle` refuses. `export_bundle`'s own message calls the substitution
"CATASTROPHIC" (`bundle.py:82-87`); four routes perform it and answer 200.
**It reaches the screen.** `viz/geo/GeometryLab.svelte:161` runs `verifyVocab` against
`client.geoTokenize(text, token)`; under this state the probe **succeeds** (the tokenizer
really is the canonical one), so `vocabVerified = true`; `geoExportModel` then 400s, so
`modelWords` stays `null`; and `label()` at `:82` falls to `tokenText(id)` — the bundled
Alice table. The tab reports the vocabulary verified and prints the wrong words. I did not
observe the DOM.
**Would it have thrown?** **No.** 200 with plausible words.
**Reachability — stated honestly.** I could **not** reach the seed state through the HTTP
API or the static client: every writer (`scratch.py:174`, `finetune.py:258`,
`jobs.py:286`, `bundle.py:223`) either stores the word list or sets `owns_vocab=False`.
I created it with a direct `store.put` (Python) and by deleting a private map entry (TS).
What is proven is (i) the code deliberately permits writing it, (ii) it survives
derivation, and (iii) two readers of the same state disagree, one of them silently. I do
not know whether a shipped path reaches it.

### F2. `routes_geo._as_float` still coerces strings — including Unicode digits — while `routes_lex._as_float` refuses them
**Severity:** high
**Where:** `code/backend/src/llm_geometry/api/routes_geo.py:308-324` — `float(value)`
inside a `try`, with only a finiteness check added. Compare `routes_lex._as_float`, which
round 5 rewrote to "bool refused, only `int`/`float` accepted".
**Reproduce:** `POST /api/geo/finetune {"text":"…","lr": <v>}` on :8020.
**Observed (verbatim):**
```
  arabic-indic "٧"         ->  202 ACCEPTED   {"job_id":"04c24a3772a3447b9c97e9cdbed8c5f4","ready":false}
  fullwidth "７"            ->  202 ACCEPTED   {"job_id":"04c24a3772a3447b9c97e9cdbed8c5f4","ready":false}
  devanagari "७"           ->  202 ACCEPTED   {"job_id":"04c24a3772a3447b9c97e9cdbed8c5f4","ready":false}
  arabic-indic "٠.٥"       ->  202 ACCEPTED   {"job_id":"97e55064aae341069576c6549c52d35b","ready":false}
  "1_000"                  ->  202 ACCEPTED   {"job_id":"0f1cb4d98e5a42fd80d217167c85be93","ready":false}
  "+7" / " 7 " / "7" / "1e3" / 1e3            ->  202 ACCEPTED
  10**400 (int)            ->  500 ** 500 **  {"error":{"type":"InternalError","message":"int too large to convert to float","detail":{}}}
```
against the same 34 values on `POST /api/lex/train {"lr":…}`:
```
  arabic-indic "٠.٥"       ->  400 REFUSED    {"error":{"type":"InvalidParamError","message":"lr must be a number, got '٠.٥'"…
  "1_000"                  ->  400 REFUSED    …
  10**400 (int)            ->  400 REFUSED    {"…":"lr must be a number a float64 can represent exactly enough to …
```
**Expected:** the charter's rule — the two stacks must agree on every value. `Number("٠.٥")`
is `NaN` in JavaScript; `float("٠.٥")` is `0.5` here. This is the exact divergence the
campaign already named, still live on the geo tab, *plus* the untyped 500 the lex rewrite
removed one module over. Two route modules that the fix report says carry "the same rule
and wording" do not.
**Would it have thrown?** No for the strings (a real fine-tune runs at a learning rate the
other stack cannot express); yes-but-untyped-500 for `10**400`.

### F3. Static-mode geo silently rewrites `steps`/`epochs` and accepts an infinite `lr`
**Severity:** medium
**Where:** `code/frontend/src/lib/staticClient/geo.ts:247` `Math.trunc(body.steps ?? …)`,
`:321` the same for `epochs`, `:251-252` `const lr = body.lr ?? …; if (!(lr > 0)) throw`.
`Infinity > 0` is `true`.
**Expected:** the backend answers `steps: 7.5` with a typed 400 saying it is "not rounded
or truncated, because a number that is not the number you asked for is worse than a
refusal" (`routes_geo._as_int`), and `lr: Infinity` with a typed 400. The public site
truncates and accepts. Round 5 applied exactly this rule to `staticClient/lex.ts`
(`asInt`/`asFloat`) and did not carry it to `staticClient/geo.ts`.
**Would it have thrown?** No.

### F4. The unk-rate boundary is pinned in Python and pinned by NOTHING in TypeScript
**Severity:** medium
**Where:** `src/lib/geoEngine/index.ts:596` and `src/lib/staticClient/geo.ts:275`.
**Reproduce:** `>=` → `>` in **both** TS sites, then `npx vitest run`.
**Observed:**
```
src/lib/geoEngine/index.ts:596:    if (unkRate > FINETUNE_MAX_UNK_RATE) {
src/lib/staticClient/geo.ts:275:    if (unkRate > FINETUNE_MAX_UNK_RATE) {
 Test Files  34 passed (34)
      Tests  815 passed | 1 skipped (816)
```
The identical mutation in Python is caught:
```
FAILED tests/integration/test_geo_derived_vocab.py::test_the_unk_bound_refuses_a_stream_that_is_exactly_at_it
1 failed, 12 passed, 604 deselected
```
Round 4 reported this ("reverting either TS `>=` to `>` breaks no test"); round 5 did not
close it, so a stream that is exactly 90 % `<unk>` can be accepted by the public build and
refused by the backend, with the loss reported either way.
**Would it have thrown?** No.

### F5. `lexEngine/bundle.ts:515` — a prototype-named tensor in a user's model file escapes as an untyped `TypeError`
**Severity:** high · **Where:** `code/frontend/src/lib/lexEngine/bundle.ts:515`
(`const shape = shapes[name]`, `name` from `Object.entries()` of the file's `weights`).
This is the **unfixed mirror** of the `staticClient/lex.ts:1567` fix round 5 shipped; only
the static path got it. Reached from the file-open in `viz/lex/ModelFile.svelte`.
**Observed (executed):**
```
bogus_weight   -> GeoEngineError: model file carries a weight "bogus_weight" that a tied 1-layer model has no slot for
toString       -> TypeError: shape.reduce is not a function
constructor    -> TypeError: shape.join is not a function
__proto__      -> TypeError: shape.join is not a function
```
**Would it have thrown?** Yes — untyped, outside the `ApiError` surface the file dialog prints.

### F6. `staticClient/arch.ts:913` and `:1073` — `ONNX_REPOS[m.model_id]` truthiness test hands a JavaScript builtin to the ONNX runtime
**Severity:** high — **wrong answer, no throw**
**Where:** `const repo = ONNX_REPOS[m.model_id]; if (!repo) …`. `m.model_id` is the *same
untrusted field* round 5 fixed one file over at `staticClient/index.ts:115`; an inherited
member is truthy, so the "no ONNX export is wired up" refusal never fires.
**Observed (executed):**
```
archGenerate RETURNED (guard bypassed): {"handed":true}
repo handed to the ONNX runtime: function function Object() { [native code] }
control — a model_id with no ONNX repo:
  ApiError: No browser (ONNX) export is wired up for some/unmapped-model — …
```
**Would it have thrown?** No.

### F7. `viz/arch/ArchInspector.svelte:217` — `{KIND_EXPLAINER[node.kind]}`, bare index with `kind` from `graph.json`
**Severity:** low · unreachable today only because `arch/tracing.py:108-125
classify_module` returns a closed enum. Renders the prototype member's text; no throw.

### F8. `lib/geoEngine/weights.ts:194` is still "safe only by luck"
**Severity:** low · `preset` is unvalidated and reaches `fixtures.square[preset]`; the
second index is `String(seed)` (always digits), so it lands on `undefined` and the typed
refusal still fires — with a misleading message. No wrong answer. Unchanged since round 4.

### F9. The nav-guard source-text regex still has a surviving mutant, and its stated premise is false
**Severity:** high · **Where:** `code/frontend/tests/unit/navGuard.test.ts:432-442`
`fix-007-teeth.md` claims the other three panels "**cannot** be driven in jsdom
(`lex/TrainPanel` constructs `new Worker(...)` directly; the two geo panels need the
backend)". **All three were driven**, with no backend and no mock:
```
GEO-TRAIN registry after click: [ 'geo-train' ]
GEO-TRAIN held nav: {"target":"info","work":[{"id":"geo-train","label":"a from-scratch training run in the Geometry Lab"}]}
GEO-FT   registry after click: [ 'geo-finetune' ]
LEX-TRAIN registry after click: [ 'lex-train' ]   (typeof Worker in jsdom: undefined)
```
Both geo panels set `busy = true` before their first `await` (`geo/TrainPanel.svelte:117`,
`geo/FinetunePanel.svelte:69`); `lex/TrainPanel` sets it at `:258`, before `new Worker` at
`:270` throws. And the tightened regex is still not teeth:
* mutant 1, `$effect(() => {…})` → bare `{…}` in `viz/lex/TrainPanel.svelte`: **caught**
  (`Tests 1 failed | 16 passed (17)`);
* mutant 2, `let busy = $state(false)` → `let busy = false` — the effect never re-runs, the
  panel never registers, i.e. *the original silent-loss bug*: **SURVIVES**,
  `Tests 17 passed (17)`.
**Would it have thrown?** No.

---

## (b) NEW defects round 5 introduced

1. **F2** — `routes_geo._as_float` was written in round 5 as the weaker sibling of the
   `routes_lex._as_float` the same round hardened. `fix-007-geo-r5.md` states only
   "`_as_float` refuses non-finite", which is true and is not the rule the lex half adopted.
2. **F10 (below)** — round 5's new schema-15 refusal message names a recovery that does not
   work. Before the message existed, the user was told nothing; now they are told something
   false.

---

## (c) The schema bump, item by item — and the sibling that moved without one

### F10. `BUNDLE_VERSION` is still 2: a pre-identity SAVED MODEL FILE is refused as "corrupt", which is the one recovery the new v14 message names
**Severity:** high
**Where:** `code/backend/src/llm_geometry/geo/bundle.py:38` and
`src/lib/geoEngine/index.ts:39` (`BUNDLE_VERSION = 2`, last moved for a *different*
reason); message at `geo/weights.py:389-397`.
**Reproduce:** `scratchpad/p2.py` — write a genuine schema-14 `geo-weights` entry, read it,
then open the model file that same build would have saved.
**Observed (verbatim):**
```
wrote a SCHEMA-14 entry, token = b03a2a823a19b8d32b45d231169cb631
  tokenizer_for:    NotFoundError: … was stored by an earlier build (cache schema v14; this build reads v15) and is not loaded …
  load_weight_set:  NotFoundError: … (same)
  export_bundle:    NotFoundError: … (same)
  files still present (nothing destroyed)? True True

  --- the recovery the message names: the model FILE that build saved ---
   import_bundle REFUSED: InvalidParamError: this model file is corrupt: its weights and vocabulary
   hash to a93dd0282b08b3baa0f6c5098c3565f5 but it declares b03a2a823a19b8d32b45d231169cb631.
```
and the same in the browser engine (`scratchpad/t1.ts`, real `GeoEngine`):
```
pre-identity FILE refused: InvalidParamError | this model file is corrupt: its weights and vocabulary
hash to dc205ce0bf5c7d6af9e707d38e78bea9 but it declares 6169c2a647b9ae3a048f2fdb34648240.
```
**Expected:** the file format's identity field changed meaning in `0d23123` exactly as the
cache key did, so it needed the same bump. `BUNDLE_VERSION = 3` would let `import_bundle`
say "this file was written before a model was named by its words as well as its weights"
and re-derive the token, instead of accusing an honest file — the user's only surviving
copy — of corruption. As shipped, the v15 message actively directs the user to it:
"Nothing was deleted, and a SAVED MODEL FILE still loads: open it again, or train the
model again." For the models the message is about (the ones with their own word list — the
whole reason a v14 entry is unreadable), that sentence is false.
**Would it have thrown?** Yes, with a message that says the wrong thing.

**Other persisted formats checked:** `MINTED_SETS_KEY … :v2` (deliberately unbumped, and
the reasoning at `staticClient/geo.ts:60-69` is sound — those payloads *are* hash-
distinguishable); `GEO_WEIGHTS_TOKEN_KEY`/`GEO_MODEL_NOTE_KEY` (no format); `LEX_BUNDLE_VERSION = 1`
(three independent copies — `staticClient/lex.ts:458`, `lexEngine/bundle.ts:79`,
`routes_lex.py:109` — unmoved, but nothing pins them equal); `ARCH_GRAPH_SCHEMA_VERSION`,
`_TRAINER_VERSION` (unmoved). `checkpoint.json`'s `metrics.checkpoint_id` is still
`be5359a1c66bda29c8c554269e589009` and matches `weights_token(ws, None)`.

---

## (d) What I confirmed genuinely fixed

* **The store miss now raises everywhere it is read.** Mutation: `weight_set_entry` returns
  `{"meta": {}, "arrays": {}, "spec": {}}` on a miss instead of raising →
  `3 failed, 24 passed` (`test_an_unknown_token_is_unknown_to_tokenize_too`,
  `test_tokenizer_for_refuses_an_evicted_model_instead_of_relabelling_it`,
  `test_a_cache_from_before_the_identity_change_says_what_happened`). Restored,
  `shasum 18c910ca351e8b562503f5102dcc5f1967d66b0a`.
* **The schema-version comparison has teeth.** Mutation: `store.get`'s
  `!= SCHEMA_VERSION` → `< SCHEMA_VERSION - 1` (a plausible "be lenient" refactor) →
  `3 failed, 30 passed`, including `test_cache.py::test_schema_version_mismatch_is_a_miss`
  and both pre-identity-cache cases. Restored, `shasum fd015d631c294e8b69f3af07d3d3810ab763e0bd`.
* **The schema-15 refusal itself: verified.** All three readers refuse with the
  format-moved explanation, and **nothing is destroyed** — both files are still on disk
  after the refusal (verbatim above). Only the *remedy* is wrong (F10).
* **`Object.create(null)` in `safetensors.ts` has teeth.** Reverting it to `{}` →
  `Tests 2 failed | 10 passed (12)`. Restored, `shasum 4057401a1a3f8ee51cd7884d9af439e6f001bad0`.
* **`DISPLAY_NAMES` has teeth.** Reverting `Object.hasOwn(...)` to `DISPLAY_NAMES[…] ?? …`
  → `Tests 5 failed | 149 passed (154)`. Restored, `shasum e1e32eb14c742ce931c5bfbdd3f023463b6b12eb`.
* **`lexEngine/vacancy.ts` `METER_FEET` is fixed** (null-prototype + frozen + `Object.hasOwn`
  at `:371-372,398`): `foot = "constructor"/"toString"/"__proto__"` all throw
  `unknown foot …`; `"iamb"` → `0.5`. **`staticClient/arch.ts:829` is fixed** (now `:836`,
  `Object.hasOwn(header.tensors, c)`). `hfDatasets.ts:129,179` fixed.
* **`_as_int` / `_as_seed` / `_edit_seed` are correct** on all 34 hostile values across
  `POST /geo/train`, `/geo/finetune`, `/geo/train_scratch`, `/geo/weights` — `"0x10"`,
  `"+7"`, `" 7 "`, `""`, `null`, `true`, `[]`, `{}`, `Infinity`, `NaN`, `2^53`, `2^53+1`,
  `"٧"`, `"７"`, `"७"`, `"1_000"`, `"0b101"`, `"0o17"`, `"1e3"` all typed 400/422; `7.0`
  accepted, `7.5` refused with the "not rounded or truncated" sentence. `-0`/`-0.0` are
  accepted as `0` in both stacks (agreement, not a defect).
* **`importWeightSet` checks the payload it is handed.** A reversed word list under an
  honest token → `false`; the honest payload → `true`, and the model keeps its own words.
* **The `navGuard` behaviour test for `VacancyPanel` is real.** `npx vitest run
  tests/unit/navGuard.test.ts` → `Tests 17 passed (17)`, two of them driving genuine
  training runs. The wiring itself is real: `nav-hold` renders at `src/App.svelte:117-131`,
  all 13 in-app nav sites go through `view.set` (`lib/stores.ts:116`), no
  `location.hash =` bypass, `beforeunload` at `stores.ts:226`.

---

## (e) Claims I could not verify

1. **Whether F1's seed state is reachable through a shipped path.** Stated in F1. I say "I
   don't know" rather than guess.
2. **Anything rendered end to end.** No e2e, no browser, no screenshot. F1's route to the
   DOM is traced through `GeometryLab.svelte:82/161/170`, not observed.
3. **Lex token/vocabulary resolution was not attacked exhaustively.** The lex vocabulary is
   rebuilt deterministically from `(source, budget, size)` (`routes_lex._resolve_budget`)
   rather than resolved from a token, so the geo store-miss shape does not transfer; I did
   not prove there is no other shape there.
4. **The q8 arm of `VACANCY_Q8_UNCERTAINTY_NATS`** — unchanged from three previous rounds;
   it needs a real browser. No q8 number was computed, guessed or extrapolated here.
5. **Linux / Python 3.10 in CI.** All runs were local macOS, Python from
   `code/backend/.venv`.
6. **Whether a TRUE collision exists in either `weightsToken`.** Not attempted this round.

### What the zero e2e coverage of the nav guard leaves unverified
The regression itself — a real browser run at step N, a tab click, the run surviving — is
tested nowhere. `lex/TrainPanel`'s registration only ever executes where `Worker` is
*absent*, so the real-worker `busy` lifecycle is untested in any suite;
`VacancyPanel`'s jsdom run uses `trainInWorker`'s inline no-Worker fallback;
`beforeunload` is asserted through a synthetic event's `defaultPrevented`, which Chrome
honours only with sticky user activation; jsdom `pushState`/`popstate` is not Chrome's;
and nothing checks that the `role="alertdialog"` is visible, unclipped by the WebGL
canvas, focused or announced. The production `VITE_DATA_MODE=static` build exercises the
guard zero times.

---

## Repo hygiene

Nothing in the repository was modified by me; every mutated file was restored from a
scratchpad `cp` backup and re-verified by `shasum` (`geoEngine/index.ts`
`5dacb22532b40990b1965dc289748c954488d267`, `staticClient/geo.ts`
`7e3f637a0112d2d3928dc2526e1d024e71f8c6b1`, `staticClient/safetensors.ts`
`4057401a1a3f8ee51cd7884d9af439e6f001bad0`, `staticClient/index.ts`
`e1e32eb14c742ce931c5bfbdd3f023463b6b12eb`, `geo/weights.py`
`18c910ca351e8b562503f5102dcc5f1967d66b0a`, `cache/store.py`
`fd015d631c294e8b69f3af07d3d3810ab763e0bd`, `geo/finetune.py`
`1baa33c843ff365899df02eeee8b74c5238a3a49`, `geo/jobs.py`
`aa3eb40e5ff3ba9650424ab7bf8e831aeb89029e`). All probes (`p1.py`, `p2.py`, `coerce.py`,
`t1.ts`) and the scratch cache directory live in the session scratchpad and are deleted.
The backend on :8020 was stopped. Ports 8000/5173/4173 were never bound. No secrets: the
probes contain generated nonsense words, content hashes and public-domain corpus text only.
