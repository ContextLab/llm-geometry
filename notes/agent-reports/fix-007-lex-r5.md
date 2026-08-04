# Fix 007 — round 5, Lexicon Lab slice (agent "LEX ROUND 5")

Date: 2026-08-04. Branch `main`, from `ee351ce`. Charter: the unassigned halves of items 5
(`Object.hasOwn` sweep) and 6 (string→number coercion) of
`notes/agent-reports/verify-007-round3.md`. Two sibling agents were editing the tree
concurrently; every file listed as theirs was read but **never written**.

`npm run test:e2e` was not run. No dev server was started.

---

## TASK 1 — the lookup sweep

### What was fixed

#### T1a. `staticClient/safetensors.ts` — a remote header could replace the tensor map's prototype
**Severity:** medium (was the residual half of the verifier's F4/item-5 "low")
**Where:** `code/frontend/src/lib/staticClient/safetensors.ts:305` (was
`const tensors: Record<string, TensorEntry> = {};`)

The keys of that map are tensor names chosen by a **remote file**.
`tensors["__proto__"] = entry` on an ordinary object literal does not create a property:
it invokes `Object.prototype`'s `__proto__` setter and replaces the map's prototype with
the attacker's entry. Two consequences, neither of which throws:

1. the declared tensor **silently vanishes** from the map — `Object.keys` returned `[]`;
2. every field of that entry (`dtype`, `shape`, `data_offsets`) became visible **on the map
   itself**, so `header.tensors.dtype` resolved to a string a remote host supplied. That is
   exactly what `staticClient/arch.ts:829` reads, with a truthiness test.

Round 3's `Object.hasOwn` in `readWindow` blocked the *consumer*; this closes the
*producer*. The map is now `Object.create(null)`, which has no `__proto__` setter and no
inherited keys at all. Verified by execution: before, `Object.hasOwn(header.tensors,
"__proto__")` is `false` and `Object.keys(header.tensors)` is `[]`; after, the tensor is a
normal own key and `readWindow("__proto__", …)` returns its real F32 values.

**Test:** `tests/unit/staticSafetensors.test.ts` →
`a remote header cannot reach through the tensor map's prototype` (2 tests). Builds a
**real** safetensors byte stream (8-byte LE header length + JSON header + F32 payload) and
serves it over a transport that honours `Range` like the CDN. **2 failed before, 12 passed
after** (the 10 real-CDN tests in that file are unaffected).

Note on the test itself: the header is written as **text**, not as an object literal —
`{ __proto__: … }` in JavaScript source sets the *literal's* prototype and
`JSON.stringify` then emits `{}`, so an object literal cannot express the file this test is
about. That cost one debugging round and is documented in the test.

#### T1b. `staticClient/index.ts:115` — `DISPLAY_NAMES[m.model_id] ?? m.model_id`
**Severity:** medium
`??` reads as a safe default and is not one: `DISPLAY_NAMES` is an object literal, so
`DISPLAY_NAMES["constructor"]` is the `Object` **function** — truthy, not nullish — and the
fallback never fires. A model whose id is any `Object.prototype` member would be listed in
the Architecture tab's model menu under the stringification of a JavaScript builtin
instead of under its own id. Now `Object.hasOwn`.

**Test:** `tests/unit/staticLexCoercion.test.ts` →
`a data-supplied key never resolves through a lookup table's prototype` (6 tests). Serves
the **real committed** `static-data` directory and renames one model's `model_id` in
`index.json` on the way through (every per-model asset path is keyed by `slug`, untouched).
**5 failed before, 6 pass after**, plus a guard that `gpt2` still gets its curated label.

#### T1c. `staticClient/lex.ts:1567` — `"shape" in entry` / `"data" in entry` on file-supplied objects
**Severity:** low (not exploitable from `JSON.parse`, which defines `__proto__` as an *own*
property rather than invoking the setter — so no inherited `shape`/`data` is reachable
today). Converted to `Object.hasOwn` anyway so the rule in the bundle loader is uniform and
no later reader has to work out which key names happen to be safe. Same commit adds a
**typed** refusal for a non-array `shape`, which previously escaped as
`(5).map is not a function` — an untyped `TypeError` the file dialog had nothing to print.

**Test:** `staticLexCoercion.test.ts` →
`refuses a weight entry whose shape is not a list, with a typed error`.

#### T1d. `staticClient/hfDatasets.ts:129,179` — `first[preferred]` and `row[column]`
**Severity:** low, **hardening only — I could not construct a failing-before test, and say
so rather than claim one.** Column names come out of a remote JSON document, but
`JSON.parse` cannot produce an object that *inherits* `text`/`content`, and `column` itself
comes from `Object.entries` (own keys only). Both lookups are now `Object.hasOwn`-guarded
because the Python half (`dict`) has no prototype chain and the two halves should not
differ in kind. **No behaviour change is demonstrable on any input the dataset viewer can
return**, and no test asserts one.

### Every lookup/coercion site found, and the judgement on each

Files I own:

| Site | Shape | Judgement |
|-|-|-|
| `staticClient/safetensors.ts:305` | remote key → `{}` assignment | **FIXED** (T1a) |
| `staticClient/safetensors.ts:85,165,173` | `Object.hasOwn(BYTES_PER/tensors, …)` | already correct (round 3) |
| `staticClient/index.ts:115` | `DISPLAY_NAMES[data key] ?? …` | **FIXED** (T1b) |
| `staticClient/lex.ts:1567` | `"shape" in entry`, `"data" in entry` | **FIXED** (T1c), not exploitable |
| `staticClient/lex.ts:509,1615-1616` | `Object.hasOwn(MINT_STRATEGIES/shapes/supplied, …)` | already correct (round 3) |
| `staticClient/lex.ts:720-722` | `shapes[name]`, `weights[name]` | safe — `name` from `Object.keys(weights).sort()` |
| `staticClient/lex.ts:1225` | `(body as Record)[control]` | safe — `control` from a literal `as const` array |
| `staticClient/lex.ts:1005` | `sizes[name]` | safe — `name` from the `DOLCH_ORDER` const |
| `staticClient/hfDatasets.ts:129,179` | remote column name | **HARDENED**, no reachable defect (T1d) |
| `staticClient/hfDatasets.ts:77,97` | `RETRY_DELAYS_MS[attempt]` | safe — array, bounded loop index |
| `staticClient/assets.ts` `files`/`presets` | `Record<string, …>` | never indexed anywhere in the tree |
| `staticClient/jobs.ts` | job registry | safe — a real `Map`, no prototype chain |
| `staticClient/transformersRuntime.ts` | all `[…]` are array indices | safe |
| `staticClient/byteSpans.ts:33` | `bs.includes(b)` | safe — array (file currently being edited by the arch agent; not touched) |
| `lib/stores.ts:25` | `VIEWS.includes(h)` | safe — array `.includes`, no prototype path |
| `viz/lex/provenance.ts:188` | `Object.hasOwn(PROVENANCES, declared)` | already correct (round 3) — and it is the **only** entry point, so every `Record<Provenance, …>` lookup below is safe |
| `viz/lex/ModelFile.svelte:119,180` | `DEFAULT_NOTES[provenance]`, `LOADED_AS[claim.provenance]` | safe — validated upstream by `provenanceFromMetrics`/`provenanceOf` |
| `viz/lex/ForwardPassPanel.svelte:50`, `SamplePanel.svelte:51` | inline `{…}[provenance]` | same — safe for the same reason |
| `viz/lex/BudgetPanel.svelte:136,143` | `BUDGET_LABELS[name]`, `sizes[name]` | safe — `name` iterates the `DOLCH_ORDER` const |
| `viz/lex/LexWeightLab.svelte:118,130,145,164` | `shapes[name]`, `active?.[name]` | safe — `name` is internal state pinned to `weightNames(cfg)` by the effect at `:111` |
| `viz/lex/SamplePanel.svelte:62` | `SPECIAL_TOKENS[id]` | safe — const array, const ids |
| `viz/lex/VacancyPanel.svelte:697` | `"seed" in parsed` | safe — TypeScript discriminated-union narrowing over a locally constructed literal, not a validator on data |
| `routes_lex.py` (all `in`/`.get`) | Python `dict` | safe by construction — no prototype chain. `getattr(params, spec.name)` at `:658` reads a dataclass's **own** `fields()`, not a wire name |

Files I do **not** own — reported, not edited:

| Site | Shape | Judgement |
|-|-|-|
| `lib/lexEngine/vacancy.ts:386` | `METER_FEET[foot]` guarded by `=== undefined` | **STILL BROKEN.** `METER_FEET["constructor"]` is the `Object` function, so `meterScore` returns **`0`** with no throw where `"bogus"` throws correctly. Latent (`foot` is not a wire parameter today) but it is the wrong-number-not-throw form. **The charter both told me to fix this and put `lexEngine/vacancy.ts` on the do-not-touch list; I obeyed the do-not-touch list. It needs one line: `Object.hasOwn(METER_FEET, foot)`.** |
| `staticClient/arch.ts:829` | `const entry = header.tensors[c]` truthiness test | **Source now closed** by T1a (a null-prototype map has nothing to inherit), but the test is still the wrong shape and should be `Object.hasOwn`. Arch agent's file. |
| `lib/geoEngine/weights.ts:194` | `(fixtures.embedding \| fixtures.square)[preset]` then `table?.[String(seed)]` | Safe **by luck**: `preset="constructor"` yields the `Object` function, and `Object["0"]` is `undefined`, so the typed refusal still fires. Worth an `Object.hasOwn` anyway. Geo agent's file. |
| `staticClient/geo.ts:73-88` | `all[token]` over a `JSON.parse`d sessionStorage map | Safe — tokens are 32-hex content hashes, and `JSON.parse` writes `__proto__` as an own property rather than invoking the setter. Reviewed, no action. |

No `Object.keys(...).includes` and no `hasOwnProperty(` call sites exist anywhere in
`code/frontend/src`.

---

## TASK 2 — string→number coercion

### T2a. `staticClient/lex.ts` `asInt` / `asFloat` — the `0x10` finding, and its class
**Severity:** high (this is the public site's path)
**Where:** `code/frontend/src/lib/staticClient/lex.ts:783-799` (12 call sites)

`Number(value)` was the whole rule, and `Number` is the widest parser in the language.
Verified against this build before the fix, verbatim from the verifier: `"7"→7`, `true→1`,
`false→0`, `null→0`, `[]→0`, `[7]→7`, `""→0`, `" 7 "→7`, **`"0x10"→16`**. I found the same
for `"0b101"→5`, `"0o17"→15`, `"1e3"→1000`, `"+7"→7`, `"\n7\n"→7`, `"   "→0` and
`new Date(0)→0`.

Both helpers now take **only a JSON number**, matching `routes_lex._as_int` exactly:
`undefined` takes the default, an explicit `null` is refused (Python's `_as_int(None)`
raises, so returning the default here made `{"seed": null}` two different runs), `7.0` is
accepted as `7` because JSON cannot express the int/float distinction, and a fractional or
non-finite value is refused rather than truncated.

One further defect found while rewriting: the refusal message used `JSON.stringify(value)`,
which renders `NaN`, `Infinity` and `-Infinity` all as the string **`null`** — so the three
numbers hardest to notice were reported back as the one value nobody sent. A `show()`
helper now prints numbers with `String`.

### T2b. `staticClient/lex.ts` — `POST /api/lex/train`'s own seed was unbounded
`Number.isInteger(1e300)` is **true**, so `asInt` alone let a seed through that no RNG here
can use and that Python answers with a typed 400 (`abs(seed) > MAX_SEED`). The seed is
echoed back in the job result, so one request body documented two different runs. The
backend's bound is now mirrored at the same place.

### T2c. `routes_lex.py::_as_float` — `_as_int`'s rewrite stopped one type short
**Severity:** high
`_as_float` was bare `float(value)`. Measured against the running app:
`{"p": Infinity}` → `500 InternalError: Out of range float values are not JSON compliant`
(the exact untyped leak the `_as_int` rewrite removed one type up). `NaN` is quieter and
worse: every `<`/`>` against it is `False`, so `if lr <= 0: raise` and
`if weight_decay < 0: raise` both wave it through and the run diverges at step 1 with a
message about the model, for a defect in the request. `float()` also reads numeric strings
Python and JavaScript disagree about — `"٠.٥"` (Arabic-Indic) is `0.5` to Python and `NaN`
to `Number` — and `float(True)` is `1.0`.

`_as_float` now mirrors `_as_int`: bool refused, only `int`/`float` accepted, non-finite
refused, and an `int` too large for a double (`10**400`, which raises `OverflowError`)
given a typed 400 instead of a 500. Every caller reads a JSON body, never a query string,
so nothing legitimate arrives as text — checked all 9 call sites.

### T2d. `viz/lex/VacancyPanel.svelte` — the `reveal first` box, beside the seed box round 3 fixed
**Severity:** medium
`Math.trunc(Number(e.currentTarget.value))` then `Math.max(1, v)` is three silent rewrites
in one line: `2.5`→2, `1e3`→1000 **under `max="99"`**, and `0` (or anything the number
input sanitizes to `""`)→1. `reveal_after` is a boundary in the vacancy map, so a
substituted value produces a different corpus, a different `vacated_sha256` and a different
loss curve, with the box still showing what was typed. It now uses the seed box's rule
(`/^\d+$/`, range-checked, **refused** with a visible `role="alert"` sentence naming the
value still in force).

### T2e. `viz/lex/LexWeightLab.svelte` — the `randomize` seed box
**Severity:** medium
`bind:value={seed}` on `<input type="number">` hands back `null` for an empty box, so
clearing the field re-drew the tensor **at seed 0** while the note under it read
"re-drawn from its initializer at seed **null**"; `2.5` and `1e3` were passed straight to
`initWeights`; and `max="9999"` blocked nothing, exactly as it blocked nothing on the
Vacancy panel before red-team finding F4. `randomize` is reproducible only by its seed and
that note is the panel's sole record of which draw the active weights are — everything
downstream (spectrum, token cloud, sampler) then measures a model the page describes
wrongly. Now parsed as text, digits-only, 0..9999, and **refused** into the panel's
existing error surface.

### Every other string→number conversion in my files, checked

| Site | Judgement |
|-|-|
| `routes_lex.py::_as_int` | already correct (round 3); re-verified against `"٧"`, `"７"`, `2**53`, long digit strings |
| `routes_lex.py::_as_bool` | accepts `"true"/"false"/"1"/"0"` — **deliberate parity** with `staticClient/lex.ts::asBool`; both stacks agree, so left alone |
| `staticClient/lex.ts::oneOf` | delegates to `asInt`; fixed transitively |
| `staticClient/lex.ts:596` `Number(x.toPrecision(6))` | safe — round-trip of a number the code produced |
| `VacancyPanel.svelte::parseSeed` | already correct (round 3) — it is the model the two new parsers copy, including its `0x10` comment |
| `VacancyPanel:676`, `ModelPanel:248`, `ForwardPassPanel:348`, `TrainPanel:548` | `Number(e.currentTarget.value)` on `<input type="range">` — the browser guarantees a numeric string inside `[min,max]`; no free text reaches these |
| `SamplePanel:121,132,137`, `TrainPanel:537,553,557,561` | `bind:value` on `type="range"` — same |
| `LexWeightLab::onCellEdit` | already guards with `Number.isFinite` and refuses |
| `safetensors.ts:297` `Number(big)` | safe — a `BigInt` already bounded by `HEADER_SANITY_LIMIT` |
| `transformersRuntime.ts:163` `Number(v)` | safe — `number \| bigint` out of a tensor |

The two free-entry number boxes in the Lexicon tab (T2d, T2e) were the **only** two; every
other numeric control is a range slider.

---

## Tests added

| File | Tests | before → after |
|-|-|-|
| `code/backend/tests/contract/test_api_lex_params.py` (extended) | 5 new parameterized groups, 25 cases | **19 failed** → 48 passed |
| `code/frontend/tests/unit/staticLexCoercion.test.ts` (new) | 154 | **72 failed** → 154 passed (`asInt`/`asFloat`); **5 failed** → 6 passed (`DISPLAY_NAMES`) |
| `code/frontend/tests/unit/lexVacancySeed.test.ts` (extended) | 11 new | **11 failed** → 20 passed |
| `code/frontend/tests/unit/lexWeightLabSeed.test.ts` (new) | 15 | **15 failed** → 15 passed |
| `code/frontend/tests/unit/staticSafetensors.test.ts` (extended) | 2 new | **2 failed** → 12 passed |

Every "before" number was measured by temporarily restoring the old code from a `cp`
backup, running the suite, and restoring the file — never `git checkout`, because two
sibling agents share this tree. No existing assertion was weakened, skipped or deleted.

The tables in `staticLexCoercion.test.ts` are the **class**, not the examples: every form
`Number` accepts and a JSON integer is not appears once, so a future rewrite reaching for
`Number`, `parseInt`, `parseFloat` or a unary `+` fails there rather than in a browser.

---

## Full-suite results

* Frontend: `npx vitest run` → **Test Files 32 passed · Tests 803 passed | 1 skipped (804)**
* Frontend: `npm run build` → `✓ built in 3.39s`
* Backend: `ruff check .` → `All checks passed!`
* Backend: `pytest -q` → **571 passed, 1 failed**
* Backend: `black --check src/llm_geometry/api/routes_lex.py tests/contract/test_api_lex_params.py` → clean

### Failures NOT from this slice — reported, not dismissed

These are in files a sibling agent is editing right now (`git status` confirms each is
modified or added by the concurrent geo/arch work). I did not touch them, and editing them
would collide.

1. `pytest tests/integration/test_geo_derived_vocab.py::test_a_cache_from_before_the_identity_change_says_what_happened`
   — `NotFoundError: weights_token '9a12df74…' is unknown`. This is the geo agent's new
   test for verifier finding F2, against `geo/weights.py`, both mid-edit.
2. `black --check` would reformat three files, **all** sibling-owned:
   `tests/unit/test_arch_word_classes.py`, `src/llm_geometry/geo/tokenizer.py`,
   `src/llm_geometry/geo/weights.py`.
3. `npm run check` reports 2 errors, both in `tests/unit/wordClasses.test.ts:69`
   (`Expected 2 arguments, but got 1`) — the arch agent's new word-classes work. Zero
   errors in any file I own or wrote.

---

## Repo hygiene

Working tree contains only intended edits. No scratch files, no screenshots, no secrets
(the new tests contain literal parameter values and public-domain corpus text only). All
mutation backups were written to `/tmp` and deleted. Ports 8000/5173/4173 were never bound;
no server was started at all.
