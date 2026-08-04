# Fix 007 — Geometry Lab, round 5 (agent "GEO ROUND 5")

Date: 2026-08-04. Branch `main`. Charter: verification round 4's F1, F2, F3 and the geo
half of item 6 (`notes/agent-reports/verify-007-round3.md`). Two sibling agents share the
tree; every file I touched is inside my slice, with two deliberate exceptions noted below.
No dev stack was started; `npm run test:e2e` was not run.

| Task | Finding | Status |
|-|-|-|
| 1 | F1 — the vocabulary substitution reached through a store miss | **fixed**, both stacks |
| 2 | F2 — a persisted format moved with no `SCHEMA_VERSION` bump | **fixed**, both stacks |
| 3 | F3 — malformed vocabularies diverge and 500 | **fixed**, both stacks |
| 4 | item 6 (geo half) — `_as_int` coercion, unbounded seed | **fixed**, plus two more of the same class |

Every fix has a test that fails before it and passes after, and every guard was killed by
mutation (§ Teeth). Mutation edits were made with `cp` backups and restored by `cp`, never
`git checkout` — a sibling agent shares this working tree.

---

## TASK 1 — F1: a store miss is not "this model reads under the shipped words"

`GET /api/geo/tokenize` answered **200** for a token the store does not have, labelling the
user's ids with Alice in Wonderland's words, while `GET /api/geo/trace` answered **404** for
the identical request. The cause was one function returning `None` for two different facts:

```python
payload = load_weight_set_vocab(weights_token, store=store)   # None on a MISS *and* on
return GeoTokenizer.from_json(payload) if payload else get_tokenizer()   # "no own vocab"
```

**Fix.** `geo/weights.weight_set_entry` is now the single place a token is resolved, and it
raises. `load_weight_set`, `load_weight_set_vocab`, `weight_set_owns_vocab` and therefore
`tokenizer_for` all go through it, so `None` means exactly one thing. The browser mirror is
`GeoEngine.tokenizerFor`, which now calls `resolveWeightSet(token)` instead of falling back
to `this.tokenizer`.

Routes checked for the same hole: `/trace`, `/vector_field`, `/weights` (GET and POST),
`/finetune`, `/train_scratch`, `/model` (GET and POST) all resolve through
`resolve_weight_set` or `tokenizer_for` and now 404 consistently. `/tokenize` was the only
one that did not.

Side effect worth stating: the token strip's own probe (`viz/geo/vocab.verifyVocab`)
compares against the bundled CANONICAL table, so under the old fallback it **succeeded** and
the tab reported the vocabulary *verified* for a model whose words it had never seen. It now
fails closed and labels fall back to ids.

## TASK 2 — F2: the identity change was a format change, and now says so

`SCHEMA_VERSION` **14 → 15** (`config.py`), with the reason in the constant's comment. A
pre-change `geo-weights-*` entry is keyed by a hash of the weights ALONE while carrying a
word list of its own; this build cannot check one against the other, which is the whole
reason it must not read them. Consequences measured in round 4 and reproduced as tests:
the user's model became unsaveable (`export_bundle` re-hashed it and accused the file of
being corrupt) and `train_canonical` could die on a conflicting-claim refusal at startup.

Refused, not migrated: a v14 entry's vocabulary is precisely the thing the identity change
made unverifiable (the store deduplicated on a weights-only hash and kept the first word
list it saw), so migrating one would carry a possible substitution forward under a fresh
digest.

The miss is **explained**. `geo/weights._missing_entry_error` asks the new
`CacheStore.stale_schema_version(key)` whether a sidecar exists at another version and, if
so, says the format moved, that nothing was deleted, and that a saved model FILE still
opens — instead of "unknown (never minted here, or evicted)", which is neither what happened
nor what to do.

Browser side, the same defect with the opposite failure mode: `restorePersistedSets`
**deleted** a refused payload on boot, silently. Now:
* `GeoEngine.importWeightSet` records WHY it refused each payload (`refusedSets`), and
  detects the pre-identity shape by hash — such a payload hashes to `weightsToken(ws)`, the
  token this build gives the same weights with no word list, while declaring one.
* `resolveWeightSet` raises that reason, so the explanation reaches the user at the moment
  the app reaches for the model.
* `restorePersistedSets` **keeps** the payload. Deleting it destroyed a trained model with
  no account of it; keeping it makes the explanation reproducible across reloads, and the
  entries stay bounded by the existing `MINTED_SETS_CAP` LRU.
* The sessionStorage key stays `:v2` **deliberately** (round 3 bumped it to `:v2` because
  pre-fix payloads were then *indistinguishable*; these are distinguishable by hash, and
  bumping the key would only make the models invisible again).
* `GeometryLab.svelte` now shows the refusal instead of switching to the shipped model in
  silence (`data-testid="geo-model-lost"`), and the token-strip fetch heals the dead token
  rather than swallowing the 404.

`importWeightSet` also validates a payload whose token is already loaded (it used to return
`true` unread — round 4's F8), which is what lets a contradictory entry be classified at all.

## TASK 3 — F3: one vocabulary format, one answer in both stacks

`GeoTokenizer.from_json` was three unguarded lines; seven malformed `vocab` blocks reached
`POST /api/geo/model` as untyped **500**s whose whole message was a Python exception string.
It now refuses each with a typed `InvalidParamError` (400) naming the cause, and:

* **`specials` is validated, not ignored.** `"specials":{"<unk>":5}` loaded 200 in Python
  and was refused by TS. Both refuse now, with the same sentence.
* **The `tokens` shape is refused in both.** It is the static site's ASSET shape; a model
  file carries `geo-tokenizer-v1`/`words`. TS accepted it (via the permissive asset loader)
  and Python 500'd. `importBundle` now uses the new strict
  `GeoTokenizer.fromModelVocabJson`, an exact mirror of Python's `from_json`;
  `fromVocabJson` stays permissive for the shipped asset, which is its actual job.

## TASK 4 — item 6, geo half (and two more of the same class)

* `routes_geo._as_int` raises instead of coercing — same rule and wording as
  `routes_lex._as_int` (duplicated, not shared, so the two tabs' route modules stay
  independent). `7.0` is still accepted as 7, because JSON cannot express the distinction
  and the TS engine reads it as 7.
* `_form_int`: multipart fields are strings on the wire and always were, so the upload form
  parses strictly (base 10, ASCII digits) and is then held to the same rule. Without this
  the strict rule would have broken `.txt` uploads.
* `_as_float` refuses non-finite: `lr: Infinity` passed `lr > 0`, started a job, and turned
  every parameter into NaN.
* `_as_seed` + `geo/config.MAX_SEED = 2**53 - 1` on `POST /api/geo/train` (the same bound
  `/api/lex/train` enforces, defined locally rather than imported across tabs).
  `{"seed": 9007199254740993}` used to answer 202 and echo back a number the browser reads
  as ...992.
* **Same coercion found elsewhere and fixed:** `geo/weights.build_weight_set` did
  `int(edit.get("seed", 0) or 0)`. A seed picks WHICH preset matrix you get, so `1.5`
  silently returned the seed-1 matrix; `-1` reached `np.random.default_rng` as an untyped
  500; `Infinity` was an `OverflowError` 500. Now `_edit_seed` (typed 422, range 0..MAX_SEED).
  Its TS mirror `editSeed` replaces `Math.trunc(Number(edit.seed ?? 0)) || 0`, which mapped
  Infinity/NaN/-1 to 0 — swallowing the static build's own "that seed is not shipped" refusal.
* `layer: true` passed `isinstance(layer, int)` and edited layer 1; booleans are refused in
  both stacks now.

---

## Tests (all real; no mocks, no fixtures standing in for behaviour)

Backend — `tests/contract/test_api_geo_params.py` (new, 45 cases):
`test_finetune_refuses_a_steps_value_that_is_not_an_integer` (11 params),
`test_finetune_accepts_a_json_float_that_is_a_whole_number`,
`test_finetune_refuses_a_non_finite_learning_rate` (3),
`test_finetune_multipart_still_parses_its_string_fields`,
`test_train_scratch_refuses_a_non_integer_epochs`,
`test_train_refuses_a_seed_javascript_cannot_read_back` (4),
`test_train_refuses_a_seed_that_is_not_an_integer` (5),
`test_weight_edit_refuses_a_seed_that_would_select_another_matrix` (7),
`test_weight_edit_refuses_a_boolean_layer`, `test_a_real_edit_still_works`,
`test_model_upload_refuses_a_malformed_vocabulary_with_a_typed_400` (7),
`test_model_upload_refuses_a_vocabulary_whose_specials_are_not_ours`,
`test_model_upload_refuses_the_sites_tokens_shaped_export`,
`test_a_real_model_file_still_loads`.

Backend — `tests/integration/test_geo_derived_vocab.py` (appended):
`test_an_unknown_token_is_unknown_to_tokenize_too`,
`test_tokenizer_for_refuses_an_evicted_model_instead_of_relabelling_it`,
`test_a_cache_from_before_the_identity_change_says_what_happened`,
`test_a_cache_from_before_the_identity_change_does_not_wedge_the_lab`.
The pre-change entry is written at schema version **14 literally**, not `SCHEMA_VERSION - 1`
— the cases are about the caches that exist in the wild, and a relative version would have
passed against the old build.

Frontend — `tests/unit/geoDerivedVocab.test.ts` (appended, 4 describes):
"refuses to tokenize under an unknown token instead of using Alice's words",
"still answers for a model it does have, with that model's own words",
"names the format change for a payload written under the weights-only identity",
"keeps saying 'evicted' for a token that really was never here",
"distinguishes a tampered payload from an out-of-date one",
"refuses a vocabulary whose declared specials are not the ones we use",
"refuses the site's `tokens`-shaped asset export inside a model file",
"gives a typed refusal for every malformed vocabulary shape",
"still loads a real file",
"refuses a non-integer, non-finite or negative seed instead of picking seed 0",
"accepts the seeds that really exist".

Frontend — `tests/unit/staticClient.test.ts` (appended, through the REAL client and the
real exported assets): "refuses an unknown token instead of tokenizing it under the shipped
word list", "keeps a pre-identity persisted payload and explains why it is not loaded".

**Two existing assertions were changed, both strengthened, neither weakened.**
`geoEngine.test.ts:499` and `geoDerivedVocab.test.ts:240` asserted the refusal with
`/unknown/` — the evicted-token wording, which passed while a trained model was being erased
with no account of why. Each now asserts the refusal AND the sentence the engine owes the
user (`/could not be restored/` plus the specific reason). No `expect(` was removed anywhere.

## Teeth — 11 mutations, 11 killed

| Mutation | Killed by |
|-|-|
| `load_weight_set_vocab` returns `None` on a store miss | 2 integration cases |
| `SCHEMA_VERSION` back to 14 | both pre-identity-cache cases |
| `from_json` back to the 3-liner | 9 contract cases |
| `_as_int` back to `int(value)` | 7 contract cases |
| `_as_seed` → `int(body.seed)` | 9 contract cases |
| `_edit_seed` → `int(... or 0)`, bool layer allowed | 8 contract cases |
| TS `tokenizerFor` fallback restored | 3 vitest cases |
| TS legacy-identity classification removed | 2 vitest cases |
| `importBundle` → `fromVocabJson` | 1 vitest case |
| TS `editSeed` → `Math.trunc(Number(...)) \|\| 0` | 1 vitest case |
| `restorePersistedSets` deletes refused payloads again | 1 vitest case |

## Suites

Final run, after the last edit, with two sibling agents also writing to this tree:

* Backend: `pytest -q` → **620 passed** (2:54); `ruff check src tests` → all checks passed;
  `black --check src tests` → 86 files unchanged.
* Frontend: `npx vitest run` → **809 passed | 1 skipped (33 files)**; `npm run check` →
  **0 errors, 0 warnings** (1180 files); `npm run build` → ok.

Two transient red states seen mid-session, both in a sibling's slice and both green by the
final run — recorded rather than dismissed, since the rule here is that an error is reported
when it is encountered:
* `svelte-check`: 2 × `Expected 2 arguments, but got 1` in `tests/unit/wordClasses.test.ts`.
* `pytest`: 5 failures in `tests/unit/test_arch_word_classes.py` /
  `test_arch_vacancy_align.py`, e.g. `AssertionError: 'don‿t': U+203F undertie, general
  category Pc … assert [] == ['don‿t']` — the F4 joiner-class work in progress.

## Files outside my slice that I touched, and why

* `code/backend/src/llm_geometry/config.py` — the `SCHEMA_VERSION` bump the charter asks for.
* `code/backend/src/llm_geometry/cache/store.py` — one additive read-only method,
  `stale_schema_version`, so the geo layer can tell "the format moved" from "evicted".

## Also observed, NOT fixed (not mine)

* A sibling agent edited `code/frontend/src/lib/geoEngine/tokenizer.ts` mid-session — the
  round-4 F6 fix, `code < 0x80` → `code < 0x7f` in `canonicalVocabJson` (Python's
  `ensure_ascii` escapes DEL). That file is inside my slice by the charter's list and I had
  already added `fromModelVocabJson`/`validateSpecials` to it, so the two changes are in one
  file and cannot be committed separately. **My commit therefore carries their F6 fix, and
  the four green files that test it** (`tests/unit/geoCanonicalVocab.test.ts`,
  `tests/fixtures/geo-canonical-vocab.json`, `tests/unit/test_geo_canonical_vocab.py`,
  `scripts/export_geo_canonical_vocab.py`) — committing the source half without its tests
  would have left `main` with an untested serialization change. Authorship is stated in the
  commit message; nothing of theirs was edited by me.
