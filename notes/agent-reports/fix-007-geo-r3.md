# Fix 007 — Geometry Lab round 3 (agent "GEO ROUND 3")

Date: 2026-08-04. Scope: `notes/agent-reports/verify-007-geo-vacancy.md` findings F1, F2,
F3, F4, F9, F10 and the lows. No server was started, stopped or restarted;
`npm run test:e2e` was not run.

Every claim below is a command I ran and its verbatim output. Where a fix is claimed, the
test that proves it is named, and the mutation that kills that test is named too.

---

## TASK 1 (critical) — the vocabulary substitution, third path: the store's dedup

### What the defect actually was

`save_weight_set` deduplicated on a content hash that covered the **weights only**, and
wrote its metadata inside `if store.get(key) is None:` — first-write-wins. So for two
models with identical weights and different word lists, whichever vocabulary was cached
first won forever. The TS engine's `registerScratchModel` resolved the same collision by
**overwriting** — last-write-wins — so the two stacks disagreed about *which* of the two
models had been corrupted.

The fix is to the **identity**, not to the caching policy: two weight sets with the same
numbers and different vocabularies are two different models, and a hash that says they are
the same is the root cause.

### The change

- `geo/weights.py` — `weights_token(ws, vocab_json=None)` now hashes the name-sorted
  float32 bytes, then (for a model with a word list of its own) a fixed separator
  `b"\x00geo-vocab-v1\x00"` and the canonical vocabulary JSON. `vocab_json=None`
  reproduces the previous hash **byte for byte**, so `checkpoint_id` never moved.
- `geo/weights.py` — new `own_vocab_json()`: a word list identical to the shipped one is
  *not* an own vocabulary (otherwise the canonical model would get two different tokens
  depending on whether it arrived as the checkpoint or through a file).
- `geo/weights.py` — `save_weight_set` normalizes `(vocab_json, owns_vocab)` and, on a key
  that already exists, **reconciles**: a second write that disagrees about the vocabulary
  claim raises `InvalidParamError` instead of silently keeping the first.
- `geo/bundle.py` — `export_bundle` re-hashes (weights + vocabulary) and refuses to write a
  file whose token would not name the stored model; `import_bundle` validates the
  vocabulary **before** the token check (the vocabulary is one of the hash's inputs) and
  hashes the *canonical* re-serialization of what the file carried, so identity cannot
  depend on a writer's key order.
- `geo/finetune.py`, `geo/jobs.py` — the fine-tune cache key uses the base's **identity**
  (`base` itself when it is a token), not a re-hash of its weights, which would have let
  two same-weights/different-words models share one cache entry.
- `lib/geoEngine/weights.ts` — `weightsToken(ws, vocabJson?)`, the same bytes.
- `lib/geoEngine/index.ts` — `ownVocabJson` / `tokenFor` / `ownedWordsFor`; every mint
  (`postWeights`, `registerFinetunedWeights`, `registerScratchModel`, `importBundle`,
  `importWeightSet`) computes the token over (weights, word list). `registerScratchModel`
  can no longer overwrite another model, because it can no longer collide with one.

A side effect worth naming: the identity hash closes the one hole the two digests could
not. A file with genuine weights, a substituted word list and `vocab_sha256` *recomputed*
over the substitute used to load with a 200 (red-team F1's crafted file). It is now
refused, because the declared `weights_token` names a model with the original words.

### Reproducing tests (backend)

`code/backend/tests/integration/test_geo_derived_vocab.py`

| test | what it pins |
|-|-|
| `test_two_models_with_identical_weights_keep_their_own_word_lists` | two vocabularies, one weight set ⇒ two tokens, two files |
| `test_loading_a_file_then_training_the_same_model_does_not_swap_the_word_list` | the verifier's end-to-end trigger (load a pre-fix file, then train) |
| `test_a_file_whose_word_list_was_swapped_after_the_fact_is_refused` | the crafted-file path, now a typed 400 |
| `test_save_weight_set_refuses_a_conflicting_vocabulary_claim` | the dedup reconciliation |
| `test_the_token_covers_the_vocabulary_byte_for_byte` | cross-language golden (same two constants pinned in TS) |

### Before / after

Mutant applied to `geo/weights.py` (`vocab_json = None` at the top of `weights_token` — the
pre-fix identity, with the rest of the fix in place):

```
=== MUTANT: pre-fix identity (weights only) ===
FAILED tests/integration/test_geo_derived_vocab.py::test_two_models_with_identical_weights_keep_their_own_word_lists
FAILED tests/integration/test_geo_derived_vocab.py::test_loading_a_file_then_training_the_same_model_does_not_swap_the_word_list
FAILED tests/integration/test_geo_derived_vocab.py::test_a_file_whose_word_list_was_swapped_after_the_fact_is_refused
FAILED tests/integration/test_geo_derived_vocab.py::test_the_token_covers_the_vocabulary_byte_for_byte
```

(The three other failures in that run — `test_finetune_refuses_an_all_unk_stream`,
`test_the_two_existing_gates_do_not_catch_a_baseline_run`,
`test_weights_route_never_leaks_a_bare_key_error` — were the isolated-run failures of
TASK 3, fixed separately below.)

Restored, whole file: `18 passed, 1 warning in 28.77s` (run **alone**, cold cache).

### Reproducing tests (TypeScript)

`code/frontend/tests/unit/geoDerivedVocab.test.ts`, describe block
*"a model's identity covers its word list [F1, third path]"*:

- `gives two models with identical weights and different words different tokens`
- `hashes the vocabulary exactly as the Python backend does` — pins
  `38cb99338fb6c40f022641b579a7e827` (no vocabulary, unchanged from before the fix) and
  `50246246e336794517fcc299b505659a` (with one), the same two constants the Python test
  pins. This is what makes "either build saves the same file" checkable.

Mutant: TS `weightsToken` stops appending the vocabulary (`if (false)`):

```
=== MUTANT M3: TS token stops covering the vocabulary ===
 × minted-set persistence hooks (static reload survival) [fixtures] > carries a loaded model's OWN vocabulary across the persistence hop
 × a model's identity covers its word list [F1, third path] > gives two models with identical weights and different words different tokens
 × a model's identity covers its word list [F1, third path] > hashes the vocabulary exactly as the Python backend does
 × persisted payloads that predate `ownsVocab` are refused, not decided [F2] > refuses a payload whose claim and payload disagree, in either direction
 Tests  4 failed | 60 passed | 1 skipped (65)
```

### One existing test changed (construction only, not assertions)

`geoEngine.test.ts > carries a loaded model's OWN vocabulary across the persistence hop`
built its "model file with its own word list" by **tampering**: it took a real bundle,
replaced `vocab`, recomputed `vocab_sha256`, and left `weights_token` alone. That file is
now correctly refused. The test now builds the file the way a writer does (declaring the
token the new word list gives it) and **additionally asserts that the tampered file is
refused** — every original assertion is intact, one is added.

---

## TASK 2 (high) — a pre-fix persistence payload is refused, not decided

`GeoEngine.importWeightSet` accepted a payload with no `ownsVocab` and no `vocabWords`,
decided it as "does not own a vocabulary", and `exportBundle` then wrote the corrupt file.
The whole defence was the `:v2` sessionStorage key name.

Now: a payload carrying **neither** the flag **nor** a word list is genuinely undecidable
(a fine-tune of a scratch model and a fine-tune of the shipped model are byte-identical in
that shape) and is **refused**. A payload whose claim and payload disagree in either
direction is refused. And the claim is *checked, not believed*: the token covers the word
list, so a payload cannot pair one model's weights with another's words and still hash
right. `restorePersistedSets` drops what `importWeightSet` refuses, as before.

Test: `geoDerivedVocab.test.ts > persisted payloads that predate `ownsVocab` are refused,
not decided [F2] > refuses a payload carrying neither an ownership flag nor a word list`.
It uses a payload with a **weights-only token**, which is what the pre-fix build actually
persisted — so the content-hash check passes and cannot stand in for the refusal.

Mutant: restore the pre-fix acceptance rule
(`const ownsVocab = declaredOwns ?? words !== undefined;`):

```
=== MUTANT M2: undecidable-payload refusal removed ===
     expect(fresh.importWeightSet(preFixToken, preFix)).toBe(false);
 Tests  1 failed | 55 passed | 1 skipped (57)
```

(The first shape of this test used a payload minted by the *new* build; the mutant survived
it, because the hash check caught it instead. That is recorded because it is exactly the
trap the finding is about — the test was rewritten to the faithful pre-fix shape.)

---

## TASK 3 (medium) — tests without teeth

### (a) The TS `exportBundle` refusal

New: `geoDerivedVocab.test.ts > refuses to SAVE a set that owns a word list it no longer
has`. The state is unreachable through the public API by design, so the test reaches it the
way a real session does — register the model, then remove its vocabulary (an eviction, a
stale restore) — and asserts the refusal **by its own message**
(`/its ids mean its own words rather than the shipped model's/`).

That precision matters: the first version asserted `/vocabulary/`, and the mutant SURVIVED,
because the new re-hash guard beneath it also throws with the word "vocabulary" in it. With
the message pinned:

```
=== MUTANT M1: exportBundle ownership guard removed (`if (false)`) ===
 Test Files  1 failed | 4 passed (5)
      Tests  1 failed | 70 passed | 1 skipped (72)
```

### (b) The singleton-merge boundary

New: `code/backend/tests/unit/test_lex_vacancy.py::test_the_singleton_merge_boundary_is_exactly_one_member`.
Built from the real corpus domain: a class of **one** must merge into the bare class, a
class of **two through seven** must not, and a two-member class must derange as a clean
transposition with both images still carrying `-ing`.

```
=== MUTANT: singleton-merge boundary < 2 -> < 8 ===
FAILED tests/unit/test_lex_vacancy.py::test_the_singleton_merge_boundary_is_exactly_one_member
1 failed, 358 passed, 1 warning in 94.48s        (tests/unit + tests/contract)
```

and in the other direction (the merge removed entirely, `if suffix and False:`):

```
FAILED tests/unit/test_lex_vacancy.py::test_a_singleton_suffix_class_is_merged_into_the_bare_class
FAILED tests/unit/test_lex_vacancy.py::test_the_singleton_merge_boundary_is_exactly_one_member
2 failed, 89 passed
```

The boundary is now pinned on both sides. (Before: `< 2 → < 8` passed all 357.)

### (c) The three order-dependent tests

`tests/integration/test_geo_derived_vocab.py` gained a module-scoped `canonical_ready`
fixture that calls the real `train_canonical()`; the three cases that resolve
`base="learned"` or `load_canonical_weight_set()` now depend on it.

Before (unmodified `main`, module run alone): `3 failed, 10 passed`.
After (module run alone, cold cache): `18 passed, 1 warning in 28.77s`.

---

## TASK 4 (medium) — the unk bound, and seed validation

### The unk bound is now `>=`

`geo/finetune.py`, `geo/jobs.py`, `lib/geoEngine/index.ts`, `lib/staticClient/geo.ts`: the
refusal is `unk_rate >= FINETUNE_MAX_UNK_RATE`. The message now prints the rate to one
decimal **and states the limit**, so an accepted rate and a refused one can no longer both
render as "(90%)". The constant's docstring records the measurement that justifies it
(modern financial prose: `n_tokens 73 n_unk 46`, 0.63 — ~0.27 below the bound), and the
panel prose in `viz/geo/FinetunePanel.svelte` now names the number.

Tests (`test_geo_derived_vocab.py`): `test_the_unk_bound_refuses_a_stream_that_is_exactly_at_it`
(a real 900/1000 stream, refused) and `test_the_unk_bound_still_accepts_a_stream_just_below_it`
(899/1000, a real fine-tune that runs) — the bound is pinned from both sides.

### `POST /api/lex/vacancy` raises instead of truncating

`api/routes_lex.py::_as_int` no longer calls `int(value)`. It accepts a JSON integer (and
an integral float such as `7.0`, which is what the TS engine reads as 7) and refuses
everything else with a typed `InvalidParamError`: `1.5`, `"7"`, `true`, `false`, `null`,
arrays, objects, and the non-finite `Infinity` / `-Infinity` / `NaN` that used to escape as
an untyped 500 (`OverflowError: cannot convert float infinity to integer`).

Also fixed on the same theme (reported by the verifier under F10, not in my charter's
numbered list, fixed because it is the same wrong-answer path): `POST /api/lex/train`
bounded the *vacancy* seed it forwards and left its own top-level `seed` unbounded —
`12345678901234567890` was accepted with a 202 and echoed back as a number JavaScript
cannot read. It now carries the same `MAX_SEED` bound.

New file: `code/backend/tests/contract/test_api_lex_params.py` (20 cases, real HTTP).

```
=== MUTANT: _as_int back to int(value), train-seed bound off ===
FAILED ...[1.5]  FAILED ...[2.0000000001]  FAILED ..."7"  FAILED ...[true]  FAILED ...[false]
FAILED ...test_a_non_finite_seed_is_a_typed_400_not_a_leaked_overflow[Infinity]
FAILED ...[-Infinity]
FAILED ...test_the_training_seed_carries_the_same_bound_as_the_vacancy_seed
8 failed, 12 passed
```

Restored: `20 passed`.

Not changed, deliberately: the static client's `asInt(body.seed, "seed", 0)`
(`lib/staticClient/lex.ts:519`, the other half of F10) belongs to another agent in this
campaign and I did not touch it. **Handoff:** that default silently substitutes `0` for a
`null` seed where the backend now returns a typed 400.

---

## Handoffs — wording for files owned by other agents

### 1. `specs/007-vacancy-transform-field/architecture.md` (agent A)

The same-inflection guarantee at **§8.3, the paragraph beginning "Because each class is
permuted onto itself"** is unconditional and is falsified 25 lines later by the
singleton-merge exception (2.74 % of vacated words on the shipped passages). Suggested
replacement for that sentence:

> Because each class is permuted onto itself, every image is a real domain word, and —
> **except for the merged singleton classes described below** — it carries the same
> inflection as the word it replaces, so the morphology a reader parses (`-ed`, `-ing`,
> `-'s`) is as intact in the swap arm as in the nonce arm. Measured over the six shipped
> Architecture passages at `p = 1, seed = 0`: 767 vacated words, 0 outside the domain, 21
> (2.74 %) in a different suffix class, every one of them from a merged singleton.

The exception paragraph itself ("**A class of one is merged into the bare class.**") is
accurate and needs no change; the boundary it describes is now pinned by
`test_the_singleton_merge_boundary_is_exactly_one_member`.

### 2. `code/frontend/src/viz/info/InfoTab.svelte` (agent A)

Two paragraphs state the vocabulary guarantee in contradictory terms, and — as of this fix
— **both are now out of date**, because the content hash covers the word list.

Current (≈ line 555):

> A file with real weights and a tampered word list would silently mislabel every point on
> the sphere, so it is refused instead.

Current (≈ line 562):

> The digests cannot police this on their own — a writer that substituted the shipped word
> list would also compute `vocab_sha256` over the substituted list, and the file would
> verify.

Suggested replacement for the second one (the first sentence is now true and can stand):

> The two digests could not police this on their own: a writer that substituted the shipped
> word list would also compute `vocab_sha256` over the substituted list, and the file would
> verify. So the content hash covers the **word list as well as the weights** — a model's
> identity is its numbers *and* what its ids mean — and a file whose vocabulary was swapped
> after the fact no longer hashes to the model it names. Where a vocabulary cannot be
> recovered at all, saving is **refused** rather than completed with the wrong words.

Please also add, in the same paragraph or a footnote, the limit that remains: a file whose
author recomputes *every* digest is self-consistent by construction and describes whatever
model it says it does. What is now impossible is our own writer producing one.

---

## Suites (full, local, macOS, Python 3.10)

Run after every change, with every mutation restored (`shasum -c` on each mutated source):

```
code/backend   $ pytest -q          532 passed, 1 warning in 157.23s (0:02:37)
code/backend   $ ruff check .       All checks passed!
code/backend   $ black --check .    82 files would be left unchanged.
code/frontend  $ npm run check      COMPLETED 1173 FILES 0 ERRORS 0 WARNINGS 0 FILES_WITH_PROBLEMS
code/frontend  $ npx vitest run     Test Files 29 passed (29) · Tests 596 passed | 1 skipped (597)
code/frontend  $ npm run build      ✓ built in 3.19s
```

Re-run in full after the commits landed (the tree also picked up two other agents'
commits in the meantime): `532 passed, 1 warning in 154.45s`, ruff and black clean,
`0 ERRORS 0 WARNINGS`, `596 passed | 1 skipped`, build ✓.

**SC-703 and the TS↔Python differential suite are undisturbed.** `tests/unit/vacancy.test.ts`
(76 cases) and `tests/unit/vacancyGolden.test.ts` are green, and the §10 numbers they print
are unchanged: `stemsTotal=1676`, `tokensVacated(p=1)=8125`, `domainTypes 2233/1940`,
`bijective=true`, `imageSize=2233` at both seeds. Nothing I changed touches the transform —
the only vacancy-side edits are a new unit test and the route's parameter coercion.

## Observed while working, not fixed (out of all four tasks)

`geo/scratch.py::train_scratch` stores only `{"embedding": …}` beside its cache entry and,
on a cache hit, returns the recorded `weights_token` without re-registering the weight set
(compare `train_canonical`, which calls `save_weight_set` again on a hit "in case it was
LRU-evicted"). If the weight artifact is evicted while the scratch entry survives, the
returned token resolves to `NotFoundError`. That is a **loud** failure, not a wrong answer,
which is why I left it: fixing it changes what the scratch cache entry stores, and nothing
in this charter touches that. Recorded here rather than dropped.

## What I did NOT do

- I did not touch `viz/arch/*`, `arch/*`, `staticClient/arch.ts`, `byteSpans.ts`,
  `specs/007-*/architecture.md`, `specs/007-*/spec.md`, `viz/info/InfoTab.svelte`,
  `viz/lex/*`, `staticClient/lex.ts`, `stores.ts`, `App.svelte`, `README.md`, or
  `specs/006-*/contracts/api-lex.md`.
- I did not run `npm run test:e2e`, and I did not start, stop or restart any server.
- `lib/lexEngine/vacancy.ts` is mine by charter but was held modified by another agent for
  the whole of my run, so I made no change to it. Nothing in this fix required one: the
  singleton-merge boundary is identical in both stacks and the TS↔Python differential suite
  covers their agreement.
