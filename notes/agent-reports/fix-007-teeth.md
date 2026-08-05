# fix-007-teeth — three tests that passed while the code they pin was broken

Charter: `verify-007-round3.md` F10 (high), F11 (medium), F12 (medium) — the three round-3
mutants that survived. Each was treated as evidence about a *class* of weak assertion, not
as an isolated chore.

Protocol, applied to all three: **apply the exact mutation → confirm the existing tests pass
(the bug) → write the new test → confirm it FAILS under the mutation → restore → confirm it
passes.** Mutations were applied with `Edit`/`perl` and reverted from `cp` backups in the
session scratchpad, verified by `shasum`; `git checkout` was never used.

---

## F10 — a test that asserts on source TEXT

**Where:** `code/frontend/tests/unit/navGuard.test.ts` (was :294-310, the `readFileSync` +
regex loop); code at `code/frontend/src/viz/lex/VacancyPanel.svelte:557-560`.

### The mutation

```
-  $effect(() => {
+  {
     if (demoBusy) registerWork(WORK_ID, "the two training runs in the Lexicon Lab's transform demo");
     else releaseWork(WORK_ID);
-  });
+  }
```

De-reactifies the wiring completely: the bare block runs once at init with
`demoBusy === false`, releases an id nobody registered, and never runs again — so the tab's
headline demonstration (two real training runs) is destroyed in silence by a tab click.

### Four states

| # | State | Command | Result |
|-|-|-|-|
| 1 | mutation live, OLD tests | `npx vitest run tests/unit/navGuard.test.ts` | `Tests  15 passed (15)` — **the bug is invisible** |
| 2 | mutation live, NEW tests added | same | `Tests  2 failed \| 15 passed (17)` |
| 3 | restored (`shasum 3ba60bff84325ce37c0b6217537b8f5f3a964a4d`) | same | `Tests  17 passed (17)` |
| 4 | restored, full frontend suite | `npx vitest run` | `Test Files 34 passed (34) · Tests 815 passed \| 1 skipped (816)` |

State-2 failure output, verbatim:

```
FAIL tests/unit/navGuard.test.ts > a panel's registration survives being driven for real >
  holds a tab switch while the vacancy demonstration is really training
AssertionError: the demonstration's training runs were never registered:
  expected [] to deeply equal [ 'lex-vacancy-demo' ]
```

### What replaced the source check

A new `describe("a panel's registration survives being driven for real")` with two tests
that mount the real `VacancyPanel` in jsdom and **really train**:

* `holds a tab switch while the vacancy demonstration is really training` — clicks
  `[data-testid="lex-vacancy-demo-run"]`, asserts `pendingWork` holds `lex-vacancy-demo`,
  then drives a real tab switch (`view.set("architecture")`) and asserts it is HELD and
  names the run; then waits for both runs to finish and asserts a later switch goes through.
* `releases the work when the tab is left, so a later navigation is not held` — unmounts
  mid-run (what a confirmed navigation does) and asserts the registry empties, so it cannot
  latch.

This is possible with **no mock** because `lib/lexEngine/index.ts:263-268` says so:
"Where `Worker` does not exist — Node, and therefore the unit tests — it runs the SAME job
function inline rather than pretending to be asynchronous: one code path, no mock." Clicking
Run in the test therefore performs two genuine 40-step training runs.

**No existing test was weakened, skipped or deleted.** The `readFileSync` loop is still
there, and was made *strictly stronger*: the regex now requires the `$effect(…)` wrapper,
not just the three statements inside it. That matters because the other three panels in the
loop **cannot** be driven in jsdom (`lex/TrainPanel` constructs `new Worker(...)` directly;
the two geo panels need the backend), so for them the tightened text check is the only gate
that exists.

Verified the tightening has teeth on a panel that cannot be driven: the same bare-block
mutation applied to `src/viz/geo/TrainPanel.svelte:56-59` →
`Tests 1 failed | 16 passed (17)`, `viz/geo/TrainPanel.svelte does not register REACTIVELY
on \`busy\``. Restored; `17 passed (17)`.

### Full audit: every test in the suite that asserts on source TEXT

`grep -rn "readFileSync\|read_text()\|inspect.getsource"` over `code/frontend/tests/` and
`code/backend/tests/`. Backend: **none** — the only two `read_text()` calls
(`test_geo_derived_vocab.py:652`, `test_cache.py:56`) read cache sidecars the code just
wrote, which is data, not source.

| Site | Reads | Verdict |
|-|-|-|
| `tests/unit/navGuard.test.ts:432` | 4 panel `.svelte` sources | **the finding — fixed.** Behaviour test added for the one driveable panel; regex tightened to require `$effect(` for the other three, which have no other gate (`grep -rn "nav-hold" tests/e2e/` → **no matches**: there is no e2e coverage of the nav guard at all). |
| `tests/unit/shell.test.ts:355` | `viz/geo/GeometryLab.svelte` source | **SAME CLASS, still open.** Two tests (`are radio groups rather than tablists`, `render through SegmentedControl`) assert on markup text. I attempted to replace them with a mount: `GeometryLab` *does* mount in jsdom without throwing, but `geo-mode`/`geo-layer` are behind `{:else}` at `GeometryLab.svelte:518` — they render only in `phase === "ready"`, which needs the real backend or a real boot, so both querySelectors return `null`. Not fabricated around: reported instead. Mitigation that already exists: `SegmentedControl`'s ARIA *is* behaviour-tested by mounting it (`shell.test.ts:203`), and the source regex does still catch the control being replaced by raw buttons. What it cannot catch is a semantic change inside `SegmentedControl` reaching these two usages. |
| `tests/e2e/docs.spec.ts:480` | `lib/staticClient/arch.ts` source | **Different, weaker-but-not-vacuous class.** It *extracts a value* (`VACANCY_Q8_UNCERTAINTY_NATS`, `VACANCY_MIN_POOLED_PRESERVED`) and compares it to the rendered prose, guarded by `expect(uncertainty, "…is gone or renamed").toBeDefined()`. It pins "the documented number is the declared constant"; it does **not** verify the constant is enforced. Acceptable for its stated claim; noted. |
| `tests/unit/lexEngine.test.ts:87,157` | `lex/dolch.py`, `lex/vocab.py` sources | **Same value-extraction class, and legitimate.** It reads the Python literal (`WORD_RE = re.compile(r"…")`) to derive the expected value for a cross-language differential, with an explicit `not.toBeNull()` guard if the source shape moves. It compares values, not presence. |
| `geoGoldenAssets.ts`, `geoScratch.test.ts:50`, `lexBundle.test.ts:52`, `lexGolden.test.ts:151`, `readmeNumbers.test.ts`, `wordClasses.test.ts`, `staticVacancy.test.ts:92`, `vacancyGolden.test.ts`, `geoDerivedVocab.test.ts:74`, `e2e/static.spec.ts`, `e2e/webgpu.spec.ts` | corpora, golden JSON, `vocab.json`, `README.md` | **Not this class** — these read DATA (fixtures and shipped assets), which is what a golden test is. |

---

## F11 — key-order independence, claimed in a comment and tested nowhere

**Where:** `code/backend/src/llm_geometry/geo/bundle.py:211`; the property is stated in the
comment at `:191-194`.

### The mutation

```
-    owned = own_vocab_json(canonical_vocab)
+    owned = own_vocab_json(vocab_json)
```

An honest third-party file — same word list, different key order and indentation, with
`vocab_sha256` correctly recomputed over its own bytes — is then **refused**.

### Four states

| # | State | Command | Result |
|-|-|-|-|
| 1 | mutation live, pre-existing tests only | `pytest -q tests/integration/test_geo_derived_vocab.py` | 24 of 24 pre-existing cases pass — **the bug is invisible** |
| 2 | mutation live, new test added | same | `1 failed, 26 passed` |
| 3 | restored (`shasum 53acb130755ebceae3871ff3ff97777c33069bb7`) | same | `27 passed, 1 warning in 28.43s` |
| 4 | restored, full backend suite | `pytest -q` | `624 passed, 1 warning in 166.09s` |

State-2 failure output, verbatim:

```
llm_geometry.errors.InvalidParamError: this model file is corrupt: its weights and
vocabulary hash to 35ba27d745a8b0f8836603ad0e54f8d8 but it declares
cb6d90c290bd7a188182d67f8e536b87. Loading it would pair the wrong vocabulary with these
weights, so it is refused.
FAILED tests/integration/test_geo_derived_vocab.py::test_a_writers_key_order_does_not_change_which_model_a_file_is
```

### The new test

`test_a_writers_key_order_does_not_change_which_model_a_file_is` — builds a real model file
for a scratch-trained weight set with its own 1,000-word list, then a second file that is
byte-different and semantically identical (`json.dumps(..., indent=2)` with unsorted keys)
with `vocab_sha256` honestly recomputed, and asserts:

1. both import to the **same** `weights_token`;
2. `tokenizer_for(...)` on the imported model returns the file's own words;
3. `export_bundle(...)["vocab"]` is the **canonical** spelling, so re-saving is byte-stable.

Two guards make sure the test tests something: `assert reordered != canonical` and
`assert json.loads(reordered) == data`.

---

## F12 — the module advertised as pinning identity did not

**Where:** `code/backend/src/llm_geometry/geo/weights.py:283-299` (`own_vocab_json`);
module `tests/integration/test_geo_derived_vocab.py`.

### The mutation

```
-    if vocab_json is None:
-        return None
-    from .tokenizer import get_tokenizer  # local import to avoid a cycle
-
-    return None if vocab_json == get_tokenizer().to_json() else vocab_json
+    return vocab_json
```

One model then has two identities depending on which door it came through.

### Four states

| # | State | Command | Result |
|-|-|-|-|
| 1 | mutation live, pre-existing tests only | `pytest -q tests/integration/test_geo_derived_vocab.py` | `24 passed, 1 warning in 28.58s` — **all five named vocabulary tests green while the bug is live** |
| 2 | mutation live, new tests added | same | `2 failed, 25 passed` |
| 3 | restored (`shasum 18c910ca351e8b562503f5102dcc5f1967d66b0a`) | same | `27 passed, 1 warning in 28.04s` |
| 4 | restored, full backend suite | `pytest -q` | `624 passed, 1 warning in 166.09s` |

State-2 failure output, verbatim — the two identities, measured:

```
>       assert as_file["weights_token"] == as_checkpoint
E       AssertionError: assert '8278367380d3...0bff9eff43ca1' == 'c59a0812e8d0...8bd0d0b669aab'
E         - c59a0812e8d0405a8818bd0d0b669aab
E         + 8278367380d3f1fecfb0bff9eff43ca1
FAILED …::test_own_vocab_json_treats_the_shipped_word_list_as_nothing_to_own
FAILED …::test_a_file_spelling_out_the_shipped_word_list_is_the_same_model_as_the_checkpoint
```

### The new tests

* `test_own_vocab_json_treats_the_shipped_word_list_as_nothing_to_own` — the contract as
  three cases: the shipped list → `None`, `None` → `None`, an own list → itself.
* `test_a_file_spelling_out_the_shipped_word_list_is_the_same_model_as_the_checkpoint` — the
  behaviour probe. The same weights saved as a checkpoint-descended set and imported as a
  file that spells the shipped word list out must land on **one** token; and the imported
  entry must report `weight_set_owns_vocab(...) is False` and read under the canonical words.

Note why the module was blind: its own helper `_bundle_for` (`:48-60`) calls
`own_vocab_json`, so writer and reader mutate together and every round-trip stayed
self-consistent. The new tests compare against `save_weight_set(..., vocab_json=None)`,
which does not route through the mutated branch — that is what breaks the symmetry.

---

## Production behaviour

**Unchanged.** All three findings were test gaps, not defects: with the code restored, every
new test passes against the shipped implementation. Nothing in `src/` differs from `HEAD`
except the two test files and this report (`git status --short`).

## Suite numbers

| Check | Result |
|-|-|
| `code/backend` `pytest -q` | `624 passed, 1 warning in 166.09s (0:02:46)` |
| `code/backend` `ruff check .` | `All checks passed!` |
| `code/backend` `black --check .` | `86 files would be left unchanged.` |
| `code/frontend` `npm run check` | `1181 FILES 0 ERRORS 0 WARNINGS 0 FILES_WITH_PROBLEMS` |
| `code/frontend` `npx vitest run` | `Test Files 34 passed (34) · Tests 815 passed \| 1 skipped (816)` |
| `code/frontend` `npm run build` | `✓ built in 3.41s` |

`npm run test:e2e` was **not** run (reserved to the campaign lead).

### SC-703 and the TS↔Python differential suite

Undisturbed. `npx vitest run tests/unit/vacancy.test.ts tests/unit/vacancyGolden.test.ts
tests/unit/staticVacancy.test.ts tests/unit/archVacancy.test.ts` → `Test Files 4 passed (4)
· Tests 237 passed (237)`, and every §10 number the run prints is identical to the recorded
baseline at both seeds:

```
stemsTotal=1676   tokensVacated(p=1)=8125   domainTypes=2233/1940
bijective=true    imageSize=2233
```

## Repo hygiene

Working tree carries only `code/backend/tests/integration/test_geo_derived_vocab.py`,
`code/frontend/tests/unit/navGuard.test.ts` and this file. Every mutation was reverted from
a scratchpad backup and confirmed by `shasum`. The one scratch probe written
(`tests/unit/probeGeoLab.test.ts`, to find out whether `GeometryLab` can be mounted) was
deleted. No ports were bound; no dev stack was started. No secrets: the diffs contain
generated nonsense words, content hashes and public-domain corpus text only.
