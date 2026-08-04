# Fix agent — Lexicon Lab UI/API (F1, F2, F4, F6)

Date: 2026-08-04. Agent: fix "LEXICON UI/API". Worktree branch off `main` @ `0ed5365`.
Charter: `notes/agent-reports/redteam-007-lex.md` findings F1, F2, F4, F6.

Commits, newest last:

| commit | scope |
|-|-|
| `bfe1043` | contract only (`specs/006-lexicon-lab-tiny/contracts/api-lex.md`) |
| `e278380` | F2 — `mint` through both stacks |
| `5e46658` | F4 — the seed input |
| `eba4db0` | F1 + F6 — provenance round trip and attribution, plus one extra defect |

**All four are fixed.** One additional defect of the same class was found while
testing and fixed in the same set; it is written up as F1b below.

Verification at the end of the run, in this worktree:

```
code/backend   pytest -q                 464 passed, 1 warning in 152.66s
code/backend   ruff check src/ tests/    All checks passed!
code/backend   black --check src/ tests/ 80 files would be left unchanged
code/frontend  npx svelte-check          1163 FILES 0 ERRORS 0 WARNINGS
code/frontend  npx vitest run            22 files, 474 passed | 1 skipped
code/frontend  npm run build             ✓ built in 3.64s
```

`npm run test:e2e` was not run (excluded by the charter). The two e2e specs that touch
the surfaces I changed were read: `vacancy.spec.ts` fills the seed box with `"7"`, which
the new parser accepts, and asserts on testids I did not remove.

The static assets under `code/frontend/public/static-data/` do not exist in a fresh
worktree (git-ignored, generated). Before generating them, 36 vitest cases failed with
`This build did not export the Lexicon Lab corpus` and `Static asset geo/checkpoint.json
is missing (HTTP 404)`. `python scripts/export_static_assets.py --quick` fixes that; CI
runs the same command before `npm run test`. No code defect was involved.

---

## F1 (high) — loading a model file made the tab claim the weights were trained

**Reproducing test:** `code/frontend/tests/unit/lexProvenance.test.ts`

- `loading a model file > keeps the untrained warnings for a file that records \`trained: false\``
- `loading a model file > says the history is unknown for a file that records no provenance`
- `loading a model file > attributes the claim to the file rather than vouching for it`
- `loading a model file > carries a hand-edited file's edited state, which no in-tab edit is holding`
- `loading a model file > does show the trained wording for a file that records a trained model` (the other direction — the fix must not be "always say untrained")

The first case loads the report's own file, verbatim:

```json
{"note":"untrained random initialization","provenance":"untrained","trained":false,"edited":false}
```

It mounts the real `LexiconLab` in jsdom over the real committed corpus
(`static-data/lex/corpus.json`), builds the bundle with the real `exportLexBundle`, and
feeds it through the real `[data-testid="lex-load-model-input"]`. The only shims are
`URL.createObjectURL` and Node's `Blob`/`File` — jsdom implements neither
`createObjectURL` nor `Blob.text()`, and they are used to carry the component's own bytes,
not to stand in for anything under test.

**Before** (`git stash push -- src/viz/lex/`, same tests):

```
× keeps the untrained warnings for a file that records `trained: false`
  → expected null not to be null
× says the history is unknown for a file that records no provenance
  → expected null not to be null
× attributes the claim to the file rather than vouching for it
  → expected '' to contain 'the file\'s own label'
× carries a hand-edited file's edited state, which no in-tab edit is holding
  → expected null not to be null
Tests  4 failed | 2 passed (6)
```

**After:** `tests/unit/lexProvenance.test.ts (7 tests) — 7 passed`.

**What changed.**

- `viz/lex/provenance.ts` — `Provenance` gains a third ORIGIN, `unrecorded`, giving six
  states (`{untrained, trained, unrecorded} × {edited, not}`). `provenanceOf` now takes
  the origin rather than a boolean; `originOf` inverts it; `provenanceFromMetrics` reads
  a file's own account in three steps (`metrics.provenance`, then the
  `trained`/`edited` booleans, then `unrecorded` with `declared: false`).
- `viz/lex/ModelFile.svelte` — `onLoaded` carries the provenance; the load line now reads
  **"weights + vocabulary verified"** (what the three digests actually establish,
  recomputed here from the file's bytes) and a separate, dashed `lex-file-claim` line
  reports what the file *claims* its weights are, attributed to the unhashed `metrics`
  block. `trained` is written as `null`, not `false`, when the tab cannot know.
- `viz/lex/LexiconLab.svelte` — `trained` → `base`, carrying `provenance` as a field. The
  active state is `provenanceOf(originOf(base.provenance), isEdited(base.provenance) ||
  edited !== null)`: a loaded file may already be hand-edited while the tab's own edit
  slot is empty.
- `SamplePanel`, `ForwardPassPanel`, `SpectrumPanel`, `TokenCloud`, `ModelFile` — a
  sentence for each new state. The two panels that use `Record<Provenance, string>` maps
  made this compiler-enforced: `svelte-check` failed on the missing keys, which is why
  the union was widened rather than a boolean added.
- `TrainPanel` props `trainedModel/trainedVocab/trainedNote` → `baseModel/baseVocab/
  baseNote`, and its disabled-fine-tune tooltip now mentions loading a file. A fine-tune
  may now depart from a loaded model that was never trained; the name had to survive that.

**Why a third origin rather than reading the booleans and defaulting.** A bundle from
`GET /api/lex/model` carries real losses and (before this work) no provenance field. It
really was trained, so defaulting an absent record to `untrained` is the same defect as
F1 with the opposite sign. The tab now says the history is unknown, and — separately, in
the F2 commit — `/api/lex/train` and the static client's trainer write
`provenance`/`trained`/`edited` into their metrics so future bundles do not land there.

---

## F1b (found while testing, high) — a model loaded before the corpus fetch landed was silently discarded

Not in the red-team report. Same defect class as F1: the load line said the file verified
and the tab then threw the model away without a word.

`shapeKey` contains `vocab?.rows ?? 0`, and the retirement `$effect` compared it against a
snapshot taken when the model was adopted. Click **↑ Load model** before the corpus fetch
returns and the snapshot is taken against `rows = 0`; the fetch then "changes the shape"
and the just-loaded model is retired. Measured, on `main`, with a real load and no wait:

```
OK LINE: loaded m.llmlex.json · 40-word dolch budget (44 rows) · model f138ca578ff3… weights + vocabulary verified
UNTRAINED WARNING PRESENT: true
ACTIVE MODEL: undefined
```

After the fix, same script:

```
UNTRAINED WARNING PRESENT: false
ACTIVE MODEL: active model: loaded from m.llmlex.json
```

Retirement now compares the base's OWN shape (`shapeKeyOf(model, vocab)`) against the
controls, and does nothing at all while `vocab === null` — there is no shape to compare
with yet, and `rows = 0` matches no real model.

**Reproducing test:** `lexProvenance.test.ts > loading a model file > keeps a model
loaded before the corpus fetch landed`. Before: `expected '' to contain 'loaded from'`.

---

## F2 (medium) — the backend silently ignored `mint`

**Reproducing tests:** `code/backend/tests/contract/test_api_lex.py`

- `test_vacancy_params_reads_every_knob_the_transform_declares` — the wiring, not the
  symptom: every non-underscore field of `VacancyParams` must survive a non-default value
  through `_vacancy_params`. It fails the day another knob is added and forgotten.
- `test_vacancy_honours_and_echoes_the_swap_mint_control`
- `test_vacancy_rejects_parameters_outside_the_contract` (+4 cases: `"bogus"`, `3`,
  `null`, and `swap` under `consistent: false`)
- `code/frontend/tests/unit/staticVacancy.test.ts` — the cross-stack parity fixture gains
  a `mint-swap-p1-seed0` case, compared field for field including `vacated_sha256`; its
  reject list gains the same three cases.

**Before:**

```
AssertionError: _vacancy_params dropped 'mint': asked for 'swap', got 'nonce'
  — the caller's setting was silently replaced by the dataclass default
KeyError: 'mint'
AssertionError: assert 200 == 400        # {"mint": "bogus"}
```

**After:** `14 passed, 37 deselected` for `-k vacancy`; the full backend suite is 464
passed. The swap control now demonstrably runs — the new fixture case's preview begins
`THE SHOEING / ROBIN SNOW … _Grandmothered by_ Brings Doorer`, real English words, which
is what §8.3 asks for, against `THE KRARRD / SORISH KLOALK` for the nonce mint.

**What changed.**

- `routes_lex.py::_vacancy_params` passes `mint=payload.get("mint", VacancyParams.mint)`.
  Validation is the dataclass's own `__post_init__`, which already refused anything
  outside `MINT_STRATEGIES` — so the typed 400 came for free once the value was passed.
  The value is handed over **unconverted**: `str(3)` would have become `"3"` and lost the
  message.
- `routes_lex.py::_vacate` now passes `type_counts(tokens)` when swapping.
  `build_vacancy_map` raises without them, so honouring `mint` alone would have replaced
  a silent wrong answer with a loud unreachable one — the control would still not work.
- The response echoes `"mint": params.mint`.
- `staticClient/lex.ts` had the identical defect (its `vacancyParamsFrom` never read
  `body.mint`, under a comment promising that could not happen). It now parses, validates
  against a `Record<MintStrategy, true>` so a third strategy fails to compile here,
  mirrors `__post_init__`'s swap+inconsistent refusal, echoes `mint`, and routes both its
  map builds through one `buildMapFor` that supplies the counts.
- `scripts/export_vacancy_api_golden.py` grows the swap case; the fixture was regenerated
  from the real route.

**Contract:** `specs/006-lexicon-lab-tiny/contracts/api-lex.md` documented neither the
request nor the response field. Amended in its own commit `bfe1043`, per the repo rule,
with the reasoning in the message. The frozen feature-002 contract is untouched
(`test_frozen_geo_contract_untouched` still passes).

---

## F4 (low) — the seed input lied about its own range

**Reproducing test:** `code/frontend/tests/unit/lexVacancySeed.test.ts` (8 cases), which
mounts the real `VacancyPanel` over a real `buildVacancyMap` result and drives the real
input.

**Before** (`git stash push -- src/viz/lex/VacancyPanel.svelte`):

```
× refuses one above its declared maximum instead of accepting it
  → expected [ 10000 ] to deeply equal []
× refuses 2^53 + 1 rather than silently applying 2^53
  → expected [ 9007199254740992 ] to deeply equal []
× refuses a negative seed rather than rewriting it     → expected [ +0 ]
× refuses a fractional seed rather than rewriting it   → expected [ 3 ]
× refuses exponent notation rather than rewriting it   → expected [ 1000 ]
× refuses hexadecimal rather than rewriting it         → expected [ +0 ]
× applies an ordinary seed and clears the error it showed before
Tests  7 failed | 1 passed (8)
```

The second line is the report's observation reproduced exactly.

**After:** 8 passed.

**What changed.** `max` is now `Number.MAX_SAFE_INTEGER`, which is also the largest value
accepted, and out-of-range or non-integer input is refused with a visible
`lex-vacancy-seed-error` naming the seed still in use. Digits only (`/^\d+$/`), because
`Number()` accepts `"1e3"`, `"0x10"`, `" 12 "` and `"3.0"` and turns each into an integer
the reader did not type. The magnitude test is done in `BigInt`:
`Number("9007199254740993") <= MAX_SAFE_INTEGER` is *true*, since the parse has already
rounded it to the bound. One of the eight cases is the invariant that catches this class
outright — the value at the declared `max` is accepted, one above it is refused.

**Bound chosen, and why it may need revisiting.** `2^53 − 1` is the largest integer
JavaScript represents exactly, and `u(stem, seed) = sha256(f"{seed}:{stem}")` consumes the
seed as digits, so above it the browser hashes a seed nobody chose while Python hashes
the one that was sent (F3). If the agent fixing F3 introduces a *tighter* engine bound,
this control's `max` should be sourced from that constant instead of from
`Number.MAX_SAFE_INTEGER`; `lexEngine/vacancy.ts` exports no such constant today, and I
did not add one because that file is not mine.

---

## F6 (low) — `metrics` outside all three digests, with a justification F1 falsified

The report is right that the exclusion is documented as intentional, and right that the
stated reason was false. I did **not** bring `metrics` under a digest. The reasoning is
written into `lib/lexEngine/bundle.ts` and into the contract rather than left in a commit
message:

- `_model_token` is the contract's hash and a bundle written by either stack must verify
  in the other; a fourth mandatory digest would refuse every bundle the backend writes.
- A token that changed when a note was edited would make the cache key depend on prose.

So the old sentence — "`metrics` … is the one block that cannot mislabel a token" — is
replaced by the narrower statement that is actually true (it cannot mislabel a token
*id*, because the vocabulary is inside `model_token`), followed by the obligation that
falls out of it: `metrics` establishes nothing, a forged `final_loss` of `1e-05`
round-trips verbatim, and **any surface that repeats the block must attribute it**. The
tab now does: `lex-file-claim`, asserted by `lexProvenance.test.ts > attributes the claim
to the file rather than vouching for it`, and the load line no longer lets the single
word "verified" cover both the digests and the label.

The contract paragraph in `api-lex.md` §"The three digests" carries the same text plus
the writer/reader obligation: a writer SHOULD record `provenance`/`trained`/`edited`
(both trainers now do), and a reader MUST treat their absence as "unknown".

**Residual, stated plainly:** a file can still declare `"provenance":"trained"` over a
random initialization and the tab will repeat that claim. It repeats it *as the file's
claim*, next to the sentence saying the block is unhashed, which is the most a format
whose provenance block is deliberately outside its digests can honestly do. Closing that
gap needs a signature or a fourth mandatory digest, i.e. a contract change with a
migration for every bundle the backend has written — out of scope here, and I have not
done it.

---

## Things I did not do, and why

- **F3, F5, F7** are another agent's (`lex/vacancy.py`, `lexEngine/vacancy.ts`,
  `specs/007-*/architecture.md`). I did not touch those files. F4's fix is the input
  control's half only; the transform's own 2^53 behaviour is theirs.
- **`viz/info/InfoTab.svelte`** is outside my file set. I checked it for the F6 claim and
  it does not repeat it, so nothing there needed correcting.
- **e2e** was not run, per the charter.
- **A tighter seed bound sourced from the engine** — see F4 above. If F3's fix adds an
  exported bound, `VacancyPanel`'s `MAX_SEED` should read it.
