# Fix 007 — Geometry Lab (agent B's findings)

Fixes for `notes/agent-reports/redteam-007-geo.md`, F1–F6. Worktree
`.claude/worktrees/agent-a6c5805673bff2158`, branch `main`.

**Status: all six fixed.** Every fix has a real test that was confirmed to fail before it
and pass after. Nothing was mocked, weakened, or skipped. Two things I could NOT verify
are stated plainly at the bottom — read that section.

---

## The shape of the F1 fix (it is the biggest change)

The old model was "a *kind* of weight set owns a vocabulary" — `scratch` and `imported`
did, `finetuned` and `edited` did not. That is wrong, and it is why the same defect
reappeared twice after `d6e9d5d`: owning a vocabulary is a property of the **derivation
chain**, not of the kind. A fine-tune of the shipped checkpoint keeps Alice's words; a
fine-tune of a scratch model must keep *that* model's words; both are `finetuned`.

So both stacks now record ownership per weight set and inherit it at every hop:

- **Backend** — `geo/weights.py` gained `inherited_vocab(base, store)` (the vocabulary a
  set derived from `base` must carry), `weight_set_owns_vocab(token)`, and an explicit
  `owns_vocab` flag in the artifact meta. `finetune()` (`geo/finetune.py:245`) and
  `mint_weight_set()` (`geo/jobs.py:282`) both call it. `mint_weight_set` also gained a
  `store` parameter so it is testable in isolation.
- **Engine** — `GeoEngine` gained a private `ownsVocab: Set<string>` and
  `inheritVocab(base, derived)`, called from `postWeights` and
  `registerFinetunedWeights`. `registerFinetunedWeights` now **requires** the base token
  (it cannot do its job without it). `ExportedWeightSet` carries `ownsVocab`.

And where inheritance is impossible, **both** writers refuse rather than substitute:
`export_bundle` raises `InvalidParamError`, mirroring the guard `exportBundle` already
had. The refusal is reachable because ownership is recorded independently of the payload
— a set can claim a word list it no longer has (a stale cache entry), and that is exactly
the case that must not silently fall back to the shipped vocabulary.

`d6e9d5d`'s claim that "the full stack was never affected" was false and I did not treat
it as a constraint; I reproduced the full-stack defect first (evidence below).

---

## F1 — derived weight sets lose their vocabulary (critical) · FIXED

**Before** (real run, this worktree's code at `0ed5365`, verbatim):

```
SCRATCH: {'weights_token': 'c59a0812e8d0405a8818bd0d0b669aab', 'final_loss': 6.89552326405302, ...}
  scratch vocab[:5] = ['badebo', 'fikapa', 'basofe', 'fifeso', 'detufi']
F1 finetune vocab[:5] = [',', '"', 'the', '.', 'and']  == scratch? False
F1 saved-file vocab[:5] = [',', '"', 'the', '.', 'and']
F1-edit: scratch[:4] ['badebo', 'fikapa', 'basofe', 'fifeso'] edited[:4] [',', '"', 'the', '.'] equal? False
F1-edit saved file vocab[:4]: [',', '"', 'the', '.']
```

**After**: all four of those come back as the scratch model's own words.

Tests (all confirmed red before the fix, green after):

| Test | File |
|-|-|
| `test_finetuning_a_scratch_model_keeps_its_vocabulary` | `code/backend/tests/integration/test_geo_derived_vocab.py` |
| `test_editing_weights_of_a_scratch_model_keeps_its_vocabulary` | same |
| `test_chained_derivation_still_carries_the_vocabulary` (scratch→edit→finetune→edit) | same |
| `test_export_refuses_when_an_owned_vocabulary_is_missing` | same |
| `carries a scratch model's words through a weight edit, into the saved file` | `code/frontend/tests/unit/geoDerivedVocab.test.ts` |
| `carries a scratch model's words through a fine-tune, into the saved file` | same |
| `keeps them through a whole chain: scratch → edit → fine-tune → edit` | same |
| `leaves a fine-tune of the SHIPPED model on the canonical vocabulary` (negative control) | same |
| `persists the ownership claim, and refuses to restore a derived set without its words` | same |

Red-before evidence for the TS half: with `inheritVocab` stubbed to a no-op,
`4 failed | 8 passed`; restored, `12 passed`.

**sessionStorage key bumped to `llm-geometry:static-weight-sets:v2`.** Payloads written by
the old build recorded ownership only through `setSource`, so a pre-fix `finetuned`
payload derived from a scratch model was stored *without* the word list it needs and is
indistinguishable from a legitimate fine-tune of the shipped model. Restoring one would
revive the corruption; the bump drops them.

**Prose updated in the same commit** (the report quoted three sentences that stated the
old behaviour): `viz/geo/TrainPanel.svelte`, `viz/geo/FinetunePanel.svelte`,
`viz/geo/GeometryLab.svelte`, `viz/info/InfoTab.svelte`.

---

## F2 — issue #6: fine-tuning tokenized with the wrong vocabulary (high) · FIXED

Fixed properly, at all three sites: `geo/finetune.py`, `lib/geoEngine/index.ts`,
`lib/staticClient/geo.ts` now tokenize with `tokenizer_for(base)` /
`engine.tokenizerFor(base)`. The pre-flight check in `geo/jobs.request_finetune` uses the
same tokenizer, so the answer matches what the job will actually train on.

**Before** (verbatim): `F2 canonical tokenizer on the ft text: n_tokens 572 n_unk 572`
with the result reported as `loss 6.614053726196289 -> 6.583433628082275`.
**After**: `unk_rate` on that same run is 0.108 — the 200 word types a 1000-word
vocabulary cannot hold out of the corpus's 1200 — and it is *reported*.

**Unk-rate guard added**, as instructed: `FINETUNE_MAX_UNK_RATE = 0.9` refuses outright
(a stream that is 90 %+ `<unk>` makes the loss a measurement of the unknown-word token,
which no caption can rescue); `FINETUNE_UNK_WARN_RATE = 0.25` is where the UI flags it.
`n_tokens` / `n_unk` / `unk_rate` travel with every result and are shown next to the loss
in `FinetunePanel.svelte`. I deliberately did **not** refuse below 0.9: fine-tuning the
shipped Alice model on modern prose legitimately unks a large share of it, and refusing
that would be a worse failure than reporting it.

Tests: `test_finetune_tokenizes_with_the_base_models_vocabulary`,
`test_finetune_refuses_an_all_unk_stream` (backend);
`reports an unk rate, and it is low when the base model knows the words`,
`refuses an almost-entirely-<unk> stream instead of reporting a loss drop` (frontend).

**GitHub issue #6: I did not close it.** The fix resolves it, but this worktree is not
the source of truth for the repo's issue tracker and I did not want to close an issue
against a branch that has not merged. The commit message references it
(`Fixes #6` is deliberately *not* used; the text says "resolves issue #6" so a human
decides). Close it when this merges.

---

## F3 — a run at the uniform baseline presented as success (medium) · FIXED

Not made to "learn" — as instructed, the run itself is correct. The defect was
presentational and it is now surfaced:

- `geo/scratch.py` gained `uniform_baseline_loss()` (= `ln(VOCAB_SIZE)`) and
  `SCRATCH_LEARNED_MARGIN = 0.5`. Results carry `uniform_baseline` and `learned`.
  Mirrored in `lib/geoEngine/scratch.ts` and `lib/staticClient/geo.ts`.
- `viz/geo/TrainPanel.svelte` renders a warning block (`geo-train-not-learned`) when
  `learned === false`: *"This run never left the uniform baseline… It has not learned
  anything from the text yet… The embedding chips in the header will still look healthy:
  they measure spread, not learning."*
- The margin of 0.5 nats is justified in the constant's comment: half a nat below
  `ln(1003)` means the distribution is at least e^0.5 ≈ 1.65× more concentrated than
  uniform. It is deliberately weaker than the suite's own `uniform − 1.0` bar, so a real
  run clears it comfortably. Absent (older cache entries) is treated as unknown, never
  as `true`.
- **The comment you asked for is written, in three places** (`geo/scratch.py`,
  `lib/geoEngine/scratch.ts`, `InfoTab.svelte`) and pinned by a test: the two existing
  gates guard against *collapse*, not against *not learning*, and the degenerate model
  scores better on both.

Tests: `test_structureless_training_is_reported_as_not_learned`,
`test_real_corpus_training_is_reported_as_learned` (positive control, 4 real epochs on
the committed corpus), `test_the_two_existing_gates_do_not_catch_a_baseline_run` —
which reproduces the red team's measurement directly (degenerate entropy/coverage >
canonical). Frontend: `classifies a structureless run as not-learned`.

The GeometryLab chip tooltips now say what they do and do not measure.

---

## F4 — incomplete model file accepted, bare `KeyError` as a 500 (medium) · FIXED

**Before** (verbatim): `F4 import_bundle ACCEPTED: {'weights_token':
'00a082546360a53eadcd95473e2dc73f', 'vocab_size': 1003}` and then
`KeyError: 'layers.0.W_V'` out of `routes_geo.py:209` as a 500.

- `geo/weights.py` gained `WEIGHT_SHAPES` (plain data — no torch construction on the file
  path) and `validate_weight_set(ws, context)`, the mirror of the engine's
  `validateWeightSet`. `import_bundle` calls it *before* the hash check, in the same order
  the engine uses.
- `GET /api/geo/weights` now raises a typed `InvalidParamError` naming the missing tensor
  and saying the set is incomplete, instead of letting `ws[name]` raise.

Tests: `test_incomplete_bundle_is_refused_on_import`,
`test_weights_route_never_leaks_a_bare_key_error` (asserts the body is *not* the string
`'layers.0.W_V'`), `test_validate_weight_set_rejects_incomplete_and_misshapen` and
`test_weight_shapes_match_the_model` (pins `WEIGHT_SHAPES` against the real torch module,
since it is a hand-maintained copy).

---

## F5 — a comment claimed a golden test that did not exist (low) · FIXED, by writing it

I wrote the golden. It is a real cross-language pin generated from the actual Python
backend, not hand-authored numbers.

The obstacle was real: torch's RNG stream is not reproducible in JS, so *whole runs*
cannot be compared (the red team measured `max|PY−TS| embedding = 1.93` on unit vectors
with losses agreeing to 0.0015 — proof that loss comparison could never catch a port
bug). The fix is to make the initialization and the repulsion sample indices **data**
rather than RNG output:

- `geo/train.py`: `uniformity_loss(embedding, idx)` now takes indices;
  `sample_uniformity_indices(gen)` draws them; `train_batch_step(...)` is the loop's
  body, extracted. `train_geo_model` calls it, so they cannot drift.
- `lib/geoEngine/scratch.ts`: the same extraction — `sampleUniformityIndices`,
  `uniformityLossAndGrad(embedding, idx, grad)`, `scratchTrainStep(...)`, `newAdamState`.
- `tests/fixtures/geo/generate.py` writes `tests/fixtures/geo/scratch_step.json`: one
  step from the canonical weight set with fixed indices — losses, **gradients**, grad
  norm, and post-step weights.
- `tests/unit/geoScratchGolden.test.ts` reproduces it and compares to ≤1e-5.

The refactor is bit-safe: re-training the canonical checkpoint with `force=True` after it
still produced `be5359a1c66bda29c8c554269e589009` — identical. Since that hash is
machine-specific (macOS vs Linux BLAS), what the suite pins is the equivalence itself:
`test_train_batch_step_reproduces_the_training_loop` drives the extracted step by hand
and demands an identical content hash and an identical loss.

**The golden has teeth** — I mutated the uniformity gradient's factor from `2` to `-2` and
it failed with `embedding: max|TS-PY| 0.00529 … expected < 0.0000026`, then restored and
re-passed. (A 0.005 % mutation is *not* caught; the tolerance is 1e-5·scale. Stated so
nobody over-reads it.) `tests/fixtures/geo/golden.json` regenerated in the same run
differs from the committed one **only in its `generated` date** — every response is
byte-identical, which is independent evidence the backend changes altered no behaviour.

---

## F6 — two `vocab_sha256` values for one model (low) · FIXED

One canonical serialization pinned in both stacks: **keys sorted, compact `,`/`:`
separators, `ensure_ascii`**. `GeoTokenizer.to_json()` gained
`separators=(",", ":")`; `lib/geoEngine/tokenizer.ts` gained `canonicalVocabJson(words)`,
which post-processes `JSON.stringify` output to escape every non-ASCII **UTF-16 code
unit** (so an astral character becomes its two surrogate escapes, exactly as
`ensure_ascii` emits). `exportBundle` uses it.

The pin is byte-identity against a file the Python backend wrote:
`is byte-identical to the vocab.json the Python backend wrote` and
`is what exportBundle writes, so a round-trip is byte-identical` compare
`canonicalVocabJson(...)` to the raw bytes of `tests/fixtures/geo/vocab.json`. Plus
`escapes non-ASCII words the way Python's ensure_ascii does` (the token regex
`[^\sa-z0-9]` really does admit accented letters and em-dashes) and the backend's
`test_vocabulary_json_is_the_pinned_canonical_serialization` /
`test_non_ascii_vocabulary_words_are_escaped_identically`.

`tests/fixtures/geo/vocab.json` was regenerated by the real backend. Files written by
older builds still load: the digest is computed over each file's own `vocab` string, so
they remain self-consistent.

---

## Contract

`specs/002-interactive-model-explorer/contracts/api.md` is amended **in its own commit**
with a note explaining why, per the repo rule. Three of these fixes change the wire:
`uniform_baseline`/`learned` on `train_scratch`, `n_tokens`/`n_unk`/`unk_rate` on
`finetune` plus its 90 % refusal, and the bundle's validation order + canonical `vocab`
spelling. While there I also recorded two pre-existing drifts the file had accumulated —
`GET /api/geo/model` documented `version: 1` and omitted `vocab_sha256`, both shipped by
feature 004 and never written down. Not mine, but I am not leaving them undocumented.

---

## Verification

- Backend: `pytest -q` — **478 tests**, all passing. `ruff check` and `black --check`
  clean at the versions `requirements.lock` pins (ruff 0.15.1, black 26.1.0).
- Frontend: `npx vitest run` — **467 passing, 1 skipped** (22 files), including the two
  new suites. `npm run check` — 0 errors, 0 warnings across 1163 files.
- Static assets regenerated with the real backend (`scripts/export_static_assets.py`) so
  the `static-export` golden source ran too.

## What I could NOT verify — read this

1. **No e2e coverage was added, and none was run.** My charter forbids `npm run test:e2e`
   (it binds the shared ports). I deliberately did not add unexecutable Playwright
   assertions: shipping test code I cannot run is exactly the kind of unverified claim
   this campaign exists to catch. The engine paths the browser actually executes
   (`GeoEngine.postWeights` / `finetune` / `exportBundle` / `importWeightSet`) are covered
   by the new unit suite instead. **Someone should still add an e2e case** covering
   "train from scratch → edit a weight → save → the file's vocabulary is unchanged",
   which is the deployed reproduction from F1.
2. **The `learned === false` banner and the unk-rate line have not been seen rendered.**
   They type-check and `svelte-check` is clean, but no screenshot exists. Same reason.
3. **The canonical checkpoint hash equality was verified on macOS only.** Linux BLAS
   legitimately produces a different checkpoint, so `be5359a1…` is not portable and is
   not asserted anywhere; the equivalence test is what CI runs.
