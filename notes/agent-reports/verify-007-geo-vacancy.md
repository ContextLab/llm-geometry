# Verification 007 — Geometry Lab + the vacancy transform (agent V1)

Date: 2026-08-04. I did not write any of this code and I did not treat the fix reports as
evidence. Every number below I computed myself; every quote is verbatim output from a run I
made. I left nothing modified under `code/` or `specs/`: every mutation below was applied,
measured, and restored, and my temporary test file was deleted. (`git status` at the time of
writing shows ~20 modified files — those belong to sibling verification agents still running;
see §(c) item 4 for how that affected my measurements.) No server was started, stopped or
restarted; `npm run test:e2e` was not run.

Scope: `fix-007-geo.md` F1–F6, `fix-007-vacancy-core.md` TASK 1–4, `fix-007-reconcile.md`
Task 1–3, against `redteam-007-geo.md` and `redteam-007-lex.md`.

**Counts: 1 critical · 2 high · 5 medium · 3 low.**

Verdicts, at a glance:

| # | Claim under test | Verdict |
|-|-|-|
| 1 | Geo F1 — vocabulary ownership travels every derivation hop | **PARTIAL** — the chains are fixed; two other paths still write the corrupt file |
| 2 | Geo F2 — base tokenizer + a 90 % `<unk>` refusal | **PARTIAL** — correct and reported, but the bound is one-sided at exactly 0.9 |
| 3 | Vacancy TASK 1 — type→type derangement, closed class intact | **VERIFIED** (one documented, quantified bend) |
| 4 | Vacancy TASK 2 — `\|seed\| ≤ 2^53−1`, raising not clamping | **PARTIAL** — the magnitude bound holds; the HTTP route still truncates non-integers |
| 5 | SC-703 invariance still holds | **VERIFIED** — 0 mismatches, controls still break it |
| 6 | TS ↔ Python parity after the rewrite | **VERIFIED** — 0 real differences in 783 fuzz cases |

---

## (a) Defects that SURVIVE the fix

### F1. The Geometry Lab can still write a saved file whose weights and word list disagree, with every digest verifying — through the cache's content-hash **dedup**, not through the derivation chain the fix repaired

**Severity:** critical
**Where:** `code/backend/src/llm_geometry/geo/weights.py:252` —
`if store.get(key) is None:  # dedup: identical content already stored`. The metadata
block that carries `vocab` and `owns_vocab` is written *inside* that branch, so for a
content hash already present the new vocabulary is **discarded, first-write-wins**.
Same file, `save_weight_set` docstring: "``vocab_json`` travels with any model whose token
ids mean words of its OWN".

**Reproduce (real HTTP, live backend, deterministic):** two corpora that differ only in the
*spelling* of their 1,200 word types, with an order-preserving renaming so the
frequency-rank tie-break at `tokenizer.py:105` (`key=lambda kv: (-kv[1], kv[0])`) yields the
identical id stream. The trainer sees identical ids, so it produces identical weights.

```
corpus A token: 272ec63f912285715d48d7b5e3812c2b
corpus B token: 272ec63f912285715d48d7b5e3812c2b  SAME TOKEN? True
B user's saved file, vocab[:4]: ['qqgobubo', 'qqsucutu', 'qqsanoda', 'qqsocizi']
  words from corpus A in that file: 1000
  words from corpus B in that file: 0
  vocab_sha256 self-consistent: True
  declared weights_token == returned token: True
  the B user's OWN first four words, tokenized under their own model: {"tokens": [{"id": 0, "text": "wwbabefa", "unk": true}, {"id": 0, "text": "wwbacuve", "unk": true}, {"id": 0, "text": "wwbadida", "unk": true}, {"id": 0, "text": "wwbagebu", "unk": true}], "n_unk": 4, "truncated": false}
```

The B user trained on their own text, pressed *Save model*, and got a file pairing their
weights with **someone else's word list**, self-consistently hashed. Their own vocabulary is
`<unk>` in their own model. This is exactly the substitution the three digests exist to
prevent, committed by the writer, reached through a path the fix did not consider.

**The same defect is reachable by simply loading a file.** `import_bundle` → `save_weight_set`
hits the same dedup, so a model file is accepted with a 200 and then read under a *different*
word list than it contains:

```
crafted file self-consistent: True | weights_token unchanged: True
POST /geo/model of the crafted file -> 200 {'weights_token': '15c15a76155a1a40aa223d0937e996fa', 'vocab_size': 1003}
re-exported vocab words[:3]: ['qalokemu', 'qabudaso', 'qacagala']      <- the CACHE's words
tokenize the file's OWN first word: {"tokens": [{"id": 0, "text": "zzlokemu", "unk": true}], "n_unk": 1, "truncated": false}
```

**A realistic trigger, using an artefact the shipped site is producing today.** Whatever
vocabulary is cached *first* for a content hash wins forever, so it is enough to load a
pre-fix model file — one of the corrupt files https://context-lab.com/llm-geometry/ writes
right now (F7) — before training the same model on the fixed build. In a fresh private cache,
with the real `import_bundle` / `train_scratch` / `export_bundle`:

```
pre-fix-shaped file: weights_token 272ec63f912285715d48d7b5e3812c2b vocab[:4] [',', '"', 'the', '.']
1) load that file first -> {'weights_token': '272ec63f912285715d48d7b5e3812c2b', 'vocab_size': 1003}
2) NOW train from scratch on the real corpus -> 272ec63f912285715d48d7b5e3812c2b final_loss 6.89588
3) the scratch model's SAVED FILE, vocab[:4]: [',', '"', 'the', '.']
   digests self-consistent: True
   words that are the user's own (qq*): 0
```

The from-scratch run built a 1,000-word vocabulary from the user's text, reported
`vocab_size 1003`, and the file it saves is Alice in Wonderland's word list.

**And the two stacks resolve the collision in opposite directions.** The TS engine's
`registerScratchModel` (`lib/geoEngine/index.ts:531-536`) overwrites unconditionally — *last*
write wins:

```
COLLIDE: same token? true | words now: [ 'zzbagigi', 'zzdegiho', 'zzdehotu' ]
```

so on the same two inputs the backend corrupts the second model and the browser corrupts the
first. A saved file's word list therefore depends on which build wrote it and in what order,
which contradicts `InfoTab.svelte:566-568` ("a model saved by the browser and the same model
saved by the Python backend are the same file").

**Expected:** a content-addressed store must not treat *metadata* as content. Either the
vocabulary digest participates in the key, or `save_weight_set` must reconcile an existing
entry (and raise when the stored `vocab` differs from the one being written) instead of
silently keeping the first. `import_bundle` must not report success for a file whose word
list it then throws away.

**Would it have thrown?** No. HTTP 200 throughout, both digests verify, the file reloads, and
every label on the sphere is confidently wrong.

---

### F2. `GeoEngine.importWeightSet` still accepts a pre-fix-shaped persistence payload and restores it half-right — the `d6e9d5d`/F1 corruption is held off by a **key name**, not by code

**Severity:** high
**Where:** `code/frontend/src/lib/geoEngine/index.ts:387` —
`const ownsVocab = payload.ownsVocab ?? SET_SOURCES_WITH_OWN_VOCAB.has(payload.setSource);`
with `SET_SOURCES_WITH_OWN_VOCAB = new Set(["scratch", "imported"])` (line 90). The whole
defence is `MINTED_SETS_KEY = "llm-geometry:static-weight-sets:v2"`
(`lib/staticClient/geo.ts:63`).

**Reproduce:** fine-tune a scratch model, export the persistence payload, strip it to exactly
what the pre-fix build wrote (`weights`, `sources`, `setSource` — no `ownsVocab`, no
`vocabWords`), and restore it into a fresh engine.

**Observed** (vitest, real engine, real scratch run):

```
V1-SHAPED PAYLOAD accepted? true | after restore, exportBundle vocab is canonical? [",","\"","the","."]
```

The set is accepted, `tokenizerFor` falls back to the shipped tokenizer, and `exportBundle`
writes a file pairing a scratch model's fine-tuned weights with Alice in Wonderland's word
list under a matching `vocab_sha256` — the original critical defect, alive in the code.
The commit's reasoning is quoted in the file itself: "Bumping the key drops them instead."
It does drop them (I confirmed nothing reads a `:v1` key — `grep -rn "static-weight-sets"
src/` returns one hit, the `:v2` constant). But the *acceptance rule* is unchanged, so any
future key bump that reads a previous generation, any payload copied between profiles, and
any other caller of `importWeightSet` revives it.

**Expected:** refuse a payload that carries neither `ownsVocab` nor `vocabWords` for a
`finetuned`/`edited` `setSource` — it is genuinely undecidable, which is the fix report's own
argument for dropping it. Deciding it as `false` is the corruption.

**Would it have thrown?** No.

---

### F3. The TS half of the "both writers refuse rather than substitute" guarantee has **no test at all** — deleting it changes nothing

**Severity:** medium
**Where:** `code/frontend/src/lib/geoEngine/index.ts:563-570` (`exportBundle`'s refusal).
`fix-007-geo.md`: "where inheritance is impossible, **both** writers refuse rather than
substitute: `export_bundle` raises `InvalidParamError`, mirroring the guard `exportBundle`
already had."

**Reproduce:** replace the guard's condition with `if (false) {` and run every geo suite.

**Observed:**

```
=== T3: exportBundle drops its refusal
 code/frontend/src/lib/geoEngine/index.ts | 2 +-
 Test Files  5 passed (5)
      Tests  66 passed | 1 skipped (67)
```

(`geoDerivedVocab`, `geoEngine`, `staticClient`, `geoEngineFinetune`, `geoScratch`.) The
Python mirror **is** covered — the same mutation there kills two tests (see §(d)). Only the
browser half, which is what the public site runs, is unpinned.

**Expected:** a case that puts a set in `ownsVocab` with nothing in `vocabs` and asserts
`exportBundle` throws. Note the same unreachable-by-design state also silently degrades
`tokenizerFor` (`index.ts:256-259`) and `tokenizer_for` (`geo/tokenizer.py:206-211`), which
fall back to the canonical vocabulary rather than refusing — only `exportBundle` refuses.

**Would it have thrown?** No — that is the point: the guard that was supposed to throw can be
removed without a single test noticing.

---

### F4. The 90 % `<unk>` bound is one-sided at exactly 0.9, and the refusal message rounds an accepted rate and a refused rate to the same "(90%)"

**Severity:** medium
**Where:** `geo/finetune.py:57` `FINETUNE_MAX_UNK_RATE = 0.9` with `if unk_rate >
FINETUNE_MAX_UNK_RATE` (line 189); mirrored at `lib/geoEngine/index.ts` and
`lib/staticClient/geo.ts`.

**Reproduce:** fine-tune the shipped model on texts of controlled `<unk>` rate.

**Observed — backend, real HTTP:**

```
unk~0.5: OK loss 5.0568 -> 4.7250  n_tokens=1000 n_unk=500 unk_rate=0.5000
unk~0.85: OK loss 3.3536 -> 2.8959  n_tokens=1000 n_unk=850 unk_rate=0.8500
unk~0.89: OK loss 3.1191 -> 2.6336  n_tokens=1000 n_unk=890 unk_rate=0.8900
unk~0.9: OK loss 3.0612 -> 2.5670  n_tokens=1000 n_unk=900 unk_rate=0.9000
unk~0.901: 400 {"error": {"type": "InvalidParamError", "message": "901 of 1000 tokens (90%) in this text are outside the active model's vocabulary…
unk~0.95: 400 …
```

**Observed — TS engine, identical numbers (so this is not a port divergence):**

```
TS unk target 0.9 (measured 0.9000): ACCEPTED loss 3.0612 -> 2.9198 n_tokens=1000 n_unk=900 unk_rate=0.9000
TS unk target 0.901 (measured 0.9010): REFUSED 901 of 1000 tokens (90%) in this text are outside the active model's vocab…
```

A stream that is *exactly* 90 % `<unk>` is accepted and reported as "loss 3.06 → 2.57", and
the refusal one token later is announced with the same displayed percentage the accepted run
had. The fix report's justification — "fine-tuning the shipped Alice model on modern prose
legitimately unks a large share of it" — is real but does not reach 0.9: I measured a
paragraph of modern financial prose under the shipped vocabulary at

```
modern prose: n_tokens 73 n_unk 46 unk_rate 0.6301
```

so the legitimate case the bound protects sits ~0.27 below it. `FINETUNE_MAX_UNK_RATE` is
documented as "a stream that is 90 %+ `<unk>`", which reads as `>=`, not `>`.

**Expected:** `>=`, and a bound justified by a measurement rather than a round number.

**Would it have thrown?** No.

---

### F5. Three of the fix's own new backend tests do not pass when their file is run alone

**Severity:** low
**Where:** `code/backend/tests/integration/test_geo_derived_vocab.py` (lines 194, 229, 268).
`tests/conftest.py:12` gives every run a fresh cache
(`os.environ.setdefault("LLM_GEOMETRY_CACHE_DIR", tempfile.mkdtemp(...))`), and these three
call `load_canonical_weight_set()` / `finetune(base="learned")` without training it.

**Observed** (unmodified `main`, cold cache):

```
3 failed, 10 passed, 1 warning in 9.14s
FAILED …::test_finetune_refuses_an_all_unk_stream
FAILED …::test_the_two_existing_gates_do_not_catch_a_baseline_run
FAILED …::test_weights_route_never_leaks_a_bare_key_error
E           llm_geometry.errors.NotFoundError: The canonical GeoTransformer checkpoint has not been trained yet; POST /api/geo/train (or call train_canonical()) first
```

After `train_canonical()` into the same cache: `13 passed, 1 warning in 8.62s`. So they pass
in a whole-suite run (as `fix-007-geo.md` reports) and fail in isolation — the failure mode a
bisect or a `-k` run hits first. Not a product defect; recorded because the fix report
presents these tests as the evidence for F1–F4 and I could not run three of them as written.

**Would it have thrown?** n/a (test hygiene).

---

### F6. `InfoTab` states the vocabulary-substitution guarantee twice, in contradictory terms — and the stronger of the two is false

**Severity:** low
**Where:** `code/frontend/src/viz/info/InfoTab.svelte:555-557` vs `:562-565`.

Verbatim, one paragraph apart:

> "A file with real weights and a tampered word list would silently mislabel every point on
> the sphere, so it **is refused instead**."

> "The digests cannot police this on their own — a writer that substituted the shipped word
> list would also compute `vocab_sha256` over the substituted list, and **the file would
> verify**."

The second sentence is the true one, and I confirmed it directly: replacing the whole word
list and recomputing `vocab_sha256` produced a file both stacks accept
(`POST /geo/model … -> 200`, TS `importBundle` returns the substituted words). A reader who
stops at the first sentence has been given a guarantee the format cannot provide.

**Would it have thrown?** No — prose.

---

### F7. The deployed static site at https://context-lab.com/llm-geometry/ contains **none** of the 007 fixes, and still reproduces the original critical defect end to end

**Severity:** high (not a defect *in* the fix — a defect in what the public is running)
**Where:** the Pages deploy, `.github/workflows/pages.yml`. The cause is simpler than a stale
deploy: **none of the fix commits have been pushed.**

```
$ git status -sb | head -1
## main...origin/main [ahead 19]
$ git log --oneline origin/main -1
0ed5365 Merge pull request #8 from ContextLab/007-vacancy-transform
```

The last Pages run and the last CI run are both `2026-08-04T18:47:13Z` on that same
`0ed5365`. So the site is serving `origin/main`, **and CI has never run on any fix commit** —
every "471 / 478 / 501 passed, `npm run check` clean" figure in the three fix reports is a
local macOS run only. (`fix-007-geo.md` itself flags one consequence: "The canonical
checkpoint hash equality was verified on macOS only.")

**Observed — string probe of the served bundle** (`assets/index-Bjyi2uBc.js`,
`last-modified: Tue, 04 Aug 2026 18:52:23 GMT`):

```
llm-geometry:static-weight-sets:v2         0
llm-geometry:static-weight-sets            1     <- the PRE-fix, unversioned key
ownsVocab                                  0
outside the active model's vocabulary      0     <- the new unk refusal
uniform baseline                           0     <- the F3 banner
1,940                                      0
1,944                                      2     <- the pre-rewrite §10 count
isVacatable                                0
```

**Observed — real clicks on that site** (train from scratch on 13,200 tokens of 1,200
invented words, 1 epoch, *Save model*, one `W_Q ← identity` edit, *Save model*):

```
TRAIN RESULT: trained a new model · final loss 6.89 · 13,200 tokens · 1 epochs — it is now the active model, with its own vocabulary
NOT-LEARNED BANNER: ABSENT
ACTIVE after edit: active model: hand-edited weights 953ad125
SAVED[0] wt=5fe1fecb06263bbe69a24e7199716fb8 vocab[0:4]=["denuso","dezefi","bamoka","fipade"]
SAVED[1] wt=953ad12508725b9dd59434b5f10c036a vocab[0:4]=[",","\"","the","."]
scratch vocab == edited vocab ? false
console errors: []
```

That is `redteam-007-geo.md` F1 verbatim, live, today. **Consequence for this whole campaign:
no 007 fix can be verified against the deployed static build until it redeploys** — every
static-mode verdict in this report is from the engine and the staticClient in vitest, not
from the shipped page.

**Would it have thrown?** No.

---

## (b) NEW defects the fix introduced

None found that the fix itself created. F1's dedup path and F6's contradictory prose both
predate the fix (dedup is original `save_weight_set` behaviour); F2 and F3 are gaps the fix
left open rather than damage it did. The one *newly introduced* asymmetry worth naming is
inside F1: `inheritVocab`/`registerScratchModel` in TS overwrite where `save_weight_set` in
Python dedups, so the two stacks now disagree about which of two colliding models keeps its
words. Counted under F1.

---

## (c) Fix claims I could NOT verify, and why

1. **Everything about the deployed static build.** See F7 — the site is running pre-fix code,
   so `redteam-007-geo.md`'s live reproductions cannot be re-run against the fix. I verified
   the static-mode code paths (`GeoEngine`, `staticClient/geo.ts`) in vitest instead, which is
   the same code the page will run once it deploys, but that is not the same evidence.
2. **`fix-007-reconcile.md` TASK 3, the q8 arm of `VACANCY_Q8_UNCERTAINTY_NATS`.** Still
   unmeasured, still correctly flagged as such in three places. I did not re-run it: it needs
   a browser q8 pass on the *new* engine, and the deployed build is the old engine (F7), so
   the run is not currently possible anywhere. The constant remains under-supported exactly as
   the fix report says.
3. **The `e2e` suites** (`docs.spec.ts` and the four new cases). Forbidden by charter; the
   two assertions that read §10 counts off the live API were re-derived by hand instead
   (§(d)) but the suites themselves remain unexecuted, as they were for every fix agent.
4. **Any measurement taken while sibling verification agents held the tree modified.** Several
   other agents were mutating `code/frontend/src` and `specs/` during my run
   (`git status` showed up to nine foreign modifications at once) and one of them deleted the
   shared scratch directory mid-run. Every measurement in this report was taken either against
   a file I had just restored, or against `git show HEAD:<path>`, or through the running dev
   stack (which serves the committed tree); one sub-verifier's first TS run reported
   `stemsTotal 1680` and was not reproducible — every clean re-run gave 1676. Treat any single
   unbracketed number from this campaign with suspicion.
5. **`fix-007-geo.md` F5's cross-language scratch golden under a real port bug.** I confirmed
   the golden exists and that the fixer's own reported mutation kills it, but I did not
   independently mutate it; my mutation budget went to the F1/F2/F3 guards, which are where
   the critical defect lives.

---

## (d) What I confirmed is genuinely fixed

**Geo F1 — the derivation chain itself.** Every hop carries the vocabulary, in both stacks,
including chains longer than the fixer tested and the "loaded from a file, then derived from"
case.

Backend, real HTTP, scratch → edit → fine-tune → edit → export → import → fine-tune → edit:

```
S      (['qalokemu', 'qabudaso', 'qacagala', 'qamoduko'], ('digest_ok', True), ('wt_declared', '15c15a76…'))
edit1  (['qalokemu', 'qabudaso', 'qacagala', 'qamoduko'], ('digest_ok', True), …)
ft     (['qalokemu', 'qabudaso', 'qacagala', 'qamoduko'], ('digest_ok', True), …)
edit2  (['qalokemu', 'qabudaso', 'qacagala', 'qamoduko'], ('digest_ok', True), …)
import 200 {'weights_token': '0e8c49921933bd0843f8d026e2223baa', 'vocab_size': 1003}
ft-of-import  (['qalokemu', 'qabudaso', 'qacagala', 'qamoduko'], …)
edit-of-that  (['qalokemu', 'qabudaso', 'qacagala', 'qamoduko'], …)
```

TS engine (vitest, real training, real files) — three probes I wrote, all green:
scratch → persist → **fresh engine (reload)** → edit → fine-tune → edit → save;
imported → fine-tuned → edited → save; and cross-stack.

**Cross-stack, both directions, byte-identical.** A Python-written scratch bundle imported
into the TS engine, derived from there, re-exported, and posted back:

```
TS-written bundle: wt 34b356819c3bb43daaebb894da8ccfd4 vocab[:3] ['qalokemu', 'qabudaso', 'qacagala']
POST to backend -> 200 {'weights_token': '34b356819c3bb43daaebb894da8ccfd4', 'vocab_size': 1003}
backend vocab bytes == TS vocab bytes: True
backend vocab_sha256 == TS: True
backend derive-from-TS-file vocab[:3]: ['qalokemu', 'qabudaso', 'qacagala']
```

That also settles **F6 (two `vocab_sha256` for one model)**: the canonical serialization is
genuinely pinned — the two builds now emit the same bytes.

**Geo F1/F2/F3 in the shipped UI, full stack, real clicks at :5173** — including the two
things `fix-007-geo.md` listed as unverified ("the `learned === false` banner and the unk-rate
line have not been seen rendered"). Both render:

```
UNK LINE: 41 of 63 tokens (65%) were outside this model's vocabulary and trained as <unk> — the loss is partly about the unknown-word token, not only your words.
LOSS LINE: loss 3.68 → 2.41 on your text
TRAIN: trained a new model · final loss 6.89 · 13,200 tokens · 1 epochs — it is now the active model, with its own vocabulary
NOT-LEARNED BANNER: This run never left the uniform baseline. A model that predicts every token equally scores 6.91 nats (that is ln 1003), and this run finished at 6.89. It has not learned anything from the text yet …
ACTIVE: active model: trained from scratch on your text b3006583
ACTIVE after edit: active model: hand-edited weights 328ffe19
SAVED[0] wt=b300658308b67d7d61441be971032733 vocab[0:4]=["fisoka","fifigo","deleso","filefe"]
SAVED[1] wt=328ffe195c7d0971175bb02ca0edb167 vocab[0:4]=["fisoka","fifigo","deleso","filefe"]
console errors: []
```

Compare `SAVED[1]` here with `SAVED[1] vocab[0:4]=[",","\"","the","."]` from the deployed
build in F7: that is the fix working.

**Geo F2 — the reported unk rate is the one actually tokenized.** Independent check against
`GET /api/geo/tokenize` under the same base:

```
independent /tokenize under the shipped model: n_tokens 18 n_unk 5 truncated False
finetune reports: n_tokens 18 n_unk 5 unk_rate 0.2778
AGREE: True
```

**Geo F4 — incomplete and misshapen model files.** All refused with typed 400s naming the
tensor; an extra tensor is refused too. (My first attempt here was a false positive: a
`pos_embedding` of `[50,3]` is the *correct* shape, `CONTEXT_WINDOW == 50`.)

```
POST subset ->        [400, …"model file is incomplete (missing: ['layers.0.W_K', …
pos_embedding [49,3]-> [400, …"model file: weight 'pos_embedding' has shape (49, 3), expected (50, 3)"…
layers.0.W_Q  [4,3] -> [400, …"model file: weight 'layers.0.W_Q' has shape (4, 3), expected (3, 3)"…
layers.0.b_in [7]   -> [400, …"model file: weight 'layers.0.b_in' has shape (7,), expected (12,)"…
extra tensor        -> [400, …"model file is incomplete (missing: none, unexpected: ['bogus']) — a GeoTransformer needs all 34 tensors, so it cannot be run"…
```

**Mutation testing — the backend geo tests have real teeth.** Five surgical mutations, each
applied to the source the test covers, run, and restored
(`tests/integration/test_geo_derived_vocab.py`, baseline `13 passed`):

| mutation | result |
|-|-|
| `inherited_vocab` always returns `(None, False)` | **3 failed**, 10 passed |
| `FINETUNE_MAX_UNK_RATE` 0.9 → 1.01 | **1 failed**, 12 passed |
| `export_bundle`'s refusal removed (falls back to the canonical vocabulary) | **2 failed**, 11 passed |
| `"learned"` hard-coded `True` | **2 failed**, 11 passed |
| `validate_weight_set` made a no-op | **4 failed**, 9 passed |

TS side, `geoDerivedVocab.test.ts` (baseline `12 passed`): `inheritVocab` no-op → **4
failed**; `importWeightSet` accepting an `ownsVocab` payload with no words → **1 failed**.
The third TS mutation survived — that is F3 above.

---

## §B — the vacancy transform

Measured by two sub-verifiers I directed (charter: compute every number yourself, never read
one off a report; mutate and restore). I read their raw output files rather than their
summaries, and re-ran the parity diff myself over their two 30 MB output dumps. Findings
**F8, F9 and F10** below are counted in this report's totals.


### TASK 1 — swap is a type→type derangement inside one suffix class · **VERIFIED**

Recomputed on the shipped corpus (`load_corpus_text`, the Gutenberg-trimmed body) and the six
default Architecture passages, in **both** stacks:

- corpus, `mint=swap`, seeds 0 and 7, `p ∈ {.25,.5,.75,1}` — `notADomainType=0` and
  `suffixClassMismatch=0` at all eight points; the TS engine agrees (`inflMismatch=0`).
- six passages, `p=1`, seed 0 — **767 vacated tokens, 0 outside the domain**, 21 (2.74 %)
  with a different suffix class, every one traceable to a *merged singleton class*
  (`('leaping','april')`, `('carving','adieu')`, `("king's",'stout')`). Singleton classes
  occur in 5 of the 6 passages, `ing` in three.
- closed class: **0** function words vacated across corpus × 6 passages × {nonce, swap} ×
  seeds {0,7} × `p ∈ {.25,.5,.75,1}`, measured from the texts; and exhaustively over the
  entire `FUNCTION_WORDS` set in three casings through `apply_word` / `transformWord`,
  **0 changed** in either stack. No eighth leaking word beyond the seven the fixer named.
- the reverse-direction attack also fails: `imagesThatAreFunctionWords = 0` and
  `imagesNotVacatable = 0` at both seeds in both stacks.
- fixed points `= 0`; the map is injective at `p ∈ {0,1}` (2,233 distinct images over 2,233
  types); `swap` still refuses at `.25/.5/.75` with the typed error in both stacks; a
  two-member class deranges as a clean transposition (`singing ↔ ringing`); a class with one
  *vacatable* type raises the same `ComputeError` in both stacks; case commutes
  (`Jack→Ask`, `DOGS→THIGHS`, `I→I`, 0 violations over the whole domain).

**§10 statistics, recomputed in Python and again in TypeScript, all exact:**
`domainTypesEligible 1940`, `corpusTypesEligible 1918`, `stemsTotal 1676`,
`tokensVacated(p=1) 8125`; swap lost image slots seed 0 `0 / 349 / 484 / 364 / 0`, seed 7
`0 / 336 / 475 / 372 / 0`.

### F8. `architecture.md` states the same-inflection guarantee unconditionally, 25 lines above the exception that falsifies it

**Severity:** low
`architecture.md:758` states the same-inflection guarantee **unconditionally**,
25 lines before `:782` states the singleton-merge exception that makes it false for 2.74 % of
the vacated words on the shipped passages. The exception is documented; the guarantee above it
is not qualified.

### F9. A surviving mutant: the singleton-merge boundary — the one place the inflection match is allowed to bend — is pinned by no test

**Severity:** medium
Changing the singleton-merge
boundary from `< 2` to `< 8` passes **all 357 backend unit + contract tests** (and all 90 in
`test_lex_vacancy.py`). The shipped corpus's smallest classes (`ies`, `est`) hold exactly 8,
so the fixture cannot see it; the only passage-level assertion is one-sided (`merged >= 4`,
which *rises* under over-merging) and checks realness, never inflection. The boundary that
defines "the one place the inflection match bends" is unpinned. Killed mutants, for contrast:
dropping the keep-set check (2 fail), re-attaching the source suffix (2 fail), `< 2 → < 9`
(fails), allowing a fixed point (3 fail); TS `isVacatable` (6 fail) and the derangement (3
fail).

### TASK 2 — the seed bound · **PARTIAL**

The **magnitude** bound is real and enforced identically everywhere. TS `vacancyParams` /
`vacancyU`:

```
MAX_SEED 9007199254740991 true
9007199254740991           vacancyU       OK -> 0.897994031497381
-9007199254740991          vacancyU       OK -> 0.8307343642207088
9007199254740992 (2^53)    vacancyU       THROW: vacancy: seed must be an integer in [-900719925474…
9007199254740993           vacancyU       THROW: …
12345678901234567890       vacancyU       THROW: …
```

and over HTTP:

```
{"p":0.5,"seed":9007199254740991}        200  seed=9007199254740991
{"p":0.5,"seed":9007199254740992}        400  InvalidParamError: seed must lie in [-9007199254740991, 9007199254740991]
{"p":0.5,"seed":12345678901234567890}    400  InvalidParamError: …
```

`redteam-007-lex.md` F3 (the `seed > 2^53` cross-language divergence) is therefore genuinely
closed.

### F10. …but `POST /api/lex/vacancy` still silently rewrites a seed it cannot use, and 500s on one it cannot parse — the very behaviour the fix says it rejected

**Severity:** medium
**Where:** `code/backend/src/llm_geometry/api/routes_lex.py`'s `_as_int` coercion, upstream of
`VacancyParams`. `fix-007-vacancy-core.md`: "**raise, never clamp** … Clamping was rejected on
the ground the red team's F4 gives — a number used that is not the number asked for." And:
"The HTTP route needs no change."

**Observed** (same route, same shape of request):

```
{"p":0.5,"seed":1.5}                     200  seed=1
{"p":0.5,"seed":2.0000000001}            200  seed=2
{"p":0.5,"seed":7.0}                     200  seed=7
{"p":0.5,"seed":"7"}                     200  seed=7
{"p":0.5,"seed":true}                    200  seed=1
{"p":0.5,"seed":false}                   200  seed=0
{"p":0.5,"seed":Infinity}                500  InternalError: cannot convert float infinity to integer
{"p":0.5,"seed":-Infinity}               500  InternalError: cannot convert float infinity to integer
{"p":0.5,"seed":NaN}                     400  InvalidParamError: seed must be an integer, got nan
```

Every one of those the **TypeScript** engine refuses with a typed error, including `1.5`,
`"7"` and `true` — so the two stacks now disagree on the whole non-integer and wrong-type
domain, in the direction where Python computes a *different seed than it was asked for* and
JS refuses. `Infinity` additionally leaks a raw Python `OverflowError` message as a 500, the
same class as geo F4. (One benign asymmetry in the other direction: `VacancyParams` refuses a
Python `float` `7.0` outright — `"seed must be an int, got 7.0"` — while JSON cannot express
the int/float distinction and the TS engine accepts `7.0` as the integer 7. Loud on both
sides, so not a wrong answer.)

Two more surfaces on the same theme: `POST /api/lex/train` bounds its *vacancy* seed (400 at
`2^53+1`) but **not** its own top-level training seed (`12345678901234567890` → 202); and the
static client's `asInt(body.seed, "seed", 0)` (`staticClient/lex.ts:519`) silently substitutes
0 for a `null` seed where the backend returns a typed 400.

The UI half of the original F4 **is** genuinely fixed: `VacancyPanel`'s `parseSeed` is
digits-only with a BigInt comparison, refuses visibly, and never rewrites what was typed
(`lexVacancySeed.test.ts`, 8/8 against the real panel).

**Would it have thrown?** No for the truncations — 200, plausible output, and the applied
seed echoed back (a caller who compares the echo can detect it; nothing says it happened).

### TASK 3 — SC-703 · **VERIFIED**

Re-run independently over seeds {0,3,7,12345}, 13–17 values of `p` (including 0.05, 0.33,
0.66, 0.95, 0.99), both prosody settings, both mints, all five Dolch budgets and two frequency
budgets, comparing token id streams element by element on the 16,000-token corpus:

```
budgets: {'pre_primer': 40, 'primer': 92, 'first': 133, 'service': 220, 'full': 314, 'frequency-top300': 300, 'frequency-top150': 150}
passed {'nonce': 504, 'swap': 112} refused {'nonce': 0, 'swap': 392}
MISMATCHES: 0 []
```

and cross-stack:

```
  all TS idStreams equal to the p=0 stream: True | lengths: [16000]
  PY streams equal TS streams at every p: True
swap: idStream available at p = [0.0, 1.0] | refused at p = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
  swap streams at p=0 and p=1 equal the nonce baseline: True
```

The controls still break it, as they must — a theorem that held for everything would be a
broken measurement:

```
   consistent=False p=1.0: idStream == baseline? False  ndiff=2521/16000
   reveal_after=2  p=1.0: idStream == baseline? False  ndiff=299/16000
   reveal_after=1  p=1.0: idStream == baseline? False  ndiff=149/16000
```

The swap rewrite did not move the theorem.

### TASK 4 / parity — TS ↔ Python differential fuzz · **VERIFIED**

783 cases (random + adversarial: both mints, `p` at the endpoints and interior, negative and
near-`MAX_SEED` seeds, `consistent=false`, `reveal_after`, `keep` sets, prosody on/off,
unicode with accents/CJK/emoji/combining marks, hyphens, apostrophes, ALL CAPS, MiXeD case,
empty and whitespace-only text, text with no eligible word, very long single tokens, and the
real shipped corpus), each emitting the full `mapping`, `forbidden`, `mintedStress`,
`remintRounds`, `bijective`, `imageSize`, `domain`, `stems`, `mappedWords`, `idStream`, the
whole vacated text, and all §10 `stats`. I re-ran the comparison myself over the two raw
output files:

```
cases: 783
field -> count of cases differing: {'mapVocabRejected': 436, 'vacatedLen': 106}
cases where the sha256 of the whole vacated text differs: 0
```

Both are **harness artefacts, not product divergences**, and I checked each:

- `vacatedLen` — Python `len()` counts code points, JS `.length` counts UTF-16 code units, so
  the two disagree by exactly the number of astral characters. The transform's own output is
  identical: `vacatedSha` (sha256 of the entire vacated text) matches in all 783 cases.
- `mapVocabRejected` — the harness recorded the exception's class name; the refusal itself
  fires in both stacks at the same 436 cases (`'InvalidParamError'` in Python,
  `'Error'` in TS).

**Zero real differences in 783 cases.** I did not check whether `preview_chars` slicing shows
the same code-point-vs-code-unit asymmetry on the wire; that is a cosmetic path I ran out of
time for and I am not claiming either way.

Mutation testing of the seed-bound tests: 6/6 mutants killed (`MAX_SEED → 2**53`, the bound
removed, and a silent clamp — each in both stacks), every source restored to a matching
`shasum`.

### F11. `fix-007-reconcile.md` TASK 1 missed the one stale copy a user can actually read

**Severity:** medium
**Where:** `code/frontend/src/lib/staticClient/lex.ts:514-515`, at HEAD:

```
        "type per occurrence and the corpus has 1680 open-class stems against 8202 vacated " +
        "tokens, so there is no supply of real words (architecture.md §8.3)",
```

That is the **wire boundary's error message** for `mint="swap", consistent=false` in static
mode — the text the reader sees. The other three copies of the same sentence were corrected in
the same rewrite:

```
HEAD:code/backend/src/llm_geometry/lex/vacancy.py:864:  "type per occurrence and the corpus has 1676 open-class stems against 8125 "
HEAD:code/frontend/src/lib/lexEngine/vacancy.ts:1104: "fresh type per occurrence and the corpus has 1676 open-class stems against 8125 " +
HEAD:code/frontend/src/viz/lex/LexiconLab.svelte:236: open-class stems against 8 125 vacated tokens, so there is no supply of real words to
```

I re-derived the true values in §B TASK 1: `stemsTotal = 1676`, `tokensVacated(p=1) = 8125`.
`fix-007-reconcile.md` reports TASK 1 as "the stale numbers on screen · DONE" and lists the
files it changed; this file is not among them. Same commit also leaves
`asInt(body.seed, "seed", 0)` on line 519, which silently defaults a `null` seed to 0 where
the backend returns a typed 400 — the other half of F10.

**Would it have thrown?** No — it *is* the throw, with two wrong numbers in it.

---

## Reproduction notes

Scratch scripts lived in
`/private/tmp/claude-501/.../scratchpad/main/` and are deleted; the only artefact left in the
repo is this file. `git status` shows no modification to any file under `code/` or `specs/`.
