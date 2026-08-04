# Red-team 007 — Geometry Lab (agent B)

Scope: `viz/geo/`, `llm_geometry/geo/`, `api/routes_geo.py`, `lib/geoEngine/`,
`lib/staticClient/geo.ts`. Both modes attacked: the deployed static site
(https://context-lab.com/llm-geometry/#geometry) and the local full stack
(:5173 → :8000). Every quote below is verbatim output from a run I made.

**Counts:** 1 critical · 1 high · 2 medium · 2 low.

---

### F1. `d6e9d5d` is only half fixed: fine-tuning or weight-editing a scratch/imported model still saves its weights under Alice in Wonderland's vocabulary, with self-consistent digests

**Severity:** critical
**Where:**
- `code/backend/src/llm_geometry/geo/finetune.py:207` — `save_weight_set(new_ws, source="finetuned", store=store)` (no `vocab_json=`)
- `code/backend/src/llm_geometry/geo/weights.py` `build_weight_set` path via `api/routes_geo.py:231` — edits save with no `vocab_json`
- `code/frontend/src/lib/geoEngine/index.ts:81` — `SET_SOURCES_WITH_OWN_VOCAB = new Set(["scratch", "imported"])`; `"finetuned"` (line 464) and `"edited"` (line 392) are absent, so `exportBundle`'s guard at line 492 never fires for them
- Live: https://context-lab.com/llm-geometry/#geometry, Train a new model → Fine-tune → ↓ Save model

**Reproduce (live static site, headless Chromium, real clicks):** paste a 13,200-token
corpus of 1,200 invented words, epochs = 1, *Train from scratch*, *↓ Save model*, then
*Fine-tune on your text* with the same corpus, *↓ Save model* again.

**Observed:**
```
TRAIN RESULT: trained a new model · final loss 6.89 · 13,200 tokens · 1 epochs — it is now the active model, with its own vocabulary
SAVED[scratch] name=geotransformer-c9675b8e.llmgeo.json token=c9675b8e0d39cb467b24ead92b166ff1 vocab[0:5]=["depa","rupi","vega","fefo","fune"]
FINETUNE RESULT: loss 6.58 → 5.58 on your text
active model chip after finetune: active model: fine-tuned on your text | 0fa8472b
SAVED[finetuned] name=geotransformer-0fa8472b.llmgeo.json token=0fa8472bfb436d386aacc9cd5d7647ed vocab[0:5]=[",","\"","the",".","and"]
scratch vocab == finetuned vocab ? false
console errors: []
```
Both files verify, and the corrupt one reloads with no complaint:
```
live_finetuned.json  vocab_sha256 declared=09cac39b0f5f7b53 actual=09cac39b0f5f7b53 SELF-CONSISTENT=true
LOAD of the corrupt fine-tuned file -> loaded live_finetuned.json · 1003-token vocabulary
active model chip: active model: loaded from live_finetuned.json | 0fa8472b
```

**The full stack is affected too** — `d6e9d5d`'s message says "The full stack was never
affected: Python has always stored the word list beside the weights". Same corpus,
`POST /api/geo/train_scratch` then `POST /api/geo/finetune` with `base=<scratch token>`:
```
scratch bundle vocab words[:8]: ['depa', 'rupi', 'vega', 'fefo', 'fune', 'nufo', 'voze', 'dima']
FT bundle vocab words[:8]: [',', '"', 'the', '.', 'and', 'to', 'a', 'she']
FT vocab == scratch vocab? False
FT vocab_sha256 self-consistent: True
POST /geo/model of the FT bundle -> 200 {'weights_token': '736aa3cceda2e4a99f4b8378ed4831e3', 'vocab_size': 1003}
tokenize under imported: {'tokens': [{'id': 5, 'text': 'the', 'unk': False}, {'id': 75, 'text': 'queen', 'unk': False}], 'n_unk': 0, 'truncated': False}
```
The weights really are the scratch model's — 5 SGD steps moved them almost nowhere:
```
||FT - SCRATCH||_F  = 0.0026621627621352673
||FT - CANONICAL||_F= 31.898061752319336
||SCR- CANONICAL||_F= 31.897991180419922
```

**A single weight edit does the same thing.** `POST /api/geo/weights` with
`base=<scratch token>`, one preset edit:
```
POST /geo/weights base=SCRATCH -> 200 {'weights_token': '8302aa402333525144dccf05fbb5e15c', 'edited': [{'layer': 0, 'matrix': 'W_Q', 'source': 'preset:identity'}]}
edited-model bundle vocab words[:6]: [',', '"', 'the', '.', 'and', 'to']
trace next_token under edited model: {'id': 618, 'text': 'written'}
```
and identically in the TS engine (same token, so this is not a port divergence):
```
imported scratch model: b7a4188a99a1a675434aad5d96ab7bf0
  tokenizerFor(imported).words[0:4]: [ 'depa', 'rupi', 'vega', 'fefo' ]
after ONE weight edit: 8302aa402333525144dccf05fbb5e15c
  tokenizerFor(edited).words[0:4]: [ ',', '"', 'the', '.' ]
  saved file vocab words[0:4]: [ ',', '"', 'the', '.' ]
  saved file re-imports cleanly: {"weights_token":"8302aa402333525144dccf05fbb5e15c","vocab_size":1003}
```

**Expected:** any set derived from a model that owns a vocabulary must inherit that
vocabulary (backend: `save_weight_set(..., vocab_json=load_weight_set_vocab(base))`;
engine: carry `this.vocabs` through `registerFinetunedWeights` and `postWeights`). Failing
that, `exportBundle` must refuse, exactly as it already does for a scratch set with no
vocabulary. The product's own words make the contract explicit:

- `viz/geo/TrainPanel.svelte:203` — "a saved model file carries its vocabulary alongside its weights, each with its own checksum."
- `viz/info/InfoTab.svelte:525` — "A file with real weights and a tampered word list would silently mislabel every point on the sphere, so it is refused instead."
- `geo/bundle.py:46` — "without this a file could carry intact weights and a FABRICATED word list and load cleanly — every label in the UI would then be confidently wrong, which is exactly what this module exists to stop."

This is the substitution those three digests exist to prevent, committed by the writer,
where no digest can catch it — i.e. the same defect class as `d6e9d5d`, reached through
two paths the fix did not cover, in both builds.

**Would it have thrown?** No. Nothing throws, nothing warns, both digests verify, the
file reloads, and the sphere relabels every point.

---

### F2. Issue #6 measured: fine-tuning a scratch model tokenizes 100 % of the text to `<unk>` and reports an encouraging loss drop

**Severity:** high
**Where:** `geo/finetune.py:157` — `ids = get_tokenizer().encode_stream(text)`;
`lib/geoEngine/index.ts:436` — `const tokenIds = this.tokenizer.encodeStream(text)`;
`lib/staticClient/geo.ts:246` — `const tokenIds = engine.tokenizer.encodeStream(text)`.
All three use the canonical tokenizer regardless of `base`.

**Reproduce:** train from scratch on text the shipped vocabulary does not contain, then
fine-tune that model on the same text.

**Observed** — the fine-tuning corpus under the tokenizer actually used:
```
n_tokens 40 n_unk 38 truncated False
first 5 [{'id': 0, 'text': 'gome', 'unk': True}, {'id': 0, 'text': 'lapi', 'unk': True}, {'id': 0, 'text': 'vade', 'unk': True}, {'id': 0, 'text': 'reda', 'unk': True}, {'id': 0, 'text': 'bozo', 'unk': True}]
```
and the number the UI puts on screen for that run, live:
```
FINETUNE RESULT: loss 6.58 → 5.58 on your text
```
A full nat of "improvement" from a model learning to emit `<unk>`, labelled *on your text*.

**Expected:** either tokenize with `tokenizer_for(base)` (the fix issue #6 describes), or
refuse / warn when the `<unk>` rate of the fine-tuning stream is near 1. There is no
unk-rate gate anywhere on this path; the only guard is `len(ids) < 2`.
The prose at `FinetunePanel.svelte:126-129` does disclose the tokenizer choice
("your text is tokenized with the **shipped** vocabulary… (issue #6)"), so the *tokenizer*
is documented — the silently plausible loss on a 95–100 % `<unk>` stream is not, and
neither is F1, which is downstream of the same omission.

**Worse than stated in the brief:** issue #6 is filed as "fine-tuning ignores a
scratch-trained model's own vocabulary". It also *destroys the saved file* (F1) and
produces a believable training curve from a meaningless run.

**Would it have thrown?** No.

---

### F3. From-scratch training ships a model that learned nothing, and the two "non-degeneracy" gates score it *better* than the real checkpoint

**Severity:** medium
**Where:** `geo/scratch.py:138-159` (no gate on `final_loss`);
`viz/geo/TrainPanel.svelte:275-279` (the result banner)

**Reproduce:** train from scratch, default epochs = 12, on 13,200 tokens of structureless
text (1,200 word types drawn i.i.d.).

**Observed:**
```
epochs=12 on a structureless corpus: {"epochs": 12, "final_loss": 6.78428, "n_distinct": 1200, "n_tokens": 13200, "seed": 0, "vocab_size": 1003, "weights_token": "08066f8fe78c86c01eb2385f866c82e6", "ready": true}
ln(1003) = 6.910750787961936
```
i.e. 0.13 nats below the "learned nothing" baseline. The repo's own bar for the same
trainer, `tests/unit/geoScratch.test.ts:104-110`, is
`const uniform = Math.log(VOCAB_SIZE); // 6.91 nats — the "learned nothing" baseline` …
`expect(result.finalLoss).toBeLessThan(uniform - 1.0)` — i.e. < 5.91. The product accepts
6.78, activates it, and says:
```
TRAIN RESULT: trained a new model · final loss 6.89 · 13,200 tokens · 1 epochs — it is now the active model, with its own vocabulary
```

The other two gates cannot catch it either — I scored the degenerate model and the shipped
one with `geo.fields.field_directional_entropy` / `coverage_uniformity`:
```
SCRATCH:   field_directional_entropy=3.2838  coverage_uniformity=0.9884  n_arrows=1003
canonical: field_directional_entropy=2.8121  coverage_uniformity=0.9005  n_arrows=1003
```
The garbage model beats the real one on both, because near-random embeddings are maximally
dispersed and maximally multi-directional. Those two metrics are gates against *collapse*,
not against *not learning*; only `final_loss` distinguishes the two, and nothing checks it.

**Expected:** compare `final_loss` against `ln(VOCAB_SIZE)` and either refuse or say so on
screen ("this run ended at the uniform-distribution baseline — it did not learn"). The
GeometryLab chip already shows `field entropy 2.83` with the tooltip "The test suite
requires ≥ 2.0" (`GeometryLab.svelte:343`), which invites exactly the wrong conclusion here.

**Would it have thrown?** No.

---

### F4. `POST /api/geo/model` accepts a structurally incomplete model file, then leaks a raw `KeyError` as a 500; the TS engine refuses the same file

**Severity:** medium
**Where:** `geo/bundle.py:120-148` — weights are decoded and hashed but never checked for
completeness or expected shapes, unlike `lib/geoEngine/index.ts:590` `validateWeightSet(ws)`

**Reproduce:** take `GET /api/geo/model?weights_token=learned`, keep only the `embedding`
tensor, recompute `weights_token` over what is left (the real `weights.weights_token`),
leave `vocab`/`vocab_sha256` untouched, `POST` it back.

**Observed:**
```
crafted token: 3ab115b3481dfdd2a3a3b67c57cf48fa
POST /geo/model (weights subset, digests all valid) -> (200, {'weights_token': '3ab115b3481dfdd2a3a3b67c57cf48fa', 'vocab_size': 1003})
/geo/trace -> 400 {"error":{"type":"InvalidParamError","message":"Weight set mismatch (missing: ['pos_embedding', 'layers.0.W_Q', …
/geo/vector_field -> 400 {"error":{"type":"InvalidParamError","message":"Weight set mismatch (missing: ['pos_embedding', …
/geo/weights -> 500 {"error":{"type":"InternalError","message":"'layers.0.W_V'","detail":{}}}
```
The static engine refuses the identical file up front:
```
TS importBundle(subset) REFUSED: InvalidParamError: Weight set mismatch (missing: pos_embedding, layers.0.W_Q, layers.0.W_K, …
```

**Expected:** `import_bundle` should run the same completeness/shape validation the engine
does, and `GET /api/geo/weights` should never surface a bare `KeyError` as a 500 —
`WeightLab.friendly()` renders `e.message`, so the user is shown the string
`'layers.0.W_V'`.

**Would it have thrown?** Partly — the load is silently accepted; two reads then 400
correctly and one 500s with an opaque message.

---

### F5. The from-scratch trainer has no cross-language golden at all, and `scratch.ts` claims one exists

**Severity:** low
**Where:** `lib/geoEngine/scratch.ts:15-17` — "which is why the golden test pins one
forward+backward step from a FIXED initialization rather than a whole run."
`tests/unit/geoScratch.test.ts` contains no such test: its assertions are corpus stats,
vocabulary equality, unit-norm initialization, `finalLoss < uniform - 1.0`, and argument
validation. `grep -r "repulsion\|REPULSION" code/frontend/tests` returns nothing, so the
Adam step and the Wang & Isola uniformity gradient — the two things fine-tuning does *not*
exercise — are pinned in neither language against the other. The fine-tune golden is the
only training pin and it is loose by design (`loss_before` to 1e-4, `loss_after` to 15 %,
`geoEngineFinetune.test.ts:85-98`).

**Measured divergence** (same corpus file, same seed 0, same 1 epoch, backend vs
`runScratchTrain`):
```
TS corpusStats: {"n_tokens":13200,"n_distinct":1200,"vocab_words_required":1000}
TS vocab === PY vocab? true
TS  final_loss: 6.894201045606132
PY  final_loss: 6.89567  (POST /api/geo/train_scratch, epochs=1, seed=0, same text)
TS weights_token: c9675b8e0d39cb467b24ead92b166ff1
PY weights_token: b7a4188a99a1a675434aad5d96ab7bf0
  max|PY-TS| embedding              = 1.9320341348648071
  max|PY-TS| layers.0.W_in          = 2.821442425251007
```
Two entirely different models (embedding rows are unit vectors, so 1.93 is near-antipodal)
whose reported losses agree to 0.0015. The *divergence* is honestly documented —
`InfoTab.svelte:1025-1026` says "streams are not portable, so two runs 'from the same seed'
are two independent runs of one recipe" — so this is a coverage gap plus a wrong code
comment, not a lie to the user. I did verify the formulas match by reading both
(`train.py:119-124` vs `scratch.ts:135-181`: same `t`, same 256-sample-with-replacement,
same `pairs = m(m−1)/2`).

**Would it have thrown?** No — and nothing would catch a real port bug here either.

---

### F6. The same model saved by the backend and by the browser are different files with different `vocab_sha256`

**Severity:** low
**Where:** `geo/bundle.py:85` uses `GeoTokenizer.to_json()`
(`json.dumps(..., ensure_ascii=True, sort_keys=True)`, `", "`/`": "` separators);
`lib/geoEngine/index.ts:500` uses `JSON.stringify` with a hand-written key order.

**Observed** — the *same* scratch model, exported by Python and re-exported by the TS
engine after `importBundle`:
```
PY vocab head: '{"format": "geo-tokenizer-v1", "specials": {"<eos>": 1, "<pad>": 2, "<unk>": 0}, "words": ["depa", "rupi", "vega", "fefo'
PY vocab sha: d42c313f4859045c
TS vocab head: "{\"format\":\"geo-tokenizer-v1\",\"specials\":{\"<unk>\":0,\"<eos>\":1,\"<pad>\":2},\"words\":[\"depa\",\"rupi\",\"vega\",\"fefo\",\"fune\",\"nuf"
TS vocab sha: d9bbae25e179331e
words equal: true
  re-export weights_token identical? true
  re-export weight bytes identical? true
  re-export vocab_sha256 identical? false
```
Both files load in both builds, so nothing is corrupt; but `vocab_sha256` is not a
model identity across builds, and a round-trip is not byte-identical. Worth pinning the
serialization if any future check compares files across modes.

**Would it have thrown?** No. Harmless today.

---

## What I tried that came back clean

- **Every digest defence on `POST /api/geo/model` holds.** Flipping one byte of the
  embedding, renaming one vocabulary word, deleting `vocab_sha256`, deleting
  `weights_token`, and declaring `version: 1` were each refused with a typed 400 and an
  accurate message ("its vocabulary hashes to 87f924df5ccea5be… but it declares
  d42c313f4859045c…", "its weights hash to e90e2e9689741068d9d2975b71940582 but it
  declares b7a4188a99a1a675434aad5d9…", "model file has no `vocab_sha256`…",
  "model file has no `weights_token`…", "model file version 1 is not supported…").
- **The `d6e9d5d` reload fix itself works.** Live: load a model file, reload the page,
  save again → `after-reload save: token 0fa8472bfb436d386aacc9cd5d7647ed == loaded token? true`,
  `after-reload save: vocab identical to the loaded file? true`.
- **Cross-mode portability works.** A Python-written scratch bundle imports into the TS
  engine with bit-identical weights and the identical `weights_token`
  (`re-export weight bytes identical? true`), and a browser-written bundle imports into
  the backend (`POST /geo/model … -> 200`). Only the JSON spelling of `vocab` differs (F6).
- **Weight-edit parity is exact.** The same edit on the same base mints the identical
  content hash in both engines (`8302aa402333525144dccf05fbb5e15c`).
- **Degenerate inputs fail loudly in both builds.** Empty text, a one-word corpus, and a
  nonexistent HF dataset: backend 400 "fine-tuning text is empty" / 400 "too short after
  tokenization (need at least 2 tokens)" / 422 "HuggingFace dataset
  'definitely/not-a-real-dataset-xyz' (split 'train') could not be loaded"; scratch 400
  "This text has only 0 distinct word types…" / "only 1 distinct word types…". The static
  engine raises the matching `InvalidParamError`s, and refuses `hf_dataset` inside
  `GeoEngine.finetune` (the staticClient resolves the dataset to text first, so the HF tab
  really does work in the browser as `FinetunePanel` claims).
- **The vector-field prose is accurate, including the parts that are easy to fudge.**
  `sequence_forces` are tangent-projected at the *token embedding* anchor with the removed
  radial magnitude reported as `normal_residual` and surfaced in the UI badge; the
  `antisymmetrize` toggle is correctly scoped to the per-point field only and
  `tangent_exact` tracks it; `layer="full"` is rejected for force mode. The arrow-length
  transform in `GeoScene.retarget` (p90 normalization plus a hard clip at 0.55/0.6) is
  described almost line for line at `GeometryLab.svelte:430-435` and
  `InfoTab.svelte:353-369` — "Arrow length is relative, not absolute… clipping anything
  longer… lengths do not compare across renders or between the two classes."
- **No console errors on the live static site** across a full session: load, train from
  scratch, save, fine-tune, save, load a file, reload, save (`console errors: []` twice).
