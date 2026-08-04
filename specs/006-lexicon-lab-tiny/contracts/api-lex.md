# API Contract: Lexicon Lab — `/api/lex/*` (features 006, 007)

**Additive.** This file adds a new namespace; it does **not** change anything in the
frozen feature-002 contract (`specs/002-interactive-model-explorer/contracts/api.md`).
No existing endpoint's path, parameters, fields, or status codes are altered. Feature
002's rules are inherited verbatim:

- **Error envelope** — every failure is
  `{ "error": { "type": <string>, "message": <string>, "detail": {} } }`.
  Types used here: `InvalidParamError` (400), `NotFoundError` (404),
  `UnsupportedModelError` (422), `ComputeError` (500), `TrainingFailedError`
  (500, via job error events).
- **Array encoding** — nested JSON lists of finite floats, row-major, rounded to
  6 significant digits. A non-finite value is an error, never a `null`.
- **Jobs/SSE** — long work returns `202 { "job_id", "ready": false }` and streams through
  the existing `GET /api/jobs/{id}/events`. The Lexicon Lab's phase label is
  **`"lex_train"`** (the `phase` field is feature 002's optional additive one).
- **Cache-hit target** — a repeated request that hits the cache answers in `< 100 ms`.

The maths behind `/spectrum` is fixed by `specs/006-lexicon-lab-tiny/architecture.md`
("Spectrum"), which the TypeScript engine implements to the same numbers (SC-605).

`model_token` is a content hash over **weights + config + vocabulary**. Two models with
byte-identical weights but different word lists are different models and get different
tokens: the vocabulary is this tab's independent variable, and a shared token would let
a cache hit serve the wrong labels.

**Feature 007 additions**, marked as such where they appear below:
`POST /api/lex/vacancy` (new path) and an **optional** `vacancy` object on
`POST /api/lex/train` (absent ⇒ the endpoint is byte for byte what it was). Their
semantics are fixed by `specs/007-vacancy-transform-field/architecture.md`, which both
stacks implement; this file specifies only what goes on the wire.

Implementation: `code/backend/src/llm_geometry/api/routes_lex.py`.
Contract tests: `code/backend/tests/contract/test_api_lex.py`.
The static build serves the same surface from
`code/frontend/src/lib/staticClient/lex.ts` — the Lexicon Lab computes in the browser
in **both** modes, so nothing in this namespace is refused or approximated there.

---

## GET /api/lex/spec

Everything the tab needs before the user touches a control. No parameters.

→ `200 {
  "corpus": { "title": "The Real Mother Goose", "year": 1916, "gutenberg_id": 10607,
              "sha256": "<64 hex>", "bytes": 110445,
              "n_tokens": <int>, "n_distinct": <int>, "n_lines": <int>, "n_chars": <int> },
  "budget_sources": ["dolch", "frequency"],
  "budgets": [ { "name": "pre_primer", "size": 40, "rows": 44 }, … ],
  "special_tokens": { "<unk>": 0, "<bos>": 1, "<eos>": 2, "<pad>": 3 },
  "generation_banned_ids": [0, 1, 3],
  "model": { "d_model_choices": [16,32,64,128], "n_layer_choices": [1,2,3,4],
             "n_head_choices": [1,2,4], "ctx_choices": [32,64,128],
             "mlp_ratio": 4, "layer_norm_eps": 1e-5,
             "defaults": { "d_model", "n_layers", "n_heads", "ctx", "tied", "dropout",
                           "budget_source", "budget" } },
  "training": { "max_steps": <int>, "grad_clip_norm": 1.0, "val_fraction": <float>,
                "onecycle": { "pct_start", "div_factor", "final_div_factor" },
                "defaults": { "steps", "lr", "batch_size", "weight_decay", "seed",
                              "sample_every" } },
  "generation": { "max_new_tokens_limit": <int>,
                  "defaults": { "temperature", "max_new_tokens", "seed" } },
  "spectrum": { "pca_components": 3, "display_k": 48 } }`

`budgets[].size` is **measured** from the shipped Dolch lists, never quoted (FR-602).
The largest is `314`, not the widely-cited 315, because `Santa Claus` contains a space
and no word tokenizer can ever match it; it is dropped and the real count reported.

## GET /api/lex/budgets

Every budget at once, measured against the shipped corpus (US-1, FR-606, FR-612).

`?source=dolch|frequency` (default `dolch`)
`&d_model=&n_layers=&n_heads=&ctx=&tied=` — optional shape controls; they only affect
the `param_count` column, so the UI can show the size/cost trade-off live.

→ `200 {
  "source": "dolch",
  "corpus": { "title", "n_tokens", "n_distinct", "n_lines", "n_chars" },
  "model": { "d_model", "n_layers", "n_heads", "ctx", "tied", "dropout" },
  "budgets": [ { "source", "budget", "size", "rows",
                 "coverage": { "total_tokens", "in_budget_tokens", "distinct_types",
                               "oov_types", "total_lines", "whole_lines_in_budget",
                               "token_coverage", "unk_rate" },
                 "param_count": <int> }, … ] }`

`param_count` is the closed form verified against the source model on 7 configurations:
`N = (2 if untied else 1)·rows·d + ctx·d + L·(12d² + 13d) + 2d`.

Errors: `400 InvalidParamError` — unknown `source`, or a `d_model`/`n_layers`/`n_heads`/
`ctx` outside the enumerated choices, or `d_model % n_heads != 0`.

## POST /api/lex/coverage

One budget measured against one corpus (SC-603, US-3, US-5). POST rather than GET
because the corpus may be a whole pasted book.

← `{ "source": "dolch"|"frequency", "budget": "pre_primer"|"primer"|"first"|"service"|"full",
     "size": <int?>, "text": <str?>, "hf_dataset": <str?>, "hf_split": <str?>,
     "max_samples": <int?> }`

All fields optional. No `text`/`hf_dataset` ⇒ the shipped corpus. `size` applies only to
`source="frequency"` (a Dolch budget **is** its list, so its size is measured, not
chosen); sending it with `source="dolch"` is a 400. `hf_dataset` reuses feature 004's
HuggingFace loader and is resolved **synchronously**, so an unusable dataset id is a 422
on this request rather than a late job error.

→ `200 { "source", "budget", "size", "rows", "coverage": {…as above…},
         "corpus": { "n_tokens", "n_distinct", "n_lines", "n_chars" },
         "oov_sample": [ { "word": <str>, "count": <int> }, … ],
         "words": [<str>, …] }`

`oov_sample` is the most frequent out-of-budget types — the measurable form of what this
budget cannot say. It is a **sample** (at most 24 entries) and is labelled as one.

## POST /api/lex/train

Train from scratch, or fine-tune an existing model on new text (US-2, FR-619).
Idempotent and single-flight on a content-hash cache key.

← `{ "source", "budget", "size",                       // the budget (from-scratch only)
     "text" | "hf_dataset" (+ "hf_split", "max_samples"),   // corpus; default = shipped
     "d_model", "n_layers", "n_heads", "ctx", "tied", "dropout",   // shape (FR-611)
     "steps", "lr", "batch_size", "weight_decay", "seed", "sample_every",
     "base": "<model_token>" }`

With `base` set this is a **fine-tune**: the base model's shape *and vocabulary* are
used and travel with the result. Sending any shape or budget control alongside `base` is
a 400 rather than being silently ignored — feature 004's issue #6 was exactly that class
of silent substitution.

→ `200 { …result…, "ready": true }` on a cache hit (`< 100 ms`)
→ `202 { "job_id": "<id>", "ready": false }` otherwise.

SSE (`phase: "lex_train"`) progress messages read
`"step 120/400 · loss 3.412 · lr 2.71e-03"`, and every `sample_every` steps they append
` · <a real sample generated from the model as it stands>` (FR-618). The terminal `done`
event's data is the result object:

`{ "model_token": "<32 hex>", "first_loss", "final_loss", "val_loss", "steps", "seed",
   "elapsed_s", "n_tokens", "vocab_size", "vocab_rows", "param_count", "sample" }`

The 200 cache-hit body additionally carries `"history": [ { "step", "loss", "lr" }, … ]`.

Errors: `400 InvalidParamError` — `steps` outside `1..MAX_STEPS`, `lr <= 0`,
`batch_size < 1`, `weight_decay < 0`, `sample_every < 1`, an out-of-enum shape control,
a corpus with no word tokens, or a corpus shorter than one full context window.
`404 NotFoundError` — unknown `base`. `500 TrainingFailedError` (via the job error
event) — a real training failure, surfaced verbatim, never replaced by a partial result.

### Feature 007: the optional `vacancy` object

← `"vacancy": { "p", "seed", "consistent", "match_prosody", "reveal_after", "keep" }`

**Optional and additive: absent, everything above is unchanged, byte for byte.**
Present, the resolved corpus is vacated (`POST /api/lex/vacancy` below defines the
parameters) *before* training, and the model is trained under the vocabulary
`specs/007-vacancy-transform-field/architecture.md` §7.2 assigns it — **mapped** when
`consistent` and `reveal_after = 0`, **rebuilt** from the vacated corpus otherwise.

The transform runs server-side rather than in the client because `/api/lex/vacancy`
deliberately returns an excerpt: sending the whole rewritten corpus back just to train
on it would move ~86 kB per request in each direction.

With `base` set the base model's vocabulary is used unchanged, as it always is; only
the text is vacated.

Under the mapped condition this is a **pure relabelling**: the token id stream is
element-for-element identical, so `first_loss`, `final_loss` and `val_loss` are
*bit-identical* to the same run on the English corpus. That is the tiny arm's result
(architecture.md §7.3), not a caveat about it. The `model_token` still differs,
because the vocabulary is part of it.

The transform's parameters are in the cache key even though `(corpus, vocabulary)`
already determines the run — so that a knob added to the transform later cannot land
on an entry made before it existed.

Errors: `400 InvalidParamError` — `vacancy` not an object, or any parameter outside
the ranges given for `/api/lex/vacancy`.

## POST /api/lex/vacancy

**Feature 007.** The vacancy transform applied to a corpus, with the statistics of
`specs/007-vacancy-transform-field/architecture.md` §10. Additive: it adds a path and
changes nothing that existed. Same corpus-source and budget rules as
`/api/lex/coverage`, because the interesting question about a vacated corpus is always
"under which vocabulary?".

← `{ "source", "budget", "size",                                   // as /coverage
     "text" | "hf_dataset" (+ "hf_split", "max_samples"),          // as /coverage
     "p": <float ∈ [0,1] = 0>, "seed": <int = 0>,
     "consistent": <bool = true>, "match_prosody": <bool = true>,
     "reveal_after": <int ≥ 0 = 0>, "keep": [<str>, …],
     "preview_chars": <int ∈ 0..20000 = 2000> }`

All fields optional. The five transform knobs are architecture.md §7.1's, in this
API's `snake_case`; `keep` must be a **list**, since a bare string would be read
letter by letter and quietly protect six single letters.

→ `200 { "p", "seed", "consistent", "match_prosody", "reveal_after", "keep": [<str>, …],
         "vocabulary_rule": "mapped" | "rebuilt",
         "words": [<str>, …],
         "budget": { "source", "budget", "size", "rows", "coverage": {…} },
         "corpus": { "n_tokens", "n_distinct", "n_lines", "n_chars" },
         "vacancy_stats": { …§10's 23 fields, camelCase… },
         "bijective": <bool>, "remint_rounds": <int>,
         "preview": <str>, "original_preview": <str>,
         "preview_chars": <int>, "truncated": <bool>,
         "vacated_chars": <int>, "vacated_sha256": "<64 hex>",
         "original_chars": <int>, "original_sha256": "<64 hex>" }`

**An excerpt and a digest, never the whole vacated corpus.** The shipped corpus is
~86 kB of body text and the panel re-runs this on every tick of the `p` slider, so
returning it whole would put megabytes on the wire across one sweep to show a reader a
screenful. Nothing needs it whole: the panel shows an excerpt (the source's own figure
is its first 400 characters), and a caller that wants to *train* on the vacated corpus
sends the same parameters to `/api/lex/train`, which vacates in place. What an excerpt
cannot do by itself is prove which text it came from, so `vacated_sha256` covers all of
it in 64 characters — and that digest is the single value the static build's
in-browser transform is checked against.

`vacancy_stats` carries §10's field names **verbatim**, camelCase inside this API's
snake_case envelope on purpose: they are a cross-language contract between
`llm_geometry/lex/vacancy.py` and `lexEngine/vacancy.ts`, not this API's naming. An
unprefixed `types*` is forbidden there; every count names its scope (`domainTypes*` vs
`corpusTypes*`). `bijective` and `remint_rounds` also appear at the top level, because
injectivity is the guarantee the mapped vocabulary rests on and a caller checking it
should not have to reach into a statistics block.

`vocabulary_rule` says which of §7.2's two rules produced `words`, and a client must
not have to infer it from the parameters: `"mapped"` is the only condition under which
the ids are the English ids, and that is the difference between an invariance result
and a coverage collapse.

Every number returned is measured on the corpus in the request. The source document's
own prosody figures are its numbers on a corpus we do not have and are transcribed
nowhere.

Errors: `400 InvalidParamError` — `p` outside `[0, 1]` or not a number,
`reveal_after < 0`, `preview_chars` outside `0..20000`, `keep` not a list of strings,
`size` with `source="dolch"`, an unknown budget, a corpus with no word tokens, or both
`text` and `hf_dataset`.

Parity: `code/frontend/tests/fixtures/vacancy-api-golden.json` is a transcript of this
route (`python scripts/export_vacancy_api_golden.py`, real app, real corpus, no mocks).
`test_api_lex.py` asserts the live route still returns it and
`tests/unit/staticVacancy.test.ts` asserts the browser's in-page implementation
reproduces it field for field, so neither stack can drift alone.

## GET /api/lex/spectrum

The geometry of a trained model's embedding (FR-620..FR-623).

`?model_token=<hash>` (required) `&matrix=embedding|readout` (default `embedding`)
`&baseline=true|false` (default `true`) `&baseline_seed=<int>` (default 0)

→ `200 {
  "model_token", "matrix", "tied": <bool>, "projection": "pca", "display_k": 48,
  "tokens": [<str>, …],                       // itos, so the cloud can be labelled
  "spectrum": {
    "rows": <V>, "d_model": <d>, "max_rank": <min(V-1, d)>,
    "eigenvalues": [<d floats desc>], "singular_values": [<d floats desc>],
    "explained_variance": [<d floats desc, sums to 1>], "total_variance": <float>,
    "effective_rank": <float>, "stable_rank": <float>, "participation_ratio": <float>,
    "frac_var_top2": <float>, "frac_var_top10": <float>, "n_dims_for_90pct": <int>,
    "pca_coords": [[x, y, z], … V rows],
    "pca_explained_variance_ratio": [<3 floats>],
    "degenerate": <bool> },
  "baseline": { "rows", "d_model", "max_rank", "effective_rank", "stable_rank",
                "participation_ratio", "frac_var_top2", "frac_var_top10",
                "n_dims_for_90pct", "total_variance", "degenerate" },
  "comparison": { "effective_rank_delta", "stable_rank_delta",
                  "participation_ratio_delta", "frac_var_top2_delta", "max_rank",
                  "effective_rank_frac_of_ceiling",
                  "baseline_effective_rank_frac_of_ceiling" } }`

- `baseline` is an **untrained model at the same shape** (FR-622). It is on by default
  because effective rank climbs with `|V|` for random matrices too: a trained curve
  without the random control cannot distinguish learning from arithmetic (SC-604).
- `comparison.*_delta` are signed. Training frequently *lowers* effective rank by
  concentrating variance, and the panel says so rather than only reporting increases.
- `projection: "pca"` is part of the payload because FR-623 requires the token cloud to
  be labelled a projection — unlike the Geometry Lab's sphere, which is native 3-D.
- `pca_coords` use a fixed sign convention (each eigenvector's largest-magnitude entry is
  positive) so Python and the browser agree on them; see `lex/spectrum.py`.

Errors: `404 NotFoundError` — unknown `model_token`. `400 InvalidParamError` —
`matrix` outside the enum, or `matrix="readout"` on a **tied** model (a tied model's
readout *is* its embedding, so it has exactly one spectrum; the source project logged
that one matrix as two spectra and this refuses to).

## POST /api/lex/generate

Generate text from a trained model (FR-605, SC-602).

← `{ "model_token": "<hash>", "prompt": <str>, "temperature": <float>,
     "max_new_tokens": <int>, "seed": <int>, "stop_at_eos": <bool> }`

→ `200 { "model_token", "prompt", "text", "words": [<str>, …], "n_words",
         "out_of_budget": [],
         "prompt_tokens": [ { "text", "id", "unk" }, … ],
         "temperature", "seed", "vocab_size", "final_loss" }`

`out_of_budget` is **always empty**: the vocabulary *is* the budget and
`<unk>`/`<bos>`/`<pad>` are masked, so in-budget output is guaranteed by construction —
no trie, no post-filter. It is returned, and checked server-side, because a guarantee
nobody verifies is a guarantee nobody can trust; if it were ever non-empty the response
is a `500 ComputeError`, never a quietly filtered string.

`<eos>` renders as a line break (a corpus of nursery rhymes is line-shaped).
Out-of-budget *prompt* words become `<unk>` on the way in and are flagged
`"unk": true` — legitimate input, reported rather than hidden.

Errors: `400 InvalidParamError` — missing `model_token`, `temperature < 0`,
`max_new_tokens` outside `1..MAX_NEW_TOKENS`. `404 NotFoundError` — unknown token.

## GET /api/lex/model

The whole model as one portable, self-describing bundle (US-8, SC-607).

`?model_token=<hash>` (required)

→ `200 { "format": "llm-geometry/lex-model", "version": 1, "model_token": "<32 hex>",
         "weights_token": "<32 hex>", "vocab_sha256": "<64 hex>",
         "config": { "vocab_rows", "d_model", "n_layers", "n_heads", "ctx", "tied",
                     "dropout" },
         "vocab": { "source", "budget", "words": [<str>, …], "specials": [<str>, …] },
         "metrics": { … the training numbers … },
         "weights": { "<name>": { "shape": [<int>, …], "data": "<base64 float32 LE>" }, … } }`

Weights are base64 little-endian float32, as the Geometry Lab's bundle already does —
a 6-significant-digit JSON array would not round-trip the weights bit-for-bit, and
SC-607 requires a reloaded model to reproduce its generation *exactly*.

**Tensor names are the PyTorch model's** (`blocks.N.*`). The browser engine calls the
same tensors `layers.N.*` and translates at the file boundary
(`src/lib/lexEngine/bundle.ts`), so one format serves both runtimes rather than one tag
meaning two payloads.

### The three digests (amended by feature 006's US-8 work)

`weights_token` and `vocab_sha256` were **added** to this response. Rationale: with a
single joint `model_token`, a mismatch says only "something is wrong" — it cannot say
which half, and there is no id for a weight set independent of the word list it happens
to be paired with, which is exactly the id the Weight Lab (US-6) mints for an edited
model. All three are computed as:

- `model_token` — `sha256(canonical(config) ‖ canonical(words) ‖ Σ_sorted[name ‖
  repr(shape) ‖ f32-LE])[:32]` (`routes_lex.py::_model_token`, unchanged).
- `weights_token` — the same weight term **alone**, `[:32]`. Same construction as
  `geo/weights.py::weights_token`.
- `vocab_sha256` — `sha256(canonical({budget, source, specials, words}))`, full 64 hex.

`canonical` is `cache/keys.py::_canonical` (sorted keys, no whitespace, `ensure_ascii`).
`metrics` is deliberately outside every digest: it is the one block that cannot mislabel
a token.

## POST /api/lex/model

Load a bundle written by `GET /api/lex/model`.

← the bundle object → `200 { "model_token", "config", "vocab_size", "vocab_rows",
"param_count" }`

Validation is strict and loud:

- `format` and `version` must match, or `400`.
- `vocab.words` length + specials must equal `config.vocab_rows`, or `400` — weights and
  a word list that describe different models must never be joined silently.
- `tied` travels in the bundle and is enforced by the weight names it implies: a tied
  bundle carrying a `head_w`, or an untied one missing it, is a `400`. The source
  project's `probe.py` dropped `tie` on reload and silently reloaded a tied checkpoint as
  an untied model; the reloaded thing was then not the model that had been saved.
- **All three digests are MANDATORY and each must equal a re-hash of the bundle's own
  contents, or `400`.** A file whose weights and label disagree is refused, not repaired.

  *Amended by feature 006's US-8 work.* This previously read "if `model_token` is present
  it must equal a re-hash" — and a bundle with the field simply **deleted** loaded
  cleanly, verified by nothing. That is precisely the hole feature 004 shipped in
  `geo/bundle.py` and had caught by a red team; leniency-on-absence is not a lesser form
  of checking, it is an opt-out an attacker controls. Missing is now treated exactly like
  wrong, with its own message, on both sides of the wire.

Both implementations of this endpoint — `api/routes_lex.py` and the browser's
`src/lib/lexEngine/bundle.ts` — write and read the same payload. That is measured, not
asserted: `tests/unit/lexBundle.test.ts` pins the TS `model_token` against one produced by
real Python, and a bundle written by the browser engine imports into the FastAPI app,
returns the identical `model_token`, generates the identical greedy text, and re-exports
byte-identical weights.
