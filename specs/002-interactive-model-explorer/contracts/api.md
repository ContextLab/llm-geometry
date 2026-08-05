# API Contract: Interactive Model Explorer (FROZEN)

Extends the 001 contract (`specs/001-core-machinery/contracts/api.md`): same error
envelope, same cache/jobs/SSE machinery, same `< 100 ms` cache-hit target. This file
is **frozen for feature 002**: `dataClient.ts`, `routes_geo.py`, `routes_arch.py`,
and all contract tests implement exactly this. Changes require editing this file
first, in its own commit, with a note in the issue.

New error `type`s (envelope unchanged): `ModelTooLargeError` (422),
`InvalidWeightEditError` (422), `TrainingFailedError` (500 via job error events).

SSE progress events MAY now carry an optional `"phase"` field
(e.g. `"download" | "graph" | "train" | "finetune"`) — additive, backward-compatible.

Array encoding: all tensors are nested JSON lists of finite floats, row-major,
rounded to 6 significant digits. Shapes documented as `(rows, cols)`.

## Additive namespaces (this file's endpoints are unchanged)

Later features add whole namespaces beside `/api/geo/*` and `/api/arch/*`. Nothing
below is altered by them — no path, parameter, field, status code, or error type in
this document changes meaning — and each addition is recorded here so that "frozen"
means *frozen*, not *undocumented*:

| Added by | Namespace | Contract |
|-|-|-|
| 006 | `/api/lex/*` (Lexicon Lab) | `specs/006-lexicon-lab-tiny/contracts/api-lex.md` |
| 007 | `POST /api/lex/vacancy`, `vacancy` on `POST /api/lex/train` | same file, "Feature 007" section |

**Why 007 needed an addition rather than a parameter on something existing.** The
vacancy transform rewrites a *corpus*, and every existing endpoint here takes a model
or a prompt. Folding it into `/api/lex/coverage` would have made that endpoint's
response mean two different things depending on a flag — the exact failure this file
is frozen to prevent. The one existing endpoint that did change, `/api/lex/train`,
gained an **optional** object whose absence is byte-for-byte the previous behaviour,
because a vacated corpus has to be tokenized under the vocabulary the transform
assigns it and shipping the ~86 kB rewritten text back and forth to achieve that
would have been a worse contract than a parameter.

---

## Geometry Lab — `/api/geo/*`

The tiny model ("GeoTransformer") is fixed-architecture: `d_model=3`, `n_layers=4`,
`n_heads=1`, `mlp_hidden=12`, `vocab_size=1003` (1000 words + `<unk>` `<eos>` `<pad>`),
`context_window=50`, weight-tied unembedding, embeddings unit-norm on S².
`layer` params take `0..3` or `"full"`. `weights_token` is always optional; omitted ⇒
canonical learned checkpoint.

### GET /api/geo/spec

→ `200 {
  "model": { "d_model": 3, "n_layers": 4, "n_heads": 1, "mlp_hidden": 12,
             "vocab_size": 1003, "context_window": 50, "tied_unembedding": true,
             "corpus": "<corpus id>", "seed": 0 },
  "special_tokens": { "unk": 0, "eos": 1, "pad": 2 },
  "checkpoint": { "status": "ready" | "missing" | "training",
                  "checkpoint_id": "<hash>" | null,
                  "final_loss": <float> | null,
                  "coverage_uniformity": <float 0..1> | null,
                  "field_directional_entropy": <float> | null,
                  "job_id": "<id>" | null } }`

`checkpoint.status == "missing"` ⇒ client POSTs `/api/geo/train`. `"training"` ⇒
subscribe to `job_id` SSE.

### POST /api/geo/train

Idempotent, single-flight. ← `{ "seed": 0 }` (optional; default 0)
→ `200 { "checkpoint_id": "<hash>", "status": "complete", "ready": true }` (cache hit)
→ `202 { "job_id": "<id>", "ready": false }` (progress via `/api/jobs/{id}/events`,
   `phase: "train"`, messages like `"epoch 7/30 · loss 4.12"`; `done` event data
   includes `{ "checkpoint_id": "<hash>" }`).

### GET /api/geo/tokenize

`?text=<str>&weights_token=<hash?>` → `200 { "tokens": [ { "id": 17, "text": "alice",
"unk": false }, … ], "n_unk": <int>, "truncated": <bool> }` (truncated ⇒ input exceeded
context_window).

`weights_token` is additive (feature 004) and OPTIONAL: models trained from scratch
carry their own vocabulary, so their ids — and which words come back `unk` — depend on
which model is active. Omitted ⇒ the canonical vocabulary, exactly as before.

### POST /api/geo/train_scratch

Train a BRAND NEW model on the caller's own corpus. Same three sources as
`/api/geo/finetune`: JSON `{ "text": <str> }`, multipart with a `.txt`/`.md` `file`
field, or JSON `{ "hf_dataset": <id>, "hf_split"?, "max_samples"? }` — exactly one.
Optional `epochs` (1…60, default 12).

Unlike fine-tuning, this builds a **fresh vocabulary from the supplied text** and
freshly initialized weights: the result is a different model whose token ids mean
different words. The canonical checkpoint is never modified.

→ `200 { "weights_token", "vocab_size", "final_loss", "uniform_baseline", "learned",
"n_tokens", "n_distinct", "epochs", "seed", "ready": true }` on a content-hash cache
hit, else `202 { "job_id", "ready": false }` with SSE `phase: "train_scratch"` and the
same fields on the `done` event.

`uniform_baseline` is `ln(vocab_size)` — the cross-entropy a model reaches by learning
nothing — and `learned` is `final_loss < uniform_baseline − 0.5`. **Both are additive
(amended 2026-08-04, red-team 007 F3).** Training on structureless text correctly ends
*at* the baseline and the trainer is right not to refuse it, but the response carried
nothing that distinguished such a run from one that learned, and a client had no honest
way to say so. The two non-degeneracy metrics on `/api/geo/spec` cannot substitute:
they guard against collapse, and a model that learned nothing scores *better* on both.
A client MUST surface `learned: false` rather than presenting the run as a trained
model. Older cached entries predate the fields and may omit them; absent means unknown,
never `true`.

→ `400 InvalidParamError` when the text has fewer distinct word types than the
vocabulary is wide. The model's vocabulary size is an architectural dimension, so a
short corpus cannot fill it; the error names the shortfall instead of padding the
vocabulary with placeholders.

### GET /api/geo/corpus_stats

`?text=<str>` → `200 { "n_tokens": <int>, "n_distinct": <int>,
"vocab_words_required": <int> }`. Lets a client show whether a corpus is big enough to
train on *before* submitting it.

### GET /api/geo/model

`?weights_token=<hash|"learned">` → `200` a portable, self-describing bundle:

`{ "format": "llm-geometry/geo-model", "version": 3, "weights_token": <hash>,
   "config": { "d_model", "n_layers", "n_heads", "mlp_hidden", "vocab_size",
               "context_window" },
   "vocab": <tokenizer JSON string>,
   "vocab_sha256": <hex SHA-256 of the `vocab` string>,
   "weights": { "<name>": { "shape": [...], "data": <base64 float32-LE> }, … } }`

The vocabulary travels WITH the weights because a scratch-trained model's ids are
meaningless without it — and so does any model DERIVED from such a one by fine-tuning
or a weight edit. Where a model's vocabulary cannot be recovered, this endpoint returns
`400 InvalidParamError` rather than substituting the shipped one.

`vocab` is a **canonical serialization**: keys sorted, compact `,`/`:` separators, and
every non-ASCII character escaped (`\uXXXX`). `vocab_sha256` is the digest of exactly
those bytes, so the Python backend and the in-browser build write byte-identical files
for the same model.

*(`vocab_sha256` was shipped by feature 004's vocabulary-integrity fix but never written
down here; the canonical serialization and the derived-model rule were added 2026-08-04
for red-team 007 F1/F6. Recorded now so "frozen" means frozen, not undocumented.)*

*(`version: 2` → `3`, 2026-08-04, red-team 007 round 5 F10. `weights_token` changed
meaning when it began hashing the word list as well as the weights — so the FIELD in the
file changed meaning, and the format had to move with it. It did not, and the consequence
was that every file written before the change, for a model with its own vocabulary, failed
the re-hash and was refused as `this model file is corrupt` — an accusation against an
intact file, and against the very file the cache's schema-bump message tells the reader to
open. `POST /api/geo/model` therefore reads **versions 2 and 3**: a version-2 payload is
checked against the weights-only hash its own format put in it, and its current identity is
re-derived from the (weights, word list) pair it carries. Refusing it instead would strand
an intact file and buy nothing — the binding a version-3 token provides is absent from
every version-2 file, including the ones that load unchanged because their word list is the
shipped one and takes no part in either hash. A file that DECLARES version 3 is held to
version 3.)*

### POST /api/geo/model

← a bundle from `GET /api/geo/model` → `200 { "weights_token", "vocab_size" }`.

Validation is strict, and in this order: format, version, every `config` field, per-
tensor decode, **completeness and shape of the whole weight set**, the `vocab` block and
a re-hash of it against the declared `vocab_sha256`, then a re-hash of the decoded
weights **and that vocabulary** against the declared `weights_token`. Any failure is
`400 InvalidParamError`, never a partial load — pairing the wrong vocabulary with a set
of weights would make every label in the UI quietly wrong.

*(The vocabulary now precedes the token check because it is one of that hash's inputs —
amendment of 2026-08-04 with the token change above. It closes the one hole the two
digests could not: a file with genuine weights, a substituted word list, and
`vocab_sha256` recomputed over the substitute used to load with a 200, because each
digest verified what it covered and nothing covered the pair.)*

*(The completeness/shape step is an amendment of 2026-08-04, red-team 007 F4: a bundle
carrying one tensor, with every digest honestly recomputed over that one tensor, was
accepted with a `200` and later surfaced as a `500` whose entire message was the bare
string `'layers.0.W_V'`. A hash only says the bytes are the bytes the file declares; it
says nothing about whether they form a model. `GET /api/geo/weights` on such a set is
now also a typed `400`, never a raw `KeyError`.)*

### GET /api/geo/trace

`?prompt=<str>&weights_token=<hash?>`
→ `200 {
  "tokens": [ { "id": …, "text": …, "unk": … } ],            # T ≤ 50
  "embeddings": [[x,y,z] × T],                                # input embeddings (unit-norm)
  "layers": [ { "layer": 0,
      "attention": [[…] × T] (T,T),                           # row-stochastic, causal
      "q": [[x,y,z] × T], "k": [[…] × T], "v": [[…] × T],
      "hidden_in": [[…] × T], "attn_out": [[…] × T],
      "mlp_out": [[…] × T], "hidden_out": [[…] × T] }, × 4 ],
  "probs": [<float> × 1003],                                  # next-token distribution
  "logits_topk": { "ids": [×10], "texts": [×10], "probs": [×10] },
  "next_token": { "id": …, "text": … } }`
→ `400` empty prompt after tokenization.

### GET /api/geo/vector_field

`?mode=next_next|force&layer=0|1|2|3|full&prompt=<str>&weights_token=<hash?>
 &temperature=<float=0>&top_m=<int=1>&antisymmetrize=<bool=false>`

→ `200 {
  "mode": "next_next" | "force", "layer": "full" | <int>,
  "points": [[x,y,z] × V],                 # embedding of every vocab token, V=1003
  "token_ids": [<int> × V],
  "arrows": [ { "origin_index": <int>,     # index into points
                "vec": [dx,dy,dz],
                "weight": <float 0..1> } … ],
  "sequence_forces": [ { "position": <int>, "vec": [dx,dy,dz],
                         "normal_residual": <float> } … ] | null,   # force mode only
  "tangent_exact": <bool> }                # true iff force mode + antisymmetrize
`

- `next_next`: for each vocab token *v* appended hypothetically to the prompt, arrow(s)
  from `points[v]` toward the embedding(s) of the following-token prediction;
  `temperature=0 & top_m=1` ⇒ one argmax arrow per point; otherwise `top_m` arrows
  weighted by probability. `layer` selects which layer's residual stream produces the
  prediction (per 001 conventions); `"full"` = final layer.
- `force`: per-point field `W_V·z` for the selected layer (`antisymmetrize=true` uses
  `(W_V−W_Vᵀ)/2`; exactly tangent), plus per-sequence-position aggregate forces
  `Σ_{j≤i} softmax(⟨K z_j, Q z_i⟩)·V z_j`.
  `layer="full"` ⇒ `400` (force mode is per-layer by definition).
- **`sequence_forces[].vec` is the aggregate force projected onto the tangent plane at
  its anchor point `z_i`**, and `normal_residual` is the magnitude of the radial
  component that projection removed (amended 2026-07-26, feature 004). Before the
  amendment `vec` carried the unprojected sum, which rendered as arrows visibly leaving
  the sphere they are anchored to. Note that `antisymmetrize` does not fix this: each
  term `W_V z_j` is tangent at `z_j`, not at the `z_i` where the sum is drawn. The
  projection is a display choice and `normal_residual` is what keeps it honest — a
  client MUST surface it rather than presenting `vec` as the whole force.

### GET /api/geo/weights

`?weights_token=<hash?>&layer=<int>&matrix=W_Q|W_K|W_V|W_O|embedding`
→ `200 { "values": [[…]], "shape": [3,3] | [1003,3] | [3,12]…,
         "source": "learned" | "edited" | "preset:<name>" }`
(`matrix=embedding` ignores `layer`.)

### POST /api/geo/weights

← `{ "base": "learned" | "<weights_token>",
     "edits": [ { "layer": <int>, "matrix": "W_Q|W_K|W_V|W_O|embedding",
                  "preset": "identity|toeplitz_fuzzy|random|random_autocorr|zero|learned" | null,
                  "values": [[…]] | null,      # exactly one of preset/values
                  "seed": <int=0> } … ] }`
→ `200 { "weights_token": "<content-hash>",
         "edited": [ { "layer": …, "matrix": …, "source": … } … ] }`
→ `422 InvalidWeightEditError` (bad shape, bad name, both/neither preset+values,
   non-finite values).

Tokens are content hashes over the *full resulting weight set* **and the vocabulary its
token ids mean**: stateless, deduplicating, valid across workers/restarts; artifacts
LRU-evicted with the cache. A model that reads under the shipped vocabulary hashes the
weights alone (so `checkpoint_id` is unchanged); a model with a word list of its own —
trained from scratch, imported from a file, or derived from one of those — hashes the
weights, a fixed separator, and the canonical vocabulary serialization.

*(Amendment of 2026-08-04, red-team 007 F1 third path. The hash used to cover the weights
alone, and the artifact store deduplicates on it and wrote its metadata first-write-wins,
so two models with identical weights and different word lists shared ONE vocabulary:
loading a model file and then training from scratch to the same weights discarded the
scratch run's 1,000 words and saved its file under the loaded file's list, with every
digest recomputed over the substitute and therefore verifying. The two stacks resolved the
collision in opposite directions — Python kept the first word list, the browser engine the
last — so which model was corrupted depended on which build wrote the file. Identical
weights with different vocabularies are different models; a hash that says otherwise is
the defect, not the caching policy.)*

### POST /api/geo/finetune

← JSON `{ "text": "<str>" | null, "hf_dataset": "<repo id>" | null,
          "hf_split": "train", "max_samples": 200,
          "steps": <int ≤ 500 = 100>, "lr": <float = 1e-2>,
          "base": "learned" | "<weights_token>" }`
   — or multipart with a `.txt`/`.md` `file` field replacing `text`.
   Exactly one source of {text, file, hf_dataset}.
→ `202 { "job_id": "<id>", "ready": false }` (SSE `phase: "finetune"`; `done` data:
   `{ "weights_token": "<new hash>", "loss_before": <float>, "loss_after": <float>,
      "n_tokens": <int>, "n_unk": <int>, "unk_rate": <float> }`)
→ `200 { "weights_token": …, "loss_before": …, "loss_after": …, "n_tokens": …,
   "n_unk": …, "unk_rate": …, "ready": true }` on content-hash cache hit.
→ `400` no source / more than one source; `422` unusable dataset id.
→ `400 InvalidParamError` when **90 % or more** of the tokenized text is `<unk>` under
   the base model's vocabulary. (The bound was `> 90 %` until 2026-08-04, so a stream
   that was EXACTLY 90 % unknown was accepted and reported as a clean loss drop, one
   token below a refusal whose message rounds to the same "(90%)".)
Never mutates the canonical checkpoint.

**Amended 2026-08-04 (red-team 007 F1/F2, issue #6).** Two changes, both to stop a
believable number coming out of a meaningless run:

1. The text is tokenized with **`tokenizer_for(base)`**, not the canonical tokenizer.
   Previously a scratch-trained model's own corpus encoded to a stream that was 100 %
   `<unk>`, and the response still reported a clean loss drop. `n_tokens` / `n_unk` /
   `unk_rate` are **additive** and report how much of the text that vocabulary actually
   knew; a client showing the loss MUST be able to show them (absent ⇒ unknown, from a
   cache entry that predates the fields — never reported as zero). At 90 % or above the
   request is refused instead.
2. The minted checkpoint **inherits `base`'s vocabulary**, as `POST /api/geo/weights`
   also now does. This is not a response-shape change; it is what `GET /api/geo/model`
   then writes into the bundle. Previously a fine-tune or an edit of a model with its
   own word list saved under the shipped Alice vocabulary, with `vocab_sha256` computed
   over the substituted list — so the file verified and every label was wrong. Where a
   vocabulary cannot be recovered, `GET /api/geo/model` now returns
   `400 InvalidParamError` rather than substituting one.

---

## Architecture Explorer — `/api/arch/*`

`model_id` follows 001 rules (resolve/capability-gate first). Additionally, models
with `total_params > ARCH_MAX_PARAMS` (config; default 1.5e9) →
`422 ModelTooLargeError` **before** any download.

### GET /api/arch/graph

`?model_id=<id>`
→ `200 {
  "model_id": …, "schema_version": <int>,
  "meta": { "n_layers": …, "hidden": …, "heads": …, "kv_heads": …, "vocab": …,
            "total_params": …, "traced_seq_len": <int> },
  "nodes": [ { "id": "<stable dotted path>",       # e.g. "model.layers.0.self_attn.q_proj"
               "kind": "embedding|linear|layernorm|rmsnorm|rope|attention_softmax|
                        residual_add|activation|mlp|lm_head|other",
               "op": "module" | "functional",
               "label": "<human label>",
               "layer": <int> | null, "group": "stem" | "layer_<k>" | "head",
               "params": [ { "name": "weight|bias", "shape": [r,c],
                             "param_path": "<state_dict key>",
                             "tied_to": "<param_path>" | null } ] } … ],
  "edges": [ { "from": "<node id>", "to": "<node id>",
               "tensor_shape": ["T", <int>] } … ] }`

Built from a traced real forward pass (hooks/`torch.fx`): functional ops (RoPE,
attention softmax, residual adds, activations) appear as `op: "functional"` nodes.
Tied tensors appear once, aliased via `tied_to`.

### GET /api/arch/weights

`?model_id=<id>&param=<param_path>&r0=<int=0>&r1=<int?>&c0=<int=0>&c1=<int?>
 &max_cells=<int=4096>`
→ `200 { "param": …, "shape": [R,C], "r0": …, "r1": …, "c0": …, "c1": …,
         "downsampled": <bool>, "grid_shape": [gr,gc], "values": [[…] gr×gc],
         "stats": { "min": …, "max": …, "mean": …, "std": … },
         "method": "exact" | "strided_mean" }`
Window ≤ `max_cells` ⇒ exact values (`downsampled: false`). 1-D params (biases,
norms) use `C=1`. → `404` unknown `param_path`.

### GET /api/arch/trace

`?model_id=<id>&prompt=<str>&system_prompt=<str?>&max_context=<int=64>`
→ `200 {
  "tokens": [ { "id": …, "text": … } ],               # T ≤ max_context (truncates left)
  "chat_template_used": <bool>,
  "truncated": <bool>,                                # prompt exceeded max_context (left-truncated)
  "layers": [ { "layer": <int>,
      "attention": [heads][T][T],                     # downsampled to ≤64×64 per head
      "attention_downsampled": <bool>,
      "hidden_norm": [<float> × T],                   # L2 of residual stream out
      "hidden_pca3": [[x,y,z] × T] } … ],             # 3-D PCA of hidden states (viz aid)
  "logits_topk": { "ids": [×10], "texts": [×10], "probs": [×10] },
  "node_activations": [ { "node_id": …, "out_norm": <float>,
                          "out_shape": ["T", …] } … ] }   # one entry per traced node
`
`node_activations` covers **every** graph node (completeness invariant shared with
`/api/arch/graph`).

### POST /api/arch/generate

← `{ "model_id": …, "prompt": "<str>", "system_prompt": "<str>" | null,
     "temperature": <float = 0.8>, "max_new_tokens": <int ≤ 128 = 64>,
     "seed": <int> | null }`
→ `200 { "text": "<full decoded reply>",
         "tokens": [ { "id": …, "text": …, "prob": <float>,
                       "topk": { "ids": [×5], "texts": [×5], "probs": [×5] } } … ],
         "finish_reason": "eos" | "length" }`
Single-turn (v1). Uses the model's chat template when available.
