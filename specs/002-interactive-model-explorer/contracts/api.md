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

`?text=<str>` → `200 { "tokens": [ { "id": 17, "text": "alice", "unk": false }, … ],
"n_unk": <int>, "truncated": <bool> }` (truncated ⇒ input exceeded context_window).

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
  `Σ_{j≤i} softmax(⟨K z_j, Q z_i⟩)·V z_j` with their normal residual magnitudes.
  `layer="full"` ⇒ `400` (force mode is per-layer by definition).

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

Tokens are content hashes over the *full resulting weight set*: stateless,
deduplicating, valid across workers/restarts; artifacts LRU-evicted with the cache.

### POST /api/geo/finetune

← JSON `{ "text": "<str>" | null, "hf_dataset": "<repo id>" | null,
          "hf_split": "train", "max_samples": 200,
          "steps": <int ≤ 500 = 100>, "lr": <float = 1e-2>,
          "base": "learned" | "<weights_token>" }`
   — or multipart with a `.txt`/`.md` `file` field replacing `text`.
   Exactly one source of {text, file, hf_dataset}.
→ `202 { "job_id": "<id>", "ready": false }` (SSE `phase: "finetune"`; `done` data:
   `{ "weights_token": "<new hash>", "loss_before": <float>, "loss_after": <float> }`)
→ `200 { "weights_token": …, "loss_before": …, "loss_after": …, "ready": true }`
   on content-hash cache hit.
→ `400` no source / more than one source; `422` unusable dataset id.
Never mutates the canonical checkpoint.

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
