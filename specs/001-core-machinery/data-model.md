# Phase 1 Data Model: Core Project Machinery

Entities are derived from the spec's **Key Entities** and Functional Requirements.
Fields list the conceptual shape (not a DB schema — storage is the filesystem cache).
Validation rules trace to FR numbers. State transitions are given for the two
stateful entities (Cache Artifact, Precompute Job).

## ModelReference

What it represents: an open-weights model the system can load.

|Field|Type|Notes|
|-|-|-|
|`model_id`|str|HuggingFace id (e.g., `gpt2`); curated or user-entered (FR-001)|
|`revision`|str|Resolved commit/revision pin for reproducibility (FR-013)|
|`source`|enum|`curated` \| `user` (FR-001)|
|`display_name`|str|Human label for the selector|
|`capabilities`|object|`{ exposes_token_probs: bool, exposes_hidden_states: bool, num_layers: int, hidden_size: int, vocab_size: int }` (FR-002)|
|`status`|enum|`supported` \| `unsupported`; unsupported carries a `reason` (FR-003)|

Validation: a `ModelReference` may be used for computation only if
`status == supported` and both capability flags are true; otherwise the API returns
an `UnsupportedModelError` (FR-002/FR-003, FR-021). `revision` MUST be pinned before
any artifact is keyed (FR-013).

## EmbeddingSet

What it represents: per-layer embedding vectors for a model's reference token set —
the canonical source for reductions.

|Field|Type|Notes|
|-|-|-|
|`model_id` / `revision`|str|Provenance|
|`reference_set`|object|`{ kind: "full_vocab" \| "subset", token_ids: int[]? , size: int }`|
|`source`|enum|`static` (input-embedding matrix) \| `contextual` (hidden states)|
|`layer`|int|`0..num_layers` (0 = embedding/base layer) (FR-002)|
|`vectors`|float32[N][d]|N = reference-set size, d = `hidden_size`|

Validation: `vectors.shape == [reference_set.size, hidden_size]`; `layer` within
`0..num_layers`. Canonical artifact from which every ReducedCoordinateSet is
regenerable (FR-005).

## NextTokenDistribution

What it represents: probability over the next token for a given context + params.

|Field|Type|Notes|
|-|-|-|
|`model_id` / `revision`|str|Provenance|
|`context`|object|`{ prefix_text: str, token_ids: int[] }` (FR-015 prompt/context prefix)|
|`temperature`|float|`>= 0`; `0` ⇒ deterministic argmax (R1)|
|`probs`|float32[vocab]|Sums to 1.0 (± tol); or `{ top_k: [{token_id, prob}], tail_mass }`|
|`top_token`|int|argmax convenience field|

Validation: `temperature >= 0` (FR-021 reject negative); `probs` non-negative and
normalized to 1.0 within tolerance; values are **real model outputs**, never
placeholders (Principle I).

## ReducedCoordinateSet

What it represents: a low-dimensional view of an EmbeddingSet.

|Field|Type|Notes|
|-|-|-|
|`model_id` / `revision`|str|Provenance|
|`embedding_ref`|key|Cache key of the source EmbeddingSet (FR-005)|
|`dims`|enum|`2d` \| `3d_sphere`|
|`method`|enum|`pca` \| `umap` (2d); `mds` \| `pca3` (3d) (R4/R5)|
|`params`|object|method params (e.g., UMAP `n_neighbors`, `min_dist`)|
|`seed`|int|Fixed for reproducibility (FR-013)|
|`coords`|float32[N][2 or 3]|For `3d_sphere`, every row is on the unit sphere (R5)|

Validation: identical `{embedding_ref, method, params, seed}` ⇒ identical `coords`
(FR-013, SC-002). For `3d_sphere`, `‖coords[i]‖ ≈ 1` for all i.

## ReferenceGrid

What it represents: the n×n grid over the 2D space and its nearest-token reference
points (FR-011).

|Field|Type|Notes|
|-|-|-|
|`reduction_ref`|key|Cache key of the source 2D ReducedCoordinateSet|
|`n`|int|grid resolution (default 25); `n >= 2`|
|`vertices`|float32[n*n][2]|grid coordinates spanning the 2D bounds|
|`reference_token_ids`|int[n*n]|nearest token id per vertex (FR-011)|

Validation: `vertices.length == n*n == reference_token_ids.length`; each reference
token is the nearest reduced point to its vertex.

## CacheArtifact (stateful)

What it represents: a stored computation result and its provenance/integrity.

|Field|Type|Notes|
|-|-|-|
|`key`|str|hash of `{schema_version, model_id, revision, artifact_type, inputs, params, seed}` (R6)|
|`artifact_type`|enum|`embeddings` \| `distribution` \| `reduction_2d` \| `reduction_3d` \| `grid` \| `token_cloud` \| `vector_field` \| `sankey` \| `manifold`|
|`spec`|object|the full, sorted key spec (inspectable sidecar)|
|`checksum`|str|content hash of the payload (FR-007)|
|`schema_version`|int|bumped when artifact format changes (FR-007)|
|`complete`|bool|set true only after atomic write finishes (R6, edge: interrupted precompute)|
|`payload_path`|path|`.npz`/`.json` under `data/processed/cache/`|

State transitions:

```
(absent) --request--> WRITING --atomic-rename+flag--> COMPLETE
WRITING --crash/cancel--> (absent or PARTIAL)   # never read as COMPLETE
COMPLETE --checksum/schema mismatch on read--> INVALID --> recompute (FR-007)
COMPLETE --regenerate request--> rebuilt from canonical source (FR-005)
```

Validation on read: `complete == true` AND `schema_version` matches AND `checksum`
verifies; else treat as miss and recompute (FR-006/FR-007). Identical key ⇒
byte-identical payload (SC-002).

## PrecomputeJob (stateful)

What it represents: an in-flight or finished computation; drives progress +
single-flight concurrency (FR-008/FR-009).

|Field|Type|Notes|
|-|-|-|
|`job_id`|str|opaque id|
|`cache_key`|str|the artifact being produced (single-flight dedup key, FR-008)|
|`status`|enum|`queued` \| `running` \| `done` \| `error`|
|`progress`|float|`0.0..1.0`|
|`message`|str|human-readable current step|
|`error`|object?|`{ type, message }` on failure (FR-021)|

State transitions:

```
queued --start--> running --(progress ticks ≥1/s)--> done   # cache_key now COMPLETE
running --exception--> error (message surfaced verbatim; no fabricated result)
queued/running + identical cache_key requested --> attach to SAME job (FR-008)
```

## InteractionParameters

What it represents: the user-controllable inputs shared across visualizations
(FR-015); these are the inputs that, with the model, form cache keys.

|Field|Type|Notes|
|-|-|-|
|`model_id`|str|from the selector (incl. arbitrary id)|
|`prefix_text`|str|editable prompt/context prefix (may be empty)|
|`temperature`|float|`>= 0`|
|`layer`|int|`0..num_layers`|

Validation: bounds enforced client- and server-side; invalid combinations yield a
clear error and never a fabricated/stale-as-fresh result (FR-021, SC-007).

## Relationships

```
ModelReference 1──* EmbeddingSet 1──* ReducedCoordinateSet 1──1 ReferenceGrid(2d)
ModelReference 1──* NextTokenDistribution
every (EmbeddingSet|Distribution|ReducedCoordinateSet|ReferenceGrid) ⇒ one CacheArtifact
every CacheArtifact produced-by one PrecomputeJob
InteractionParameters ⇒ select/identify the artifacts to fetch
```
