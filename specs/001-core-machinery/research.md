# Phase 0 Research: Core Project Machinery

Decisions that resolve the Technical Context. Each is recorded as Decision /
Rationale / Alternatives. Load-bearing API specifics are confirmed for real by the
integration tests in implementation (Constitution Principle I — real verification);
anything not yet exercised by a real run is marked `TODO(verify:)`.

## R1. Extracting next-token probability distributions

- **Decision**: Run a forward pass on the tokenized context, take the final-position
  logits `logits[:, -1, :]`, divide by temperature `T` (clamped to a small floor to
  avoid div-by-zero), and apply softmax to obtain a probability over the full
  vocabulary. `T = 0` is treated as the deterministic argmax (probability mass 1.0
  on the top token). Distributions are stored sparse-friendly (full vector for
  small vocab; top-k + tail-mass option reserved for large vocab).
- **Rationale**: This is the canonical, exact next-token distribution an open-weights
  causal-LM exposes; it is what the Sankey/vector-field views require. Temperature
  scaling of logits before softmax is the standard sampling-temperature definition.
- **Alternatives**: Sampling repeatedly to estimate the distribution (lossy, slow) —
  rejected for the exact distribution; reserved only for the *temperature-fan*
  visualization variant downstream, not for the canonical distribution here.

## R2. Per-layer token embeddings + reference token set

- **Decision**: Expose two embedding sources: (a) **static** input-embedding matrix
  (`model.get_input_embeddings().weight`, shape `[vocab, d]`) as the base "layer 0"
  location of every token — cheap, one tensor; (b) **contextual** per-layer hidden
  states via `output_hidden_states=True`, yielding a tuple of `num_layers+1` tensors,
  computed by feeding each reference token as a single-token input in batches and
  taking the last-position hidden state per layer. The **reference token set** is
  configurable; default = full vocabulary for the static source, and a configurable
  subset (default: full vocab, batched) for contextual layers, with the chosen set
  recorded in the cache key.
- **Rationale**: The vector-field view needs a per-token location per layer; static
  embeddings give an instant, exact base layer, while `output_hidden_states` is the
  documented HF mechanism for per-layer representations. Batching keeps the full-vocab
  pass within the precompute budget; caching makes it a one-time cost.
- **Alternatives**: Only static embeddings (loses the layer slider's contextual
  story) — rejected; running multi-token contexts for every vocab item (combinatorial)
  — rejected for the base machinery, deferred to context-specific precomputes keyed by
  the prompt prefix.
- `TODO(verify:)` exact `hidden_states` tuple length/order and last-token indexing
  confirmed by an integration test on `sshleifer/tiny-gpt2` and `distilgpt2`.

## R3. Open-weights capability detection (FR-002/FR-003)

- **Decision**: Before computing, verify the model (i) resolves on the Hub and is
  loadable without auth errors, (ii) is a causal/decoder LM exposing logits over the
  tokenizer vocabulary, and (iii) returns hidden states when requested. Any failure
  raises a specific, typed error (`UnsupportedModelError` with a human message);
  there is **no fallback** to another model or to fabricated data.
- **Rationale**: Directly enforces FR-002/FR-003 and the constitution's "fail loudly,
  never mock" rule. Gated/private/non-LM/closed models fail fast with a clear reason.
- **Alternatives**: Trust the id and let downstream crash cryptically — rejected
  (poor UX, violates FR-021's clear-error requirement).

## R4. 2D reduction: PCA (default) + UMAP (option)

- **Decision**: Default 2D reduction is **PCA** (`sklearn.decomposition.PCA`,
  deterministic) for reproducibility (FR-013) and speed. **UMAP** (`umap-learn`) is
  offered as a selectable method with a fixed `random_state` (documented to make it
  deterministic at the cost of single-threaded embedding). The reduction method +
  params + seed are part of the cache key.
- **Rationale**: PCA is exactly reproducible with zero seeding caveats and is fast on
  CPU; UMAP gives nicer cluster structure when desired. Offering both satisfies the
  spec's "UMAP or PCA" wording while keeping the default deterministic.
- **Alternatives**: UMAP-only (nondeterminism risk vs FR-013) — rejected as default;
  t-SNE (not requested, less stable) — rejected.

## R5. 3D spherical reduction

- **Decision**: Produce a 3D embedding (classical/metric MDS via
  `sklearn.manifold.MDS`, or PCA-to-3D as a fast deterministic fallback) and project
  each point onto the unit sphere by L2-normalization; tokens are then placed on a
  radius-2 sphere for the manifold view. Method + seed recorded in the cache key.
  True geodesic spherical-MDS is noted as a future refinement.
- **Rationale**: Gives a reproducible, well-defined "tokens on a sphere" substrate
  that the downstream manifold visualization warps; matches `project_description.md`'s
  "spherical MDS or similar" while staying tractable and deterministic.
- **Alternatives**: Full geodesic spherical MDS optimization (heavier, harder to make
  reproducible) — deferred to the manifold feature; the order-invariant warp-combining
  question in `project_description.md` is explicitly **out of scope** for core
  machinery (it belongs to the manifold visualization feature).

## R6. Cache: keys, integrity, single-flight

- **Decision**: A cache artifact's key is a stable hash of an explicit, sorted spec:
  `{schema_version, model_id, model_revision, artifact_type, inputs, params, seed}`.
  Artifacts are written atomically (temp file + rename) with a small JSON sidecar
  storing the full spec, a content checksum, and a `complete` flag. On read, the
  sidecar is validated (schema version + checksum + complete); any mismatch →
  treat as miss and recompute. A `jobs.registry` keyed by cache key provides
  **single-flight**: concurrent identical requests await one running computation.
- **Rationale**: Implements FR-004..FR-008 (regenerable, identical-on-hit, detect
  corruption/partial, dedup concurrent work). Atomic write + `complete` flag prevents
  treating an interrupted precompute as a hit (edge case).
- **Alternatives**: Hashing opaque pickles (not portable/inspectable) — rejected;
  no integrity flag (risks serving partial artifacts) — rejected.

## R7. Progress streaming

- **Decision**: Backend exposes precompute progress via **Server-Sent Events (SSE)**
  from a job's progress channel; the frontend subscribes and animates a determinate/
  indeterminate progress indicator, falling back to short-interval polling if SSE is
  unavailable. Progress updates emitted **≥ 1/second** during long precomputes.
- **Rationale**: SSE is the simplest one-way server→client stream, ideal for progress;
  satisfies FR-009 / SC-003 (continuous feedback, never frozen).
- **Alternatives**: WebSocket (bidirectional, heavier than needed) — rejected;
  polling-only (laggier) — kept only as fallback.

## R8. Frontend stack (user-selected)

- **Decision**: **Svelte + TypeScript + Vite**, with D3.js (2D) and Three.js (3D)
  used minimally in the core-machinery preview; Svelte stores for shared interaction
  state; Vitest (unit) + Playwright (e2e + screenshots).
- **Rationale**: User-selected (2026-06-14). Compile-time reactivity and minimal
  runtime overhead best fit the 60 fps + "beautiful, clean" constitution bar; the
  viz libraries own their own canvas/WebGL so framework re-render pressure is low.
- **Alternatives**: React, vanilla TS — both viable; not chosen by the user.

## R9. Default + test models

- **Decision**: Default app model = **`gpt2`** (small, open, fast, ubiquitous).
  Tests use **`sshleifer/tiny-gpt2`** for fast real runs and assert at least one
  real-output path on **`distilgpt2`**/`gpt2`. Curated menu seeds a few small
  open-weights models; arbitrary HF ids are accepted (R3 gates them).
- **Rationale**: Keeps the full pipeline runnable in minutes on the CPU reference
  machine and keeps CI-like tests fast while still being **real** models (no mocks).
- **Alternatives**: A larger default (slower first run, exceeds budget on CPU) —
  rejected for the default; still selectable.

## R10. Reproducible environment

- **Decision**: Provide two reproducible cross-stack mechanisms (Constitution V
  allows "containerized AND/OR pinned dependencies"): (a) a primary `environment.yml`
  for conda/mamba that pins **both** Python 3.11 and Node 20 in one isolated env and
  pip-installs the backend; (b) a repo-root `Dockerfile` (same dual stack) for hosts
  with a Docker daemon. Backend deps come from `pyproject.toml` with exact pins
  frozen in `requirements.lock`; frontend deps pinned via `package.json` + lockfile.
  `README.md`/`CLAUDE.md`/`quickstart.md` document both; `quickstart.md` is the
  runnable proof. Cache dir git-ignored.
- **Rationale**: Enforces Principle V (reproducible, independently runnable) and
  Principle IV (docs co-updated). The conda/mamba env is the verified primary path
  because it needs no Docker daemon and still pins both stacks.
- **Alternatives**: Keep the legacy Python 3.7 conda Dockerfile (cannot run modern
  transformers/Svelte) — rejected. Docker-only — rejected as sole mechanism since a
  daemon may be unavailable; the conda/mamba env covers that case.

## Open questions deferred (not blocking this feature)

- Order-invariant combination of per-token sphere warps — `project_description.md`
  flags this as research; it belongs to the **manifold visualization** feature, not
  core machinery.
- Multi-tenant/hosted deployment and horizontal cache sharing — out of scope
  (local-first per spec assumption).
