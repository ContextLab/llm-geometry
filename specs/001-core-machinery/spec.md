# Feature Specification: Core Project Machinery

**Feature Branch**: `001-core-machinery`

**Created**: 2026-06-14

**Status**: Draft

**Input**: User description: "let's implement the core project machinery"

## Overview

`llm-geometry` will ship three interactive web visualizations of a transformer's
embedding space (vector fields, Sankey diagrams, and a reachable-thoughts
manifold). All three share the same underlying needs: access to an open-weights
model that exposes token-level probabilities, the same expensive computations
(per-layer embeddings and next-token distributions), the same dimensionality
reductions, the same precompute-and-cache discipline, and the same family of
interaction controls (model selector, editable prompt/context prefix, temperature,
layer selector) and progress experience.

This feature builds that **shared foundation** — the "machinery" the three
visualizations run on — so each visualization can later be specified and built as
a thin layer on top of a proven substrate. It deliberately does **not** build any
of the three visualizations themselves; instead it includes a minimal verification
surface (a data preview / placeholder page) that proves the foundation works
end-to-end, from model selection through cached data to a responsive web page.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Turn a model into cached geometric data, with visible progress (Priority: P1)

A person chooses an open-weights model (from a curated menu or by entering an
arbitrary HuggingFace model identifier). The system loads the model and computes
the quantities the visualizations depend on — per-layer embeddings for a reference
set of tokens, and next-token probability distributions for given contexts — then
stores the results in a cache keyed by model, inputs, and parameters. The first
time a given combination is requested it is computed once while an animated
progress indicator communicates what is happening and roughly how long it will
take; every later request for the same combination is served instantly from the
cache and yields identical results.

**Why this priority**: This is the engine. Nothing else in the project — no
reduction, no visualization, no interaction — can exist without a reliable way to
get token-level data out of an open-weights model and keep it. It is the smallest
slice that delivers standalone value: a reusable, cached geometric dataset.

**Independent Test**: Request data for a model/context combination that has never
been computed; confirm an animated progress indicator appears and advances, a
cache artifact is produced, the produced distributions are real model outputs (not
placeholders), and a second identical request returns instantly with byte-identical
results.

**Acceptance Scenarios**:

1. **Given** an empty cache and a model selected from the curated menu, **When** the
   user requests its geometric data, **Then** the system loads the model, shows an
   advancing progress indicator throughout, computes per-layer embeddings and
   next-token distributions, and writes a cache artifact tagged with the model,
   inputs, and parameters used.
2. **Given** a model/context/parameter combination already present in the cache,
   **When** the same combination is requested again, **Then** results are returned
   from the cache without recomputation and are identical to the originally
   computed values.
3. **Given** an arbitrary open-weights HuggingFace model identifier entered by the
   user, **When** the model exposes token-level probabilities, **Then** the system
   computes and caches its data without any code change.
4. **Given** an identifier that does not exist, is not open-weights, or does not
   expose token-level probabilities, **When** the user requests its data, **Then**
   the system reports a clear, specific error and produces no cache artifact and no
   fabricated result.

---

### User Story 2 - Make high-dimensional geometry explorable in 2D and 3D (Priority: P2)

Given a model's cached embeddings, the system produces the low-dimensional
representations the visualizations draw on: a 2D reduction (UMAP or PCA) used by the
vector-field and trajectory views, including an *n×n* reference grid whose vertices
are snapped to their nearest tokens ("reference points"); and a 3D spherical
reduction (spherical MDS or equivalent) used by the manifold view, placing tokens on
a sphere. These reductions are themselves precomputed and cached, are regenerable
from the canonical embeddings, and are reproducible given the same inputs and
parameters.

**Why this priority**: The raw embeddings from P1 are not directly viewable. This
slice converts them into the common geometric substrate every visualization needs,
and proves the reduction and reference-point machinery works — without yet drawing
any specific visualization.

**Independent Test**: For a model whose embeddings are already cached, request the
reduced coordinates; confirm a 2D coordinate set with resolvable nearest-token
reference points and a 3D spherical coordinate set are produced, cached, and
identical on a second run with the same parameters.

**Acceptance Scenarios**:

1. **Given** cached embeddings for a model, **When** a 2D reduction is requested,
   **Then** a 2D coordinate set is produced and cached, and each vertex of an *n×n*
   grid laid over the space resolves to its nearest token.
2. **Given** cached embeddings for a model, **When** a 3D spherical reduction is
   requested, **Then** tokens are placed on a sphere and the result is produced and
   cached.
3. **Given** a reduction already cached, **When** it is requested again with the same
   parameters, **Then** the cached result is returned and is reproducible
   (identical) to the original.
4. **Given** a request to regenerate a reduction, **When** the canonical embeddings
   are present, **Then** the reduction can be rebuilt from them without re-running
   the model.

---

### User Story 3 - Explore through a shared, instant, polished web shell (Priority: P3)

A person opens a web page served by the project. A shared application shell hosts
the page and offers the reusable controls every visualization will need: a model
selector (including entry of an arbitrary open-weights model), an editable
prompt/context prefix, a temperature control, and a layer selector. Changing any
control fetches the corresponding cached data through a shared data-access layer and
updates the page. When the data is already cached the page responds instantly; when
it is not, a smooth progress/transition animation plays while the computation from
P1/P2 runs, and the page updates the moment results are ready. A minimal data-preview
(or placeholder render) demonstrates this round trip end-to-end; it is explicitly not
one of the three real visualizations.

**Why this priority**: This is the shared front-end foundation and interaction/UX
contract every visualization inherits. It proves the whole machinery is usable, fast,
and beautiful, and gives downstream visualization features a ready-made shell and
control set to drop into.

**Independent Test**: Load the shell, change the model, prompt/context prefix,
temperature, and layer controls in turn; confirm each change fetches the correct
cached data and updates the preview, that a cached change feels instant, and that an
uncached change shows a progress animation and then resolves to real data.

**Acceptance Scenarios**:

1. **Given** the shell is open and the requested data is cached, **When** the user
   changes any shared control, **Then** the preview updates near-instantly with the
   data corresponding to the new control values.
2. **Given** the shell is open and the requested data is not yet cached, **When** the
   user changes a control, **Then** a progress/transition animation appears
   immediately and the preview updates with real results when computation finishes.
3. **Given** the user enters an arbitrary open-weights model identifier in the
   selector, **When** it is valid and supported, **Then** the shell drives the same
   compute/cache/preview flow for it without code changes.
4. **Given** an invalid or unsupported control combination, **When** it is submitted,
   **Then** the shell surfaces a clear message and never displays fabricated or
   stale-as-fresh data.

---

### Edge Cases

- **Unsupported model**: an entered identifier that does not exist, is gated/private,
  is not open-weights, or hides per-token probabilities → a clear, specific error;
  no cache artifact; no silent fallback to another model and no fabricated data.
- **Insufficient hardware**: a model too large for available memory/compute →
  graceful failure with an informative message; documented hardware expectations for
  the curated default model.
- **Interrupted precompute**: a compute job that is cancelled or crashes mid-way →
  the partial result is never treated as a complete cache hit; it is detected and
  recomputed on next request.
- **Corrupted or stale cache**: an artifact that fails an integrity/version check →
  detected and recomputed rather than served; results are never silently wrong.
- **Concurrent identical requests**: two requests for the same uncomputed
  model/inputs → the expensive work runs once and is shared, not duplicated.
- **Empty cache / first run**: the very first request shows the full progress
  experience with continuous feedback; the interface never appears frozen.
- **Model download offline/partial**: loss of connectivity during a model download →
  a clear message; a partial download is never treated as a usable model.
- **Reproducibility**: repeated runs with the same model, inputs, parameters, and any
  random seeds produce identical cached results; any nondeterminism is controlled by
  a documented, fixed seed.

## Requirements *(mandatory)*

### Functional Requirements

**Model access**

- **FR-001**: The system MUST load open-weights models from HuggingFace by identifier,
  supporting both a curated menu of preconfigured models and arbitrary user-entered
  open-weights model identifiers.
- **FR-002**: The system MUST obtain genuine token-level next-token probability
  distributions and per-layer token embeddings from the loaded model, and MUST reject
  (with a clear error) any model that cannot supply token-level probabilities.
- **FR-003**: The system MUST NEVER fall back to a different model, a closed API, or
  fabricated values when the requested model cannot be served; it MUST fail loudly
  with a specific, actionable message.

**Computation, caching & precompute**

- **FR-004**: The system MUST precompute expensive quantities once and cache them,
  keyed by a combination of model identity, inputs (e.g., context prefix), and
  parameters (e.g., temperature, layer, grid size, reduction settings).
- **FR-005**: Every cached artifact MUST be regenerable from its canonical source
  (the model and its parameters), so the cache can be deleted and rebuilt.
- **FR-006**: The system MUST return cached results without recomputation on a cache
  hit, and those results MUST be identical to the originally computed values.
- **FR-007**: The system MUST detect incomplete, corrupted, or version-mismatched
  cache artifacts and recompute them rather than serve them.
- **FR-008**: The system MUST ensure a given expensive computation runs at most once
  even when multiple identical requests arrive concurrently.
- **FR-009**: The system MUST report progress for any computation that is not
  instantaneous, with continuous feedback (the interface MUST NOT appear frozen) and a
  signal of expected completion.

**Dimensionality reduction & reference geometry**

- **FR-010**: The system MUST produce and cache a 2D reduction of a model's embeddings
  (UMAP or PCA) suitable for the vector-field and trajectory views.
- **FR-011**: The system MUST lay an *n×n* grid over the 2D space and resolve each grid
  vertex to its nearest token ("reference point"), with *n* configurable.
- **FR-012**: The system MUST produce and cache a 3D spherical reduction of a model's
  embeddings (spherical MDS or equivalent) that places tokens on a sphere, suitable
  for the manifold view.
- **FR-013**: Reductions MUST be reproducible: identical inputs and parameters
  (including any random seed) MUST yield identical cached outputs.

**Shared web shell & interaction controls**

- **FR-014**: The system MUST serve interactive web pages from a shared application
  shell that downstream visualizations can build on.
- **FR-015**: The shell MUST provide reusable, shared controls: a model selector
  (including arbitrary open-weights model entry), an editable prompt/context prefix, a
  temperature control, and a layer selector.
- **FR-016**: Changing any shared control MUST fetch the corresponding cached data
  through a shared data-access layer and update the page; cached responses MUST feel
  instant and uncached responses MUST show an immediate progress/transition animation.
- **FR-017**: The system MUST include a minimal verification surface (a data preview or
  placeholder render) that exercises the full model→compute→cache→serve→display path,
  while NOT implementing any of the three production visualizations.
- **FR-018**: The shell and its animations/transitions MUST be smooth and responsive,
  meeting the performance budget defined for this feature.

**Reproducibility, environment & documentation**

- **FR-019**: The system MUST ship a reproducible environment (containerized and/or
  pinned dependencies) and runnable setup-and-use instructions sufficient for an
  independent party to recreate the environment and produce cached data for the
  default model with no undocumented steps.
- **FR-020**: Dependency manifests and setup/usage documentation MUST be kept in sync
  with what the code actually requires as this feature is built.
- **FR-021**: All user-facing errors MUST be clear and specific, and the system MUST
  never present hypothetical, placeholder, or stale data as a real, current result.

### Key Entities *(include if feature involves data)*

- **Model Reference**: an open-weights model the system can load — its HuggingFace
  identifier, source/menu metadata, and capability flags (notably whether it exposes
  token-level probabilities).
- **Embedding Set**: per-layer embedding vectors for a model's reference set of tokens;
  the canonical source from which reductions are derived.
- **Next-Token Distribution**: for a given context (and parameters such as temperature),
  the probability of emitting each possible next token.
- **Reduced Coordinate Set**: a low-dimensional representation of an Embedding Set — 2D
  (UMAP/PCA) or 3D spherical (MDS) — tagged with its method, parameters, and seed.
- **Reference Points / Grid**: the *n×n* grid laid over the 2D space and the nearest
  token resolved to each vertex.
- **Cache Artifact**: a stored computation result keyed by model + inputs + parameters
  + version, with provenance and an integrity/version marker and status.
- **Precompute Job**: an in-flight or completed computation with status and progress
  used to drive the progress indicator and concurrency control.
- **Interaction Parameters**: the user-controllable inputs shared across
  visualizations — model selection, prompt/context prefix, temperature, and layer
  selection.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: After its data is cached, changing any shared control updates the preview
  with a perceived-instant response — the interface begins responding in under 100 ms
  and shows the result in under 1 second.
- **SC-002**: A cache hit returns results identical to the original computation 100% of
  the time; deleting the cache and rebuilding for the same inputs reproduces identical
  results.
- **SC-003**: The first-time precompute for the curated default model completes within
  a documented time budget on the documented reference machine, with progress feedback
  updating at least once per second and never appearing frozen.
- **SC-004**: Shell animations and transitions render smoothly at the feature's target
  frame rate (60 fps / ~16 ms per frame) on the documented reference machine.
- **SC-005**: An independent person, starting from a clean machine and the project's
  documented instructions alone, can set up the environment and produce cached data for
  the default model with zero undocumented steps.
- **SC-006**: An arbitrary valid open-weights model can be added through the selector
  and produces cached data and a preview with no code changes.
- **SC-007**: Every invalid or unsupported model/control request produces a clear,
  specific error 100% of the time, and the system never displays a fabricated or
  silently substituted result.

## Assumptions

- **Scope is the shared foundation only.** The three production visualizations
  (vector fields, Sankey diagrams, manifold) are out of scope here and will each be
  specified and built as separate downstream features that consume this machinery.
  This feature includes only a minimal data-preview/placeholder surface to prove the
  foundation end-to-end.
- **Open-weights, token-probability-exposing models only.** Closed APIs that hide
  per-token probabilities are out of scope by constitutional constraint.
- **Default model is small and fast.** A small, widely-available open-weights model
  (e.g., a GPT-2-class model) is preconfigured as the development/default so the full
  pipeline runs quickly on a typical developer machine; the curated menu and
  arbitrary-model entry make larger models available later.
- **Local/self-hosted first.** The machinery is initially run and served locally (or
  on a single host) for development, but is designed so it can later be deployed; the
  exact deployment target is a planning decision, not a spec constraint.
- **Concrete performance budgets and the reference machine** (target frame rate,
  interaction-latency ceilings, maximum first-precompute time, default model, default
  grid size *n*) are finalized in the plan, consistent with the example targets in the
  Success Criteria.
- **Reference token set** for embeddings/reductions is the model's vocabulary (or a
  documented, configurable subset where the full vocabulary is impractical), chosen in
  the plan.
