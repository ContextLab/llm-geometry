# Feature Specification: Interactive Model Explorer

**Feature Branch**: `002-interactive-model-explorer`

**Created**: 2026-07-24

**Status**: In progress

**Input**: GitHub issue #1 ("getting started") + the implementation plan posted at
https://github.com/ContextLab/llm-geometry/issues/1#issuecomment-5071465097 (the plan
comment is the authoritative expansion of this spec; design decisions D1–D10 and the
red-team resolutions in §3b are incorporated by reference).

Two new tabs on the existing web app, sharing one invariant: **every visual element
maps 1-to-1 onto a real model component, and everything is manipulable.**

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Explore a real LLM's architecture (Priority: P1)

A student opens the **Architecture** tab, sees the full component diagram of a real
open-weights model (default `HuggingFaceTB/SmolLM2-135M-Instruct`) built from a traced
forward pass — every step present, including parameterless ops (RoPE, softmax,
residual adds, activations). They click any matrix block to zoom into its actual
values; they type a prompt (and optional system prompt) and watch tokenization and a
per-layer animated trace of the real forward pass; they generate a short reply.

**Why this priority**: it is the issue's "superficial demo" and the entry point for
everything else; it reuses the most existing machinery.

**Independent Test**: open tab → diagram renders with ≥1 node per traced op; click
`q_proj` → exact values match `state_dict()`; type prompt → trace animates; generate →
real tokens stream back.

**Acceptance Scenarios**:

1. **Given** a cold cache, **When** the tab opens, **Then** a phase-labeled progress
   animation runs (download → graph build) and the diagram appears without errors.
2. **Given** the diagram, **When** any weight node is clicked, **Then** a zoomable
   heatmap shows real values (downsampled tiles; exact values ≤4096-cell windows) and
   tied weights (`lm_head` ≡ `embed_tokens`) display as one aliased tensor.
3. **Given** a typed prompt, **When** 400 ms pass without edits, **Then** a single
   trace request fires (stale ones aborted) and the animation reflects the real
   tensors returned.
4. **Given** a model over the size ceiling, **When** it is selected, **Then** a
   plain-language rejection appears before any download starts.

### User Story 2 - Understand geometry with a tiny transparent model (Priority: P1)

A student opens the **Geometry** tab. A tiny from-scratch decoder-only transformer
(`d_model=3`, 4 layers, 1 head, ~1000-word vocab, ≤50-token context) has been trained
(once, cached, with animated progress) on a real public-domain corpus. Its 3-D token
embeddings sit on a sphere. The student types a prompt and sees vector fields over the
sphere — either the "next-next token" field or the paper's attention-force field
(`Σ softmax(⟨K z_j, Q z_i⟩)·V z_j`, arXiv:2607.13295) — per layer or full model. They
edit W_Q/W_K/W_V/W_O/embedding per layer (cell-by-cell or via presets: identity,
fuzzy-diagonal Toeplitz, random, random+autocorrelated, zero, learned) and watch the
field change. They view the attention matrix as a heatmap. They fine-tune the tiny
model on pasted text, an uploaded .txt/.md, or an HF dataset id and watch embeddings
move — never mutating the canonical checkpoint.

**Why this priority**: it is the issue's "deeper" demo and its scientific core.

**Independent Test**: train-on-first-open completes; sphere shows ~1000 embedding
points with non-degenerate learned vector field (directional-entropy gate); setting
W_V per-layer to identity visibly changes arrows; fine-tune on pasted text mints a new
`weights_token` and moves points.

**Acceptance Scenarios**:

1. **Given** a cold cache, **When** the tab opens, **Then** training runs as a job
   with epoch-labeled SSE progress and the result is cached for all later loads.
2. **Given** the learned checkpoint, **When** the non-degeneracy and
   coverage-uniformity tests run, **Then** both pass their thresholds.
3. **Given** a weight edit (preset or cells), **When** it is submitted, **Then** a
   content-hash `weights_token` is returned, all geo endpoints accept it, and a page
   refresh (sessionStorage) preserves the edit.
4. **Given** a prompt containing out-of-vocab words, **When** it is tokenized, **Then**
   `<unk>` substitutions are visibly marked in the tokenization strip.
5. **Given** a fine-tune request (text, file, or HF dataset), **When** the job
   completes, **Then** a *new* checkpoint token is returned and the canonical
   `learned` artifact is unchanged on disk.

### User Story 3 - Everything stays consistent with the existing app (Priority: P2)

The three existing views keep working: new tabs use their own stores (no semantic
collisions), the sidebar shows only controls relevant to the active view, and all new
computations flow through the existing cache + single-flight jobs + SSE machinery.

**Acceptance Scenarios**:

1. **Given** any sequence of tab switches, **When** views change, **Then** each view's
   state persists independently and no store value leaks across meanings.
2. **Given** identical repeated requests, **When** they hit the backend, **Then** the
   cache serves hits and concurrent identical work is single-flighted.

## Requirements *(mandatory)*

- **FR-101**: Architecture graph MUST be built from a traced forward pass and include
  parameterless ops; a completeness test MUST diff traced ops against graph nodes.
- **FR-102**: Weight views MUST serve real values (server-side downsampling allowed;
  exact under zoom), with tied tensors represented once and aliased.
- **FR-103**: The tiny model MUST use true 3-D embeddings (`d_model=3`) with unit-norm
  + spherical-uniformity regularization; no dimensionality reduction in the Geometry
  tab. Training MUST be real, seeded, cached, and non-degeneracy-gated.
- **FR-104**: All weight edits MUST round-trip through content-addressed cache entries
  (`weights_token` = content hash; LRU eviction; multi-worker safe).
- **FR-105**: Both vector-field modes (next-next; attention-force with antisymmetrize
  toggle scoped to the per-point `V·z` field) MUST be selectable per layer or full
  model.
- **FR-106**: Fine-tuning MUST accept pasted text, uploaded .txt/.md, and HF dataset
  ids; MUST run real SGD; MUST mint new checkpoints, never mutating `learned`.
- **FR-107**: Every new failure class MUST surface a designed inline error state
  (no silent failures, no raw stack traces); models over the parameter ceiling MUST be
  rejected at selection time.
- **FR-108**: All interactions MUST debounce/abort stale requests (cancel-and-restart).
- **FR-109**: No mock objects or simulated data anywhere — tests and runtime use real
  models, real corpora, real training, real HTTP.
- **FR-110**: Contract in `contracts/api.md` is frozen; frontend and backend implement
  against it; contract tests enforce it.

## Success Criteria *(mandatory)*

- **SC-101**: Cold-start of each tab completes with animated, phase-labeled progress;
  warm loads are instant (cache hits < 100 ms server-side).
- **SC-102**: Architecture completeness test passes for SmolLM2-135M-Instruct and
  Qwen2.5-0.5B-Instruct.
- **SC-103**: Geometry Lab non-degeneracy + coverage gates pass on the canonical
  checkpoint; W_V=identity produces a visibly and numerically different field.
- **SC-104**: Full local gate (ruff, black --check, backend pytest, svelte-check,
  vitest, vite build, Playwright e2e) and the new CI PR gate are green.

## Non-goals (explicit deferrals, flagged in the issue)

- In-browser-only inference (transformers.js/WebGPU) — follow-up issue.
- Training exploration of the big (SmolLM2-class) model — follow-up; the ingestion
  path ships now against the tiny model.
- Multi-turn chat history; full a11y audit; non-decoder architectures (schema is
  extensible per D10, implementations deferred).
