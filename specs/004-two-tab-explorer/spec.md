# Feature Specification: Two-Tab Explorer — removals, fixes, and real training

**Feature Branch**: none — landed directly on `main` (owner's call; every push redeploys
GitHub Pages, so each commit must be green locally before it is pushed).

**Created**: 2026-07-26

**Status**: In progress

**Input**: Owner request (2026-07-26): drop the three embedding-geometry views, then
diagnose/debug/fix seven defects across the two remaining explorer tabs. Design approved
in-session before any code was written.

**Goal**: launch the live site, fully red-teamed, with all identified issues fixed.

---

## Context

The app currently ships five tabs. Three of them (vector field, Sankey, manifold) are the
feature-001 embedding-geometry views; two (Architecture Explorer, Geometry Lab) are the
feature-002 explorers that feature 003 made runnable as a static GitHub Pages build.

This feature removes the first three outright and repairs the second two. After it, the
app is a **two-tab explorer**, and the shared left "Controls" sidebar disappears with the
views it served — both remaining tabs already own their controls.

Every root cause below was verified by reading the code before the design was written;
line references are to the state of `main` at commit `e8e4f7c`.

---

## User Stories

### US-1: A two-tab app with nothing dead behind it (P1)

The vector-field, Sankey, and manifold views are deleted — frontend components, backend
compute modules, their tests, their exporter presets, their e2e specs, and their
references in docs. Nothing unreachable remains.

**Acceptance**: the app shows exactly two tabs; no deleted module is importable; the full
test suite passes with no skipped-because-deleted tests; `grep` finds no references to the
removed views outside historical notes and the superseded 001 spec.

### US-2: Replies that are actually worth reading (P1)

The Architecture Explorer's chat produces coherent answers to ordinary prompts.

**Acceptance**: "What is the capital of France?" yields a correct, fluent reply on the
default model, captured as a screenshot from the live static site. Base models without a
chat template are labeled as text-completion models rather than silently prompted as
chatbots.

### US-3: A trace animation that obeys the user (P1)

Playback through the traced forward pass no longer fights the layer control, and all
attention heads of a layer can be seen at once.

**Acceptance**: with playback running, a manually chosen layer stays chosen; an explicit
"follow playhead" toggle restores auto-follow; the per-layer detail shows a grid of every
head's attention matrix with the selected head highlighted.

### US-4: Honest claims everywhere (P1)

The UI promises only what the build can do. The Architecture model picker offers a curated
list — the "arbitrary HuggingFace id" affordance is removed — and every user-visible
string in both tabs matches actual behavior.

**Acceptance**: a red-team pass over both tabs finds no claim contradicted by behavior; a
tracking issue exists for expanding model support later.

### US-5: Working documentation links (P1)

Every external link in the app resolves.

**Acceptance**: each URL referenced by the app returns HTTP 200 from an unauthenticated
client.

### US-6: A sphere you can hold still (P2)

Auto-rotation is user-controlled and off by default; dragging the sphere pauses it.

**Acceptance**: the toggle is visible in the scene, rotation is stopped on load, and an
orbit drag never fights the animation.

### US-7: Forces that lie on the sphere (P1)

The Geometry Lab's aggregate sequence forces are tangent to the sphere where they are
drawn, and both the forces and the token path are occluded by the sphere's near surface.

**Acceptance**: screenshots from the front and the back of the sphere show far-side
geometry hidden; `normal_residual` reports how much radial component was projected away;
antisymmetrization is on by default.

### US-8: Train a real model from scratch, keep it, bring it back (P1)

A user can train a fresh GeoTransformer on their own pasted text or on a real HuggingFace
dataset, then save the resulting model to a file and load it back later.

**Acceptance**: training from scratch on user text builds a fresh vocabulary from that
text, produces a model whose loss falls over training, and drives every Geometry view;
saving downloads a file that, when re-loaded in a new session, restores the identical
model (verified by weight hash).

---

## Requirements

### Removal

- **FR-401**: The vector-field, Sankey, and manifold views MUST be deleted from the
  frontend, together with every control used only by them (`LayerSlider`,
  `ModelSelector`, `PromptPrefix`, `ResponseAnimator`, `SwarmControls`,
  `ManifoldControls`, `Temperature`, `StaticPresetPicker`, `vizMath`).
- **FR-402**: Their backend counterparts MUST be deleted —
  `compute/{vector_field,sankey,manifold,token_cloud,distributions,printable}.py`,
  `reduce/`, `precompute.py`, `api/routes.py` — along with their tests and their sections
  of `scripts/export_static_assets.py`. Modules still used by the two remaining tabs
  (`models/loader.py`, `compute/context.py`, `cache/`, `jobs/`) MUST be kept. Each
  deletion MUST be justified by real import analysis, not substring search.
- **FR-403**: `ExportBar`/`exportFigure` MUST be retained and wired to the Geometry Lab's
  WebGL canvas rather than deleted with its former callers.
- **FR-404**: Docs (`README.md`, `CLAUDE.md`, `code/*/README.md`) and the 001 spec MUST be
  updated to describe the two-tab app; the 001 spec is marked superseded rather than
  deleted, so the history stays readable.

### Response quality

- **FR-405**: Generation MUST apply standard decoding constraints — `top_p ≈ 0.9`,
  `top_k ≈ 50`, `repetition_penalty ≈ 1.1` — in BOTH the transformers.js runtime and the
  backend's `arch/generate.py`, and the two MUST stay mirrored. Unfiltered full-vocab
  sampling (`top_k: 0, top_p: 1.0`) is the primary cause of the current output quality
  and MUST NOT remain.
- **FR-406**: The default Architecture model MUST be `Qwen/Qwen2.5-0.5B-Instruct`.
- **FR-407**: A model whose tokenizer has no chat template MUST be labeled in the UI as a
  base/text-completion model, and MUST NOT be presented as if it answers questions.

### Architecture animation

- **FR-408**: The trace playhead MUST NOT overwrite a user-selected layer. Auto-follow
  becomes an explicit toggle (default on) that disengages when the user moves the layer
  control.
- **FR-409**: The per-layer detail MUST show every attention head of the selected layer
  simultaneously as small multiples, with the selected head highlighted and click-to-
  enlarge for the full-size view.
- **FR-410**: Before fixing, the reported "animations don't update correctly" symptom MUST
  be reproduced in a real browser and the observed behavior recorded, so the fix addresses
  the actual defect rather than the first plausible cause.

### Data sources and honesty

- **FR-411**: The Geometry Lab MUST accept a real HuggingFace dataset as a training or
  fine-tuning corpus, read from the public datasets-server rows API (no auth, CORS-
  enabled), selectable by dataset / config / split / text column, in addition to pasted
  text.
- **FR-412**: The Architecture model picker MUST offer a curated list only; the free-text
  "any open-weights HuggingFace id" input and every claim implying arbitrary model support
  MUST be removed.
- **FR-413**: A GitHub issue MUST be filed to track expanding model support later (curated
  growth, or automatic filtering of open-weights ONNX repositories).
- **FR-414**: Every user-visible string in both tabs MUST be audited against actual
  behavior by adversarial review; anything the build cannot do MUST NOT be claimed.

### Links

- **FR-415**: The repository MUST be made public (owner-authorized), which is the root
  cause of the 404ing "run the full stack" links, and EVERY external URL the app
  references MUST then be verified to return HTTP 200 unauthenticated.

### Geometry scene

- **FR-416**: Sphere auto-rotation MUST be user-controllable, default OFF, and MUST pause
  when the user drags the sphere.
- **FR-417**: `geoAntisymmetrize` MUST default to true, so the per-point field is exactly
  tangent by construction.
- **FR-418**: Aggregate sequence forces MUST be projected onto the tangent plane at the
  anchor point `z_i` before display. Antisymmetrizing `W_V` alone is NOT sufficient: each
  term `W_V z_j` is tangent at `z_j`, not at the `z_i` where the aggregate is drawn. The
  removed radial component MUST continue to be reported as `normal_residual`.
- **FR-419**: Force arrows and the token path MUST be depth-tested against the sphere so
  far-side geometry is occluded, while remaining visible on the near side.

### Training and persistence

- **FR-420**: A user MUST be able to train a GeoTransformer from scratch on arbitrary text
  (pasted or from a HuggingFace dataset). Training MUST build a fresh vocabulary from that
  text and freshly initialized unit-norm embeddings — it is a genuinely new model, not a
  re-fit of the shipped one.
- **FR-421**: From-scratch training MUST run in BOTH runtimes: the TypeScript engine (so
  the static site really has the feature) and the backend (so the full stack keeps
  parity).
- **FR-422**: A trained or edited model MUST be savable to a single file and loadable back
  in a later session. The file carries schema version, vocabulary, model config, weights,
  and provenance (corpus hash, step count, final loss, weight hash). Loading MUST validate
  the file and reject anything malformed with a plain-language error — never a silent
  partial load.
- **FR-423**: Contract changes required by FR-420..422 MUST land in their own commit with
  a note, per the project's frozen-contract rule.
- **FR-424**: No mocks, anywhere. Training tests train real models on real text; dataset
  tests hit the real datasets-server; generation tests run real models.

---

## Success Criteria

- **SC-401**: The app ships two tabs; backend `pytest`, frontend `vitest`,
  `svelte-check`, both Playwright projects, and both linters pass locally and in CI.
- **SC-402**: A screenshot from the deployed site shows a correct, fluent reply to "What
  is the capital of France?".
- **SC-403**: Screenshots from opposite sides of the sphere show correct occlusion, and a
  force-mode capture reports a near-zero `normal_residual` after projection.
- **SC-404**: A model trained from scratch in the browser, saved, and re-loaded in a fresh
  session reproduces the same weight hash.
- **SC-405**: Red-team subagents drive both tabs end-to-end via Playwright with screenshot
  evidence and find no false claim and no silently broken control; all findings are fixed
  and re-verified.
- **SC-406**: The live site at https://context-lab.com/llm-geometry/ serves the two-tab
  app and every external link in it resolves.

---

## Known limits (stated, not hidden)

- **Training is platform-divergent at the bit level** (macOS vs Linux BLAS). The
  TypeScript↔Python golden test therefore pins ONE forward+backward step from a fixed
  initialization, not a whole training run. Whole-run cross-platform equality is not
  achievable and is not claimed.
- **ONNX exports do not expose attention weights or hidden states**, so live per-prompt
  tracing of arbitrary models in the browser remains out of reach; this is why FR-412
  removes the arbitrary-model promise rather than trying to satisfy it.
- Pushing to `main` redeploys the live site with no review gate. Mitigation: every commit
  is verified locally first, and the removal and each fix land as separate commits so any
  regression is a one-commit revert.
