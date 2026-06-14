<!--
SYNC IMPACT REPORT
Version change: 1.0.0 → 2.0.0
Rationale: MAJOR — the principle set was restructured. Former Principle I
(Accuracy & Verification) and former Principle IV (Academic Honesty & Integrity)
were merged into a single principle, and a new principle (Reproducibility & Open
Release) was added. Merging/removing a standalone principle is a backward-
incompatible redefinition, which mandates a MAJOR bump.

Principle changes:
  - I. Accuracy & Verification + IV. Academic Honesty & Integrity
        → merged into I. Accuracy, Verification & Academic Integrity
  - II. Single Source of Truth — unchanged
  - III. User Experience Is Paramount — unchanged
  - V. Documentation Stays Current — renumbered to IV (content unchanged)
  - NEW: V. Reproducibility & Open Release

Section changes:
  - Engineering Constraints & Standards — reproducibility reframed as a constraint
    that references new Principle V; cross-references renumbered
  - Development Workflow & Quality Gates — cross-references renumbered
  - Governance — documentation co-update cross-reference renumbered (V → IV)

Templates / docs reviewed for alignment:
  - ✅ .specify/templates/plan-template.md — "Constitution Check" gate references
       this file dynamically; no edit required
  - ✅ .specify/templates/spec-template.md — aligned; no mandatory-section changes
  - ✅ .specify/templates/tasks-template.md — reproducibility, setup-instruction,
       and release-readiness work fit the existing Polish phase; no structural change
  - ✅ CLAUDE.md — consistent (already documents the reproducible environment and
       the need to keep setup/dependencies in sync)
  - ✅ .specify/templates/commands/ — directory absent; nothing to update
Follow-up TODOs: none
-->

# LLM Geometry Constitution

## Core Principles

### I. Accuracy, Verification & Academic Integrity

This project is intended to become a scientific publication and MUST be conducted accordingly.
Every factual claim, citation, numeric result, and reference MUST be verified against a primary
source before it is recorded — by running the analysis directly, performing a web search, reading
the relevant paper, or inspecting the actual image, file, or data. Results MUST reflect what was
actually observed; "expected", hypothetical, simulated, or placeholder results MUST NEVER be
reported or presented as real findings. Items that are not yet verified MUST be explicitly flagged
(e.g., `TODO(verify): ...`) rather than stated as established fact. Work MUST be committed
frequently so the repository preserves a FULL, honest record of the project's evolution, and files
that cannot be easily reproduced MUST NOT be deleted or overwritten.

Rationale: Scientific credibility rests on a faithful, auditable trail and on never confusing
intention or expectation with evidence — verification (of inputs) and honesty (about outputs) are
the same discipline applied at both ends.

### II. Single Source of Truth

Every function, data object, figure, image, and file MUST exist in exactly ONE canonical
location. Edits MUST be made IN PLACE; duplicate, backup, or `*_v2` / `*_old` / `*_copy`
variants are prohibited. Project history is preserved exclusively through Git commits, never
through parallel files. Derived artifacts (compiled PDFs, cached precomputations, exported
figures) MUST be regenerable from their single canonical source.

Rationale: Duplication produces drift and contradictory results; Git already provides reliable,
auditable history without cluttering the working tree.

### III. User Experience Is Paramount

All visualizations and interactions MUST be smooth, fast, and delightful to engage with. The
system MUST respond instantly whenever feasible, using multi-threading and any other
optimizations needed to maximize performance and responsiveness. When an instant response is
not possible, the interface MUST immediately show a progress indicator and/or transition
animation that signals the request is being processed and sets expectations for when the result
will be ready. Final animations MUST render fast and responsively — pre-computing and caching
results when that is what it takes to meet this bar.

Rationale: These are interactive tools for understanding LLMs; perceived performance and polish
determine whether they actually communicate the underlying ideas.

### IV. Documentation Stays Current

All documentation MUST always be up to date. Any change to the approach, results, data, code, or
repository structure MUST, within the same change set, update every affected document — including
README files, internal agent files (`CLAUDE.md`), Spec Kit artifacts (`.specify/`, specs, plans,
tasks), AND paper sources (`.tex` files and figures). A change is not complete until its
documentation reflects reality.

Rationale: Stale documentation is actively misleading; co-updating docs with the change that
caused them prevents the record from silently diverging from the work.

### V. Reproducibility & Open Release

Everything MUST be designed from the outset for eventual public release of the complete project —
all code, all data, and all associated software. Computational environments MUST be reproducible
(containerized and/or with pinned dependencies), and every component MUST ship with clear,
runnable instructions for both setting it up and using it. The released artifacts alone MUST be
sufficient for an independent party to recreate the project's environment and reproduce its
results, without access to any private machine, undocumented step, or unshared resource.

Rationale: Open, reproducible release is a baseline expectation for credible computational
science, and it is far cheaper to maintain continuously than to retrofit at publication time.

## Engineering Constraints & Standards

These constraints operationalize the principles above and apply to all work in this repository.

- **Real verification, never mocks**: Tests and validations MUST exercise real functionality —
  real model calls, real inputs, real files, and real outputs — including the behavior of
  external libraries, models, and resources. Mock-only tests MUST NOT substitute for confirming
  that real components work. If real functionality cannot be validated, the code MUST fail
  loudly rather than silently fall back. (Enforces Principle I.)
- **Cached computation**: Expensive computations MUST be precomputed once and cached, with every
  cached artifact regenerable from its canonical source. (Enforces Principles II & III.)
- **Reproducible environments & instructions**: Dependency manifests (container/requirements)
  MUST be updated whenever dependencies change, and setup and usage instructions MUST be kept
  accurate and runnable as the project evolves. (Enforces Principle V.)
- **Open-weights models for probabilistic visualizations**: Any visualization that depends on
  token-level probability distributions MUST use open-weights models that expose those
  distributions; closed APIs that hide per-token probabilities MUST NOT be relied upon for
  these results. (Enforces Principles I & V.)
- **Per-feature performance budgets**: Each feature's plan MUST define concrete, measurable
  performance/responsiveness targets (e.g., target frame rate, interaction latency, max
  precompute time) and the strategy (threading, caching, progress UI) used to meet them.
  (Enforces Principle III.)

## Development Workflow & Quality Gates

- **Spec-Driven Development**: Features flow through the Spec Kit pipeline
  (specify → clarify → plan → tasks → analyze → implement). Each plan MUST include a
  Constitution Check that passes before implementation begins.
- **Commit discipline**: Commit frequently and in place. Every commit MUST leave documentation
  consistent (Principle IV) and MUST NOT introduce duplicate or backup files (Principle II).
- **Verification before completion**: No task may be marked complete, and no result reported,
  without evidence from a real run or a direct source check (Principle I). Claims of "done",
  "passing", or "fixed" require the supporting command output or artifact.
- **Documentation co-update gate**: A change touching approach, results, data, code, or
  structure is incomplete until the affected READMEs, `CLAUDE.md`, `.specify/` artifacts, and
  paper sources are updated in the same change set (Principle IV).

## Governance

This constitution supersedes all other development practices for this repository. When guidance
conflicts, the constitution wins; direct, explicit user instructions take precedence over
inferred defaults.

- **Amendments**: Proposed via a commit or PR that edits this file, states the rationale, and
  bumps the version per the policy below. Dependent templates and documentation MUST be updated
  in the same change (Principle IV).
- **Versioning policy**: Semantic versioning. MAJOR = backward-incompatible governance or
  principle removal/redefinition; MINOR = a new principle/section or materially expanded
  guidance; PATCH = clarifications, wording, and non-semantic refinements.
- **Compliance review**: Every plan's Constitution Check and every `/speckit-analyze` run MUST
  verify compliance with these principles. Violations MUST be remediated, or explicitly
  justified in the plan's Complexity Tracking, before work proceeds.

**Version**: 2.0.0 | **Ratified**: 2026-06-14 | **Last Amended**: 2026-06-14
