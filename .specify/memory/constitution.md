<!--
SYNC IMPACT REPORT
Version change: (unversioned template) → 1.0.0
Rationale: Initial ratification — template placeholders replaced with concrete,
project-specific principles. MAJOR (1.0.0) marks first adoption.

Principle slots resolved:
  - [PRINCIPLE_1] → I. Accuracy & Verification
  - [PRINCIPLE_2] → II. Single Source of Truth
  - [PRINCIPLE_3] → III. User Experience Is Paramount
  - [PRINCIPLE_4] → IV. Academic Honesty & Integrity
  - [PRINCIPLE_5] → V. Documentation Stays Current
Sections resolved:
  - [SECTION_2_NAME] → Engineering Constraints & Standards
  - [SECTION_3_NAME] → Development Workflow & Quality Gates
Removed sections: none

Templates / docs reviewed for alignment:
  - ✅ .specify/templates/plan-template.md — "Constitution Check" gate references
       this file dynamically; no edit required (already generic)
  - ✅ .specify/templates/spec-template.md — mandatory sections unchanged; aligned
  - ✅ .specify/templates/tasks-template.md — doc-update & verification tasks fit
       existing Polish phase; no structural change required
  - ✅ CLAUDE.md — consistent with all five principles
  - ✅ .specify/templates/commands/ — directory absent; nothing to update
Follow-up TODOs: none
-->

# LLM Geometry Constitution

## Core Principles

### I. Accuracy & Verification

Every factual claim, citation, numeric result, and reference MUST be verified against a
primary source before it is recorded — by running the analysis directly, performing a web
search, reading the relevant paper, or inspecting the actual image, file, or data. Unverified
assertions MUST NOT be committed to code, documentation, or the paper. When verification is
still pending, the item MUST be explicitly flagged (e.g., `TODO(verify): ...`) rather than
stated as established fact.

Rationale: This work targets scientific publication; a single unchecked claim can invalidate
a result and erode trust in everything around it.

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

### IV. Academic Honesty & Integrity

This project is intended to become a scientific publication and MUST be conducted accordingly.
Results MUST reflect what was actually observed — "expected", hypothetical, simulated, or
placeholder results MUST NEVER be reported or presented as real findings. Work MUST be committed
frequently so the repository preserves a FULL, honest record of the project's evolution. Files
that cannot be easily reproduced MUST NOT be deleted or overwritten.

Rationale: Scientific credibility depends on a faithful, auditable trail and on never confusing
intention or expectation with evidence.

### V. Documentation Stays Current

All documentation MUST always be up to date. Any change to the approach, results, data, code, or
repository structure MUST, within the same change set, update every affected document — including
README files, internal agent files (`CLAUDE.md`), Spec Kit artifacts (`.specify/`, specs, plans,
tasks), AND paper sources (`.tex` files and figures). A change is not complete until its
documentation reflects reality.

Rationale: Stale documentation is actively misleading; co-updating docs with the change that
caused them prevents the record from silently diverging from the work.

## Engineering Constraints & Standards

These constraints operationalize the principles above and apply to all work in this repository.

- **Real verification, never mocks**: Tests and validations MUST exercise real functionality —
  real model calls, real inputs, real files, and real outputs — including the behavior of
  external libraries, models, and resources. Mock-only tests MUST NOT substitute for confirming
  that real components work. If real functionality cannot be validated, the code MUST fail
  loudly rather than silently fall back. (Enforces Principles I & IV.)
- **Reproducible & cached computation**: Computational environments MUST be reproducible
  (pinned dependencies / the containerized environment), and dependency manifests MUST be
  updated whenever dependencies change. Expensive computations MUST be precomputed once and
  cached, with every cached artifact regenerable from its canonical source. (Enforces
  Principles II & III.)
- **Open-weights models for probabilistic visualizations**: Any visualization that depends on
  token-level probability distributions MUST use open-weights models that expose those
  distributions; closed APIs that hide per-token probabilities MUST NOT be relied upon for
  these results. (Enforces Principle I.)
- **Per-feature performance budgets**: Each feature's plan MUST define concrete, measurable
  performance/responsiveness targets (e.g., target frame rate, interaction latency, max
  precompute time) and the strategy (threading, caching, progress UI) used to meet them.
  (Enforces Principle III.)

## Development Workflow & Quality Gates

- **Spec-Driven Development**: Features flow through the Spec Kit pipeline
  (specify → clarify → plan → tasks → analyze → implement). Each plan MUST include a
  Constitution Check that passes before implementation begins.
- **Commit discipline**: Commit frequently and in place. Every commit MUST leave documentation
  consistent (Principle V) and MUST NOT introduce duplicate or backup files (Principle II).
- **Verification before completion**: No task may be marked complete, and no result reported,
  without evidence from a real run or a direct source check (Principles I & IV). Claims of
  "done", "passing", or "fixed" require the supporting command output or artifact.
- **Documentation co-update gate**: A change touching approach, results, data, code, or
  structure is incomplete until the affected READMEs, `CLAUDE.md`, `.specify/` artifacts, and
  paper sources are updated in the same change set.

## Governance

This constitution supersedes all other development practices for this repository. When guidance
conflicts, the constitution wins; direct, explicit user instructions take precedence over
inferred defaults.

- **Amendments**: Proposed via a commit or PR that edits this file, states the rationale, and
  bumps the version per the policy below. Dependent templates and documentation MUST be updated
  in the same change (Principle V).
- **Versioning policy**: Semantic versioning. MAJOR = backward-incompatible governance or
  principle removal/redefinition; MINOR = a new principle/section or materially expanded
  guidance; PATCH = clarifications, wording, and non-semantic refinements.
- **Compliance review**: Every plan's Constitution Check and every `/speckit-analyze` run MUST
  verify compliance with these principles. Violations MUST be remediated, or explicitly
  justified in the plan's Complexity Tracking, before work proceeds.

**Version**: 1.0.0 | **Ratified**: 2026-06-14 | **Last Amended**: 2026-06-14
