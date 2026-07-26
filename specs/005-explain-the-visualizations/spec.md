# Feature 005 — Explain the visualizations

**Status**: implemented · **Date**: 2026-07-26 · **Landed on**: `main`

## Why

A researcher working in this area opened https://context-lab.com/llm-geometry/ and could not
tell what was being shown. That is a documentation defect, not a taste disagreement: both tabs
render real tensors from real models, and neither one said enough for a reader to know what the
numbers were, what the controls did, or which claims the deployment could actually keep.

Feature 004 shipped correct visualizations with one-line captions. This feature supplies the
missing text. The audience is assumed to be **mathematically sophisticated** — equations are
stated outright rather than analogized away.

## User stories

- **US-1** — As a first-time visitor, I can find a single place that explains what the project is,
  what each tab shows, and what the mathematics behind it is, without reading the source.
- **US-2** — As someone looking at a tab, I can learn what is on screen *in that tab*, next to the
  thing being explained, without navigating away.
- **US-3** — As a reader deciding whether to trust a picture, I can find out what is computed live,
  what is precomputed, what is ported, and what this deployment cannot do.
- **US-4** — As someone about to change a control, I can find out what it changes mathematically
  and what to expect on screen — including which controls barely move the picture and why.

## Functional requirements

- **FR-501** — A third tab, **Info**, is a standalone reference: framing, notation, per-tab
  explanations, the honesty table, known limits, and verified references.
- **FR-502** — The Info tab states the GeoTransformer's forward pass as equations matching
  `geo/model.py` exactly: unscaled scores, no layer norm, tied unembedding, learned absolute
  positions.
- **FR-503** — Both field modes are defined mathematically, including that the aggregate attention
  forces are projected onto the tangent plane **at the anchor** and that antisymmetrizing `W_V`
  does not make them tangent there.
- **FR-504** — Each explorer tab carries always-visible orientation prose plus collapsible
  deep-dives (`Explain`) placed beside what they explain.
- **FR-505** — Every documented control names what it changes and what to watch, including the
  cases where the visible effect is small (`W_Q`/`W_K` at temperature 0).
- **FR-506** — The Info tab states what is real and where it runs, differentiating the full stack
  from the static build, and names the static build's real limitation (ONNX exports expose no
  hidden states).
- **FR-507** — Known limits are stated with their tracking issues (#4, #5) rather than omitted.
- **FR-508** — Every external link is verified to resolve, opens in a new tab, and carries
  `rel="noopener"`.
- **FR-509** — Documented numbers are pinned by tests against the running system, so the prose
  fails CI rather than silently rotting when a constant changes.
- **FR-510** — The active tab is mirrored into `location.hash`, so the explanation can be linked,
  bookmarked, and reached with Back. An unknown fragment falls back to the landing tab.
- **FR-511** — A first-time visitor gets one always-visible pointer to the Info tab from the
  landing tab, which retires itself once they have opened it.
- **FR-512** — Where a picture encodes something the reader cannot recover from it — arrow lengths
  are renormalized per render and clipped, the whole-matrix heat map is a quantized overview —
  the documentation says so rather than letting the picture imply precision it does not have.

## Non-functional requirements

- **NFR-501** — No new runtime dependency. Mathematics is set in Unicode + CSS; adding KaTeX would
  have meant self-hosting fonts under the static build's CSP for a typographic gain only.
- **NFR-502** — Explainers are `<details>`, so they work without JS, are keyboard-reachable, and
  are collapsed by default — the visualization stays above the fold.
- **NFR-503** — The Info tab renders identically in the static build; it depends on no endpoint.

## Success criteria

- **SC-501** — A reader can answer, from the site alone: what the sphere is, why it is not a
  projection, what both arrow types mean, and what the badge numbers measure.
- **SC-502** — Every numeric claim in the documentation matches its source constant; verified by
  an adversarial pass over the code, not by the author's memory.
- **SC-503** — All three tabs, both e2e projects, and the full backend suite pass.

## Found by the red team, deferred with issues

- **Issue #6** — fine-tuning tokenizes with the shipped vocabulary regardless of the base model,
  so fine-tuning a from-scratch model uses the wrong word list and discards its vocabulary. A real
  defect, not a documentation error; the docs state the limitation and link to it.
- **Issue #7** — hover-only tooltips (unreachable by keyboard or touch), `role="tablist"` on
  controls that are not tabs, and the notation glossary marked up as a table.

## Out of scope

- Rewriting the visualizations themselves. This feature adds explanation; 004's behavior stands.
- Widening the model menu (issue #4) or pinning ONNX mirrors (issue #5) — both are *documented*
  here as limits, not fixed.

## Also landed in this change

The nightly CI run of 2026-07-26 failed: HuggingFace returned `429 Too Many Requests` and
`scripts/export_static_assets.py` hard-failed on the first rate-limited metadata call. Hub metadata
calls now go through `llm_geometry.models.hub.hub_call`, which retries 429/5xx with exponential
backoff (honouring `Retry-After`) and never swallows the error. The static export additionally
refuses to publish a model whose revision did not resolve to a real commit sha, so a rate-limited
run cannot quietly bake unpinned weight URLs into the deployed artifact.
