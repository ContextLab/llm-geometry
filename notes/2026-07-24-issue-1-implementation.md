# Issue #1 implementation — living notes (started 2026-07-24)

Goal: complete implementation of issue #1 via feature `002-interactive-model-explorer`.
Plan (authoritative, red-teamed twice before posting):
https://github.com/ContextLab/llm-geometry/issues/1#issuecomment-5071465097

## State

- [x] Plan drafted, red-teamed (gap analysis 13 findings + adversarial critic 11
      findings — all incorporated), posted to issue #1.
- [x] Branch `002-interactive-model-explorer` (off 001-core-machinery).
- [x] `specs/002-interactive-model-explorer/spec.md` (FR-101..110, SC-101..104).
- [x] `specs/002-interactive-model-explorer/contracts/api.md` — **FROZEN** contract
      for `/api/geo/*` + `/api/arch/*`. All agents implement against it.
- [x] Batch 0 code stubs (commit 50a38ec) + progress comment posted.
- [x] Batch 1 (parallel): A1 geo (995894d, 54 tests, ALL GATES PASS: loss 4.885,
      coverage 0.9005, entropy 2.812) · A2 arch (00d6be2, 43 tests, traced graph
      424/340 nodes SmolLM2/Qwen) · A3 frontend (bfe389f, 20 tests). Reports in
      notes/agent-reports/batch1-A{1,2,3}.md.
- [x] Integration bug found by combined-suite gate + FIXED (92e5854):
      models/loader.py globally disabled autograd on every HF load (latent 001
      bug) -> geo training failed after any HF test. Fix: model.requires_grad_(False)
      + local no_grad. Also repo-wide ruff/black conformance. 162/162 green.
- [x] Batch 1 red-team: backend ACCEPT-WITH-FIXES (force math verified to ~1e-8;
      2 MAJOR: global RNG leak, unlocked forwards) + frontend REQUEST-CHANGES
      (ApiError hardening, abort on weight getters, component tests). ALL fixed
      (a1327c5) + 2 extra bugs the new tests exposed (svelte server-build
      resolution; heatmap editor bind race). Issue comment posted.
- [x] Batch 2: B1 geo routes (+geo/jobs.py; ADDITIVE Job.phase/result on shared
      registry; python-multipart dep) · B2 arch routes (gate-first, _sig6
      encoding). 36 contract tests. 198/198 green. Commit 93478d5. Comment posted.
- [ ] Batch 3 (running): B3 Architecture Explorer view (viz/arch/) · B4 Geometry
      Lab view (viz/geo/) — both verify live in a real browser via scripts/dev.sh.
- [ ] Batch 2+3 red-team + fixes + screenshots comment.
- [ ] Batch 2 (parallel): B1 geo routes · B2 arch routes (+contract tests).
- [ ] Batch 3 (parallel): B3 Architecture Explorer view · B4 Geometry Lab view.
- [ ] Batch 2+3 red-team + fixes + issue comments with screenshots.
- [ ] Batch 4: e2e, CI (.github/workflows), docs.
- [ ] Full local gate → cleanup → push → PR → final issue comment.

## Key design decisions (details in the issue comment)

- GeoTransformer: d_model=3 (true 3-D, non-negotiable), 4 layers, 1 head,
  mlp_hidden=12, vocab 1003 (=1000+unk/eos/pad), ctx 50, tied unembedding,
  unit-norm embeddings + spherical-uniformity (repulsion) regularizer.
  Non-degeneracy gate: directional entropy of learned field above threshold;
  fallback if it fails: 2 heads / wider MLP / curated corpus. Corpus: real
  public-domain text (Alice in Wonderland).
- weights_token = content hash of full weight set, persisted via cache/store.py.
  Fine-tune mints new checkpoints; canonical `learned` immutable.
- Arch graph from traced forward pass (hooks/torch.fx) incl. functional ops
  (RoPE, softmax, residual adds, activations); tied weights aliased once.
  Default model HuggingFaceTB/SmolLM2-135M-Instruct; ARCH_MAX_PARAMS gate 1.5e9.
- Two field modes: next_next (issue's definition) and force (paper
  arXiv:2607.13295: dz_i/dt = Σ softmax(⟨K z_j, Q z_i⟩)·V z_j; antisymmetrize
  toggle is per-point V·z only — aggregate force shows normal residual).
- Per-view stores (no global collisions); 400 ms debounce + AbortController;
  sessionStorage for weights_token; errors get designed inline states.
- CI tiered: PR gate = smallest real model (SmolLM2-135M) + lint + frontend +
  e2e with HF cache; nightly = full multi-model suite. No mocks anywhere.

## Environment facts

- Baselines green pre-change: backend 65 passed (~30 s warm); svelte-check 0 errors;
  vitest 5 passed. No CI existed before this feature.
- Repo: ContextLab/llm-geometry. Existing e2e: 13 specs. Watch for stale
  uvicorn on :8000 shadowing new ones (`lsof -iTCP:8000`).
- Subagents in this session tend to end with stub final messages ("Complete.") —
  their real reports must be extracted from the transcript JSONL
  (`python3` parser over assistant text blocks; see session history) or re-requested.

## Progress log

- 2026-07-24 11:00 Read issue; explored codebase; researched references
  (paper = "On Transformer Dynamics"; ELIZA demo loop = chat→trace→edit→retest;
  SmolLM2-135M still the right tiny fully-open model — SmolLM3 is 3B-only).
- 2026-07-24 11:3x Plan red-teamed (2 rounds), revised, posted to issue #1
  (comment 5071465097).
- 2026-07-24 11:4x Branch + spec + frozen contract written.
