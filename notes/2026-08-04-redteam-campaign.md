# Session notes — the red-team campaign against all four tabs (2026-08-04)

**PAUSED mid-flight.** Read this first on resume.

## Where things stand

- PR #8 (feature 007, the vacancy transform) is **merged** to `main` as `0ed5365`.
- The Pages deploy for it ran and is live — bundle `B4JW7yLY` → `Bjyi2uBc`, verified by use.
- Everything after that merge is **LOCAL ONLY**. `main` is ~24 commits ahead of
  `origin/main`. **Nothing from the campaign has been pushed, run in CI, or run on Linux.**
  The deployed site is entirely pre-fix.

## THE IMMEDIATE NEXT STEP

```bash
cd code/frontend && npm run test:e2e        # 65 tests, ~6 min, all three projects
```

The last commit (`1b70966`) contains two changes that are **reasoned but NOT verified** —
the confirming run was interrupted after 1 test when the machine was suspended. Do not
trust them until that run is green:

1. `src/lib/stores.ts` — mid-session hash canonicalization (a real bug, see below)
2. `tests/e2e/static.spec.ts` — an updated assertion

Before that run, the suite stood at **63 passed / 2 failed**, and those 2 are exactly what
`1b70966` addresses. Everything else is green (see "Verified state" below).

## What happened, in order

1. Merged PR #8, verified the live deploy **by using it** (real 41.7 s training run in the
   browser: loss 5.790 → 2.244; real vacancy scoring, 158 s on Qwen2.5-0.5B at q8/webgpu).
2. **Red-team round 1** — 4 agents, one per surface, both data modes. **37 findings.**
3. **5 fix branches** in isolated worktrees, disjoint file ownership, all merged.
4. A **reconcile** pass for cross-cutting debt the rewrite created.
5. **Verify round 2** — 3 fresh agents, none reviewing its own work, instructed to
   *refute*. Found the critical bug had **survived**, plus a hole nobody reported.
6. **Round 3** — 3 more fixers. All merged.
7. First **e2e run of the campaign** → the 2 failures above.

## The findings that mattered

- **Geo F1 (critical), survived two rounds.** Vocabulary substitution with self-consistent
  digests. Round 1 fixed the derivation chain; round 2 found it still reachable through
  `save_weight_set`'s content-hash **dedup** (first-write-wins in Python, last-write-wins in
  TS — the two stacks corrupted *different* models). Round 3 fixed the root cause: the
  identity itself. `weights_token` now covers weights **and** the canonical vocabulary JSON.
  Side effect worth keeping: a tampered file with a recomputed `vocab_sha256` is now
  refused — the hole the two digests provably could not close.
- **Arch F1 → vacancy TASK 1.** The `swap` control emitted non-words (`wented`, `kitser`)
  because the map was stem-keyed over a pool of inflected types. Fixing it exposed a bigger
  bug: `stem_and_suffix` split the **closed class** open (`after→aft+er`, `this→thi+s`, and 5
  more), so words the experiment holds fixed were being vacated **in both arms**. Swap is now
  a type→type derangement inside one suffix class. Measured effect on the headline:
  `unknown_form` 0.2726 → 0.2872 — about **a third of the SE**, so no conclusion moves.
- **The error bar, wrong twice in opposite directions.** First it printed a ± it had not
  earned (interval excluded the fp32 truth). After the fix it printed `± 0.2 (quantization,
  measured)` while the contract said that bound must **not** be called measured. Now reads
  "retained bound — not re-measured since the swap rewrite", pinned both ways.
- **`in` as a validity test** — found in **4** places because one agent was told to sweep for
  the pattern rather than patch the instance. The worst: a remote safetensors header with
  `dtype:"constructor"` passed the gate and `decodeScalars` decoded it as **F16**, returning
  real-looking numbers. All now `Object.hasOwn`.
- **Docs.** The original audit traced 60+ numeric claims with **zero drift** — then the
  transform rewrite moved the statistics and one copy of four rotted, at the **static wire
  boundary**, the path the public site runs. Also: the Dolch grade-split claim was **false**
  (an agent read the 1936 article; the 220 words and 95 nouns are in it, the grade split is
  not — Dolch groups by part of speech).

## Things I got wrong, recorded so they are not re-learned

- `d6e9d5d`'s commit message claimed "the full stack was never affected". **False** — Python
  passed `vocab_json` for 2 of 4 sources.
- I transcribed `2.74%` from an agent report into the normative contract, then my own spot
  check said `0.00%`. **My check was wrong** (one map over concatenated passages; the code
  builds one per passage). The number was right — but re-deriving it found two that were
  not: "five of the six passages have a singleton suffix class" (**all six** do, wrong in 4
  places) and "smallest class holds 8 types" (**7** — `n't`).
- I merged five branches locally and kept going without pushing. A verifier caught it.

## Method notes that paid off

- **Isolated worktrees + disjoint file ownership.** Zero merge conflicts across 5 branches.
  The one integration failure was a *stale fixture* (a parity case generated before the swap
  rewrite) — and regenerating it proved the TS engine reproduces the new Python swap
  byte-for-byte.
- **Verifiers must not be fixers.** Every round-2 agent found something its round-1
  counterpart had missed or overclaimed.
- **"Sweep for the pattern, don't patch the instance"** turned 2 known bugs into 4 found.
- **Mutation testing caught worthless tests** (17/19 had teeth; a TS export refusal could be
  deleted with 66 tests still passing).
- **e2e is not optional.** No agent could run it (it binds the shared ports), and it found a
  bug that a fixer *and* a verifier had both signed off on.

## Remaining work

1. `npm run test:e2e` → confirm 65/65. **Blocking everything else.**
2. Re-run the full suite (rule: fixing one check invalidates the others):
   backend `pytest -q` + `ruff` + `black --check`; frontend `npm run check` + `vitest` + `build`.
3. **Push** (~24 commits). Watch CI — Linux, Python 3.10, real model downloads. `HF_TOKEN`
   is wired into both workflows now.
4. Pages redeploy → verify the live site **by using it**, not by the workflow tick.
5. **Verify round 4** — fresh agents against the round-3 fixes. The critical bug has now
   survived two verification rounds; do not declare done without another.
6. Unfixed / deliberately deferred, all documented in the agent reports:
   - `ArchitectureExplorer:238` unregistered with the nav guard
   - The 3 geo tablists: `role="tab"` present but no `tabpanel`, no `aria-controls`,
     `tabindex 0` on every tab, no keydown handler (issue #7, **left open**)
   - Issue #6 fixed but **left open for a human to close**
   - A file whose author recomputes *every* digest is still self-consistent by construction
     (needs a signature + migration)
   - The q8 arm of `VACANCY_Q8_UNCERTAINTY_NATS` — only a browser can re-measure it
7. Clean up: `.claude/worktrees/` (auto-removed, but verify), scratch `.mjs`/`zz*.test.ts`
   probes, and `git worktree list` should be back to one entry.

## Verified state at pause (local, macOS, warm cache)

| Check | Result |
|-|-|
| backend `pytest -q` | **532 passed** |
| `ruff` / `black` | clean / 82 files unchanged |
| `svelte-check` | **1173 files, 0 errors, 0 warnings** |
| `vitest` | **596 passed, 1 skipped** |
| `npm run build` | ✓ |
| `npm run test:e2e` | **63 passed / 2 failed** → both addressed in `1b70966`, unconfirmed |

Agent reports (all findings, with verbatim evidence) are in `notes/agent-reports/`:
`redteam-007-{arch,geo,lex,docs-shell}.md`, `fix-007-*.md`, `verify-007-*.md`.
