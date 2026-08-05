# Session notes — the red-team campaign against all four tabs (2026-08-04)

> **UPDATE after rounds 5–8.** Everything below the "Where things stand" section is the
> record as of the pause; the campaign continued. Current state is at the END of this file,
> under "Rounds 5–8". Read that first.

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

---

# Rounds 5–8 (current state)

## Where it actually stands

`main` = `8826d36`, pushed. Pages deployed and verified **by use**. CI green through
`34027ae` except one backend failure, fixed in `8826d36` (running at the time of writing).

Local, all independently re-run by me rather than taken from agent reports:

| Check | Result |
|-|-|
| backend `pytest` | **680 passed** |
| `ruff` / `black` | clean / 88 files unchanged |
| `svelte-check` | **1186 files, 0 errors, 0 warnings** |
| `vitest` | **851 passed, 1 skipped** |
| `npm run test:e2e` | **68/68**, incl. webgpu on a real `apple/metal-3 (shader-f16)` adapter |

## The arc, because the SHAPE of each round's findings is the useful signal

| Round | What it found |
|-|-|
| 1 (4 agents) | 37 findings — behavioural: wrong numbers, crashes, silent substitutions |
| 2 (verify) | the critical bug SURVIVED its fix, via a path nobody had considered |
| 3 (fix) | fixed it at the *identity* level, not the caching policy |
| 4 (verify) | 27 findings — critical HELD, but the same wrong ANSWER reached around it |
| 5 (fix) | mostly structural: constants, one Unicode table, un-conflated sentinels |
| 6 (verify) | 16 findings, **3 REFUTED** — incl. a fix that REOPENED its own defect |
| 7 (fix) | real pins with proven teeth; every fix mutation-verified |
| 8 (verify) | running |

Convergence is behaviour → prose → structure → test teeth. It is NOT monotonic: round 5
reopened a defect and shipped a facade, which is why rounds kept being worth running.

## Defects that recurred, and what finally worked

- **Vocabulary substitution with self-consistent digests** — 3 fix attempts. Only the third
  held, because it changed the model's IDENTITY (`weights_token` now covers weights **and**
  canonical vocab JSON) instead of the caching policy. Side benefit: a tampered file with a
  recomputed `vocab_sha256` is now refused — the hole the two digests could never close.
- **A stale/unearned error bar** — wrong 3 times: a ± never earned; a retained bound called
  "measured"; then numbers from a configuration that no longer existed. Fixed structurally
  (constants + interpolation) — and that fix was a FACADE (see below) until round 7.
- **Inherited-key lookups** (`x in OBJ`, truthiness index, `{}` as a table) — **five
  consecutive rounds each found MORE**: 2 → 4 → 3 → 2 → 3. Worst: a remote safetensors
  header naming a tensor `__proto__` ran `Object.prototype`'s setter (tensor vanished, its
  `dtype`/`shape` readable on the map); a `dtype:"constructor"` decoded as F16 and returned
  real-looking numbers; `ONNX_REPOS["constructor"]` handed the `Object` function to the ONNX
  runtime. Assume the next sweep is also incomplete.

## Tests that were green for the wrong reason (the campaign's real lesson)

- A test module built its fixtures with the function under test (`_bundle_for` →
  `own_vocab_json`), so writer and reader mutated together: **20 passed while one model had
  two identities**.
- The "structural" constants fix: changing `VACANCY_FP32_REFERENCE.unknownForm`
  0.2872 → 0.4872 **passed all 815 tests**. The Python pin asserted its own literal; the TS
  pin did `toContain(String(constant))` — a tautology. Exchanging two interpolation slots
  also passed 815/815 and restored the exact original defect.
- A nav-guard test asserted on **source text** via `readFileSync` + regex: removing the
  `$effect` wrapper (de-reactifying the wiring entirely) left it green.
- A brand-new e2e test could never have passed on any build — it drove a mechanism the code
  documents as inapplicable (`goto("/#lexicon")` makes Back leave the DOCUMENT, which is
  `beforeunload`'s job, not the in-app prompt's).
- 5 of 7 mutations survived one round; 3 survived all 596 unit tests.

**So: "all tests pass" was repeatedly true and repeatedly meaningless.** Mutation testing is
the only thing that reliably distinguished a pin from a decoration. Require four-state
evidence: mutate → old test passes → new test fails → restore (`shasum`-verified).

## Things I got wrong (recorded so they are not re-learned)

- `d6e9d5d`'s message claimed "the full stack was never affected". False — 2 of 4 sources.
- I called the constants+interpolation change a structural fix that meant "prose can no
  longer disagree with the measurement". It was a facade; a verifier refuted it.
- I transcribed `2.74%` from an agent report into the normative contract, then my own spot
  check said `0.00%` — **my check was wrong** (one map over concatenated passages; the code
  builds one per passage). Re-deriving it correctly found two OTHER figures that were wrong.
- I reported the webgpu project produced no output. It ran and passed; I had grepped a log
  I truncated with `tail`.
- I nearly reported a live TS↔Python divergence in the word-alphabet rule. The UI probe was
  racing with re-render; testing both stacks directly showed they agree on all 10 cases.
- I merged five branches locally and kept going for 36 commits without pushing.

## Method notes that paid off

- **Isolated worktrees + disjoint file ownership**: 5 concurrent fix branches, zero
  conflicts. The one integration failure was a stale fixture, and regenerating it PROVED
  the TS engine reproduces the new Python swap byte-for-byte.
- **Verifiers must never be the fixer.** Every verification round found something its
  fixer counterpart had missed or overclaimed.
- **"Sweep for the pattern, don't patch the instance"** turned 2 known bugs into 4, then 3
  more, then 3 more.
- **Charter the verifier with the failure history**, not a task list. Round 8's brief opens
  with "the central bug survived two complete fixes; assume it is happening again".
- Agents cannot run e2e (it binds the shared ports). Run it yourself — it found a bug a
  fixer AND a verifier had both signed off on.
- Tell agents to use `cp`/`shasum` backups for mutation testing, **never `git checkout`** —
  siblings have uncommitted work.
- `black src/` is repo-wide; scope it while others are editing.

## Open / carried forward

1. **Verification round 8** — results pending.
2. **A warm-cache flake I could not reproduce**: `test_geo_finetune…mints_new_checkpoint`
   failed for one agent on a warm cache (`assert result["cached"] is False`); passes for me
   5/5 cold and warm. May have been fixed incidentally by the round-7 identity/caching work.
   **CI's `restore-keys` are the real test** — watch for it there; do not treat absence as
   proof.
3. Known-and-open, all documented in the agent reports: `ArchitectureExplorer:238`
   unregistered with the nav guard; the 3 geo tablists have `role="tab"` but no `tabpanel`,
   no `aria-controls`, `tabindex 0` on every tab, no keydown handler (**issue #7, left
   open**); issue #6 fixed but **left open for a human to close**; a file whose author
   recomputes EVERY digest is still self-consistent by construction (needs a signature +
   migration); the q8 arm of `VACANCY_Q8_UNCERTAINTY_NATS` can only be re-measured in a
   browser.
4. One source-text assertion remains (`shell.test.ts:355`) with a stated reason —
   `geo-mode`/`geo-layer` need `phase === "ready"`, unreachable in jsdom.

All reports: `notes/agent-reports/{redteam,fix,verify}-007-*.md`.
