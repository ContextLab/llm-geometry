# Feature 006 — Lexicon Lab

**Date**: 2026-08-03/04 · **Spec**: `specs/006-lexicon-lab-tiny/`
**Asked for**: a new tab built from `~/Desktop/tiny-models` — an interactive, modifiable,
retrainable, animatable version of that project's tiny constrained-vocabulary model.

## What the source material actually was

Five analysis agents read the bundle and RAN its code. Reports:
`notes/agent-reports/006-source-{model-arch,vocabulary,eval,paper-claims}.md` and
`006-corpus-sourcing.md`.

The bundle is, in its own audit's words, **"a proposal plus partial instrumentation—not a
result package."** No trained checkpoint. No generated corpus. No run manifest. 8 of its 9
paper figures are schematics — one legend reads "schematic data", one caption "I have
drawn my bet, not a finding". Only four things survive its own audit as demonstrated:
budget cardinalities, single-word out-of-budget detection, vacancy nesting/stability, and
one demo fingerprint.

**It also does not run.** `tiny_lm.py:174` selects the `mps` device and
`torch.linalg.svdvals` has no MPS kernel; it is called before training step 1. Reproduced:
`NotImplementedError` on any Mac. 17 defects found in total.

### The four that mattered most

| Defect | Why it mattered |
|-|-|
| `--tie` logs E and U as the same matrix | corrupts the exact rank evidence that file exists to produce |
| `fingerprint.py` meter score | every word's stress pattern starts `1`, so the anapest score converges to the template's 1/3 density. *"and I do not like green eggs and ham"* scores **0.333**; the nonsense corpus scores **0.346**. It does not measure meter. Note the audit lists this number under "What held up" — it is reproducible, but it is not a measurement. |
| `mask_decode.py` trie | re-opens the root after a completed word. Verified with real GPT-2: after `" ham"` it permits `"eat"` → **`" hameat"`**, the exact failure its docstring claims to prevent |
| `dolch.py` first grade has `giving` | the published list has `going`; the word occurs 27 times in our corpus, so the slip cost real coverage |

Plus a provenance problem: `MASK_50`, documented as "fifty of the commonest words in
English", is verbatim the vocabulary of a 1960 picture book (`eggs`, `ham`, `Sam`, `goat`,
`fox`, `mouse`, `anywhere`). Not shipped.

## What we built instead

A tab where **the vocabulary budget is the control**. Not a port — the source's own
integration doc says the two are "not checkpoint-compatible" and warns "Never make up
browser curves when bundles are absent." So every number is generated live from a model
that actually trained.

- **Corpus**: PG #10607, *The Real Mother Goose* (1916), committed whole, digest-verified.
  Chosen over Lear/Stevenson/others because it is the only large candidate with zero
  `[Illustration]` markers, zero tables, zero transcriber notes — and because 9.1% of its
  lines are exact repeats (Alice: 0.8%), which is the property a tiny model can learn.
- **Two budget sources at matched |V|**: Dolch (prescribed, 1936) vs corpus frequency
  (descriptive). The descriptive budget wins at every size on its own corpus — 70.7% vs
  60.8% coverage at |V|=314. That comparison is the tab's most interesting control.
- **In-budget by construction**: the vocabulary IS the budget, so no trie and no filter.
  The source's decoder bug is not inherited, it is designed out.
- **Left out on purpose** (FR-624..626): minting/vacancy (no parameter-matched control),
  meter/rhyme (broken instrument), and the paper's T1–T6 predictions.

## Decisions worth remembering

- **314, not 315.** `Santa Claus` has a space and no word tokenizer can match it. The
  source shipped it, making its "315" budget silently 314 wide. We drop it and report the
  measured number. Budget sizes are computed from the data, never quoted.
- **Effective rank is an entropy, not a count.** Algebraic rank plateaus exactly at
  `min(|V|−1,d)`; effective rank only approaches it, with decelerating increments. I wrote
  the spec loosely at first and an agent's measurement corrected it.
- **The confound is the feature.** A random-init model's effective rank climbs
  36.21 → 64.38 → 79.11 → 95.73 → 104.16 across the five budgets *with no learning at
  all*. So the panel never draws a rank curve without the `min(|V|−1,d)` ceiling and an
  untrained baseline beside it. A "staircase" against |V| is the null result.
- **PCA sign convention.** Eigenvectors are defined only up to sign, so two correct
  implementations produce mirror-image clouds while every scalar agrees. Fixed
  (largest-magnitude entry positive) and written into the contract — without it the golden
  test fails on `pca_coords` alone and looks like a numerical bug.
- **Deliberate divergence from the source**: weight decay applies to weight matrices only,
  not LayerNorm/biases/embeddings (the source decays everything). Marked `[DIFFERS]` in the
  contract and stated in the UI rather than changed quietly.

## The key bias is dead — found by the parity test

The golden TS↔Python test met ≤1e-5 everywhere except one category, and chasing that
exception found real mathematics rather than a bug.

`qkv_b`'s **K slice has identically zero gradient**: adding `b_k` to every key shifts
`scores_ij` by `q_i·b_k/√dh`, constant along `j`, and softmax ignores a constant shift
along its axis. Verified on the real model — `|∂L/∂b_q|` = 2.8e-3, `|∂L/∂b_v|` = 1.7e-2,
`|∂L/∂b_k|` = **3.5e-10**. That is `d` dead parameters per layer.

Why it showed up as a parity failure: AdamW's first step is scale-invariant
(`m̂ = g`, `√v̂ = |g|` at `t=1`, so the update is `−lr·sign(g)`), so a parameter whose true
gradient is below Adam's `ε` has its step decided by roundoff — float32 and float64
disagree by up to a full `lr`. The fix was NOT to loosen the tolerance: the test splits on
the optimizer's own `ADAM_EPS`, holds well-conditioned weights to 1e-5 (worst 2.11e-6),
holds the degenerate ones to one step's reach, and asserts the exempt fraction stays under
2% (measured 0.4–1.5%) so it cannot grow to hide a real regression.

Measured parity, all ≤1e-5 as required:

| quantity | max deviation |
|-|-|
| one-cycle LR schedule | 0 (exact) |
| forward logits | 3.70e-7 |
| loss | 2.85e-8 |
| gradients (every tensor) | 6.97e-8 |
| spectrum scalars | 1.40e-15 |
| PCA coordinates | 1.64e-15 |
| AdamW step (well-conditioned) | 2.11e-6 |

The PCA figure is only that good because the sign convention was fixed first; without it
the clouds mirror and every scalar still agrees.

## Cross-agent catches

Worth noting because it is the argument for running these in parallel rather than serially:

- The model agent recomputed a parameter count from the architecture report and found it
  inconsistent — 912,128 requires V=400, but the row said V=395. The formula was right; the
  report had labelled *word types* as *embedding rows*. Corrected in the report.
- The spectrum agent found my SC-604 wording wrong (see above) and found the PCA sign gap
  the contract was silent on.
- The eval agent disproved a number the source's own audit had filed under "What held up".

## Verification

Backend: **333 passed** (151 pre-existing + the new suites), ruff + black clean.
Real training, default config: loss **5.771 → 2.531** in 400 steps / 5.3 s; first loss
equals `ln(318)` exactly, as it must for an untrained model over 318 rows.
SC-602 (zero out-of-budget words) holds across 5 budgets × 4 seeds × 3 temperatures.

## Throughput: profiled, not guessed

~1336 ms/step at the original defaults meant a ~9-minute default run. The profile
disproved the obvious hypothesis: allocation was NOT a factor (40 MB of typed arrays costs
1.15 ms) and GC was 0.4%; all three matmul kernels simply ran at a uniform 0.85 GMAC/s.

4×4 register blocking, routing the transposing kernels through the blocked one, and
head-contiguous attention took it to **465 ms/step (2.87×)** — **bit-identical**, verified
by hashing every weight after a 12-step run. Unrolling interleaves independent accumulator
chains and never reassociates one, which is why parity survived. Two other optimizations
were tried and **reverted** for measuring ~0.

Defaults then moved `ctx 64→32`, `batch 32→16` (both languages), taking a full run from
**193.2 s → 44.7 s**. The context change is justified on QUALITY, not speed: at ctx 64 the
model gets a *better* train loss and a *worse* val loss, because a 64-token window over a
book of nursery rhymes spans several unrelated ones and it memorizes.

## Red team — 5 findings, 2 CRITICAL, 2 of them mine

Report: `notes/agent-reports/006-redteam-lexicon.md`. Everything below was reproduced.

**CRITICAL — the two runtimes trained on different data.** Python emitted `<eos>` after
every line (19,071 tokens / 3,071 markers); the browser flat-tokenized to 16,000 with
none. So the browser model could never learn where a line ends — a real 200-step run
produced one unbroken line — while the UI claimed both ran "the same recipe" and blamed
divergence on platform BLAS. **Root cause was this spec**: `architecture.md` said "the
token stream" without ever defining how text becomes one, so the two implementations
filled the gap differently. Now specified, and both sides must reproduce 19,071/3,071.

**CRITICAL — "in budget by construction" was defeatable, with a post-filter hiding it.**
`NaN > -Infinity` is false, so with non-finite weights the greedy argmax never updates and
falls through to its initialiser `next = 0` — which is `<unk>`, bypassing the -inf ban.
Stripping specials from `words` then removed the evidence: exactly the post-filter the
prose says does not exist. Reproduced live with clicks alone (a `×2` preset ~140 times):
40 words printed under a green "every one drawn from the 314-word budget" badge. Python
raised; TS had no guard. Now both refuse, naming the offending tensor.

**MINE — the effective-rank prose was wrong twice in one sentence.** I wrote "decelerating
increments (+29.3 … +8.4)". The real increments are +28.18, +14.73, **+16.62**, +8.42 —
the first figure was copied from an agent's report without recomputing, and the sequence
is not decelerating at all. Both documents now carry the measured values and an explicit
instruction not to describe that curve's shape without re-measuring.

**MINE — SC-606 was unmet.** `docs.spec.ts` had zero Lexicon assertions, leaving every
number in the new tab's prose unpinned — in a feature that had already produced two
stale-number defects. Now pinned against live `/api/lex/*` measurements.

**HIGH — one weight-lab click made five panels lie at once.** `isTrained` ignored `edited`,
so after editing an untrained model the spectrum said "these two are the same model" beside
0.00 vs 57.67, the cloud said "random Gaussian matrix", and save said it "writes the random
initialization". Replaced with a 4-state provenance (`untrained` / `trained` /
`edited-untrained` / `edited-trained`) read by every panel.

Also: the Info tab still described a two-explorer app and claimed "Both tabs run against
real PyTorch" — false for a tab that never calls the backend; and the corpus checksum chip
advertised a verification the browser never performed, over the untrimmed file rather than
the loaded body. The browser now genuinely rehashes what it loaded and refuses a mismatch.

## Verification

Backend **336 passed**, ruff + black clean. Frontend **247 passed** (1 skipped),
`svelte-check` 0 errors / 0 warnings over 1147 files, production build clean.
Golden TS↔Python parity ≤1e-5 on forward logits (3.70e-7), loss (2.85e-8), gradients
(6.97e-8), spectrum scalars (1.40e-15), PCA coords (1.64e-15), AdamW step (2.11e-6).

Verified in a real browser: coverage counters match the Python measurement exactly
(60.8% / 39.2% / 215 of 3,071); a 400-step d=32 run produces **13 lines of verse** with
`<eos>` at a 10% rate against the stream's 16.1%; all six non-finite attack variants
refuse at both temperatures; 0 console errors.
