# Feature 006 — Lexicon Lab

**Feature Branch**: `006-lexicon-lab-tiny`
**Created**: 2026-08-03
**Status**: approved
**Source material**: `~/Desktop/tiny-models` (audited 2026-08-03; see `notes/agent-reports/006-*`)

## What this is

A third explorer tab: a **tiny word-level transformer whose vocabulary budget is the
control you move**. Pick a budget, pick the model's dimensions, train it from scratch in
your browser on real public-domain nursery rhymes, and watch three things respond
together — the loss, the text it generates, and the geometry of its embedding matrix.

The question it makes explorable is the one the source project poses: **what can a
bounded vocabulary learn and say, and what can it not?**

## Why it is NOT a port of the source project

The source (`tiny-models`) is, in its own audit's words, "a proposal plus partial
instrumentation—not a result package." It ships **no trained checkpoint, no corpus, and
no run manifest**; 8 of its 9 paper figures are schematics (one legend reads "schematic
data", another caption "I have drawn my bet, not a finding"). Its own integration doc
says llm-geometry and it are "not checkpoint-compatible" and instructs: "Never make up
browser curves when bundles are absent."

So this tab **generates every number live from a model that actually trained**. It
imports ideas and corrected code, never claims.

### Defects in the source that this feature must not inherit

Each verified by running the source code (`notes/agent-reports/006-source-*`):

| Source defect | Handling here |
|-|-|
| `tiny_lm.py:174` selects `mps`; `svdvals` has no MPS kernel → crashes before step 1 | spectrum via `d×d` Gram eigendecomposition, no SVD |
| `--tie` makes `spectra` log E and U as the same matrix | tied models report **one** spectrum, labelled tied |
| `probe.py:38` drops `tie` on reload → tied checkpoint reloads untied | `tied` travels in the bundle and is asserted on load |
| `--mint` writes a duplicate `itos` entry; the minted row is dead | minting not implemented (FR-624) |
| `dolch.py` first grade has `giving`; the real list has `going` | corrected, with a regression test naming both |
| `MASK_50` documented as "the commonest words in English"; it is verbatim the *Green Eggs and Ham* (1960) vocabulary | **not shipped** |
| `DOLCH_NOUNS` contains `"Santa Claus"`, unmatchable by a word tokenizer | dropped; noun count stated as measured |
| `fingerprint.py` meter score: every word's stress pattern starts `1`, so the anapest score converges to the template's 1/3 density regardless of input | **meter/rhyme metrics not shipped** (FR-625) |
| `mask_decode.py` trie re-opens the root after a completed word → emits `" hameat"` | not needed: this model is natively word-level over the budget |

## User Scenarios & Testing

**US-1 (P1) — Move the budget.** Choose a vocabulary budget and see the model's parameter
count, its coverage of the corpus, and what fraction of the corpus it cannot express,
before training anything.

**US-2 (P1) — Train from scratch, live.** Press train and watch loss, generated samples,
and the embedding spectrum update together, in the browser, on a model really learning.

**US-3 (P2) — Compare a prescribed vocabulary with a descriptive one.** At the same `|V|`
switch between the 1936 Dolch list and the corpus's own top-N words, and see coverage,
loss, and geometry differ.

**US-4 (P2) — Read the geometry honestly.** The spectrum panel shows effective rank
against the `min(|V|−1, d)` ceiling and against an untrained random-init model, so rank
rising with `|V|` without learning is visible rather than hidden.

**US-5 (P2) — Train on my own text.** Paste text, upload a file, or pull a HuggingFace
dataset; the budget is rebuilt from it.

**US-6 (P3) — Modify the model.** Edit weights and see generation and geometry respond.

**US-7 (P3) — Watch it run.** Step the forward pass and see attention and the residual
stream for a chosen prompt.

**US-8 (P3) — Keep it.** Save the trained model to a file and load it back.

## Functional requirements

### Vocabulary and corpus
- **FR-601** Two budget *sources*, switchable at matched `|V|`: `dolch` (the real 1936
  lists, corrected) and `frequency` (top-N types of the active corpus).
- **FR-602** Dolch budget sizes are the real cumulative list sizes, **measured from the
  data, not quoted**: 40, 92, 133, 220, 314. The last is 314 rather than the often-cited
  315 because `Santa Claus` is dropped (FR-601 note); the source shipped it and silently
  had a 314-word "315" budget. `frequency` offers the same sizes, so comparison is at
  matched `|V|`.
- **FR-603** Every budget carries `<unk>`, `<bos>`, `<eos>`, `<pad>`. Reported `|V|` is the
  budget size; total embedding rows is `|V| + 4`, displayed separately.
- **FR-604** Out-of-budget tokens map to `<unk>` in training inputs and targets.
- **FR-605** `<unk>`, `<bos>` and `<pad>` are **masked at generation**, so generated text is
  in-budget by construction. No trie, no post-filter.
- **FR-606** For the active budget the UI always shows: token coverage, `<unk>` rate, and
  the count of corpus lines wholly in-budget.
- **FR-607** Default corpus: Project Gutenberg #10607, *The Real Mother Goose* (1916),
  committed whole with its PG header/footer intact, sha256
  `d514f0fd2cd40967eb6cf35b140a6cddc11200126e07d76603fae3f88bf1e0ab`, 110,445 bytes.
- **FR-608** User corpora: paste, file upload, or HuggingFace dataset (reusing feature
  004's `hfDatasets` path).

### Model
- **FR-609** Pre-norm decoder-only transformer, faithful to the source's *shape*: learned
  absolute positions; multi-head self-attention with a packed QKV projection **with
  bias**; MLP `d → 4d → d` with exact-erf GELU; LayerNorm (eps 1e-5, affine) before
  attention, before the MLP, and finally before readout; readout `Linear(d, |V|,
  bias=False)`.
- **FR-610** `tied` is a user control. When tied, readout weights are the embedding.
- **FR-611** Configurable: `d_model` ∈ {16,32,64,128}, `n_layers` ∈ 1..4, `n_heads` ∈
  {1,2,4} with `d_model % n_heads == 0` enforced in the UI, `ctx` ∈ {32,64,128}.
- **FR-612** Parameter count displayed live, from the formula verified against the source
  on 7 configurations:
  `N = (2 if untied else 1)·|V|·d + ctx·d + L·(12d² + 13d) + 2d`
- **FR-613** Dropout is **0 by default** and exposed as a control. (The source hard-codes
  0.1 and does not expose it; for a live demo determinism matters more.)

### Training
- **FR-614** AdamW, weight decay on matrices only — **not** LayerNorm, biases, or
  embeddings. This deliberately DIFFERS from the source, which decays every parameter;
  the difference is documented in the UI.
- **FR-615** One-cycle LR schedule with `lr` as the peak, matching the source's shape.
- **FR-616** Gradient clipping at global norm 1.0.
- **FR-617** Training runs in a **worker** in the browser and in PyTorch on the backend,
  from the same recipe. Whole-run bit-equality is NOT claimed (documented).
- **FR-618** Live during training: step, loss, LR, elapsed, and a sample generated at a
  user-set interval.
- **FR-619** Fine-tune an existing model on new text, keeping its vocabulary. Feature
  004's issue #6 must not be repeated: the active model's vocabulary is used and travels
  with the result.

### Geometry
- **FR-620** Spectrum of the embedding matrix (and the readout, when untied), computed
  from the `d×d` Gram matrix of the **column-mean-centred** matrix.
- **FR-621** Effective rank `exp(−Σ pᵢ ln pᵢ)` with `pᵢ = σᵢ²/Σσⱼ²`, plus stable rank
  `‖A‖²_F/‖A‖²₂` and participation ratio `1/Σpᵢ²`.
- **FR-622** The panel draws the **`min(|V|−1, d)` ceiling** and an **untrained
  random-init baseline** at the same shape, so the mechanical bound is visible.
- **FR-623** A token cloud is a **PCA projection** and is labelled as such — the Geometry
  Lab's sphere is native 3-D and explicitly is not PCA. Explained variance is displayed.

### Out of scope (stated, not silently omitted)
- **FR-624** Nonce-word minting / the "vacancy" experiment. The source's generator is
  sound and deterministic, but the experiment needs a parameter-matched control that does
  not exist. Tracked as a follow-up issue.
- **FR-625** Meter, rhyme, and the prosody "fingerprint". The source's meter score does
  not measure meter (proved: "and I do not like green eggs and ham" scores 0.333 against
  a nonsense corpus's 0.346). Shipping it would ship a broken instrument.
- **FR-626** The paper's T1–T6 predictions. This tab measures; it does not adjudicate.

## Success criteria

- **SC-601** A cold visitor can train a model to visibly falling loss in **under 60 s** in
  a browser at the default configuration.
- **SC-602** Generated text contains **zero** out-of-budget words, verified
  programmatically at every budget size.
- **SC-603** Switching budget source at matched `|V|` measurably changes coverage, and the
  numbers shown match a backend computation of the same quantity exactly.
- **SC-604** The effective rank of a **random-init** model at `d=128` rises strictly across
  budgets 40→314 — measured **36.21, 64.38, 79.11, 95.73, 104.16**, with no learning at
  all. That is the confound, reproducible in the UI rather than hidden by it.
  The two ranks behave differently and the UI must not conflate them: **algebraic** rank
  plateaus exactly at the `min(|V|−1,d)` ceiling (128 from the `first` budget onward),
  while **effective** rank only approaches it — it is an entropy, not a count.
  The rise is strict but **not smooth**: the increments are +28.18, +14.73, **+16.62**,
  +8.42, so it is not a monotonically decelerating curve. (An earlier draft of this line
  claimed "+29.3 … decelerating"; both were wrong, the first copied without recomputing.
  Do not describe the shape beyond "strictly rising" without re-measuring.)
- **SC-605** TS and Python agree to ≤1e-5 on forward pass, loss, and every spectrum
  statistic, for a fixed set of weights (golden test).
- **SC-606** Every constant quoted in the tab's prose is pinned by a test, as feature 005
  established.
- **SC-607** Saving and reloading a model reproduces its generation exactly.

## Known limits (stated in the UI)

- Whole-run training equality between browser and Python is not achievable
  (platform-divergent BLAS, non-portable RNG). The recipe is identical; a run is not.
- The corpus is nursery rhymes, not Dr. Seuss, whose work is under copyright. The Dolch
  list is a real 1936 pedagogical word list, not Seuss's vocabulary.
- Effective rank is bounded by `min(|V|−1, d)`. The tab draws that bound because a
  staircase in rank against `|V|` is expected **even for random matrices**.
- Nothing here validates the source paper's predictions.
