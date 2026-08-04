# Lexicon Lab — the exact model, training recipe, and spectrum

**This file is the contract.** The PyTorch backend and the TypeScript browser engine both
implement exactly this, and a golden test holds them to ≤1e-5. If you change anything
here, change both implementations and the golden vectors in the same commit.

Derived from `tiny-models/tiny-seuss/train/tiny_lm.py` by running it
(`notes/agent-reports/006-source-model-arch.md`). Deliberate departures from the source
are marked **[DIFFERS]** with a reason; everything else is faithful.

## Shapes

`V` = embedding rows (`budget_size + 4` specials) · `d` = `d_model` · `L` = layers ·
`H` = heads · `dh = d/H` · `ctx` = context length.

| Parameter | Shape | Initialization |
|-|-|-|
| `embed` | `(V, d)` | `N(0, 0.02²)` |
| `pos` | `(ctx, d)` | `N(0, 0.02²)` |
| `ln1_g`, `ln1_b` | `(d,)` each | ones, zeros |
| `qkv_w` | `(3d, d)` | **xavier-uniform**, bound `sqrt(6/(3d+d))` |
| `qkv_b` | `(3d,)` | zeros |
| `proj_w` | `(d, d)` | `N(0, 0.02²)` |
| `proj_b` | `(d,)` | zeros |
| `ln2_g`, `ln2_b` | `(d,)` each | ones, zeros |
| `fc1_w` | `(4d, d)` | `N(0, 0.02²)` |
| `fc1_b` | `(4d,)` | zeros |
| `fc2_w` | `(d, 4d)` | `N(0, 0.02²)` |
| `fc2_b` | `(d,)` | zeros |
| `lnf_g`, `lnf_b` | `(d,)` each | ones, zeros |
| `head_w` | `(V, d)` | tied ⇒ **is** `embed`; else `N(0, 0.02²)` |

The mixed init is not a mistake to clean up: the source's `_init` only matches
`nn.Linear`/`nn.Embedding`, so the packed QKV parameter keeps PyTorch's xavier default
(measured std 0.0627, bound 0.10825) while every other matrix gets `N(0, 0.02²)`.
Reproducing it keeps us honest about what the source model actually is.

Parameter count (verified on 7 configs):
`N = (2 if untied else 1)·V·d + ctx·d + L·(12d² + 13d) + 2d`

## Forward pass

Input `x: (B, T)` of token ids, `T ≤ ctx`. LayerNorm is `eps = 1e-5`, affine, over the
last axis.

```
h = embed[x] + pos[:T]                                  # (B, T, d)
h = dropout(h)

for each layer:
    a          = layernorm(h, ln1_g, ln1_b)
    qkv        = a @ qkv_w.T + qkv_b                    # (B, T, 3d)
    q, k, v    = split(qkv, 3, axis=-1)                 # each (B, T, d)
    reshape each to (B, H, T, dh)
    scores     = q @ kᵀ / sqrt(dh)                      # (B, H, T, T)
    scores     = scores + causal_mask                   # -inf strictly above diagonal
    A          = softmax(scores, axis=-1)
    A          = dropout(A)                             # dropout on ATTENTION WEIGHTS
    o          = A @ v                                  # (B, H, T, dh)
    o          = merge_heads(o)                         # (B, T, d)
    o          = o @ proj_w.T + proj_b
    h          = h + o                                  # NOTE: no dropout on this branch

    m          = layernorm(h, ln2_g, ln2_b)
    m          = gelu(m @ fc1_w.T + fc1_b)              # exact erf GELU
    m          = m @ fc2_w.T + fc2_b
    m          = dropout(m)
    h          = h + m

h      = layernorm(h, lnf_g, lnf_b)
logits = h @ head_w.T                                   # (B, T, V), NO bias
```

`gelu(x) = 0.5·x·(1 + erf(x/√2))` — the exact form, not the tanh approximation. The
source uses `nn.GELU()` with default `approximate='none'`.

### The key bias is a dead parameter

`qkv_b` carries biases for Q, K and V, but the **K slice has identically zero gradient**.
Adding `b_k` to every key shifts `scores_ij` by `q_i·b_k / √dh`, which is constant along
`j` — and softmax is invariant to a constant shift along the axis it normalizes. So
`∂L/∂b_k ≡ 0` analytically.

Measured on the real implementation (d=32, L=2, one backward pass): `|∂L/∂b_q|` = 2.8e-3
and `|∂L/∂b_v|` = 1.7e-2, while `|∂L/∂b_k|` = 3.5e-10 — numerically zero, seven orders
down. That is `d` dead parameters per layer (128 of 122,496 at the default shape).

This is inherited from the source's packed-QKV projection and is kept for fidelity, not
because it is a good idea; many implementations set `bias=False` on the key projection for
exactly this reason. It matters for two practical purposes:

1. **Parity testing.** AdamW's first step is scale-invariant — at `t=1`, `m̂ = g` and
   `√v̂ = |g|`, so the update is `−lr·g/(|g| + ε) = −lr·sign(g)` whatever the magnitude.
   A parameter whose true gradient is below `ε` therefore has its step decided by
   roundoff, and float32 vs float64 can disagree by a full `lr`. The golden test splits on
   the optimizer's own `ADAM_EPS` rather than loosening its tolerance.
2. **Anything that counts "trainable" parameters** should know some of them are not.

Dropout placement is the source's and is unusual: on the embedding sum, on the attention
weights, and after the second MLP linear — **not** on the attention residual branch.
**[DIFFERS]** default is `0.0` here, not the source's hard-coded `0.1`, and it is exposed
as a control: a live demo people re-run should be deterministic by default.

## Training

- Loss: mean cross-entropy of `logits[:, :-1]` against `x[:, 1:]`, `ignore_index = PAD_ID (3)`.
- Optimizer: **AdamW**, `betas = (0.9, 0.999)`, `eps = 1e-8`.
  - **[DIFFERS]** weight decay applies **only to 2-D weight matrices**: `qkv_w`,
    `proj_w`, `fc1_w`, `fc2_w`, and `head_w` when untied. NOT `embed`, NOT `pos`, NOT any
    bias, NOT LayerNorm gains. The source decays every parameter including LayerNorm and
    embeddings, which is a known anti-pattern; we follow the standard convention and say
    so in the UI.
  - **[DIFFERS]** `beta1` is fixed at 0.9. The source's OneCycleLR also cycles beta1
    between 0.95 and 0.85; that adds a second schedule to mirror exactly across two
    languages for no pedagogical gain.
- LR schedule: one-cycle with `lr` as the **peak**, matching the source's shape.
  - `initial = lr / 25`, `final = initial / 1e4`, `pct_start = 0.3`, cosine both phases.
  - Step `i` of `S` (0-indexed), `w = round(pct_start·S)`:
    - `i < w`:  `p = i/w`,        `lr_i = initial + (lr − initial)·(1 − cos(πp))/2`
    - `i ≥ w`:  `p = (i−w)/(S−w)`, `lr_i = final + (lr − final)·(1 + cos(πp))/2`
- Gradient clipping: global L2 norm 1.0, applied before the step.
- **Token stream construction** (text → ids). This was previously left unstated, and the
  two implementations filled the gap differently — Python inserted line markers, the
  browser did not, so the browser model never saw a line boundary and could not learn
  verse while the UI claimed both ran the same recipe. It is now part of the contract:

  > Split the text into lines. For each line with at least one word token, emit that
  > line's ids **followed by `<eos>`**. Blank lines emit nothing. Nothing is prepended.

  For the shipped corpus this yields exactly **19,071 tokens including 3,071 `<eos>`** —
  a number both sides must reproduce, asserted by a test. `<eos>` is what makes the model
  produce verse rather than one undifferentiated run of words, which is the entire reason
  generation renders it as a line break.

- Batching: contiguous 95/5 train/val split of the token stream; each batch draws `B`
  start offsets uniformly **with replacement** from the training span; a window is
  `ctx+1` tokens so the last token has a target.
- Determinism: a seeded RNG drives init and batch selection. Bit-equality between the
  browser and Python is **not** claimed (platform BLAS, non-portable RNG streams).

## Generation

Greedy at `temperature = 0`, else sample from `softmax(logits[-1] / T)` after setting the
logits of `GENERATION_BANNED_IDS = (UNK_ID, BOS_ID, PAD_ID)` to `-inf`. `<eos>` is
sampleable and ends the line. Because the vocabulary **is** the budget, in-budget output
is guaranteed by construction — no trie, no post-filter, no possibility of the source's
`" hameat"` defect.

## Spectrum

For a matrix `A` of shape `(V, d)` (the embedding, or the readout when untied):

1. Column-mean-centre: `Ac = A − mean(A, axis=0)`. Centring is why the maximum attainable
   rank is `min(V−1, d)`.
2. Gram: `G = Acᵀ Ac`, shape `(d, d)`. Symmetric eigendecomposition gives eigenvalues
   `λᵢ = σᵢ²` directly — **no SVD**. This is what makes it 2 ms in a browser, and it
   sidesteps the source's crash (`torch.linalg.svdvals` has no MPS kernel).
3. Clamp `λᵢ ← max(λᵢ, 0)` (negative values are float error), sort descending,
   `σᵢ = sqrt(λᵢ)`.
4. With `pᵢ = λᵢ / Σλⱼ`:
   - `effective_rank = exp(−Σ pᵢ ln pᵢ)`, terms with `pᵢ = 0` contribute 0
   - `stable_rank    = Σλⱼ / λ₁`
   - `participation  = 1 / Σ pᵢ²`
   - `frac_var_top2`, `frac_var_top10`, `n_dims_for_90pct`
5. Ceiling reported alongside: `min(V−1, d)`.
6. PCA coordinates: `Ac @ E[:, :3]` where `E` are the top-3 eigenvectors of `G`. These are
   a **projection** and must be labelled as such — unlike the Geometry Lab's sphere.
   Report the explained-variance ratio of each component.

   **Sign convention (required for parity).** An eigenvector is only defined up to sign,
   so two correct implementations can return mirror-image clouds and every scalar
   statistic will still agree. Fix it: for each eigenvector, if its largest-magnitude
   entry is negative, negate the whole vector. Without this the golden test fails on
   `pca_coords` alone while matching everywhere else.

> **Effective rank is an entropy, not a count.** `min(|V|−1, d)` bounds it, but it reaches
> that bound only when the spectrum is perfectly flat. Measured on untrained models at
> `d=128`, algebraic rank hits the 128 ceiling exactly from the `first` budget onward,
> while effective rank is still only 104.2 at `|V|=314` and still climbing. Its increments
> across the five budgets are +28.18, +14.73, +16.62, +8.42 — strictly rising overall but
> **not** monotonically decelerating, so do not describe the curve's shape beyond "rising"
> without measuring it again. Any UI text must keep the two ranks distinct.
