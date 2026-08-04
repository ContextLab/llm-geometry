# 006 — Exact specification of `tiny-seuss/train/tiny_lm.py`

Source of truth: `/Users/jmanning/Desktop/tiny-models/tiny-seuss/train/tiny_lm.py` (245 lines).
All line numbers below refer to that file. Every architectural claim was **verified by
running the real code** under `torch 2.12.0`
(`/Users/jmanning/llm-geometry/code/backend/.venv/bin/python`), not read off by eye.

**Imports:** the module imports only `argparse, json, math, os, re, sys` and
`torch`, `torch.nn as nn`, `torch.nn.functional as F` (lines 17–26). **There are no
first-party imports** — the file is fully self-contained. Its only consumer inside the
repo is `tiny-seuss/eval/probe.py:31` (`from train.tiny_lm import TinyLM, Vocab, tokenize`).

A near-duplicate exists at `/Users/jmanning/Desktop/tiny-models/TonyModelsTools/tiny_lm.py`.
It differs in exactly three hunks (verified by `diff`): no MPS device branch, no
corpus-length guard, and `log = []` instead of `log = [{"step": 0, **spectra(model)}]`.
**Port the `tiny-seuss` copy**; it is the newer one.

---

## 1. Exact architecture

### 1.1 Tokenizer and vocabulary (needed to fix |V|)

```python
SPECIALS = ["<pad>", "<bos>", "<eos>", "<nl>", "<unk>"]          # line 28
WORD_RE = re.compile(r"[A-Za-z'][A-Za-z'-]*|[.,!?;:]")           # line 29
```

```python
def tokenize(text):                                              # lines 32-40
    out = []
    for line in text.splitlines():
        if not line.strip():
            out.append("<eos>")
            continue
        out.extend(w.lower() for w in WORD_RE.findall(line))
        out.append("<nl>")
    return out
```

Word-level, lowercased. A blank line emits `<eos>`; every non-blank line ends with `<nl>`.

```python
class Vocab:                                                     # lines 43-58
    def __init__(self, tokens, mint=()):
        types = sorted(set(tokens) - set(SPECIALS))
        self.itos = SPECIALS + list(mint) + types
        self.stoi = {s: i for i, s in enumerate(self.itos)}
        self.mint_ids = [self.stoi[m] for m in mint]
```

So id 0=`<pad>`, 1=`<bos>`, 2=`<eos>`, 3=`<nl>`, 4=`<unk>`, then minted tokens, then
the corpus types in `sorted()` (ASCII) order. `encode` maps unknown surface forms to
`<unk>`; `decode` is `" ".join(self.itos[i] for i in ids)` (line 58).

Measured on the shipped corpora:

| corpus | tokens | \|V\| |
|-|-|-|
| `tiny-seuss/data/demo_jabber.txt` | 3663 | 395 |
| `artifacts/fixture/corpus.txt` | 2048 | 35 |

### 1.2 Block

```python
class Block(nn.Module):                                          # lines 61-75
    def __init__(self, d, h, dropout=0.1):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(),
                                 nn.Linear(4 * d, d), nn.Dropout(dropout))

    def forward(self, x, mask):
        h = self.ln1(x)
        a, w = self.attn(h, h, h, attn_mask=mask, need_weights=True,
                         average_attn_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x, w
```

### 1.3 Top level

```python
class TinyLM(nn.Module):                                         # lines 78-92
    def __init__(self, vocab_size, d=128, layers=4, heads=4, ctx=128,
                 dropout=0.1, tie=False):
        super().__init__()
        self.ctx = ctx
        self.tok = nn.Embedding(vocab_size, d)
        self.pos = nn.Embedding(ctx, d)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(d, heads, dropout)
                                     for _ in range(layers)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab_size, bias=False)
        if tie:
            self.head.weight = self.tok.weight
        self.apply(self._init)
```

### 1.4 Answers to the specific questions

| Question | Answer | Evidence |
|-|-|-|
| Layers | `layers=4` default, `nn.ModuleList` of identical `Block`s | line 79, 86-87 |
| d_model | `d=128` default | line 79 |
| n_heads | `heads=4` default | line 79 |
| head_dim | `d // heads` = 32 at defaults; `nn.MultiheadAttention` asserts `embed_dim % num_heads == 0` (verified: `AssertionError: embed_dim must be divisible by num_heads`) | runtime probe: `blk.attn.head_dim == 32` |
| MLP shape | `d -> 4d -> d` | line 66-67 |
| MLP activation | **exact GELU (erf), not tanh** — runtime repr is `GELU(approximate='none')` | line 66 + runtime probe |
| Normalization | `nn.LayerNorm`, `eps=1e-5`, `elementwise_affine=True` (weight+bias), **pre-norm** (ln1 before attn, ln2 before MLP, residual added to the *un*-normalized `x`), plus a **final** `lnf` before the readout | lines 64, 70-74, 88, 111; runtime `ln1.eps == 1e-05` |
| Positional encoding | **learned absolute** `nn.Embedding(ctx, d)`, added to token embedding. No RoPE, no sinusoids, no ALiBi | line 84, 104 |
| Readout | `nn.Linear(d, vocab_size, bias=False)`. **Untied by default**; `--tie` aliases `head.weight = tok.weight` (verified: same `data_ptr()`) | lines 89-91 |
| Bias terms | LayerNorms: yes. Attention `in_proj_bias` (3d) and `out_proj.bias` (d): **yes** (PyTorch default `bias=True`). MLP both `nn.Linear`: **yes**. Readout: **no** | runtime named_parameters dump |
| Dropout | Single `dropout=0.1` used in three places: (a) post-embedding `self.drop`, (b) **attention-weight** dropout inside `nn.MultiheadAttention`, (c) after the second MLP `nn.Linear`. **There is no residual/output dropout on the attention branch** — `x = x + a` is raw | lines 62, 65, 67, 80, 85 |

### 1.5 Initialization — the single most important porting gotcha

```python
    @staticmethod
    def _init(m):                                                # lines 94-99
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
```

`nn.MultiheadAttention.in_proj_weight` is a bare `nn.Parameter`, **not** an `nn.Linear`,
so `_init` **never touches it**. It keeps PyTorch's own default,
`xavier_uniform_(in_proj_weight)`, i.e. U(-b, b) with b = sqrt(6/(3d+d)).
By contrast `out_proj` is a `NonDynamicallyQuantizableLinear`, which **is** a subclass of
`nn.Linear`, so it *does* get `normal_(std=0.02)`.

Measured at d=128 (b = 0.108253…):

```
blocks.0.attn.in_proj_weight       std=0.06269  min=-0.1082 max=+0.1083   <- xavier_uniform
blocks.0.attn.in_proj_bias         std=0.00000  (all zeros)               <- torch default constant_(0)
blocks.0.attn.out_proj.weight      std=0.01990                            <- normal_(0.02)
blocks.0.attn.out_proj.bias        std=0.00000
tok.weight / pos.weight / mlp.*.weight / head.weight   std≈0.0200
all LayerNorm weight = 1.0, bias = 0.0                                    <- untouched by _init
all Linear biases = 0.0
```

So the exact init to reproduce is:
- `tok`, `pos`, `head`, `mlp.0.weight`, `mlp.2.weight`, `attn.out_proj.weight` → N(0, 0.02²)
- `attn.in_proj_weight` → U(±sqrt(6/(4d)))
- all biases → 0
- all LayerNorm γ=1, β=0
- **no** depth-scaled residual init (no 1/sqrt(2L) GPT-2 trick)

`self.apply(self._init)` runs **after** the tie (line 92 after 90-91), so with `--tie`
the shared tensor is re-drawn from N(0, 0.02²) twice — distributionally identical, but
it consumes two draws from the RNG stream, which matters for bit-exact seed matching.

---

## 2. Exact forward pass, in computed order

```python
    def forward(self, idx, targets=None, return_hidden=False):   # lines 101-119
        B, T = idx.shape
        p = torch.arange(T, device=idx.device)
        x = self.drop(self.tok(idx) + self.pos(p))
        mask = torch.triu(torch.full((T, T), float("-inf"), device=idx.device), 1)
        hiddens, attns = [], []
        for blk in self.blocks:
            x, w = blk(x, mask)
            hiddens.append(x)
            attns.append(w)
        x = self.lnf(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.reshape(-1), ignore_index=0)
```

Let `idx ∈ {0..V-1}^{B×T}`, `E ∈ R^{V×d}` (`tok.weight`), `P ∈ R^{ctx×d}` (`pos.weight`),
`H` heads, `d_h = d/H`.

**(0) Causal mask.** `M ∈ R^{T×T}`, `M[i,j] = -inf if j > i else 0`. Rebuilt every forward.

**(1) Embed.**
```
x⁰ = Dropout_p( E[idx] + P[0:T] )          # positions are always 0..T-1, never offset
```

**(2) For ℓ = 1 … L:**
```
h  = LayerNorm(x^{ℓ-1}; γ_ℓ1, β_ℓ1, eps=1e-5)

# packed QKV: W_in ∈ R^{3d×d} = [W_q; W_k; W_v],  b_in ∈ R^{3d} = [b_q; b_k; b_v]
Q  = h W_qᵀ + b_q ;  K = h W_kᵀ + b_k ;  V = h W_vᵀ + b_v
# reshape (B,T,d) -> (B,H,T,d_h), head n takes rows [n·d_h : (n+1)·d_h] of each of W_q/W_k/W_v

A_n     = softmax_j( Q_n K_nᵀ / sqrt(d_h) + M )        for each head n
A'_n    = Dropout_p(A_n)                                # train mode only; THIS is what is returned
O       = concat_n( A'_n V_n ) W_oᵀ + b_o

x^{ℓ-½} = x^{ℓ-1} + O                                   # <-- no dropout on this branch

g  = LayerNorm(x^{ℓ-½}; γ_ℓ2, β_ℓ2)
x^{ℓ} = x^{ℓ-½} + Dropout_p( GELU_exact(g W_1ᵀ + b_1) W_2ᵀ + b_2 )

hiddens[ℓ] = x^{ℓ}        # POST-block, i.e. after the MLP residual
attns[ℓ]   = A'           # shape (B, H, T, T) because average_attn_weights=False
```
`GELU_exact(z) = z·Φ(z) = 0.5 z (1 + erf(z/√2))`.

**(3) Final norm + readout.**
```
z      = LayerNorm(x^L; γ_f, β_f)
logits = z W_headᵀ                                      # (B, T, V), no bias
```

**(4) Loss (only when `targets` given).**
```
loss = mean over {(b,t) : targets[b,t] ≠ 0} of  -log softmax(logits[b,t])[targets[b,t]]
```
`ignore_index=0` = `<pad>`.

Verified shapes at `V=320, d=128, L=4, H=4, B=2, T=7`:
`logits (2,7,320)`, `hiddens` length 4 each `(2,7,128)`, `attns[0] (2,4,7,7)`.
Causality verified: `att[0][0,0,0,1:].abs().max() == 0.0`.

### 2.1 Two traps for a faithful port

1. **The returned attention weights are post-dropout.** Measured row sums in train mode:
   `[1.1111, 1.1111, 0.81, 1.1111, 0.7864, 0.9446]`; in eval mode: all `1.0`.
   If you visualize `attns` from a training-mode forward you are showing a
   dropout-corrupted, non-stochastic-matrix version of attention.
2. **`need_weights=True` disables PyTorch's fused SDPA path.** Numerics/perf differ from
   a `scaled_dot_product_attention` implementation; a TS port matching the math is fine,
   but do not expect bit-identity with a fused-kernel PyTorch rewrite.

### 2.2 Generation

```python
    @torch.no_grad()
    def generate(self, idx, n=64, temperature=1.0, top_k=None, allow=None):   # 121-136
        for _ in range(n):
            logits, _ = self(idx[:, -self.ctx:])
            logits = logits[:, -1, :] / max(1e-6, temperature)
            if allow is not None:
                m = torch.full_like(logits, float("-inf"))
                m[:, allow] = 0.0
                logits = logits + m
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            nxt = torch.multinomial(F.softmax(logits, -1), 1)
            idx = torch.cat([idx, nxt], 1)
        return idx
```

Order is: crop to last `ctx` → logits of last position → divide by `max(1e-6, temperature)`
→ additive `allow` budget mask → top-k truncation → softmax → multinomial. Always samples;
there is no greedy branch.

---

## 3. Training

```python
    torch.manual_seed(args.seed)                                            # 173
    dev = ("cuda" if torch.cuda.is_available() else
           "mps" if torch.backends.mps.is_available() else "cpu")           # 174-175
    ...
    toks = tokenize(open(args.data).read())                                 # 178
    mint = [m for m in args.mint.split(",") if m]                           # 179
    vocab = Vocab(toks, mint)                                               # 180
    data = torch.tensor(vocab.encode(toks), dtype=torch.long)               # 181
    n = int(0.95 * len(data))                                               # 182
    tr, va = data[:n], data[n:]                                             # 183
```

- **Split:** a single contiguous 95 / 5 prefix/suffix split of the token stream. No shuffle.
- **Optimizer** (line 195):
  `torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)`.
  Betas/eps are PyTorch defaults (0.9, 0.999), eps 1e-8, `amsgrad=False`.
  **Weight decay 0.01 is applied to every parameter**, including LayerNorm γ/β, all biases,
  and both embedding matrices. There is no no-decay parameter group.
- **Schedule** (lines 196-197):
  `torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps)`.
  All other args are torch defaults: `pct_start=0.3`, `anneal_strategy='cos'`,
  `div_factor=25`, `final_div_factor=1e4`, `cycle_momentum=True`,
  `base_momentum=0.85`, `max_momentum=0.95`. Because AdamW exposes `betas` and not
  `momentum`, **OneCycleLR cycles `beta1`**.

  I reproduced the schedule in closed form to **max abs error 0.0** over all 3000 steps
  (`max_lr=3e-4, total_steps=3000`). With
  `anneal(s,e,pct) = e + (s-e)/2·(cos(π·pct)+1)`, `init = max_lr/25`,
  `min = init/1e4`, `k₁ = floor(0.3·S) − 1`:

  ```
  k ≤ k₁ :  lr(k)    = anneal(init,   max_lr, k/k₁)          beta1(k) = anneal(0.95, 0.85, k/k₁)
  k > k₁ :  lr(k)    = anneal(max_lr, min,    (k−k₁)/(S−1−k₁)) beta1(k) = anneal(0.85, 0.95, (k−k₁)/(S−1−k₁))
  ```
  At defaults: `init = 1.2e-5`, peak `3e-4` at index 899, final `1.2e-9`.
  Note the consequence: **the first training step runs at 1.2e-5 and the last at ~1.2e-9.**

- **Batch / context construction** (lines 199-204):
  ```python
    def batch(split):
        d = tr if split == "train" else va
        ix = torch.randint(len(d) - args.ctx - 1, (args.batch,))
        x = torch.stack([d[i:i + args.ctx] for i in ix]).to(dev)
        y = torch.stack([d[i + 1:i + 1 + args.ctx] for i in ix]).to(dev)
        return x, y
  ```
  i.i.d. uniform start offsets **with replacement**, no epochs, no permutation, no
  document boundaries. `x` and `y` are the classic shift-by-one pair, both length `ctx`.
  Every position contributes to the loss.

- **Loss:** token-level mean cross-entropy over the whole `B×ctx` block, `ignore_index=0`
  (line 115-116). Since `<pad>` is never emitted by `tokenize`, `ignore_index` is a no-op
  in practice.

- **Step body** (lines 210-218):
  ```python
    for step in range(1, args.steps + 1):
        model.train()
        x, y = batch("train")
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
  ```
  Global grad-norm clip at **1.0**. No grad accumulation, no AMP, no `torch.compile`.

- **Eval** (lines 219-230): every 100 steps and at step 1, mean of **10 freshly sampled
  random validation batches** (so val loss is stochastic, not a fixed held-out set), plus
  `ppl = math.exp(min(20, vl))` and the `spectra` probe; `log.json` is rewritten in full
  each time.

- **Seeding / determinism** (line 173): `torch.manual_seed(args.seed)` and nothing else.
  No `torch.use_deterministic_algorithms`, no cuDNN flags, no separate data generator.
  The model init, the dropout masks, the batch offsets, and the final sampling all draw
  from one global stream, so **any change to eval cadence or to the tie flag shifts the
  whole downstream stream.** Runs are not reproducible across devices (CPU vs MPS vs CUDA).

- **Geometry probe** (lines 139-153): every eval step, for `model.tok.weight` and
  `model.head.weight` (both `(V,d)`), subtract the column-wise mean over the vocabulary
  (`A - A.mean(0, keepdim=True)`), take `svdvals`, set `p = s²/Σs²`, and log
  `stable_rank = Σs²/max s²`, `participation = 1/Σp²`, `erank = exp(-Σ p log p)`,
  `top5_frac = Σ p[:5]`.

---

## 4. CLI args

| Flag | Type | Default | Notes |
|-|-|-|-|
| `--data` | str | **required** | corpus path, read with `open(args.data).read()` |
| `--out` | str | `"runs/tiny"` | `os.makedirs(..., exist_ok=True)` |
| `--d` | int | `128` | d_model |
| `--layers` | int | `4` | |
| `--heads` | int | `4` | must divide `--d` |
| `--ctx` | int | `128` | also the size of the positional table |
| `--batch` | int | `32` | |
| `--steps` | int | `3000` | also `OneCycleLR.total_steps` |
| `--lr` | float | `3e-4` | this is OneCycle's **max_lr**, not the starting lr |
| `--tie` | `store_true` | `False` | |
| `--mint` | str | `""` | comma-separated; help text: `"comma-separated minted tokens with no training data"` |
| `--seed` | int | `0` | |

**Not exposed:** dropout (hard-coded 0.1 at lines 62/80; `TinyLM` is constructed
positionally at 191-192 and never receives it), weight decay (0.01, line 195), grad-clip
norm (1.0, line 216), eval cadence (100, line 219), the 95/5 split (line 182),
`temperature`/`top_k` for the end-of-run sample (0.9 / 40, line 237), and device.

---

## 5. Checkpoint format

```python
    torch.save({"model": model.state_dict(), "itos": vocab.itos,
                "args": vars(args)}, os.path.join(args.out, "ckpt.pt"))     # 232-233
```

Written **once, at the very end of training.** Verified by loading a real run:

```
top-level keys: ['model', 'itos', 'args']
args: {'data': ..., 'out': ..., 'd': 32, 'layers': 2, 'heads': 2, 'ctx': 32,
       'batch': 8, 'steps': 200, 'lr': 0.0003, 'tie': False, 'mint': '', 'seed': 0}
itos: list[str], len == V, ['<pad>','<bos>','<eos>','<nl>','<unk>', ...]
model: 29 float32 entries for L=2, keys:
  tok.weight (V,d), pos.weight (ctx,d),
  blocks.{i}.ln1.{weight,bias}, blocks.{i}.ln2.{weight,bias},
  blocks.{i}.attn.in_proj_weight (3d,d), blocks.{i}.attn.in_proj_bias (3d,),
  blocks.{i}.attn.out_proj.weight (d,d), blocks.{i}.attn.out_proj.bias (d,),
  blocks.{i}.mlp.0.weight (4d,d), blocks.{i}.mlp.0.bias (4d,),
  blocks.{i}.mlp.2.weight (d,4d), blocks.{i}.mlp.2.bias (d,),
  lnf.weight, lnf.bias, head.weight (V,d)
```
State-dict entry count = `2 + 14·L + 3`. **No optimizer state, no scheduler state, no RNG
state, no step counter — training cannot be resumed.** With `--tie`, `head.weight` is
still present in the dict and byte-equal to `tok.weight` (verified).

Two sibling artifacts are also written to `--out`:
- `log.json` — a JSON list, rewritten at every eval. First element is
  `{"step": 0, <8 spectra keys>}`; every later element is
  `{"step", "train", "val", "ppl", <8 spectra keys>}`. The 8 keys are
  `{embed,unembed}_{stable_rank,participation,erank,top5_frac}`.
- `sample.txt` — 120 sampled tokens, `.replace(" <nl>", "\n").replace(" <eos>", "\n\n")`
  (lines 236-239).

---

## 6. Broken / half-implemented / would-not-run

**Severity ordered. Everything below was reproduced, not inferred, unless flagged.**

1. **HARD CRASH on Apple Silicon — lines 174-175 + 145.** The device probe selects `mps`,
   but `torch.linalg.svdvals` has no MPS kernel, and `spectra` is called at line 209
   *before the first training step*. Real run output:
   ```
   |V|=395  tokens=3663  minted=[]
   params=0.052M
   Traceback ... line 209, in main
       log = [{"step": 0, **spectra(model)}]
     File ".../tiny_lm.py", line 145, in spectra
       s = torch.linalg.svdvals(A)
   NotImplementedError: The operator 'aten::_linalg_svd.U' is not currently implemented
   for the MPS device.
   ```
   The script is **unrunnable as written on any Mac** unless
   `PYTORCH_ENABLE_MPS_FALLBACK=1` is set. (The `TonyModelsTools` copy has no MPS branch
   and does not hit this.) Fix for a port: compute `spectra` on a CPU copy of the weights.

2. **`--mint` does not do what its help text says — lines 46-48.** If a minted token also
   occurs in the corpus, `itos` gets **two** entries for it and `stoi` keeps the *later*
   (corpus) index, so the minted slot becomes a dead, never-encoded embedding row and
   `mint_ids` points at the trained corpus row. Reproduced:
   ```
   Vocab(tokenize("the cat sat\nthe dog ran\n"), mint=["cat","zzz"])
   itos = ['<pad>','<bos>','<eos>','<nl>','<unk>','cat','zzz','cat','dog','ran','sat','the']
   stoi['cat'] = 7 ;  mint_ids = [7, 6] ;  encode(['cat']) = [7]
   ```
   The whole point of a minted token — "no training data" — is silently violated. There is
   no collision check anywhere.

3. **`Vocab.mint_ids` is computed and never used.** `grep -rn "mint_ids"` over the entire
   repo returns exactly one hit: its own assignment at line 48. The minted tokens
   therefore receive no special treatment during training at all.

4. **`eval/probe.py:38` silently drops weight tying.**
   `m = TinyLM(len(itos), a["d"], a["layers"], a["heads"], a["ctx"])` — `tie` is in
   `ck["args"]` but never passed. Reproduced with a real `--tie` checkpoint:
   `args tie = True -> probe reconstructs tied? False`. `load_state_dict` succeeds (both
   keys exist and are equal), so it fails *quietly*: the reloaded model has two independent
   matrices, and any further training or in-place edit desynchronizes them.

5. **`<bos>` (id 1) is dead.** `tokenize` (lines 32-40) never emits it and nothing else
   inserts it. Line 236 reads
   `bos = torch.tensor([[vocab.stoi["<eos>"]]], device=dev)` — the variable is named `bos`
   but the seed token is `<eos>`. So the model has an embedding row it never sees as input.

6. **`ignore_index=0` (line 116) is a no-op.** Id 0 is `<pad>`, which `tokenize` never
   produces and which the contiguous-slice batcher never inserts.

7. **`import sys` (line 22) is unused.** `grep -n "sys\."` returns nothing.

8. **With `--tie`, the `spectra` log silently duplicates itself.** `model.tok.weight` and
   `model.head.weight` are the same tensor, so `embed_*` and `unembed_*` are byte-identical
   (verified). Four of the eight logged numbers are then meaningless — and `unembed_erank`
   would be read as independent evidence in exactly the rank-staircase experiment this file
   exists to serve.

9. **Off-by-one in the batcher — line 201.** `torch.randint(len(d) - ctx - 1, ...)` gives
   max start `len(d)-ctx-2`, so the final token of each split is **never used as a target**.
   Harmless, but a port must copy it to match sample streams.

10. **Validation is not a fixed set — line 222.** `sum(model(*batch("val"))[1].item()
    for _ in range(10)) / 10` redraws 10 random windows every eval, so `val` and `ppl`
    carry sampling noise and are not comparable point-to-point across steps.

11. **Unclosed file handles / locale-dependent decoding — lines 178, 229, 238.**
    `open(args.data).read()`, `json.dump(log, open(..., "w"), indent=1)`, and
    `open(...).write(...)` all leak handles and use the platform default encoding. On a
    non-UTF-8 locale, line 178 mis-decodes the corpus.

12. **No checkpointing during training, only at line 232.** Combined with (10) and the
    `log.json`-only trajectory, an interrupted run loses everything. `docs/LLM_GEOMETRY_
    INTEGRATION.md` asks for "exact singular-value/effective-rank trajectories" and
    "optional checkpoints" — the per-step weights needed for a step scrubber are not saved.

13. **`--steps` is load-bearing for the scheduler, and OneCycleLR is exactly exhausted.**
    Verified: 3000 `sched.step()` calls succeed; a 3001st raises
    `ValueError: Tried to step 3001 times. The specified number of total steps is 3000`.
    So any port that adds a warmup step, a resume, or an extra eval step will crash.
    `--steps 0` gives an empty loop and an invalid `total_steps`.

14. **Weight decay 0.01 on LayerNorm γ/β, all biases, and both embeddings** (line 195).
    Almost certainly unintended; it is not the GPT-2/nanoGPT convention. Flagging it
    because a port that "does the sensible thing" will not reproduce these curves.

15. **Docstring example references a file that does not exist.** Line 14:
    `python train/tiny_lm.py --data data/dolch315.txt ...`. `tiny-seuss/data/` contains
    only `demo_jabber.txt`. Same for `tiny-seuss/README.md:168-169`.

16. **Requirements disagree on the torch pin.** `tiny-seuss/requirements.txt` says
    `torch>=2.0`; `requirements/train.txt` says `torch==2.8.0`. I verified everything above
    on 2.12.0. Since the init scheme depends on `nn.MultiheadAttention` internals
    (finding 1.5), this pin is not cosmetic.

17. **`generate` edge cases — lines 122-136.** `allow=torch.tensor([])` makes every logit
    `-inf`, `F.softmax` returns NaN, and `torch.multinomial` raises. `temperature <= 0` is
    clamped to `1e-6` rather than rejected or switched to greedy. Neither is guarded.
    *(Reasoned from the code, not executed.)*

**Things I checked and found NOT broken:** the corpus-length guard at lines 184-188 fires
correctly (`ValueError: corpus too short for ctx=128: train=1945, val=103`); the causal
mask is correct; `svdvals` returns descending singular values so `p[:5]` really is the top
5; `probe.load` works under torch ≥2.6's `weights_only=True` default because the payload is
only tensors, `str`, and plain scalars; a full 200-step run completes and writes all three
artifacts.

**I don't know:** whether the intended `dolch315.txt` corpus yields exactly |V| = 320 — the
file is not in the repo, so I could not tokenize it. The docstring says "With |V| ~ 320",
and 315 Dolch words + 5 specials = 320 only if every budget word occurs in the corpus.

---

## 7. Parameter count

Exact closed form (verified against `sum(p.numel())` for 7 configurations, all exact):

```
N = (2 if untied else 1)·V·d          # tok + head
  + ctx·d                             # learned positions
  + L·(12·d² + 13·d)                  # per block, see breakdown
  + 2·d                               # lnf γ,β

per block = 4d      (ln1 γβ + ln2 γβ)
          + 4d² + 4d   (in_proj 3d² + 3d, out_proj d² + d)
          + 8d² + 5d   (mlp 4d²+4d, then 4d²+d)
          = 12d² + 13d
```

| V | d | L | H | ctx | tie | params |
|-|-|-|-|-|-|-|
| 320 | 128 | 4 | 4 | 128 | no | **891,648** (0.892 M) ← *defaults, docstring's \|V\|≈320* |
| 320 | 128 | 4 | 4 | 128 | yes | 850,688 |
| 400 | 128 | 4 | 4 | 128 | no | 912,128 ← *defaults on the only shipped corpus, `demo_jabber.txt`* |

> Corrected 2026-08-03: this row originally read `|V| = 395`, which is the corpus's
> distinct **word type** count. The model's embedding has 400 **rows** (395 types plus
> the specials), and 400 is what produces 912,128; the formula gives 910,848 at 395.
> Caught while implementing feature 006 — the formula was right, the label was not.
> Everywhere in feature 006 the two are named separately (`budget_size` vs `rows`).
| 320 | 64 | 2 | 2 | 64 | no | 145,152 |
| 45 | 16 | 1 | 1 | 32 | no | 5,264 |
| 45 | 16 | 1 | 1 | 32 | yes | **4,544** ← *smallest sensible: mask50 budget, 1 layer, 1 head* |
| 320 | 3 | 1 | 1 | 32 | yes | 1,209 ← *llm-geometry's d=3 shape, for scale comparison* |

The trainer's own printout confirms the top row shape:
`print(f"params={nparam/1e6:.3f}M")` (line 194) reported `params=0.052M` for the
`V=395, d=32, L=2, H=2, ctx=32` smoke run.

**Constraints on "smallest sensible":** `d % heads == 0` is enforced by
`nn.MultiheadAttention` (`AssertionError: embed_dim must be divisible by num_heads`), and
lines 184-188 require `min(len(train), len(val)) > ctx + 1`, i.e. with a 95/5 split the
corpus needs roughly `20·(ctx+2)` tokens. At `ctx=32` that is ~680 tokens.

At the default `d=128`, **the four blocks are 793,088 of the 891,648 parameters (89%)**;
the embedding+readout pair is only 81,920. Note the methodological consequence flagged in
`AUDIT.md`: "Effective rank is bounded by `min(|V|-1, d)`" — with `d=128` and `|V|=320`
the rank ceiling is 128, so the embedding spectrum saturates for reasons of shape, not
learning.

---

## Port checklist (the things most likely to be gotten wrong)

1. `attn.in_proj_weight` is **xavier_uniform**, everything else is **N(0, 0.02²)**.
2. Pre-norm, with a **final** LayerNorm, LayerNorm eps **1e-5**, affine.
3. **Exact erf GELU**, not tanh.
4. **No dropout on the attention residual branch**; dropout is on the *attention weights*,
   on the embedding sum, and after the second MLP linear.
5. QKV packed as one `(3d, d)` matrix in `[q; k; v]` row order, **with bias**; readout has
   **no** bias.
6. `--lr` is OneCycle's `max_lr`; the actual step-1 lr is `max_lr/25`.
7. AdamW cycles **beta1** between 0.95 and 0.85 under OneCycleLR (`cycle_momentum=True`).
8. Weight decay hits **every** parameter.
9. Batch offsets are i.i.d. uniform on `[0, len(split) - ctx - 1)` — with replacement.
10. Returned attention weights are **post-dropout** in train mode.
