# Source review: `tiny-seuss/eval/` measurement tools

Scope: `/Users/jmanning/Desktop/tiny-models/tiny-seuss/eval/{probe,fingerprint,mask_decode}.py`
plus their dependencies `synth/lexicon.py`, `wordlists/dolch.py`, `train/tiny_lm.py`.

Every number below was **reproduced by running the real code** (Python 3, real PyTorch 2.13,
real GPT-2 tokenizer from HF, real corpus `data/demo_jabber.txt`). No mocks. Where I did not
verify something, it is marked "I don't know".

Line numbers are as of the files read on 2026-08-03.

---

## 0. Executive verdict

| tool | correct as written? | browser-reusable? |
|-|-|-|
| `probe.py` — `spectrum()` | **Yes**, math is sound | **Yes — best candidate.** ~2 ms for 320×64 |
| `probe.py` — `hidden_geometry()` | Yes, one undocumented choice | Yes, if you already run the forward pass |
| `probe.py` — `minted_field()` | **No** — mislabelled quantity | Yes, sub-ms, but fix semantics first |
| `probe.py` — probes (B) and (C) | **Not implemented at all** | n/a |
| `fingerprint.py` — token stats | Yes (one approximation) | **Yes**, O(N), instant |
| `fingerprint.py` — `meter_*` | **No — broken.** Measures nothing | Cheap but must not ship as-is |
| `fingerprint.py` — `rhyme_rate` | **Heuristic**, verified errors both ways | Cheap; needs a hand table |
| `mask_decode.py` — `BudgetTrie` | **No — audit confirmed** | Trie is cheap; loop needs a full LM |

---

## 1. `probe.py`

### 1.1 What it measures

Module docstring (`probe.py:1-21`) promises three probes:

> ```
> (A) RANK. Full spectrum of E and U, stable rank, participation ratio, effective
>     rank. With |V| ~ 320 this is exact and free.
> ```
> ```
> (B) FIELD GROWTH for a minted token. ... We track, per checkpoint: distance from
>     initialization, rank of the span of its contextual occurrences, and the KL
>     between its next-token distribution and the corpus unigram.
> ```
> ```
> (C) VACANCY RESPONSE for jabberwockified corpora. ... Train across p and read
>     off how much of the loss is carried by closed-class scaffolding alone.
> ```

**Only (A) is implemented.** For (B), only the KL is computed — there is no
"distance from initialization" (the initialization is never loaded or stored), no
"rank of the span of its contextual occurrences", and no per-checkpoint sweep
(`train/tiny_lm.py:232` writes exactly one `ckpt.pt`, at the end). For (C), nothing:
`grep -rn "closed.class\|function word\|per_class\|class_loss\|scaffold" eval/ train/`
returns exactly one hit — the docstring line itself (`probe.py:18`). There is no
per-token-class loss computation anywhere in the repo.

### 1.2 Exact formulas — `spectrum(W, center=True)` (`probe.py:45-59`)

```python
A = W.detach().float()
if center:
    A = A - A.mean(0, keepdim=True)
s = torch.linalg.svdvals(A)
p = (s ** 2) / (s ** 2).sum()
```

So with singular values σ₁ ≥ … ≥ σ_r of the **column-mean-centred** matrix, and
p_i = σ_i² / Σ_j σ_j² (i.e. fraction of total variance/energy in direction i):

| output key | formula | line |
|-|-|-|
| `singular_values` | σ (full list; truncated to 32 in `main`, `probe.py:131`) | 52 |
| `stable_rank` | Σσ² / max σ² = ‖A‖²_F / ‖A‖²₂ | 53 |
| `participation_ratio` | 1 / Σ p_i² (inverse participation ratio) | 54 |
| **`effective_rank`** | **exp(−Σ p_i ln p_i)** — exponential of the Shannon entropy (in **nats**) of the normalized **squared** singular values | 55 |
| `frac_var_top2` | p₁+p₂ | 56 |
| `frac_var_top10` | Σ_{i≤10} p_i | 57 |
| `n_dims_for_90pct` | `int((cumsum(p) < 0.90).sum()) + 1` — smallest k with Σ_{i≤k} p_i ≥ 0.90 | 58 |

Verified numerically on a real `TinyLM(V=320, d=64)`:
`stable_rank 31.20, participation_ratio 53.15, effective_rank 57.80,
frac_var_top2 0.0633, frac_var_top10 0.2749, n_dims_for_90pct 50`.
On a rank-1 matrix all four rank measures return exactly `1.0` and
`n_dims_for_90pct = 1` — the edge case is handled correctly.

**Definitional caveat.** The code uses **squared** singular values for p. Some of the
literature defines "effective rank" using p_k = σ_k / Σσ_i (unsquared). The two differ.
I did not fetch the source paper, so I cannot state which convention the author intended
— **I don't know**. If this is ported, state the convention explicitly in the UI.

**Centering caveat (correctly acknowledged upstream).** `A.mean(0)` averages over the
**vocabulary** rows, so one rank is spent on the mean; max attainable rank is
min(|V|−1, d). `train/tiny_lm.py:206-208` says so:

> "Step zero is the essential random-initialization null. Rank is bounded by
> min(|V|-1, d); never interpret saturation without this baseline and width controls."

That is the right guardrail and `probe.py` inherits it — but `probe.py` itself never
emits the step-0 null, so a user running only `probe.py` gets no baseline.

**Duplicate implementation.** `train/tiny_lm.py:139-153` (`spectra()`) computes the same
four quantities with different key names (`embed_erank`, `embed_participation`,
`embed_stable_rank`, `embed_top5_frac`). Two copies of the same math that can drift.

### 1.3 `hidden_geometry()` (`probe.py:62-76`)

Non-overlapping windows of `ctx` tokens (up to `max_chunks=64`), one forward pass with
`return_hidden=True`, flatten `hs[layer]` to (B·T, d), then `spectrum()`; returns every
key except `singular_values`, prefixed `residual_`.

Undocumented choice: `hiddens` are appended **inside** the block loop
(`train/tiny_lm.py:107-110`), so `hs[-1]` is the last block's output **before**
`self.lnf` (`train/tiny_lm.py:111`). The residual-stream spectrum is therefore
pre-final-LayerNorm. Defensible, but it should be stated — LayerNorm materially
changes the spectrum.

Verified: 40 chunks × ctx 64 on a 2-layer d=64 model → 9 ms total (forward + SVD).

### 1.4 `minted_field()` (`probe.py:79-111`) — **mislabelled**

```python
E = model.tok.weight.detach()
U = model.head.weight.detach()
...
e = E[i]
logits = U @ e
pd = F.softmax(logits, -1)
```

The docstring (`probe.py:82-83`) calls this the token's

> "readout row's induced next-token distribution"

and the module docstring (`probe.py:13-14`) calls it

> "the KL between its next-token distribution and the corpus unigram"

**It is neither.** `U @ e` pushes the *raw input embedding* straight through the
*unembedding*, skipping every transformer block, the positional embedding, and the final
LayerNorm `lnf`. It is a layer-0 logit-lens reading **without** LayerNorm. It is not the
model's next-token distribution for that token in any context. Any claim of "field
growth" resting on `kl_from_unigram` is measuring drift in E·Uᵀ, not in the model's
predictive distribution.

Other outputs: `embed_norm` = ‖e‖₂; `cos_to_mean_embed` = cos(e, mean of E);
`readout_entropy_bits` = −Σ p log₂ p (correctly in bits, `probe.py:98`);
`kl_from_unigram` = Σ p (ln p − ln q) = KL(readout ‖ unigram), in **nats** while the
entropy beside it is in **bits** — mixed units in the same record;
`nearest` = top-8 cosine neighbours in E with self suppressed via `sims[i] = -2`.

### 1.5 Concrete defects in `probe.py`

| # | line | defect |
|-|-|-|
| P1 | 38 | `TinyLM(len(itos), a["d"], a["layers"], a["heads"], a["ctx"])` **drops `tie`**. Verified: a `tie=True` checkpoint's `state_dict` contains both `tok.weight` and `head.weight` (equal tensors); loading into an untied model reports "All keys matched successfully" and yields two *independent but numerically identical* matrices. `res["embed"]` and `res["unembed"]` then come out **identical**, silently — for a script whose headline output is "compare the E and U spectra". |
| P2 | 91 | `logits = U @ e` is not a next-token distribution (§1.4). |
| P3 | 135 | `v = Vocab(toks)` is dead code — `v` is never referenced again. |
| P4 | 13-18 | Docstring promises (B) distance-from-init, span rank, per-checkpoint tracking, and all of (C). None exist. |
| P5 | 57 | `frac_var_top10` silently returns `1.0` whenever d < 10 (verified: d=4 → 1.0). Same for `frac_var_top2` when d < 2. |
| P6 | 98 vs 102 | entropy in bits, KL in nats, in the same dict. |
| P7 | — | No step-0 / random-init null is emitted, though the training script insists one is required. |

### 1.6 Fixes needed to trust `probe.py`

1. Pass `tie=a.get("tie", False)` at line 38, or assert `not a["tie"]` before reporting
   `unembed` separately.
2. Either rename `minted_field`'s outputs to what they are (`embed_logitlens_*`) or run a
   real forward pass with the token in context and read the actual next-token logits.
3. Emit the step-0 null alongside the trained numbers, or read `runs/*/log.json` (which
   `train/tiny_lm.py:209` already writes with a `step: 0` record) and report the delta.
4. Delete line 135; implement or delete probes (B)-remainder and (C).
5. Guard `frac_var_top{2,10}` when d is small; unify units.

---

## 2. `fingerprint.py`

### 2.1 The audit numbers reproduce exactly

Running `fingerprint(open('data/demo_jabber.txt').read())` gave, verbatim:

```
"n_lines": 400,
"n_tokens": 3214,
"ttr": 0.12134411947728686,
"zipf_slope": -0.16719125426313466,
"meter_anapest": 0.34639262126762127,
"rhyme_rate": 0.01,
```

matching the audit's "400 lines, 3214 tokens, TTR 0.12134, Zipf slope -0.16719,
anapest score 0.34639, rhyme rate 0.01" digit-for-digit. Also produced:
`n_types 390, heaps_beta 0.2647, unigram_entropy_bits 8.3662,
bigram_entropy_bits 11.6018, cond_entropy_bits 3.2355, mean_syllables 1.2113,
monosyllable_frac 0.8062, mean_line_len 8.035, line_len_sd 0.1969,
meter_iamb 0.4668, refrain_rate 0.0, anaphora_rate 0.01,
end_word_concentration 0.1125, hapax_frac 0.1103`.

This corroborates `README.md:145-146`:
> "The tracked demo independently reproduces a Zipf slope near -0.16 and rhyme rate
> 0.01, but its TTR is 0.121, not 0.098."

### 2.2 Metric-by-metric: exact or heuristic

Tokenizer for all of these: `WORD_RE = re.compile(r"[A-Za-z']+(?:-[A-Za-z']+)*")`
(`synth/lexicon.py:21`), lowercased (`fingerprint.py:89`). Punctuation is dropped.

| metric | formula | exact? |
|-|-|-|
| `n_lines` | count of non-blank lines (`fingerprint.py:88`) | exact |
| `n_tokens`, `n_types`, `ttr` | \|toks\|, \|set\|, ratio | **exact but length-dependent** — see §2.4 |
| `heaps_beta` | OLS slope of log V vs log N at 12 evenly spaced prefixes (`fingerprint.py:32-52`) | exact OLS; depends on `points=12` |
| `zipf_slope` | OLS slope of log(freq) vs log(rank) over the **top 100** types, ranks 1…N (`fingerprint.py:55-64`) | exact OLS; depends on `top=100`; ties untreated |
| `unigram_entropy_bits` | −Σ(c/n)log₂(c/n) over types | exact |
| `bigram_entropy_bits` | same over `Counter(zip(toks, toks[1:]))` | exact — but bigrams **cross line and stanza boundaries**, since `toks` comes from `WORD_RE.findall(text)` over the whole text |
| `cond_entropy_bits` | `entropy(bi) - entropy(uni)` (`fingerprint.py:107`) | **approximate.** H(X,Y) − H(X) = H(Y\|X) only if the marginal is over the *first* elements. Here `uni` covers all N tokens while `bi` has N−1 pairs, so it is off by boundary terms |
| `mean_syllables`, `monosyllable_frac` | `len(stress(t))` | **heuristic for 95% of tokens** — see §2.3 |
| `mean_line_len`, `line_len_sd` | population SD (÷N), recomputed in-loop (O(n²), irrelevant at this size) | exact |
| **`meter_anapest`, `meter_iamb`** | mean over lines of `Budget.meter_score` | **broken** — see §2.3 |
| **`rhyme_rate`** | mean over `(0,1),(2,3),…` line pairs of `couplet_rhyme` | **heuristic, with verified errors both ways** — see §2.3 |
| `refrain_rate`, `anaphora_rate` | Σ of counts of items occurring >1, ÷ n_lines (`fingerprint.py:78,80`) | exact but odd: counts **all** occurrences, not c−1, so a line appearing twice contributes 2 |
| `end_word_concentration` | share of lines whose last **whitespace** token is in the top 10 (`fingerprint.py:81-82`) | exact but **inconsistent**: `l.split()[-1]` keeps trailing punctuation, so `eat.` ≠ `eat`, unlike `rhyme_rate` which uses `WORD_RE` |
| `hapax_frac` | types with count 1 ÷ n_types | exact |

### 2.3 The meter score is broken — and I can prove it

`synth/lexicon.py:5-8` claims:

> "315 words is small enough to hand-verify in an afternoon, so meter is not
> estimated, it is known. Everything downstream (rejection sampling, meter scoring,
> the anapest detector) is therefore exact rather than heuristic"

**This is false as implemented.** `stress()` (`synth/lexicon.py:71-80`) ends with:

```python
n = _rule_syllables(w)
return "1" if n == 1 else "1" + "0" * (n - 1)
```

Every fallback pattern **begins with `1`**, and every monosyllable is unconditionally
`"1"` — i.e. **stressed**. So *the*, *a*, *to*, *of*, *and*, *is* all scan as stressed.

Measured on `data/demo_jabber.txt`:
- hand-table (`STRESS_TABLE`, 62 entries) covers **161 / 3214 tokens = 5.0%**. The other 95% use the spelling heuristic.
- the concatenated scan string has character counts **exactly `{'1': 3214, '0': 679}`** against 3214 tokens — i.e. **precisely one `1` per token**, confirming the structural claim.
- 82.6% of all scan positions are `1`.

`meter_score` (`synth/lexicon.py:127-139`) tiles the template and counts positional
agreement:

```python
target = (pat * (len(s) // len(pat) + 1))[: len(s)]
return sum(a == b for a, b in zip(s, target)) / len(s)
```

With a near-all-`1` scan string, agreement with `"001"` converges to the **density of
`1` in the template = 1/3**, and with `"01"` to **1/2**. Observed: `meter_anapest
0.3464 ≈ 1/3`, `meter_iamb 0.4668 ≈ 1/2`. The reported anapest score is a
**positional coincidence, not a measurement of meter.**

Decisive true-positive test — textbook anapests, every word in-budget:

```
0.333  111111111     | and I do not like green eggs and ham
0.333  111111111111  | in the house with a mouse in the box with a fox
0.250  11111111      | i will not eat them here or there
```

A perfectly anapestic line scores **0.333 — identical to the nonsense corpus's 0.346**.
The instrument cannot distinguish anapestic verse from random monosyllables. Anything
built on `meter_anapest` (including the rejection sampler in `synth/generate.py`, which
I did not audit in depth) is affected.

**Rhyme is heuristic, with errors in both directions.** `Budget.rhyme_key`
(`synth/lexicon.py:169-183`) consults a 60-entry hand table then applies 21 ordered
spelling rewrites. Verified live:

| pair | keys | verdict |
|-|-|-|
| here / there | `er_` / `er_` | correct |
| **were / here** | `er_` / `er_` | **false positive** |
| eat / street | `Et` / `Et` | correct |
| cat / hat | `at` / `at` | correct |
| go / snow, do / you, tree / key | via hand table | correct |
| **one / done** | `e` / `on_` | **false negative** |

The source comment is honest about this (`synth/lexicon.py:141-144`): "Orthographic
rimes are unreliable ... For a 315-word budget you can replace this entirely with a hand
table; this is the version that degrades gracefully". The *module* docstring's "exact
rather than heuristic" claim is the one that is wrong.

**Couplet pairing bug.** `fingerprint.py:96-97` pairs `lines[i], lines[i+1]` for
`i in range(0, len-1, 2)` — but `lines` has already had blank lines stripped
(`fingerprint.py:88`). Any stanza with an **odd** number of lines shifts every
subsequent pairing by one, so an AABB corpus is read as ABBA from that point on.
`rhyme_rate 0.01` on the demo cannot distinguish "no rhymes" from "pairing desynced".

### 2.4 The headline comparison is confounded

The stated purpose (`fingerprint.py:4-8`) is to run the tool on the Dolch synthesis and
on a private Seuss transcript and "Compare vectors, not text." Two problems:

1. **Unequal instruments.** `STRESS_TABLE` and `_RHYME_HAND` are *Dolch-specific*. Run
   against a real Seuss text, out-of-budget words fall to the spelling heuristic and the
   default `"1" + "0"*(n-1)` stress. The two corpora are therefore scanned with
   different effective precision, in a direction that is not controlled.
2. **TTR is length-dependent.** Raw types/tokens falls monotonically with corpus size;
   comparing a 3214-token synthesis to a book-length transcript is comparing lengths as
   much as vocabularies. `heaps_beta` partially mitigates this and should be the
   headline number instead.

Also: `fingerprint()` constructs a `Budget` (`fingerprint.py:87`) but never calls
`.violations()` or reports coverage, so the fingerprint contains **no OOV rate** — the
one number that would make the budget comparison interpretable.

### 2.5 Fixes needed to trust `fingerprint.py`

1. **Stress assignment.** Add an explicit unstressed set for closed-class monosyllables
   (`the a to of and is in on at for`, …) and make `stress()` return `"0"` for them; or,
   better, do what the docstring already promises and hand-set all 315 entries. Until
   then, drop `meter_*` from the output rather than shipping a number that reads as
   "34% anapestic".
2. **Meter scoring.** Even with correct stress, positional agreement against a tiled
   template penalizes legal anapestic variation (headless lines, feminine endings).
   Score best-alignment over phase offsets, or match feet rather than positions.
3. **Rhyme.** Replace `rhyme_key` with a full hand table for the 315 words (the file
   already recommends this), and report rhyme over *detected* stanza structure rather
   than assuming AABB over blank-stripped lines.
4. **Comparability.** Report OOV/coverage; report standardized TTR (MTLD or a fixed
   token-window subsample) alongside raw TTR; state explicitly that bigrams cross line
   boundaries; make `end_word_concentration` use `WORD_RE`.
5. Fix `cond_entropy_bits` to use the marginal over the first elements of the bigram
   list, or rename it `joint_minus_unigram_bits`.

---

## 3. `mask_decode.py` — audit **CONFIRMED**

### 3.1 How it is supposed to work

Docstring (`mask_decode.py:10-15`):

> "for a subword tokenizer the mask is not a set of token ids, it is a set of ALLOWED
> PATHS through the token lattice, because budget words fragment. We build a trie over
> the tokenizations of the budget and mask to whatever continues some live path. Getting
> this wrong (masking to "tokens that appear in some budget word") silently permits
> out-of-budget words and quietly ruins the experiment."

`BudgetTrie.__init__` (`mask_decode.py:32-45`) encodes case/space variants of each budget
word and inserts each id sequence as a path, marking terminals with `"$"`
(`_insert`, 47-51). `generate` (64-95) keeps a `state` node, masks logits to
`allowed(state)`, and on a completed word re-opens the root plus separators.

### 3.2 Defect A — unspaced starts are permanently at the root, so words concatenate

**Root cause — `mask_decode.py:41-45`:**

```python
forms = {w, w.lower(), w.capitalize()}
variants = set()
for f in forms:
    variants.add(f)                 # <-- line 43: bare, UNSPACED form, unconditionally
    if with_space:
        variants.add(" " + f)
```

Line 43 inserts the space-less spelling of every budget word as a root path. Those
spellings are only ever legal at the very start of the text, but they stay in the root
forever.

**Trigger site — `mask_decode.py:73-75`:**

```python
allow = trie.allowed(state)
if trie.complete(state):
    allow = allow | set(trie.allowed(trie.root)) | set(sep_ids)
```

Once a word completes, the whole root — *including all the unspaced starts* — becomes
legal, with no requirement that the next token begin a new orthographic word.

**Verified with the real GPT-2 tokenizer and `BUDGETS['mask50']`:**

```
n root children: 199
UNSPACED root starts (99): ['A', 'B', 'E', 'I', 'a', 'i', 'in', 'on', 'or', 'am', 'if',
  'and', 'The', 'are', 'so', 'In', 'any', 'the', 'me', 'be', 'here', ..., 'ham', ...]
SPACED root starts (100): [' a', ' the', ' in', ' and', ' be', ' I', ...]

complete after " ham"? True
surface if we now pick unspaced start: ' hameat'
```

**`" hameat"` is exactly the failure the docstring says the trie prevents.** Half the
root's children (99 of 199) are unspaced, so at every word boundary the mask permits an
out-of-budget surface string. The `hamper`-from-`ham` example in `README.md:96-100` is
the same class of bug, arrived at from the other direction.

The same unspaced-root leak fires at two more sites: `mask_decode.py:76-78`
(`if not allow: state = trie.root; allow = trie.allowed(trie.root)`) and
`mask_decode.py:91` (`nxt_state = trie.step(trie.root, t) or trie.root`).

**Fix.** Build two root maps: `root_initial` (bare forms, used only when the emitted text
is empty or the previous token was a separator/newline) and `root_continuation`
(space-prefixed forms only). Re-open `root_continuation` at line 75.

### 3.3 Defect B — prompt state is never validated and never walked

**`mask_decode.py:69-70`:**

```python
ids = tok.encode(prompt, return_tensors="pt").to(device)
state = trie.root
```

The prompt is tokenized and then `state` is hard-set to the root. Three consequences:

1. **No budget validation.** `Budget.violations()` exists (`synth/lexicon.py:103-111`) and
   is never called on the prompt. An out-of-budget prompt is accepted silently, which
   defeats the stated purpose — "whatever Seuss-like quality appears here is attributable
   to the budget and the meter, because nothing else was supplied" (`mask_decode.py:6-8`).
2. **No partial-word reconstruction.** A prompt ending mid-word leaves `state = root`, so
   the model is forced to start a *fresh* budget word immediately after the fragment.
   Verified tokenizations: `'I do not like' -> [40, 466, 407, 588]` (last token a complete
   ` like`) but `'I do not lik' -> [40, 466, 407, 4300]` (a fragment) — the code cannot
   tell these apart.
3. **Separators are illegal on step 1.** Because `state = root` and
   `trie.complete(root)` is `False`, the branch at line 74 does not fire, so `sep_ids`
   are excluded from the first step's `allow` set. Generation can never begin with
   punctuation or a newline.

**Fix.** Walk the prompt's token ids through the trie to recover the true state, raising
on any prompt that leaves the lattice; treat a prompt that ends at a complete node (or
ends in a separator) as "boundary" so `sep_ids` and `root_continuation` are available
immediately.

### 3.4 Further defects found

| # | line | defect |
|-|-|-|
| M1 | 41-45, 75 | Defect A (§3.2) |
| M2 | 69-70 | Defect B (§3.3) |
| M3 | 72 | `model(ids)` recomputes the **entire** forward pass every step — no KV cache. O(steps²) in sequence length. |
| M4 | 71-94 | No stop condition. The loop always runs `max_new` steps and can end with `state` incomplete, emitting a trailing **prefix** of a budget word that is not itself a budget word. |
| M5 | 84 | `torch.topk(logits, min(top_k, (logits > -1e30).sum().item()))` — if the count is 0, `topk(k=0)` returns an empty tensor and `v[-1]` raises `IndexError`. Lines 76-78 make this unlikely but not impossible. |
| M6 | 113 | `sep = [tok.encode(s, add_special_tokens=False)[0] for s in [",", ".", "!", "\n"]]` takes only the **first** token of each separator, and only the unspaced variants. |
| M7 | 89-92 | On a "complete-but-extendable" node the code silently prefers continuing the current word. Verified: with `mask50` + GPT-2 there is exactly **1** such node (`'The'`), so the practical impact here is small — but the preference is undocumented and would matter for a larger budget. |
| M8 | README:99-100 | **"Implemented as a trie; unit test in the file."** There is no test in `mask_decode.py`, and `find . -name "test*"` over the whole repo returns **nothing**. The claimed test does not exist. Had it existed, defect A would have been caught. |

---

## 4. Browser feasibility (vocab ≤ 400, d ≤ 64)

Measured on CPU PyTorch; JS will be slower by a constant factor but the asymptotics are
what matter at this size.

| computation | complexity | measured | live in browser? |
|-|-|-|-|
| `spectrum()` on E or U (400×64) | O(V·d² + d³) via Gram; SVD is O(V·d²) ≈ 1.6 MFLOP | **2.0 ms** | **Yes, trivially.** Recompute per frame if you want |
| `hidden_geometry()` — forward pass | O(L·B·T²·d) ≈ 67 MFLOP at L=4,B=64,T=64,d=64 | **9 ms** incl. SVD | **Yes**, if the forward pass already runs in-browser |
| `hidden_geometry()` — SVD of (B·T)×d | O(B·T·d² + d³) ≈ 17 MFLOP | (in the 9 ms) | Yes |
| `minted_field()` — `U@e`, cosines, top-k | O(V·d) ≈ 26 kFLOP, plus O(V log k) | **0.4 ms** | **Yes**, sub-millisecond |
| `fingerprint()` token stats (TTR, entropies, hapax, Heaps) | O(N) with hashing; Zipf adds O(V log V) | instant on 3214 tokens | **Yes** |
| `fingerprint()` meter + rhyme | O(total syllables) and O(lines) — pure string ops | instant | **Cheap, but broken** (§2.3). Do not ship as-is |
| `BudgetTrie` construction | O(Σ words × variants × tokens/word) ≈ 300 encodes for mask50 | fast | Yes |
| `mask_decode` per-step masking | O(\|allow\|) + O(\|V_tokenizer\|) to materialize the mask (50257 for GPT-2) | fast per step | Mask yes — but the **LM forward** dominates, and M3 makes it O(steps²) |

**Key optimization for a port.** Every statistic in `spectrum()` depends only on the
*squared* singular values, so you never need a full SVD in JS: form the d×d Gram matrix
G = ÃᵀÃ (Ã = column-centred), take its eigenvalues λ_i via symmetric Jacobi, and set
p_i = λ_i / Σλ_j. That is O(V·d² + d³) with d ≤ 64 — a 64×64 symmetric eigensolve, well
under 10 ms in plain JS, no linear-algebra dependency. `singular_values` for display are
just √λ.

### Recommendation for a browser visualization

- **Port directly:** `spectrum()` (via the Gram trick), `hidden_geometry()` (if you already
  have the forward pass), the `fingerprint()` token-level statistics.
- **Port with a semantic fix:** `minted_field()` — cheap and visually compelling
  (nearest-neighbour lists, readout entropy), but relabel or re-derive the "next-token
  distribution" (§1.4).
- **Do not port until fixed:** `meter_anapest` / `meter_iamb` (structurally broken),
  `rhyme_rate` (heuristic with demonstrated false positives *and* false negatives, plus
  a stanza-pairing bug).
- **Port the idea, not the code:** `BudgetTrie` is a good interactive object — showing
  the live allowed-token set at each step is exactly the kind of thing a visualization
  should surface — but rebuild the root split (§3.2) and the prompt walk (§3.3) first,
  and note that it needs a real LM (transformers.js) to be meaningful.

---

## 5. Things I did not verify

- The intended source/convention for `effective_rank` (squared vs unsquared singular
  values). **I don't know** which the author meant.
- `synth/generate.py` and `synth/jabberwockify.py` were only skimmed for their imports.
  `generate.py` uses `meter_score` for rejection sampling, so §2.3 propagates into corpus
  construction, but I did not trace that path.
- `mask_decode.generate()` end-to-end against a real GPT-2 *model* (I verified the trie
  and tokenizer behaviour with the real GPT-2 tokenizer; I did not download the weights).
  The `" hameat"` result is a property of the mask, established directly from the trie.
- Whether the 62-entry `STRESS_TABLE` values are themselves correct English stress. I
  checked coverage (5.0% of demo tokens), not accuracy.
