# Red-team 007 — Architecture Explorer (agent A)

Date: 2026-08-04. Branch `main` @ `eeac0a3`.
Targets: live static build `https://context-lab.com/llm-geometry/#architecture` and the
already-running local full stack (`:5173` → `:8000`). No servers were started or stopped;
all browser work used the repo's own Playwright driven from throwaway node scripts.

Counts: **4 high**, **3 medium**, **3 low**.

---

### F1. The `swap` control contains word forms that are not English words, so both headline differences are biased
**Severity:** high
**Where:** `code/backend/src/llm_geometry/arch/vacancy_score.py:12-14` (the documented property),
`code/backend/src/llm_geometry/lex/vacancy.py:283-302` (`stem_and_suffix`) +
`build_vacancy_map` swap branch; mirrored in `code/frontend/src/lib/lexEngine/vacancy.ts`.
Visible in both stacks' `swap` preview pane (`arch-vac-preview-swap`).
**Reproduce:**
```
cd code/backend && ./.venv/bin/python -c "
from llm_geometry.lex.vacancy import *
from llm_geometry.lex.vocab import tokenize
from llm_geometry.arch.vacancy_score import default_passages
p=default_passages()[0]; tk=tokenize(p)
m=build_vacancy_map(vacancy_domain(tk), VacancyParams(p=1.0,seed=0,consistent=True,
   match_prosody=True,reveal_after=0,keep=frozenset(),mint='swap'), type_counts(tk))
for w in ['jumped','leaping','huffed','after','November']:
    s,suf=stem_and_suffix(w); print(w, s, suf, m.mapping.get(s.lower()), surface_form(m.mapping[s.lower()],s,suf,0))"
```
**Observed** (verbatim):
```
'jumped': stem='jump' suffix='ed' replacement='went' surface='wented'
'leaping': stem='leap' suffix='ing' replacement='thy' surface='thying'
'huffed': stem='huff' suffix='ed' replacement='sacks' surface='sacksed'
'after': stem='aft' suffix='er' replacement='kits' surface='kitser'
'November': stem='Novemb' suffix='er' replacement='huffed' surface='huffeder'
```
and, verbatim from the DEPLOYED site's `swap` preview (default corpus, gpt2, p=1, seed=0):
```
"She is flewed, thying over a lamed."
...
Candlestick Thirty Alack wented myself
```
Measured over the six shipped default passages at p=1, seed=0: 776 words are vacated;
**195 of them (25.1 %) have a swap form absent from `/usr/share/dict/words`**, and **35
(4.5 %) are still absent under a generous regular-inflection expansion** — `kitser`,
`sacksed`, `huffeder`, `housted`, `minler`, `lauked`, `cakler`, `homked`, `clothter`,
`mintery`, `maidns`, `Timler`, `neary`, `kyloe`.
**Expected:** `vacancy_score.py:12-14` states the control's defining property —
"every vacated stem replaced by a REAL, frequency-rank-matched English word. Equally
nonsensical, **ordinarily tokenized, every form known**". The swap pool is drawn from
domain *types*, many of which are themselves inflected (`went`, `sacks`, `huffed`, `kits`),
and `surface_form` re-attaches the SOURCE word's suffix to them. `stem_and_suffix` also
produces spurious splits (`after` → `aft`+`er`, `November` → `Novemb`+`er`), which
compounds it. Because the whole decomposition is `nll(nonce) − nll(swap)` = "the cost of
unknown form", a swap variant that already carries unknown forms **shrinks the headline
`unknown_form` and inflates `wrong_content`**. Direction of the bias is unambiguous;
I did not measure its magnitude and do not claim one. Corroborating: swap tokenizes to
2858 tokens against english's 2754 (+3.8 %), so it is not tokenization-neutral either.
**Would it have thrown?** No. It is the shipped default on both stacks.

---

### F2. Ordinary short passages return HTTP 500 `InternalError: non-finite value in response payload`
**Severity:** high
**Where:** `code/backend/src/llm_geometry/arch/vacancy_score.py:479-484` (`se = math.nan`
when `nPairs <= 1`) → `code/backend/src/llm_geometry/api/encoding.py` (`jsonable_6sig`
rejects non-finite). Panel path: uncheck "score the shipped corpus excerpts", paste text,
Score.
**Reproduce:**
```
curl -s -X POST http://localhost:8000/api/arch/vacancy-score -H 'content-type: application/json' \
  -d '{"model_id":"gpt2","passage":"I like cats and dogs."}'
```
**Observed:**
```
{"error":{"type":"InternalError","message":"non-finite value in response payload","detail":{}}} (HTTP 500)
```
The underlying value is real and intentional — calling `vacancy_score` in-process returns
`"nats": 0.0, "se": NaN, "nPairs": 1` — but the encoder refuses NaN, so the *designed*
"standard error undefined at n = 1" path is unreachable through the API and surfaces as an
opaque internal error. `plainError` falls through to `default: return e.message`, so the
red box in the panel literally reads `non-finite value in response payload`.
Neighbouring inputs are equally bad:
```
[the dog]           -> ComputeError "no tokens to average — the passage has no scored positions" (HTTP 500)
[The dog barked.]   -> ComputeError "no tokens to average — the passage has no scored positions" (HTTP 500)
[Hello world]       -> ComputeError "passage 0 has no word that survives the transform..."      (HTTP 500)
[the the]           -> InternalError "non-finite value in response payload"                     (HTTP 500)
```
**Expected:** a typed 400 naming the actual cause ("this passage yields N ≤ 1 paired
preserved tokens; a standard error needs at least 2"). Note the static stack does NOT do
this — it gates on `VACANCY_MIN_POOLED_PRESERVED = 700` and refuses with prose, and its
`num()` formatter renders `NaN` as `—`. So the two stacks diverge in behaviour here too.
**Would it have thrown?** Yes — but as an untyped internal error on a first-try user input.

---

### F3. A NEGATIVE "cost of unknown form" is rendered as a resolved result, with a negative percentage share and the conclusion asserted anyway
**Severity:** high
**Where:** `code/frontend/src/viz/arch/VacancyScorePanel.svelte:85-94` (`resolved`,
`formShare`) and `:252-268` (the verdict). The `{:else}` branch's final sentence is
unconditional.
**Reproduce (full stack UI):** Architecture tab → gpt2 → untick "score the shipped corpus
excerpts" → paste `The cat sat on the mat and the dog ran to the tree in the park. It was a
good day for a walk with the boy and his ball.` → p = `0.5`, seed = `4` → Score.
**Observed** (verbatim from the rendered panel at `http://localhost:5173/#architecture`):
```
the cost of unknown form
nll(nonce) − nll(swap)
-0.355nats
± 0.136 (sampling, 22 paired tokens)
upper bound — see below
```
```
That juxtaposition is the whole result: a word's form is worth exactly nothing to a model
trained from scratch with no lexical entries, and -0.355 nats to one that has them —
against 0.871 nats for simply saying the wrong thing, i.e. -69% of the total damage. Even
where a location exists, losing it costs far less than losing the content.
```
Backend seed sweep on the same passage at p = 0.5 (gpt2): seeds 0, 2, 4, 5, 6, 7 all give a
negative `unknown_form`; seeds 4 and 7 pass the panel's own `|nats| > 2·se` "resolved" test
(`-0.3548 ± 0.1359` and `-0.5548 ± 0.2687`), i.e. the panel *promotes* them to the
conclusion branch.
**Expected:** a negative `nonce − swap` means the nonce variant was EASIER to predict than
the swap variant — the effect the panel exists to measure ran backwards. It is also
labelled "upper bound" while being negative, and `formShare` prints `-69%` of "the total
damage". Either the sign must be guarded (refuse / flag, as the code already does for an
unresolved effect) or the closing sentence must be conditioned on
`unknownForm.nats < wrongContent.nats && unknownForm.nats > 0`.
**Would it have thrown?** No.

---

### F4. Static build drops the measured ±0.2 nat quantization uncertainty from the "both costs together" line, and the stated interval then excludes the float32 truth
**Severity:** high
**Where:** `code/frontend/src/viz/arch/VacancyScorePanel.svelte:305-311` renders
`{d.expr} = {nats(d.nats)} ± {num(d.se)}` and never reads `d.quantizationUncertaintyNats`,
which `code/frontend/src/lib/staticClient/arch.ts:452-462` DOES attach to the `total`
difference (`quantizationUncertaintyNats: VACANCY_Q8_UNCERTAINTY_NATS` = 0.2). The two
headline cards (`:232-234`) render it correctly; only the secondary row loses it.
**Reproduce:** deployed site → Architecture → GPT-2 → Score (defaults). ~91 s in wasm/q8.
**Observed** (verbatim from `https://context-lab.com/llm-geometry/#architecture`):
```
both costs together
nll(nonce) − nll(english) = 0.879 ± 0.074
```
against the same computation on the full stack at float32 (this session, 6 default
passages, gpt2): `total 0.9892 ± 0.0595`. The stated static interval `0.879 ± 0.074`
= `[0.805, 0.953]` **does not contain 0.9892**. The headline card immediately above it, on
the same page, correctly prints `± 0.059 (sampling, 847 paired tokens) · ± 0.2
(quantization, measured)`.
**Expected:** the quantization uncertainty is the dominant term for this difference —
`arch.ts:163-165` records `nonce − english : fp32 0.9892 q8 0.8790 |Δ| = 0.110`, which is
the very measurement `VACANCY_Q8_UNCERTAINTY_NATS = 0.2` was derived from. A number whose
real uncertainty is ±0.2 must not be printed as ±0.074.
**Would it have thrown?** No.

---

### F5. Chat's per-token tooltip uses the CURRENT temperature slider, not the temperature the reply was generated at
**Severity:** medium
**Where:** `code/frontend/src/viz/arch/ArchChat.svelte:89-99` (`tokenTip` reads
`$archTemperature`) and `:162` (`probColor(t.prob)`); `result` is invalidated on a model
change (`:40-46`) but not on a temperature change.
**Reproduce (full stack):** gpt2, prompt `The capital of France is`, temperature slider to
`0`, Generate reply. Then drag the temperature slider to `1.2` WITHOUT re-running, and read
the first token's tooltip.
**Observed** (verbatim `aria-label` before and after moving the slider, same reply):
```
before: greedy pick (chosen with certainty) · the model's own top-5: " the" 8.5% · " now" 4.8% · ...
after : chance of being drawn at T=1.20: 100.0% · the model's own top-5: " the" 8.5% · " now" 4.8% · ...
```
**Expected:** the reply carries `prob` values computed at the temperature it was generated
at. After the slider moves, the panel asserts that a token whose plain-softmax probability
is 8.5 % had a 100.0 % chance of being drawn at T = 1.20. The temperature the result was
produced at should be captured alongside `result` (as `forModel` already is), or `result`
dropped when the slider moves.
**Would it have thrown?** No.

---

### F6. Any passage containing a non-ASCII letter inside a word returns HTTP 500, and the docstring claims this cannot happen
**Severity:** medium
**Where:** `code/backend/src/llm_geometry/arch/vacancy_score.py:215-247`
(`preserved_token_indices`; the docstring at :228-231 says "It cannot happen with the
byte-level pretokenizers of the curated models"), mirrored at
`code/frontend/src/lib/staticClient/byteSpans.ts:128-149`. Root cause: `WORD_RE` is
`[A-Za-z]+(?:['\-][A-Za-z]+)*` (ASCII only), so `naïvely` is two words `na` + `vely`, one
of which can be preserved while the other is vacated — and the tokenizer emits one token
covering both.
**Reproduce:**
```
curl -s -X POST http://localhost:8000/api/arch/vacancy-score -H 'content-type: application/json' \
  -d '{"model_id":"gpt2","passage":"The cat sat on the mat, naïvely. Café résumé is on the table and the dog ran."}'
```
**Observed:**
```
{"error":{"type":"ComputeError","message":"token 7 spans both a preserved and a vacated word ('na', 'vely'); refusing to attribute it","detail":{"index":7,"words":["na","vely"]}}} (HTTP 500)
```
**Expected:** refusing rather than mis-attributing is the right call, but (a) the comment
asserting it is unreachable is false, and (b) any reader pasting French, German, Spanish,
or English with `café` / `naïve` / `résumé` / `coöperate` gets an opaque 500 with a token
index. Emoji and CJK are fine (verified: `🙂🙂` and `猫が座った。` both score cleanly), so
this is specifically Latin letters with diacritics *inside* a word.
**Would it have thrown?** Yes.

---

### F7. The panel exposes p ∈ (0,1) at 0.05 steps, where the swap map is provably non-injective and the Lexicon Lab refuses the analogous request; p = 0 prints a pure identity as a measurement
**Severity:** medium
**Where:** `code/frontend/src/viz/arch/VacancyScorePanel.svelte:163-174` (`min=0 max=1
step=0.05`, no explanation of what `p` is). The refusal exists only in
`code/backend/src/llm_geometry/lex/vacancy.py:1212-1222` (`map_vocab_words`), which the
Architecture Explorer's `vacate_text` path never calls.
**Reproduce:**
```
cd code/backend && ./.venv/bin/python -c "
from llm_geometry.lex.vacancy import *
from llm_geometry.lex.vocab import tokenize
from llm_geometry.arch.vacancy_score import default_passages
tk=tokenize(default_passages()[0])
m=build_vacancy_map(vacancy_domain(tk), VacancyParams(p=0.5,seed=0,consistent=True,
  match_prosody=True,reveal_after=0,keep=frozenset(),mint='swap'), type_counts(tk))
print('injective_at_every_p', m.injective_at_every_p)"
```
**Observed:** `injective_at_every_p False`. And through the API at p = 0.5 nothing is
refused: `{'wrong_content': 0.3938, 'unknown_form': -0.031, 'total': 0.3627}`.
At p = 0.05 and p = 0.0 the API returns `{'wrong_content': 0.0, 'unknown_form': 0.0,
'total': 0.0}` with `se = 0.0`, which the panel renders as `0.000 nats ± 0.000 (sampling,
20 paired tokens)` for "the cost of wrong content" — a rendered measurement of an identity
(all three variants are the same string), with nothing on screen saying so.
**Expected:** either mirror the Lexicon Lab's `p ∈ {0,1}` refusal here, or state on the
control that intermediate `p` breaks the swap control's injectivity (contract §5.2a), and
special-case the degenerate p→0 case where the variants are character-identical.
**Would it have thrown?** No.

---

### F8. Static mislabels an exported EXACT tile as `downsampled: true, method: "strided_mean"`
**Severity:** low
**Where:** `code/frontend/src/lib/staticClient/arch.ts:562-581` — the over-budget branch
hardcodes `downsampled: true` / `method: "strided_mean"` and ignores `tile.downsampled` /
`tile.method`. Triggered for every 1-D parameter, because `ArchInspector.svelte:66-71` asks
for the overview with `max_cells: 128` while the tile has R rows.
**Observed:** `https://context-lab.com/llm-geometry/static-data/arch/gpt2/tiles.json`
contains, verbatim:
```
{'param': 'transformer.h.0.ln_1.weight', 'shape': [768, 1], 'grid_shape': [768, 1],
 'downsampled': False, 'method': 'exact', 'vmin': 0.0418614, 'vmax': 0.252667, ...}
```
yet the static client will return that tile with `downsampled: true, method:
"strided_mean", grid_shape: [768,1]`, i.e. a full-resolution (but uint8-quantized) strip
described as a strided mean.
**Expected:** carry `tile.downsampled` / `tile.method` through.
**Would it have thrown?** No.

---

### F9. `nChars` (and therefore `bitsPerChar`) is code points in Python and UTF-16 units in JS
**Severity:** low
**Where:** `vacancy_score.py:412-430` / `:433-451` use `len(scored.text)`;
`code/frontend/src/lib/staticClient/transformersRuntime.ts:396` uses `nChars: text.length`.
`nChars` is returned by both stacks; `bitsPerChar` is derived from it in Python.
**Reproduce:**
```
python3 -c "print(len('The cat sat on the mat. 🙂🙂 It was a very good day for the dog and the bird.'))"
node -e "console.log('The cat sat on the mat. 🙂🙂 It was a very good day for the dog and the bird.'.length)"
```
**Observed:** `75` vs `77`.
**Expected:** the module's own docstring (`vacancy_score.py:32-34`) says "Python indexes
code points where JavaScript indexes UTF-16 units, so the two disagree on the same string
… bytes are the only safe contract unit" — and then the payload reports a character count
anyway. Either count UTF-8 bytes on both sides, or document `nChars` as stack-dependent.
Impact today is bounded: static refuses `bitsPerChar` (`null`), so only `nChars` diverges.
**Would it have thrown?** No.

---

### F10. The size gate's rejection message quotes a ceiling the model did not actually exceed on the config-estimate path
**Severity:** low
**Where:** `code/backend/src/llm_geometry/arch/gate.py:89-103`. `effective_ceiling` is
`ARCH_MAX_PARAMS * 0.8` = 1.2 B when `source == "config_estimate"`, but the message
interpolates `ARCH_MAX_PARAMS` (1.5 B). `ArchitectureExplorer.svelte:332-335` also states
flatly "Models are capped at **1.5B parameters**".
**Observed** (code, verbatim):
```python
effective_ceiling = (
    ARCH_MAX_PARAMS if source == "safetensors_metadata" else ARCH_MAX_PARAMS * 0.8
)
if total > effective_ceiling:
    raise ModelTooLargeError(
        f"Model '{mid}' has ~{total:,} parameters, over the Architecture Explorer "
        f"ceiling of {ARCH_MAX_PARAMS:,}. ...
```
so a 1.3 B model with no safetensors metadata is told it "has ~1,300,000,000 parameters,
over the … ceiling of 1,500,000,000".
**Expected:** quote `effective_ceiling` and say why it is lower, and soften the "capped at
1.5B" prose. **I did not trigger this live** — it needs a 1.2–1.5 B model whose repo has no
safetensors index, and I did not find one worth the download. The claim rests on the code
quote only.
**Would it have thrown?** Yes (it is an error path), but with a self-contradictory message.

---

## What I tried that came back clean

Cross-stack and cross-source numeric checks — all agreed:

- **Model capabilities vs HuggingFace configs.** `/api/models` reports Qwen2.5-0.5B
  24 layers / hidden 896 / 14 heads / 2 KV / vocab 151936, SmolLM2-360M 32/960,
  SmolLM2-135M 30/576, gpt2 12/768/50257 — all correct.
- **Parameter counts.** `graph.meta.total_params` = 124,439,808 for gpt2 and 494,032,768
  for Qwen2.5-0.5B, i.e. the standard published counts. Summing node params gives
  163,037,184 for gpt2; the difference is exactly the tied `lm_head.weight`
  (50257×768 = 38,597,376), so **tied weights are correctly counted once** and the alias
  carries `"tied_to": "transformer.wte.weight"` with a `tied →` badge in the inspector.
- **Op counts.** Qwen graph has 340 nodes / 387 edges and the trace has 340
  `node_activations` with **zero** ids in one and not the other; the UI's "all 340 traced
  ops" matches. Kind histogram is self-consistent (168 linear = 24×7, 49 rmsnorm = 24×2+1,
  48 residual_add, 24 softmax, 24 activation, 25 rope = 24 layers + the shared
  `model.rotary_emb`). gpt2: 149 nodes, 149 acts, "149 traced ops" in the UI.
- **`n_params` in `static-data/index.json`** (291/291/273/149) is the count of exported
  weight tensors and reconciles exactly with each architecture's per-layer tensor list.
- **Static precomputed trace vs a fresh backend trace** (gpt2, "The capital of France
  is"): identical token ids, identical top-10 ids/texts, probabilities agreeing to 5
  decimal places (` the` 0.0845924 vs 0.084591), and the worst relative `out_norm`
  discrepancy across all 149 ops was **9.4e-6**.
- **Static weight inspector vs backend `/api/arch/weights`.** Clicked
  `transformer.h.0.attn.c_attn` on the deployed site, zoomed to an exact window; the panel
  showed `rows 147–210 · cols 517–580 … min -0.7605 · max 1.046 · μ -0.0003268 · σ 0.2172`
  and the backend returned `{'min': -0.760522, 'max': 1.04598, 'mean': -0.000326779,
  'std': 0.217221}`. Real HTTP Range reads to `huggingface.co` were observed
  (`range=bytes=272053223-272053478` etc.), no console errors.
- **safetensors dtype decoding.** Re-implemented `decodeScalars`' BF16 (`u16 << 16` into
  the f32 bit pattern) and F16 (incl. subnormals, ±Inf, NaN, 65504) decoders in node and
  compared against `torch.bfloat16` / `torch.float16` round-trips on 15 adversarial values:
  no mismatches.
- **Static vacancy tokenization vs backend.** Both stacks produced 2754 / 2858 / 3810
  tokens and 847 preserved tokens on the shipped default set for gpt2 — identical
  tokenization, so the only difference really is the dtype, as `arch.ts:160-161` claims.
  The q8 vs fp32 gap on `swap − english` (0.644 vs 0.7166, |Δ| = 0.073) reproduces the
  constant in that comment.
- **The q8 refusals fire on the deployed site**, verbatim and in place: `nonce − swap`
  refused, absolute `nllPreserved`/`nllAll`/`bitsPerChar` shown as `—` with the absolute
  refusal, per-passage table refused. The `What ran` line correctly said
  `gpt2 on the static stack at q8/ wasm, p = 1, seed = 0`.
- **`logitsSanity` is reachable and not spuriously firing.** `RUNTIME_LADDER` no longer
  contains `q4f16`; `selfCheck` runs on every pipeline build; headless Chromium logged
  `No available adapters.` and fell through to `wasm/q8`, which passed and produced real
  numbers.
- **Decoding constants are actually mirrored:** `ARCH_TOP_P 0.9 / ARCH_TOP_K 50 /
  ARCH_REPETITION_PENALTY 1.1 / ARCH_MAX_NEW_TOKENS 128 / ARCH_WEIGHTS_MAX_CELLS 4096 /
  ARCH_DEFAULT_MAX_CONTEXT 64` all match `transformersRuntime.ts` and `arch.ts`.
- **Trace truncation.** A 200-word prompt returned `truncated: true` with exactly 64
  tokens, and the elision chip is rendered before the first surviving token.
- **Scrubber/playhead correspondence.** `order[idx]` is `trace.node_activations[idx]` and
  the playhead prints that op's own `out_norm` / `out_shape`; the per-layer panels are
  explicitly labelled "layer N out" and the Explain says the top-10 is the full-pass
  distribution, so nothing claims to be a mid-forward state that isn't one.
- **Full-stack browser pass** (gpt2, live trace + inspector + chat): zero console errors,
  zero page errors, zero failed requests.
- **Vacancy input validation:** `p = -0.5` / `p = 1.5` → typed 400; `passages: []` → typed
  400; whitespace-only passage → typed 400; a huge negative seed is accepted and echoed.
- **Emoji and CJK passages score correctly** through the byte-span aligner (verified end to
  end on gpt2).

## Not covered by this pass

- WebGPU (this machine's headless Chromium has no adapter; `webgpu/q8` was skipped, as the
  brief predicted).
- Qwen / SmolLM2 in-browser generation and vacancy scoring — only gpt2 was run live in the
  browser, for time.
- Concurrency: simultaneous `/trace` + `/generate` on one model (`_TRACE_LOCK` looked
  correct on inspection but I did not race it).
