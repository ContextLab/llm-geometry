# Red-team C — Lexicon Lab (feature 007, the vacancy transform)

Date: 2026-08-04. Agent: red-team C. Scope: `viz/lex/`, `llm_geometry/lex/`,
`api/routes_lex.py`, `lib/lexEngine/`, `lib/staticClient/lex.ts`, contract
`specs/007-vacancy-transform-field/architecture.md`.

Both modes attacked: local full stack (`localhost:5173` → `:8000`) and the live static build
(`https://context-lab.com/llm-geometry/#lexicon`). Nothing was fixed; no file under `code/`
or `specs/` was edited.

**Counts: 1 high, 4 medium, 2 low.** The headline invariance theorem (SC-703) survived every
attack, and I found **no** TS↔Python divergence in the transform itself for any seed
representable as a float64 — see "what came back clean" for the volume behind that claim.

---

### F1. Loading any saved model file makes the whole tab claim the weights were trained — including a file that records `"trained": false`

**Severity:** high
**Where:** `code/frontend/src/viz/lex/LexiconLab.svelte:412-425` (`onLoadedModel`, which sets
`trained = { model, vocab: modelVocab, note }` unconditionally);
`code/frontend/src/viz/lex/ModelFile.svelte:106-140` (the loader reads only `metrics.note` and
discards `metrics.provenance` / `metrics.trained`); `provenance.ts:18` (`provenanceOf(true, false)
→ "trained"`).

**Reproduce:** open `#lexicon`, train nothing, click **↓ Save model**, reload the page, click
**↑ Load model** and pick the file just saved. Script: `prov.mjs` (scratchpad, deleted).
Verified on **both** `http://localhost:5173/#lexicon` and
`https://context-lab.com/llm-geometry/#lexicon` — identical output.

**Observed** (live site, verbatim):

Before saving, with nothing trained:

```
"saveUntrainedNote": "Nothing has been trained yet, so a save right now writes the random
 initialization — a real model file, of a model that has learned nothing."
"spectrumUntrained": "This model has not been trained. The bars, the baseline and both rank
 markers coincide because they are the same random initialization — train it and watch them
 separate, or fail to."
```

The file it writes records the truth:

```
SAVED bundle metrics: {"note":"untrained random initialization","provenance":"untrained","trained":false,"edited":false}
```

After reloading the page and loading that same file back:

```
"fileOk": "loaded untrained.llmlex.json · 314-word dolch budget (318 rows) · model 7f87bcd524a4… verified"
"fileError": null
"saveUntrainedNote": null
"spectrumUntrained": null
"sampleHeader": "Generate from the model you trained  PROMPT  TEMPERATURE 0.90 …"
BODY contains "from the model you trained": true
BODY contains "has not been trained": false
CONSOLE ERRORS: []
```

**Expected:** the tab's provenance is the point of `provenance.ts` ("What the weights on screen
ACTUALLY are… Five sentences then described a model that was not on screen"). The bundle carries
`provenance`, `trained` and `edited` precisely so a reader downstream does not have to guess
(`ModelFile.svelte:85-90`), and the loader throws all three away. After a load, **every**
provenance-conditioned sentence in the tab — SamplePanel's header, SpectrumPanel's
`lex-spectrum-untrained` block, TokenCloud's and ForwardPassPanel's equivalents, and ModelFile's
own "Nothing has been trained yet" warning — flips to the trained wording, and the untrained
warnings vanish entirely. The spectrum panel then invites the reader to read a
random-initialization eigenvalue spread as a training result.

**Would it have thrown?** No. `lex-file-error` is null, the console is empty, and the success
line ends in the word "verified".

---

### F2. `POST /api/lex/vacancy` and `POST /api/lex/train` silently ignore `mint` — `swap` returns nonce output under a byte-identical digest

**Severity:** medium
**Where:** `code/backend/src/llm_geometry/api/routes_lex.py:594-601` — `_vacancy_params`
constructs `VacancyParams(p=…, seed=…, consistent=…, match_prosody=…, reveal_after=…, keep=…)`
with **no `mint=`**, so the dataclass default `"nonce"` always wins. `mint` appears nowhere else
in the file (`grep -n "mint" routes_lex.py` returns only the words "minted", "re-mint",
"remint_rounds" and the *comment* at line 574 that anticipates "the swap-mint control of §8.3").

**Reproduce:**

```
curl -s -X POST localhost:8000/api/lex/vacancy -H 'content-type: application/json' \
  -d '{"text":"the little dog ran over the lazy brown mountain and the happy children sang loudly today","p":1.0,"seed":0,"mint":"nonce","preview_chars":200}'
curl -s -X POST localhost:8000/api/lex/vacancy -H 'content-type: application/json' \
  -d '{… same …,"mint":"swap",…}'
```

**Observed:**

```
NONCE: 'the skoufenty scarrt glaimp over the flaumous bor sturrdum and the squrthy kliesowle plish peardly refon'
sha 3c4f7f081ed0d2d92ee24d32c3c77bf30ac4b88f7144de2691f2bacfd8e24672
SWAP : 'the skoufenty scarrt glaimp over the flaumous bor sturrdum and the squrthy kliesowle plish peardly refon'
sha 3c4f7f081ed0d2d92ee24d32c3c77bf30ac4b88f7144de2691f2bacfd8e24672
```

A garbage value is equally silent: `{"p":0.5,"mint":"bogus"}` → `200 OK rule=mapped words=314
vac=4470`.

**Expected:** either the parameter is honoured, or it is refused. The response echoes
`p`, `seed`, `consistent`, `match_prosody`, `reveal_after` and `keep` back to the caller
(`routes_lex.py:739-750`) — deliberately, so a client can see what was applied — but not `mint`,
so a caller has no way to detect that the control it asked for was not used. Two aggravating
facts: (a) the frontend has a live `mint` control (`VacancyPanel.svelte:648`,
`data-testid="lex-vacancy-mint"`) that a reader will assume the API mirrors, and (b) the
`_vacancy_key` cache-key builder at `routes_lex.py:604-621` reads the dataclass's own fields
*on purpose* — "so that a knob added to the transform is in the cache key the day it is added" —
which means the machinery downstream of the parse is already correct and only the parse is
missing. Contract §8.3 makes `mint` a first-class parameter; `api-lex.md`'s wire list does not
include it, so this is a gap between the two contracts as much as a code defect.

Mitigating: the Lexicon Lab never calls `/api/lex/*` (`grep -n "lex" src/lib/dataClient.ts
src/lib/clientProvider.ts` returns nothing — the tab computes in the browser in both modes), so
no *shipped* screen is affected today. The route is a public documented surface and a contract
test target, so it is still wrong.

**Would it have thrown?** No — HTTP 200, plausible output, matching digest.

---

### F3. TS ↔ Python divergence for `seed > 2^53`: different maps, different corpus, no error in either stack

**Severity:** medium
**Where:** `code/backend/src/llm_geometry/lex/vacancy.py:349`
(`f"{seed}:{stem.lower()}"` — Python stringifies an arbitrary-precision int exactly) vs
`code/frontend/src/lib/lexEngine/vacancy.ts:412`
(`` `${seed}:${stem.toLowerCase()}` `` — JS stringifies the *rounded double*).
`vacancyU`'s guard is `Number.isInteger(seed)`, which `9007199254740993` passes after rounding.
Neither `VacancyParams.__post_init__` nor `vacancyParams` bounds the magnitude, and
`_as_int` on the route accepts any integer.

**Reproduce:** run both engines on the same case list with seeds straddling 2^53 (differential
driver, scratchpad). Backend-reachable directly:
`curl -X POST localhost:8000/api/lex/vacancy -d '{"p":1,"seed":9007199254740993}'` → `200`.

**Observed** (same text, same `p = 1.0`, seed `9007199254740993`):

```
AGREE   seed 9007199254740992   (= 2^53)
DIVERGE seed 9007199254740993
   TS vacated: 'the sterndleerid faung perrn over the gnernkous steent wrurcker and troobid chargleenow wup wealrly unveell'
   PY vacated: 'the swellelle menk yersk over the gourder trerv youndleel and thrailter klargener doal talkly redrup'
DIVERGE seed 9007199254740995
DIVERGE seed 1152921504606846983
DIVERGE seed -9007199254740993
DIVERGE seed 12345678901234567890
AGREE   seed 2147483648
AGREE   seed 4294967297
```

5 038 field-level differences across those five cases — the whole `mapping`, `forbidden`,
`mintedStress`, vacated text and every §10 statistic move.

**Expected:** contract §4 goes to considerable trouble to make `u` structurally identical across
the two languages (`(top64 >> 11) / 2**53`, departure #2, "a cross-language equality that depends
on a subtle proof is one refactor away from being false"). The same care is not applied to the
*input* of the digest. The fix is a bound, not arithmetic — but the contract states none, so both
implementations are "correct" per the document while disagreeing. This is the exact failure class
§4 exists to prevent, moved one line upstream.

**Would it have thrown?** No. Both stacks return a complete, self-consistent, plausible result.

---

### F4. The seed box accepts values above its own declared `max` and silently rewrites what the user typed

**Severity:** low
**Where:** `code/frontend/src/viz/lex/VacancyPanel.svelte:573-585` — `<input type="number"
min="0" max="9999" step="1">` whose handler is
`const v = Math.trunc(Number(e.currentTarget.value)); if (Number.isFinite(v)) onSeed(Math.max(0, v));`
There is no upper clamp; `max` on a number input does not block a typed value.

**Reproduce:** type `9007199254740993` into `[data-testid="lex-vacancy-seed"]`.

**Observed:**

```
seed field now reads: 9007199254740992
injectivity: Measured just now at p = 0.00, nonce: 2,233 distinct images from 2,233 domain types — 0 lost image slots.
errors: []
```

**Expected:** either reject the out-of-range value visibly, or clamp it to the declared `max`.
Silently substituting a different seed is a small instance of the same defect class as F3: the
number used is not the number asked for, and nothing says so.

**Would it have thrown?** No.

---

### F5. `architecture.md` §5.2a's "measured" swap-collision counts (191 / 246 / 190) do not reproduce; the live UI shows 244 / 322 / 233

**Severity:** medium
**Where:** `specs/007-vacancy-transform-field/architecture.md:327` and
`specs/007-vacancy-transform-field/spec.md:156`.

Contract, verbatim:

> "Measured, to make it concrete rather than merely proved: the frequency-rank swap below
> produces 191 / 246 / 190 colliding types at `p = 0.25 / 0.5 / 0.75` on the shipped corpus, and
> 0 at `p ∈ {0, 1}`."

**Reproduce:** build the swap map over `vacancy_domain(tokenize(load_corpus_text()))` with
`type_counts`, push every domain type through `surface_form` at each `p`, and count. Or open
`#lexicon`, choose **swap**, and read `[data-testid="lex-vacancy-lost-slots"]`.

**Observed** — Python engine, shipped corpus, seed 0:

```
p=0.0 : distinct images=2233 of 2233 types | lost slots=0
p=0.25: distinct images=1989 of 2233 types | lost slots=244 | groups=244 | types involved=488
p=0.5 : distinct images=1911 of 2233 types | lost slots=322 | groups=322 | types involved=644
p=0.75: distinct images=2000 of 2233 types | lost slots=233 | groups=233 | types involved=466
p=1.0 : distinct images=2233 of 2233 types | lost slots=0
```

The live browser agrees to the unit:

```
Measured just now at p = 0.25, swap: 1,989 distinct images from 2,233 domain types — 244 lost image slots.
Measured just now at p = 0.50, swap: 1,911 distinct images from 2,233 domain types — 322 lost image slots.
Measured just now at p = 0.75, swap: 2,000 distinct images from 2,233 domain types — 233 lost image slots.
```

I swept seeds 0..11 under three natural readings (lost image slots; types involved in a
collision; the same restricted to corpus types) and **none** produces 191 / 246 / 190:

```
seed 0: lost=[244, 322, 233] involved=[488, 644, 466] corpus_lost=[237, 314, 228]
seed 6: lost=[219, 336, 243] involved=[438, 672, 486] corpus_lost=[211, 327, 238]
seed 7: lost=[228, 331, 256] involved=[456, 662, 512] corpus_lost=[223, 323, 250]
seed 10: lost=[228, 301, 213] involved=[456, 602, 426] corpus_lost=[223, 288, 204]
```

**Expected:** the code is right and the document is stale — the panel measures live and its
number is the true one. But `architecture.md` is normative and this is one of its explicitly
*measured* claims, in the same file that forbids transcribing the source document's own figures
(§10: "Do not transcribe them into any UI string, test, or doc"). The `0` at `p ∈ {0, 1}` half of
the sentence does reproduce.
**Uncertainty I cannot resolve:** the document does not state which seed or which counting
definition it used, so I cannot prove the number was never true — only that it is not reachable
by any of the three readings I tried, at any of twelve seeds, against the code as shipped.

**Would it have thrown?** No — a doc-level claim, unasserted by any test.

---

### F6. The `metrics` block is outside all three digests, and the backend stores and re-serves fabricated numbers verbatim

**Severity:** low (documented as intentional — reported because F1 falsifies the stated
justification)
**Where:** `routes_lex.py:183-196` (`_model_token` hashes config + words + weights only),
`:208-237` (`_weights_token`, `_vocab_digest`), `:1310-1330` (import path);
`lib/lexEngine/bundle.ts:37-39`.

**Reproduce:** export a bundle, perturb one weight by `+0.12345`, recompute all three digests
honestly, replace `metrics`, POST it, then GET it back.

**Observed:**

```
forged token c0bb9d6a4605a15a9a98e852b04db85a
{"model_token":"c0bb9d6a4605a15a9a98e852b04db85a","config":{…},"vocab_size":92,"param_count":5360}   HTTP 200
metrics served back: {'base': None, 'elapsed_s': 0.0, 'final_loss': 1e-05, 'first_loss': 2e-05,
                      'seed': 3, 'steps': 1000000, 'val_loss': 1e-05}
```

**Expected:** `bundle.ts:37` states the design intent — "`metrics` is provenance and is
deliberately outside every digest — **it is the one block that cannot mislabel a token**". That
claim is false as shipped: `ModelFile.svelte:122` reads `loaded.metrics.note` straight into the
UI's status line, and (F1) the block's `provenance`/`trained` fields are the *only* record of
whether the weights were ever trained. A `final_loss` of `1e-05` on a random-init model is a
plausible wrong number with a green checkmark beside it. Either the block should be inside a
digest, or nothing read from it should reach the screen.

**Would it have thrown?** No — HTTP 200 both ways.

---

### F7. Latent asymmetry: `domainTypesVacated` is computed by two different rules in the two stacks

**Severity:** low
**Where:** `lexEngine/vacancy.ts:1403-1418` (`countVacatedByMap` requires
`surfaceForm(stem, suffix, nonce, seed) !== t`, i.e. the image must actually differ) vs
`llm_geometry/lex/vacancy.py:1324-1326` (`domain_vacated = {t for t in domain_eligible if
vacancy_u(stem_and_suffix(t)[0], params.seed) < params.p}` — no such test).

**Observed:** no case in ~2 100 differential inputs made the two disagree, because condition B
(nonce) and B₁ (swap) both forbid a surface form equal to its own source type. So this is latent,
not live — I did not observe a wrong number from it.

**Expected:** §5.8 spends five bullets on exactly this class of asymmetry ("one stack stored it;
the other rebuilt it…", "TypeScript called it `map` and Python `mapping`"). A statistic defined
two ways in two files is one condition-B regression away from becoming a real divergence, and the
test suite would not catch it because it compares outputs, not definitions.

**Would it have thrown?** No.

---

## What I tried that came back clean

**1. TS ↔ Python differential fuzzing of the vacancy transform — 0 differences.**
A driver in each stack emitting `mapping`, `mintedStress`, `forbidden` (sorted), `remintRounds`,
`bijective`, `imageSize`, `injectiveAtEveryP`, `domain` size, the whole vacated text, the
post-transform token list, `mapVocabWords`, per-word probes (`stemAndSuffix`, `isEligible`,
`vacancyU` as an exact float, `stress`, `syllables`, `ruleSyllables`, `transformWord`) and all
23 §10 statistics; compared field-by-field with exact string equality and 1e-12 relative
tolerance on floats.

- **1 800 randomized cases** over 6 generator seeds: random `p` (including 6-decimal values),
  random seeds (`0, 7, 1, 12345, 999999, -3, 2^31`), all four conditions, both prosody settings,
  `keep` sets, `mint ∈ {nonce, swap}`, and texts built from unicode (`café naïve 日本語 Ωmega
  über ﬁ ٣ Ⅻ 🙂`), hyphens, apostrophes, ALL CAPS, MiXeD case, prefix families, digits, empty
  strings, whitespace-only strings, and texts with no eligible word at all. **0 differences.**
- **131 adversarial stress cases**: 14-syllable manufactured words to drive the mint loop past
  its `a ≥ 400` / `a ≥ 800` relaxations, dense prefix + suffix families to force re-mint rounds,
  `mint="swap"` with **no** counts argument, whole texts case-folded four ways, unicode-only
  texts, and `p` set exactly to a stem's `u`. **0 differences.**
- **The real committed corpus × 86 parameter combinations** (4 seeds × 9 `p` × 2 prosody, plus
  swap at `p ∈ {0, 0.5, 1}` × 2 seeds, plus `consistent=false` and `revealAfter=2` at 2 seeds ×
  2 `p`). **0 differences** — including the full 2 104-entry map, the full `forbidden` set, and
  the whole 86 kB vacated text byte for byte.

**2. The invariance theorem (SC-703) — could not be broken.**
Token id streams compared element for element over 6 seeds × 17 values of `p` (including
`0.05, 0.33, 0.66, 0.95, 0.99`) × 2 prosody settings × all five Dolch budgets and a 500-word
frequency budget: **0 mismatches**. Then on the *real backend* with real PyTorch, 25-step runs at
`p ∈ {0, .25, .5, .75, 1}` × `seed ∈ {0, 7}`:

```
p=0.0  s=0 first=4.56533 final=2.2677 val=2.46495 rows=96 hist=e0c6b0a0aa0574cb
p=1.0  s=7 first=4.56533 final=2.2677 val=2.46495 rows=96 hist=e0c6b0a0aa0574cb
```

— identical for all ten runs, including a digest over the full 25-step loss history, while
`model_token` correctly *differs* (`966b10f4…` at `p=0`, `100cfaff…` at `p=0.5`, `8d529fe3…`
at `p=1`), proving the vocabulary really did move underneath an unmoved loss.

**3. Nesting, stability, injectivity-at-every-`p`, and the case-commuting invariant — 0 failures.**
Over 6 seeds on the real corpus: vacated stem sets nest across 17 values of `p`; every stem's
nonce is byte-identical at every `p` where it is vacated; the type map is injective at all 17
values of `p` over the full 2 233-type domain (not just the endpoints); and
`lower(transformWord(w)) == transformWord(lower(w))` for every corpus type in its original,
upper, capitalised and lower forms — 4 × 2 211 × 6 checks, all passing.

**4. The `swap` refusal is airtight.** `mapVocabWords` refuses at
`p ∈ {5e-324, 1e-323, 1e-300, 1e-12, 0.0001, 0.25, 0.5, 0.75, 0.999999999999999, 1−2⁻⁵³}` and
allows only `p ∈ {0, 1}`, at seeds 0/3/7 — denormals included, no bypass found. Collisions are
exactly 0 at both endpoints. `mint="swap"` with `consistent=false` raises with the §8.3 supply
argument. The UI carries the engine's refusal verbatim and substitutes **nothing**: at swap +
`p=0.5` the coverage counters, the invariance verdict and the vocabulary all go empty rather than
falling back to the rebuilt rule.

**5. Every §10 number reproduces exactly on the shipped corpus body.**
`domainTypesTotal=2233`, `corpusTypesTotal=2211`, `domainTypesEligible=1944`,
`corpusTypesEligible=1922`, `stemsTotal=1680`, `tokensTotal=16000`, `tokensVacated=8202`,
swap pool `1944` types against `1680` stems; `corpusTypesVacated` `0/461/954/1430/1922` (seed 0)
and `0/434/975/1440/1922` (seed 7); `domainTypesVacated` `0/469/966/1448/1944` and
`0/440/985/1455/1944` — all matching §10 verbatim. §5.8's named cases hold: at seed 7,
`remintRounds = 1`, `hang → smeeg`, and `"wak" ∈ forbidden` (the superseded nonce is stored, not
reconstructed). *Caveat for future readers: these numbers only reproduce against
`load_corpus_text()`, the Gutenberg-trimmed body (86 408 chars, 16 000 tokens). Against the raw
committed file they read 2 717 / 2 736 / 19 050 — I lost twenty minutes to that before checking.*

**6. Save / load — the three digests hold, in both directions.**
A bundle written by the **browser** was accepted by the **backend**, which independently
recomputed and matched `model_token = 7f87bcd524a4cd673a58cc945bd51537`. Deleting any one of the
three digests is fatal, not skipped:

```
delete model_token   -> 400 "bundle has no usable 'model_token', so its contents cannot be verified — refusing to load it."
delete weights_token -> 400 (same shape)
delete vocab_sha256  -> 400 (same shape)
```

Reversing the word list without touching the digests:

```
400 "bundle declares model_token '7f87bcd524a4cd673a58cc945bd51537' but its own contents hash to
     '7fd4dbac4d56e856e7db7c36029c6add'; refusing to load a file whose weights and label disagree"
```

The only block I could forge past all three checks is `metrics` (F6). A re-import of an
already-known token dedups rather than overwriting, so even a metrics lie cannot corrupt an
existing cache entry — only a fresh one.

**7. Budgets — the coverage counter matches the backend to the displayed digit.**
All five Dolch budgets read in the live browser against `GET /api/lex/budgets`:

| budget | rows | UI token cov. | backend | UI oov | backend | UI whole lines | backend |
|-|-|-|-|-|-|-|-|
| pre_primer | 40 + 4 | 27.9% | 0.278562 | 2,172 | 2172 | 5 / 3,071 | 5 |
| primer | 92 + 4 | 41.0% | 0.409625 | 2,120 | 2120 | 20 | 20 |
| first | 133 + 4 | 49.7% | 0.497125 | 2,080 | 2080 | 45 | 45 |
| service | 220 + 4 | 55.1% | 0.550813 | 1,996 | 1996 | 96 | 96 |
| full | 314 + 4 | 60.8% | 0.607875 | 1,919 | 1919 | 215 | 215 |

Frequency budgets (`top40…top314`) likewise. Under the mapped condition the coverage counters do
**not** move as `p` sweeps 0 → 1 (60.8% / 39.2% / 1 919 at every `p`), which is the invariance
showing up in a second place. Under the controls they collapse as documented — `inconsistent`:
44.5% coverage, 8 383 OOV types, "token id streams differ"; `reveal`: 45.5%, 2 970 OOV, and
`corpusTypesVacated` reads 1 275 / 1 922, i.e. measured from the two texts rather than from map
membership (the §10 fix, still holding). Pasted text with a frequency budget preserves ids
correctly (`the → the`, `dog → scarrt`, `fox → florng`, order preserved, 15/17 tokens in budget).
A real HuggingFace dataset (`roneneldan/TinyStories`, 20 samples) runs the mapped rule end to end
with `bijective: true`; a script-based dataset (`karpathy/tiny_shakespeare`) fails **loudly** with
a typed `ComputeError` rather than degrading.

**8. Pinned constants that are still true.** `SHIPPED_LINE_COUNT = 3071` (measured: 3 071
token-producing lines); `SHIPPED_BODY_LINE = 619` (0-based index 619 is exactly
`LITTLE BO-PEEP`); `corpus.py`'s "9.1% of its non-blank lines are exact duplicates" (measured
9.106% on strip-normalised lines); Dolch sizes 40 / 92 / 133 / 220 / 314, and the `314`-not-`315`
explanation.

**9. API error envelope.** `p` out of `[0,1]`, non-numeric `p`, negative `reveal_after`, `keep` as
a bare string, `preview_chars > 20000`, an unknown budget, `size` with `source="dolch"`, a corpus
with no word tokens, and an empty corpus all return typed `400 InvalidParamError`s with the
messages `api-lex.md` specifies. No console errors were produced anywhere in the tab during any
browser run, local or live.
