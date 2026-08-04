# Verification V2 — red team 007 Architecture + Lexicon fixes

Date: 2026-08-04. Branch `main` @ `33c5ce6`. Agent: VERIFICATION V2 (did not write any of
the code under test).

Method: the already-running shared dev stack (`:5173` → `:8000`), driven with the repo's
own Playwright from throwaway node scripts, plus direct `POST /api/arch/vacancy-score` and
`POST /api/lex/vacancy` calls against the real backend with real gpt2. **145 real scoring
runs** (110 vacancy-score sweeps + 35 adversarial passages). No server was started, stopped
or restarted; `npm run test:e2e` was not run; no MCP browser tool was used. Nothing was
fixed. Every mutation made for a teeth-check was reverted and `git status` verified clean.

Counts: **4 high**, **7 medium**, **5 low**.

Verdict per charter item: **1** error bars — PARTIAL (fp32 re-measurement VERIFIED; the
"measured" label REFUTED). **2** negative cost — VERIFIED for every path I could reach.
**3** short/odd passages — PARTIAL (no 500s, no NaN; the diacritic class is only half closed).
**4** provenance — PARTIAL (the reported defect is gone; a new one re-opens it).
**5** `mint` — PARTIAL (backend VERIFIED, static diverges on prototype keys).
**6** ribbon VERIFIED, painter VERIFIED, no twelfth `view.set` path — but the guard is
PARTIAL: nothing registers the work it is supposed to protect.

Findings F3b (consequence 1), F9 and F10 were located by two verification sub-agents working
under the same charter; I reproduced the `in`-operator behaviour, the source lines and the
cross-stack `mint` half of F3b myself before recording them. Everything else in this file I
observed directly.

---

## (a) Defects that SURVIVE the fix

### F1. Both error-bar renderers label the q8 uncertainty **"measured"**, which the tree's own normative document forbids in as many words
**Severity:** high
**Where:** `code/frontend/src/viz/arch/vacancyVerdict.ts:106` (`errorBarTerms`, the headline
cards) and `:115` (`secondaryLine`, the secondary rows). Reached in static mode for
`wrong_content` and `total`, which `code/frontend/src/lib/staticClient/arch.ts:508,533`
attach `quantizationUncertaintyNats: VACANCY_Q8_UNCERTAINTY_NATS` (= 0.2) to.
**Reproduce:** read the two lines; both are unconditional once the stack attached a value.
**Observed** (verbatim, `vacancyVerdict.ts:104-108` and `:112-117`):
```ts
const terms = [`± ${num(d.se)} (sampling, ${d.nPairs.toLocaleString()} paired tokens)`];
if (d.quantizationUncertaintyNats) {
  terms.push(`± ${d.quantizationUncertaintyNats} (quantization, measured)`);
}
```
```ts
const terms = [`± ${num(d.se)} (sampling)`];
if (d.quantizationUncertaintyNats) {
  terms.push(`± ${d.quantizationUncertaintyNats} (quantization, measured)`);
}
```
So the static panel renders, on the two numbers it reports,
`± 0.2 (quantization, measured)`.

Against, verbatim, `specs/007-vacancy-transform-field/architecture.md:875-879` — written by
the reconcile fix agent in this same campaign:
> So `VACANCY_Q8_UNCERTAINTY_NATS = 0.2` is **not currently a like-for-like
> measurement of the shipped configuration**; it is retained only because it exceeds every gap
> ever observed here, and lowering a bound without a measurement is worse. … **Until then, do
> not quote it as "measured on the shipped swap".**

and `notes/agent-reports/fix-007-reconcile.md:319`:
> "the constant is unchanged, its status is now stated in three places, and it **must not be
> described as measured on the shipped swap** until someone runs it."

**Expected:** FR-720a (`specs/007-vacancy-transform-field/spec.md:108-113`) — "a stated ±
that was never measured is a fabricated error bar and is worse than no number". The word
`measured` in the rendered string is the claim FR-720a governs, and it is the one surface a
reader actually sees when they run the scorer. The three places the reconcile agent
corrected (a source docstring, the Info tab's Known limits, and `architecture.md`) are all
*elsewhere*; the panel that prints the number was not among them. This is the campaign's
signature failure mode — a fix asserted as complete that stops one file short of the surface.
**Would it have thrown?** No.

---

### F2. The word-alphabet refusal only covers Unicode **letter** runs, so any non-letter joiner inside a word still produces a silently scored fragment rewrite — including the ordinary curly apostrophe
**Severity:** high
**Where:** `code/backend/src/llm_geometry/arch/vacancy_score.py:120`
(`WORDLIKE_RE = re.compile(r"[^\W\d_]+(?:['\-][^\W\d_]+)*")`) and its mirror
`code/frontend/src/lib/staticClient/byteSpans.ts:139`
(`const WORDLIKE_RE = /\p{L}+(?:['\-]\p{L}+)*/gu;`). Both admit **only** ASCII `'` and `-`
as internal joiners, so a word joined by any other character is two separate "wordlike"
runs, each of which `WORD_RE` matches *entirely* — and `fragmented_words` /
`fragmentedWords` therefore flags nothing.

`notes/agent-reports/fix-007-arch.md` claims F6 closed: *"Both stacks now refuse up front,
naming the word … `café` is the word `caf` to `WORD_RE`, so the transform rewrote a fragment
and the endpoint **returned** the score. That is the plausible-wrong-answer class, not a
crash."* The class is not closed; only its ASCII-letter instance is.

**Reproduce (full stack API, and reproduced identically in the browser):**
```
curl -s -X POST http://localhost:8000/api/arch/vacancy-score -H 'content-type: application/json' \
  -d '{"model_id":"gpt2","p":1.0,"seed":0,"passage":"The cat’s don’t stop and the dog won’t go to the tree in the park today okay."}'
```
**Observed** — HTTP 200, a full score, and the swap variant, verbatim from the rendered
panel at `http://localhost:5173/#architecture`:
```
  wrong_content: 2.406nats
     err: ± 1.309 (sampling, 10 paired tokens)
  unknown_form: -0.585nats
     bound: negative — a cost cannot have a negative upper bound; see below
  SWAP PREVIEW: "The want’s big’t wish and the cat clean’t go to the park in the take away yellow."
```
`don’t` was split into `don` + `t` and rewritten to `big’t`; `won’t` → `clean’t`. Neither
is a word, which is precisely the property `swap` is defined by.

Two more joiners, same result (API, HTTP 200 each):
```
[soft hyphen U+00AD] eng='… the co\xadoperate dog ran …'  swp='… the co\xadwood cat way …'   non='… the co\xadklultidous scarrt …'
[ZWJ U+200D]         eng='The cat‍sat on the mat …'  swp='The want‍wish on the bird …'
```
The *straight* apostrophe is handled correctly (`don't` is preserved intact), so the same
sentence typed two ways gives two different answers, and the wrong one is the one a reader
gets by pasting from a browser, Word, or macOS with smart quotes on.
**Would it have thrown?** No — HTTP 200, plausible numbers, no console error.

---

### F3. `identity` is `p == 0` rather than "the three variants are the same string", so a passage with no vacatable word renders an identity as a measurement, with a false explanation
**Severity:** medium
**Where:** `code/backend/src/llm_geometry/arch/vacancy_score.py:757`
(`identity = float(p) == 0.0`) and its mirror `code/frontend/src/lib/staticClient/arch.ts:922`
(`p === 0`). The F7 fix report says the presentation defect was closed; it was closed only
for the `p = 0` route to it.
**Reproduce (full stack UI):** Architecture → gpt2 → untick "score the shipped corpus
excerpts" → paste `the of and a to in is it you that he was for on are with as I his they be`
→ p = `1`, seed = `0` → Score.
**Observed** (verbatim from the rendered panel; the API confirms
`identity=False identityNote=None` on all three differences and
`identical(eng==swap==nonce)=True`):
```
  wrong_content: 0.000nats
     err: ± 0.000 (sampling, 20 paired tokens)
  unknown_form: 0.000nats
     err: ± 0.000 (sampling, 20 paired tokens)
     bound: upper bound — see below
  total:  nll(nonce) − nll(english) = 0.000 ± 0.000 (sampling)
  VERDICT: … and 0.000 ± 0.000 nats to one that has them — an effect this sample does not
  resolve, because the standard error is more than half of it. Score more text (the pooled
  corpus excerpts above are the measured configuration) before reading anything into the sign.
  SWAP PREVIEW: "the of and a to in is it you that he was for on are with as I his they be"
```
`a a a a a a a a a a` behaves the same way (9 pairs, all zeros, variants identical).
Reachable in the full stack from the first passage a reader types. In the static build the
`VACANCY_MIN_POOLED_PRESERVED = 700` gate makes it very hard to reach (it would need ~700
closed-class tokens), so treat this as a full-stack defect with a latent static twin — the
condition is wrong in both.
**Expected:** every open-class word is closed-class here, so nothing is vacated and all
three variants are one string — the same by-construction zero `p = 0` produces, and the
panel has a correct branch for it that this path never reaches. Three separate statements
are false: the effect is not "unresolved" (`se` is exactly 0, and `0 > 0/2` is false, so the
stated reason is not even self-consistent), "score more text" cannot change it, and the
headline card calls an identity an `upper bound`. The condition should be
`english == swap == nonce`, which is what both stacks' own comments say identity means.
**Would it have thrown?** No.

---

### F3b. Both new string validators use `key in objectLiteral`, so JavaScript's own prototype keys validate — and in the Lexicon Lab that erases every provenance caveat on the page
**Severity:** high
**Where:** `code/frontend/src/viz/lex/provenance.ts:177`
(`if (typeof declared === "string" && declared in PROVENANCES)`, against
`const PROVENANCES: Record<Provenance, true> = {…}` at `:142`) and the identical pattern at
`code/frontend/src/lib/staticClient/lex.ts:500`
(`if (!(typeof mint === "string" && mint in MINT_STRATEGIES))`, against
`const MINT_STRATEGIES: Record<MintStrategy, true> = { nonce: true, swap: true }` at `:535`).
Both are code the F1/F2 fix commits introduced.
**Reproduce:**
```
node -e "const P={untrained:1,trained:1,unrecorded:1,'edited-untrained':1,'edited-trained':1,'edited-unrecorded':1};
for (const k of ['constructor','toString','valueOf','__proto__','hasOwnProperty','isPrototypeOf','banana'])
  console.log(k,'->',k in P)"
```
**Observed:**
```
constructor -> true
toString -> true
valueOf -> true
__proto__ -> true
hasOwnProperty -> true
isPrototypeOf -> true
banana -> false
```
*Consequence 1 (the Lexicon Lab).* A bundle whose `metrics.provenance` is `"constructor"`
returns `{ provenance: "constructor", declared: true }`, which then misses every panel's
`Record<Provenance, string>` lookup. Observed live at `http://localhost:5173/#lexicon` with
a real browser-saved **random-initialization** bundle, verbatim:
```
lex-file-ok:    "loaded case.llmlex.json · 314-word dolch budget (318 rows) · model 7f87bcd524a4… weights + vocabulary verified"
lex-file-claim: "The file describes these weights as function Object() { [native code] }. That is the file's own label…"
lex-spectrum-untrained / -unrecorded / -edited / lex-save-* / lex-forward-*:  ALL null
consoleErrors: []
```
Every untrained warning, every "unrecorded" warning and every edited warning vanishes, under
a line ending in the word `verified`. That is finding F1's original symptom restored by a
different door — the `unrecorded` fallback exists precisely so an unknown file gets a caveat,
and this input routes around it into a state with *no* caveat at all.
*Consequence 2 (`mint`, cross-stack).* `{"mint": "constructor"}` on the same request:
```
backend : 400 {"error":{"type":"InvalidParamError","message":"mint must be one of ['nonce', 'swap'], got 'constructor'","detail":{"mint":"constructor"}}}
static  : passes the wire check at lex.ts:500, is cast `mint as MintStrategy`, and reaches
          lexEngine/vacancy.ts:1098-1099, which throws an UNTYPED
          `Error("vacancy: unknown mint strategy \"constructor\"")`
```
`fix-007-lex-ui.md` states the goal as *"the two stacks answer this request with the same
typed error rather than one 400 and one 200"*; for these six keys they do not.
**Expected:** `Object.prototype.hasOwnProperty.call(...)`, a `Set`, or a
`null`-prototype object. The fix report's own claim — *"validates against a
`Record<MintStrategy, true>` so a third strategy fails to compile here"* — is a
**compile-time** guarantee being relied on for a **runtime** membership test on untrusted
wire data.
**Would it have thrown?** The `mint` case throws (untyped, in the wrong stack). The
provenance case does **not** — it renders a clean, caveat-free page for a random init.
*(Consequence 1 was located by a sub-agent; I reproduced the `in`-operator behaviour and the
source lines myself, and independently found consequence 2.)*

---

### F4. `spec.md` FR-720a and the Info tab still assert the q8 bound as an existing measurement
**Severity:** medium
**Where:** `specs/007-vacancy-transform-field/spec.md:109-110`;
`code/frontend/src/viz/info/InfoTab.svelte:963-965` and `:972-976`;
`code/frontend/src/lib/staticClient/arch.ts:476-477` (docstring).
**Observed**, verbatim:
```
- **FR-720a** … Pooled `nonce − english` and `swap − english` qualify under q8
  (|Δ| ≤ 0.054 nats).
```
```
<b>A pool below 700 preserved tokens: refused</b> — that is the size at which the bound was
measured, and below it the honest answer is no number.
```
```
carries <b>±0.2</b> nats of quantization uncertainty stated beside the sampling standard
error, quoted to one decimal place because that is all the measurement supports.
```
```
 * (contract §8.3a, FR-720a). Pure policy, separated from the measurement so it can be
 * asserted directly: pooled `swap − english` and `nonce − english` carry a stated,
 * MEASURED quantization uncertainty; …
```
**Expected:** the reconcile agent's own Known-limits entry, ~180 lines below the second
quote in the same file, says the opposite — "the `q8` side has not [been re-measured] …
no gap can be computed for the configuration that ships … **0.2 is kept because it is larger
than every gap ever measured here … not because it has been re-derived.**" The `|Δ| ≤ 0.054`
in FR-720a and the "700 preserved tokens" gate were both derived from the pre-rewrite texts
(the current run pools **856**, not ~700). A reader who reads the requirement, or the
sentence above Known limits, is told a bound exists.
**Would it have thrown?** No.

---

### F5. The static client's `swap` + `consistent:false` refusal still quotes the pre-rewrite corpus counts, and now disagrees with the backend's message word for word
**Severity:** medium (a stack divergence the fix pass introduced)
**Where:** `code/frontend/src/lib/staticClient/lex.ts:513-515`.
**Observed** — the same refusal from the two stacks:
```
backend  (POST /api/lex/vacancy, mint=swap, consistent=false):
  400 InvalidParamError: mint='swap' requires consistent=True: the inconsistent control needs
  a fresh type per occurrence and the corpus has 1676 open-class stems against 8125 vacated …

static   (lex.ts:513):
  "mint = 'swap' requires consistent = true — the inconsistent control needs a fresh " +
  "type per occurrence and the corpus has 1680 open-class stems against 8202 vacated " +
```
`code/backend/src/llm_geometry/lex/vacancy.py:864` and
`code/frontend/src/lib/lexEngine/vacancy.ts:1104` were both updated to `1676` / `8125`, and
`InfoTab.svelte:910` says `1,676 stems cannot cover 8,125 vacated tokens`. The static
client's copy was missed. `grep -rn "8202"` over `code/*/src` returns this one line.
**Why nothing caught it:** the parity fixture
`code/frontend/tests/fixtures/vacancy-api-golden.json` has no `rejects` key (its 7 cases are
all 200s), and `tests/unit/staticVacancy.test.ts:277-280` asserts rejects only with
`.rejects.toMatchObject({ type: "InvalidParamError" })` — the message is never compared. A
future divergence of any refusal *message* between the stacks is undetectable by the suite.
**Would it have thrown?** It throws — with the wrong number in it, in only one of two stacks.

---

### F6. The panel's "backwards" paragraph explains the state with a cause that can no longer occur
**Severity:** low
**Where:** `code/frontend/src/viz/arch/VacancyScorePanel.svelte:300-307`.
**Observed** (verbatim):
> That happens at intermediate <code>p</code> and on short passages, where the swap
> variant's own replacements can be rarer than the nonce forms that replace them; it is not
> the measured configuration.

**Expected:** the F7 fix in the same commit made intermediate `p` a typed 400 in both stacks
(verified below), so no reader can ever reach a backwards verdict "at intermediate `p`". The
sentence names an unreachable cause as the primary explanation.
**Would it have thrown?** No.

---

### F7. The `p = 0` identity still prints a sampling error bar and an "upper bound" label on the headline cards
**Severity:** low
**Where:** `VacancyScorePanel.svelte:264-273` — `errorBarTerms(d)` and `upperBoundLabel(d)`
are rendered without consulting `d.identity`; only the verdict paragraph branches on it.
**Observed** (live, p = 0, the red team's passage):
```
  wrong_content: 0.000nats
     err: ± 0.000 (sampling, 29 paired tokens)
  unknown_form: 0.000nats
     bound: upper bound — see below
  VERDICT: … exactly nothing to one that has them either — but by construction, not by
  measurement: At p = 0 no stem is vacated, … not a measurement of anything.
```
**Expected:** `fix-007-arch.md` claims the panel shows *"exactly nothing … but by
construction, not by measurement"* **"instead of `0.000 ± 0.000 nats (sampling, 20 paired
tokens)`"**. The "instead of" is not true — the card still prints it, three lines above the
paragraph that says it is not a measurement. The verdict is right; the number beside it is
still dressed as a measurement.
**Would it have thrown?** No.

---

### F8. None of these fixes are on the deployed site; the original F4 defect is still live
**Severity:** low (deployment state, not code) — recorded because the brief's premise is the
live site.
**Where:** `https://context-lab.com/llm-geometry/assets/index-Bjyi2uBc.js`.
**Observed:** the deployed bundle contains `(sampling, ${…} paired tokens)` **once** and the
post-fix secondary-row form `(sampling)` **zero** times; `cost cannot have a negative upper
bound`, `unrecorded`, `1,940` and `349 / 484 / 364` are all absent. The last Pages deploy
(`gh run list --workflow=pages.yml`) succeeded at `2026-08-04T18:47:13Z` UTC = 14:47 EDT,
before the fix merges (16:35–17:09 EDT).
**Expected:** the live site still renders `nll(nonce) − nll(english) = 0.879 ± 0.074`, the
number the campaign was opened over. The fixes exist only in `main`.
**Would it have thrown?** No.

---

### F9. The provenance sentences are repeated in the tab's own second-person voice, unattributed — the attribution lives only in the Save/load panel
**Severity:** medium (the residual `fix-007-lex-ui.md` F6 admits, confirmed live and wider
than the admission)
**Where:** `code/frontend/src/viz/lex/SamplePanel.svelte:42`,
`SpectrumPanel.svelte:227-256`, `ForwardPassPanel.svelte:236` — none of them mentions the
file. `ModelFile.svelte`'s `lex-file-claim` is the only attributed surface.
**Observed** — a bundle `{"provenance":"trained","trained":false,"edited":false}` over a real
random initialization:
```
sample header: "Generate from the model you trained"
hasNotTrainedPhrase: false;  no spectrum caveat;  no forward-pass caveat
```
**Expected:** the fix report's own words — *"It repeats it as the file's claim, next to the
sentence saying the block is unhashed, which is the most a format whose provenance block is
deliberately outside its digests can honestly do."* That is true of `lex-file-claim`; it is
not true of "**the model you trained**", which is the page asserting it in its own voice, in a
different panel, with no adjacent caveat. `{"provenance":"banana","trained":true}` reaches
the same state through step 2 of `provenanceFromMetrics`.
**Would it have thrown?** No.

---

### F10. A save can mislabel *where* a model was trained
**Severity:** low
**Where:** `code/frontend/src/viz/lex/ModelFile.svelte:74`, `DEFAULT_NOTES["edited-trained"]`.
**Reproduce:** train via `POST /api/lex/train`, `GET /api/lex/model`, load that bundle into
the tab, edit a weight, Save.
**Observed** (verbatim, from the written file):
```json
{"note":"trained in the Lexicon Lab, then hand-edited in the Weight Lab · embed zeroed",
 "provenance":"edited-trained","trained":true,"edited":true}
```
Those weights were trained by the backend route, never in this tab.
**Would it have thrown?** No.

---

### F11. TASK 4's guard is sound, but only three panels ever register — the Lexicon Lab's own vacancy demo runs two real trainings that a tab switch still destroys in silence
**Severity:** high
**Where:** `code/frontend/src/viz/lex/VacancyPanel.svelte:507-509`
(`let demoAbort: AbortController | null = null;` … `onDestroy(() => demoAbort?.abort());`)
with **no** `registerWork` anywhere in the file.
**Reproduce:** `grep -rn "registerWork" code/frontend/src` — verbatim, the complete list:
```
src/viz/lex/TrainPanel.svelte:147:    if (busy) registerWork(WORK_ID, "a training run in the Lexicon Lab");
src/viz/geo/FinetunePanel.svelte:38:  if (busy) registerWork(WORK_ID, "a fine-tuning run in the Geometry Lab");
src/viz/geo/TrainPanel.svelte:57:   if (busy) registerWork(WORK_ID, "a from-scratch training run in the Geometry Lab");
src/lib/stores.ts:73:export function registerWork(id: string, label: string): void {
```
**Observed** live, mid-demo, at `http://localhost:5173/#lexicon`:
```
DURING DEMO:              {"pending":[],"busy":true,"label":"training…"}
AFTER TAB CLICK MID-DEMO: {"bar":false,"onInfo":true,"demoStillThere":false}
```
Two real training runs destroyed, no alertdialog, nothing said.
**Expected:** `fix-007-reconcile.md` TASK 4 argues the guard belongs in the store *"rather
than in the eleven `view.set` call sites deliberately: with it in the call sites, the twelfth
would be the one that lost a run."* The reasoning is right and the guard is right — but the
gap simply moved to the **other** side of the contract. The registry only protects work that
opts in, and the panel that runs the tab's *headline demonstration* does not. The same file
carries two `view.set("architecture")` buttons (`:779`, `:1102`), so the app invites the
click that destroys the work. `viz/arch/ArchitectureExplorer.svelte:238` (`traceCtl?.abort()`)
is unregistered for the same reason.
**Would it have thrown?** No.
*(Located by a sub-agent; I reproduced the `grep` myself — the four lines above are the whole
registration surface, and `VacancyPanel`/`GeometryLab`/`ArchInspector`/`ArchitectureExplorer`
all hold abortable in-flight work outside it.)*

---

### F12. The navigation guard has no `beforeunload`, so a reload or a closed tab still discards a registered run without asking
**Severity:** medium
**Where:** `code/frontend/src/lib/stores.ts` / `App.svelte` — `grep -rn "beforeunload" src/`
returns no hits.
**Observed:** reload with work registered → `native dialogs fired during reload: []`.
**Expected:** the Known-limits entry the same fix added says the app "now holds and names any
such navigation including Back". Reload and close are the two commonest ways a reader leaves,
and they are not covered. (The `Back` half *is* covered — verified.)
**Would it have thrown?** No.

---

### F13. `stores.ts:113` says "eleven `view.set` call sites"; there are thirteen
**Severity:** low
**Where:** `code/frontend/src/lib/stores.ts:113`. Cosmetic — the design argument is unaffected,
and the count is the kind of number this repo pins elsewhere.
**Would it have thrown?** No.

---

## (b) NEW defects the fix introduced

- **F3b** — both `key in objectLiteral` validators are new code from the F1 and F2 fix
  commits. The provenance half is the worse one: it re-opens F1's exact symptom.
- **F5** — the static/backend refusal messages now disagree, and the static one is stale.
  Before the reconcile pass both stacks said `1680` / `8202`; the pass corrected two of the
  three copies.
- No other regression found. The intermediate-`p` refusal, the identity flag, the `mint`
  parse, the fragmented-word check and the negative-cost verdict all behave correctly on the
  paths they do cover, and I found no case where a fix made a previously correct number wrong.

---

## (c) Claims I could NOT verify, and why

1. **The q8 arm of `VACANCY_Q8_UNCERTAINTY_NATS`.** Unchanged from the reconcile agent's own
   position: it needs a real browser running *this* build's static scorer. The deployed
   build is pre-fix (F8), and I may not start a static dev server. I did not compute, guess
   or extrapolate a q8 number.
2. **The `backwards` verdict branch in the live UI.** I hunted it across **110 real scoring
   runs** — 4 passage sets × up to 30 seeds at `p = 1`. Negative `nonce − swap` is common on
   short passages (18 of 30 runs on one set), but only **one** configuration reached the
   panel's `|nats| > 2·se` promotion test: four short passages pooled, seed 5,
   `-0.4141 ± 0.1970` (ratio 2.10) over 48 pairs — and that is reachable only through the
   API's `passages[]` array, which the panel's single-textarea control does not expose. So
   the backwards paragraph is verified at the pure-function level (`verdictKind` returns
   `"backwards"` for the red team's exact `-0.3548 ± 0.1359`) and by reading the template
   (a dedicated `{:else if verdict === "backwards"}` branch that renders no share and no
   conclusion), **not** by seeing it rendered. The adjacent negative-but-unresolved path I
   did verify live, and it suppresses the share and the conclusion correctly.
3. **WebGPU.** No adapter on this machine, as the brief predicts.
4. **Qwen / SmolLM2**, in either stack. Only gpt2 was run.
5. **The static build's panel rendering, end to end.** The deployed site is pre-fix (F8) and I
   may not start a `VITE_DATA_MODE=static` server, so the static arm of F1 and of the F4 fix is
   verified by reading `staticClient/arch.ts` and by the panel's unit tests, not by seeing the
   page. The full-stack arm was driven live at several `p`, seeds and passages.
6. **`vacancyParamsFrom` could not be driven directly** — it is not exported from
   `staticClient/lex.ts`, so the static half of the `mint` matrix is source-read plus the
   `constructor` probe, not a live call.

---

## (d) What I confirmed is genuinely fixed

**The fp32 arm — re-measured independently and reproduced exactly.** A fresh
`POST /api/arch/vacancy-score {"model_id":"gpt2","p":1.0,"seed":0}` against the running
backend returned, verbatim:
```
variants  english 2754 tok / 856 preserved · swap 2766 / 856 · nonce 3792 / 856
wrong_content  nll(swap)  − nll(english) = 0.690417 ± 0.0538944   856 pairs
unknown_form   nll(nonce) − nll(swap)    = 0.287211 ± 0.0449636
total          nll(nonce) − nll(english) = 0.977628 ± 0.0590321
```
— every figure `fix-007-reconcile.md` TASK 3 claims (0.6904 ± 0.0539, 0.9776 ± 0.0590, 856
preserved, 2754/2766/3792). `test_the_fp32_arm_quoted_in_the_static_client` passes against
a real gpt2 run (`1 passed in 7.07s`) and pins all of them.

**F4's arithmetic half.** Every number `VacancyScorePanel.svelte` can render, and the terms it
carries — the charter's "no term it has not earned, every term it is entitled to" audit:

| rendered number | full stack (fp32) | static (q8) |
|-|-|-|
| headline `wrong_content` | `± se (sampling, N paired tokens)`, no quantization term — correct, there is none | `± se (sampling, N) · ± 0.2 (quantization, <b>measured</b>)` — both terms present; **label wrong, F1** |
| headline `unknown_form` | `± se (sampling, N)`; `upper bound` caption, or the negative caption | refused with `VACANCY_UNKNOWN_FORM_REFUSAL`, no number at all — correct |
| secondary `total` | `= x ± se (sampling)` — correct | `= x ± se (sampling) · ± 0.2 (quantization, <b>measured</b>)` — **F4's dropped term is restored**; label wrong (F1) |
| variants table `nllPreserved` / `nllAll` / `bitsPerChar` | real fp32 numbers | `pooledStats` returns `null` for all three → rendered `—`, beside `VACANCY_ABSOLUTE_REFUSAL` — correct |
| variants table `nTokens` / `nPreservedTokens` / `nChars` | counts, no ± claimed — correct | same — correct |
| per-passage table | real per-passage rows | `passages: null` + `VACANCY_PER_PASSAGE_REFUSAL` — correct |
| tiny arm | `0 nats, exactly` / `an identity, not a rounding` — correct | same |
| verdict `unresolved` branch | `{nats} ± {se} nats`, **no** quantization term | unreachable today (static always refuses `unknown_form`), but it would be a fabrication if a q8 `unknown_form` were ever reported — latent |

So: no number in either stack carries a term it did not earn, and after the fix none drops one
it did. `nllPreserved`/`nllAll`/`bitsPerChar` print `—` rather than a q8 number. The remaining
problems with the quantization term are its **label** (F1) and the documents that back it (F4).
The `unresolved` verdict row is the one place a future quantized difference could still print a
bare `± se` — worth closing before anything is added to `VACANCY_MEASURED_DTYPES`.

**Intermediate `p` is refused, in the API and in the UI.** `p ∈ {0.05, 0.5, 0.95}` →
`400 InvalidParamError` citing §5.2a in both; `p ∈ {0, 1}` are accepted. The panel shows the
refusal with "Set p = 1" / "Set p = 0" actions and substitutes nothing.

**A negative cost is no longer dressed as a result** (live, `-0.585 ± 0.543` and
`-0.112 ± 0.361`): the bound caption reads `negative — a cost cannot have a negative upper
bound; see below`, **no** `% of the total damage` is printed, and the closing sentence draws
no conclusion. The `-69%` share and the unconditional finding are gone.

**`p = 0` is labelled an identity** in the verdict paragraph, with the stack's own note.

**The tiny-arm juxtaposition is honest on every path.** In all five live scenarios it
rendered `0nats, exactly` / `an identity, not a rounding`, independent of the pretrained
arm's sign, and the verdict's opening clause ("worth **exactly nothing** to a model trained
from scratch with no lexical entries") is the one claim in that sentence that is always true.
No rendering path was found in which a negative or refused pretrained number changed what the
tiny arm asserts.

**F2/F6 input handling: 35 adversarial passages, zero 500s, zero NaN, zero silent wrong
answers except F2's joiner class.** 1-, 2-, 3- and 5-word passages, `the the`, `Hello world`,
empty, whitespace, tabs/newlines, emoji-only, CJK-only, Arabic, Hebrew, digits, punctuation,
a single character and a 5101-token passage all return typed `400 InvalidParamError`s naming
the actual cause — including the red team's exact five inputs and the new context-length
gate. Diacritics (`a café`, `naïvely`, German, Spanish, Greek, the `ﬁ` ligature) are refused
by name with the `WORD_RE` explanation. Emoji, CJK, Cyrillic and Arabic *mixed into* English
score cleanly and are byte-identical across the three variants, as documented.

**`mint` is honoured, validated and echoed, in both stacks.** Backend: `nonce`/`swap`
produce different text and different `vacated_sha256` (`3c4f7f08…` vs `999504196f2cbcc2…`);
missing defaults to `nonce`; `"bogus"`, `3`, `null`, `[]`, `"SWAP"`, `" swap "` and `true`
all → `400 InvalidParamError: mint must be one of ['nonce', 'swap'], got …`;
`swap`+`consistent:false` and `swap`+`p=0.5` are refused. `staticClient/lex.ts:499-517`
mirrors the same checks in the same order (source-read; the function is not exported, so I
could not drive it directly — the message text differs, see F5).

**Teeth checks — every mutation I tried made the new tests fail.**

| mutation | test | result |
|-|-|-|
| `secondaryLine` drops the quantization term | `archVacancyPanel.test.ts` | `1 failed` — `expect(line).toContain("± 0.2 (quantization, measured)")` |
| `verdictKind` returns `"conclusion"` for a negative | `archVacancyPanel.test.ts` | `1 failed` — `expected 'conclusion' to be 'backwards'` |
| `_vacancy_params` drops `mint=` | `tests/contract/test_api_lex.py -k vacancy` | `4 failed` (knob test, echo test, reject test, golden fixture) |
| `_vacancy_params` drops `reveal_after=` | same | `test_vacancy_params_reads_every_knob_the_transform_declares` `1 failed` |
| `provenanceFromMetrics` always returns `trained` | `lexProvenance.test.ts` | `3 failed / 4 passed` |
| `ModelFile.svelte` passes a literal `"trained"` to `onLoaded` | `lexProvenance.test.ts` | `3 failed / 4 passed` |

The last row answers the brief's question directly: the "reads every knob" test really does
catch a *different* dropped parameter, not just the one it was written for.

**Lexicon provenance (F1/F1b) — the fix holds on every path except F3b and F9.** Verified
live at `http://localhost:5173/#lexicon`:

- the red team's original `{"provenance":"untrained","trained":false,"edited":false}` file →
  `lex-spectrum-untrained` ("This model has not been trained…"), `lex-save-untrained` and
  `lex-forward-untrained` all present, and `"from the model you trained"` absent. **The
  reported defect is gone.**
- no `metrics` block, and `metrics` without `provenance`/`trained` → the new `unrecorded`
  state, claim line: *"The file does not record whether these weights were ever trained, so
  this page does not say either."*
- contradictory metrics resolve conservatively: `{"trained":false}` beside a real
  `first_loss`/`final_loss`/`steps` curve stays **untrained** (a loss curve is not promoted
  to evidence); `{"provenance":"untrained","trained":true}` stays untrained.
- `provenance` of `"banana"`, `123` and `null` → `unrecorded`, correctly.
- a real **backend**-written bundle (20-step train, `metrics.provenance: "trained"`) loads and
  reads trained; a **browser**-written bundle round-trips as untrained.
- **F1b** reproduced with the corpus fetch blocked until after the load: `data-ready=0` at
  load time, and `lex-file-ok` plus the `"trained weights"` badge are identical before and
  after the fetch lands. The model is no longer silently retired.
- load(trained) → train → load(untrained file) correctly drops to `"random init weights"`;
  load → edit → save → load keeps `edited` (`lex-spectrum-edited`, badge *"the loaded file
  (trained, then hand-edited) weights"*).

All mutations were reverted; `git status --porcelain` shows no modification under `code/`.

**Reconcile TASK 2 (the ribbon) and TASK 4's guard mechanism — both hold under attack.**

- **The ribbon (CLAIM A): no counterexample in 20 configurations.** Real browser, real corpus
  (`{"chars":86408,"title":"The Real Mother Goose"}`), `mint ∈ {nonce, swap}` × `seed ∈ {0,7}`
  × `p ∈ {0, .25, .5, .75, 1}`; for all 8 rows × 5 cells of each, `u`, the image and a real
  `transformWord(witness, map, params)` on a corpus witness were recomputed independently.
  Every configuration printed `FINDINGS: []`, every cell `match=true realMatch=ok`, 8/8 rows
  kept, and the header switches between `["stem",…]` and `["type",…]` as it should. The
  original bug is confirmed to have been real: `u_type_towards = 0.447` vs
  `u_stem_toward = 0.794`, so the old hash *would* have shown `towards` vacated at `p = 0.5`
  where the transform does not. Under `swap` at intermediate `p` the engine's refusal renders
  verbatim **and** the ribbon still shows correct real swap forms (`ride→hill`,
  `lanes→takes`, `highness→neighbors`).
- **The corpus painter (CLAIM B): correct, and the search for more was exhaustive.** Sweeping
  all 2 211 corpus types for `isEligible(stem) ≠ isVacatable(word)` yields exactly four —
  `after(aft)`, `always(alway)`, `does(doe)`, `this(thi)` — and paging every corpus window at
  `p = 1` renders all four `kept`. (`during`, `having` and `unless` are not in the corpus at
  all, so the report's "seven" is the closed-class count, not the rendered one.)
- **No twelfth `view` path exists.** `current` is module-private; the only writers are
  `view.set` (guarded), `view.update` (routes through `view.set`), `syncFromUrl` (guarded) and
  `confirmNavigation` (intentional). Driven against the app's *actual* store instance:
  tab click, info-pointer, `view.update`, `location.hash =`, `goBack`, and a synthetic
  `popstate` after both `pushState` and `replaceState` all produced
  `bar=true role=alertdialog` with the view unchanged; Stay and Discard both behave. The
  fixer's structural argument is sound — what is missing is registration, not interception
  (F11).
- **Teeth, 4 more mutations, 4 caught** (baseline 19/19 green): `keysAreTypes → false` →
  2 failed (`expected 'stem' to be 'type'`); `isVacatable` reverted to the stem predicate →
  1 failed (`'after' is painted open`); unguarded `view.set` → 4 failed; the `popstate` hold
  removed → 1 failed.

**Repo hygiene note, not a finding against these fixes:** an untracked
`code/frontend/tests/unit/zzverify.test.ts` (header: *"TEMPORARY verification probe — delete
after the run"*, importing `src/lib/geoEngine`) appeared in the working tree at 17:19 during
this session. It is not mine and not either Lexicon/Architecture sub-agent's — it belongs to a
concurrent GeoEngine agent, so I left it rather than deleting another agent's in-flight file.
Whoever closes this campaign should confirm it is gone.
