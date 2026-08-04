# Verification V3 — the fixes for red-team D (Info tab, shell, docs, a11y)

Independent verification of `notes/agent-reports/fix-007-docs-shell.md` and
`fix-007-reconcile.md` against `notes/agent-reports/redteam-007-docs-shell.md`, under
`notes/2026-08-04-redteam-brief.md`. I wrote none of this code and took no number off any
report — every figure below was re-derived here, on `main` at `33c5ce6`.

Method: the running local stack (`:5173` → `:8000`, never started/stopped/restarted), my
own Node scripts against the repo's installed Playwright (scratchpad, since deleted), the
backend venv (`code/backend/.venv`) driving the real `llm_geometry.lex.vacancy` /
`arch.vacancy_score` code, and `git diff` on the fix commits. `npm run test:e2e` was NOT
run; no MCP browser tools were used. Nothing was fixed.

**Counts: 0 critical · 1 high · 4 medium · 9 low.**

Headline: the shell fixes (F1, F5–F9) are real and hold up under a real browser and a real
keyboard. The number audit found **one surviving pre-rewrite pair in a user-facing string
on the path the public site runs**, and the "corrections" to `README.md` / `CLAUDE.md`
introduced two new inaccuracies of their own.

---

## (a) Defects surviving

### V1. The static build's `swap + consistent=false` refusal still quotes the pre-rewrite counts
**Severity:** high
**Where:** `code/frontend/src/lib/staticClient/lex.ts:512-517`
**Reproduce:** `rg -n "open-class stems against" code specs`
**Observed:** four copies of one sentence; three were updated by the rewrite and one was not.
```
code/backend/src/llm_geometry/lex/vacancy.py:864 : "…the corpus has 1676 open-class stems against 8125 "
code/frontend/src/lib/lexEngine/vacancy.ts:1104  : "…the corpus has 1676 open-class stems against 8125 "
code/frontend/src/lib/staticClient/lex.ts:514    : "…the corpus has 1680 open-class stems against 8202 vacated "
specs/006-lexicon-lab-tiny/contracts/api-lex.md:207: "(§8.3: 1 680 open-class stems against 8 202 vacated"
```
Measured here on the shipped corpus through the real engine (and confirmed against
`POST /api/lex/vacancy {p:1,seed:0}` on the running backend):
```
stemsTotal 1676   tokensVacated(p=1) 8125   domainTypesEligible 1940
```
**Expected:** 1676 / 8125. `staticClient/lex.ts` is the **static** wire boundary — i.e. the
code path the deployed site runs — so this is the copy a public user is most likely to be
shown, and it is the one copy that is wrong. `lexEngine/vacancy.ts` (which the same build
also loads) disagrees with it by 4 stems and 77 tokens.
**Would it have thrown?** No. It *is* the thrown error's text; the throw is correct and its
justification is two stale numbers.

### V2. `swap_pools`'s docstring quotes `195 of 776` as a present-tense measurement of an engine that no longer exists
**Severity:** low
**Where:** `code/backend/src/llm_geometry/lex/vacancy.py:622-625`
**Observed:**
> "Measured over the six shipped Architecture passages at ``p = 1, seed = 0``: 195 of 776
> vacated words (25.1%) had a swap form absent from ``/usr/share/dict/words``."

Re-measured here on the shipped engine, same passages, same `p` and seed:
```
passages 6   words 1521   vacated 767   swap forms outside the passage's own domain: 0
```
**Expected:** 776 is not the current vacated count (767 is), and `fix-007-reconcile.md`
itself lists this figure as one it could not reproduce. The paragraph is *about* the first
implementation, but the sentence is written in the present tense with no marker that the
number belongs to code that was deleted — which is exactly the failure mode the campaign
is chasing. Contrast `staticClient/arch.ts:149-203`, which does mark its superseded numbers.
**Would it have thrown?** No.

### V3. `VacancyScorePanel` quotes the pre-rewrite float32 value as "the float32 truth"
**Severity:** low
**Where:** `code/frontend/src/viz/arch/VacancyScorePanel.svelte:361-364`
**Observed:**
> "the interval it stated, [0.805, 0.953], excludes the float32 truth **0.9892**."

`staticClient/arch.ts:184` records that the same quantity was re-measured after the rewrite
as `nonce − english : fp32 0.9776 ± 0.0590`, and marks 0.9892 as belonging to texts that
no longer exist. This comment carries no such marker.
**Would it have thrown?** No.

### V4. The three `role="tablist"` groups the fixer left alone expose state but not the pattern
**Severity:** medium (issue #7)
**Where:** `viz/geo/TrainPanel.svelte:230-234`, `viz/geo/FinetunePanel.svelte:154-163`,
`viz/geo/AttentionView.svelte:23-25`
**Reproduce:** my `a11y2.mjs` — real Chromium, Geometry Lab loaded to `data-ready="1"`,
DOM walk of every `[role=tablist]`.
**Observed:** the fixer's stated reason for leaving them — *"They do at least set
`role="tab"` + `aria-selected`, so the state is exposed"* — is **TRUE**. What the live DOM
also shows:
```
{"kids":[{"t":"paste text","role":"tab","sel":"true","ti":0,"ctl":null}, …],"panels":0}
{"kids":[{"t":"layer 0","role":"tab","sel":"true","ti":0,"ctl":null}, …],"panels":0}
```
`document.querySelectorAll('[role=tabpanel]').length === 0` on the whole page; every tab
carries `tabindex 0` (no roving tabindex, so each group is N tab stops); `aria-controls` is
`null` everywhere; and `rg -n "onkeydown|keydown"` over all three files returns **nothing**,
so there is no arrow-key navigation.
**Expected:** the `tablist` role is a promise of APG keyboard behaviour that is not
implemented, and a `tablist` with no `tabpanel` is an incomplete ARIA 1.2 pattern. The
fixer's characterisation is accurate; the defect is unfixed, not misdescribed.
**Would it have thrown?** No.

### V5. `dolch.py` points a reader at a test file that does not exist
**Severity:** low
**Where:** `code/backend/src/llm_geometry/lex/dolch.py:13` and `:32`
**Observed:** line 13 — *"``test_dolch.py`` pins both the presence of ``going`` and the
absence of ``giving``"*; line 32 — *"Counts here are MEASURED by `test_dolch.py`, never
hard-coded prose"*. `find . -name 'test_*dolch*'` returns nothing; the real pins live in
`code/backend/tests/unit/test_lex_vocab.py`, which at `:50` **does** hard-code
`{pre_primer: 40 … full: 314}`. The TypeScript port cites a file that exists.
**Would it have thrown?** No.

### V6. `specs/006` contract carries the same stale pair as V1
**Severity:** medium
**Where:** `specs/006-lexicon-lab-tiny/contracts/api-lex.md:207`
**Observed:** *"(§8.3: 1 680 open-class stems against 8 202 vacated tokens …)"*
**Expected:** 1 676 / 8 125. This is a contract document; `specs/007/architecture.md:797`
already carries the corrected pair, so the two specs now disagree.
**Would it have thrown?** No.

---

## (b) NEW defects introduced by the fixes

### V7. `README.md` now states the Geometry vocabulary in the wrong units — the "correction" regressed it
**Severity:** medium
**Where:** `README.md:16`; introduced by `f130828`
**Reproduce:** `git diff f130828^..f130828 -- README.md`
**Observed:**
```
-   ~1000-word vocab) really trained on Alice in Wonderland, …
+   1003-word vocab) really trained on Alice in Wonderland, …
```
`geo/config.py:22-23` — `VOCAB_WORDS = 1000  # word/punctuation types drawn from the
corpus` and `VOCAB_SIZE = 1003  # VOCAB_WORDS + the three specials below`. The app's own
prose keeps the two apart: `GeometryLab.svelte:324-325` says *"1000-word vocab"* and *"Its
1003 token embeddings"*; `TokenStrip.svelte:21` says *"not in the 1000-word vocabulary"*.
**Expected:** the vocabulary is 1000 **words** in 1003 **rows**. `fix-007-docs-shell.md`
justifies the change as *"the GeoTransformer's vocabulary is **1003**, not '~1000'"*, and
cites `docs.spec.ts:97` asserting `spec.model.vocab_size === 1003` — which is the row count,
not the word count. The pre-fix text was closer to correct than the post-fix text.
**Would it have thrown?** No.

### V8. The new "Which spec is which" table omits feature 003, which the same file references
**Severity:** low
**Where:** `CLAUDE.md:145-152` (the table is new in `f130828`)
**Observed:** the table lists 007, 006, 005, 004, 002, 001 and is introduced as the map of
the specs. `ls specs/` returns **seven** directories including `003-static-pages-site`, and
`CLAUDE.md:61` itself reads *"**Static build** (feature 003)"*.
**Would it have thrown?** No.

### V9. "any navigation away from a running tab is held … including browser **Back**" is false for the Back that actually loses the run
**Severity:** medium
**Where:** `code/frontend/src/viz/info/InfoTab.svelte:1178-1186` (Known limits, added by
`33c5ce6`); mechanism at `lib/stores.ts:172-194`
**Reproduce:** `navhold.mjs` — cold-load `http://localhost:5173/#lexicon` in a fresh tab,
click **Train from scratch**, wait ~9 s, press browser Back.
**Observed:**
```
running: step 84/400 · loss 3.314 · lr 2.38e-3 · 9.0s
-- browser Back while running --
after Back: lex-view= 0 url= about:blank hold= 0
```
The run was destroyed with no prompt and no `beforeunload`. The hold only exists in
`popstate`, which does not fire for a traversal that leaves the document — the exact case
for a reader who opened the deployed `#lexicon` link and then pressed Back. The in-document
case **does** work (`navhold2.mjs`: land on `/`, click Lexicon, train, Back → hold shown,
`history.length` unchanged at 3, *Stay* keeps the run, *Discard* really ends it).
**Expected:** either the sentence is qualified, or a `beforeunload` covers the leaving case.
As written the page promises a guarantee the app does not keep.
**Would it have thrown?** No.

### V10. The one-entry-per-navigation guard is dead code, and the test the fix report says pins it does not
**Severity:** low
**Where:** `code/frontend/src/lib/stores.ts:164`
**Observed:** mutation M2 — deleting `if (window.location.hash !== target)` and pushing
unconditionally leaves `shell.test.ts` **green**, including
`does not push an entry for re-selecting the tab already showing`. Svelte's `writable`
already skips a `set` to an equal string, so the subscriber never re-fires and the guard is
never reached on any path the tests exercise. (The *behaviour* is nonetheless correct in a
real browser: `history.length` was 3 before and 3 after three re-clicks of the showing tab.)
**Expected:** `fix-007-docs-shell.md` claims *"That is pinned by its own test, so the guard
cannot be dropped as a 'simplification'."* **Refuted** — it can be dropped silently.
**Would it have thrown?** No.

### V11. Neither history listener is individually pinned
**Severity:** low
**Where:** `code/frontend/src/lib/stores.ts:195-196`
**Observed:** mutation M3 — deleting `window.addEventListener("popstate", syncFromUrl)`
leaves all 30 shell tests green; deleting `hashchange` instead is also green. In jsdom with
hash-only URLs the two are mutually redundant, so the Back/Forward tests do not distinguish
the listener the fix actually added.
**Would it have thrown?** No.

### V12. FR-706's replacement wording over-scopes a purity claim the code scopes to `nonce`
**Severity:** low
**Where:** `specs/007-vacancy-transform-field/spec.md:58-59`
**Observed:** the new text ends *"so the map stays a pure function of `(domain, seed,
match_prosody)`."* Under `mint="swap"` it is not: `build_vacancy_map` raises
`InvalidParamError` without `counts` (`vacancy.py:1181-1186`) and `_assign_swap_class(pool,
seed, match_prosody)` permutes a **frequency-ranked** pool. `vacancy.py:1173-1174` scopes
the identical sentence to *"the **nonce** map"*; FR-706 drops the scope.
**What is right:** there is genuinely **no** `avoid` parameter in any signature in either
stack (grepped), so the substance of the fix stands. This is a scoping error in the new
wording, not a re-introduced API.
**Would it have thrown?** No.

### V13. "the published list has `going`" is qualified in three places and left bare in the Info tab
**Severity:** low
**Where:** `code/frontend/src/viz/info/InfoTab.svelte:620-622`
**Observed:** *"a transcription slip in the first-grade list, which had `giving` where the
**published list** has `going`"* — with no qualifier, in the paragraph immediately after the
section that establishes the grade split is **not** in the 1936 article. `dolch.py:10` and
`lexEngine/dolch.ts:23-24` both add the qualifier explicitly ("refers to that conventional
graded list, not to the 1936 article"). Same bare phrasing at `LexiconLab.svelte:630`.
**Would it have thrown?** No.

### V14. `README.md` documents a lint command that cannot run in the environment it documents
**Severity:** low
**Where:** `README.md:91` against `README.md:53`
**Observed:** the venv is built with `pip install -e ".[test]"`; `pyproject.toml`
`[project.optional-dependencies] test = ["pytest>=8.1","pytest-asyncio>=0.23","httpx>=0.27"]`
and `requirements.txt` contains neither `ruff` nor `black` (they are in `requirements.lock`
only). `ruff check src/ tests/ && black --check src/ && pytest -q` is command-not-found in
that venv. It also drifts from CI (`ci.yml:73` runs `black --check src/ tests/`).
**Would it have thrown?** Yes, at the shell — but the reader is told it is the pre-push gate.

### V15. `CLAUDE.md` carries two statements that were already false before this round
**Severity:** low
**Where:** `CLAUDE.md:88`, `CLAUDE.md:96`
**Observed:** *"`speckit-constitution` | Fill the project principles (currently template
placeholders)"* — `.specify/memory/constitution.md` is fully written (`**Version**: 2.0.0 |
**Ratified**: 2026-06-14`, real Core Principles / Governance sections, no placeholder
tokens). And *"The `specs/` directory does not exist until the first feature is
specified"* — it exists with seven features. Reported per brief rule 2 rather than dismissed.
**Would it have thrown?** No.

---

## (c) Unverifiable from here

1. **The q8 arm of `VACANCY_Q8_UNCERTAINTY_NATS`.** Unchanged since the reconcile report
   said so; it needs a real browser run of the static scorer. I did not run it and I did not
   attempt to derive a |Δ|. The three places that now disclose this
   (`staticClient/arch.ts:149-203`, InfoTab Known limits, `architecture.md` §8.3a) are
   consistent with each other and with the fp32 numbers I found pinned by
   `test_the_fp32_arm_quoted_in_the_static_client`. **I did not run that test.**
2. **`lex/config.py:84-86`** — held-out loss `2.333` / `2.294` / `2.258` at 400/500 steps.
   Each needs a real ~50 s training run; I did not do them. **I don't know** whether they
   are current.
3. **The six e2e cases** (four added by the docs/shell agent, two comment-edited by the
   reconcile agent) remain unexecuted — the charter forbids `npm run test:e2e`. I
   reproduced every behaviour they assert with my own Playwright scripts and all passed, so
   the *behaviour* is verified and the *Playwright wiring* is not.
4. **The 1936 article's contents.** I did not obtain the PDF. I verified only that no text
   anywhere in the repo still credits it with the graded sublists, and that the surviving
   wording is internally consistent. Where the grade split was first published remains
   unknown, and every place that mentions it now says so.
5. **`0.333` vs a nonsense corpus's `0.346`** rests on `notes/agent-reports/006-source-eval.md:187`
   (`"meter_anapest": 0.34639…`) rather than on a source constant. Same disposition as the
   original red team: substantiated by a tracked measurement, not re-derivable from `src/`.

---

## (d) Confirmed fixed

### The numbers — every claim re-derived here, on the current code

Computed from `code/backend/.venv` driving the real modules, and cross-checked against the
running backend. Not read off any report.

| Claim, and where it appears | Re-derived here | ✓ |
|-|-|-|
| domain **2,233** / corpus **2,211** / eligible **1,940** / corpus-eligible **1,918** / stems **1,676** / vacated **8,125** of **16,000** (`InfoTab:767-771`, `:910`) | `POST /api/lex/vacancy {p:1,seed:0}` at 5 values of `p`, and independently through `vacancy_domain`/`build_vacancy_map`/`vacate_text` | ✓ |
| swap loses **349 / 484 / 364** slots at `p=.25/.5/.75`, **0** at both endpoints, seed 0 (`InfoTab:900-903`) | `{vmap.apply_word(t,params).lower() for t in domain}`, computed independently of the test that pins it: seed 0 → `0/349/484/364/0`; seed 7 → `0/336/475/372/0` | ✓ |
| suffix-class sizes **1,280 / 216 / 106 / 92 / 81 / 67 / 53 / 29 / 8 / 8** summing to 1,940 (`InfoTab:867-869`) | `swap_pools()` exactly, total 1940 = `domainTypesEligible` | ✓ |
| "**0** of **767** … every swap form is a type of that passage's own domain" (`InfoTab:883`) | 6 passages, 1521 words, **767** vacated, **0** images outside the passage's own domain | ✓ |
| "five of the six default passages have [a singleton class]" (`InfoTab:891`) | measured pre-fold: 5 of 6, with `ing` the singleton in 3 — matching `swap_pools`'s own docstring | ✓ |
| "smallest class holds 8" on the shipped corpus | `est` 8, `ies` 8 | ✓ |
| **137** function words · **11** suffixes · **61** stress entries · **5.1%** table coverage | `len(FUNCTION_WORDS)==137`, `SUFFIXES` is 11 in that order, `len(STRESS_TABLE)==61`, `stressFromTableBefore = 0.0514375` | ✓ |
| corpus **110,445** bytes / `d514f0fd2cd4…`; **86,408** bytes / `03769632905e…`; **16,000** word tokens; `going` **27**× | file size + sha256, `trim_gutenberg` body, `WORD_RE.findall` | ✓ |
| **70.7%** vs **60.8%**, **39.2%**, **215 of 3,071**, **9,726 of 16,000**, out of **2,211** | `POST /api/lex/coverage`: `0.70675` / `0.607875` / `0.392125` / `215` / `3071` / `9726` / `2211` | ✓ |
| Dolch **40 / 92 / 133 / 220 / 314**, **95** nouns, **318** rows | `dolch_sizes()` exactly; `rows 318` from the API | ✓ |
| `min(\|V\|−1, d)`, **57%** / **42%** / **95%** / **17%** of the model | `param_count`: `57.1 / 41.7 / 94.6 / 16.6` at the stated configurations | ✓ |
| **1.5B** cap, **20%** margin → **1.2B**; **4096** cells; **64×64**; last **64** tokens; **128** new tokens; top-k **50** ∩ top-p **0.9**; penalty **1.1**; **400 ms**; trace prompt capped at **12** tokens | `ARCH_*` in `config.py`; `gate.py:67` `ARCH_MAX_PARAMS * 0.8`; `graph.py:32-33` | ✓ |
| **4** layers, **1** head, MLP **12**, ctx **50**, vocab **1003**, `<unk>=0 <eos>=1 <pad>=2`; **30** epochs, lr **2e-2**, batch **64**, stride **10**, repulsion **0.3**; **≥0.80** / **≥2.0** nats, `ln 64 ≈ 4.16`, `ln 1003 ≈ 6.91`; fine-tune ≤**500**/**100**/**1e-2**, refuse >**90%** `<unk>`; scratch needs **1000** types | `geo/config.py`, `finetune.py:57` `FINETUNE_MAX_UNK_RATE = 0.9`, `scratch.py:137` | ✓ |
| "entropy **3.28** vs **2.81**, coverage **0.988** vs **0.900**" | `geo/scratch.py:65`, verbatim | ✓ |
| **59°**, **90th-percentile** scaling | `fields.py:153`, `GeoScene.svelte` | ✓ |
| **±0.2** nats, pool floor **700**, and the gaps **0.054 / 0.073 / 0.110** | `staticClient/arch.ts:203`, `:219`, `:149-203` — the Info tab's Known-limits paragraph is a faithful transcription of that docstring, including the "not re-derived" disclosure | ✓ |
| eps **1e-5** equivalent, `N(0, 0.02²)`, `2 heads of width 32`, `32 × 64` positions, `12d²+13d` breakdown | `lex/model.py:69` `INIT_STD = 0.02`; `lex/config.py` `DEFAULT_D_MODEL=64 / N_HEADS=2 / CTX=32`; `param_count` docstring verified against 4 configurations | ✓ |
| `lex/config.py:61` states **19,071** tokens (F13) | 16,000 word tokens + 3,071 `<eos>` | ✓ |

**No stale `1,944` / `1,680` / `8,202` / `244 / 322 / 233` survives anywhere in `code/*/src`
or `specs/007`** — except the two occurrences reported as V1 and V6.

### The equations and the transform's mechanism (item 2)

Re-checked line by line against the code as it now stands, not against the prior audit.

- `InfoTab:271-278` vs `geo/model.py::_run` (105-114) and `readout` (131-132):
  `q = h @ W_Q.T` ✓, `scores = q @ k.transpose(1,2) + causal  # unscaled ⟨k_j,q_i⟩` ✓ (the
  "no `1/√d`" claim is verbatim in the source), `(attn @ v) @ W_O.T` ✓,
  `gelu(h @ W_in + b_in) @ W_out + b_out` with `W_in:(3,12)` ✓, `logits = h @ E.T` ✓.
- `InfoTab:312-315` vs `fields.py::next_next_field`, `InfoTab:340-343` and `:399-401` vs
  `force_field`: `w_eff = 0.5 * (w_v - w_v.T) if antisymmetrize else w_v` and
  `vecs = points @ w_eff.T` — per-point field only, aggregate untouched, tangent projection
  anchored at the embedding ✓.
- **The rewritten transform's equations** — `InfoTab:736-741`:
  `[A-Za-z]+(?:['-][A-Za-z]+)*` is `vocab.py:33` `WORD_RE` verbatim; "vacatable iff the
  WORD is open class and its STEM is" is `is_vacatable` (test 1 then `is_eligible`)
  exactly; `u = (sha256("seed:stem")[:8] as uint64 >> 11) / 2**53` is `vacancy_u` character
  for character; "vacate iff u < p" ✓; "too short" = `len(stem) > 2` ✓;
  `good-bye` never moves ✓.
- **The rewritten swap mechanism prose** — `InfoTab:866-893` — matches `swap_pools` +
  `_assign_swap_class`: whole type → whole type, partitioned by the suffix the splitter
  takes off, ranked `(count desc, type asc)`, deranged within the class, a singleton class
  folded into the bare class. **The old "stem replacement + re-attached suffix" description
  is gone**; nothing in `src/` still describes swap as assembling a form. The nonce arm's
  "stem vacated, suffix re-attached, `dog's → <nonce>'s`" is still correct and still applies
  only to `nonce` ✓.

### The shell (items 3 and 4) — driven in a real browser

`shell.mjs` / `hash.mjs` / `a11y.mjs`, Chromium, zero console errors and zero page errors:
```
cold /            url=/            shown=arch-view
click geometry → lexicon → info    /#geometry /#lexicon /#info
Back 1  /#lexicon   Back 2  /#geometry   Forward 1  /#lexicon
Back 3  /#geometry  Back 4  /  (arch-view)
history.length before 3 re-clicks of the showing tab = 3, after = 3
cold #geometry → Back → about:blank        (leaves the site; no trap)
deep links: #architecture #geometry #lexicon #info all resolve
reload mid-history → /#info, then Back → /#lexicon → /#geometry
```
Hash canonicalization:
```
cold #Info → /#info · #INFO → /#info · #LeXiCoN → /#lexicon · #Geometry → /#geometry
cold #not-a-tab → /#architecture (arch-view), and clicking Architecture keeps /#architecture
mid-session #not-a-tab on the Info tab → stays on info, URL left at /#not-a-tab
  then reload → /#architecture (arch-view)      ← docs.spec.ts:51-58's contract, intact
```
**`docs.spec.ts:41-63` was not weakened.** `git diff f130828^..f130828` and
`git diff 33c5ce6^..33c5ce6` on that file show the URL test's body unchanged in both
commits — every hunk is an addition after line 62, plus three comment-only edits at
`:394-412`. I re-ran its exact scenario (`page.goto`, not `location.hash`) and it passes.
One observation, not a defect: a percent-encoded fragment (`#info%20`, what a browser
produces from a typed trailing space) is not canonicalized, because `.trim()` does not
remove `%20`.

### Accessibility (item 5) — driven with a real keyboard and the CDP AX tree

```
#architecture first tab stops: Architecture(cur=page) → Geometry → Lexicon → Info → …
nav: {"role":null,"label":"views","tag":"NAV"}, exactly one aria-current="page", follows clicks
geo-mode  {"role":"radiogroup","label":"field","kids":[{"next-next","radio",checked:true,ti:0},{"force","radio",checked:false,ti:-1}]}
geo-layer {"role":"radiogroup","label":"layer", 5 radios, exactly one ti:0}
ArrowRight → force(ti 0) · ArrowRight → next-next (wraps) · ArrowLeft → force · End → force · Home → next-next
Tab out of geo-mode lands on geo-layer's single roving stop, not on its second option
TOC: "Known limits" → focus H3#limits (tabIndex −1); "Notation" → H3#notation; next Tab lands inside the section (y≈14184)
CDP AX on #architecture: UNNAMED interactive nodes = 0; combobox name = "Model"
```
F5, F6, F7 and F9 are all genuinely fixed, and the roving tabindex is correct including the
disabled-option case. The nav-hold bar is a real `role="alertdialog"` naming the work and
the destination; *Stay* keeps the run and the registration, *Discard* really ends it
(verified by returning to the tab: no step line).

### Provenance (item 6)

No text anywhere in `src/`, `specs/`, `README.md` or `CLAUDE.md` still credits the 1936
article with graded sublists. `dolch.py`, `lexEngine/dolch.ts`, `BudgetPanel.svelte:52`,
`InfoTab` (the budget bullet, "Why 314 and not 315", and the reference entry) all carry the
corrected wording, and the Python and TypeScript word data are byte-identical across all six
lists. `avoid` exists in no signature in either stack. Residue: V12 (FR-706's scope) and V13
(one unqualified "published list").

### Test teeth (mutation testing)

19 mutations applied to the source and reverted; the tree was verified clean afterwards.
**17 of 19 failed the covering test (teeth).** The two survivors are V10 and V11.
`lexVacancyRibbon.test.ts` is fully behavioural — real transform, real map, real mount — and
is the strongest of the three. `shell.test.ts` mounts real components for most of its
assertions but pins the Geometry Lab's two controls by regexing `GeometryLab.svelte`'s
source (a rename or a prop reorder evades it); `navGuard.test.ts` pins all three panels'
registrations the same way. Both files say so in their own comments; noted because
source-text assertions are weaker than they read.

---

## Verdicts

| # | Item | Verdict |
|-|-|-|
| 1 | The numbers, re-derived from scratch | **PARTIAL** — every figure in the Info tab and all four tabs' prose re-derives correctly; two stale copies survive outside the prose (V1 high, V6), plus V2/V3, and the README "correction" regressed one (V7) |
| 2 | Equations and the transform's mechanism | **VERIFIED** — all five equation blocks and the whole swap/nonce mechanism description match the current code |
| 3 | Shell / F1 | **VERIFIED** for tab navigation; V9 is a documentation over-claim about Back, V10/V11 are test-coverage gaps |
| 4 | Hash canonicalization / F8 | **VERIFIED** — both behaviours correct, `docs.spec.ts:54-58` demonstrably not weakened |
| 5 | A11y / F5, F6, F7, F9 | **VERIFIED**; the fixer's claim about the three untouched tablists is **true**, and V4 is what they still lack |
| 6 | Provenance / F11, T5 | **PARTIAL** — the false claim is gone everywhere; V12 and V13 remain |
| 7 | README / CLAUDE.md | **PARTIAL** — four tabs and 007-current are accurate and every referenced path exists, but V7, V8, V14, V15 |

No secrets, no scratch files and no screenshots left behind; the scratchpad was deleted and
the working tree is clean.
