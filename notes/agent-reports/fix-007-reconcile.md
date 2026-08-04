# Fix agent — RECONCILE (the leftovers of the 007 red-team campaign)

Date: 2026-08-04. Main checkout, branch `main`, after all five fix branches merged.
No server was started, stopped or restarted; `npm run test:e2e` was not run.

Answering the "needs the owning agents" list of `fix-007-vacancy-core.md`, F11/F12 of
`fix-007-docs-shell.md`, and F2 of `redteam-007-docs-shell.md`.

**Every number below was re-derived here, on the code as it now stands.** Where a
measurement was impossible I say so rather than transcribing one.

---

## TASK 1 — the stale numbers on screen · DONE

### What I measured, and against what I was told

Run on the shipped corpus through the real engine (`load_corpus_text` → `trim_gutenberg` →
`tokenize` → `vacancy_domain` / `build_vacancy_map` / `vacate_text` / `vacancy_stats`):

```
bytes 86408   word tokens 16000
domainTypesTotal = 2233   corpusTypesTotal = 2211
domainTypesEligible = 1940   corpusTypesEligible = 1918
stemsTotal = 1676   tokensVacated (p=1) = 8125
```

Lost image slots, `|domain| − |{T_p(t)}|`, computed independently of the test that pins it:

```
seed 0  p = 0 / .25 / .5 / .75 / 1 :  0 / 349 / 484 / 364 / 0
seed 7                             :  0 / 336 / 475 / 372 / 0
```

Swap pools, one per suffix class, `swap_pools(domain, counts, keep_set)`:

```
''  1280 · s 216 · ed 106 · er 92 · ing 81 · 's 67 · es 53 · ly 29 · est 8 · ies 8
total 1940  (= domainTypesEligible, as it must be)
```

**Every figure my charter gave me reproduced exactly.** 1,940 / 1,676 / 8,125 and
349 / 484 / 364. Nothing to correct in the brief.

Two figures I was *not* given and derived myself, because the sentence I was rewriting
needed them and the ones in `vacancy.py`'s docstring (195 of 776, 25.1 %) were measured on
the pre-fix engine and are not re-derivable:

```
six default Architecture passages, p = 1, seed = 0
  words 1521   vacated 767   swap forms outside the passage's own domain: 0
  passages with a singleton suffix class: 5 of 6
```

So the Info tab now states the guarantee as a number I measured (**0** of **767**) and the
old defect as "about a quarter", attributed to before the rewrite, rather than repeating a
count I cannot reproduce.

### Files changed

| file | change |
|-|-|
| `viz/info/InfoTab.svelte` | 1,944→**1,940**, 1,680→**1,676**, 8,202→**8,125** (twice); 244/322/233→**349/484/364** plus the counting definition and the seed |
| `viz/lex/LexiconLab.svelte` | the same pair in the `vacBuild` comment: 1 676 / 8 125 |
| `arch/vacancy_score.py` | the `swap` variant's one-line description |
| `viz/arch/VacancyScorePanel.svelte` | the same sentence in its module docstring |
| `viz/lex/VacancyPanel.svelte` | the mint explainer and the `swap` radio's tooltip |
| `tests/e2e/docs.spec.ts` | four comments beside assertions that read the live API |

### The now-false "doubly-inflected replacement" sentence

Deleted and replaced. It said swap's pool was "**1,944** eligible domain types against the
map's **1,680** stems" and that "a source type carrying a suffix may receive an already
inflected replacement". Neither half survives: the map is no longer keyed on stems, and
**nothing is assembled**, so a doubly-inflected surface cannot exist. The replacement prose
states what the transform does now — a type→type derangement inside one suffix class, the
ten class sizes above, the frequency ranking, and the two honest consequences (a per-class
pool whose tail degrades the frequency match; a singleton class folded into the bare class,
which happens on passages and not on the shipped corpus, whose smallest class holds 8).

### One thing I changed that was not on the list

The eligibility equation block and its paragraph described only the stem test. The
whole-word closed-class test is what stopped `after → kitser`, and it is the reason the §8
measurement is taken over words that really survive, so it is now stated on the page.

---

## TASK 2 — `VacancyPanel`'s ribbon read types as stems · DONE, with a real test

**The defect, reproduced.** `ribbon` did
`[...map.mapping.keys()] … .map((stem) => ({ stem, u: vacancyU(stem, seed) … }))`.
Under `mint = "swap"` those keys are **types**, so the panel hashed the wrong string. The
filter `corpusStems.has(s)` compared type keys against a set of *stems*, so every inflected
type was silently dropped and the surviving rows were mostly Dolch words that happen to
equal their own stem.

**Reproducing test:** `code/frontend/tests/unit/lexVacancyRibbon.test.ts` (7 cases; real
transform, real map, real component mounted in jsdom).

Before (the shipped code, with the fix's `hashedStem` reverted to the identity it was):

```
× prints the u the transform actually used, under mint = "swap"
  → u printed for 'planted': expected '0.008' to be '0.101'
× shows each row's own replacement and nests it, under mint = "swap"
  → 'towards' at p = 0.25: expected 'puddings' to be 'towards'
```

That second line is the whole defect class in one assertion: a word the transform does
**not** vacate at `p = 0.25`, shown as vacated, with a replacement, and nothing thrown.

Before (the filter and the header, on HEAD):

```
× keeps inflected types as rows under swap instead of dropping them
  → no inflected type among ["told","orchard","garden","want","bread","sing","pick","ran"]
× names its rows for what they are: stems under nonce, types under swap
  → expected 'stem' to be 'type'
```

After: `7 passed`.

**The fix** branches on `map.mint` rather than relabelling: `keysAreTypes`, a `hashedStem`
that splits a swap key and leaves a nonce key alone (`water` must not become `wat`), a
`corpusKeys` filter built in the same units as the keys, and a header/caption that name
what the rows are.

### A second defect in the same file, found while fixing the first

`segmentsOf` classified a word with `isEligible(stemAndSuffix(word)[0])` — the *stem-level*
predicate. The splitter breaks the closed class open, so the seven function words
`after · always · does · during · having · this · unless` were painted "**open class, not
yet vacated**" in a panel whose legend says the colours come from the real map. They are
never vacated at any `p`. Now `isVacatable(word)`, with the legend saying why.

Test: `paints a function word the splitter breaks open as preserved, not as open class`.
Before: `'after' is painted open: expected 'open' to be 'kept'`. After: passes.

---

## TASK 3 — `VACANCY_Q8_UNCERTAINTY_NATS` · fp32 arm RE-MEASURED, q8 arm NOT VERIFIABLE

### What I measured

Real gpt2 at float32, the six default passages, `p = 1, seed = 0`, through
`vacancy_score()` — 18 real forward passes:

```
tokens      english 2754   swap 2766   nonce 3792
preserved   856 in every variant
wrong_content  nll(swap)  − nll(english) = 0.6904 ± 0.0539   856 pairs
unknown_form   nll(nonce) − nll(swap)    = 0.2872 ± 0.0450
total          nll(nonce) − nll(english) = 0.9776 ± 0.0590
```

The docstring quoted `2754/2858/3810`, `847 preserved`, `fp32 0.7166` and `fp32 0.9892`.
All four token counts, the preserved count and both fp32 values had moved. Again, this
matches what my charter told me — independently derived.

### What I could NOT verify, and why the constant is now under-supported

The justification was a **paired** comparison: fp32 against q8 **on the same texts**. The
swap rewrite changed the texts. The fp32 arm I re-ran; the q8 arm needs transformers.js in
a real browser (`VACANCY_MEASURED_DTYPES` is why), which I cannot do here.

**So no |Δ| exists for the shipped configuration.** Subtracting the old q8 numbers (0.6440,
0.8790) from the new fp32 ones would compare two different passage sets and manufacture
exactly the bound FR-720a forbids — the defect this campaign just fixed (arch F4). I did
not do it, and I did not invent a q8 number.

**I did not change the constant.** 0.2 exceeds every q8-vs-fp32 gap ever recorded on this
contrast (0.054 in the prototype study; 0.073 and 0.110 in this build pre-rewrite), and
lowering a bound without a measurement is the worse error. But it is **not** currently a
like-for-like measurement of what ships, and all three places that talk about it now say so:

- `lib/staticClient/arch.ts` — the docstring states both arms, marks the q8 half as
  belonging to texts that no longer exist, states plainly that 0.2 is retained rather than
  re-derived, and names the one thing that would restore it.
- `viz/info/InfoTab.svelte` — a new **Known limits** entry says the same to a reader, in
  prose; the what-is-real table's "measured error bound" no longer overclaims.
- `specs/007-vacancy-transform-field/architecture.md` §8.3a — a block quote under the
  quantization verdict, so the normative document carries it too.

**Open, and owned by nobody yet:** a browser q8 run of the static scorer on the six default
passages against `0.6904` / `0.9776`. Until that happens the constant must not be quoted as
"measured on the shipped swap".

### So this cannot rot again

`test_the_fp32_arm_quoted_in_the_static_client`
(`code/backend/tests/integration/test_arch_vacancy_score.py`) pins the token counts, the
preserved counts and both fp32 differences against a real gpt2 run. If the transform moves,
the comment fails a test instead of quietly becoming false — which is precisely how it
became false the first time. Runtime ~8 s on a warm cache.

---

## TASK 4 — a tab switch destroyed a training run in silence · FIXED, and warned

**Chosen: warn, not preserve.** Preserving would mean keeping the Lexicon Lab mounted while
the reader is elsewhere, and one-tab-at-a-time is what keeps a WebGL context, a loaded
transformer and a training worker from all existing at once. The worker also reports into a
panel that is gone. So the run really cannot survive — and the honest fix is to stop
destroying it without asking.

**Mechanism** (`code/frontend/src/lib/stores.ts`): a `pendingWork` registry. Whoever owns
destructible work registers it; `view.set` is **guarded** rather than plain, so a navigation
while the registry is non-empty is held in `pendingNavigation` instead of performed. The
guard is in the store rather than in the eleven `view.set` call sites deliberately: with it
in the call sites, the twelfth would be the one that lost a run.

**Back is covered too.** `syncFromUrl` holds a `popstate` that would leave a running tab and
puts the address bar back on the tab still showing. That costs the Forward stack, which is
the cheaper of the two things on offer.

**Registered by:** `viz/lex/TrainPanel.svelte` (the reported case) and, for the same
structural reason, `viz/geo/TrainPanel.svelte` and `viz/geo/FinetunePanel.svelte` — both
also lose an in-flight run to an unmount. Each registers from an `$effect` on the same
`busy` flag it disables its button with, and releases in `onDestroy`, so a confirmed
navigation leaves the registry empty rather than latched.

**UI** (`App.svelte`): an inline `role="alertdialog"` bar naming the work and the
destination, with *Stay and let it finish* / *Discard it and go to X*. Not `window.confirm`
— it cannot name the run, and it is suppressed in some embedding contexts. Not a modal: the
reader can keep watching the run while deciding, and doing nothing keeps the work.

**Reproducing test:** `code/frontend/tests/unit/navGuard.test.ts` (12 cases; real store,
real `App`, real history entries).

Before (guard removed, everything else identical): `5 failed | 7 passed`

```
× holds the navigation and names the work instead of discarding it
× stays put when the reader declines, and the run keeps its registration
× holds browser Back too, and puts the address bar back on the running tab
× names the work and the destination, and does not switch on its own
× its Stay button keeps the tab and its Discard button leaves it
```

After: `12 passed`.

**What is asserted by reading source, and why.** The three panels' registrations are pinned
by reading their source, not by driving them: a run needs a real `Worker` and jsdom has
none (`typeof Worker === "undefined"`, measured). The *store and shell behaviour* — the part
that was broken — is exercised for real. This is the same read-the-source technique
`docs.spec.ts` already uses for `VACANCY_Q8_UNCERTAINTY_NATS` and `shell.test.ts` for
`GeometryLab`, and it is stated in the test file rather than left to be discovered.

**Documentation matched:** Known limits said only "closing the page ends the model". A new
entry says that leaving the tab ends a *run* in either lab, that the app now holds and names
any such navigation including Back, that doing nothing keeps the run, and that a *finished*
model is unaffected (which the red team verified — model token `03769632905e` survived a
round trip).

---

## TASK 5 — two false provenance claims · FIXED

**`llm_geometry/lex/dolch.py`.** Applied the docs agent's exact wording for the "clean 1936
provenance" sentence and for line 3. Also qualified "The published list has `going`" — that
refers to the conventional graded list, not to the 1936 article, and the two sat three lines
apart.

**Other files repeating it — I checked all 13 occurrences of "1936" in `src/`, `tests/`,
`specs/`, `README.md` and `CLAUDE.md`.** Two more carried the claim and are fixed:

- `lib/lexEngine/dolch.ts` — a word-for-word port of the Python file, with the same
  `(Dolch, 1936)` header and the same "the published list has `going`". Given a matching
  provenance paragraph.
- `viz/lex/BudgetPanel.svelte:52` — the budget picker's tooltip said "**the real graded
  sight-word lists published by Edward William Dolch in 1936**", which is the false claim in
  one sentence. Reworded to what the article does and does not contain.

Clean: `lex/vocab.py` ("the real 1936 pedagogical lists"), `LexiconLab.svelte:539`
("published by Edward William Dolch in 1936 for teaching reading"), the `specs/006` lines,
and `viz/info/InfoTab.svelte`, which the docs agent already corrected in full.

**`specs/007-vacancy-transform-field/spec.md` FR-706.** Applied the exact replacement from
`fix-007-docs-shell.md`.

---

## Verification — the full local suite, after the last change

```
code/backend   python -m pytest -q     501 passed, 1 warning in 166.70s
code/backend   ruff check src/ tests/  All checks passed!
code/backend   black --check           81 files would be left unchanged
code/frontend  npm run check           1171 FILES 0 ERRORS 0 WARNINGS
code/frontend  npx vitest run          28 files · 558 passed | 1 skipped (559)
code/frontend  npm run build           ✓ built in 3.50s
```

Both stacks were re-run in full after the last edit (the Info-tab and architecture.md
prose for Task 3), not only after the code changes.

`npm run test:e2e` was NOT run, per charter. Two e2e tests read these numbers off the live
API and compare them with the prose, and should be the first thing checked by whoever runs
the suite next:

- `the documented counts are what the transform really produces` — now needs
  1,940 / 1,676 / 8,125 in the Info tab, which it has.
- `the documented swap collisions are the ones the engine measures` — reads the panel at
  three `p` and asserts the Info tab contains `349 / 484 / 364`, which it does.

No test was weakened, skipped or deleted. Nothing was mocked. No secrets; no scratch files
or screenshots left behind. `.claude/worktrees/` (five real git worktrees other agents left
behind) is now git-ignored rather than showing up as untracked content — the worktrees
themselves were not touched.

---

## What I could not verify

1. **The q8 arm of `VACANCY_Q8_UNCERTAINTY_NATS`** (Task 3). It needs a real browser. The
   constant is unchanged, its status is now stated in three places, and it must not be
   described as measured on the shipped swap until someone runs it.
2. **The `195 of 776` / 25.1 % figure** in `vacancy.py`'s `swap_pools` docstring. It is a
   measurement of the pre-fix engine, which no longer exists in the tree, so I could not
   reproduce it. I did not repeat it as a number in user-facing prose; I measured the
   post-fix counterpart instead (0 of 767) and stated the old one qualitatively.
3. **The four new `docs.spec.ts` cases added by the docs/shell agent**, and my own edits to
   that file, remain unexecuted — the charter forbids running e2e here.
4. **Where the Dolch grade-level split was first published.** Unknown, and now said to be
   unknown in all four places that mention it.
