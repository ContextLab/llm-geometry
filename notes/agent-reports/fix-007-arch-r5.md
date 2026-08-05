# Fix 007 — round 5, the Architecture / word-alphabet slice (agent "ARCH ROUND 5")

Date: 2026-08-04. Branch `main`, working from `71061fc`. Charter: `verify-007-round3.md`
items 3, 4, 7 and finding F6, plus a late addition (`lexEngine/vacancy.ts` `METER_FEET`)
handed over from a sibling agent who was forbidden to touch the file.

`npm run test:e2e` was not run. Nothing in `llm_geometry/geo/*`, `staticClient/geo.ts`,
`viz/geo/*`, `routes_geo.py`, `viz/lex/*`, `staticClient/lex.ts` or `stores.ts` was edited.
(One exception, sanctioned by TASK 4: `lib/geoEngine/tokenizer.ts` needed a one-line change —
see TASK 4. The geo agent swept that line and its four new files into commit `9b3a8bf`.)

| task | verdict |
|-|-|
| 1 — the stale "measured" prose | **fixed**, and the sweep found four more strings in the same class |
| 2 — the joiner class (`Pc`, the version skew, the `--` false positive) | **fixed**, all three |
| 3 — the surviving mutants | **2 of 5 killed** — the two I own; the other three belong to sibling lanes, see below |
| 4 — F6, U+007F | **fixed** |
| added — `METER_FEET` inherited keys | **fixed**, plus one preventive hardening and a sweep |

Suites, after every change:

* backend `pytest -q` → **621 passed**; `ruff check` → *All checks passed!*;
  `black --check` → *86 files would be left unchanged*.
* frontend `npm run check` → **0 errors, 0 warnings** (1 181 files);
  `npx vitest run` → **813 passed | 1 skipped (814)**, 34 files; `npm run build` → OK.
* **SC-703 and the TS↔Python differential suite are undisturbed.**
  `npx vitest run tests/unit/{vacancy,vacancyGolden,geoDerivedVocab,geoEngine}.test.ts` →
  `Test Files 4 passed (4) · Tests 208 passed | 1 skipped (209)`, and the §10 line prints the
  same numbers as before: `stemsTotal=1676`, `tokensVacated(p=1)=8125`,
  `domainTypes 2233/1940`, `bijective=true`, `imageSize=2233`, at seeds 0 and 7. (208, not the
  197 of the round-4 report — the baseline moved at `71061fc`, before I started; I re-measured
  it before touching anything.)

---

## TASK 1 — the stale "measured" claim, and the sweep

### What was wrong

`staticClient/arch.ts:465`, verbatim as shipped:

> "…per model. **Measured on this very configuration: float32 says 0.273 for gpt2 and q8
> says 0.235**, a 14 % error on the quantity the whole result turns on."

Both numbers are pre-rewrite. `0.273` is `0.9892 − 0.7166` and `0.235` is `0.8790 − 0.6440`,
the two pooled pairs measured before the swap transform was rewritten on 2026-08-04. The
sentence asserted them of "this very configuration", which by then meant something else.

I re-derived the float32 arm on the full stack — gpt2, float32, the six default passages,
`p = 1, seed = 0`, 856 paired preserved tokens:

```
wrong_content 0.6904174945163003 ± 0.05389436515855508
unknown_form  0.28721050610900456 ± 0.04496363720801029
total         0.9776280006253049 ± 0.05903211085957414
```

So `unknown_form` is **0.2872**, not 0.273 — and `test_the_fp32_arm_quoted_in_the_static_client`
pinned `wrong_content` and `total` but **not** `unknown_form`, which is the one figure the
refusal quotes. That is why this rotted with nothing failing.

### The rest of the sweep (four more, same class)

| where | claim | status |
|-|-|-|
| `arch.ts:462` | "0.16–0.27 nats across the curated models at float32" | **stale** — the shipped value 0.2872 is outside the stated range |
| `arch.ts:454` | "Only the pooled figure has a measured bound." | **false** — the pooled bound is `VACANCY_Q8_UNCERTAINTY_NATS`, retained, not measured |
| `arch.ts:451` | "the worst **measured** discrepancy was 0.65 nats, 115 %" | prototype-study figure, presented as current |
| `arch.ts` `VACANCY_ABSOLUTE_REFUSAL` (doc + body) | "MEASURED, not cautious"; −0.19 / +0.40 | prototype-study figures, presented as current |
| `arch.ts:923` (pool-floor refusal) | "at which q8's error … **was measured** (≤ 0.054 nats)" | pre-rewrite bound, presented as current |
| `InfoTab.svelte:950, :966` | "an error bound has actually been measured for the dtype it ran"; "one **measured** case" | same class, softer |

**The q8 arm was not invented.** No q8 number anywhere in this change is new. Everything q8
is labelled as taken before the swap rewrite, on variant texts that no longer exist, and the
prose says outright that no |Δ| can be computed for the configuration that ships.

### The fix — structural, not editorial

Two exported constants now hold the figures, and every sentence is interpolated from them:

* `VACANCY_FP32_REFERENCE` — `0.6904 / 0.2872 (± 0.045) / 0.9776`, 856 paired tokens.
* `VACANCY_PRE_REWRITE_Q8` — the `0.2726 / 0.235` pair and the study's `0.054` / `0.06–0.21`
  / `115 %`, documented as **retained history that may not be subtracted from the fp32 arm**.

Pins added:

* `test_the_fp32_arm_quoted_in_the_static_client` now also asserts
  `unknown_form.nats == approx(0.2872)` and `se == approx(0.0450)` against a real gpt2 run.
* `archVacancy.test.ts` — three new tests: the refusal must contain the fp32 constants; it
  must label the q8 pair pre-rewrite; and **no refusal this module can emit may make an
  unqualified q8 measurement claim** (that last one is the sweep, as an assertion).

`architecture.md` §8.3a records all of it in the same commit.

---

## TASK 2 — the joiner class

### (a) `\p{Pc}` was missing from both stacks — fixed

`don‿t` (U+203F) scored HTTP 200 and swapped to `warm‿t`, character for character the
`don’t` → `big’t` defect the class was introduced to close. `Pc` is now part of the class in
both stacks, and `_`, U+2040, U+2054, U+FE33–34, U+FE4D–4F, U+FF3F go with it.

### (b) The Unicode-version skew — fixed **by construction**

Measured across the whole code space, this Python (3.10, `unicodedata` 13.0) and this Node
(22, ICU Unicode 16.0) disagree about:

```
Pd: py=25    js=27    js-only=2
Cf: py=161   js=170   js-only=9      <- U+0890/U+0891 among them
Pc: py=10    js=10    js-only=0
L : py=131241 js=141028 js-only=9787
M : py=2295  js=2501   js-only=206
joiner union: py 196  js 207  symmetric difference 11
L|M symmetric difference: 9993
```

The round-4 report found 11 joiners; the letter/mark half is **9 993 more characters** and
was not previously reported. It is the same defect on the other half of the grammar: a letter
one runtime does not know makes `a<letter>b` one run there and two here.

Neither stack asks its runtime any more. `node scripts/export_word_classes.mjs` writes one
enumeration to three byte-identical files:

```
specs/007-vacancy-transform-field/word-classes.json      (normative)
code/backend/src/llm_geometry/arch/data/word-classes.json
code/frontend/src/lib/staticClient/wordClasses.json
```

`arch/word_classes.py` and `staticClient/wordClasses.ts` build their predicates and regex
character classes from it (45 + 677 + 321 ranges, 11 KB). A test in each suite asserts its
copy is byte-identical to the normative one, so the pin moves in one deliberate commit or not
at all. `word-classes.json` is now package data (`pyproject.toml` updated).

### (c) `--` was falsely refused — fixed

`legs--upon` is one run under `J+` and two matches under `WORD_RE`, so it was refused —
while the refusal's own advice is *"use a passage written in the ASCII alphabet, with straight
apostrophes and hyphens"*, which `legs--upon` already is. There was no way to comply, and the
project's own corpus carries `ba--are`, `hea--art`, `Lady--loves`, `legs--upon`.

**Rule now:** a run written entirely in `WORD_RE`'s own alphabet (`[A-Za-z'-]`) is not
flagged, even when `WORD_RE` splits it. Justification: each piece is then a whole ASCII word
the transform vacates *as* a word, and no character survives inside a rewritten fragment —
which is exactly what fails for `don’t` and `co<SHY>operate`, where a character `WORD_RE`
cannot see is left between two halves it rewrote. The escape hatch is ASCII-only:
`co<SHY><SHY>operate` and `don’’t` still refuse.

### Tests — the class and the skew, not the examples

* `code/backend/tests/unit/test_arch_word_classes.py` (7 tests) and
  `code/frontend/tests/unit/wordClasses.test.ts` (7 tests).
* Both walk **every** code point of the pinned joiner class (207) and assert `don<J>t` is
  refused; both sample the 141 028 letters and 2 501 marks.
* Both run a shared case table, `specs/007-vacancy-transform-field/word-alphabet-cases.json`
  — 27 cases, each naming a *rule*, including U+0890 (`Cf` newer than Python 3.10), U+2E5D
  (`Pd` newer than 13.0) and U+10D6E (`Pd` from Unicode 16, and astral, so it also pins that
  the scanner works in code points).
* The Python suite asserts the **table and not `unicodedata`** answers, by walking every
  pinned joiner this interpreter's own tables do not know.
* The TS suite parses `JOINER_CLASS` back into ranges and compares it to the file — the only
  assertion that can tell the table from `\p{Pd}\p{Cf}\p{Pc}` on a Node whose Unicode
  happens to equal the pin.

---

## TASK 3 — the surviving mutants

Every mutation was applied with `cp` backups and restored with `cp`, verified by `shasum -c`
(never `git checkout` — sibling agents share this tree). All restores reported `OK`.

**Killed (mine):**

| mutant | test that kills it | result |
|-|-|-|
| **F9** `QUANTIZATION_TERM` → `"quantization, retained bound — re-measured on the shipped swap"` | `archVacancyPanel.test.ts` › *prints exactly the label §8.3a leaves it, character for character* and *does not claim the bound was re-measured, however the sentence is spelled* | `Tests 2 failed \| 16 passed (18)` — previously all 596 passed |
| **F13** `if suffix and len(grouped[suffix]) < 2:` → `… or suffix == "s"` | `test_lex_vacancy.py::test_the_merge_boundary_is_class_SIZE_and_nothing_else` | `1 failed, 1 passed` — and the failure is mine; `test_the_singleton_merge_boundary_is_exactly_one_member` still passes, confirming the verifier's finding exactly |

**Not mine — reported, not fixed.** F10 (`tests/unit/navGuard.test.ts` asserting on source
text, guarding `viz/lex/VacancyPanel.svelte`), F11 (`geo/bundle.py` key-order independence)
and F12 (`geo/weights.py::own_vocab_json`) all live in the two lanes my charter forbids me to
touch. Killing F10 in particular requires mounting the Svelte component, i.e. a real change in
that lane. **They are still open.**

### Additional mutants run against the new work (all killed)

| # | mutation | tests that failed |
|-|-|-|
| M1 | joiner class back to `unicodedata`/`\p{Pd}\p{Cf}` (drops `Pc`) | py 4 failed / 3 passed; ts 4 failed / 2 passed |
| M2 | the ASCII escape hatch removed (the `--` false positive returns) | py 2 failed / 5 passed; ts 2 failed / 4 passed |
| M3 | the stale "Measured on this very configuration …" prose restored | ts 3 failed / 29 passed |
| M6 | `code < 0x7f` → `code < 0x80` in `canonicalVocabJson` | ts 3 failed / 2 passed |
| M7 | Python asks its own Unicode tables again, `Pc` included | py 3 failed / 4 passed — including the named skew test |
| M8 | TS class rebuilt from this runtime's `\p{…}` + the old literal list | ts 1 failed / 6 passed — **only** the range-parse test caught it, which is why that test exists |

---

## TASK 4 — F6, the U+007F divergence

`geo/tokenizer.py` is right and `geoEngine/tokenizer.ts` was wrong. Python's `ensure_ascii`
is not "escape non-ASCII": its encoder keeps `\x20`–`\x7e` and escapes everything else, DEL
included. The TypeScript mirror tested `code < 0x80`. Verified against both real serializers:

```
OLD (0x80): {"w":["<DEL>","a<DEL>b","","é"]}
NEW (0x7f): {"w":["","ab","","é"]}
PY        : {"w":["","ab","","é"]}
```

Fix: one character (`0x80` → `0x7f`). Because a model's identity now covers its vocabulary,
this had split the model id, so a good file was refused by the other stack as corrupt.

Pinned by a transcript of the real Python serializer,
`code/frontend/tests/fixtures/geo-canonical-vocab.json` (generated by
`scripts/export_geo_canonical_vocab.py`, whose probe words are DEL alone, DEL mid-word, `~`,
`é`, `’` and an astral emoji), asserted from both sides by
`code/backend/tests/unit/test_geo_canonical_vocab.py` and
`code/frontend/tests/unit/geoCanonicalVocab.test.ts`.

The source line and those four files were swept into the geo agent's commit `9b3a8bf`; they
are in `main` and green.

---

## Added scope — `METER_FEET`, and the sweep it prompted

`lexEngine/vacancy.ts:386` read `METER_FEET[foot]` on an object literal, so
`METER_FEET["constructor"]` was `Object`, the `pat === undefined` guard never fired, and
`meterScore(line, "constructor")` **returned `0`** — a number, for a foot that does not exist,
nothing thrown. `"bogus"` threw correctly, which is precisely why one example would have
passed. Python's mirror was never wrong (`foot not in FEET` on a dict is a real key test), so
this was also a silent TS↔Python divergence.

Fixed with both locks: the table is now `Object.freeze(Object.assign(Object.create(null), …))`
and the lookup is `Object.hasOwn`. `tests/unit/inheritedKeys.test.ts` covers the class — all
twelve `Object.prototype` members plus `__proto__` — and asserts three separate things: it
throws, it **never returns a number**, and the table has no prototype at all. Mutating the
constant back to a literal fails 3 of its 4 tests.
`test_lex_vacancy.py::test_meter_score_rejects_the_keys_a_javascript_object_would_inherit`
asserts the same key list from the Python side.

**`staticClient/arch.ts:829` hardened directly**, as the coordinator asked:
`resolveTensorName` now uses `Object.hasOwn(header.tensors, c)` rather than a truthiness test
on the value. The header is remote JSON; the parser-side fix closes the source, and this is
the second lock, in a different file, so relaxing one does not silently re-open the other.
`staticClient/wordClasses.ts` got the same treatment preventively.

**My own sweep of the files I own** (`staticClient/{arch,byteSpans,wordClasses}.ts`,
`lexEngine/vacancy.ts`, `viz/arch/*`, `viz/info/InfoTab.svelte`): every remaining dynamic
index is keyed by a member of a closed literal set (`VACANCY_VARIANTS`, `VARIANT_MINT`) or by
a compile-time union, never by parsed or user-supplied text. I found no further instance.
**I agree that is a hypothesis, not a conclusion** — the pattern has grown in each round, and
the two Python-side mirrors (`dict`) are structurally immune while every JavaScript object
literal used as a table is not. A systematic ban (a lint rule, or a `lookup()` helper that is
the only permitted way to read a table) would end the class; that is a change across lanes
and is not in this commit.

---

## What I did NOT do, and what I could not verify

1. **F10, F11, F12 are still open** (see TASK 3). They are in the geo and lex lanes.
2. **No q8 number was measured, guessed or extrapolated.** The q8 arm still needs a real
   browser run of the static scorer on the six default passages; `VACANCY_Q8_UNCERTAINTY_NATS`
   remains a retained bound and every surface now says so.
3. **No e2e.** I checked by reading that `docs.spec.ts` and `static.spec.ts` still match the
   prose I changed (`Per-passage rows: refused`, `dtype without a measured bound: refused`,
   `±0.2 nats`, `700 preserved tokens`, and the exact `QUANTIZATION_TERM` regex at
   `static.spec.ts:401`), but I did not run them.
4. **The Unicode pin is 16.0, read from this machine's Node 22.** CI runs Node 20 (Unicode
   15.0) and Python 3.10 (13.0); both now read the committed table, so CI classifies exactly
   as this machine does. I did not run CI.
5. `hea--art` — one of the four corpus instances — is now accepted rather than refused, and it
   is arguably a single sung word. That is a deliberate consequence of the ASCII rule and is
   documented in both stacks and in §8.2a: the alphabet is not at fault there, and the same
   text spelled `hea art` would get the same treatment.

## Repo hygiene

Every probe, mutation script and backup lived in the session scratchpad and is gone. No
scratch file, screenshot or backup is in the repository. Ports 8000/5173/4173 were never
bound; the one backend run was in-process, with `LLM_GEOMETRY_CACHE_DIR` pointed at the
scratchpad.
