# Fix — Architecture round 3 (red-team 007, verification V2 items 1, 2, 3)

Date: 2026-08-04. Branch `main`, from `33c5ce6`. Agent: FIX ARCH ROUND 3.
Charter: `notes/agent-reports/verify-007-arch-lex.md` findings **F1** (error-bar label),
**F2** (joiner class), **F3** (identity), plus the `backwards`-reachability question.

No server was started, stopped or restarted; `npm run test:e2e` was not run. Nothing under
`viz/lex/`, `staticClient/lex.ts` or `provenance.ts` was touched (the concurrent agent's slice).
Every temporary revert made to capture "before" evidence was restored and re-verified green.

---

## TASK 1 (F1) — the rendered error-bar label said `measured`; it is not

**The defect.** `vacancyVerdict.ts:106` and `:115` printed, unconditionally once the stack
attached a value:

```
± 0.2 (quantization, measured)
```

against `architecture.md:880`, verbatim:

> Until then, do not quote it as "measured on the shipped swap".

`VACANCY_Q8_UNCERTAINTY_NATS = 0.2` was derived from q8-vs-fp32 comparisons taken on the
PROTOTYPE swap's variant texts; the swap rewrite replaced those texts and the q8 arm cannot be
re-run without a real browser. So the code asserted the one provenance claim FR-720a governs, on
the one surface a reader reads — the same defect class the campaign already fixed once (arch F4).

**The fix.** One exported constant, used by both renderers, plus the bound's full standing
rendered once under the headline pair:

- `QUANTIZATION_TERM = "quantization, retained bound — not re-measured since the swap rewrite"`
  (`code/frontend/src/viz/arch/vacancyVerdict.ts`);
- `quantizationNote(d)` — the paragraph the panel now shows at
  `data-testid="arch-vac-quantization-note"`: retained rather than fresh, the texts it was
  measured on no longer exist, re-deriving it needs a browser q8 run, and it is kept unchanged
  because it exceeds every gap ever observed. It quotes the number the stack attached, never a
  literal of its own.

**No q8 value was invented or re-derived.** I cannot measure it without a browser; the constant
is byte-for-byte unchanged and only its *label* moved.

**Prose changed in the same commit** (code, contract, spec and UI now agree):

| file | what changed |
|-|-|
| `specs/007-vacancy-transform-field/spec.md` FR-720a | "measured error bound" → an error bound whose **real standing is stated**; adds the RETAINED-not-current paragraph and "no surface may call it measured on the shipped swap" |
| `specs/007-vacancy-transform-field/architecture.md` §8.3a | adds "The rendered label now complies", naming `QUANTIZATION_TERM` and the test that pins it |
| `viz/info/InfoTab.svelte` (2 sentences) | "the size at which the bound was measured" → "the **retained** bound"; the ±0.2 bullet now says the panel labels it a retained bound |
| `lib/staticClient/arch.ts` (2 docstrings) | "a stated, MEASURED quantization uncertainty" → a bound RETAINED from a measurement whose texts the rewrite replaced |

The e2e docs pins (`docs.spec.ts:488-489`, `±0.2 nats` and `700 preserved tokens`) are untouched
by the rewording — both literals are still present in `InfoTab.svelte`.

**Test:** `archVacancyPanel.test.ts` → `F1 (round 3) — the quantization term does not call itself
measured`, 3 cases. The key one iterates all four rendered surfaces (two headline cards, two
secondary lines) and asserts `not.toMatch(/(?<!re-)measured/)` and `toMatch(/retained/)`. The
F4 block's two literals were updated to reference `QUANTIZATION_TERM` rather than retyping it.

**Before/after (real runs):**

```
# HEAD's vacancyVerdict.ts restored in place, tests unchanged:
   × F1 (round 3) … > never labels the retained bound 'measured', on either surface
   × F1 (round 3) … > says on the page what the bound's real standing is
   × F4 … > states the quantization uncertainty on a SECONDARY row too
   × F4 … > states nothing it was not given: no quantization term at float32
   × F4 … > headline cards carry both terms, unchanged
      Tests  5 failed | 11 passed (16)
# after (file restored):
      Tests  16 passed (16)
```

---

## TASK 2 (F2) — the joiner half of the word alphabet, closed as a class

**The defect.** Both stacks scanned Unicode **letter** runs
(`WORDLIKE_RE = [^\W\d_]+(?:['\-][^\W\d_]+)*` / `/\p{L}+(?:['\-]\p{L}+)*/gu`), admitting only
ASCII `'` and `-` as internal joiners. Any other joiner ended the run, leaving two runs `WORD_RE`
matched **in full** — so `fragmented_words` flagged nothing, the transform rewrote half a word,
and the endpoint returned HTTP 200. Reproduced verbatim against the real backend before the fix:

```
english = 'The cat’s don’t stop and the dog won’t go to the tree in the park today okay.'
swap    = 'The want’s big’t wish and the cat clean’t go to the park in the take away yellow.'
nonce   = 'The chooz’s dud’t skint and the scarrt boft’t go to the jaust in the shork refon shesid.'
```

`big’t` and `clean’t` are not words, which is the one property `swap` is defined by.

**The fix — the class, not the character.** A joiner is now anything that binds two letters
without being one:

- Unicode general category **`Pd`** (dash punctuation): hyphens of every width — U+002D, U+2010,
  U+2011 non-breaking, U+2012–U+2015, U+FE63, U+FF0D;
- Unicode general category **`Cf`** (invisible format): U+00AD soft hyphen, U+200B–U+200F
  (ZWSP/ZWNJ/ZWJ/bidi), U+2060 word joiner, U+FEFF;
- **combining marks** (`M*`), folded into the letter atom rather than treated as joiners — a mark
  belongs to its letter. NFC composes most away but not all: `b` + U+0301 has no precomposed
  form, so `cab́le` stayed the two `WORD_RE` words `cab` + `le`;
- named literals with no distinguishing property: `'` U+0027, `‘` U+2018, `’` U+2019, `ʹ` U+02B9,
  `ʼ` U+02BC, `՚` U+055A, `′` U+2032, `＇` U+FF07, `·` U+00B7, `‧` U+2027, `−` U+2212.

Grammar in both stacks: `(L M*)+ ( J+ (L M*)+ )*`. A **trailing** joiner is punctuation and is
excluded, so `"The cat — the dog ran."` (spaced em dash) still scores.

- Backend `arch/vacancy_score.py`: `WORD_JOINER_CATEGORIES`, `WORD_JOINER_CHARS`, and a
  hand-written `wordlike_runs()` scanner — Python's `re` has no `\p{Pd}`/`\p{Cf}`/`\p{M}`, so it
  scans where JS matches. `re` is no longer imported there.
- Static `staticClient/byteSpans.ts`: `WORD_JOINER_CLASS` + `LETTER_RUN`, same grammar as a
  `u`-flagged regex.
- Both refusal messages updated identically ("ASCII letters **joined by the ASCII apostrophe and
  hyphen** only", and the `don’t` → `big’t` example beside the `café` → `washé` one).

**Tests — the class, not the example.** A 22-row case table, duplicated verbatim in both suites
so the stacks are pinned to the same answers:

- backend `tests/unit/test_arch_vacancy_align.py::test_the_whole_joiner_class_is_refused_not_just_the_one_seen`
  (`JOINER_CASES`), plus negative cases (`don't`, `co-operate`, plain English) and the
  "names the run the reader wrote" assertions;
- frontend `tests/unit/archVacancy.test.ts` → `refuses the whole joiner class…`,
  `still admits every joiner WORD_RE really does handle`, `names the joined word whole…`
  (`JOINER_CASES`);
- end to end on a real gpt2 run:
  `tests/integration/test_arch_vacancy_score.py::test_a_curly_apostrophe_is_refused_through_the_endpoint_too`
  — curly apostrophe, soft hyphen and ZWJ passages, each a typed 400 naming the word.

**Before/after:**

```
# HEAD's vacancy_score.py restored, tests unchanged:
FAILED tests/integration/…::test_a_curly_apostrophe_is_refused_through_the_endpoint_too
# HEAD's byteSpans.ts restored:
   × the word alphabet (§8.2) — red team F6 > refuses the whole joiner class, not just the character that was reported
   × … > names the joined word whole, so the reader can find it
   × … > refuses such a passage with a typed error naming the word
# after: 42 passed (backend align + score), 29 passed (archVacancy.test.ts)
```

**A deliberate widening, stated:** `cat—dog` (unspaced em dash) is now refused. It was not merely
cosmetic — `cat` and `dog` are vacated independently there, and a BPE piece spanning the dash
would hit the `preserved_token_indices` guard as an opaque 500. A typed 400 naming the word is
the better of the two, and a refusal is never a wrong answer.

---

## TASK 3 (F3) — an identity detected from the texts, not from `p`

**The defect.** `identity` was `float(p) == 0.0` (backend `:757`) and `p === 0` (static `:922`).
An all-closed-class passage has nothing to vacate at **any** `p`, so its three variants are one
string and every difference is 0 by construction — but it took the `p = 1` route and rendered
`0.000 ± 0.000 (sampling, 20 paired tokens)`, an `upper bound` caption, and "score more text".

**The fix.**

- backend: `identity = all(texts["english"] == texts["swap"] == texts["nonce"] for … in prepared)`;
- static: `staticVacancyDifferences(…, { identical, p })`, with `identical` computed from the
  prepared variant texts;
- the `identityNote` branches on which route was taken — the `p = 0` note is unchanged, and the
  new one says the text has no word the transform vacates and that **more text of this kind
  cannot change it**;
- `vacancyVerdict.ts` no longer dresses an identity as a measurement (this also closes the
  verifier's **F7**, which is the same sentence one line up): `errorBarTerms` returns
  `an identity across N paired tokens — 0 by construction` instead of a `± 0.000 (sampling…)`,
  `secondaryLine` returns `expr = 0.000 (an identity: the three variants are one string)`, and
  `upperBoundLabel` returns `not a bound — the three variants are one string…`. The token count
  is real and is kept; the ± is not, and goes.
- `spec.md` gains **FR-720b** stating the condition and the three things that must not be printed.

**Tests:**

- `tests/integration/test_arch_vacancy_score.py::test_an_all_closed_class_passage_is_an_identity_not_a_measurement`
  — a real gpt2 run on the red team's exact passage; asserts the previews are one string,
  `identity is True` on all three differences, and that the note explains *this* route;
- `tests/unit/archVacancy.test.ts` → `calls an identity an identity from the TEXTS, not from p`
  — both routes' notes, and that `identical: false, p: 0` is **not** an identity (the condition
  is the texts in both directions);
- the pre-existing `test_p_zero_is_the_identity_and_costs_nothing` still passes unchanged.

**Before/after:**

```
# HEAD's vacancy_score.py restored:
>           assert d["identity"] is True, d["id"]
E           AssertionError: wrong_content
E           assert False is True
# HEAD's staticClient/arch.ts restored:
   × what the quantized static build may say (§8.3a, FR-720a) > calls an identity an identity from the TEXTS, not from p
# after: both green.
```

---

## The `backwards` verdict branch — NOT dead code, and the verifier's 110 runs missed it

The verifier could not reach it through the UI and could reach it only through the API's
`passages[]` array. I swept the panel's **own two controls** — a single custom passage in the
textarea, `p = 1`, seed 0–11 — over four short passages against the running backend, 48 real
scoring runs:

```
p0 s10  -0.6596 +- 0.3049  n=8   backwards
p2 s10  -0.9510 +- 0.3880  n=10  backwards
p3 s4   -1.0923 +- 0.4651  n=9   backwards
p3 s11  -1.2337 +- 0.3115  n=9   backwards
RUNS 48  BACKWARDS 4
```

So the branch is reachable from the UI in roughly one run in twelve on a short passage; the
verifier's negative result was an artifact of passage length, not of the control surface. **It
must stay.** Nothing was deleted.

That run also settles the verifier's **F6** (low): the paragraph named "intermediate `p`" as the
primary cause, and intermediate `p` is now a typed 400 — an unreachable cause stated as the
explanation. The sentence now reads "That happens on short passages…", which is exactly what the
48 runs show, with the intermediate-`p` route named as the one that used to exist and is now
refused. Changed in the same commit as the behaviour it describes.

---

## What I could NOT verify

1. **The q8 arm of `VACANCY_Q8_UNCERTAINTY_NATS`.** Unchanged from the two previous agents'
   position: it needs a real browser running this build's static scorer, which I may not start.
   I did not compute, guess or extrapolate a q8 number — the fix is entirely in what the label
   claims.
2. **The static panel rendered end to end.** `npm run test:e2e` is out of scope for this charter,
   so the static arm of all three tasks is verified by unit tests over the real functions plus
   `npm run build`, not by a screenshot.
3. **Models other than gpt2.** Only gpt2 was scored. The joiner and identity changes are
   model-independent (they run before any weights are touched, and on the variant strings), but
   I did not run Qwen or SmolLM2.

---

## Suites

Run in full after the last change.

| suite | result |
|-|-|
| backend `pytest -q` | `504 passed, 1 warning in 170.87s` |
| backend `ruff check .` | `All checks passed!` (fixed one `F401` my edit created: `re` is no longer used) |
| backend `black --check .` | `81 files would be left unchanged` (reformatted `vacancy_score.py` once, then clean) |
| frontend `npm run check` | `1173 FILES 0 ERRORS 0 WARNINGS 0 FILES_WITH_PROBLEMS` |
| frontend `npx vitest run` | `Test Files 29 passed (29)` · `Tests 591 passed \| 1 skipped (592)` |
| frontend `npm run build` | `✓ built in 3.15s` |

The frontend counts include the concurrent agent's in-flight files (`viz/lex/*`,
`staticClient/lex.ts`) and the untracked `zzverify.test.ts`; the backend `504` is the whole
suite on real models.

No test was weakened, skipped or deleted. Three existing assertions were updated to the new
message/label text (`ASCII letters only` → `ASCII letters joined by`, and the two
`(quantization, measured)` literals → `QUANTIZATION_TERM`); each became *more* specific, not
less, and each is accompanied by a new assertion that the old false claim is absent.

**Repo hygiene:** every temporary revert restored and re-verified; the sweep script lives in the
session scratchpad, not in the repo. The untracked
`code/frontend/tests/unit/zzverify.test.ts` noted by the verifier is still present and is not
mine — whoever closes this campaign should confirm it is removed.
