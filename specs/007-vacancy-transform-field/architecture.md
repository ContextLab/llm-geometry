# Feature 007 — the vacancy transform: TS ↔ Python contract

**Status:** normative. Both stacks implement *this document*, not each other.

This is the file that feature 006 taught us to write first. In 006 the contract omitted one
sentence — how a corpus becomes a token stream — and the two stacks silently trained on
different data for a day. Everything here that reads like pedantry is load-bearing; if a
sentence looks obvious, it is because someone would otherwise have guessed differently.

Source material: `~/Desktop/TinyModelsDoc/tiny_models.tex` §"The vacancy transform"
(`\label{sec:how-vacancy}`) and `tiny-seuss/synth/jabberwockify.py`. We port the *design* and
correct the *implementation*; §9 lists every deliberate departure with its reason. As in 006,
we never port a claim we have not measured ourselves.

---

## 0. What the instrument is for

The doc's T4 is a 2×2 over *location* (a token has a prior embedding) and *field* (a token's
distributional neighbourhood is fully specified by context):

|  | no field | field supplied |
|-|-|-|
| **no location** | (i) random init, no data | (iii) **vacancy** — nonce form, full syntactic support |
| **location** | (ii) minting at a hub centroid | (iv) normal word learning |

The vacancy transform manufactures condition (iii) at a controlled rate `p` on any corpus:
closed-class words, inflectional suffixes, syntax and line structure are preserved exactly;
open-class stems are replaced by phonotactically legal nonce forms carrying the same syllable
count and stress pattern.

**The result the tiny arm can prove.** For a *word-level model trained from scratch*, a word's
"location" is nothing but a row index — the model never sees the letters. Under the conditions
of §7 the transform is therefore a **pure relabelling of the vocabulary**, and the model is
*exactly* invariant to it: same token id stream, same loss, bit for bit. That is stronger than
the doc's prediction (it predicts (iii) ≫ (ii); in the tiny regime (iii) ≡ (iv) identically),
and §7 states it as a theorem with a test that would catch it becoming false.

An invariance is only worth showing against something that breaks it. §6 defines the three
control conditions the doc itself calls for, and §8 defines the pretrained arm — a model that
*does* have locations — which is where the number that says what location was worth comes from.

---

## 1. Word segmentation

The transform rewrites a raw text in place. It finds words with **exactly the tokenizer's
regex** and passes everything else — whitespace, punctuation, digits, line breaks — through
unchanged, byte for byte.

```
WORD_RE = /[A-Za-z]+(?:['\-][A-Za-z]+)*/g          # TS: lexEngine/vocab.ts
WORD_RE = re.compile(r"[A-Za-z]+(?:['\-][A-Za-z]+)*")   # Py: lex/vocab.py
```

This is **not** the regex the source used (`[A-Za-z][A-Za-z']*`, which splits `good-bye` into
two words). Using the tokenizer's own regex is a hard requirement: the relabelling theorem of
§7 is false the moment the transform's idea of a word differs from the trainer's.

Each match is replaced by the output of `transformWord` (§5). **Every output is itself a
single, complete `WORD_RE` match** — checked, not assumed (§7.3). Therefore
`tokenize(vacate(text))` has exactly the same length and ordering as `tokenize(text)`, and
because line breaks are untouched, the `<eos>`-per-line rule produces the same number of
`<eos>` in the same places.

TS must construct a fresh `RegExp(WORD_RE.source, "g")` per call — the shared literal carries
`lastIndex`. (This bit `vocab.ts` already; do not re-learn it.)

---

## 2. Eligibility: what may be vacated

### 2.1 The closed class

`FUNCTION_WORDS` is the source's curated list, ported verbatim, whitespace-split and
lowercased:

```
a an the this that these those my your his her its our their some any all both each
every no none i me you he she it we they him them us who whom whose which what where
when why how is am are was were be been being do does did done have has had having
will would shall should can could may might must not and or but so if then than as of
to in on at by for with from into onto up down out off over under again once here there
very too also only just even still yet ever never always about after before while
because though although unless until since during between among against through above
below near far one two three four five six seven eight nine ten
```

The source carries a warning we keep: an earlier version of it added short Dolch service words
to the closed class, which silently protected content verbs (`run`, `eat`, `see`, `get`, `let`,
`put`) and understated the vacancy rate. The closed class is **this curated list only**. Do not
union it with a Dolch budget.

Callers may extend it with a `keep` set; the effective set is `FUNCTION_WORDS ∪ lower(keep)`.

### 2.2 The eligibility test

A word is eligible for vacancy iff, after suffix splitting (§3), its **stem** satisfies all of:

1. `lower(stem) ∉ keepSet`
2. `stem` matches `^[A-Za-z]+$` — ASCII letters only
3. `len(stem) > 2`

Test 2 is why hyphenated and apostrophised words behave as they do, and both stacks must agree
on it exactly:

- `good-bye` — no suffix matches, stem is `good-bye`, which contains a hyphen, so **test 2
  fails and the word is never vacated**.
- `don't` — the `n't` suffix splits it to stem `do`, which fails test 3 (and test 1).
- `dog's` — the `'s` suffix splits it to stem `dog`, which passes; output is `<nonce>'s`.

Python must use `re.fullmatch(r"[A-Za-z]+", stem)`, **not** `str.isalpha()`. `isalpha()` is
Unicode-aware and would accept letters JS's `^[A-Za-z]+$` rejects. Nothing in the shipped
corpus exercises the difference; a pasted corpus would.

---

## 3. Suffix splitting

Inflectional morphology is preserved: the stem is vacated, the suffix is re-attached, so the
nonce still looks inflected and the syntax still parses.

```
SUFFIXES = ["ing", "edly", "est", "ies", "'s", "n't", "ed", "es", "er", "ly", "s"]
```

Tried **in this order**; the first match wins. A suffix `s` matches iff
`lower(word).endswith(s)` **and** `len(word) - len(s) >= 3`. The split slices the **original**
word, so case is preserved.

```
EXCEPTIONS = {brother, father, mother, sister, never, over, under, morning, giving, thing}
```

A word whose lowercase form is in `EXCEPTIONS` is **never split** (stem = word, suffix = `""`).
This list comes from the audited copy of the source, not the copy in the zip: without it
`brother → broth+er` and `morning → morn+ing`, which the source itself flags as a known
artifact. It is a spelling heuristic, not a morphological analyser, and it is wrong on words
outside the list (`ladder → ladd+er`). That is acceptable — the nonce still carries a
consistent identity and an inflected-looking surface — but it must be *documented in the UI*,
not quietly tolerated.

---

## 4. The vacancy decision — nesting

A stem is vacated iff `u(stem) < p`, where `u` depends only on the stem and the seed:

```
digest = sha256(utf8(f"{seed}:{lower(stem)}"))        # 32 bytes
top64  = int(digest[0:8], big-endian)                 # first 8 bytes
u      = (top64 >> 11) / 2**53
vacate iff u < p
```

**The `>> 11` is mandatory and is a departure from the source** (which used
`top64 / 2**64`).

The reason is *not* that the source's expression diverges across the two languages. It was
measured, and it does not: `int / 2**64` in CPython is a single correctly-rounded division,
while `Number(bigint) / 2**64` in JS rounds to float64 and then divides by a power of two —
which is exact — so the two agree. This was checked on 200 006 values, including random 64-bit
integers and hand-picked ties at the rounding boundary (`(1<<63)|((1<<11)-1)`, `(1<<64)-1`, and
neighbours). Every one matched bit for bit.

The reason is that `(top64 >> 11) / 2**53` needs **no such argument**. A 53-bit integer over
2⁵³ is exactly representable, so `u` is self-evidently the same double in both languages, and
the property survives a reimplementation that assembles the value differently — from two 32-bit
halves, say, where the rounding argument above stops holding. A cross-language equality that
depends on a subtle proof is one refactor away from being false; this one does not.

- TS: `Number(BigInt("0x" + hex.slice(0, 16)) >> 11n) / 2 ** 53`
- Py: `(int.from_bytes(digest[:8], "big") >> 11) / 2 ** 53`

`p` is compared as given. Callers must pass the identical double; the UI emits `p` at two
decimal places and both stacks parse it as float64.

**Nesting.** `u` is a function of `(seed, stem)` alone — not of `p`, not of traversal order, not
of which other words exist. So `{stems vacated at p} ⊆ {stems vacated at p'}` for `p < p'`,
which is the first of the two properties that make a `p`-sweep interpretable.

---

## 5. Minting — stability

### 5.1 Why the source's minter cannot be ported as written

The source mints with `random.Random(f"{seed}:{k}")` and then guards uniqueness with a
`used` set on a long-lived `Minter`. Two consequences, both of which break the stability
property the source claims for itself:

1. **`used` is order-dependent.** If stem *A* mints `flim` and stem *B* would too, *B* retries —
   but only if *A* was minted first. At `p = 0.5` only *B* may be vacated, so *B* gets `flim`;
   at `p = 1.0` both are, so *B* gets something else. The nonce for a word then depends on `p`,
   which is exactly what stability forbids.
2. **The give-up path is order-dependent.** After 400 failed attempts the source returns
   `syllable + str(len(self.used))` — a counter of how many words happened to be minted before.

There is also the practical problem that Python's Mersenne Twister seeded from a string is not
reproducible in TypeScript without reimplementing MT19937 and `Random.choice`'s masking.

### 5.2 What we do instead

**The map is built once over the whole type set, in a canonical order, independent of `p`.**
The map at any `p` is then the restriction of that single map to `{stem : u(stem) < p}`.
Nesting and stability become structural facts rather than properties to be hoped for.

```
buildVacancyMap(types, seed, matchProsody, avoid) -> Map<stem, nonce>
    stems := sorted({ stemOf(t) for t in types if eligible(stemOf(t)) })   # ASCII sort, ascending
    used  := {}
    for stem in stems:                       # canonical order — never p, never document order
        nonce := mint(stem, seed, matchProsody, forbidden = used ∪ avoid)
        used.add(nonce); map[stem] = nonce
```

`types` is the union of the corpus's type set and the budget's word list (§7.2 explains why the
budget must be in the domain). `avoid` is the lowercased corpus type set, so **a nonce can
never collide with a real word of the corpus** — the source accepts an `avoid` parameter and
then never passes one, which lets a minted form silently merge with an English type.

### 5.3 The deterministic byte stream

`random.Random` is replaced by a sha256 counter stream, which is trivially identical in both
languages:

```
bytesFor(seed, stem, salt, counter) = sha256(utf8(f"{seed}:mint:{stem}:{salt}:{counter}"))
```

Consume the stream 4 bytes at a time, big-endian, as an unsigned 32-bit integer; refill from
the next `counter` when exhausted. A choice from a list is `list[nextU32() % len(list)]`.
(All lists here are shorter than 256, so the modulo bias is aesthetic, not statistical — but
both stacks must bias *identically*, which they do.)

TS: `((b0 << 24) | (b1 << 16) | (b2 << 8) | b3) >>> 0`. The `>>> 0` is required; without it JS
produces a negative number and `%` returns a negative index.

### 5.4 The phonotactic tables

Ported verbatim from the source, order significant (the index into each list is what the byte
stream selects, so reordering silently changes every nonce):

- `ONSETS` — 47 entries, `b … sq`
- `NUCLEI` — 19 entries, `a … er`
- `CODAS` — 49 entries, beginning with the empty string
- `UNSTRESSED_TAILS` — 13 entries, `y … ing`
- unstressed-onset prefixes — `["a", "be", "re", "de", "un", "en"]`
- reduced coda set for unstressed syllables — `["", "", "l", "n", "r", "s"]` (the duplicated
  empty string doubles its weight; keep it)

### 5.5 The mint loop

For `salt = 0, 1, 2, …`:

1. `pattern` := the stem's stress pattern (§6) when `matchProsody`, else `"1"`.
   `nSyl` := `len(pattern)`.
2. Build a candidate: for syllable `i`, if `pattern[i] == "1"` emit
   `choice(ONSETS) + choice(NUCLEI) + choice(CODAS)`; else if `i == 0` emit
   `choice(prefixes)`; else emit `choice(UNSTRESSED_TAILS)`. Stressed syllables in a
   non-initial unstressed position use the reduced coda set.
3. Collapse runs: `re.sub(r"([bcdfghjklmnpqrstvwxz])\1{2,}", r"\1\1", w)`.
4. Accept iff `len(w) >= 3` **and** `w ∉ forbidden` **and** `syllables(w) == nSyl`.
5. On `salt >= 400`, drop the syllable-count check. On `salt >= 800`, drop the length check.
   These relaxations are deterministic and order-independent, unlike the source's counter.
   Reaching `salt >= 1200` raises — it has never happened and if it does we want to know.

### 5.6 Stability

`mint` depends only on `(seed, stem, matchProsody, forbidden)`, and `forbidden` depends only on
the canonically-ordered prefix of stems before it. Nothing depends on `p`, on the document, or
on the order words are encountered while rewriting. Therefore **a stem's nonce is the same at
every `p`** — the second property the `p`-sweep needs.

### 5.7 Seams

Re-attaching a suffix can produce a seam (`wee` + `er` → `weeer`). When
`nonce[-1] == suffix[0]`, replace the nonce's last character with
`"lnrtk"[u32(sha256(f"{seed}:seam:{stem}:{suffix}")) % 5]`. Deterministic, order-independent;
the source used a shared RNG here, which is order-dependent.

---

## 6. Prosody

### 6.1 Provenance — read this before quoting a number

`STRESS_TABLE` is ported verbatim from `tiny-seuss/synth/lexicon.py` (61 polysyllables of the
Dolch list). The source describes it as **"seeded by rule and then overridden by a hand table"**
and its own status table lists the stress table under *not yet exercised*: *"seeded by rule;
wants roughly an hour of human checking."*

So: **we do not claim exact prosody, and no UI string may.** The doc's argument that a closed
lexicon *buys* exact prosody is sound in principle and false of this table today. Two
consequences, both mandatory:

1. The table covers Dolch words. The shipped corpus is *The Real Mother Goose* with ~2 200
   types, most of which are not in it, so most words fall through to the spelling rule.
2. Every prosody statistic the UI shows must be accompanied by `stressTableCoverage` — the
   fraction of tokens whose stress came from the hand table rather than the rule. That number
   is the honesty of every other prosody number on the panel.

### 6.2 Syllable rule (fallback)

```
w := lower(word), strip leading/trailing "'" and "-", then delete every non [a-z]
if w is empty: return 1
n := count of matches of /[aeiouy]+/ in w
if w ends with "e" and n > 1 and w does not end with "le" | "ee" | "ye": n -= 1
return max(1, n)
```

The source has a further `if w.endswith("le") …: pass` branch. It is dead code — a `pass` — and
is *not* ported; the behaviour above is byte-identical to the source's.

### 6.3 `stress(word)`

Lookup order, exactly:

1. `MINTED_STRESS[lower(word)]` — the intended pattern of a form we minted ourselves, so
   prosody scoring on a vacated corpus is exact *for the minted forms*
2. `STRESS_TABLE[word]` — **case-sensitive**, for `Christmas`
3. `STRESS_TABLE[lower(word)]`
4. the rule: `"1"` if `n == 1` else `"1" + "0" * (n - 1)`

`syllables(word) := len(stress(word))`.

### 6.4 Meter

`meterScore(line, foot)` = the fraction of syllable positions in the line's concatenated stress
string that match the repeating foot, `0.0` for a line with no syllables.
`anapest = "001"`, `iamb = "01"`, `trochee = "10"`, `dactyl = "100"`. Reported as the mean over
lines that produce at least one token.

---

## 7. Conditions, vocabulary, and the invariance theorem

### 7.1 Parameters

| name | type | default | meaning |
|-|-|-|-|
| `p` | float ∈ [0,1] | `0` | fraction of eligible **types** vacated |
| `seed` | int | `0` | selects both `u` and the nonce assignment |
| `consistent` | bool | `true` | one nonce per source type, corpus-wide |
| `matchProsody` | bool | `true` | nonce carries the stem's syllable count and stress |
| `revealAfter` | int | `0` | first N occurrences of a vacated stem keep the English form |
| `keep` | set | `{}` | extra words added to the closed class |

`consistent = false` derives the nonce from `(stem, occurrenceIndex)` in document order, so
every occurrence is a fresh type. This condition deliberately has **no** stability property;
it is the source's "inconsistent assignment" control and its purpose is to destroy the field
while holding the vacancy rate fixed.

### 7.2 Vocabulary under vacancy

Two rules, and which applies depends on the condition:

**Mapped vocabulary** (`consistent = true`, `revealAfter = 0`) — the budget's word list is
pushed through the *same* `transformWord`, **preserving order**. Since the map is injective,
`itos_p = SPECIALS ++ [transformWord(w) for w in words]` assigns every word the id its
pre-image had. This is why the map's domain must include the budget's words as well as the
corpus's types (§5.2): a budget word absent from the corpus still needs an image.

**Rebuilt vocabulary** (every other condition) — the budget is rebuilt from `C_p` by the tab's
normal rule (the Dolch list as-is, or `frequencyBudget(C_p, N)`). Coverage then collapses, and
the collapse is the measurement.

### 7.3 The invariance theorem

> **Theorem.** With `consistent = true` and `revealAfter = 0`, and the vocabulary mapped as
> above, for every `p`, `seed`, budget, and value of `matchProsody`:
>
> `tokenStream(vacate(C, p), V_p)` equals `tokenStream(C, V)` element for element.
>
> **Corollary.** Training is bit-identical: `runTraining` is a function of
> `(cfg, tokens, seed, hyperparameters)` and `cfg` depends on the vocabulary only through
> `vocabRows`, which is unchanged.

Why it holds: §1 gives a bijection between word occurrences that preserves order and line
structure; §5.2 makes the type map injective on the union domain; §7.2 makes `stoi_p(map(w))`
= `stoi(w)` for budget words and sends every non-budget type to a non-budget type, so
`<unk>` lands in exactly the same places.

**The theorem is a test, not a comment** (SC-703). It is asserted on the real corpus across all
five Dolch budgets and a frequency budget, `p ∈ {0, 0.25, 0.5, 0.75, 1}`, `seed ∈ {0, 7}`, and
both settings of `matchProsody`. If a future change to the tokenizer, the suffix list, or the
minter breaks it, that test fails.

**Injectivity is verified, not assumed.** After building the map, compute the image of the
actual type set. Two distinct source types can in principle collide through the
stem+suffix construction (`nonce(a) + "s"` colliding with `nonce(b)`), which the `used` set
does not rule out. If `|image| < |types|`, re-mint the colliding stems at a higher salt in
canonical order and repeat; raise after 8 rounds. In practice this loop runs zero times — but
the theorem depends on it, so it is checked every build and reported as `bijective` in the
statistics.

### 7.4 What this predicts, stated before we measure it

Three of the knobs — `p`, `seed`, `matchProsody` — are **invisible** to a word-level model
trained from scratch. Only the knobs that break type identity (`consistent = false`,
`revealAfter > 0`) can change a loss. That is the honest tiny-arm result, and it is worth
saying plainly rather than dressing a null up as a curve: *for this model class, all of a
word's meaning is field and none of it is form.* The prosody control matters for the corpus as
an artifact and not at all for this model — which is itself a finding about what a word-level
model can see.

---

## 8. The pretrained arm (Architecture Explorer)

The tiny arm has no model with a location, so it cannot say what a location is worth. This arm
does: it runs a **real pretrained model** over a passage and its vacated twin.

### 8.1 The measurement

Mean negative log-likelihood, in nats per token, restricted to the tokens of **preserved**
words — the closed-class scaffolding, which is character-identical in both passages. Formally,
`the __ __ did __ and __`: does a model that knows English still predict the scaffolding when
the content is vacant? Carroll's claim is that a reader does; this puts a number on it.

Reported per passage (English `E`, vacated `J`):

- `nllPreserved` — the headline; mean NLL over tokens belonging to preserved words
- `nllAll` — every scored token, for context
- `bitsPerChar` — `nllAll * nTokens / (ln 2 * nChars)`, comparable across tokenizations
- `nTokens`, `nPreservedTokens`, `nChars`

and the deltas `ΔnllPreserved = nllPreserved(J) - nllPreserved(E)`.

### 8.2 Alignment

Tokens must be attributed to words. **Determine empirically which mechanism the installed
transformers.js actually provides before writing the implementation** — offsets if the
tokenizer exposes them, otherwise incremental decode with a character cursor. Whichever is
used, it is verified by reconstructing the passage from the token spans and asserting equality
with the input; a mismatch raises rather than mis-attributes.

Do not tokenize word-by-word to force alignment. It suppresses cross-word merges and changes
the NLL being reported.

### 8.3 The swap control — what makes the number interpretable

`ΔnllPreserved > 0` on its own is uninterpretable, because at least three things change at once
when content words are vacated:

1. the forms are unknown, so the model has no lexical entry to condition on;
2. nonce forms fragment into many subword tokens, so the context is longer and stranger;
3. the passage says something nonsensical.

Only (1) is "location". A caveat cannot separate them; a control can.

**Swap.** Mint by drawing a *real English word* instead of a nonce form — same eligibility, same
`u(stem) < p` decision, same suffix handling, same map-injectivity guarantee. The replacement is
drawn deterministically from the corpus's own open-class types by **frequency rank**: the stem's
rank `r` among open-class types selects a replacement from a deterministic window around `r`,
excluding the stem itself and anything already used. So the swapped passage is equally
nonsensical, but every form is a known word with ordinary tokenization.

This makes the minting strategy a parameter:

```
mint: "nonce" | "swap"          # default "nonce"
```

and decomposes the measurement:

- `nll(swap) − nll(english)` — the cost of **wrong content** (3)
- `nll(nonce) − nll(swap)` — the cost of **unknown form**, i.e. (1) together with (2)

The second difference is the closest this instrument gets to "what location was worth", and the
UI must report it as *that difference*, never `nll(nonce) − nll(english)` alone. Residual (2) is
not separable without a tokenizer-level control and the UI must say so rather than pretend the
remainder is pure location.

`mint: "swap"` preserves every property of §7: the map is still injective (verified the same
way), still nested in `p`, still stable in `(seed, stem)`. The invariance theorem of §7.3
therefore holds for `swap` exactly as for `nonce` — the tiny model is equally blind to both,
which is itself the check that the swap control is implemented correctly.

### 8.4 Confound, stated in the UI

The vacated passage has genuinely higher entropy, so *every* prediction in it gets worse —
including the scaffolding. `ΔnllPreserved > 0` is therefore expected; its **magnitude** is the
result, and it is only interpretable against the tiny arm's exact zero. The UI must show both
arms together, and must not present `ΔnllPreserved > 0` as a surprise.

Both stacks (backend PyTorch, static transformers.js) must produce the same numbers for the
same model and passage, to the tolerance the existing arch parity tests use.

---

## 9. Deliberate departures from the source

| # | Source | Here | Why |
|-|-|-|-|
| 1 | `[A-Za-z][A-Za-z']*` | the tokenizer's `WORD_RE` | otherwise the transform and the trainer disagree about `good-bye` and §7.3 is false |
| 2 | `top64 / 2**64` | `(top64 >> 11) / 2**53` | not a bug in the source — the two languages were *measured* to agree on it (§4). Exact representability makes the agreement structural instead of a proof that a refactor could invalidate |
| 3 | `random.Random(str)` | sha256 counter stream | MT19937 seeded from a string is not reproducible in TS |
| 4 | map built lazily while rewriting | map built once over all types in canonical order | the source's `used` set and give-up counter make the nonce depend on `p`, breaking its own stability claim |
| 5 | `avoid` accepted, never passed | `avoid` = corpus type set | a minted form could otherwise merge with a real English type |
| 6 | give-up = `syllable + str(len(used))` | deterministic salt relaxation | order-independence |
| 7 | seam fix via shared RNG | seam fix via hash of `(stem, suffix)` | order-independence |
| 8 | injectivity assumed | injectivity verified, re-mint on collision | §7.3 depends on it |
| 9 | zip copy of `split_suffix` | audited copy's `EXCEPTIONS` | `brother → broth+er` is a known artifact the audited copy fixes |
| 10 | "exact prosody" | measured prosody + `stressTableCoverage` | the source's own status table calls the stress table unverified |
| 11 | `vacated` count reported as `len(self.map)` | count of stems actually vacated | the zip copy reports the wrong number; the audited copy fixes it |

Departures 4, 6, 7 and 8 are corrections to bugs that break properties the source *claims*.
Departure 2 is **not** a bug fix — the source's expression was tested and is fine; the change
buys structural rather than argued cross-language equality.

---

## 10. Statistics contract

`vacancyStats(originalText, vacatedText, map, p, seed, …)` returns, with these exact names:

```
typesTotal, typesEligible, typesVacated,
tokensTotal, tokensVacated,
meanSyllablesBefore, meanSyllablesAfter,
meanAnapestBefore,  meanAnapestAfter,
stressTableCoverageBefore, stressTableCoverageAfter,
bijective, imageSize, remintRounds
```

Both stacks compute these from the same definitions; the golden fixture (§11) pins them.

For reference, the source reports mean anapest `0.351 → 0.345` and mean syllables
`1.224 → 1.211` on *its* corpus. **Those are its numbers on a corpus we do not have. Do not
transcribe them into any UI string, test, or doc.** Compute ours on Mother Goose and quote
only what we measured — this is the same rule that caught the fabricated "+29.3 …
decelerating" in feature 006.

---

## 11. Golden fixture

`code/frontend/tests/fixtures/vacancy-golden.json`, generated by
`scripts/export_vacancy_golden.py`, consumed by `code/frontend/tests/unit/vacancyGolden.test.ts`.
Same shape and discipline as `lex-golden.json`: `format`, `tolerance`, `git_sha`, generator
versions, then cases.

Pinned per case, on the **real committed corpus**:

- `u(stem)` for a fixed list of 24 stems spanning eligible/ineligible and both budgets — the
  exact float64, which is the whole point of departure 2
- the full `map` at `seed ∈ {0, 7}` — every stem→nonce pair
- the first 400 characters of the vacated corpus at `p ∈ {0, 0.35, 0.7, 1}`, seed 0 — this is
  the source's own figure, reproduced on our corpus
- `vacancyStats` for each of those, all fields
- the nesting assertion: `vacated(0.35) ⊆ vacated(0.7) ⊆ vacated(1.0)` as explicit id sets
- the stability assertion: the nonce for each of 24 stems is identical at every `p` where it
  is vacated
- the token id stream digest under the mapped vocabulary at each `p` — all equal, which is
  §7.3 pinned as data rather than as an assertion in one language

Strings are compared exactly. Only the prosody means use `tolerance`.
