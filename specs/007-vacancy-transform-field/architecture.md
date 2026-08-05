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

A word is eligible for vacancy iff **the whole word** passes test 1 and, after suffix
splitting (§3), its **stem** passes tests 2–4:

1. `lower(word) ∉ keepSet` — the WHOLE word, before any splitting
2. `lower(stem) ∉ keepSet`
3. `stem` matches `^[A-Za-z]+$` — ASCII letters only
4. `len(stem) > 2`

`isVacatable(word)` is tests 1–4; `isEligible(stem)` is 2–4 and is the stem-level predicate
only. Both stacks export both, and every caller that judges a WORD or a TYPE calls the former.

**Test 1 is not redundant, and it was missing.** §3's splitter is a spelling heuristic, so it
breaks the closed class open: `after → aft + er`, `this → thi + s`, `does → doe + s`, and the
same for `always`, `during`, `having`, `unless`. None of those stems is a function word, so the
stem tests passed all seven and they were vacated — at seed 0 `after` came out as `kitser` —
while §0 claims the closed-class scaffolding survives character for character and §8 takes its
whole measurement over the words that survive. Protection has to be applied to the word the
reader keeps, not to the fragment the splitter happens to produce. Measured cost of the fix on
the shipped corpus: `domainTypesEligible` 1944 → 1940, `corpusTypesEligible` 1922 → 1918,
`stemsTotal` 1680 → 1676, `tokensVacated` at `p = 1` 8202 → 8125.

Test 3 is why hyphenated and apostrophised words behave as they do, and both stacks must
agree on it exactly:

- `good-bye` — no suffix matches, stem is `good-bye`, which contains a hyphen, so **test 3
  fails and the word is never vacated**.
- `don't` — `n't` does *not* split it, because §3's length rule requires
  `len(word) - len(suffix) >= 3` and `5 - 3 = 2`. The stem is therefore `don't`, which contains
  an apostrophe and fails test 3. Never vacated. (An earlier draft of this document said the
  suffix splits it to `do`; that was wrong about the mechanism, though right about the outcome.)
- `dog's` — `'s` splits it (`5 - 2 = 3`) to stem `dog`, which passes; output is `<nonce>'s`.

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

**The seed is bounded: `|seed| ≤ 2⁵³ − 1 = 9007199254740991`, enforced in both stacks.** The
care above is spent on the digest's *output*; the same care is owed to its *input*. The digest
is taken over `f"{seed}:{stem}"`, and Python stringifies an arbitrary-precision integer exactly
while JavaScript stringifies the nearest float64 — so at `2⁵³ + 1` Python hashes
`"9007199254740993:little"` and JavaScript hashes `"9007199254740992:little"`, and the two
stacks build entirely different maps, vacate different corpora, and report different statistics
with nothing raised on either side. Measured: `2⁵³` agrees (it is representable); `2⁵³ + 1`,
`2⁵³ + 3`, `−(2⁵³ + 1)` and `12345678901234567890` diverge in every field.

`MAX_SEED = 2**53 - 1` is checked in `VacancyParams` **and** in `u` itself, since `u` is public
and a caller can reach it directly. It **raises**; it does not clamp. A clamp would use a seed
nobody asked for — the same defect one level up, and the browser's seed box already had it
(typing `9007199254740993` silently became `9007199254740992`). `Number.isInteger` is not a
sufficient guard on the JavaScript side: `9007199254740993` passes it, having already been
rounded on the way in.

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
buildVacancyMap(types, params) -> Map<stem, nonce>
    domain := { lower(t) for t in types }
    stems  := sorted({ stemOf(t) for t in domain if eligible(stemOf(t)) })  # ASCII sort, ascending
    used   := {}
    for stem in stems:                       # canonical order — never p, never document order
        nonce := mint(stem, seed, matchProsody, forbidden = used ∪ domain)
        used.add(nonce); map[stem] = nonce
```

**There is no caller-supplied `avoid` parameter.** The domain is always avoided, implicitly.

The first implementations both gave `avoid` a default of empty and left it to the caller to pass
the type set. Both stacks agreed with each other, so no parity test caught it — but the map is
then a function of *what the caller remembered to pass*. Measured: at seed 0 the same corpus and
seed produce `remintRounds = 0` with the domain passed and `1` without, and **different nonces**
either way. Both maps are valid; that is the problem. One caller passing it and another not —
the panel and the golden fixture, say — is a silent divergence with no failing test.

Since condition B below already requires that no surface form equal any domain type, avoiding
the domain at mint time is not an extra policy, only the cheaper way to reach the same fixed
point. Making it implicit costs nothing and makes the map a pure function of
`(domain, seed, matchProsody)`.

**The domain is `corpus types ∪ the full Dolch list`** — the *full* list, always, never the
active budget. §7.2 explains why budget words must be in the domain at all; the reason it is the
full list is that the domain must not depend on which budget the reader has selected, or
switching budgets would re-mint the corpus in front of them and the stability the panel is
demonstrating would look false. A frequency budget needs no special case, since its words are
corpus types by construction.

Making the domain the forbidden set (below) turns this from a convenience into a requirement,
and the earlier draft of this paragraph is now wrong in an instructive way. It said the map was
"identical across all five Dolch domains" — true when `avoid` was a caller-passed corpus type
set independent of the domain, false now. A **smaller domain forbids less**, so it mints
differently: building over `corpus ∪ dolch_budget(name)` for any name below `full` moves exactly
one stem, `jam → floor` instead of `scirmp`, because `floor` is a full-list Dolch word that
never occurs in *Mother Goose* and so is only forbidden when the full list is in the domain.

That is a *reason* the domain must be the full list rather than a coincidence that it may be.
`vacancyDomain` makes the smaller domains unreachable, and the test asserts the property that
actually matters — **the map does not move when the active budget changes** — rather than the
stronger claim that happened to hold before.

The source accepts an `avoid` parameter and then never passes one, which lets a minted form
silently merge with an English type. We do not repeat that by making it optional — see above.

**Both stacks expose a `vacancyDomain(types)` / `vacancy_domain(types)` helper** that applies the
union rule, and every call site uses it. Python had one and TypeScript did not, which is the kind
of asymmetry that ends with two call sites building the domain two different ways.

Both take an **iterable of types, not a text**. Python must reject a bare `str` explicitly:
`Iterable[str]` happily accepts one and iterates it character by character, so
`vacancy_domain(corpus_text)` silently yields a domain of single letters. Raise a `TypeError`
naming `tokenize()`, rather than returning a subtly wrong answer.

**The check is over surface forms, not bare nonces, and it must hold at every `p`.** This is
the second defect the first implementation exposed. A bare-nonce check is not enough:

> At `seed = 7`, the stem `hang` minted the nonce `wak`. No corpus type equals `wak`, so a
> bare-nonce `avoid` check passes. But the corpus contains `hanged`, whose surface form is
> `wak` + `ed` = `waked` — and the corpus *also* contains the English word `waked`. At `p = 1`
> both are vacated and nothing collides, which is why a check performed only at full vacancy
> sees nothing. At `p = 0.25` and `p = 0.5`, `hanged` is vacated and `waked` is not, so two
> distinct source types both map to `waked`. Injectivity fails, and with it §7.3.

So define, over the domain:

- `pairs` := `{(stem, suffix)}` for every domain type whose stem is eligible
- `surface(stem, suffix)` := the assembled, lowercased output of §5.7

and require **both**:

- **A.** the surface forms are pairwise distinct
- **B.** no surface form equals any lowercased type in the domain — whether or not that type is
  itself eligible

B is deliberately conservative: it forbids a minted form from equalling a word that would always
have been vacated alongside it. That costs a re-mint and buys a condition that is independent of
`p`, which is what the theorem needs. Because un-vacated words map to themselves, minted words
map into the surface set, and A and B keep those two sets internally distinct and mutually
disjoint, injectivity holds **simultaneously for every `p`** rather than at `p = 1` only.

On violation, re-mint the offending stems at a higher salt in canonical order and re-check;
raise after 8 rounds.

### 5.2a What A and B are standing in for, and what `mint = "swap"` can therefore satisfy

B is **sufficient, not necessary**. Writing the necessary condition out is what makes the swap
control of §8.3 statable at all, because swap draws its replacements *from* the domain and so
violates B by construction.

Fix `p` and let `T_p` be the type map: `T_p(t) = surface(t)` when `t`'s stem is eligible and
`u(stem(t)) < p`, and `T_p(t) = t` otherwise. `T_p` is injective for **every** `p` iff both:

- **A.** the surface forms are pairwise distinct — unchanged; and
- **B′.** for domain types `t₁` (eligible stem) and `t₃`, if `surface(t₁) = t₃` then `t₃`'s stem
  is eligible **and** `u(stem(t₃)) < u(stem(t₁))`.

*Why.* The only way two types can merge is that one moved onto another that had not: an image
`surface(t₁)` colliding with an un-vacated `t₃`. That is possible at some `p` exactly when
`u(stem(t₁)) < p ≤ u(stem(t₃))`, i.e. exactly when `u(stem(t₃)) ≥ u(stem(t₁))` — which B′ forbids.
The case `t₃ = t₁` is included, so B′ also rules out a type that silently fails to vacate (the
`tak → tak` defect of §5.8). Image-on-image collisions are A.

**B ⟹ B′ vacuously** (B makes B′'s premise unsatisfiable), so nothing about the nonce strategy
changes and nothing in this document about it is weakened. B stays the *enforced* rule for
`mint = "nonce"`: it is cheaper to check, it costs one re-mint on the shipped corpus, and it is
the reason §7.3 holds simultaneously at every `p`.

**Theorem (why `swap` cannot have that).** Suppose the map is stable in `p` (§5.6) and every
image is a domain type — both true of swap by construction. `T_p` injective for every `p` forces
`T_p` to be a bijection of the domain onto itself, hence to map the vacated set `V_p` onto `V_p`.
The `V_p` are nested and grow one *stem family* at a time, so a bijection mapping every `V_p`
onto itself maps each stem family onto itself: `u(stem(σ(s))) = u(stem(s))`, hence `σ = id`.
**So no non-trivial swap is injective at intermediate `p`.** Measured, to make it concrete rather
than merely proved, **at seed 0, under this counting definition**:

> **lost image slots** `= |domain| − |{T_p(t) : t ∈ domain}|` — how many of the domain's 2 233
> slots the type map at `p` fails to reach. It is what `lex-vacancy-lost-slots` measures live in
> the panel, and it is what both engines' tests pin.

The frequency-rank swap of §8.3 loses **349 / 484 / 364** slots at `p = 0.25 / 0.5 / 0.75` on
the shipped corpus at seed 0 (**336 / 475 / 372** at seed 7), and **0** at `p ∈ {0, 1}`.

*This sentence previously read "191 / 246 / 190 colliding types", with neither a seed nor a
counting definition attached, and no test read it. It did not reproduce under any of the three
natural readings at any of twelve seeds; the numbers above are measured by
`test_swap_collisions_at_intermediate_p_are_measured_under_one_definition` (Python) and
`§5.2a: swap's lost image slots, measured under one stated definition` (TypeScript), so a
figure quoted here can no longer rot unobserved. A "measured" claim in this document must
name its seed and its definition and be pinned by a test — the same rule §10 states for the
source document's prosody figures.*

What swap *can* satisfy, and does, is the condition at **full vacancy**, where the un-vacated set
is exactly the ineligible types:

- **B₁.** no surface form equals an **ineligible** domain type, and no surface form equals its own
  source type.

A + B₁ make `T_1` a bijection of the domain, so the invariance theorem of §7.3 holds for
`mint = "swap"` at `p ∈ {0, 1}` — and *provably cannot* hold at `0 < p < 1`. The engine therefore
**refuses** the mapped vocabulary of §7.2 for `swap` at intermediate `p`, with a typed error
naming this theorem, rather than shipping a vocabulary with two words on one row. `vacateText`
itself is unrestricted: the pretrained arm of §8.3 measures a passage, and a passage does not
need an injective map. `VacancyMap.injectiveAtEveryP` reports which of the two regimes a map is
in — `true` for nonce, `false` for swap — so no caller has to infer it.

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
- `CODAS` — **46** entries, `"" … zzle`
- `UNSTRESSED_TAILS` — 13 entries, `y … ing`

An earlier draft said `CODAS` had 49. It has 46 — I miscounted. Both implementations copied the
source verbatim, flagged the discrepancy, and correctly took "verbatim" over the tally, so
nothing diverged. **The lists are normative; the counts here are commentary.** If they ever
disagree again, the source lists win, because a nonce is a function of the strings and their
indices and not of a number in a document.
- unstressed-onset prefixes — `["a", "be", "re", "de", "un", "en"]`
- reduced coda set for unstressed syllables — `["", "", "l", "n", "r", "s"]` (the duplicated
  empty string doubles its weight; keep it)

### 5.5 The mint loop

A mint call carries a **base salt** `S` (0 for the first build; §5.8 sets it for re-mints) and
runs an **attempt counter** `a = 0, 1, 2, …`. The byte stream of §5.3 is keyed on
`salt = S + a`; **the quality thresholds below are on `a`, not on `salt`.**

That distinction is load-bearing. Read the other way — thresholds on the absolute salt — a
re-mint at `S = 1001` would begin with every quality check already relaxed, so the replacement
nonce would not be prosody-matched, and a second round at `S = 2001` would exceed the give-up
bound and raise, contradicting §5.2's "raise after 8 rounds". Counting attempts per call means a
re-mint is held to exactly the same standard as an original mint, which is what makes the
seed-7 replacement (`hang → smeeg`) monosyllabic like the word it replaces.

For `a = 0, 1, 2, …`:

1. `pattern` := the stem's stress pattern (§6) when `matchProsody`, else `"1"`.
   `nSyl` := `len(pattern)`.
2. Build a candidate: for syllable `i`,
   - if `pattern[i] == "1"` emit `choice(ONSETS) + choice(NUCLEI) + choice(CODAS)`
   - else if `i == 0` emit `choice(prefixes)`
   - else emit `choice(UNSTRESSED_TAILS)`

   Exactly three branches, in that order. An earlier draft added a sentence about "stressed
   syllables in a non-initial unstressed position", which is self-contradictory — a syllable is
   stressed or it is not. **The reduced coda set of §5.4 is therefore unreachable**, exactly as
   it is in the source, where `_syl(stressed=False)` is never called. It is retained in §5.4 for
   fidelity to the source's tables and because the byte stream's list indices must not shift.
   **Do not "fix" this in either stack** — doing so would change every multi-syllable nonce.
3. Collapse runs: `re.sub(r"([bcdfghjklmnpqrstvwxz])\1{2,}", r"\1\1", w)`.
4. Accept iff `len(w) >= 3` **and** `w ∉ forbidden` **and** `syllables(w) == nSyl`.
5. On `a >= 400`, drop the syllable-count check. On `a >= 800`, drop the length check.
   These relaxations are deterministic and order-independent, unlike the source's counter.
   Reaching `a >= 1200` raises — it has never happened and if it does we want to know.

### 5.6 Stability

`mint` depends only on `(seed, stem, matchProsody, forbidden)`, and `forbidden` depends only on
the canonically-ordered prefix of stems before it. Nothing depends on `p`, on the document, or
on the order words are encountered while rewriting. Therefore **a stem's nonce is the same at
every `p`** — the second property the `p`-sweep needs.

### 5.7 Seams, and the case-commuting invariant

Re-attaching a suffix can produce a seam (`wee` + `er` → `weeer`). When
`nonce[-1] == suffix[0]`, replace the nonce's last character with
`"lnrtk"[u32(sha256(f"{seed}:seam:{stem}:{suffix}")) % 5]`. Deterministic, order-independent;
the source used a shared RNG here, which is order-dependent.

**Everything in the transform is computed on the lowercased word.** The stem and the suffix are
lowercased before the seam test, before the seam hash, and before the surface form is assembled;
`matchCase` is then applied **to the whole assembled surface form**, with the *original whole
word* as the case source.

This is not a stylistic preference. The first implementation of this contract followed the
source and sliced the suffix case-preserved, then ran the seam test against it. So `gums` →
`flels` while `GUMS` → `FLESS`: `suffix[0]` was `s` in one and `S` in the other, the seam test
fired in one and not the other, and one source **type** acquired two distinct surface forms.
The tokenizer lowercases, so those are two different types — and §7.3 is false.

> **Invariant (normative, and a test).** For every word `w`:
> `lower(transformWord(w)) == transformWord(lower(w))`.
>
> The transform must **commute with lowercasing**, because the tokenizer lowercases. Any step
> that branches on a character's case — seam tests, hash inputs, table lookups other than the
> deliberate case-sensitive `STRESS_TABLE` probe of §6.3 — violates it. Assert it over the whole
> real corpus, and over each type upper-cased, capitalised, and lower-cased.

---

### 5.8 Details the first implementation had to invent — now pinned

These were gaps, not choices. Both stacks do it this way or the golden fixture fails.

- **Re-mint selection.** When conditions A/B of §5.2 fail, re-mint **only the losing stem** —
  the one later in canonical (ASCII-ascending) order among those involved in the collision — at
  **base salt** `1000 * round + previousBaseSalt + 1`. Round counts from 1. Winners keep their
  nonce, so a re-mint never cascades. The attempt counter restarts at `a = 0` inside the new
  call, so §5.5's quality checks apply in full (see the note there).
- **`consistent = false` key, and where its prosody comes from.** The per-occurrence nonce is
  minted for the key `f"{stem}#{idx}"`, `idx` being the 0-based occurrence index of that **stem**
  in document order. The `#` is not a legal `WORD_RE` character, so the key can never collide
  with a stem.

  **Condition B applies to the per-occurrence path too.** It was enforced for the map and not
  for this control, and the gap is observable: at seed 7, `p = 1`, the stem `tak` minted the
  nonce `tak`, so `Taking → Taking` — a token that silently failed to vacate, leaving
  `corpusTypesVacated` one short of the consistent path's 1918. §7.1 says this control has no
  *stability* property, which is about a nonce being reused across occurrences; it does not
  license a word quietly surviving the transform. A control whose vacancy rate is not actually
  the stated rate is not a control. So a per-occurrence nonce must equal neither any domain type
  **nor the stem it replaces**, and the same re-mint loop applies.

  **The key feeds the byte stream and the uniqueness check only. The stress pattern comes from
  `stress(stem)`, never from `stress(key)`.** This was under-specified and the two stacks split
  on it: one passed the key into the minter, so the pattern became `stress("little#0") = "10"`
  instead of `stress("little") = "100"`, and `Little` minted as `Wrerken` rather than
  `Wrerkenle`. §7.1 says the nonce carries *the stem's* syllable count and stress, so the key
  must not reach the prosody lookup. Caught by the golden fixture, not by either test suite.
- **Minted stress is passed, never global.** `stress(word, mintedStress)` takes the map as an
  argument. The source used a module-level `MINTED_STRESS` dict mutated by `register_minted`,
  which makes two concurrently-live maps corrupt each other — and the Lexicon Lab holds several
  at once (one per condition being compared).
- **`forbidden` is STORED, not reconstructed, and includes superseded re-mint nonces.** One
  stack stored it; the other rebuilt it as `domain ∪ mapping.values()`, which silently drops
  every nonce that a re-mint round replaced (`wak` at seed 7). Nothing observable diverges from
  that today — all digests agree — but the two sets are genuinely different, and the
  per-occurrence path of the `consistent = false` control now *draws against* `forbidden`, so it
  is one unlucky hash away from mattering. Superseded nonces stay forbidden because they were
  rejected for a reason: reusing one can recreate the very collision the re-mint resolved.
  **Both stacks now store it**, and both assert `"wak" ∈ forbidden` at seed 7 — the superseded
  nonce of `hang`, which re-minted to `smeeg`. It is the one case the shipped corpus produces, so
  it is the one the test names; a reconstruction from `mapping.values()` fails that assertion.
  The re-mint loop draws against the same accumulated set, so a re-mint can never hand a stem a
  form some other stem has already given up.
- **`VacancyMap`'s stem→nonce field is `mapping` in both stacks.** TypeScript called it `map`
  and Python `mapping`, which is the third naming asymmetry this feature produced (after the
  missing `vacancyDomain` helper and the `avoid` default) and cost a debugging round for
  anything driving both. `mapping` wins because a field called `map` sitting next to Python's
  builtin reads badly. The remaining fields — `mintedStress`, `remintRounds`, `bijective`,
  `imageSize`, `forbidden`, `domain` — already agree and are normative.
- **Statistic field names are camelCase in both stacks**, exactly as §10 spells them, including
  in the Python JSON. This deviates from the rest of the Python API, which is snake_case; the
  vacancy block is a nested object, so it is self-consistent and it lets the golden fixture
  compare the two stacks key-for-key without a translation table.

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

1. `mintedStress[lower(word)]` — the intended pattern of a form we minted ourselves, so
   prosody scoring on a vacated corpus reflects what we built *for the minted forms*. Passed in
   as an argument, never a module global (§5.8).
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
| `mint` | `"nonce"` \| `"swap"` | `"nonce"` | invent the replacement, or draw a real word (§8.3) |

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

**Injectivity is verified, not assumed** — by conditions A and B of §5.2, which are checked at
map-build time and reported as `bijective` and `remintRounds` in the statistics.

Two traps, both found by implementing this document rather than by reading it:

1. Checking `|image| == |types|` **at `p = 1` only** is insufficient. At full vacancy every
   eligible type has moved, so a minted form cannot collide with a surviving English word; the
   collision only exists at intermediate `p`. Condition B is `p`-independent precisely so this
   cannot recur.
2. Checking **bare nonces** rather than assembled surface forms is insufficient, for the same
   reason: the collision arrives through the suffix.

The test asserts injectivity at every `p` in the grid, not just at the endpoints.

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

### 8.2a The word alphabet, and the table both stacks classify from

`WORD_RE` is `[A-Za-z]+(?:['-][A-Za-z]+)*` — ASCII letters joined by the ASCII apostrophe and
hyphen. A passage containing anything else does not contain *those words* as far as the
transform is concerned, so vacating rewrites a fragment and the endpoint returns a plausible
number. `check_word_alphabet` / `checkWordAlphabet` refuse such a passage up front.

A run a reader would call one word is `(L M*)+ ( J+ (L M*)+ )*` — letters with the combining
marks that sit on them, then joiner-separated continuations. The three classes are:

| class | membership |
|-|-|
| letter | Unicode general category `L` |
| mark | `M` — part of the letter it follows, never a joiner |
| joiner | `Pd` ∪ `Cf` ∪ `Pc`, plus named apostrophes and word-internal points (`'`, `’`, `·`, `׳`, `・`, …) that carry no property distinguishing them from quotation marks |

Two corrections, both from measured wrong answers (2026-08-04, round 5):

- **`Pc` (connector punctuation) was missing from both stacks.** `don‿t` (U+203F) scored
  HTTP 200 and swapped to `warm‿t` — character for character the `don’t` → `big’t` defect the
  joiner class had been introduced to close. `_`, U+2040, U+2054, U+FE33–U+FE34,
  U+FE4D–U+FE4F and U+FF3F did the same.
- **The two stacks read different Unicode tables.** Python 3.10 — the version CI pins —
  carries Unicode 13.0; Node 22 carries 16.0. Measured over the whole code space, the two
  disagreed about **9 993 letters and marks and 11 joiners**. U+0890 and U+0891 are genuinely
  `Cf`, the backend's own declared category, and the backend scored them while the browser
  refused them. **The classes are therefore not read from either runtime.** They come from
  `word-classes.json` in this directory — a committed enumeration at one pinned Unicode
  version, copied byte-identically into `llm_geometry/arch/data/` and
  `src/lib/staticClient/`, regenerated by `node scripts/export_word_classes.mjs`, and
  compared to the normative copy by a test in each suite. Moving the pin is a deliberate
  commit that moves both stacks together. Reverting either stack to `\p{…}` or to
  `unicodedata.category` re-opens the divergence and fails those tests.

**A run written entirely in `WORD_RE`'s own alphabet is not refused, even when `WORD_RE`
splits it.** `J+` accepts a run of joiners where `WORD_RE` accepts exactly one, so the
Gutenberg em-dash convention `legs--upon` was refused — while the refusal's own advice is
"use a passage written in the ASCII alphabet, with straight apostrophes and hyphens", which
`legs--upon` already is. There was no way to comply, and this project's corpus contains
`ba--are`, `hea--art`, `Lady--loves`, `legs--upon`. The split is harmless there: each piece is
a whole ASCII word the transform vacates *as* a word, and no character survives inside a
rewritten fragment. That is exactly what fails for `don’t` and `co<SHY>operate`, where a
character `WORD_RE` cannot see is left between two halves of a word it rewrote — so those
still refuse. The escape hatch is ASCII-only: `co<SHY><SHY>operate` is still refused.

`word-alphabet-cases.json` in this directory is the shared case table; both suites run it and
must return the listed runs exactly.

### 8.3 The swap control — what makes the number interpretable

`ΔnllPreserved > 0` on its own is uninterpretable, because at least three things change at once
when content words are vacated:

1. the forms are unknown, so the model has no lexical entry to condition on;
2. nonce forms fragment into many subword tokens, so the context is longer and stranger;
3. the passage says something nonsensical.

Only (1) is "location". A caveat cannot separate them; a control can.

**Swap.** Replace each vacated word with a *real English word* instead of a nonce form — same
eligibility, same `u(stem) < p` decision, and the injectivity §5.2a shows is available to it. The
map is over **whole types, not stems**, and **nothing is re-assembled**: the image *is* a domain
type, hence a real word, ordinarily tokenized.

Construction. Partition the vacatable domain types into **suffix classes** — the suffix §3 splits
off, so the ten classes on the shipped corpus are `'' s ed er ing 's es ly ies est`. Rank each
class by `(corpus count descending, type ascending)`, the tie rule `frequencyBudget` already uses.
Then permute each class onto itself with **no fixed point**, in ASCII-ascending order of the type,
in three stages:

1. **the draw** — attempt `a` reads an offset `δ ∈ [-w, -1] ∪ [1, w]` from the byte stream of §5.3
   under the tag `swap` (never `mint`, so the two streams cannot alias) and proposes
   `pool[(r + δ) mod m]`, where `r` is the type's own rank and `m` the class size; `w` starts at
   32 and doubles every 64 attempts up to `m`. A candidate is accepted iff it is not already used,
   is not the type itself, and — while `a < 1024` and `matchProsody` — carries the type's stress
   pattern;
2. **deterministic completion**, if 4 096 draws all landed on used entries: scan outward from the
   type's own rank (`+1, −1, +2, −2, …`), taking the first free entry. This is reached in ordinary
   runs — the last type of a class has one free image out of `m` — and it is exhaustive, so it
   cannot fail while any image is free;
3. **the endgame exchange**, if the only free image is the type itself (possible only for a
   class's last type): exchange with the ASCII-first assigned type whose image is not this one.
   Both entries stay non-identity and the images stay distinct.

Stages 2 and 3 drop the prosody preference, which is stated rather than hidden: they run only
where the class has nothing left, and a real word of the wrong stress is a far smaller departure
than a form that is not a word. Measured at seed 0 on the shipped corpus: 1 918 of 1 940 types
(98.9 %) get a stress-matched real word.

Because each class is permuted onto itself, every image is a real domain word, and — **except for
the merged singleton classes described below** — it carries the same inflection as the word it
replaces, so the morphology a reader parses (`-ed`, `-ing`, `-'s`) is as intact in the swap arm as
in the nonce arm. Measured over the six shipped Architecture passages at `p = 1, seed = 0`: 767
vacated words, 0 outside the domain, 21 (2.74 %) in a different suffix class, every one of them
from a merged singleton. §3's spelling heuristic becomes harmless
here: `November → Novemb + er` is never re-assembled, it only puts `November` in the `er` class.

**Why not stems.** The first implementation drew a replacement for the STEM and re-attached the
SOURCE word's suffix to it. The pool holds inflected types, so that produced forms which are not
English words at all: `jump` + `ed` drawing `went` gave `wented`; `leap` + `ing` drawing `thy`
gave `thying`; `huff` + `ed` drawing `sacks` gave `sacksed`; `aft` + `er` drawing `kits` gave
`kitser`. Measured over the six shipped Architecture passages at `p = 1, seed = 0`: **195 of 776
vacated words (25.1 %)** had a form absent from `/usr/share/dict/words`, and **165 (21.3 %)** were
not words of their own domain. That falsifies the one property the control exists for, and it
biases the decomposition below — `nll(nonce) − nll(swap)` is *the cost of unknown form*, and the
swap arm was carrying unknown forms of its own. Measured after the fix, same configuration:
**0** of 767 vacated words is outside its domain, and the swap variant's token count falls from
2 858 to 2 766 against english's 2 754 (+3.8 % → +0.4 %), so it is now very nearly
tokenization-neutral as well.

**What "real word" means here, exactly.** The image is a type of the passage's own domain — the
text's own vocabulary, plus the full Dolch list. That is a verifiable property with no external
dictionary, and it is what both stacks' tests assert. It also means the guarantee is only as
English as the source text: *Mother Goose* contains `intery`, `cutery`, `kyloe` and `lauk`, so the
swap arm may too. What it can never contain is a form the source text did not.

**A class of one is merged into the bare class.** A class with a single member cannot be permuted
without a fixed point, and on a PASSAGE-sized domain that is the common case rather than a corner:
all six shipped Architecture passages have such a class (`ing` in three of them). Those
types join the bare class, which the full Dolch list keeps at ≥ 194 members. This is the one place
the inflection match bends, and it bends in the only direction that keeps the property the control
exists for — the replacement is still a real domain word, merely an uninflected one. Refusing
instead would refuse all six shipped passages; assembling a form would put a non-word back
into the arm whose whole claim is that every form is known. If even the bare class cannot be
permuted, the engine **raises**.

Because the replacements are real English words, they are **not** registered in `mintedStress` —
their stress comes from the table or the rule like any other English word, so `stressFromMinted`
is 0 on both sides of a swap and `stressFromTable`/`stressFromRule` say what they always say.

`consistent = false` is **refused** under `swap`: that control needs a fresh type per occurrence,
and the corpus has 1 676 open-class stems against 8 125 vacated tokens, so there is no supply. It
raises rather than quietly reusing words and reporting a rate it is not achieving.

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

`mint: "swap"` is still **nested** in `p` (the `u(stem) < p` decision is untouched, and it is
still taken on the STEM, so swap and nonce vacate exactly the same tokens — the decomposition
depends on that) and still **stable** (the map is built once, in canonical order, independently
of `p`). `VacancyMap.mint` records which strategy built the map, because its KEYS differ: stems
under `nonce`, types under `swap`. `VacancyMap.stems` carries the stem set either way, so §10's
`stems*` counts mean one thing under both.
Injectivity is where it and `nonce` part company, and §5.2a proves why they must: an earlier draft
of this paragraph claimed swap "preserves every property of §7", and that claim is false — a map
whose images are domain types and which does not depend on `p` cannot be injective at intermediate
`p` unless it is the identity. So the invariance theorem of §7.3 holds for `swap` at `p ∈ {0, 1}`,
where it is asserted exactly as for `nonce`, and the mapped vocabulary is refused in between. At
full vacancy the tiny model is exactly as blind to `swap` as to `nonce`, which is the check that
the control is implemented correctly; the pretrained arm measures at full vacancy, so nothing the
control exists for is lost.

### 8.3a What was measured, and what the static build may therefore say

Measured before implementing: 6 × 250-word real-corpus passages × 4 conditions (english /
frequency-matched real-word swap / nonce at two seeds) × {gpt2, SmolLM2-135M}, ~700 preserved
closed-class tokens per condition. ONNX fp32 ≡ torch to 5.3e-4 nats, and the two stacks'
tokenizations were identical (0 id mismatches across 48 texts), so the alignment of §8.2 is
sound and fp32 is the reference.

**The result (fp32).** `nonce − english` ≈ **0.92–1.03 nats**, of which `nonce − swap` is only
**0.06–0.21**.

> **These are the PROTOTYPE's numbers, and the shipped swap is not the one they were measured
> with** (2026-08-04). That study's swap re-attached the source suffix to a stem replacement, so
> a quarter of its swap forms were not English words — the defect §8.3 now describes. Re-measured
> on the shipped configuration (gpt2, float32, the six default passages, `p = 1, seed = 0`,
> 856 paired preserved tokens): `wrong_content = 0.690 ± 0.054`, `unknown_form = 0.287 ± 0.045`,
> `total = 0.978 ± 0.059`. Before the fix, the same run read `0.717 ± 0.054` / `0.273 ± 0.041` /
> `0.989 ± 0.060` over 847 pairs. So the bias was in the direction the argument predicts —
> `unknown_form` understated by **0.015 nats (5.4 % of itself)** and `wrong_content` overstated by
> about the same — and it is a fraction of the sampling standard error rather than a
> conclusion-changing effect. The ratio it feeds is unchanged in substance: ~70 % of the damage is
> wrong content, ~30 % unknown form, on this model and this passage set. The 0.06–0.21 range above
> is left as the historical record of a different measurement, not restated as the shipped one. So roughly 80–90 % of the damage is *wrong content* and only 10–20 % is *unknown
form*. Taken with the tiny arm's exact zero, that is the 2×2:

| | what a word's form is worth |
|-|-|
| tiny, trained from scratch (no locations) | exactly 0 |
| pretrained (has locations) | ~0.1 of ~1.0 nats, i.e. 10–20 % |

Even for a model that *has* locations, losing the location costs far less than losing the
content — the doc's T4 prediction that field ≫ location, on a model it did not consider.

**The quantization verdict.** The app ships quantized ONNX, and the effect above is small:

- **Absolute NLL is unusable.** q8 shifts `nllPreserved` by −0.19 nats (gpt2) and **+0.40**
  (SmolLM2) — the sign is not even stable across models.
- **Pooled differences do cancel**: `|Δ_q8 − Δ_fp32| ≤ 0.054` nats on every contrast, against a
  sampling standard error of 0.12–0.22.

  > **The q8 arm has not been re-measured since the swap rewrite** (2026-08-04). Every
  > q8-vs-fp32 gap on record — the 0.054 above, and this build's own 0.073 (`swap − english`)
  > and 0.110 (`nonce − english`) — was taken on the OLD variant texts. The fp32 side has been
  > re-measured on the shipped transform (0.6904 and 0.9776; pinned by
  > `test_the_fp32_arm_quoted_in_the_static_client`), the q8 side cannot be, because it needs a
  > real browser. So `VACANCY_Q8_UNCERTAINTY_NATS = 0.2` is **not currently a like-for-like
  > measurement of the shipped configuration**; it is retained only because it exceeds every gap
  > ever observed here, and lowering a bound without a measurement is worse. Restoring the
  > derivation needs a browser q8 run of the static scorer on the six default passages, against
  > those fp32 numbers. Until then, do not quote it as "measured on the shipped swap".
  >
  > **The rendered PROSE now complies too** (2026-08-04, round 5). Round 3 fixed the label and
  > left the sentences. `VACANCY_UNKNOWN_FORM_REFUSAL` still read *"Measured on this very
  > configuration: float32 says 0.273 for gpt2 and q8 says 0.235"* — both values pre-rewrite,
  > both presented as current — and `VACANCY_PER_PASSAGE_REFUSAL` still said *"Only the pooled
  > figure has a measured bound"*, of a bound that is retained. The figures now live in two
  > named constants, `VACANCY_FP32_REFERENCE` (0.6904 / **0.2872** / 0.9776 over 856 paired
  > tokens) and `VACANCY_PRE_REWRITE_Q8` (the 0.2726 / 0.235 pair, labelled as history), and
  > the sentences are interpolated from them. `unknown_form` is the number the `nonce − swap`
  > refusal quotes and was the one figure of the three that `test_the_fp32_arm_quoted_in_the_static_client`
  > did not pin; it is pinned now. `archVacancy.test.ts` fails if a refusal claims a current
  > q8 measurement or drops the constants.
  >
  > **The rendered label now complies** (2026-08-04, round 3). Both error-bar renderers printed
  > `± 0.2 (quantization, measured)` on the two numbers the static panel reports — the one
  > surface a reader actually sees, and the exact claim this paragraph forbids. They print
  > `QUANTIZATION_TERM` (`vacancyVerdict.ts`) instead: *"quantization, retained bound — not
  > re-measured since the swap rewrite"*, with the full standing of the bound beside the
  > numbers. `archVacancyPanel.test.ts` fails if any renderer calls it measured again.
- **Per-passage differences do not**: worst case 0.65 nats, **115 %** of that passage's fp32
  delta.
- **`nonce − swap` is destroyed.** Its true value is 0.06–0.21; q8's error on it is 14–23 %
  pooled, up to 0.28 per passage, with one sign flip in six passages per model.
- The error is **not** a constant offset that a baseline could remove: q8 compresses extreme
  surprisal, and 2.7 % of gpt2's preserved tokens carry `|e| > 5` nats, all on 15–20-nat
  line-initial function words — precisely the tokens this measurement is about. Median and
  trimmed means do not rescue it.
- **q4f16 — the app's first-choice dtype — could not be measured outside a browser** (session
  init fails on the onnxruntime-node CPU EP for both models). Until it is measured in a real
  browser, the deployed default path has **no error bar at all**.
  **RESOLVED (2026-08-04):** measured in a real browser on a real Apple Metal-3 adapter — on
  gpt2 and both SmolLM2 exports q4f16 returns logits identical at every position (SmolLM2:
  exactly 0, every NLL = ln V), so it has no error bar because it measures nothing. It is no
  longer requested: the app's ladder is now `webgpu/q8 → wasm/q8`, both rungs gated by a
  load-time non-degeneracy check, so **q8 is the dtype whose measured bounds apply here**.

**Policy.** The full stack (torch/fp32) reports everything. The static build may report a
number only where there is a measured bound for the dtype it actually ran:

1. pooled `nonce − english` and `swap − english`, with the quantization uncertainty stated;
2. **never** `nonce − swap`, and **never** a per-passage delta — these are refused with a typed
   error naming the full stack, exactly as the static build already refuses elsewhere;
3. if the dtype in use has no measured bound, refuse rather than invent one. A stated ±
   that was never measured is a fabricated error bar, which is worse than no number.

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
| 12 | eligibility tested on the stem alone | tested on the WHOLE word first, then the stem (§2.2) | the suffix splitter breaks seven function words open (`after → aft + er`), and the stem test alone vacated all seven |
| 13 | seed unbounded | `\|seed\| ≤ 2⁵³ − 1`, enforced in both stacks (§4) | beyond it Python hashes the exact integer and JavaScript the rounded double, so the two build different maps and neither raises |

Departures 4, 6, 7 and 8 are corrections to bugs that break properties the source *claims*.
Departure 2 is **not** a bug fix — the source's expression was tested and is fine; the change
buys structural rather than argued cross-language equality.

---

## 10. Statistics contract

`vacancyStats(originalText, vacatedText, map, p, seed, …)` returns, with these exact names:

```
domainTypesTotal, domainTypesEligible, domainTypesVacated,
corpusTypesTotal, corpusTypesEligible, corpusTypesVacated,
stemsTotal, stemsVacated,
tokensTotal, tokensVacated,
meanSyllablesBefore, meanSyllablesAfter,
meanAnapestBefore,  meanAnapestAfter,
stressFromTableBefore,  stressFromTableAfter,
stressFromMintedBefore, stressFromMintedAfter,
stressFromRuleBefore,   stressFromRuleAfter,
bijective, imageSize, remintRounds
```

Both stacks compute these from the same definitions; the golden fixture (§11) pins them.

**Counting: the scope is in the name.** This section cost two round trips between the stacks,
both times because "types" is ambiguous between the **corpus** (2 211 types of *Mother Goose*)
and the **domain** (2 233 = corpus ∪ the full Dolch list). The two stacks agreed on
`tokensVacated` to the token (8 125 at `p = 1`) and disagreed only on the type counts — a
reporting gap, never a disagreement about the transform.

An unprefixed `types*` is therefore **forbidden**. Every count names its scope:

- `domainTypes{Total,Eligible,Vacated}` — over the domain of §5.2. This is what governs the map
  and the vocabulary, so it is the diagnostic number.
- `corpusTypes{Total,Eligible,Vacated}` — over the corpus's own type set. This is what the panel
  shows a reader, because the 22 domain-only words (`funny`, `squirrel`, `today`, …) are in the
  budget but never appear in the text, and counting words the reader cannot see inflates the
  vacancy rate they are being shown.

  **`corpusTypesVacated` is measured from the two texts, not from map membership.** A type
  counts as vacated iff at least one of its occurrences actually changed. Under
  `revealAfter > 0` the two are not the same number and the stacks split on it — one measured
  the texts (663) and one asked whether the stem was in the vacated set (1334), over-reporting
  by 2×, because a type whose every occurrence falls inside the reveal window is still listed
  in the map. Under `revealAfter = 0` the readings coincide, which is why it took a control
  condition to expose. Measuring the texts is the definition that matches what this number
  claims to the reader.
- `stemsTotal` — distinct vacatable stems, i.e. `|VacancyMap.stems|`; `stemsVacated` — stems with
  `u(stem) < p`. **Taken from `stems`, never from `|mapping|`**, which counts stems under `nonce`
  and types under `swap` (§8.3) and would silently change meaning with the strategy.

  **`domainTypesVacated` is MAP MEMBERSHIP, by one rule in both stacks**: the type is vacatable
  (§2.2) and `u(stem) < p`. TypeScript additionally required the image to *differ* from the type;
  the two readings coincide, since conditions B and B₁ both forbid an image equal to its own
  source, so no case ever disagreed — which is exactly why it had to be pinned rather than left as
  two definitions of one statistic. Both stacks now assert the coincidence directly.
- `tokensTotal` / `tokensVacated` — over the **corpus** token stream

`domainTypesEligible ≥ stemsTotal` always, since inflected forms share a stem, and
`domainTypesEligible = corpusTypesEligible + 22` on the shipped corpus. At `p = 1` every
eligible stem vacates, because `u ∈ [0, 1)` by construction, so `stemsVacated == stemsTotal` and
`{domain,corpus}TypesVacated == {domain,corpus}TypesEligible` — identities worth asserting,
since the first of them is what exposed all of this.

Measured on the shipped corpus (seed 0 / seed 7, `p = 0/.25/.5/.75/1`):
`corpusTypesVacated` 0/461/951/1427/1918 and 0/433/972/1436/1918;
`domainTypesVacated` 0/469/963/1445/1940 and 0/439/982/1451/1940.
Totals at `p = 1`: `domainTypesTotal` 2 233, `domainTypesEligible` 1 940, `corpusTypesTotal`
2 211, `corpusTypesEligible` 1 918, `stemsTotal` 1 676, `tokensTotal` 16 000, `tokensVacated`
8 125. (Before §2.2's whole-word test these read 1 944 / 1 922 / 1 680 / 8 202.)

**Where each token's stress came from.** The first draft asked for a single
`stressTableCoverage`, which is ambiguous the moment minted forms exist: read literally it counts
only the hand table (measured 1.2 % after full vacancy), and read as "stress we actually know" it
also counts minted forms, whose pattern we chose ourselves (41.6 %). Rather than pick, report the
**three-way split**, which is unambiguous and strictly more informative:

- `stressFromTable` — the hand table of §6.1. This is the honesty number for English words, and
  it is the one FR-712/SC-708 require beside every prosody statistic.
- `stressFromMinted` — forms we minted, whose intended pattern we registered. Known by
  construction, but *asserted* rather than verified: §5.5 accepts a candidate on syllable
  **count**, so the count is checked and the pattern is not.
- `stressFromRule` — the spelling heuristic of §6.2, i.e. a guess.

The three are token-weighted fractions and sum to 1 on each side.

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

- `u(stem)` for a fixed list of 28 stems spanning eligible/ineligible and both budgets — the
  exact float64, which is the whole point of departure 2
- the full `map` at `seed ∈ {0, 7}` — every stem→nonce pair
- the first 400 characters of the vacated corpus at `p ∈ {0, 0.35, 0.7, 1}`, seed 0 — this is
  the source's own figure, reproduced on our corpus
- `vacancyStats` for each of those, all fields
- the nesting assertion: `vacated(0.35) ⊆ vacated(0.7) ⊆ vacated(1.0)` as explicit id sets
- the stability assertion: the nonce for each of 28 stems is identical at every `p` where it
  is vacated
- the token id stream digest under the mapped vocabulary at each `p` — all equal, which is
  §7.3 pinned as data rather than as an assertion in one language

Strings are compared exactly. Only the prosody means use `tolerance`.
