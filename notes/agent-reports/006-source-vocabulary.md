# Vocabulary machinery in `tiny-seuss` — source analysis

Target: `/Users/jmanning/Desktop/tiny-models/tiny-seuss/`
Files analyzed: `wordlists/dolch.py`, `synth/lexicon.py`, `synth/jabberwockify.py`, `synth/generate.py`
Date: 2026-08-03. Every claim below is grounded in a verbatim quote from the source, or in output I produced by executing the code.

**Repo status:** `tiny-seuss/` is **not a git repository** (`fatal: not a git repository`). It sits inside a parent directory `/Users/jmanning/Desktop/tiny-models/` which contains an independent audit (`AUDIT.md`), a LaTeX manuscript, a test suite, and an integration plan referencing `llm-geometry`.

---

## 1. The Dolch word lists

### Are they present as real data?

**Yes.** They are hard-coded string literals in `wordlists/dolch.py`, not fetched or generated. Example (lines 15–17):

```
DOLCH_PRE_PRIMER = """a and away big blue can come down find for funny go help
here I in is it jump little look make me my not one play red run said see the
three to two up we where yellow you""".split()
```

### Exact cardinalities (measured by executing the module)

| List | Constant | Count |
|-|-|-|
| Pre-primer | `DOLCH_PRE_PRIMER` | 40 |
| Primer | `DOLCH_PRIMER` | 52 |
| First grade | `DOLCH_FIRST` | 41 |
| Second grade | `DOLCH_SECOND` | 46 |
| Third grade | `DOLCH_THIRD` | 41 |
| **Service total** | `DOLCH_SERVICE` | **220** |
| Nouns | `DOLCH_NOUNS` | 95 |
| **All** | `DOLCH_ALL` | **315** |
| Decode mask | `MASK_50` | 50 |

### The budgets

Defined at lines 59–66 under the comment `# Graded budgets, so you can sweep |V| as an independent variable.`

| Key | Contents | Size (measured) |
|-|-|-|
| `mask50` | `MASK_50` | 50 |
| `preprimer` | pre-primer | 40 |
| `primer` | pre-primer + primer | 92 |
| `first` | pre-primer + primer + first | 133 |
| `service220` | all five service levels | 220 |
| `dolch315` | service + nouns | 315 |

The README's "40 / 92 / 133 / 220 / 315" (line 59: `settings**: 40 / 92 / 133 / 220 / 315.`) matches exactly. All six budgets are duplicate-free (verified: `len(v) == len(set(w.lower() for w in v))` holds for every budget; the module's own `__main__` block asserts this).

**Note:** the README also mentions numbers that do **not** appear in the code — `"the 50-word bet, the 236-word school list"` (line 12). There is no 236-word list anywhere in `wordlists/dolch.py`.

### Are these the REAL published Dolch lists?

**Almost.** I diffed the code word-for-word against the Wikipedia `Dolch word list` article (fetched via Special:Export). Four of five service levels are **byte-identical**:

| Level | Diff vs. published list |
|-|-|
| Pre-primer (40) | identical |
| Primer (52) | identical |
| **First grade (41)** | **code has `giving`; published list has `going`** |
| Second grade (46) | code omits `left`, which Wikipedia includes |
| Third grade (41) | identical |
| Nouns (95) | matches, incl. `good-bye` and `Santa Claus` |

Two findings, with different verdicts:

1. **`giving` is a transcription error for `going`.** Confirmed against a second independent source (mrsperkins.com), which answered directly: *"Does the First Grade list contain 'going' or 'giving'? Yes, it contains 'going.'"* This is a **real defect to fix on reimplementation.** It also propagates: `giving` appears in `lexicon.py`'s `STRESS_TABLE` (`"giving": "10"`) and in `jabberwockify.py`'s `split_suffix` exceptions set.

2. **The `left` omission is correct; Wikipedia is wrong.** Wikipedia lists 2nd grade as 47 words, making its non-noun total 221 — one over the canonical 220 that Wikipedia itself states. The code's 46-word 2nd grade yields exactly 40+52+41+46+41 = 220. The second source also confirmed `left` is absent. **Do not "fix" this.**

### Licensing / provenance as stated in the code

The module docstring (lines 1–13) states:

> `DOLCH_SERVICE (220) + DOLCH_NOUNS (95) = the 1936 Dolch sight-word lists, published in E.W. Dolch, "A Basic Sight Vocabulary," Elementary School Journal 36 (1936). Public domain. This is the *kind* of list that publishers handed to trade authors in the 1950s as a reading-level budget.`

and closes:

> `Nothing in this file is an expressive work; these are word budgets.`

The README repeats it (line 53): `Synthesize metered rhymed verse under the Dolch lists (1936; 220 service words + 95 nouns; public domain; essentially the kind of list Seuss worked from)`.

The 1936 citation is accurate as to author, title, journal, and year. The "public domain" assertion is asserted, not sourced.

### ⚠️ `MASK_50` is misdescribed in the docstring

The docstring says (lines 9–11):

> `MASK_50 is the decode-time mask: fifty of the commonest words in English. It is used here as a constraint on generation, never as content.`

and the inline comment (line 52) says `# Fifty common English words. Constraint only.`

**This is false.** `MASK_50` is the verbatim vocabulary of *Green Eggs and Ham* (1960). I verified the full 50-word list against published sources; it matches item-for-item, including `anywhere`, `boat`, `box`, `eggs`, `fox`, `goat`, `ham`, `mouse`, `Sam`, and `train` — none of which are among "the commonest words in English." The README's `"the 50-word bet"` (line 12) is the tell: Dr. Seuss wrote the book to win a $50 bet with Bennett Cerf under a 50-word constraint.

**Implication for reimplementation:** the licensing framing that covers the 1936 Dolch lists does *not* transparently cover this list, and the docstring's provenance claim is inaccurate. A bare word list is likely uncopyrightable, but the file should say what the list actually is. This deserves an explicit decision, not a silent copy.

---

## 2. `synth/lexicon.py` — budget enforcement + prosody

Docstring (lines 1–13) states the design claim:

> `The point of a closed lexicon that nobody usually notices: you can carry an EXACT pronunciation dictionary. 315 words is small enough to hand-verify in an afternoon, so meter is not estimated, it is known.`

**That claim is not met by the code.** `STRESS_TABLE` has only **61 entries** for a 315-word budget; everything else falls through to a spelling heuristic. The docstring concedes the mechanism (`STRESS below is seeded by rule and then overridden by a hand table`), and the README is more honest (lines 186–188): `The current table is incomplete and partly heuristic, so meter is presently estimated, not known. A reviewed pronunciation table and tests are required before making an exactness claim.` The parent `AUDIT.md` finding #4 says the same: `Prosody is not exact. Most of the 315 words use a spelling heuristic, some table entries are incorrect`.

### What it provides

**Tokenization.** One regex, `WORD_RE = re.compile(r"[A-Za-z']+(?:-[A-Za-z']+)*")` — handles hyphenated words (`good-bye` tokenizes as one token) and apostrophes (`don't`).

**Stress / syllables.** `stress(word)` resolves in priority order: `MINTED_STRESS` → `STRESS_TABLE` (case-sensitive, then lowercased) → `_rule_syllables()` heuristic returning `"1" + "0"*(n-1)`. `syllables(word)` is `len(stress(word))`. `register_minted(d)` populates `MINTED_STRESS` so meter scoring on a jabberwockified corpus uses intended patterns — it is called from exactly one place, `eval/fingerprint.py:157`, gated behind a `--stress-map` argument.

**`Budget` class.** Constructor `Budget(name="dolch315", extra=())`; `self.words` is a **lowercased set**. Key methods:
- `tokens(text)` → `WORD_RE.findall(text)`
- `violations(text)` → list of tokens not in the budget
- `ok(text)` → `not self.violations(text)`
- `scan(line)` → concatenated stress string
- `meter_score(line, foot="anapest")` → fraction of syllable positions matching a repeating pattern (`anapest`=`001`, `iamb`=`01`, `trochee`=`10`, `dactyl`=`100`)
- `rhyme_key(word)` / `couplet_rhyme(l1, l2)` → orthographic rime normalization (`_RIME_NORM`, 21 rules) with a 68-entry hand table `_RHYME_HAND` for open syllables. `couplet_rhyme` requires `a != b` (identical end-words don't count as rhyme).

**Inflection is off by default.** `self.allow_inflection = False`, with the comment: `Seuss's publisher list allowed the bare forms; we do the same and make the choice visible rather than silently permitting -s/-ed.` When enabled, `_stem_ok` strips `("s","es","ed","ing","er","est","ly","'s","n't")`. Verified: `Budget('dolch315').violations('the cats ran')` → `['cats']`.

**🐛 Defect: `"Santa Claus"` can never validate.** `DOLCH_NOUNS += ["Santa Claus"]` puts a space-containing entry into `self.words` (as `"santa claus"`), but `WORD_RE` cannot produce a token containing a space. Verified:

```
>>> b.violations('Santa Claus came')
['Santa', 'Claus']
```

So one of the 315 budget words is unreachable — the effective usable vocabulary is 314, and any generated stanza mentioning Santa Claus is rejected. `STRESS_TABLE` also carries a `"Santa Claus": "101"` entry that `stress()` can never reach through normal tokenization.

**Minor:** `STRESS_TABLE` contains one key not in `DOLCH_ALL` — `summer` (harmless, unreachable).

`report(budget, text, foot)` returns `n_lines, n_tokens, ttr, coverage, oov, meter_mean, rhyme_rate`.

---

## 3. `synth/jabberwockify.py` — the nonce-word / vacancy system

### What "vacancy" means

Docstring (lines 3–9):

> `Carroll's trick, stated operationally: closed-class scaffolding is fully intact -- function words, inflectional morphology, syntax, meter -- while open-class slots carry nonce forms. A reader parses "the slithy toves did gyre and gimble" because every grammatical signal survives and only lexical content is vacant. Such a token has a FIELD (its syntactic/semantic neighborhood is fully specified by context) and no LOCATION (no prior embedding).`

### The `p` parameter

> `p            fraction of open-class TYPES vacated (0 = English, 1 = full Jabberwocky). Sweep it; the comprehension/loss curve against p is the actual measurement.`

`p` operates on **types, not tokens**. The decision is a deterministic hash threshold, not a draw:

```python
def _u(self, key):
    """Deterministic uniform in [0,1) per (word, seed). Independent of p
    and of traversal order."""
    h = hashlib.sha256(f"{self.seed}:{key}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2 ** 64

def _decide(self, stem):
    """Vacate iff u(word) < p. This makes the vacated sets NESTED as p
    grows -- everything vacated at p is still vacated at p' > p -- so a
    p-sweep varies only the vacancy set and never the assignment."""
    return self._u(stem.lower()) < self.p
```

**Verified nesting empirically.** On a fixed sentence at seed 0:

| p | vacated types |
|-|-|
| 0.0 | (none) |
| 0.25 | brother, ran, squirrel, want |
| 0.50 | + little |
| 0.75 | + away |
| 1.00 | + sleep, tonight |

Each set is a strict superset of the last. **Confirmed `nested: True`.**

### How nonce words are generated

`class Minter` — `"""Phonotactically legal English nonwords, prosody-matched."""` Syllables are assembled from four hand-written inventories: `ONSETS` (47 entries), `NUCLEI` (19), `CODAS` (46, includes `""`), `UNSTRESSED_TAILS` (13).

`Minter.mint(n_syl, pattern)` loops up to 400 times:
- For a stressed position: `onset + nucleus + coda`.
- For an unstressed position at index 0: a prefix from `["a","be","re","de","un","en"]`.
- For an unstressed position elsewhere: an entry from `UNSTRESSED_TAILS`.
- Collapses 3+ repeated consonants: `re.sub(r"([bcdfghjklmnpqrstvwxz])\1{2,}", r"\1\1", w)`.
- Rejects if `len(w) < 3`, already used, in `avoid`, or if `syllables(w) != n_syl`.
- Fallback after 400 tries: `self._syl(True) + str(len(self.used))` — with the comment `# give up gracefully rather than loop forever`.

### Determinism / seeding

**Fully deterministic and seeded.** Two independent mechanisms:
1. Vacancy selection: SHA-256 over `f"{seed}:{word}"` (above).
2. Nonce assignment: `_nonce_for` **re-seeds the minter per word** before minting:
   ```python
   # per-word minter so the nonce for a word is fixed by (word, seed)
   # alone -- stable across p and across corpora.
   self.minter.rng = random.Random(f"{self.seed}:{k}")
   ```

Verified: identical output for two `Jabberwockifier(p=1.0, seed=0)` instances; different output at `seed=1`. CLI exposes `--seed` (default 0).

### `consistent` vs `inconsistent`

> `consistent   one nonce form per source type, corpus-wide, so the minted token has a stable distributional signature to grow a field from. Setting this False gives the control condition: same vacancy rate, no learnable identity.`

Implemented by whether the `stem → nonce` map is consulted and written (`if self.consistent and k in self.map: return self.map[k]`). Verified on `"the squirrel and the squirrel and the squirrel"`:

- consistent: `the byrk and the byrk and the byrk`
- inconsistent: `the byrk and the hos and the swort`

**Caveat from the parent `AUDIT.md` (finding #6):** `The inconsistent vacancy control changes the number of nonce types and thus vocabulary/model size; it does not isolate identity by itself.`

### `reveal_after`

> `reveal_after N occurrences are left un-vacated, seeding a partial location. reveal_after=0 is the pure case.`

Verified — `reveal_after=2` on `"squirrel squirrel squirrel squirrel"` → `squirrel squirrel byrk byrk`.

### `match_prosody`

> `match_prosody  minted form carries the syllable count and stress of the word it replaces, so meter is untouched and p is the only variable.`

The intended pattern is recorded in `self.stress_map[nonce] = pat` and dumped to the `--map` JSON as `{"map": ..., "stress": ...}`, for `register_minted()` to reload. Verified: `away` (stress `01`) → `rejam`, with `stress_map['rejam'] == '01'`. Note `stress('rejam')` returns `"10"` from the fallback heuristic — the intended pattern survives **only** via the map file, so a downstream consumer that skips `--stress-map` silently gets wrong meter.

### Closed class

`FUNCTION_WORDS` is a **137-word curated list**, with an explicit warning:

> `# NOTE: closed class is the curated list ONLY. An earlier version added short Dolch service words, which silently protected content verbs (run, eat, see, get, let, put) and understated the vacancy rate. Keep this explicit.`

`OPEN_CLASS_ALWAYS = {w.lower() for w in DOLCH_SERVICE} - FUNCTION_WORDS` (113 words) is computed at line 56 and **never used anywhere in the repo** — dead code.

Open-class test: `_is_open_class(stem)` requires `stem.lower() not in self.keep and stem.isalpha() and len(stem) > 2`.

### Known artifact (documented in-source)

```
# Known artifact: suffix stripping is orthographic, so 'brother' -> 'broth'+'er'
# and 'never' -> 'nev'+'er'. This does not hurt the experiment ... but
# it does mean the -er in the output is sometimes not a real morpheme.
```

`split_suffix` guards this with a 9-word exceptions set: `{"brother","father","mother","sister","never","over","under","morning","giving","thing"}`. It is **incomplete** — `together` is not in it, so it splits as `togeth` + `er`, and prosody is matched to the *stem*, not the whole word. `SUFFIXES` is order-sensitive: `["ing","edly","est","ies","'s","n't","ed","es","er","ly","s"]`, and requires `len(lw) - len(s) >= 3`. `match_case` preserves ALL-CAPS and Title-case. A seam guard prevents `wee`+`er` → `weeer`.

**CLI:** `--in --out --map --p --seed --reveal-after --inconsistent --no-prosody`. Note the flags are negations of the constructor defaults (`consistent=not args.inconsistent`, `match_prosody=not args.no_prosody`).

---

## 4. `synth/generate.py` — paid-API corpus synthesis

### Provider and model

**Anthropic**, via the official SDK. `from anthropic import Anthropic; client = Anthropic()` (line 99–100), authenticated by `ANTHROPIC_API_KEY` (docstring: `export ANTHROPIC_API_KEY=...`). `requirements.txt` pins `anthropic>=0.40`.

```python
DEFAULT_MODEL = "claude-sonnet-4-6"
```

I verified this against the model catalog: **`claude-sonnet-4-6` is a real, currently-active model ID** (Claude Sonnet 4.6). Overridable with `--model`.

Call shape (lines 73–80): `client.messages.create(model=..., max_tokens=4000, temperature=temperature, system=SYSTEM, messages=[{"role":"user","content":msg}])`, then `"".join(b.text for b in r.content if b.type == "text")`.

> ⚠️ **Reimplementation note:** `temperature` is passed on every call (default `1.0`). On current-generation models — including Claude Opus 5, Opus 4.8/4.7, and Claude Sonnet 5 — sampling parameters are **rejected with a 400**. The code works as written only because it targets Sonnet 4.6. If you upgrade the model, `temperature` must be removed and variance steered by prompt instead.

### The prompt

`SYSTEM` (lines 30–42), verbatim:

```
You write children's verse under an absolute word budget.

RULES, in priority order:
1. Every word you use must appear in the ALLOWED list, exactly as spelled.
   No plurals, no tenses, no compounds unless they are in the list.
2. Anapestic tetrameter: da-da-DUM da-da-DUM da-da-DUM da-da-DUM.
   Substitution at the line head is fine.
3. Rhymed couplets, AABB.
4. Each stanza must be about something -- a small event, a refusal, a
   bargain, a chase, a boast, a sulk. Not decorative word salad.

Output ONLY stanzas separated by a blank line. No titles, no commentary,
no numbering.
```

The user message inlines the full sorted budget: `f"ALLOWED ({len(budget_words)} words):\n" + " ".join(sorted(budget_words))`, plus `f"\n\nWrite {n_stanzas} stanzas of 8 lines each.\nTheme for this batch: {seed}\n"` and `"Vary the stanzas. Use rare words from the list, not only the commonest ones."`

`SEEDS` is a list of **16 themes**, one chosen per call via `rng.choice(SEEDS)` — e.g. `"someone refuses a thing and is asked again"`, `"a rule is announced and immediately broken"`, `"something small that will not stay put"`.

### The gates (stanza-level rejection sampling)

The design is stated up front:

> `we do NOT trust the large model to obey the budget. It won't, reliably, at 315 words. Instead we generate over-length, then filter at the *stanza* level against the exact validator, and keep only clean stanzas. Yield is the thing to watch — it is itself a measurement (how hard is it to say anything under |V|=k?), so we log it rather than tuning it away.`

Per stanza (split on `\n\n`), in order — each is a `continue`:

| # | Gate | Condition |
|-|-|-|
| 1 | **Vocabulary** | `if not b.ok(stanza)` — zero OOV tokens, exact validator |
| 2 | **Length** | `if len(lines) < 4` |
| 3 | **Meter** | mean `b.meter_score(l)` over lines `< args.meter_min` (default **0.60**, anapest) |
| 4 | **Rhyme** | `--require-rhyme` (default **True**): every couplet must satisfy `b.couplet_rhyme` |
| 5 | **Dedup** | SHA-256 of the stanza already in `seen` |

Kept stanzas are streamed to disk (`fh.write(stanza + "\n\n")`, `fh.flush()`), and yield is logged per call: `kept {n}/{N}  yield={pct}  {elapsed}s`.

Failure handling is fail-fast, no fallbacks: `raise RuntimeError(f"API call {calls} failed: {e}") from e`, plus a `--max-calls` hard stop (default 10000, `help="hard stop; prevents invalid credentials or zero yield hanging forever"`) and a terminal `raise RuntimeError(f"kept {len(kept)}/{args.stanzas} after {calls} calls")`.

On success it writes `<out>.stats.json` with the `report()` fields plus `yield`, `budget`, `budget_size`, `seed`, `model`, `calls`, `require_rhyme`.

Defaults: `--budget dolch315 --stanzas 4000 --per-call 10 --meter-min 0.60 --temperature 1.0 --seed 0 --out data/corpus.txt`. Cost note in the docstring: `~4k stanzas of 8 lines is roughly 400 calls at 10 stanzas/call.`

### Does any generated corpus exist?

**No.** There is no `data/dolch315.txt`, no `*.stats.json`, and `runs/` is empty. Confirmed by the parent `AUDIT.md` finding #1:

> `No generated base corpus, checkpoint, training log, source-to-nonce map, run manifest, or test suite was supplied. Paper curves are manually entered schematics/predictions.`

and finding #3:

> `The original generator enforced vocabulary and a heuristic meter threshold, but not rhyme. It was unseeded, used process-randomized hash, and could retry API failures forever. These mechanics are now corrected, but no corpus has yet been regenerated with the corrected pipeline.`

So the gates and seeding you see in `generate.py` are **post-audit repairs that have never been run to completion.** The yield number — which the docstring calls a measurement — is unmeasured.

---

## 5. Corpus data actually on disk

### Inside `tiny-seuss/`

| Path | Size | Lines | What it is |
|-|-|-|-|
| `data/demo_jabber.txt` | 18,760 B | 400 | The only corpus file. **Synthetic fixture, not real verse.** |
| `runs/` | — | — | **Empty** |
| `{wordlists,synth,train,eval}/` | — | — | **Empty dir** — an unexpanded brace literal from a `mkdir` |

**`data/demo_jabber.txt` is word salad, not generated verse.** Measured with the project's own `report()`:

```
n_lines: 400,  n_tokens: 3214,  ttr: 0.12134411947728686,
coverage: 0.568,  meter_mean: 0.346,  rhyme_rate: 0.01
```

Those numbers identify it precisely. README lines 143–147:

> `**Unverified control value from the original proposal.** The supplied bundle did not include the generator, seed, or source artifact for the reported uniform-sampling baseline. The tracked demo independently reproduces a Zipf slope near -0.16 and rhyme rate 0.01, but its TTR is 0.121, not 0.098. Regenerate and provenance-track all baselines before citing them.`

My measured TTR is 0.1213 and rhyme rate is 0.0100 — an exact match. `AUDIT.md` corroborates: `The shipped demo fingerprint is reproducible: 400 lines, 3214 tokens, TTR 0.12134, Zipf slope -0.16719, anapest score 0.34639, rhyme rate 0.01.`

So it is a **jabberwockified uniform-random-sample baseline** — the control condition, not the treatment. Its first line reads `gerg wind up did once would soon where`. Of 390 token types, 179 are in-budget and 211 are nonce. Meter 0.35 and rhyme 0.01 are at chance. **No `--map` JSON accompanies it**, so the nonce→source mapping and intended stress patterns for this file are unrecoverable.

### Parent repo `/Users/jmanning/Desktop/tiny-models/`

| Path | Size | What it is |
|-|-|-|
| `artifacts/fixture/corpus.txt` | 7,104 B | 8 hand-written lines × 32 repetitions |
| `artifacts/fixture/manifest.json` | 228 B | Self-labeled `"scientific_result": false` |
| `scripts/offline_fixture.py` | — | Generator for the above |
| `TonyModelsTools/tiny-seuss.zip` | 37,660 B | Snapshot of an earlier bundle |

`scripts/offline_fixture.py` hard-codes `LINES` (`"I see a cat on the tree"`, `"the cat can see me by the tree"`, …) and writes `"\n".join(LINES * 32)`. Its manifest declares itself honestly:

```json
{ "kind": "deterministic offline smoke fixture",
  "generator": "scripts/offline_fixture.py",
  "repetitions": 32,
  "sha256": "9db871f2f8af42f62de926992fd7c446e4cee0a167940bd68d245a375ca4754d",
  "scientific_result": false }
```

`.gitignore` excludes `data/generated/` and `artifacts/*` — so generated corpora were never intended to be committed.

**Bottom line: there is no real Dr.-Seuss-like corpus anywhere on disk.** Everything present is either a word-salad control or a hand-written smoke fixture. Both are correctly labeled as such by the project's own documentation.

---

## Reimplementation checklist

**Fix:**
1. `DOLCH_FIRST`: `giving` → `going` (verified transcription error; also update `STRESS_TABLE` and `split_suffix` exceptions).
2. `"Santa Claus"` is unreachable through `WORD_RE`. Either drop it, hyphenate it, or special-case multiword entries in `Budget.tokens`.
3. Correct the `MASK_50` docstring — it is the *Green Eggs and Ham* vocabulary, not "fifty of the commonest words in English." Make the provenance decision explicit.
4. Remove or use `OPEN_CLASS_ALWAYS` (dead code).
5. Drop `temperature` if targeting any model newer than Sonnet 4.6 (400 error).

**Do not "fix":**
- The absence of `left` from 2nd grade. The code is right; Wikipedia's 47-word 2nd grade contradicts its own stated 220 total.

**Preserve (these are the load-bearing design decisions):**
- Hash-based `_decide` — it is what makes p-sweeps nested and reproducible.
- Per-word minter re-seeding — it is what makes nonce identity stable across p and across corpora.
- Stanza-level rejection sampling with yield logging, and fail-fast on API errors.
- `allow_inflection = False` as a visible choice.

**Carry forward as known-unresolved:**
- Prosody is estimated, not exact (61/315 hand-set). Don't claim otherwise.
- Suffix stripping is orthographic; the exceptions set is incomplete (`together` → `togeth`+`er`).
- Intended nonce stress survives only via the `--map` file; consumers that skip `--stress-map` silently score wrong meter.
- Yield has never been measured — the pipeline has not been run to completion.

---

## Sources

- [Dolch word list — Wikipedia](https://en.wikipedia.org/wiki/Dolch_word_list)
- [220 Dolch Words and 95 Nouns — Mrs. Perkins](https://mrsperkins.com/dolch-words-all.html)
- [Full Dolch Word List — Florida DOE](https://www.fldoe.org/core/fileparse.php/16294/urlt/SightWord.pdf)
- [Green Eggs and Ham — Wikipedia](https://en.wikipedia.org/wiki/Green_Eggs_and_Ham)
- [Green Eggs and Ham uses 50 words — planspace.org](https://planspace.org/20200815-green_eggs_and_ham_uses_50_words/)
- [Did Dr. Seuss Write 'Green Eggs and Ham' on a Bet? — Snopes](https://www.snopes.com/fact-check/green-eggs-and-ham/)
