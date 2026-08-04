"""The vacancy transform — field without location, at a controlled rate `p`.

Carroll's trick, stated operationally: the closed-class scaffolding is left character-for-
character intact — function words, inflectional morphology, punctuation, syntax, line
structure — while open-class stems are replaced by phonotactically legal nonce forms that
carry the same syllable count and stress. A reader parses *the slithy toves did gyre and
gimble* because every grammatical signal survives and only lexical content is vacant. Such
a token has a FIELD (its neighbourhood is fully specified by context) and no LOCATION (no
prior embedding).

Normative source: `specs/007-vacancy-transform-field/architecture.md`. That document — not
the Python or the TypeScript — is what both stacks implement, so **every departure from the
original `tiny-seuss/synth/jabberwockify.py` here is one the contract lists in its §9**:

1. Words are found with the tokenizer's own :data:`~llm_geometry.lex.vocab.WORD_RE`, not the
   source's `[A-Za-z][A-Za-z']*`. Otherwise the transform and the trainer disagree about
   `good-bye` and the relabelling theorem (contract §7.3) is simply false.
2. ``u = (top64 >> 11) / 2**53``, not ``top64 / 2**64``. A 64-bit integer over 2**64 is not
   exactly representable as a float64, so Python and JavaScript can land on different
   doubles for the same digest and disagree about a word at the boundary.
3. `random.Random` is replaced by a sha256 counter stream: MT19937 seeded from a string is
   not reproducible in TypeScript.
4. The map is built **once over the whole type set in canonical order**, never lazily while
   rewriting. The source's `used` set and give-up counter make a word's nonce depend on `p`
   and on document order, which breaks the stability property the source claims for itself.
5. The domain is avoided **implicitly** — there is no caller-supplied `avoid` parameter — so
   a minted form can never merge with a real English type and the map is a pure function of
   `(domain, seed, match_prosody)`. The source accepts an `avoid` argument and never passes
   one; making it optional reproduces the same failure a level up, where the map depends on
   what the caller remembered (measured: seed 0 gives `remintRounds` 0 with the domain passed
   and 1 without, with different nonces either way — both valid, which is the problem).
6. The give-up path is a deterministic salt relaxation, not `syllable + str(len(used))`.
7. Seams (`wee` + `er`) are repaired from a hash of `(stem, suffix)`, not a shared RNG.
8. Injectivity is **verified** over assembled surface forms, at every `p`, and re-minted on
   collision — the theorem depends on it, so it is checked on every build and reported as
   ``bijective`` / ``remintRounds``.
9. `split_suffix` carries the audited copy's ``SPLIT_EXCEPTIONS`` (`brother`, not
   `broth`+`er`).
10. No claim of exact prosody: :data:`STRESS_TABLE` is 61 hand entries over the Dolch list,
    described by its own author as "seeded by rule; wants roughly an hour of human
    checking", so every prosody number ships with the three-way `stressFrom*` split beside it.
11. ``corpusTypesVacated`` counts types actually vacated, measured from the output text, not
    ``len(map)``. Every count in §10 names its scope — corpus or domain — because an
    unprefixed "types" is ambiguous between the two and cost the stacks two round trips.

The properties that make a `p`-sweep interpretable are structural here rather than hoped
for: `u` depends only on `(seed, stem)`, so vacated sets are **nested** in `p`; the map is
built independently of `p`, so a stem's nonce is **stable** across the whole sweep.

**Two minting strategies** (``VacancyParams.mint``, contract §8.3). ``"nonce"`` invents the
replacement; ``"swap"`` draws a REAL English word from the domain's own open-class types by
frequency rank, which is the control that separates *wrong content* from *unknown form* for
the pretrained arm. They differ in exactly one property, and §5.2a proves the difference is
forced rather than a shortcoming of this implementation: a swap map's images ARE domain
types, so a vacated word can land on one that has not moved, and no `p`-stable such map is
injective at intermediate `p` unless it is the identity. Swap is therefore a bijection of
the domain at full vacancy — where the pretrained arm measures, and where the invariance
theorem holds for it exactly as for nonce — and :func:`map_vocab_words` refuses the rest
rather than manufacturing a vocabulary with two words on one row.

Two defects that the first implementation of this contract exposed are now fixed in the
contract itself, and each is covered by a named test here:

* **The transform commutes with lowercasing** (§5.7): everything — the seam test, the seam
  hash, the assembly — happens on the lowercased word, and `match_case` is applied to the
  WHOLE assembled surface with the original whole word as the case source. Slicing the suffix
  case-preserved made `gums` -> `flels` but `GUMS` -> `FLESS`, giving one type two surfaces.
* **Injectivity is checked over surface forms and holds at every `p`** (§5.2 conditions A and
  B). A bare-nonce check at `p = 1` missed `hang` -> `wak`, whose surface `wak` + `ed` is the
  real English word `waked`; the two collided at `p = 0.25` and `p = 0.5`, where `waked`
  itself was not vacated.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from ..errors import ComputeError, InvalidParamError
from .dolch import dolch_budget
from .vocab import WORD_RE

# --- the closed class -------------------------------------------------------------------

#: Contract §2.1, ported verbatim from the source and whitespace-split. The source carries a
#: warning we keep: an earlier version unioned this with the short Dolch service words, which
#: silently protected content verbs (`run`, `eat`, `see`, `get`, `let`, `put`) and understated
#: the vacancy rate. The closed class is THIS CURATED LIST ONLY.
FUNCTION_WORDS: frozenset[str] = frozenset("""a an the this that these those my your his her its our
their some any all both each every no none i me you he she it we they him them
us who whom whose which what where when why how is am are was were be been being
do does did done have has had having will would shall should can could may might
must not and or but so if then than as of to in on at by for with from into onto
up down out off over under again once here there very too also only just even
still yet ever never always about after before while because though although
unless until since during between among against through above below near far
one two three four five six seven eight nine ten""".lower().split())

# --- suffix splitting -------------------------------------------------------------------

#: Contract §3, order significant — the first match wins.
SUFFIXES: tuple[str, ...] = ("ing", "edly", "est", "ies", "'s", "n't", "ed", "es", "er", "ly", "s")

#: Words that are never split. From the AUDITED copy of the source (contract §3, departure 9);
#: without it `brother -> broth+er` and `morning -> morn+ing`, which the source itself flags as
#: a known artifact. This is a spelling heuristic, not a morphological analyser, and it stays
#: wrong outside the list (`ladder -> ladd+er`) — acceptable, because the nonce still carries a
#: consistent identity and an inflected-looking surface, but it must be stated in the UI rather
#: than quietly tolerated.
SPLIT_EXCEPTIONS: frozenset[str] = frozenset(
    {
        "brother",
        "father",
        "mother",
        "sister",
        "never",
        "over",
        "under",
        "morning",
        "giving",
        "thing",
    }
)

#: The stem must be ASCII letters only. `str.isalpha()` is Unicode-aware and would accept
#: letters JavaScript's `^[A-Za-z]+$` rejects; nothing in the shipped corpus exercises the
#: difference, a pasted corpus would (contract §2.2).
_STEM_RE = re.compile(r"[A-Za-z]+")

#: A vacated word must itself be a single complete `WORD_RE` match — checked, not assumed.
_WHOLE_WORD_RE = re.compile(WORD_RE.pattern + r"\Z")

# --- the phonotactic tables -------------------------------------------------------------
# Contract §5.4, ported verbatim, ORDER SIGNIFICANT: the index into each list is what the
# byte stream selects, so reordering silently changes every nonce in both stacks.

ONSETS: tuple[str, ...] = (
    "b", "br", "bl", "d", "dr", "f", "fl", "fr", "g", "gl", "gr", "h",
    "j", "k", "kl", "kr", "l", "m", "n", "p", "pl", "pr", "r", "s", "sk",
    "sl", "sm", "sn", "sp", "st", "str", "sw", "t", "tr", "th", "thr",
    "v", "w", "wr", "y", "z", "sh", "shr", "ch", "gn", "sc", "sq",
)  # fmt: skip

NUCLEI: tuple[str, ...] = (
    "a", "e", "i", "o", "u", "ai", "ee", "ea", "oo", "ou", "oa", "ie",
    "y", "au", "ur", "ir", "or", "ar", "er",
)  # fmt: skip

CODAS: tuple[str, ...] = (
    "", "b", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t", "v",
    "z", "sh", "ch", "th", "ck", "ff", "ll", "mp", "nd", "ng", "nk", "nt",
    "sk", "sp", "st", "ft", "lt", "lk", "rd", "rk", "rm", "rn", "rt", "ble",
    "dle", "gle", "kle", "tle", "mble", "ndle", "ffle", "zzle",
)  # fmt: skip

UNSTRESSED_TAILS: tuple[str, ...] = (
    "y", "le", "er", "ow", "en", "el", "ish", "ous", "id",
    "ic", "um", "ent", "ing",
)  # fmt: skip

#: Prefixes for an unstressed FIRST syllable.
UNSTRESSED_ONSETS: tuple[str, ...] = ("a", "be", "re", "de", "un", "en")

#: The reduced coda set for an unstressed syllable. The duplicated empty string doubles its
#: weight; keep it (contract §5.4).
#:
#: **This table is UNREACHABLE and that is correct.** §5.5 step 2 has exactly three branches
#: and none of them draws from here, exactly as in the source, where `_syl(stressed=False)`
#: is never called. An earlier draft of the contract implied a fourth branch; it was
#: self-contradictory and has been removed. It is retained here for fidelity to the source's
#: tables and because **adding a use would shift the list indices the byte stream selects and
#: change every multi-syllable nonce in both stacks**. Do not "fix" it.
REDUCED_CODAS: tuple[str, ...] = ("", "", "l", "n", "r", "s")

# --- prosody ----------------------------------------------------------------------------

#: Contract §6.1, ported verbatim from `tiny-seuss/synth/lexicon.py`: 61 polysyllables of the
#: Dolch list, "seeded by rule and then overridden by a hand table", and listed by the source
#: itself under *not yet exercised*. So we do not claim exact prosody and no UI string may:
#: the shipped corpus is *The Real Mother Goose*, most of whose types are not in this table
#: and therefore fall through to :func:`rule_syllables`. Every prosody statistic must be shown
#: with ``stressFromTable``, which is the honesty of every other prosody number.
#: `Santa Claus` is retained verbatim even though a word tokenizer can never match it.
STRESS_TABLE: dict[str, str] = {
    "away": "01", "funny": "10", "little": "100", "yellow": "10",
    "into": "10", "over": "10", "pretty": "10", "under": "10",
    "after": "10", "again": "01", "any": "10", "every": "100",
    "giving": "10", "once": "1", "open": "10",
    "always": "100", "around": "01", "because": "01", "before": "01",
    "seven": "10", "eight": "1", "myself": "01", "never": "10",
    "only": "10", "today": "01", "together": "0100", "better": "10",
    "carry": "10", "many": "10", "upon": "01", "very": "100",
    "apple": "10", "baby": "10", "birthday": "100", "brother": "10",
    "chicken": "10", "children": "100", "Christmas": "10",
    "farmer": "10", "flower": "10", "garden": "10", "good-bye": "01",
    "horse": "1", "kitty": "10", "letter": "10", "money": "10",
    "morning": "10", "mother": "10", "paper": "100", "party": "10",
    "picture": "10", "rabbit": "10", "robin": "10", "squirrel": "1",
    "table": "10", "water": "10", "window": "10", "Santa Claus": "101",
    "father": "10", "sister": "10", "summer": "10",
}  # fmt: skip

#: Contract §6.4. `anapest` is the Seuss engine (and Byron's *The Destruction of
#: Sennacherib*, which is where he got it).
FEET: dict[str, str] = {"anapest": "001", "iamb": "01", "trochee": "10", "dactyl": "100"}

_VOWEL_GROUP_RE = re.compile(r"[aeiouy]+")
_NON_LOWER_RE = re.compile(r"[^a-z]")

# --- minting ----------------------------------------------------------------------------

#: Collapse a run of three or more identical consonants to two (contract §5.5 step 3).
_RUN_RE = re.compile(r"([bcdfghjklmnpqrstvwxz])\1{2,}")

#: Contract §5.5 step 5: deterministic, order-independent relaxations, unlike the source's
#: give-up counter. Reaching :data:`MINT_MAX_SALT` raises — it has never happened and if it
#: does we want to know.
#:
#: **These thresholds are on the ATTEMPT COUNTER `a`, not on the absolute salt.** A mint call
#: carries a base salt `S`; the byte stream is keyed on `S + a` for `a = 0, 1, 2, …`, and a
#: re-mint restarts `a` at 0. So a re-mint is held to exactly the same quality bar as an
#: original mint — which is why the seed-7 re-mint of `hang` comes back monosyllabic
#: (`smeeg`) rather than falling out of a loop with every check already relaxed.
MINT_RELAX_SYLLABLES_SALT = 400
MINT_RELAX_LENGTH_SALT = 800
MINT_MAX_SALT = 1200

#: Contract §5.2/§7.3: injectivity is verified, not assumed, and re-minted on collision.
MAX_REMINT_ROUNDS = 8

# --- the swap control (contract §8.3) ----------------------------------------------------

#: Half-width of the frequency-rank window a swap replacement is drawn from: attempt `a`
#: proposes ``pool[(r + delta) % len(pool)]`` for ``delta`` in ``[-w, -1] | [1, w]``.
SWAP_WINDOW = 32

#: The window doubles every this many attempts, up to the whole pool — the same deterministic
#: relaxation §5.5 uses, and necessary for the same reason: "anything already used" depletes a
#: fixed window for the stems late in canonical order.
SWAP_WIDEN_EVERY = 64

#: Attempt at which the prosody filter is dropped, mirroring :data:`MINT_RELAX_SYLLABLES_SALT`.
SWAP_RELAX_PROSODY = 1024

#: Reaching this many attempts raises rather than looping — the pool is finite, so unlike
#: minting this bound is reachable in principle and we want to know if it ever is.
SWAP_MAX_ATTEMPTS = 4096

#: The two minting strategies of contract §7.1 / §8.3.
MINT_STRATEGIES: tuple[str, ...] = ("nonce", "swap")

#: Contract §5.8: a re-mint jumps to a fresh region of the byte stream deterministically.
REMINT_SALT_STRIDE = 1000

#: Characters a seam repair may substitute (contract §5.7).
_SEAM_CHARS = "lnrtk"


# --- word segmentation ------------------------------------------------------------------


def _reject_bare_str(value: object, func: str, param: str) -> None:
    """Raise if a text was passed where an iterable of TYPES was expected.

    ``Iterable[str]`` happily accepts a ``str`` and iterates it character by character, so
    ``vacancy_domain(corpus_text)`` silently yields a domain of single letters. That failure
    surfaces much later and somewhere else — as `stem 'real' is outside the vacancy map's
    domain` — so it is caught here, at the actual mistake.

    ``TypeError`` rather than :class:`InvalidParamError`: this is a wrong argument TYPE, not a
    value out of range, and it can never reach an HTTP boundary as a user's fault.
    """
    if isinstance(value, str):
        raise TypeError(
            f"{func}({param}=...) takes an iterable of word types, not a text; "
            f"pass tokenize(text) or set(tokenize(text)), not the text itself"
        )


def stem_and_suffix(word: str) -> tuple[str, str]:
    """Split an inflectional suffix off `word` (contract §3).

    The suffixes are tried in :data:`SUFFIXES` order and the first match wins. A suffix `s`
    matches iff ``lower(word).endswith(s)`` and ``len(word) - len(s) >= 3``. The split slices
    the ORIGINAL word, so case is preserved. Words in :data:`SPLIT_EXCEPTIONS` are never
    split.

    Preserving the suffix is what keeps the syntax parseable: the stem is vacated and the
    suffix re-attached, so the nonce still looks inflected.
    """
    lower = word.lower()
    if lower in SPLIT_EXCEPTIONS:
        return word, ""
    for suffix in SUFFIXES:
        if lower.endswith(suffix) and len(word) - len(suffix) >= 3:
            cut = len(word) - len(suffix)
            return word[:cut], word[cut:]
    return word, ""


def is_eligible(stem: str, keep: frozenset[str] = FUNCTION_WORDS) -> bool:
    """Is this stem open-class, i.e. may it be vacated at all? (Contract §2.2.)

    `keep` is the EFFECTIVE closed class — :data:`FUNCTION_WORDS` unioned with any caller
    extras. Pass :attr:`VacancyParams.keep_set`, never :attr:`VacancyParams.keep`, or the
    function words lose their protection.

    Test 2 is what makes hyphenated and apostrophised words behave as they do, and both
    stacks must agree on it exactly: `good-bye` matches no suffix, so its stem contains a
    hyphen and it is **never** vacated; `dog's` splits to `dog`, which passes, giving
    ``<nonce>'s``.
    """
    return stem.lower() not in keep and _STEM_RE.fullmatch(stem) is not None and len(stem) > 2


def match_case(src: str, new: str) -> str:
    """Carry `src`'s capitalisation onto `new`, so `Jack` becomes `Flim`, not `flim`.

    Applied to the WHOLE assembled surface form with the ORIGINAL WHOLE WORD as `src`
    (contract §5.7), never to the nonce alone with a case-preserved suffix appended after —
    that is what broke `GUMS`.
    """
    if src.isupper() and len(src) > 1:
        return new.upper()
    if src[:1].isupper():
        return new[:1].upper() + new[1:]
    return new


# --- the vacancy decision ---------------------------------------------------------------


def vacancy_u(stem: str, seed: int) -> float:
    """The stem's position in [0, 1) — vacate iff ``u(stem) < p`` (contract §4).

    ``u`` is a function of ``(seed, stem)`` alone: not of `p`, not of traversal order, not of
    which other words exist. That is what makes the vacated sets NESTED as `p` grows, which is
    the first of the two properties a `p`-sweep needs to be interpretable.

    The ``>> 11`` is mandatory and is a departure from the source (which used
    ``top64 / 2**64``). A 64-bit integer divided by 2**64 is not exactly representable as a
    float64, so Python and JavaScript can round to different doubles for the same digest and
    disagree about a word at the boundary. Shifting to 53 bits makes the numerator exactly
    representable, so ``u`` is *the same double* in both languages.
    """
    digest = hashlib.sha256(f"{seed}:{stem.lower()}".encode("utf-8")).digest()
    return (int.from_bytes(digest[:8], "big") >> 11) / 2**53


# --- prosody ----------------------------------------------------------------------------


def rule_syllables(word: str) -> int:
    """The spelling fallback for syllable counting (contract §6.2).

    The source has a further ``if w.endswith("le") ...: pass`` branch; it is dead code and is
    not ported, so this is byte-identical to the source's behaviour.
    """
    w = _NON_LOWER_RE.sub("", word.lower().strip("'-"))
    if not w:
        return 1
    n = len(_VOWEL_GROUP_RE.findall(w))
    if w.endswith("e") and n > 1 and not w.endswith(("le", "ee", "ye")):
        n -= 1
    return max(1, n)


def stress_source(word: str, minted_stress: Mapping[str, str] | None = None) -> str:
    """Where :func:`stress` got its answer: ``"minted"``, ``"table"``, or ``"rule"``.

    ``"table"`` is the fraction the UI must publish next to every prosody number — it is the
    part that a human has (nominally) checked. ``"minted"`` is exact by construction but is a
    form we invented, and ``"rule"`` is a spelling guess.
    """
    lower = word.lower()
    if minted_stress and lower in minted_stress:
        return "minted"
    if word in STRESS_TABLE or lower in STRESS_TABLE:
        return "table"
    return "rule"


def stress(word: str, minted_stress: Mapping[str, str] | None = None) -> str:
    """The stress pattern of `word` as a string of ``0``/``1`` (contract §6.3).

    Lookup order, exactly: the minted patterns (so prosody scoring on a vacated corpus is
    exact *for the forms we minted*), then :data:`STRESS_TABLE` case-sensitively (for
    `Christmas`), then case-insensitively, then the spelling rule.

    `minted_stress` is a parameter rather than the source's module-level global: a global
    would make the answer depend on which corpora had been transformed earlier in the process.
    """
    lower = word.lower()
    if minted_stress and lower in minted_stress:
        return minted_stress[lower]
    if word in STRESS_TABLE:
        return STRESS_TABLE[word]
    if lower in STRESS_TABLE:
        return STRESS_TABLE[lower]
    n = rule_syllables(lower)
    return "1" if n == 1 else "1" + "0" * (n - 1)


def syllables(word: str, minted_stress: Mapping[str, str] | None = None) -> int:
    """``len(stress(word))`` — the syllable count implied by the stress pattern."""
    return len(stress(word, minted_stress))


def scan(line: str, minted_stress: Mapping[str, str] | None = None) -> str:
    """The line's concatenated stress string, e.g. ``0100100100``.

    Words are found with the tokenizer's regex but NOT lowercased, so the case-sensitive
    :data:`STRESS_TABLE` lookup can still fire.
    """
    return "".join(stress(t, minted_stress) for t in WORD_RE.findall(line))


def meter_score(
    line: str, foot: str = "anapest", minted_stress: Mapping[str, str] | None = None
) -> float:
    """Fraction of syllable positions in `line` that match the repeating `foot`.

    ``0.0`` for a line with no syllables (contract §6.4).
    """
    if foot not in FEET:
        raise InvalidParamError(
            f"unknown foot {foot!r}; expected one of {sorted(FEET)}",
            {"foot": foot},
        )
    s = scan(line, minted_stress)
    if not s:
        return 0.0
    pattern = FEET[foot]
    target = (pattern * (len(s) // len(pattern) + 1))[: len(s)]
    return sum(a == b for a, b in zip(s, target)) / len(s)


# --- the deterministic byte stream ------------------------------------------------------


class _ByteStream:
    """A sha256 counter stream (contract §5.3), trivially identical in both languages.

    ``random.Random`` cannot be ported: MT19937 seeded from a string is not reproducible in
    TypeScript without reimplementing the generator *and* `Random.choice`'s masking.
    """

    __slots__ = ("_prefix", "_counter", "_block", "_offset")

    def __init__(self, seed: int, stem: str, salt: int, tag: str = "mint") -> None:
        #: `tag` separates the minting stream from the swap-draw stream of §8.3. It is part
        #: of the hashed prefix, so the two can never alias however the salts line up.
        self._prefix = f"{seed}:{tag}:{stem}:{salt}:"
        self._counter = 0
        self._block = hashlib.sha256(f"{self._prefix}0".encode("utf-8")).digest()
        self._offset = 0

    def u32(self) -> int:
        """The next big-endian unsigned 32-bit word, refilling from the next counter."""
        if self._offset + 4 > len(self._block):
            self._counter += 1
            self._block = hashlib.sha256(f"{self._prefix}{self._counter}".encode("utf-8")).digest()
            self._offset = 0
        value = int.from_bytes(self._block[self._offset : self._offset + 4], "big")
        self._offset += 4
        return value

    def choice(self, options: Sequence[str]) -> str:
        """``options[u32() % len(options)]``.

        Every list here is shorter than 256, so the modulo bias is aesthetic rather than
        statistical — but both stacks must bias IDENTICALLY, which this does.
        """
        return options[self.u32() % len(options)]


def _mint(
    key: str,
    seed: int,
    match_prosody: bool,
    forbidden: frozenset[str] | set[str],
    start_salt: int = 0,
    stem: str | None = None,
) -> tuple[str, str, int]:
    """Mint one nonce for `key`, returning ``(nonce, intended stress pattern, salt)``.

    Contract §5.5. Depends only on ``(seed, key, match_prosody, forbidden, start_salt)`` —
    and `forbidden` depends only on the canonically-ordered prefix of stems before this one —
    so a stem's nonce is the same at every `p`. That is the stability property (§5.6).

    **`key` feeds the byte stream and the uniqueness check ONLY; the stress pattern comes
    from `stem`.** They differ under ``consistent=False``, where the key is
    ``f"{stem}#{idx}"`` (§5.8). Letting the key reach the prosody lookup gives
    ``stress("little#0") == "10"`` instead of ``stress("little") == "100"``, so `Little`
    mints as `Wrerken` rather than `Wrerkenle` — §7.1 says the nonce carries *the stem's*
    syllable count and stress, and a mint key is not a word. `stem` defaults to `key`, which
    is the consistent case where they are the same string.

    `start_salt` is the base salt `S`; the byte stream is keyed on ``S + a`` for the attempt
    counter ``a = 0, 1, 2, …``, and the quality relaxations are thresholds on ``a``. The
    ACCEPTING salt (``S + a``) is returned because §5.8 needs it: a re-mint restarts at
    ``REMINT_SALT_STRIDE * round + previousSalt + 1``, with its own ``a`` back at 0.

    Three branches in step 2, in that order, exactly as §5.5 spells them out. There is no
    fourth: :data:`REDUCED_CODAS` is unreachable here, as it is in the source, where
    ``_syl(stressed=False)`` is never called. **Do not "fix" this** — a fourth branch would
    shift every list index the byte stream selects and change every multi-syllable nonce.
    """
    pattern = stress(key if stem is None else stem) if match_prosody else "1"
    n_syl = len(pattern)
    for attempt in range(MINT_MAX_SALT):  # `a` of §5.5; the stream is keyed on `S + a`
        salt = start_salt + attempt
        rnd = _ByteStream(seed, key, salt)
        parts: list[str] = []
        for i, mark in enumerate(pattern):
            if mark == "1":
                parts.append(rnd.choice(ONSETS) + rnd.choice(NUCLEI) + rnd.choice(CODAS))
            elif i == 0:
                parts.append(rnd.choice(UNSTRESSED_ONSETS))
            else:
                parts.append(rnd.choice(UNSTRESSED_TAILS))
        w = _RUN_RE.sub(r"\1\1", "".join(parts))
        long_enough = len(w) >= 3 or attempt >= MINT_RELAX_LENGTH_SALT
        right_length = syllables(w) == n_syl or attempt >= MINT_RELAX_SYLLABLES_SALT
        if long_enough and right_length and w not in forbidden:
            return w, pattern, salt
    raise ComputeError(
        f"could not mint a nonce for {key!r} in {MINT_MAX_SALT} attempts",
        {"key": key, "stem": stem, "seed": seed, "pattern": pattern, "start_salt": start_salt},
    )


def type_counts(tokens: Iterable[str]) -> dict[str, int]:
    """Occurrences per lowercased type — the frequency source the swap control ranks by.

    Takes the TOKEN STREAM (``tokenize(text)``), not the type set: a set has no frequencies,
    and ``mint="swap"`` needs them. Passing the deduplicated domain here would silently rank
    every type equally and give a map that is alphabetical rather than frequency-matched, so
    :func:`build_vacancy_map` requires this to be passed explicitly and raises without it
    rather than inventing a fallback.
    """
    _reject_bare_str(tokens, "type_counts", "tokens")
    counts: dict[str, int] = {}
    for t in tokens:
        key = t.lower()
        counts[key] = counts.get(key, 0) + 1
    return counts


def swap_pool(domain: Iterable[str], counts: Mapping[str, int], keep: frozenset[str]) -> list[str]:
    """The replacement pool of contract §8.3: the domain's open-class TYPES by frequency rank.

    Ordered by ``(count descending, type ascending)`` — the tie rule
    :func:`~llm_geometry.lex.vocab.frequency_budget` already uses, so "frequency rank" means
    one thing in this codebase. A type absent from the corpus (the 22 Dolch-only words) has
    count 0 and ranks last, alphabetically among its equals.

    **Types, not stems.** The stem set is exactly the set of keys the map assigns, so drawing
    from it would consume the pool exactly and leave an A-collision with nowhere to move. On
    the shipped corpus the pool is 1944 types against 1680 stems, which is the slack the
    re-draw rounds spend.
    """
    return sorted(
        {t.lower() for t in domain if is_eligible(stem_and_suffix(t)[0], keep)},
        key=lambda t: (-counts.get(t, 0), t),
    )


def swap_rank(
    stem: str, family: Iterable[str], pool: Sequence[str], counts: Mapping[str, int]
) -> int:
    """Where `stem` sits in the frequency-ranked pool (contract §8.3).

    Not ``pool.index(stem)``: 375 of the shipped corpus's 1680 eligible stems are not domain
    types at all — `hang` and `gum` reach the map only as the stems of `hanged` and `gums` —
    so a lookup would raise on a fifth of them. The rank is instead the position the stem's
    own key would take in the pool's order, with the stem's frequency taken over its whole
    INFLECTIONAL FAMILY: `hang` is as frequent as `hanged` + `hanging` make it, which is the
    frequency a reader of the corpus actually meets.

    Defined as the number of pool entries whose key sorts strictly before the stem's, so it
    is a plain count and cannot be read two ways; the binary search is only how it is
    computed. Ties fall to the stem's own alphabetical position, exactly as the pool order
    does.
    """
    freq = sum(counts.get(t, 0) for t in family)
    key = (-freq, stem)
    lo, hi = 0, len(pool)
    while lo < hi:
        mid = (lo + hi) // 2
        other = pool[mid]
        if (-counts.get(other, 0), other) < key:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _draw_swap(
    stem: str,
    seed: int,
    match_prosody: bool,
    pool: Sequence[str],
    rank: Mapping[str, int],
    used: frozenset[str] | set[str],
    suffixes: Sequence[str],
    claimed: frozenset[str] | set[str],
    barred: frozenset[str] | set[str],
    start_salt: int = 0,
) -> tuple[str, int, list[str]]:
    """Draw one real-word replacement for `stem` (contract §8.3).

    Returns ``(word, salt, surface forms)`` — the forms the stem now owns, one per suffix it
    occurs with, so the caller can claim them.

    Deterministic in ``(seed, stem, pool, used, claimed, start_salt)`` and independent of `p`,
    exactly as :func:`_mint` is, so §5.6's stability property survives the swap control.

    Attempt `a` widens the window every :data:`SWAP_WIDEN_EVERY` attempts and drops the
    prosody filter at :data:`SWAP_RELAX_PROSODY`; both relaxations are functions of `a` alone,
    never of how many stems happen to have been assigned first. The byte stream carries the
    ``swap`` tag, so it can never alias the ``mint`` stream at the same salt.

    **Conditions A and B₁ are enforced here, at draw time, not only checked afterwards.** A
    candidate is rejected if any of the surfaces it would produce is already claimed by an
    earlier stem, is an ineligible domain type, or equals the very type it replaces. Checking
    only afterwards costs 29 collision rounds on the shipped corpus at seed 0 and does not
    converge inside the 8 §5.2 allows: the pool holds inflected types, so a bare stem drawing
    `years` and a suffixed one drawing `year` land on the same surface, and re-drawing one of
    them at random walks into the next such pair. Enforcing at draw time makes the map
    correct by construction and the verification loop below a check rather than a search.
    """
    n = len(pool)
    if n == 0:
        raise ComputeError(
            "the swap pool is empty — no domain type has an eligible stem, so there is no "
            "real word to draw (contract §8.3)",
            {"stem": stem, "seed": seed},
        )
    r = rank[stem]
    pattern = stress(stem) if match_prosody else None
    family = {stem + suffix for suffix in suffixes}
    for attempt in range(SWAP_MAX_ATTEMPTS):
        salt = start_salt + attempt
        # The doubling is capped at 20 before the min, not because 20 is meaningful but
        # because JavaScript's `<<` takes its shift count modulo 32: an uncapped
        # `attempt // 64` reaches 63 at the give-up bound and the two stacks would compute
        # different widths from the same attempt. 32 << 20 already exceeds any pool.
        width = min(SWAP_WINDOW << min(attempt // SWAP_WIDEN_EVERY, 20), n)
        offset = _ByteStream(seed, stem, salt, tag="swap").u32() % (2 * width)
        delta = offset - width if offset < width else offset - width + 1
        candidate = pool[(r + delta) % n]
        if candidate == stem or candidate in used or candidate in family:
            continue
        if pattern is not None and attempt < SWAP_RELAX_PROSODY and stress(candidate) != pattern:
            continue
        forms = [surface_form(candidate, stem, suffix, seed) for suffix in suffixes]
        if any(f in claimed or f in barred or f in family for f in forms):
            continue
        if len(set(forms)) != len(forms):
            continue
        return candidate, salt, forms
    raise ComputeError(
        f"could not draw a swap replacement for {stem!r} in {SWAP_MAX_ATTEMPTS} attempts — "
        "the pool is finite, so report this rather than raising the bound",
        {"stem": stem, "seed": seed, "pool": n, "used": len(used), "start_salt": start_salt},
    )


def _seam_fix(nonce: str, suffix: str, stem: str, seed: int) -> str:
    """Repair a seam like ``wee`` + ``er`` -> ``weeer`` (contract §5.7).

    Deterministic in `(stem, suffix)`, so it is order-independent — the source used a shared
    RNG here, which is not. The substitution is applied once, not looped: the hash is fixed,
    so a loop would never terminate on a repeat.

    `stem` and `suffix` must already be LOWERCASED; see :func:`surface_form`.
    """
    if not suffix or not nonce or nonce[-1] != suffix[0]:
        return nonce
    digest = hashlib.sha256(f"{seed}:seam:{stem}:{suffix}".encode("utf-8")).digest()
    return nonce[:-1] + _SEAM_CHARS[int.from_bytes(digest[:4], "big") % len(_SEAM_CHARS)]


def surface_form(nonce: str, stem: str, suffix: str, seed: int) -> str:
    """The assembled, LOWERCASED output of §5.7 — seam repaired, suffix re-attached.

    This is the unit the injectivity conditions of §5.2 are stated over, and the only place
    the transform assembles anything, so the case-commuting invariant

        ``lower(transform_word(w)) == transform_word(lower(w))``

    holds by construction: nothing downstream of here branches on case.
    """
    key, tail = stem.lower(), suffix.lower()
    return _seam_fix(nonce, tail, key, seed) + tail


# --- parameters -------------------------------------------------------------------------


@dataclass(frozen=True)
class VacancyParams:
    """The knobs of contract §7.1.

    Three of them — `p`, `seed`, `match_prosody` — are INVISIBLE to a word-level model
    trained from scratch, because such a model never sees the letters: with
    ``consistent=True`` and ``reveal_after=0`` the transform is a pure relabelling of the
    vocabulary and training is bit-identical. Only the knobs that break type identity
    (``consistent=False``, ``reveal_after>0``) can move a loss. That is the honest tiny-arm
    result and it is worth stating plainly rather than dressing a null up as a curve.
    """

    #: Fraction of eligible TYPES vacated. Compared as given; the UI emits two decimal
    #: places and both stacks parse it as a float64.
    p: float = 0.0
    #: Selects both `u` and the nonce assignment.
    seed: int = 0
    #: One nonce per source type, corpus-wide. ``False`` is the source's "inconsistent
    #: assignment" control: same vacancy rate, no learnable identity, and DELIBERATELY no
    #: stability property.
    consistent: bool = True
    #: The nonce carries the stem's syllable count and stress.
    match_prosody: bool = True
    #: The first N occurrences of a vacated stem keep their English form, seeding a partial
    #: location. ``0`` is the pure case.
    reveal_after: int = 0
    #: Extra words added to the closed class. The effective set is
    #: ``FUNCTION_WORDS | lower(keep)`` — see :attr:`keep_set`.
    keep: frozenset[str] = frozenset()
    #: How a replacement is produced (contract §8.3). ``"nonce"`` invents a phonotactically
    #: legal form; ``"swap"`` draws a REAL English word from the domain's own open-class types
    #: by frequency rank, so the passage stays equally nonsensical while every form remains a
    #: known word with ordinary tokenization. That is the control that separates "wrong
    #: content" from "unknown form" in the pretrained arm — and §5.2a proves it can only be
    #: injective at full vacancy, which is where the pretrained arm measures.
    mint: str = "nonce"

    _keep_set: frozenset[str] = field(default=frozenset(), repr=False, compare=False)

    def __post_init__(self) -> None:
        if isinstance(self.p, bool) or not isinstance(self.p, (int, float)):
            raise InvalidParamError(f"p must be a number, got {self.p!r}", {"p": self.p})
        if not math.isfinite(self.p) or not 0.0 <= self.p <= 1.0:
            raise InvalidParamError(f"p must lie in [0, 1], got {self.p!r}", {"p": self.p})
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise InvalidParamError(f"seed must be an int, got {self.seed!r}", {"seed": self.seed})
        if isinstance(self.reveal_after, bool) or not isinstance(self.reveal_after, int):
            raise InvalidParamError(
                f"reveal_after must be an int, got {self.reveal_after!r}",
                {"reveal_after": self.reveal_after},
            )
        if self.reveal_after < 0:
            raise InvalidParamError(
                f"reveal_after must be >= 0, got {self.reveal_after}",
                {"reveal_after": self.reveal_after},
            )
        if self.mint not in MINT_STRATEGIES:
            raise InvalidParamError(
                f"mint must be one of {list(MINT_STRATEGIES)}, got {self.mint!r}",
                {"mint": self.mint},
            )
        if self.mint == "swap" and not self.consistent:
            raise InvalidParamError(
                "mint='swap' requires consistent=True: the inconsistent control needs a fresh "
                "type per occurrence and the corpus has 1680 open-class stems against 8202 "
                "vacated tokens, so there is no supply of real words to draw (contract §8.3)",
                {"mint": self.mint, "consistent": self.consistent},
            )
        # Same trap as `vacancy_domain`: `keep="little"` would quietly protect six letters.
        _reject_bare_str(self.keep, "VacancyParams", "keep")
        for w in self.keep:
            if not isinstance(w, str):
                raise InvalidParamError(
                    f"keep must contain strings, got {w!r}", {"keep": sorted(map(str, self.keep))}
                )
        object.__setattr__(self, "keep", frozenset(self.keep))
        object.__setattr__(self, "_keep_set", FUNCTION_WORDS | {w.lower() for w in self.keep})

    @property
    def keep_set(self) -> frozenset[str]:
        """The EFFECTIVE closed class: :data:`FUNCTION_WORDS` plus the caller's extras."""
        return self._keep_set


# --- the map ----------------------------------------------------------------------------


class _RewriteState:
    """Per-rewrite bookkeeping for the order-DEPENDENT conditions only.

    ``consistent=True, reveal_after=0`` — the condition the invariance theorem is stated for —
    touches nothing here except the counter, and the counter never changes an output.

    `used` is seeded from the map's STORED :attr:`VacancyMap.forbidden` (§5.8) — the domain
    plus every nonce ever handed out, superseded ones included — not from
    ``mapping.values()``, which drops the superseded ones and is the reconstruction the two
    stacks disagreed on.
    """

    __slots__ = ("counts", "used")

    def __init__(self, used: set[str]) -> None:
        self.counts: dict[str, int] = {}
        self.used = used


@dataclass(frozen=True)
class VacancyMap:
    """A `p`-independent stem -> nonce assignment, plus the evidence that it is injective.

    Built once over the whole type set in canonical order (contract §5.2), so the map at any
    `p` is just the restriction of this one map to ``{stem : u(stem) < p}``. Nesting and
    stability are therefore structural facts rather than properties to be hoped for.

    The two mappings are plain dicts for cheap lookup and must not be mutated by callers; the
    only writer is the inconsistent-assignment control, which registers the patterns of the
    forms it mints so that prosody scoring stays exact.
    """

    #: ``lower(stem) -> nonce``.
    mapping: dict[str, str]
    #: ``nonce -> intended stress pattern``, for :func:`stress`'s first lookup.
    minted_stress: dict[str, str]
    seed: int
    match_prosody: bool
    #: The lowercased domain this map was built over — :func:`vacancy_domain`, i.e. the
    #: corpus's types plus the full Dolch list. It is also the set a nonce may never equal,
    #: so a minted form can never silently merge with a real English word. Kept because §10's
    #: `domainTypes*` counts are over it and it is NOT recoverable from the corpus text:
    #: budget-only words have images but never appear.
    domain: frozenset[str]
    #: ``|image|`` of the domain, at full vacancy.
    image_size: int
    #: ``image_size == type_count``. The relabelling theorem depends on it.
    bijective: bool
    #: How many collision-driven re-mint rounds were needed. Expected to be 0.
    remint_rounds: int
    #: Every replacement ever handed out, PLUS the whole domain — including nonces a re-mint
    #: round superseded (contract §5.8). STORED, never reconstructed as
    #: ``domain | mapping.values()``: that reconstruction silently drops the superseded ones
    #: (`wak` at seed 7), and the per-occurrence path of the ``consistent=False`` control
    #: draws against this set, so reusing a form that was rejected for cause can recreate the
    #: very collision the re-mint resolved.
    forbidden: frozenset[str]
    #: Is the map injective at EVERY `p`, or only at full vacancy? ``True`` for
    #: ``mint="nonce"``, where condition B keeps every image out of the domain; ``False`` for
    #: ``mint="swap"``, whose images ARE domain types — §5.2a proves no `p`-stable swap can do
    #: better, and :func:`map_vocab_words` refuses the cases where it matters.
    injective_at_every_p: bool

    @property
    def type_count(self) -> int:
        """``|domain|`` — what `image_size` must equal for the map to be injective."""
        return len(self.domain)

    def nonce_for(self, stem: str) -> str | None:
        """The nonce assigned to `stem`, or ``None`` if the stem is outside the domain."""
        return self.mapping.get(stem.lower())

    def apply_word(self, word: str, params: VacancyParams) -> str:
        """Transform a single word in isolation.

        Convenience for one-off queries and for the order-INDEPENDENT conditions. Rewriting a
        text goes through :func:`vacate_text`, which threads the occurrence counters that
        ``reveal_after`` and ``consistent=False`` need.
        """
        return self._transform(word, params, _RewriteState(set(self.forbidden)))

    def _transform(self, word: str, params: VacancyParams, state: _RewriteState) -> str:
        stem, suffix = stem_and_suffix(word)
        if not is_eligible(stem, params.keep_set):
            return word
        key = stem.lower()
        if vacancy_u(key, params.seed) >= params.p:
            return word
        seen = state.counts.get(key, 0) + 1
        state.counts[key] = seen
        if seen <= params.reveal_after:
            return word

        if params.consistent:
            nonce = self.mapping.get(key)
            if nonce is None:
                raise ComputeError(
                    f"stem {key!r} is outside the vacancy map's domain — the map must be "
                    "built over the union of the corpus types and the budget's words",
                    {"stem": key},
                )
        else:
            # The inconsistent-assignment control: the nonce is derived from
            # (stem, occurrence index) in document order, so every occurrence is a fresh
            # type. This condition has NO stability property; destroying the field while
            # holding the vacancy rate fixed is its entire purpose.
            # §5.8 pins the key: `f"{stem}#{idx}"` with `idx` the 0-based occurrence index
            # of the STEM in document order. `#` is not a legal `WORD_RE` character, so the
            # key can never collide with a real stem — and the key must NOT reach the
            # prosody lookup, or the pattern becomes `stress("little#0")` rather than
            # `stress("little")` and the nonce loses the stem's syllable count.
            #
            # **Condition B applies here too** (§5.8): the nonce may equal neither a domain
            # type nor THE STEM IT REPLACES, and `{key}` is not redundant with `self.domain`
            # — a stem need not be a type. Measured: at seed 7, `p = 1`, `tak` minted `tak`,
            # so `Taking -> Taking` and one token silently failed to vacate
            # (`corpus_types_vacated` 1921 against the consistent path's 1922). §7.1 denies
            # this control a *stability* property, which is about a nonce being reused
            # across occurrences; it does not license a word surviving the transform, and a
            # control whose vacancy rate is not the stated rate is not a control. Adding the
            # stem to `forbidden` puts it through §5.5's ordinary re-mint loop, so the
            # replacement is held to the same quality bar as any other nonce.
            # `state.used` starts as the map's STORED `forbidden` (domain + every nonce ever
            # handed out, superseded ones included — §5.8), so this is exactly the set the
            # TypeScript control draws against. `{key}` is not redundant with it: a stem need
            # not be a type.
            nonce, pattern, _salt = _mint(
                f"{key}#{seen - 1}",
                params.seed,
                params.match_prosody,
                state.used | {key},
                stem=key,
            )
            state.used.add(nonce)
            self.minted_stress.setdefault(nonce, pattern)

        out = match_case(word, surface_form(nonce, key, suffix, params.seed))
        if _WHOLE_WORD_RE.fullmatch(out) is None:
            raise ComputeError(
                f"vacating {word!r} produced {out!r}, which is not a single complete word "
                "token — the token stream would no longer align with the original",
                {"word": word, "output": out},
            )
        return out


def _image_of(word: str, mapping: Mapping[str, str], keep: frozenset[str], seed: int) -> str:
    """A lowercased type at full vacancy, used only by the injectivity check."""
    stem, suffix = stem_and_suffix(word)
    if not is_eligible(stem, keep):
        return word
    nonce = mapping.get(stem.lower())
    if nonce is None:
        return word
    return surface_form(nonce, stem, suffix, seed)


def vacancy_domain(types: Iterable[str]) -> list[str]:
    """The map's domain: the corpus's types UNION the **full** Dolch list (contract §5.2).

    Always the full list, **never the active budget**. A budget word absent from the corpus
    still needs an image, or the mapped vocabulary of §7.2 has a hole in it — but if the
    domain tracked the *active* budget, switching budgets in the UI would rebuild the map and
    re-mint the corpus underneath a panel whose whole claim is that nonces are stable. Using
    the full list makes the map a function of `(corpus, seed, match_prosody)` alone.

    Since the domain is also the set a nonce may not equal (§5.2, no caller-supplied
    `avoid`), a SMALLER domain genuinely mints differently: measured, building over
    ``corpus ∪ dolch_budget(name)`` for any name below `full` moves exactly one stem, because
    `floor` is a full-list Dolch word that never appears in the corpus and so is forbidden in
    the full domain and free in the smaller ones. Unioning the full list here is what makes
    that unreachable, so there is only ever one map
    (`test_the_map_does_not_move_when_the_active_budget_changes`).

    Takes an iterable of TYPES, not a text: see :func:`_reject_bare_str`.
    """
    _reject_bare_str(types, "vacancy_domain", "types")
    return sorted({t.lower() for t in types} | {w.lower() for w in dolch_budget("full")})


def build_vacancy_map(
    types: Iterable[str],
    params: VacancyParams,
    counts: Mapping[str, int] | None = None,
) -> VacancyMap:
    """Assign every eligible stem a nonce, once, in canonical order (contract §5.2).

    `types` is the domain — pass :func:`vacancy_domain(corpus_types)`, which pins it to the
    corpus's types plus the full Dolch list.

    **The domain is avoided implicitly; there is no caller-supplied `avoid` parameter.** With
    one, the map depends on what the caller remembered to pass: measured, at seed 0 the same
    corpus and seed give ``remint_rounds`` 0 with the domain passed and 1 without, and
    different nonces either way. Both maps are valid — that is the problem, because one call
    site passing it and another not (the panel and the golden fixture, say) is a silent
    divergence with no failing test. Condition B below already forbids a surface form equal to
    any domain type, so avoiding the domain at mint time is not extra policy, only the cheaper
    route to the same fixed point. The map is now a pure function of
    ``(domain, seed, match_prosody)``.

    `p` is deliberately unused: the map is built over ALL eligible stems and restricted to
    ``{u < p}`` at rewrite time. That is what makes nesting and stability structural.

    **Injectivity is verified over assembled SURFACE FORMS, and the condition is
    `p`-independent** (§5.2). Both must hold over the domain:

    * **A.** the surface forms are pairwise distinct
    * **B.** no surface form equals any lowercased domain type, eligible or not

    A bare-nonce check is not enough — the collision arrives through the suffix — and a check
    performed at `p = 1` only is not enough either, because at full vacancy every eligible
    type has moved and nothing is left for a minted form to collide with. B is deliberately
    conservative: it forbids a minted form from equalling a word that would always have been
    vacated alongside it, and that costs a re-mint but buys a condition independent of `p`,
    which is what the theorem needs. Measured cost on the shipped corpus: one re-mint at
    seed 7, where `hang` first minted `wak` and `hanged` surfaced as the real word `waked`.

    On violation only the LOSING stem is re-minted — the one later in ASCII-ascending order
    among those involved — at salt ``1000 * round + previousSalt + 1``, so a re-mint never
    cascades (§5.8).

    **Under ``mint="swap"`` the replacement is a real English word and B cannot apply**, since
    the replacement is a domain type by construction. §5.2a works out what B was standing in
    for and what swap can therefore satisfy: A unchanged, and **B₁** — no surface form equals
    an INELIGIBLE domain type, and none equals its own source type. That makes the map a
    bijection of the domain at full vacancy, which is where the pretrained arm measures, and
    §5.2a proves no `p`-stable swap can do better. :attr:`VacancyMap.injective_at_every_p`
    records which regime the map is in and :func:`map_vocab_words` refuses the rest.

    `counts` is the corpus's per-type occurrence count (:func:`type_counts`), REQUIRED by
    ``mint="swap"`` and unused by ``mint="nonce"`` — the nonce map stays a pure function of
    ``(domain, seed, match_prosody)``, asserted in the tests. Swap raises without it rather
    than falling back to an alphabetical rank, which would be a frequency match in name only.
    """
    _reject_bare_str(types, "build_vacancy_map", "types")
    keep = params.keep_set
    type_set = {t.lower() for t in types}
    swapping = params.mint == "swap"
    if swapping and counts is None:
        raise InvalidParamError(
            "mint='swap' needs the corpus's type counts to rank the replacement pool by "
            "frequency (contract §8.3); pass counts=type_counts(tokenize(text))",
            {"mint": params.mint},
        )

    pairs: list[tuple[str, str]] = []
    stem_set: set[str] = set()
    families: dict[str, set[str]] = {}
    suffixes: dict[str, list[str]] = {}
    eligible_types: set[str] = set()
    for t in sorted(type_set):  # ASCII order, so `suffixes[stem]` is canonical
        stem, suffix = stem_and_suffix(t)
        if not is_eligible(stem, keep):
            continue
        pairs.append((stem.lower(), suffix.lower()))
        stem_set.add(stem.lower())
        families.setdefault(stem.lower(), set()).add(t)
        suffixes.setdefault(stem.lower(), []).append(suffix.lower())
        eligible_types.add(t)

    # Condition B's scope: EVERY domain type under `nonce`, and only the types that can never
    # be vacated under `swap` — §5.2a's B₁, which is what full-vacancy injectivity needs and
    # all a map drawing from the domain can possibly satisfy.
    barred = type_set - eligible_types if swapping else type_set

    mapping: dict[str, str] = {}
    minted_stress: dict[str, str] = {}
    salts: dict[str, int] = {}
    # `forbidden = used ∪ domain` — the domain, always, with nothing left to the caller. It
    # accumulates and is never pruned, so a superseded nonce stays out of circulation (§5.8),
    # and it is STORED on the map rather than reconstructed from `mapping.values()`.
    forbidden: set[str] = set(type_set)
    pool: list[str] = []
    rank: dict[str, int] = {}
    if swapping:
        assert counts is not None  # narrowed by the guard above
        pool = swap_pool(type_set, counts, keep)
        rank = {s: swap_rank(s, families[s], pool, counts) for s in sorted(stem_set)}
        used: set[str] = set()
        claimed_forms: set[str] = set()
        for stem in sorted(stem_set):
            word, salt, forms = _draw_swap(
                stem,
                params.seed,
                params.match_prosody,
                pool,
                rank,
                used,
                suffixes[stem],
                claimed_forms,
                barred,
            )
            used.add(word)
            claimed_forms.update(forms)
            forbidden.add(word)
            mapping[stem] = word
            salts[stem] = salt
    else:
        for stem in sorted(stem_set):
            nonce, pattern, salt = _mint(stem, params.seed, params.match_prosody, forbidden)
            forbidden.add(nonce)
            mapping[stem] = nonce
            minted_stress[nonce] = pattern
            salts[stem] = salt

    rounds = 0
    while True:
        claimed: dict[str, str] = {}  # surface -> the stem that owns it
        losers: set[str] = set()
        for stem, suffix in pairs:
            form = surface_form(mapping[stem], stem, suffix, params.seed)
            if form in barred or form == stem + suffix:  # condition B (B₁ under swap)
                losers.add(stem)
            if form in claimed:  # condition A
                losers.add(max(stem, claimed[form]))
            else:
                claimed[form] = stem
        if not losers:
            break
        if rounds >= MAX_REMINT_ROUNDS:
            raise ComputeError(
                f"vacancy map still collides after {MAX_REMINT_ROUNDS} re-mint rounds",
                {"stems": sorted(losers)[:20], "rounds": rounds, "mint": params.mint},
            )
        rounds += 1
        for stem in sorted(losers):
            others = {n for s, n in mapping.items() if s != stem}
            start_salt = REMINT_SALT_STRIDE * rounds + salts[stem] + 1
            if swapping:
                # A superseded REPLACEMENT returns to the pool — unlike a superseded nonce,
                # which stays forbidden forever. The pool is finite (1944 real words against
                # 1680 stems on the shipped corpus), so retiring words permanently would
                # starve later rounds; and a real word cannot "recreate the collision it was
                # rejected for" the way a nonce can, because the collision was with a
                # different stem's surface, which has itself moved.
                elsewhere = {
                    surface_form(mapping[s], s, suffix, params.seed)
                    for s, suffix in pairs
                    if s != stem
                }
                word, salt, _forms = _draw_swap(
                    stem,
                    params.seed,
                    params.match_prosody,
                    pool,
                    rank,
                    others,
                    suffixes[stem],
                    elsewhere,
                    barred,
                    start_salt=start_salt,
                )
                forbidden.add(word)
                mapping[stem] = word
                salts[stem] = salt
                continue
            nonce, pattern, salt = _mint(
                stem,
                params.seed,
                params.match_prosody,
                forbidden,
                start_salt=start_salt,
            )
            minted_stress.pop(mapping[stem], None)
            forbidden.add(nonce)
            mapping[stem] = nonce
            minted_stress[nonce] = pattern
            salts[stem] = salt

    seen = {_image_of(t, mapping, keep, params.seed) for t in type_set}
    return VacancyMap(
        mapping=mapping,
        minted_stress=minted_stress,
        seed=params.seed,
        match_prosody=params.match_prosody,
        domain=frozenset(type_set),
        image_size=len(seen),
        bijective=len(seen) == len(type_set),
        remint_rounds=rounds,
        forbidden=frozenset(forbidden),
        injective_at_every_p=not swapping,
    )


# --- rewriting --------------------------------------------------------------------------


def vacate_text(text: str, vmap: VacancyMap, params: VacancyParams) -> str:
    """Rewrite `text` in place, vacating every eligible stem with ``u(stem) < p``.

    Words are found with **exactly the tokenizer's regex** and everything else — whitespace,
    punctuation, digits, line breaks — passes through unchanged, byte for byte. Every output
    is itself a single complete `WORD_RE` match (checked in :meth:`VacancyMap._transform`),
    so ``tokenize(vacate(text))`` has the same length and ordering as ``tokenize(text)``, and
    because line breaks are untouched the ``<eos>``-per-line rule produces the same number of
    ``<eos>`` in the same places.
    """
    state = _RewriteState(set(vmap.forbidden))
    return WORD_RE.sub(lambda m: vmap._transform(m.group(0), params, state), text)


def map_vocab_words(words: Sequence[str], vmap: VacancyMap, params: VacancyParams) -> list[str]:
    """Push a budget's word list through the same transform, PRESERVING ORDER (§7.2).

    Since the map is injective, ``itos_p = SPECIALS ++ map_vocab_words(words, ...)`` assigns
    every word the id its pre-image had, which is why the token id stream is unchanged and
    training is bit-identical.

    This rule is only valid in the condition it is stated for. Under ``consistent=False`` or
    ``reveal_after > 0`` a source type no longer has a single image, so the budget must be
    REBUILT from the vacated corpus instead — the collapse in coverage is the measurement.
    Calling this there would quietly manufacture a vocabulary that matches no corpus.
    """
    _reject_bare_str(words, "map_vocab_words", "words")
    if not params.consistent or params.reveal_after:
        raise InvalidParamError(
            "the mapped vocabulary is only defined for consistent=True, reveal_after=0; "
            "every other condition rebuilds the budget from the vacated corpus",
            {"consistent": params.consistent, "reveal_after": params.reveal_after},
        )
    if not vmap.injective_at_every_p and 0.0 < params.p < 1.0:
        # §5.2a: swap's images ARE domain types, so at intermediate `p` a vacated type can
        # land on an un-vacated one and two budget words would share a row. That is not a
        # defect to be re-drawn away — the theorem there shows no `p`-stable swap avoids it —
        # so the mapped vocabulary is refused, exactly as it is for the two controls above.
        raise InvalidParamError(
            f"mint='swap' has no mapped vocabulary at p={params.p}: its replacements are "
            "domain types, so a vacated type can collide with an un-vacated one and the map "
            "is injective only at full vacancy (contract §5.2a). Use p=0 or p=1, or rebuild "
            "the budget from the vacated corpus",
            {"mint": params.mint, "p": params.p},
        )
    state = _RewriteState(set(vmap.forbidden))
    return [vmap._transform(w, params, state) for w in words]


# --- statistics -------------------------------------------------------------------------


@dataclass(frozen=True)
class _Prosody:
    """One side of the prosody statistics, token-weighted.

    ``from_table + from_minted + from_rule == 1`` by construction: :func:`stress_source`
    partitions the tokens.
    """

    mean_syllables: float
    mean_anapest: float
    from_table: float
    from_minted: float
    from_rule: float


def _prosody(text: str, minted_stress: Mapping[str, str] | None) -> _Prosody:
    words = WORD_RE.findall(text)
    if not words:
        return _Prosody(0.0, 0.0, 0.0, 0.0, 0.0)
    total_syllables = sum(syllables(w, minted_stress) for w in words)
    sources = [stress_source(w, minted_stress) for w in words]
    lines = [ln for ln in text.splitlines() if WORD_RE.search(ln)]
    anapest = (
        sum(meter_score(ln, "anapest", minted_stress) for ln in lines) / len(lines)
        if lines
        else 0.0
    )
    return _Prosody(
        mean_syllables=total_syllables / len(words),
        mean_anapest=anapest,
        from_table=sources.count("table") / len(words),
        from_minted=sources.count("minted") / len(words),
        from_rule=sources.count("rule") / len(words),
    )


def vacancy_stats(
    original: str, vacated: str, vmap: VacancyMap, params: VacancyParams
) -> dict[str, float | int | bool]:
    """The statistics contract (§10), with exactly these field names.

    **Every count names its scope; an unprefixed ``types*`` is forbidden.** "Types" is
    ambiguous between the corpus (2 211 types of *Mother Goose*) and the domain of §5.2
    (2 233 = corpus plus the full Dolch list), and the ambiguity cost two round trips between
    the two stacks — which agreed on ``tokensVacated`` to the token and disagreed only on
    what they were counting.

    * ``domainTypes*`` — over :attr:`VacancyMap.domain`. This governs the map and the
      vocabulary, so it is the diagnostic number, and it is a property of ``(map, p)``: the
      22 domain-only Dolch words (`funny`, `squirrel`, `today`, …) have images but never
      appear in the text, so they cannot be measured from it.
    * ``corpusTypes*`` — over the corpus's own type set. **This is what the panel shows a
      reader**: counting words the reader cannot see inflates the vacancy rate being shown.
      ``corpusTypesVacated`` and ``tokensVacated`` are MEASURED from the two texts, so they
      respect ``reveal_after`` and the inconsistent-assignment control — the source reports
      ``len(self.map)``, which is the size of the assignment and not what was vacated.
    * ``stemsTotal`` is ``|map|``; ``stemsVacated`` counts stems with ``u(stem) < p``.

    At ``p = 1`` every eligible stem vacates, because ``u ∈ [0, 1)`` by construction, so
    ``stemsVacated == stemsTotal`` and ``*TypesVacated == *TypesEligible``. The first of
    those identities is what exposed the scope confusion.

    The prosody numbers ship with a THREE-WAY split of where each token's stress came from,
    token-weighted and summing to 1 on each side (§10). A single "table coverage" number was
    ambiguous the moment minted forms existed — read literally it counts only the hand table,
    read as "stress we actually know" it also counts forms we minted, and the two readings
    differ by a factor of thirty. So:

    * ``stressFromTable*`` — the 61-entry hand table of §6.1, the honesty number for English
      words;
    * ``stressFromMinted*`` — forms we minted and registered a pattern for. Known by
      construction but ASSERTED rather than verified: §5.5 accepts a candidate on syllable
      COUNT, so the count is checked and the pattern is not;
    * ``stressFromRule*`` — the spelling heuristic of §6.2, i.e. a guess.

    The source's own numbers (mean anapest 0.351 -> 0.345, mean syllables 1.224 -> 1.211) are
    its numbers on a corpus we do not have. They are not transcribed anywhere.
    """
    before_words = WORD_RE.findall(original)
    after_words = WORD_RE.findall(vacated)
    if len(before_words) != len(after_words):
        raise ComputeError(
            f"vacating changed the token count ({len(before_words)} -> {len(after_words)}); "
            "the token streams no longer align",
            {"tokens_before": len(before_words), "tokens_after": len(after_words)},
        )

    keep = params.keep_set
    types = {w.lower() for w in before_words}
    eligible = {t for t in types if is_eligible(stem_and_suffix(t)[0], keep)}
    changed = [b.lower() for b, a in zip(before_words, after_words) if b.lower() != a.lower()]

    domain_eligible = {t for t in vmap.domain if is_eligible(stem_and_suffix(t)[0], keep)}
    domain_vacated = {
        t for t in domain_eligible if vacancy_u(stem_and_suffix(t)[0], params.seed) < params.p
    }
    stems_vacated = sum(1 for s in vmap.mapping if vacancy_u(s, params.seed) < params.p)

    # The original text is English, so it is scored WITHOUT the minted patterns; `avoid`
    # guarantees no English type is also a nonce, so `stressFromMintedBefore` is 0 by
    # construction rather than by omission.
    before = _prosody(original, None)
    after = _prosody(vacated, vmap.minted_stress)

    return {
        "domainTypesTotal": len(vmap.domain),
        "domainTypesEligible": len(domain_eligible),
        "domainTypesVacated": len(domain_vacated),
        "corpusTypesTotal": len(types),
        "corpusTypesEligible": len(eligible),
        "corpusTypesVacated": len(set(changed)),
        "stemsTotal": len(vmap.mapping),
        "stemsVacated": stems_vacated,
        "tokensTotal": len(before_words),
        "tokensVacated": len(changed),
        "meanSyllablesBefore": before.mean_syllables,
        "meanSyllablesAfter": after.mean_syllables,
        "meanAnapestBefore": before.mean_anapest,
        "meanAnapestAfter": after.mean_anapest,
        "stressFromTableBefore": before.from_table,
        "stressFromTableAfter": after.from_table,
        "stressFromMintedBefore": before.from_minted,
        "stressFromMintedAfter": after.from_minted,
        "stressFromRuleBefore": before.from_rule,
        "stressFromRuleAfter": after.from_rule,
        "bijective": vmap.bijective,
        "imageSize": vmap.image_size,
        "remintRounds": vmap.remint_rounds,
    }
