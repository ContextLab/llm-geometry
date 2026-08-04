"""The Dolch sight-word lists (Dolch, 1936) — the *prescribed* vocabulary budget.

Edward William Dolch published these lists in 1936; they are a factual word list of
long-standing pedagogical use and are not an expressive work.

This is a CORRECTED transcription of the lists in the `tiny-models` source project. Two
changes, both verified rather than assumed (`notes/agent-reports/006-source-vocabulary.md`):

1. First grade had ``giving``. The published list has ``going`` — it already contains
   ``give`` separately, so ``giving`` was a transcription slip, not a variant. Checked
   against the published list; ``test_dolch.py`` pins both the presence of ``going`` and
   the absence of ``giving`` so the slip cannot come back.

2. ``Santa Claus`` is dropped from the usable budget. It contains a space, so a
   word-level tokenizer can never match it as one token — in the source it silently made
   the "315" budget a 314-word budget. We drop it explicitly and report the real count
   rather than shipping a number that is off by one.

NOT ported: the source's ``MASK_50``. Its docstring calls it "fifty of the commonest
words in English", but it is verbatim the vocabulary of a well-known 1960 picture book
(it contains `eggs`, `ham`, `Sam`, `goat`, `fox`, `box`, `mouse`, `train`, `anywhere` —
not common English words). Shipping it under that description would be repeating a false
claim, and shipping it accurately would raise a provenance question we have no reason to
take on. The graded Dolch lists below carry the same idea with clean 1936 provenance.

Counts here are MEASURED by `test_dolch.py`, never hard-coded prose: published sources
disagree about the noun list's length (one widely-cited table gives 95, another 97, and
its own stated 220 service-word total is inconsistent with its per-grade rows). We
therefore state what this file actually contains.
"""

from __future__ import annotations

# --- the graded service words --------------------------------------------------------

PRE_PRIMER = """a and away big blue can come down find for funny go help here I in is
it jump little look make me my not one play red run said see the three to two up we
where yellow you""".split()

PRIMER = """all am are at ate be black brown but came did do eat four get good have he
into like must new no now on our out please pretty ran ride saw say she so soon that
there they this too under want was well went what white who will with yes""".split()

# `going`, NOT `giving` — see the module docstring.
FIRST = """after again an any as ask by could every fly from give going had has her him
his how just know let live may of old once open over put round some stop take thank
them then think walk were when""".split()

SECOND = """always around because been before best both buy call cold does don't fast
first five found gave goes green its made many off or pull read right sing sit sleep
tell their these those upon us use very wash which why wish work would write
your""".split()

THIRD = """about better bring carry clean cut done draw drink eight fall far full got
grow hold hot hurt if keep kind laugh light long much myself never only own pick seven
shall show six small start ten today together try warm""".split()

# --- the nouns -----------------------------------------------------------------------
# `Santa Claus` is deliberately absent: see the module docstring.

NOUNS = """apple baby back ball bear bed bell bird birthday boat box boy bread brother
cake car cat chair chicken children Christmas coat corn cow day dog doll door duck egg
eye farm farmer father feet fire fish floor flower game garden girl good-bye grass
ground hand head hill home horse house kitty leg letter man men milk money morning
mother name nest night paper party picture pig rabbit rain ring robin school seed sheep
shoe sister snow song squirrel stick street sun table thing time top toy tree watch
water way wind window wood""".split()

# --- cumulative budgets --------------------------------------------------------------
# Each budget NESTS in the next, which is what makes |V| a clean independent variable:
# growing the budget only ever ADDS words, so a comparison across budgets is not
# confounded by words leaving.

_SERVICE = PRE_PRIMER + PRIMER + FIRST + SECOND + THIRD

DOLCH_BUDGETS: dict[str, list[str]] = {
    "pre_primer": PRE_PRIMER,
    "primer": PRE_PRIMER + PRIMER,
    "first": PRE_PRIMER + PRIMER + FIRST,
    "service": _SERVICE,
    "full": _SERVICE + NOUNS,
}

#: Display labels, in ascending order of size. The number is filled in at runtime from
#: the real length so a label can never disagree with the list it names.
DOLCH_ORDER = ["pre_primer", "primer", "first", "service", "full"]


def dolch_budget(name: str) -> list[str]:
    """The lower-cased, de-duplicated word list for a named budget, in a stable order."""
    if name not in DOLCH_BUDGETS:
        raise KeyError(f"unknown Dolch budget {name!r}; expected one of {DOLCH_ORDER}")
    seen: set[str] = set()
    out: list[str] = []
    for w in DOLCH_BUDGETS[name]:
        lw = w.lower()
        if lw not in seen:
            seen.add(lw)
            out.append(lw)
    return out


def dolch_sizes() -> dict[str, int]:
    """Measured size of every budget. The UI reads this instead of quoting literals."""
    return {name: len(dolch_budget(name)) for name in DOLCH_ORDER}
