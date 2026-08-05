"""The PINNED Unicode word classes both stacks classify code points from.

``wordlike_runs`` (this package) and ``WORDLIKE_RE``
(``code/frontend/src/lib/staticClient/byteSpans.ts``) implement one grammar over three
classes — letter, combining mark, joiner. Each used to read those classes out of its own
runtime: Python via :mod:`unicodedata`, JavaScript via ``\\p{L}``/``\\p{M}``/``\\p{Pd}``.

That is agreement by coincidence, and the coincidence does not hold. Python 3.10 — the
version CI pins — carries Unicode 13.0; Node 22 carries Unicode 16.0. Measured across the
whole code space, the two disagree about **9 993 letters and marks and 11 joiners**, every
one of them a character the newer table knows and the older does not. Two of the eleven are
U+0890 and U+0891, which are genuinely ``Cf`` — the backend's own declared joiner category —
so ``don<U+0890>t`` was a wordlike run in the browser (refused) and two separate words in
Python (scored, and rewritten as ``warm<U+0890>t``). A silent wrong answer produced by
nothing worse than the two stacks being built at different times.

So neither stack asks its runtime any more. ``arch/data/word-classes.json`` is a committed
enumeration of the three classes at one pinned Unicode version, the browser carries a
byte-identical copy at ``src/lib/staticClient/wordClasses.json``, and both are compared to
the normative copy in ``specs/007-vacancy-transform-field/`` by a test in each suite.
Regenerate all three together with ``node scripts/export_word_classes.mjs``; moving the pin
is then one deliberate commit that moves both stacks at once.

The table is not an optimization and must not be replaced by a category lookup that "means
the same thing" — it is the only reason the two stacks mean the same thing.
"""

from __future__ import annotations

import json
from bisect import bisect_right
from functools import lru_cache
from importlib.resources import files
from typing import Any

#: The committed table, as loaded. Package data, so a pip install carries it.
_TABLE_RESOURCE = "word-classes.json"


@lru_cache(maxsize=1)
def word_classes() -> dict[str, Any]:
    """The pinned table, parsed once."""
    raw = (files("llm_geometry.arch") / "data" / _TABLE_RESOURCE).read_text(encoding="utf-8")
    table = json.loads(raw)
    if table.get("format") != "word-classes-v1":
        raise ValueError(f"arch/data/{_TABLE_RESOURCE}: unknown format {table.get('format')!r}")
    return table


def _parse_ranges(spec: str) -> tuple[list[int], list[int]]:
    """``"41-5a,61"`` -> two parallel sorted lists of inclusive bounds.

    Split into starts/ends so membership is a single :func:`bisect_right` rather than a scan
    of 677 ranges per character; ``wordlike_runs`` calls this per code point of a passage.
    """
    starts: list[int] = []
    ends: list[int] = []
    for part in spec.split(","):
        if not part:
            continue
        lo, _, hi = part.partition("-")
        a = int(lo, 16)
        b = int(hi, 16) if hi else a
        if a > b or (starts and a <= ends[-1]):
            raise ValueError(f"word-classes.json: ranges must ascend and not overlap, got {part!r}")
        starts.append(a)
        ends.append(b)
    return starts, ends


@lru_cache(maxsize=8)
def _class_ranges(name: str) -> tuple[list[int], list[int]]:
    cls = word_classes()["classes"][name]
    starts, ends = _parse_ranges(cls["ranges"])
    for cp_hex in cls.get("named", {}):
        cp = int(cp_hex, 16)
        i = bisect_right(starts, cp) - 1
        if i < 0 or cp > ends[i]:
            starts.append(cp)
            ends.append(cp)
    order = sorted(range(len(starts)), key=lambda i: starts[i])
    return [starts[i] for i in order], [ends[i] for i in order]


def _in_class(name: str, ch: str) -> bool:
    starts, ends = _class_ranges(name)
    cp = ord(ch)
    i = bisect_right(starts, cp) - 1
    return i >= 0 and cp <= ends[i]


def is_letter(ch: str) -> bool:
    """A letter (Unicode general category ``L*``) at the pinned version."""
    return _in_class("letter", ch)


def is_mark(ch: str) -> bool:
    """A combining mark (``M*``) at the pinned version — part of the letter it sits on."""
    return _in_class("mark", ch)


def is_joiner(ch: str) -> bool:
    """A character that BINDS two letters into one written word.

    ``Pd`` (dashes of every width) ∪ ``Cf`` (invisible format characters: soft hyphen, ZWSP,
    ZWNJ, ZWJ, the bidi marks, U+2060, U+FEFF) ∪ ``Pc`` (connector punctuation: ``_``,
    U+203F undertie, U+2040, U+2054, the fullwidth and presentation forms) ∪ the named
    apostrophes and word-internal points, which carry no property separating them from
    ordinary quotation marks.

    ``Pc`` was missing from both stacks until 2026-08-04: ``don<U+203F>t`` scored 200 and
    swapped to ``warm<U+203F>t``, character for character the ``don’t`` → ``big’t`` defect
    the class was introduced to close.
    """
    return _in_class("joiner", ch)
