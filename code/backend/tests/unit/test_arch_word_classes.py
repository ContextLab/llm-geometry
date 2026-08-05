"""The word-alphabet class, and the Unicode-version skew that used to split the stacks.

Three things are pinned here, and each of them is a defect that shipped:

1. **The class, not the examples.** ``Pc`` (connector punctuation) was missing from both
   stacks, so ``don‿t`` scored HTTP 200 and swapped to ``warm‿t`` — character for character
   the ``don’t`` → ``big’t`` defect the joiner class had been introduced to close. The tests
   below walk **every** code point of the pinned joiner table rather than a list of the
   characters someone happened to try.

2. **The skew.** ``wordlike_runs`` used to ask :mod:`unicodedata`, which on the Python this
   project pins is Unicode 13.0, while the browser asked its own regex engine, which is
   Unicode 16.0. The two disagreed about 9 993 letters and marks and 11 joiners; U+0890 is
   genuinely ``Cf`` and the backend scored it while the browser refused it. Both stacks now
   read one committed table, so the assertions here are about the TABLE being the authority —
   a revert to :mod:`unicodedata` fails them on this interpreter.

3. **The false refusal the joiner fix introduced.** ``legs--upon`` — the Gutenberg em-dash
   convention, present in this project's own corpus — is written entirely in ``WORD_RE``'s
   own alphabet, so refusing it told the reader to do the thing they had already done.
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import pytest

from llm_geometry.arch.vacancy_score import (
    WORD_RE_JOINERS,
    check_word_alphabet,
    fragmented_words,
    wordlike_runs,
)
from llm_geometry.arch.word_classes import is_joiner, is_letter, is_mark, word_classes
from llm_geometry.errors import InvalidParamError

#: Repo root: tests/unit/<this file> -> tests -> backend -> code -> root.
REPO_ROOT = Path(__file__).resolve().parents[4]
SPEC_DIR = REPO_ROOT / "specs" / "007-vacancy-transform-field"
NORMATIVE_TABLE = SPEC_DIR / "word-classes.json"
PACKAGE_TABLE = (
    REPO_ROOT / "code" / "backend" / "src" / "llm_geometry" / "arch" / "data" / "word-classes.json"
)
CASES_FILE = SPEC_DIR / "word-alphabet-cases.json"


def _parse(spec: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for part in spec.split(","):
        if not part:
            continue
        lo, _, hi = part.partition("-")
        out.append((int(lo, 16), int(hi, 16) if hi else int(lo, 16)))
    return out


def _members(name: str) -> list[int]:
    cls = json.loads(NORMATIVE_TABLE.read_text(encoding="utf-8"))["classes"][name]
    cps = [cp for lo, hi in _parse(cls["ranges"]) for cp in range(lo, hi + 1)]
    cps += [int(h, 16) for h in cls.get("named", {})]
    return sorted(set(cps))


def test_the_backend_table_is_byte_identical_to_the_normative_one() -> None:
    """The whole point of the pin: one enumeration, copied, never re-derived per stack.

    The browser's copy is checked against the same file by
    `tests/unit/wordClasses.test.ts`. If either drifts, the two stacks are back to agreeing
    by coincidence and a passage one of them scores is one the other refuses.
    """
    assert (
        PACKAGE_TABLE.read_bytes() == NORMATIVE_TABLE.read_bytes()
    ), "regenerate both with `node scripts/export_word_classes.mjs`"
    table = word_classes()
    assert table["command"] == "node scripts/export_word_classes.mjs"
    assert table["classes"]["joiner"]["categories"] == ["Pd", "Cf", "Pc"]


def test_the_pinned_table_and_not_unicodedata_answers_the_question() -> None:
    """A revert to `unicodedata.category` fails here on the interpreter CI pins.

    Not a hypothetical: Python 3.10 is Unicode 13.0 and Node 22 is Unicode 16.0, so the two
    stacks classified 11 joiners differently — U+0890 and U+0891 among them, which really are
    `Cf`. The backend saw two words where the browser saw one, scored, and rewrote a fragment.
    """

    def version(v: str) -> tuple[int, ...]:
        return tuple(int(p) for p in v.split("."))

    if version(unicodedata.unidata_version) >= version(word_classes()["unicodeVersion"]):
        pytest.skip(
            f"this interpreter is Unicode {unicodedata.unidata_version}, at or past the pin — "
            "the divergence this test demonstrates needs an older table than the pinned one"
        )
    # The NAMED joiners carry no joiner category anywhere, so they are not evidence about
    # versions; nor are the two WORD_RE accepts, which are never fragments.
    named = {
        int(h, 16)
        for h in json.loads(NORMATIVE_TABLE.read_text(encoding="utf-8"))["classes"]["joiner"].get(
            "named", {}
        )
    }
    newer = [
        cp
        for cp in _members("joiner")
        if cp not in named and unicodedata.category(chr(cp)) not in ("Pd", "Cf", "Pc")
    ]
    assert newer, "expected this interpreter's Unicode to be older than the pin"
    for cp in newer:
        assert is_joiner(chr(cp)), f"U+{cp:04X} is in the pinned table and must be a joiner here"
        assert fragmented_words(f"don{chr(cp)}t") == [f"don{chr(cp)}t"]


def test_every_joiner_in_the_pinned_class_binds_two_letters() -> None:
    """The CLASS, walked exhaustively — 207 code points, not the ones that were reported."""
    joiners = _members("joiner")
    assert len(joiners) > 200
    missed = [cp for cp in joiners if not is_joiner(chr(cp))]
    assert missed == []
    # ASCII `'` and `-` are joiners too, but WORD_RE accepts them, so `don'`+j+`t` is a
    # single match rather than a fragment. Every OTHER joiner must be refused.
    unflagged = [
        cp
        for cp in joiners
        if chr(cp) not in WORD_RE_JOINERS
        and fragmented_words(f"don{chr(cp)}t") != [f"don{chr(cp)}t"]
    ]
    assert unflagged == [], "joiners the refusal misses: " + ", ".join(
        f"U+{cp:04X}" for cp in unflagged
    )


def test_connector_punctuation_is_in_the_class() -> None:
    """`Pc`, named on its own because its absence was a live wrong answer.

    `don‿t` answered HTTP 200 and swapped to `warm‿t` under the round-3 class.
    """
    for ch in "_‿⁀⁔︳︴﹍﹎﹏＿":
        assert unicodedata.category(ch) == "Pc"
        assert is_joiner(ch), f"U+{ord(ch):04X} is Pc and must be a joiner"
        assert wordlike_runs(f"don{ch}t") == [f"don{ch}t"]
        with pytest.raises(InvalidParamError, match="word alphabet"):
            check_word_alphabet(f"the cat don{ch}t sit")


def test_a_run_written_in_word_res_own_alphabet_is_not_refused() -> None:
    """The false refusal the joiner fix introduced.

    `--` is the Gutenberg em-dash convention and this project's own corpus carries
    `ba--are`, `hea--art`, `Lady--loves`, `legs--upon`. The refusal told the reader to "use a
    passage written in the ASCII alphabet, with straight apostrophes and hyphens", which is
    what they had done — there was no way to comply.
    """
    for text in ("legs--upon", "ba--are", "hea--art", "Lady--loves", "don''t", "a---b"):
        assert fragmented_words(text) == [], text
        check_word_alphabet(text)
    # …and the escape hatch is ASCII-only: a repeated INVISIBLE joiner still refuses.
    assert fragmented_words("co­­operate") == ["co­­operate"]
    assert fragmented_words("don’’t") == ["don’’t"]
    # …and it is about the alphabet, not about length: one curly apostrophe still refuses.
    assert fragmented_words("don’t") == ["don’t"]


def test_letters_and_marks_come_from_the_pinned_table_too() -> None:
    """The other 9 993 characters of the skew.

    A letter this runtime does not know is not a letter to `wordlike_runs`, so
    `a<letter>b` is two runs here and one in the browser — the same split, on the other
    half of the grammar. Sampled across the table because walking 141 028 letters per test
    run is not worth the seconds.
    """
    letters = _members("letter")
    marks = _members("mark")
    assert len(letters) > 140_000 and len(marks) > 2_400
    for cps, predicate in ((letters, is_letter), (marks, is_mark)):
        for cp in cps[:: max(1, len(cps) // 400)]:
            assert predicate(chr(cp)), f"U+{cp:04X}"
    assert not is_letter("-") and not is_mark("-")
    assert not is_letter(" ") and not is_joiner(" ")


def test_the_shared_case_table_holds_here_exactly() -> None:
    """The same table `tests/unit/wordClasses.test.ts` runs, so the two stacks are compared
    on identical inputs and identical expected outputs rather than on two prose lists."""
    cases = json.loads(CASES_FILE.read_text(encoding="utf-8"))
    assert cases["format"] == "word-alphabet-cases-v1"
    assert len(cases["cases"]) >= 25
    for case in cases["cases"]:
        assert (
            fragmented_words(case["text"]) == case["fragmented"]
        ), f"{case['text']!r}: {case['why']}"
