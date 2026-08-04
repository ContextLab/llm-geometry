"""The vocabulary budget is this feature's independent variable, so its properties are
pinned here: the corrected Dolch data, budget nesting, and the coverage arithmetic the UI
displays.

Real data throughout — the shipped corpus, the real word lists. No fixtures, no mocks.
"""

from __future__ import annotations

import pytest

from llm_geometry.lex.config import PAD_ID, SPECIAL_TOKENS, UNK_ID, param_count
from llm_geometry.lex.corpus import load_corpus_text
from llm_geometry.lex.dolch import DOLCH_ORDER, FIRST, NOUNS, dolch_budget, dolch_sizes
from llm_geometry.lex.vocab import build_vocab, frequency_budget, tokenize


@pytest.fixture(scope="module")
def corpus() -> str:
    return load_corpus_text()


# --- the corrections ------------------------------------------------------------------


def test_first_grade_has_going_not_giving():
    """The source transcribed `going` as `giving`.

    Both words are named explicitly so that re-introducing the slip fails loudly rather
    than silently shrinking coverage — `going` occurs 27 times in the shipped corpus.
    """
    assert "going" in FIRST
    assert "giving" not in FIRST
    # `give` is genuinely on the published list too; the fix must not have replaced it.
    assert "give" in FIRST


def test_no_multiword_entries_anywhere():
    """`Santa Claus` cannot be matched by a word tokenizer, so it is not in the budget.

    Shipping it made the source's "315" budget silently 314 words wide.
    """
    for name in DOLCH_ORDER:
        assert not [w for w in dolch_budget(name) if " " in w]
    assert not [w for w in NOUNS if " " in w]


def test_budget_sizes_are_measured_not_asserted():
    """These five numbers appear in the UI; they must come from the data."""
    assert dolch_sizes() == {
        "pre_primer": 40,
        "primer": 92,
        "first": 133,
        "service": 220,
        "full": 314,
    }


def test_budgets_nest():
    """|V| is only a clean independent variable if growing it merely ADDS words."""
    for smaller, larger in zip(DOLCH_ORDER, DOLCH_ORDER[1:]):
        assert set(dolch_budget(smaller)) <= set(dolch_budget(larger))


def test_green_eggs_vocabulary_is_not_shipped():
    """The source's MASK_50 is a 1960 picture book's vocabulary, mislabelled.

    Spot-check words that are distinctive to it and absent from any real Dolch list.
    """
    every_dolch_word = set(dolch_budget("full"))
    for w in ("ham", "eggs", "sam", "goat", "anywhere", "mouse", "fox", "boat"):
        if w in ("boat",):  # `boat` IS a real Dolch noun — the others are not
            continue
        assert w not in every_dolch_word, f"{w!r} leaked in from the picture-book mask"


# --- tokenization ---------------------------------------------------------------------


def test_tokenizer_keeps_internal_punctuation():
    """`don't` and `good-bye` are real Dolch entries and must survive as single tokens."""
    assert tokenize("Don't say good-bye!") == ["don't", "say", "good-bye"]
    assert "don't" in dolch_budget("full")
    assert "good-bye" in dolch_budget("full")


def test_tokenizer_drops_punctuation_entirely():
    """A budget of N words should spend all N rows on words, not on commas."""
    assert tokenize("a, b. c; d!") == ["a", "b", "c", "d"]


# --- vocabulary assembly --------------------------------------------------------------


def test_specials_lead_and_unk_is_zero(corpus):
    v = build_vocab("dolch", "full", corpus)
    assert v.itos[: len(SPECIAL_TOKENS)] == SPECIAL_TOKENS
    assert v.stoi["<unk>"] == UNK_ID == 0
    assert v.stoi["<pad>"] == PAD_ID
    # |V| is the WORD count; rows is what the embedding matrix actually has.
    assert v.budget_size == 314
    assert v.rows == 314 + len(SPECIAL_TOKENS)


def test_out_of_budget_encodes_to_unk(corpus):
    v = build_vocab("dolch", "pre_primer", corpus)
    ids = v.encode(["the", "supercalifragilistic", "cat"])
    assert ids[0] == v.stoi["the"]
    assert ids[1] == UNK_ID
    # `cat` is a Dolch NOUN, so it is out of budget at pre-primer.
    assert ids[2] == UNK_ID


def test_frequency_budget_is_deterministic_and_alphabetical_on_ties():
    text = "b b a a c"  # a and b tie at 2; c has 1
    assert frequency_budget(text, 2) == ["a", "b"]
    assert frequency_budget(text, 3) == ["a", "b", "c"]


def test_frequency_budget_beats_dolch_on_its_own_corpus(corpus):
    """The descriptive budget should cover its own corpus better at matched |V|.

    This is US-3's whole point, so it is asserted rather than assumed. If it ever
    reverses, the comparison the tab invites is no longer the one it describes.
    """
    for budget in DOLCH_ORDER:
        d = build_vocab("dolch", budget, corpus)
        f = build_vocab("frequency", budget, corpus)
        assert d.budget_size == f.budget_size, "the comparison must be at matched |V|"
        assert f.coverage(corpus).token_coverage > d.coverage(corpus).token_coverage


def test_coverage_arithmetic_is_self_consistent(corpus):
    v = build_vocab("dolch", "full", corpus)
    c = v.coverage(corpus)
    assert c.in_budget_tokens <= c.total_tokens
    assert c.token_coverage + c.unk_rate == pytest.approx(1.0)
    assert 0 <= c.whole_lines_in_budget <= c.total_lines
    assert c.oov_types <= c.distinct_types


def test_coverage_rises_monotonically_with_budget(corpus):
    """Nesting guarantees this; a regression here means the budgets stopped nesting."""
    seen = 0.0
    for budget in DOLCH_ORDER:
        cov = build_vocab("dolch", budget, corpus).coverage(corpus).token_coverage
        assert cov >= seen
        seen = cov


# --- the parameter formula ------------------------------------------------------------


@pytest.mark.parametrize(
    "rows,d,layers,ctx,tied,expected",
    [
        # Verified against the source implementation (006-source-model-arch.md).
        (318, 64, 2, 64, True, 124_544),
        (318, 128, 4, 128, False, 891_136),
    ],
)
def test_param_count_matches_the_verified_formula(rows, d, layers, ctx, tied, expected):
    assert param_count(rows, d, layers, ctx, tied) == expected


def test_param_count_at_the_real_defaults():
    """Pinned separately from the formula cases so it TRACKS the defaults.

    `DEFAULT_CTX` moved 64 → 32 when browser throughput was fixed, which changed this
    number. A test that only exercised hand-written shapes would not have noticed, and the
    prose quoting "the default model" would have quietly gone stale.
    """
    from llm_geometry.lex.config import (
        DEFAULT_CTX,
        DEFAULT_D_MODEL,
        DEFAULT_N_LAYERS,
        DEFAULT_TIED,
    )

    rows = dolch_sizes()["full"] + len(SPECIAL_TOKENS)
    n = param_count(rows, DEFAULT_D_MODEL, DEFAULT_N_LAYERS, DEFAULT_CTX, DEFAULT_TIED)
    assert n == 122_496, (
        f"the default configuration now has {n:,} parameters, not 122,496 — update every "
        "sentence that quotes it (specs/006-lexicon-lab-tiny/architecture.md, the UI)"
    )


def test_untied_adds_exactly_one_embedding_matrix():
    tied = param_count(318, 64, 2, 64, True)
    untied = param_count(318, 64, 2, 64, False)
    assert untied - tied == 318 * 64
