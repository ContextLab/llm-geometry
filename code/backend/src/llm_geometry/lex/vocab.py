"""Vocabulary budgets — the independent variable of this tab.

Two budget *sources*, deliberately offered at matched sizes:

* ``dolch``     — the real 1936 pedagogical lists. A budget somebody PRESCRIBED.
* ``frequency`` — the top-N types of the corpus you are actually training on. A budget
                  the corpus DESCRIBES.

Comparing them at the same |V| is the point: same number of words, different words,
measurably different coverage and geometry.

Out-of-budget tokens become ``<unk>`` during training and are BANNED at generation, so a
model can only ever speak in-budget. The `<unk>` rate is reported rather than hidden — it
is the measurable form of what a budget cannot say.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from .config import (
    BUDGET_SOURCES,
    SPECIAL_TOKENS,
    UNK_ID,
)
from .dolch import DOLCH_ORDER, dolch_budget

#: Words are letters plus internal apostrophes/hyphens, so `don't` and `good-bye` — both
#: real Dolch entries — survive tokenization as single tokens. Punctuation is dropped
#: rather than tokenized: a budget of N *words* should spend all N rows on words.
WORD_RE = re.compile(r"[A-Za-z]+(?:['\-][A-Za-z]+)*")


def tokenize(text: str) -> list[str]:
    """Lower-cased word tokens. The only tokenizer in this feature."""
    return [m.group(0).lower() for m in WORD_RE.finditer(text)]


@dataclass(frozen=True)
class Coverage:
    """What a budget can and cannot express about a specific corpus."""

    total_tokens: int
    in_budget_tokens: int
    distinct_types: int
    oov_types: int
    total_lines: int
    whole_lines_in_budget: int

    @property
    def token_coverage(self) -> float:
        return self.in_budget_tokens / self.total_tokens if self.total_tokens else 0.0

    @property
    def unk_rate(self) -> float:
        return 1.0 - self.token_coverage

    def as_dict(self) -> dict[str, float | int]:
        return {
            "total_tokens": self.total_tokens,
            "in_budget_tokens": self.in_budget_tokens,
            "distinct_types": self.distinct_types,
            "oov_types": self.oov_types,
            "total_lines": self.total_lines,
            "whole_lines_in_budget": self.whole_lines_in_budget,
            "token_coverage": self.token_coverage,
            "unk_rate": self.unk_rate,
        }


@dataclass(frozen=True)
class LexVocab:
    """A budget, resolved into ids.

    Row layout is ``SPECIAL_TOKENS`` first, then the budget words in a stable order, so
    ``UNK_ID..PAD_ID`` mean the same thing in every model and a saved model's ids stay
    meaningful.
    """

    words: tuple[str, ...]  # budget words only, WITHOUT the specials
    source: str  # "dolch" | "frequency"
    budget_name: str  # e.g. "full", or "top314" for frequency budgets

    @property
    def itos(self) -> tuple[str, ...]:
        return SPECIAL_TOKENS + self.words

    @property
    def stoi(self) -> dict[str, int]:
        return {w: i for i, w in enumerate(self.itos)}

    @property
    def budget_size(self) -> int:
        """|V| as the user set it — the number of real WORDS."""
        return len(self.words)

    @property
    def rows(self) -> int:
        """Embedding rows, which is |V| plus the specials. Displayed separately."""
        return len(self.itos)

    def encode(self, tokens: Iterable[str]) -> list[int]:
        s = self.stoi
        return [s.get(t, UNK_ID) for t in tokens]

    def decode(self, ids: Iterable[int]) -> list[str]:
        n = len(self.itos)
        return [self.itos[i] if 0 <= i < n else "<unk>" for i in ids]

    def coverage(self, text: str) -> Coverage:
        """Measure this budget against a corpus. Used to populate the UI's counters."""
        vocab = set(self.words)
        toks = tokenize(text)
        types = set(toks)
        lines = [ln for ln in text.splitlines() if WORD_RE.search(ln)]
        whole = sum(1 for ln in lines if all(t in vocab for t in tokenize(ln)))
        return Coverage(
            total_tokens=len(toks),
            in_budget_tokens=sum(1 for t in toks if t in vocab),
            distinct_types=len(types),
            oov_types=len(types - vocab),
            total_lines=len(lines),
            whole_lines_in_budget=whole,
        )


def frequency_budget(text: str, size: int) -> list[str]:
    """The `size` most frequent types of `text`.

    Ties are broken alphabetically so the budget is a deterministic function of the
    corpus — the same guarantee the Geometry Lab's tokenizer makes.
    """
    from collections import Counter

    counts = Counter(tokenize(text))
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [w for w, _ in ordered[:size]]


def build_vocab(source: str, budget: str, corpus_text: str, size: int | None = None) -> LexVocab:
    """Resolve a (source, budget) pair against a corpus into a concrete vocabulary.

    For ``dolch`` the budget name selects a graded list and `size` is ignored — the list
    IS the budget. For ``frequency`` the budget is the top-`size` types of this corpus;
    when `size` is omitted it defaults to the matching Dolch budget's size, which is what
    makes the two comparable.
    """
    if source not in BUDGET_SOURCES:
        raise ValueError(f"unknown budget source {source!r}; expected {BUDGET_SOURCES}")

    if source == "dolch":
        if budget not in DOLCH_ORDER:
            raise ValueError(f"unknown Dolch budget {budget!r}; expected {DOLCH_ORDER}")
        return LexVocab(tuple(dolch_budget(budget)), source="dolch", budget_name=budget)

    n = size if size is not None else len(dolch_budget(budget if budget in DOLCH_ORDER else "full"))
    words = frequency_budget(corpus_text, n)
    return LexVocab(tuple(words), source="frequency", budget_name=f"top{n}")
