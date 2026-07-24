"""Deterministic word-level tokenizer for the Geometry Lab.

Lowercases, normalizes typographic punctuation (curly quotes → ASCII), and splits
into word / number / single-punctuation tokens. The vocabulary is the top-1000 token
types of the training corpus by frequency (ties broken alphabetically, so building is
fully deterministic), preceded by the three specials ``<unk>=0 <eos>=1 <pad>=2``.
Punctuation marks count as "word types" here — they are among the most frequent
types and an LM over the corpus is meaningless without them.

``encode`` marks out-of-vocab positions (``unk``) and truncates at CONTEXT_WINDOW per
the frozen `/api/geo/tokenize` contract; ``truncate_side`` chooses which end survives
("right" keeps the first 50 tokens — the tokenize-preview behavior; "left" keeps the
last tokens — what LM conditioning wants). ``to_json``/``from_json`` round-trip the
exact vocabulary.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

from ..errors import InvalidParamError
from .config import (
    CONTEXT_WINDOW,
    EOS_ID,
    EOS_TOKEN,
    PAD_ID,
    PAD_TOKEN,
    UNK_ID,
    UNK_TOKEN,
    VOCAB_SIZE,
    VOCAB_WORDS,
)
from .corpus import load_corpus_text

# Typographic → ASCII normalization applied before splitting.
_NORMALIZE = str.maketrans(
    {
        "’": "'",  # ’
        "‘": "'",  # ‘
        "“": '"',  # “
        "”": '"',  # ”
        " ": " ",  # nbsp
    }
)

# Words (with internal apostrophes: "alice's", "won't"), numbers, or any single
# non-space symbol (punctuation, dashes, underscores, …).
_TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)*|[0-9]+|[^\sa-z0-9]")

_SPECIAL_TEXTS = {UNK_ID: UNK_TOKEN, EOS_ID: EOS_TOKEN, PAD_ID: PAD_TOKEN}

# Tokens that should not be preceded by a space when detokenizing.
_NO_SPACE_BEFORE = set(".,;:!?)]}'…") | {"''"}
_NO_SPACE_AFTER = set("([{")


def split_words(text: str) -> list[str]:
    """Deterministically split ``text`` into lowercase word/punctuation tokens."""
    return _TOKEN_RE.findall(text.lower().translate(_NORMALIZE))


@dataclass(frozen=True)
class EncodedText:
    ids: list[int]
    texts: list[str]
    unk: list[bool]
    n_unk: int
    truncated: bool

    def tokens(self) -> list[dict]:
        """Contract-shaped token list: [{"id", "text", "unk"}, …]."""
        return [{"id": i, "text": t, "unk": u} for i, t, u in zip(self.ids, self.texts, self.unk)]


class GeoTokenizer:
    """Fixed-vocabulary word-level tokenizer (deterministic, JSON-serializable)."""

    def __init__(self, words: list[str]) -> None:
        if len(words) != VOCAB_WORDS:
            raise InvalidParamError(
                f"GeoTokenizer requires exactly {VOCAB_WORDS} words, got {len(words)}"
            )
        if len(set(words)) != len(words):
            raise InvalidParamError("GeoTokenizer vocabulary contains duplicates")
        self.words = list(words)
        self.id_to_text: dict[int, str] = dict(_SPECIAL_TEXTS)
        for offset, word in enumerate(self.words):
            self.id_to_text[len(_SPECIAL_TEXTS) + offset] = word
        self.text_to_id: dict[str, int] = {
            text: tid for tid, text in self.id_to_text.items() if tid not in _SPECIAL_TEXTS
        }
        assert len(self.id_to_text) == VOCAB_SIZE

    # -- construction ----------------------------------------------------------------

    @classmethod
    def from_corpus_text(cls, text: str) -> "GeoTokenizer":
        counts = Counter(split_words(text))
        # Deterministic order: by descending frequency, then alphabetically.
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        words = [w for w, _ in ranked[:VOCAB_WORDS]]
        if len(words) < VOCAB_WORDS:
            raise InvalidParamError(
                f"Corpus has only {len(words)} distinct token types; " f"{VOCAB_WORDS} are required"
            )
        return cls(words)

    # -- encode / decode -------------------------------------------------------------

    def encode(
        self,
        text: str,
        *,
        truncate: bool = True,
        truncate_side: str = "right",
        max_tokens: int = CONTEXT_WINDOW,
    ) -> EncodedText:
        pieces = split_words(text)
        truncated = False
        if truncate and len(pieces) > max_tokens:
            truncated = True
            pieces = pieces[:max_tokens] if truncate_side == "right" else pieces[-max_tokens:]
        ids: list[int] = []
        unk: list[bool] = []
        for piece in pieces:
            tid = self.text_to_id.get(piece)
            if tid is None:
                ids.append(UNK_ID)
                unk.append(True)
            else:
                ids.append(tid)
                unk.append(False)
        return EncodedText(ids=ids, texts=pieces, unk=unk, n_unk=sum(unk), truncated=truncated)

    def encode_stream(self, text: str) -> list[int]:
        """Encode without truncation (training/fine-tuning token streams)."""
        return self.encode(text, truncate=False).ids

    def decode(self, ids: list[int]) -> str:
        out: list[str] = []
        for tid in ids:
            text = self.id_to_text.get(int(tid))
            if text is None:
                raise InvalidParamError(f"Token id {tid} is out of range (0..{VOCAB_SIZE - 1})")
            if not out:
                out.append(text)
            elif text in _NO_SPACE_BEFORE or out[-1] in _NO_SPACE_AFTER:
                out.append(text)
            else:
                out.append(" " + text)
        return "".join(out)

    # -- serialization ---------------------------------------------------------------

    def to_json(self) -> str:
        return json.dumps(
            {
                "format": "geo-tokenizer-v1",
                "specials": {UNK_TOKEN: UNK_ID, EOS_TOKEN: EOS_ID, PAD_TOKEN: PAD_ID},
                "words": self.words,
            },
            ensure_ascii=True,
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, payload: str) -> "GeoTokenizer":
        data = json.loads(payload)
        if data.get("format") != "geo-tokenizer-v1":
            raise InvalidParamError(f"Unknown tokenizer format {data.get('format')!r}")
        return cls(list(data["words"]))


@lru_cache(maxsize=1)
def get_tokenizer() -> GeoTokenizer:
    """The canonical tokenizer, built deterministically from the real corpus."""
    return GeoTokenizer.from_corpus_text(load_corpus_text())
