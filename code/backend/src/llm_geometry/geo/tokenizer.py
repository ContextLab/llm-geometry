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
from typing import Any

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

#: The one vocabulary format a model file may carry (``GeoTokenizer.to_json``).
_VOCAB_FORMAT = "geo-tokenizer-v1"

#: Special-token ids a vocabulary may declare, under either spelling. Mirrors
#: ``geoEngine/tokenizer.fromModelVocabJson`` — see :meth:`GeoTokenizer.from_json`.
_EXPECTED_SPECIALS = {
    UNK_TOKEN: UNK_ID,
    EOS_TOKEN: EOS_ID,
    PAD_TOKEN: PAD_ID,
    "unk": UNK_ID,
    "eos": EOS_ID,
    "pad": PAD_ID,
}


def _validate_specials(specials: Any) -> None:
    """Refuse a vocabulary whose declared special ids are not the ones we use.

    Ignoring the block (which is what "parse only `words`" amounts to) let a file say
    ``"<unk>": 5`` and load with 200 while every id it labels is read with ``<unk> = 0``.
    """
    if specials is None:
        return
    if not isinstance(specials, dict):
        raise InvalidParamError(
            f"vocabulary `specials` must be an object, got {type(specials).__name__}"
        )
    for token, expected in _EXPECTED_SPECIALS.items():
        declared = specials.get(token)
        if declared is not None and declared != expected:
            raise InvalidParamError(
                f"vocabulary: special {token} has id {declared!r}, expected {expected}"
            )


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
        """The CANONICAL vocabulary serialization — byte-identical in both builds.

        `vocab_sha256` is a digest of these exact bytes, so the spelling is part of the
        format, not a formatting preference. Python used ``", "``/``": "`` separators
        while the browser used ``JSON.stringify``'s compact ones with its own key
        order, and the same model therefore had two different digests depending on
        which build saved it. Pinned here (and in
        ``lib/geoEngine/tokenizer.canonicalVocabJson``) as: keys sorted, compact
        separators, ``ensure_ascii`` so the bytes do not depend on the encoding.
        """
        return json.dumps(
            {
                "format": "geo-tokenizer-v1",
                "specials": {UNK_TOKEN: UNK_ID, EOS_TOKEN: EOS_ID, PAD_TOKEN: PAD_ID},
                "words": self.words,
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, payload: str) -> "GeoTokenizer":
        """Parse the canonical serialization, refusing anything else with a TYPED error.

        This is the vocabulary half of loading a model file, and it used to be three
        unguarded lines: ``json.loads``, ``data.get``, ``list(data["words"])``. Seven
        malformed ``vocab`` blocks therefore reached the user as untyped HTTP 500s whose
        whole message was a Python exception string (``'list' object has no attribute
        'get'``, ``unhashable type: 'list'``, ``'words'``) — on a file the user chose, so
        the one thing the message had to say was which part of THEIR file was wrong.

        ``specials`` is validated rather than ignored, and the ``tokens``-shaped export is
        refused rather than crashing, because the browser engine
        (``geoEngine/tokenizer.fromModelVocabJson``) does exactly this: while Python
        ignored ``specials`` a file declaring ``<unk> = 5`` loaded here with 200 and was
        refused there, so the two stacks disagreed about whether a file was valid at all.
        """
        if not isinstance(payload, str):
            raise InvalidParamError(f"a vocabulary must be JSON text, got {type(payload).__name__}")
        try:
            data = json.loads(payload)
        except ValueError as exc:  # JSONDecodeError
            raise InvalidParamError(f"vocabulary is not valid JSON: {exc}")
        if not isinstance(data, dict):
            raise InvalidParamError(
                "vocabulary must be a JSON object with a `words` array, got "
                f"{type(data).__name__}"
            )
        if data.get("format") != _VOCAB_FORMAT:
            raise InvalidParamError(
                f"Unknown tokenizer format {data.get('format')!r} (expected " f"{_VOCAB_FORMAT!r})"
            )
        _validate_specials(data.get("specials"))
        words = data.get("words")
        if words is None and "tokens" in data:
            raise InvalidParamError(
                "vocabulary has `tokens` but no `words`: `tokens` is the static site's "
                "asset shape (specials included), while a model file carries the "
                f"{_VOCAB_FORMAT!r} `words` list of exactly {VOCAB_WORDS} entries"
            )
        if not isinstance(words, list) or any(not isinstance(w, str) for w in words):
            raise InvalidParamError("vocabulary `words` must be an array of strings")
        return cls(words)


@lru_cache(maxsize=1)
def get_tokenizer() -> GeoTokenizer:
    """The canonical tokenizer, built deterministically from the real corpus."""
    return GeoTokenizer.from_corpus_text(load_corpus_text())


def tokenizer_for(weights_token: str | None, store: Any = None) -> GeoTokenizer:
    """The tokenizer whose words give THIS model's token ids their meaning.

    Models trained from scratch on a user's own text carry their own vocabulary, so
    reading their ids with the canonical one would label every token wrongly — and so
    does anything DERIVED from such a model, because a fine-tune or a weight edit
    changes the numbers, not what the ids mean (`weights.inherited_vocab`). Only sets
    descended from the canonical checkpoint have no stored vocabulary, and those are
    the ones the canonical tokenizer is right for.

    A token this store does not have RAISES (``weights.weight_set_entry``); it does not
    fall back. The fallback was the same corruption reached through a different door: an
    evicted or unknown token answered 200 from ``GET /api/geo/tokenize`` with the user's
    ids read under Alice in Wonderland's word list, while ``GET /api/geo/trace`` answered
    404 for the identical request — so the two routes disagreed about whether the model
    existed, and the one that said yes was the one that was wrong.
    """
    if not weights_token or weights_token == "learned":
        return get_tokenizer()
    from .weights import load_weight_set_vocab

    payload = load_weight_set_vocab(weights_token, store=store)
    return GeoTokenizer.from_json(payload) if payload else get_tokenizer()
