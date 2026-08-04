"""Token→word alignment for the pretrained arm (contract §8.2, FR-718).

Real tokenizers, real text, no mocks — but no model weights, so this file is fast: it
downloads tokenizer files only. The alignment is the part of the measurement that can be
silently wrong (a mis-attributed token still produces a plausible number), so it is
tested harder than the arithmetic around it.
"""

from __future__ import annotations

import unicodedata

import pytest
from transformers import AutoTokenizer

from llm_geometry.arch.vacancy_score import (
    byte_decoder,
    default_passages,
    preserved_token_indices,
    preserved_word_indices,
    token_byte_spans,
    variant_texts,
    word_spans,
)
from llm_geometry.errors import ComputeError

# Every curated model is a byte-level BPE; gpt2 and SmolLM2 have no normalizer and Qwen
# has an NFC one, which is the case the up-front normalization exists for.
TOKENIZERS = ["gpt2", "HuggingFaceTB/SmolLM2-135M-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"]

TEXTS = [
    "The cow jumped over the moon, and the little dog laughed.",
    # Multi-byte characters that BPE splits ACROSS tokens — the case per-token decoding
    # corrupts and the case HF's own offsets overlap on.
    "café naïve — “owl” ≈ ç√ 東京 end",
    "  leading and\ttabbed\nnewlines  ",
    "don't good-bye o'clock",
]


@pytest.mark.parametrize("model_id", TOKENIZERS)
@pytest.mark.parametrize("text", TEXTS)
def test_byte_spans_reconstruct_the_text_exactly(model_id: str, text: str) -> None:
    tok = AutoTokenizer.from_pretrained(model_id)
    normalized = unicodedata.normalize("NFC", text)
    ids = tok(normalized, add_special_tokens=False)["input_ids"]
    pieces = tok.convert_ids_to_tokens(ids)
    spans = token_byte_spans(list(pieces), normalized)

    raw = normalized.encode("utf-8")
    assert len(spans) == len(ids)
    # A true partition: contiguous, covering, in order. That is what lets a per-token
    # quantity be summed over a word without double-counting.
    assert spans[0][0] == 0
    assert spans[-1][1] == len(raw)
    for (a, b), (c, _d) in zip(spans, spans[1:]):
        assert a <= b == c
    assert b"".join(raw[a:b] for a, b in spans) == raw


def test_byte_spans_raise_when_the_pieces_do_not_rebuild_the_text() -> None:
    """A mismatch must RAISE, never mis-attribute (FR-718)."""
    tok = AutoTokenizer.from_pretrained("gpt2")
    text = "The cow jumped over the moon."
    pieces = list(tok.convert_ids_to_tokens(tok(text, add_special_tokens=False)["input_ids"]))
    with pytest.raises(ComputeError, match="alignment failed"):
        token_byte_spans(pieces[:-1], text)


def test_decomposed_input_is_caught_rather_than_silently_shifted() -> None:
    """NFD input rebuilds to different bytes — the check fires instead of drifting."""
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    decomposed = unicodedata.normalize("NFD", "café swirl")
    assert decomposed != unicodedata.normalize("NFC", decomposed)
    ids = tok(decomposed, add_special_tokens=False)["input_ids"]
    pieces = list(tok.convert_ids_to_tokens(ids))
    with pytest.raises(ComputeError):
        token_byte_spans(pieces, decomposed)
    # …and NFC first (what `vacancy_score` does) makes it exact.
    nfc = unicodedata.normalize("NFC", decomposed)
    nfc_ids = tok(nfc, add_special_tokens=False)["input_ids"]
    spans = token_byte_spans(list(tok.convert_ids_to_tokens(nfc_ids)), nfc)
    assert spans[-1][1] == len(nfc.encode("utf-8"))


def test_byte_decoder_is_the_inverse_of_bytes_to_unicode() -> None:
    table = byte_decoder()
    assert len(table) == 256
    assert sorted(table.values()) == list(range(256))
    # 0x20 is NOT printable in this scheme, so it is rendered as U+0120 ("Ġ") — which is
    # exactly why a piece cannot be read as text and needs this table to be measured.
    assert table["Ġ"] == 0x20
    assert " " not in table
    assert table["Ċ"] == 0x0A


def test_leading_space_tokens_are_attributed_to_their_word() -> None:
    """The overlap rule, not "starts inside": byte-level BPE folds the space in."""
    tok = AutoTokenizer.from_pretrained("gpt2")
    text = "the cow and the moon"
    ids = tok(text, add_special_tokens=False)["input_ids"]
    spans = token_byte_spans(list(tok.convert_ids_to_tokens(ids)), text)
    words = word_spans(text)
    assert [w.word for w in words] == ["the", "cow", "and", "the", "moon"]
    # "the", "and", "the" preserved; "cow", "moon" vacated.
    preserved = frozenset({0, 2, 3})
    got = preserved_token_indices(spans, words, preserved)
    assert got, "no token was attributed to a preserved word"
    decoded = "".join(tok.decode([ids[i]]) for i in got)
    assert decoded.replace(" ", "") == "theandthe"


def test_a_token_spanning_a_preserved_and_a_vacated_word_raises() -> None:
    """Ambiguous attribution is refused, not resolved by a heuristic."""
    words = word_spans("the cow")
    # One synthetic token covering both words: the failure mode the assertion guards.
    with pytest.raises(ComputeError, match="spans both"):
        preserved_token_indices([(0, 7)], words, frozenset({0}))


def test_variants_preserve_word_count_and_the_scaffolding() -> None:
    passage = default_passages(count=1)[0]
    texts = variant_texts(passage, p=1.0, seed=0, match_prosody=True, keep=frozenset())
    assert set(texts) == {"english", "swap", "nonce"}
    words, preserved = preserved_word_indices(texts)
    assert preserved, "full vacancy left no scaffolding at all"
    # Preserved means character-identical in EVERY variant — the property §8.1 rests on.
    for name in ("swap", "nonce"):
        variant = word_spans(texts[name])
        assert len(variant) == len(words)
        for i in preserved:
            assert variant[i].word == words[i].word
    # …and the vacated words really did move, in both variants and differently.
    vacated = [i for i in range(len(words)) if i not in preserved]
    assert vacated
    swap_words = word_spans(texts["swap"])
    nonce_words = word_spans(texts["nonce"])
    assert any(swap_words[i].word != words[i].word for i in vacated)
    assert any(nonce_words[i].word != swap_words[i].word for i in vacated)


def test_default_passages_are_deterministic_and_sized() -> None:
    a = default_passages()
    b = default_passages()
    assert a == b
    assert len(a) == 6
    assert len({t for t in a}) == 6, "the default set repeats a passage"
    for text in a:
        assert len(word_spans(text)) >= 200
