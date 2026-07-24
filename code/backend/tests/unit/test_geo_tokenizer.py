"""Unit tests for the Geometry Lab tokenizer (real corpus, no mocks — FR-109)."""

from __future__ import annotations

import pytest

from llm_geometry.errors import InvalidParamError
from llm_geometry.geo.config import (
    CONTEXT_WINDOW,
    EOS_ID,
    PAD_ID,
    UNK_ID,
    VOCAB_SIZE,
    VOCAB_WORDS,
)
from llm_geometry.geo.corpus import load_corpus_text
from llm_geometry.geo.tokenizer import GeoTokenizer, get_tokenizer, split_words

SAMPLE = (
    "Alice was beginning to get very tired of sitting by her sister on the bank, "
    "and of having nothing to do."
)


def test_vocab_size_and_specials():
    tok = get_tokenizer()
    assert len(tok.words) == VOCAB_WORDS
    assert len(tok.id_to_text) == VOCAB_SIZE
    assert tok.id_to_text[UNK_ID] == "<unk>"
    assert tok.id_to_text[EOS_ID] == "<eos>"
    assert tok.id_to_text[PAD_ID] == "<pad>"
    # Specials never collide with corpus words.
    assert all(w not in ("<unk>", "<eos>", "<pad>") for w in tok.words)


def test_determinism_two_independent_builds():
    text = load_corpus_text()
    tok_a = GeoTokenizer.from_corpus_text(text)
    tok_b = GeoTokenizer.from_corpus_text(text)
    assert tok_a.words == tok_b.words
    enc_a, enc_b = tok_a.encode(SAMPLE), tok_b.encode(SAMPLE)
    assert enc_a.ids == enc_b.ids
    assert enc_a.texts == enc_b.texts


def test_lowercasing_and_typographic_normalization():
    tok = get_tokenizer()
    assert tok.encode("ALICE Alice alice").ids == [tok.text_to_id["alice"]] * 3
    # Curly apostrophe normalizes to ASCII: “Alice’s” tokenizes like "alice's".
    curly, ascii_ = tok.encode("Alice’s"), tok.encode("Alice's")
    assert curly.ids == ascii_.ids


def test_unk_marking():
    tok = get_tokenizer()
    enc = tok.encode("alice met a supercalifragilistic gryphon")
    assert enc.texts[3] == "supercalifragilistic"
    assert enc.unk[3] is True
    assert enc.ids[3] == UNK_ID
    assert enc.n_unk == sum(enc.unk) >= 1
    assert enc.unk[0] is False  # "alice" is certainly in the top-1000
    tokens = enc.tokens()
    assert tokens[3] == {"id": UNK_ID, "text": "supercalifragilistic", "unk": True}


def test_truncation():
    tok = get_tokenizer()
    words = split_words(load_corpus_text())[: CONTEXT_WINDOW * 2]
    long_text = " ".join(words)
    enc = tok.encode(long_text)
    assert enc.truncated is True
    assert len(enc.ids) == CONTEXT_WINDOW
    assert enc.texts == words[:CONTEXT_WINDOW]  # default keeps the first tokens
    left = tok.encode(long_text, truncate_side="left")
    assert left.texts == words[-CONTEXT_WINDOW:]  # LM conditioning keeps the last
    short = tok.encode("alice was beginning")
    assert short.truncated is False
    stream = tok.encode_stream(long_text)
    assert len(stream) == len(words)  # encode_stream never truncates


def test_json_round_trip():
    tok = get_tokenizer()
    clone = GeoTokenizer.from_json(tok.to_json())
    assert clone.words == tok.words
    assert clone.encode(SAMPLE).ids == tok.encode(SAMPLE).ids
    assert clone.to_json() == tok.to_json()


def test_decode_and_reencode_idempotence():
    tok = get_tokenizer()
    enc = tok.encode(SAMPLE)
    text = tok.decode(enc.ids)
    assert "alice" in text and "," in text
    assert tok.encode(text).ids == enc.ids  # decode → encode is a fixed point
    with pytest.raises(InvalidParamError):
        tok.decode([VOCAB_SIZE])  # out of range
