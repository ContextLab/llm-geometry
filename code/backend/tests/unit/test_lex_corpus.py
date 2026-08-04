"""The shipped corpus is the ground every number in the Lexicon Lab stands on.

If these bytes change, every coverage figure, every loss curve and every generated line
changes with them — so the digest is checked rather than trusted, and the Gutenberg
licence text is asserted present because we redistribute the file whole.
"""

from __future__ import annotations

import pytest

from llm_geometry.lex.config import (
    CORPUS_BYTES,
    CORPUS_GUTENBERG_ID,
    CORPUS_PATH,
    CORPUS_SHA256,
)
from llm_geometry.lex.corpus import corpus_sha256, load_corpus_text, trim_gutenberg
from llm_geometry.lex.vocab import tokenize


def test_corpus_is_committed_and_unmodified():
    assert CORPUS_PATH.exists(), "the corpus ships with the package; it is not downloaded"
    assert CORPUS_PATH.stat().st_size == CORPUS_BYTES
    assert corpus_sha256() == CORPUS_SHA256


def test_gutenberg_licence_survives_in_the_committed_file():
    """We redistribute the file whole, which is how the PG licence is satisfied.

    Trimming happens at USE time, not on disk. If someone "cleans up" the file by
    stripping the header, the licence obligations change — so this fails first.
    """
    raw = CORPUS_PATH.read_text(encoding="utf-8")
    assert "PROJECT GUTENBERG" in raw.upper()
    assert f"{CORPUS_GUTENBERG_ID}" in raw or "Mother Goose" in raw


def test_trimming_removes_the_header_and_the_licence_footer():
    body = load_corpus_text()
    assert "*** START OF THE PROJECT GUTENBERG" not in body
    assert "*** END OF THE PROJECT GUTENBERG" not in body
    # The licence footer talks about the Foundation; the rhymes do not.
    assert "Literary Archive Foundation" not in body
    assert len(body) < CORPUS_BYTES  # trimming actually removed something


def test_body_is_the_rhymes():
    body = load_corpus_text()
    lowered = body.lower()
    # Spot-check lines that are unmistakably this book.
    assert "humpty dumpty" in lowered
    assert "the house that jack built" in lowered


def test_corpus_is_big_enough_to_train_on():
    toks = tokenize(load_corpus_text())
    assert len(toks) > 15_000, f"only {len(toks)} tokens — too little to train a model on"


def test_verification_refuses_altered_bytes(tmp_path):
    """A corpus that silently changed would move every number in the tab."""
    fake = tmp_path / "tampered.txt"
    fake.write_text("*** START OF THE PROJECT GUTENBERG ***\nhello\n", encoding="utf-8")
    with pytest.raises(ValueError, match="refusing to train on unverified text"):
        load_corpus_text(path=fake)
    # ...but an explicit opt-out still works, for user-supplied corpora.
    assert load_corpus_text(path=fake, verify=False).strip() == "hello"


def test_trim_is_a_no_op_without_markers():
    """User-supplied text has no Gutenberg markers and must pass through untouched."""
    assert trim_gutenberg("just some text") == "just some text"
