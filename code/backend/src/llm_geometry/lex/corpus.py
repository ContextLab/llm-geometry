"""The shipped training corpus.

Project Gutenberg #10607, "The Real Mother Goose" (1916) — public domain in the USA,
committed whole so no run depends on a download. The PG header and licence footer stay
in the file (satisfying the licence as-is); they are trimmed when the text is USED, the
same way the Geometry Lab handles Alice.

Chosen over the alternatives because it is the only large candidate with zero
`[Illustration]` markers, zero ASCII tables and zero transcriber notes, and because
nursery rhymes repeat whole lines: 9.1% of its non-blank lines are exact duplicates,
against 0.8% for Alice. That line-level repetition is the property a tiny model can
actually learn from.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from .config import (
    CORPUS_PATH,
    CORPUS_SHA256,
    GUTENBERG_END_MARKER,
    GUTENBERG_START_MARKER,
)


def corpus_sha256(path: Path = CORPUS_PATH) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_corpus_text(path: Path = CORPUS_PATH, verify: bool = True) -> str:
    """The book's body, with the Gutenberg header and licence footer removed.

    `verify` re-checks the committed bytes against the recorded digest. A corpus that
    silently changed would move every number in the tab, so this fails loudly rather
    than training on something unexpected.
    """
    raw = path.read_text(encoding="utf-8")
    if verify:
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != CORPUS_SHA256:
            raise ValueError(
                f"corpus at {path} has digest {actual}, expected {CORPUS_SHA256} — "
                "refusing to train on unverified text"
            )
    return trim_gutenberg(raw)


def trim_gutenberg(raw: str) -> str:
    """Drop everything up to the START marker's line and from the END marker on."""
    start = 0
    end = len(raw)
    lines = raw.splitlines(keepends=True)
    pos = 0
    for line in lines:
        if GUTENBERG_START_MARKER in line:
            start = pos + len(line)
        elif GUTENBERG_END_MARKER in line:
            end = pos
            break
        pos += len(line)
    return raw[start:end].strip("\n")
