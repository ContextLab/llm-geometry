"""Real corpus acquisition for the Geometry Lab.

Alice's Adventures in Wonderland (Project Gutenberg ebook #11). The raw file lives at
``data/raw/alice-in-wonderland.txt`` and is committed to the repo (public domain), so
normal runs and CI never touch the network. If the file is somehow absent it is
downloaded once from Project Gutenberg and integrity-checked against the recorded
sha256 — a mismatch is a hard error, never silently accepted (Constitution I).
"""

from __future__ import annotations

import hashlib
import tempfile
import urllib.request
from pathlib import Path

from ..errors import ComputeError
from .config import (
    CORPUS_PATH,
    CORPUS_SHA256,
    CORPUS_URLS,
    GUTENBERG_END_MARKER,
    GUTENBERG_START_MARKER,
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_corpus(path: Path = CORPUS_PATH) -> Path:
    """Return the path to the verified raw corpus, downloading it once if absent."""
    if path.exists():
        digest = _sha256(path)
        if digest != CORPUS_SHA256:
            raise ComputeError(
                f"Corpus file {path} failed integrity check "
                f"(sha256 {digest} != expected {CORPUS_SHA256}). "
                "Delete the file to re-download, or restore it from git."
            )
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for url in CORPUS_URLS:
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                data = resp.read()
        except Exception as exc:  # noqa: BLE001 — every URL failure is reported below
            last_error = exc
            continue
        digest = hashlib.sha256(data).hexdigest()
        if digest != CORPUS_SHA256:
            last_error = ComputeError(
                f"Downloaded corpus from {url} has sha256 {digest}, " f"expected {CORPUS_SHA256}"
            )
            continue
        # Atomic write so a partial download can never be mistaken for the corpus.
        fd_path = tempfile.NamedTemporaryFile(dir=str(path.parent), delete=False, suffix=".tmp")
        try:
            fd_path.write(data)
            fd_path.flush()
        finally:
            fd_path.close()
        Path(fd_path.name).replace(path)
        return path
    raise ComputeError(
        f"Could not obtain the corpus (tried {', '.join(CORPUS_URLS)}): {last_error}"
    )


def load_corpus_text(path: Path = CORPUS_PATH) -> str:
    """Return the corpus body with the Project Gutenberg header/footer stripped."""
    raw = ensure_corpus(path).read_text(encoding="utf-8")
    lines = raw.splitlines()
    start = 0
    end = len(lines)
    for i, line in enumerate(lines):
        if GUTENBERG_START_MARKER in line:
            start = i + 1
        elif GUTENBERG_END_MARKER in line:
            end = i
            break
    body = "\n".join(lines[start:end]).strip()
    if not body:
        raise ComputeError(f"Corpus at {path} is empty after stripping Gutenberg markers")
    return body
