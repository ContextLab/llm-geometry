"""Regenerate the cross-stack canonical-vocabulary fixture from the REAL Python serializer.

    python scripts/export_geo_canonical_vocab.py

Writes `code/frontend/tests/fixtures/geo-canonical-vocab.json`: the exact bytes
`GeoTokenizer.to_json` produces for a word list containing every character that has ever
made the two stacks' serializers disagree, plus its sha256.

WHY. `vocab_sha256` is a digest of those exact bytes, and since a model's identity covers
its vocabulary the digest is part of the model id. Python's `ensure_ascii` keeps
`\\x20`-`\\x7e` and escapes everything else — DEL (U+007F) included — while
`canonicalVocabJson` (`src/lib/geoEngine/tokenizer.ts`) tested `code < 0x80` and left DEL
raw. The same word list therefore hashed to two different values, so a model file saved by
the full stack was refused by the static build as corrupt, and vice versa. The geo token
regex `[^\\sa-z0-9]` admits any single non-space symbol, so DEL really can reach a word
list.

Both suites assert against this one file — `tests/unit/test_geo_canonical_vocab.py` here and
`tests/unit/geoCanonicalVocab.test.ts` in the browser stack — so the fixture is a transcript
of the real serializer rather than a string somebody typed, and neither stack can drift
alone.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "backend" / "src"))

from llm_geometry.geo.tokenizer import VOCAB_WORDS, GeoTokenizer, get_tokenizer  # noqa: E402

#: The characters that decide the two serializers' agreement, in the order they broke it.
PROBE_WORDS = [
    "\x7f",  # DEL: escaped by ensure_ascii, left raw by `code < 0x80` — the divergence
    "a\x7fb",  # …and not only alone: mid-word, where a word list really carries it
    "~",  # the last character ensure_ascii keeps — the other side of the boundary
    "é",  # plainly non-ASCII: escaped by both stacks before and after the fix
    "’",  # the curly apostrophe every editor emits
    "\U0001f600",  # astral: two surrogate escapes in both stacks
]

OUT = ROOT / "code" / "frontend" / "tests" / "fixtures" / "geo-canonical-vocab.json"


def main() -> None:
    base = [w for w in get_tokenizer().words if w not in PROBE_WORDS]
    words = base[: VOCAB_WORDS - len(PROBE_WORDS)] + PROBE_WORDS
    if len(words) != VOCAB_WORDS or len(set(words)) != VOCAB_WORDS:
        raise SystemExit("probe words collided with the shipped vocabulary")

    canonical = GeoTokenizer(words).to_json()
    fixture = {
        "format": "geo-canonical-vocab-v1",
        "contract": "code/backend/src/llm_geometry/geo/tokenizer.py::GeoTokenizer.to_json",
        "command": "python scripts/export_geo_canonical_vocab.py",
        "note": __doc__.split("WHY.", 1)[1].strip().replace("\n", " "),
        "probeWords": PROBE_WORDS,
        "words": words,
        "canonicalJson": canonical,
        "vocabSha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }
    OUT.write_text(json.dumps(fixture, indent=1, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}  sha256 {fixture['vocabSha256']}")


if __name__ == "__main__":
    main()
