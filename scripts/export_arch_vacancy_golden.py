"""Pin the pretrained arm's default passage set across both stacks (contract §8.3a).

The six excerpts the measurement was made on are cut from the shipped corpus by
`llm_geometry.arch.vacancy_score.default_passages`, and the browser cuts its own from
`static-data/lex/corpus.json` with `defaultVacancyPassages` in
`src/lib/staticClient/arch.ts`. Those two are the same measurement or they are not the
same measurement; this writes the digests that decide it.

Only digests and counts are written — the corpus text itself is already committed, and
duplicating 9 kB of it into a fixture would just be a second copy to keep in sync.

    python scripts/export_arch_vacancy_golden.py

Regenerate whenever `default_passages` or the corpus changes; `tests/unit/archVacancy.
test.ts` fails until you do, which is the point.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "backend" / "src"))

from llm_geometry.arch.vacancy_score import (  # noqa: E402
    DEFAULT_PASSAGE_COUNT,
    DEFAULT_PASSAGE_WORDS,
    default_passages,
)
from llm_geometry.lex.corpus import corpus_sha256  # noqa: E402
from llm_geometry.lex.vocab import WORD_RE  # noqa: E402

OUT = ROOT / "code" / "frontend" / "tests" / "fixtures" / "arch-vacancy-passages.json"


def main() -> None:
    passages = default_passages()
    payload = {
        "note": (
            "Digests of the default passage set of contract §8.3a. Written by "
            "scripts/export_arch_vacancy_golden.py from the real corpus; asserted "
            "against the browser's own cut in tests/unit/archVacancy.test.ts."
        ),
        "corpus_sha256": corpus_sha256(),
        "count": DEFAULT_PASSAGE_COUNT,
        "words_per_passage": DEFAULT_PASSAGE_WORDS,
        "passages": [
            {
                "index": i,
                "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "n_words": len(WORD_RE.findall(text)),
                "n_chars": len(text),
                "head": re.sub(r"\s+", " ", text[:60]).strip(),
            }
            for i, text in enumerate(passages)
        ],
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    for row in payload["passages"]:
        print(f"  #{row['index']} {row['n_words']} words  {row['sha256'][:12]}…  {row['head']!r}")


if __name__ == "__main__":
    main()
