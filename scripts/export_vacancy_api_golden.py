#!/usr/bin/env python
"""Emit the vacancy **API** golden vectors — the endpoint/static-client parity fixture.

`scripts/export_vacancy_golden.py` pins the TRANSFORM across the two stacks (§11 of
`specs/007-vacancy-transform-field/architecture.md`). This script pins the layer above it:
what `POST /api/lex/vacancy` actually puts on the wire, so that

* `code/backend/tests/contract/test_api_lex.py` can assert the live FastAPI route still
  returns exactly this, and
* `code/frontend/tests/unit/staticVacancy.test.ts` can assert the browser's
  `staticClient.lexVacancy()` returns exactly this too.

Two tests, one file, and therefore one claim: **the full stack and the static build answer
the same request with the same numbers, the same vocabulary, and the same sha256 of the
whole vacated corpus.** FR-722 is that sentence; this fixture is what makes it checkable
rather than asserted. If either side drifts, one of the two tests fails and names the
field.

The route is exercised through FastAPI's `TestClient` against the REAL app with the REAL
committed corpus — no mocks, no stubbed transform, no hand-written expectations. Every
value in the output was produced by running the endpoint.

Usage (from the backend venv, at the repo root):

    python scripts/export_vacancy_api_golden.py
    python scripts/export_vacancy_api_golden.py --out /tmp/vacancy-api-golden.json

Determinism: the transform is a pure function of `(corpus, params)` and the route adds no
clock or randomness, so regenerating this twice must produce byte-identical bytes.
`--generated` is a date rather than a timestamp for exactly that reason.

WHAT THE CASES COVER, and why each is here rather than being one more of the same:

* `p1-seed7-pre_primer` — full vacancy at the seed that needs a re-mint (`remintRounds` 1,
  where `hang` first minted `wak` and `hanged` surfaced as the real word `waked`). Pins the
  DEFAULT `preview_chars`, so a drift in either stack's default fails here.
* `p035-seed0-full` — the source's own figure `p`, against the largest Dolch budget, so the
  mapped word list being pinned is 314 words long.
* `p0-seed0-pre_primer` — the identity boundary. `u ∈ [0, 1)`, so nothing vacates and the
  vacated digest must equal the original's. A transform that quietly did something at
  `p = 0` would show up nowhere else.
* `control-inconsistent` — `consistent = false`. The vocabulary is REBUILT rather than
  mapped (§7.2) and coverage collapses; that collapse is the measurement (FR-715), and it
  is the case where the two stacks previously minted different per-occurrence forms.
* `control-reveal-after-2` — `revealAfter = 2`. The other rebuilt condition, and the one
  where `corpusTypesVacated` is measured from the two TEXTS rather than from map
  membership (§10) — the definition the stacks split on.
* `p07-seed0-frequency100` — a frequency budget, whose word list is drawn from the corpus
  rather than from a fixed list, plus a non-default `preview_chars`.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code" / "backend" / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from llm_geometry.api.app import app  # noqa: E402
from llm_geometry.api.routes_lex import (  # noqa: E402
    VACANCY_PREVIEW_CHARS,
    VACANCY_PREVIEW_MAX,
)
from llm_geometry.lex.corpus import load_corpus_text  # noqa: E402

# Reuse the transform exporter's git-sha helper rather than writing a second one.
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from export_vacancy_golden import git_sha, sha256_text  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "code" / "frontend" / "tests" / "fixtures" / "vacancy-api-golden.json"

FORMAT = "vacancy-api-golden-v1"

#: Both stacks round every float in a response to 6 significant digits before it reaches a
#: caller (`api/encoding.py::jsonable_6sig` and `staticClient/lex.ts::sig6`), so the wire
#: values are IDENTICAL, not merely close. The tests compare with `toBe` / `==`; this bound
#: exists only so a future float that escapes the rounding has a documented allowance
#: instead of a silently loosened assertion.
TOLERANCE = 0.0

#: `(label, request body)`. The body is sent verbatim; the corpus is the shipped one in
#: every case, because the point of the fixture is parity on the text both stacks ship.
CASES: tuple[tuple[str, dict[str, Any]], ...] = (
    (
        "p1-seed7-pre_primer",
        {"p": 1.0, "seed": 7, "source": "dolch", "budget": "pre_primer"},
    ),
    (
        "p035-seed0-full",
        {"p": 0.35, "seed": 0, "source": "dolch", "budget": "full", "preview_chars": 400},
    ),
    (
        "p0-seed0-pre_primer",
        {"p": 0.0, "seed": 0, "source": "dolch", "budget": "pre_primer", "preview_chars": 400},
    ),
    (
        "control-inconsistent",
        {
            "p": 0.5,
            "seed": 0,
            "consistent": False,
            "source": "dolch",
            "budget": "primer",
        },
    ),
    (
        "control-reveal-after-2",
        {
            "p": 0.5,
            "seed": 0,
            "reveal_after": 2,
            "source": "dolch",
            "budget": "primer",
            "preview_chars": 400,
        },
    ),
    (
        "p07-seed0-frequency100",
        {
            "p": 0.7,
            "seed": 0,
            "match_prosody": False,
            "source": "frequency",
            "budget": "full",
            "size": 100,
            "preview_chars": 400,
        },
    ),
)


def build_document(generated: str) -> dict[str, Any]:
    client = TestClient(app)
    corpus = load_corpus_text()

    cases: list[dict[str, Any]] = []
    for label, body in CASES:
        response = client.post("/api/lex/vacancy", json=body)
        if response.status_code != 200:
            raise SystemExit(
                f"case {label!r}: the route returned {response.status_code}: {response.text}"
            )
        cases.append({"label": label, "request": body, "response": response.json()})

    return {
        "format": FORMAT,
        "generated": generated,
        "git_sha": git_sha(),
        "command": "python scripts/export_vacancy_api_golden.py",
        "source": (
            "POST /api/lex/vacancy on the real FastAPI app with the real committed "
            "corpus, through fastapi.testclient — real route, real transform, no mocks"
        ),
        "contract": "specs/002-interactive-model-explorer/contracts/api.md",
        "tolerance": TOLERANCE,
        "endpoint": "/api/lex/vacancy",
        "defaults": {
            "preview_chars": VACANCY_PREVIEW_CHARS,
            "preview_max": VACANCY_PREVIEW_MAX,
        },
        "encoding": (
            "exactly what the route serves: every float already rounded to 6 significant "
            "digits by api/encoding.py::jsonable_6sig, which staticClient/lex.ts::sig6 "
            "reproduces, so every field compares EXACTLY. `vacated_sha256` is over the "
            "UTF-8 bytes of the WHOLE vacated corpus — 86 kB pinned in 64 characters."
        ),
        "corpus": {
            "path": "code/backend/src/llm_geometry/lex/data/real-mother-goose.txt",
            "sha256": sha256_text(corpus),
            "chars": len(corpus),
        },
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--generated", default=date.today().isoformat())
    args = parser.parse_args()

    document = build_document(args.generated)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
    print(
        f"wrote {args.out} ({args.out.stat().st_size / 1024:.0f} KB, {len(document['cases'])} cases)"
    )


if __name__ == "__main__":
    main()
