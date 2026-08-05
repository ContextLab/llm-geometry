#!/usr/bin/env python
"""Re-measure the float32 vacancy reference and write the record both stacks are pinned to.

The static build cannot run a real model at build time, so the three float32 figures it
quotes in `VACANCY_FP32_REFERENCE` (`code/frontend/src/lib/staticClient/arch.ts`) are
literals in TypeScript. Literals rot: three times in this campaign a sentence went on
asserting a number the transform no longer produced, and nothing failed. The chain that
stops it now is

    real gpt2 run  ->  specs/007-vacancy-transform-field/fp32-reference.json  ->  the TS constant

with `test_the_fp32_arm_quoted_in_the_static_client`
(`code/backend/tests/integration/test_arch_vacancy_score.py`) checking the first arrow
against a live run of the real model, and `tests/unit/archVacancy.test.ts` checking the
second. This script writes the middle link, from the same live run.

Run it whenever the transform legitimately moves, and commit the JSON, the TypeScript
constant and the sentences that quote them in ONE commit:

    code/backend/.venv/bin/python scripts/measure_vacancy_fp32.py

Six passages x three variants = 18 real forward passes of ~300 tokens on gpt2/CPU.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "specs" / "007-vacancy-transform-field" / "fp32-reference.json"


def main() -> int:
    from llm_geometry.arch.vacancy_score import default_passages, vacancy_score

    result = vacancy_score("gpt2", default_passages(), p=1.0, seed=0)
    diffs = {d["id"]: d for d in result["differences"]}
    record = {
        "format": "vacancy-fp32-reference-v1",
        "note": (
            "A RECORDING of one real run, written by scripts/measure_vacancy_fp32.py. "
            "Both stacks are pinned to it: the backend integration test re-runs the real "
            "model and asserts these numbers, and the frontend unit test asserts "
            "VACANCY_FP32_REFERENCE (staticClient/arch.ts) rounds to them. Editing this "
            "file without re-running the model makes the backend test fail against gpt2."
        ),
        "model": result["model_id"],
        "revision": result["revision"],
        "dtype": result["dtype"],
        "params": {"passages": "default", "p": result["p"], "seed": result["seed"]},
        "tokens": {v["id"]: v["pooled"]["nTokens"] for v in result["variants"]},
        "preserved": {v["id"]: v["pooled"]["nPreservedTokens"] for v in result["variants"]},
        "differences": {
            name: {
                "nats": diffs[name]["nats"],
                "se": diffs[name]["se"],
                "nPairs": diffs[name]["nPairs"],
            }
            for name in ("wrong_content", "unknown_form", "total")
        },
    }
    RECORD.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
