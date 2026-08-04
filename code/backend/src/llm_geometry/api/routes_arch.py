"""Architecture Explorer routes (`/api/arch/*`).

Implements the frozen contract in specs/002-interactive-model-explorer/contracts/api.md.
Owned by Batch-2 agent B2; Batch 0 registers the (empty) router so parallel work
touches disjoint files.

Every endpoint runs the pre-download size gate first (FR-107): an oversized model gets
a 422 ``ModelTooLargeError`` envelope before any weights could start downloading.
Payloads follow the contract-wide array encoding — plain nested JSON lists with floats
rounded to 6 significant digits, numpy scalars coerced to Python — via one helper.
Errors surface only as the typed-error envelope (no fallbacks, Constitution I).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from ..arch import build_graph, check_model_size, generate, trace_forward, weight_window
from ..arch.vacancy_score import vacancy_score
from ..config import ARCH_DEFAULT_MAX_CONTEXT, ARCH_WEIGHTS_MAX_CELLS
from ..errors import InvalidParamError
from .encoding import jsonable_6sig

router = APIRouter(prefix="/arch", tags=["arch"])


# Contract-wide array encoding — shared helper (red-team round 2 NIT-6).
_sig6 = jsonable_6sig


@lru_cache(maxsize=64)
def _gate(model_id: str) -> None:
    """Size-gate a model BEFORE anything could download (FR-107).

    Successful passes are memoized per process (the gate consults hub metadata, so
    repeat requests stay fast); failures always re-raise because ``lru_cache`` never
    caches exceptions — a rejected model stays rejected.
    """
    check_model_size(model_id)


class GenerateBody(BaseModel):
    """`POST /api/arch/generate` request body, per the frozen contract.

    Bounds (``max_new_tokens <= 128``, ``temperature >= 0``, non-empty prompt) are
    enforced by ``arch.generate`` so violations return the typed 400 envelope rather
    than a framework validation response.
    """

    model_id: str
    prompt: str = ""
    system_prompt: str | None = None
    temperature: float = 0.8
    max_new_tokens: int = 64
    seed: int | None = None


class VacancyScoreBody(BaseModel):
    """`POST /api/arch/vacancy-score` request body (feature 007, contract §8).

    `passages` defaults to the six evenly spaced excerpts of the shipped corpus the
    measurement of §8.3a used, so an empty request reproduces the reference configuration.
    `passage` is the singular sugar the panel uses when a reader edits one excerpt.

    The transform's other knobs are deliberately absent: `consistent` and `reveal_after`
    are fixed at the theorem's condition, because the number this endpoint produces is
    only interpretable beside the tiny arm's exact zero, and that zero holds only there.
    """

    model_id: str
    passage: str | None = None
    passages: list[str] | None = None
    p: float = 1.0
    seed: int = 0
    match_prosody: bool = True
    keep: list[str] = []


@router.get("/graph")
def arch_graph(model_id: str) -> dict[str, Any]:
    """The cached traced architecture graph (nodes/edges/meta/schema_version)."""
    _gate(model_id)
    return _sig6(build_graph(model_id))


@router.get("/weights")
def arch_weights(
    model_id: str,
    param: str,
    r0: int = 0,
    r1: int | None = None,
    c0: int = 0,
    c1: int | None = None,
    max_cells: int = ARCH_WEIGHTS_MAX_CELLS,
) -> dict[str, Any]:
    """A real window of one parameter: exact within budget, strided-mean beyond."""
    _gate(model_id)
    return _sig6(weight_window(model_id, param, r0=r0, r1=r1, c0=c0, c1=c1, max_cells=max_cells))


@router.get("/trace")
def arch_trace(
    model_id: str,
    prompt: str = "",
    system_prompt: str | None = None,
    max_context: int = ARCH_DEFAULT_MAX_CONTEXT,
) -> dict[str, Any]:
    """Per-layer activations + per-node norms from one real traced forward pass."""
    _gate(model_id)
    return _sig6(
        trace_forward(model_id, prompt, system_prompt=system_prompt, max_context=max_context)
    )


@router.post("/generate")
def arch_generate(body: GenerateBody) -> dict[str, Any]:
    """Real autoregressive generation: greedy at temperature 0, seedable otherwise."""
    _gate(body.model_id)
    return _sig6(
        generate(
            body.model_id,
            body.prompt,
            system_prompt=body.system_prompt,
            temperature=body.temperature,
            max_new_tokens=body.max_new_tokens,
            seed=body.seed,
        )
    )


@router.post("/vacancy-score")
def arch_vacancy_score(body: VacancyScoreBody) -> dict[str, Any]:
    """What a word's FORM is worth to a model that has one for it (contract §8).

    Three variants of each passage — English, a real-word swap, and nonce — are scored by
    three real forward passes each, and the mean NLL over the tokens of the words that
    survive all three is reported per variant, together with the two differences that
    decompose the damage: `nll(swap) − nll(english)` is the cost of **wrong content**,
    `nll(nonce) − nll(swap)` the cost of **unknown form**. `nll(nonce) − nll(english)`
    is returned but flagged `headline: false` — it is their sum and conflates them.

    This is the full stack, at float32, so it reports everything. The static build runs a
    quantized export and may report only what the measurement of §8.3a bounded for that
    dtype; it refuses the rest by name rather than inventing an error bar.
    """
    _gate(body.model_id)
    if body.passage is not None and body.passages is not None:
        raise InvalidParamError(
            "send either `passage` (one) or `passages` (several), not both — they would "
            "silently disagree about what was scored"
        )
    passages = body.passages
    if body.passage is not None:
        passages = [body.passage]
    return _sig6(
        vacancy_score(
            body.model_id,
            passages,
            p=body.p,
            seed=body.seed,
            match_prosody=body.match_prosody,
            keep=frozenset(body.keep),
        )
    )
