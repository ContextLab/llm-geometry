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
from ..config import ARCH_DEFAULT_MAX_CONTEXT, ARCH_WEIGHTS_MAX_CELLS
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
