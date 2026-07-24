"""Geometry Lab routes (`/api/geo/*`).

Implements the frozen contract in specs/002-interactive-model-explorer/contracts/api.md.
Owned by Batch-2 agent B1; Batch 0 registers the (empty) router so parallel work
touches disjoint files.
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/geo", tags=["geo"])
