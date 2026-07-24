"""Deterministic cache keys.

A key is a stable hash over an explicit, sorted spec. Identical inputs always yield
the identical key (and therefore the identical cached payload), which is what makes
cache hits reproducible (FR-004, FR-013, SC-002).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from ..config import SCHEMA_VERSION


def _canonical(obj: Any) -> str:
    """Canonical JSON: sorted keys, no insignificant whitespace, stable for hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def make_cache_key(
    *,
    model_id: str,
    revision: str | None,
    artifact_type: str,
    inputs: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    seed: int | None = None,
) -> tuple[str, dict[str, Any]]:
    """Return ``(key, spec)`` for an artifact.

    ``spec`` is the full, inspectable description stored alongside the artifact; ``key``
    is a filesystem-safe id derived from it. The schema version is part of the spec so
    that a format change invalidates every prior key (FR-007).
    """
    spec: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "model_id": model_id,
        "revision": revision or "",
        "artifact_type": artifact_type,
        "inputs": inputs or {},
        "params": params or {},
        "seed": seed,
    }
    digest = hashlib.sha256(_canonical(spec).encode("utf-8")).hexdigest()[:32]
    return f"{artifact_type}-{digest}", spec
