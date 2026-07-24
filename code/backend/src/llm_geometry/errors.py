"""Typed errors with HTTP mapping.

Every user-facing failure is a typed error carrying a clear message and an HTTP
status, so the API can render a consistent error envelope and NEVER fall back to a
substitute model or fabricated data (FR-003, FR-021, Constitution I).
"""

from __future__ import annotations

from typing import Any


class LLMGeometryError(Exception):
    """Base error. Subclasses set ``http_status`` and ``error_type``."""

    http_status: int = 500
    error_type: str = "InternalError"

    def __init__(self, message: str, detail: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail or {}

    def to_envelope(self) -> dict[str, Any]:
        return {"error": {"type": self.error_type, "message": self.message, "detail": self.detail}}


class UnsupportedModelError(LLMGeometryError):
    """Model is missing, not open-weights, gated, not a causal LM, or hides token
    probabilities — and there is no fallback (FR-002/FR-003)."""

    http_status = 422
    error_type = "UnsupportedModelError"


class InvalidParamError(LLMGeometryError):
    """A parameter is out of range or malformed (e.g., negative temperature)."""

    http_status = 400
    error_type = "InvalidParamError"


class ComputeError(LLMGeometryError):
    """A real computation failed; the underlying reason is surfaced verbatim."""

    http_status = 500
    error_type = "ComputeError"


class NotFoundError(LLMGeometryError):
    """A referenced resource (e.g., a job id) does not exist."""

    http_status = 404
    error_type = "NotFoundError"


class ModelTooLargeError(LLMGeometryError):
    """The model exceeds the Architecture Explorer parameter ceiling
    (``ARCH_MAX_PARAMS``) and is rejected before any weights are downloaded
    (FR-107, contracts/api.md feature 002)."""

    http_status = 422
    error_type = "ModelTooLargeError"


class InvalidWeightEditError(LLMGeometryError):
    """A Geometry Lab weight edit is unusable: bad matrix name/shape, both or neither
    of preset/values supplied, non-finite values, or an edit that cannot satisfy the
    unit-norm embedding constraint (feature 002 contract, `/api/geo/weights`)."""

    http_status = 422
    error_type = "InvalidWeightEditError"
