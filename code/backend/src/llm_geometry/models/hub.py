"""Retry wrapper for HuggingFace Hub metadata calls.

The Hub rate-limits unauthenticated clients, and CI runners share an egress IP with
every other project on the runner fleet. A single ``429 Too Many Requests`` used to
take down the whole static export: the nightly run of 2026-07-26 died on
``list_repo_files("gpt2")`` after the size gate had already logged a 429 for the same
repo seconds earlier.

``hub_call`` retries the transient statuses (429 + 5xx) with exponential backoff,
honouring ``Retry-After`` when the Hub sends it. It deliberately does NOT swallow the
error: after the last attempt the original exception propagates, so a genuine outage
still fails loudly rather than producing a half-built artifact.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# 5 attempts with a 2s base doubles to 2+4+8+16 = 30s of waiting in the worst case —
# long enough to ride out the Hub's per-minute buckets, short enough that a real
# outage fails the build promptly instead of hanging a job for its whole timeout.
HUB_RETRY_ATTEMPTS = 5
HUB_RETRY_BASE_DELAY = 2.0
HUB_RETRY_MAX_DELAY = 30.0

# 429 = rate limited; 5xx = Hub/CDN having a bad minute. Everything else (401 gated,
# 404 missing, 403 forbidden) is a real answer about the repo and must not be retried.
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


def _status_of(exc: BaseException) -> int | None:
    """HTTP status carried by a hub exception, if it has one.

    huggingface_hub raises ``HfHubHTTPError`` subclasses that keep the originating
    ``response`` object; httpx and requests responses both expose ``status_code``.
    """
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return int(status) if isinstance(status, int) else None


def _retry_after(exc: BaseException) -> float | None:
    """Seconds the Hub asked us to wait, when it says so explicitly."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    try:
        raw = headers.get("Retry-After")
    except AttributeError:
        return None
    if raw is None:
        return None
    try:
        # Only the delta-seconds form; the HTTP-date form is rare and not worth
        # parsing badly. An unparseable value falls back to exponential backoff.
        return max(0.0, float(str(raw).strip()))
    except ValueError:
        return None


def is_retryable(exc: BaseException) -> bool:
    """True when `exc` is a transient hub failure worth retrying."""
    return _status_of(exc) in RETRYABLE_STATUS


def hub_call(fn: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
    """Call a HuggingFace Hub function, retrying transient rate limits / 5xx.

    Raises the original exception once attempts are exhausted, or immediately for any
    status that describes the repo itself (404 missing, 401/403 gated).
    """
    delay = HUB_RETRY_BASE_DELAY
    for attempt in range(1, HUB_RETRY_ATTEMPTS + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if attempt == HUB_RETRY_ATTEMPTS or not is_retryable(exc):
                raise
            wait = _retry_after(exc)
            if wait is None:
                wait = delay
                delay = min(delay * 2, HUB_RETRY_MAX_DELAY)
            wait = min(wait, HUB_RETRY_MAX_DELAY)
            logger.warning(
                "hub %s returned %s (attempt %d/%d) — retrying in %.1fs",
                getattr(fn, "__name__", repr(fn)),
                _status_of(exc),
                attempt,
                HUB_RETRY_ATTEMPTS,
                wait,
            )
            time.sleep(wait)
    raise AssertionError("unreachable: loop either returns or raises")
