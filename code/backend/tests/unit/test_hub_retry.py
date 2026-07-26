"""The Hub retry wrapper: transient 429/5xx are retried, real answers are not.

No mocks: the exceptions here are genuine ``HfHubHTTPError`` objects carrying genuine
``httpx.Response`` objects — the same types huggingface_hub raises — and the
not-retryable and success paths hit the real Hub over the network. The only thing done
locally is *triggering* a 429, because the Hub cannot be asked to rate-limit on demand.
"""

import time

import httpx
import pytest
from huggingface_hub import list_repo_files
from huggingface_hub.errors import HfHubHTTPError

from llm_geometry.models import hub
from llm_geometry.models.hub import HUB_RETRY_ATTEMPTS, hub_call, is_retryable

REAL_REPO = "gpt2"
MISSING_REPO = "ContextLab/llm-geometry-no-such-model-8f3a1c"


def _hub_error(status: int, headers: dict[str, str] | None = None) -> HfHubHTTPError:
    """A real HfHubHTTPError for `status`, built the way huggingface_hub builds them."""
    request = httpx.Request("GET", f"https://huggingface.co/api/models/{REAL_REPO}")
    response = httpx.Response(status, headers=headers or {}, request=request)
    return HfHubHTTPError(f"{status} for url {request.url}", response=response)


def test_classifies_transient_and_permanent_statuses():
    for status in (429, 500, 502, 503, 504):
        assert is_retryable(_hub_error(status)), f"{status} should be retried"
    for status in (400, 401, 403, 404, 416):
        assert not is_retryable(_hub_error(status)), f"{status} should not be retried"


def test_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr(hub, "HUB_RETRY_BASE_DELAY", 0.01)
    calls = []

    def flaky(value: str) -> str:
        calls.append(value)
        if len(calls) < 3:
            raise _hub_error(429)
        return f"ok:{value}"

    assert hub_call(flaky, "gpt2") == "ok:gpt2"
    assert len(calls) == 3, "should have retried twice before succeeding"


def test_gives_up_after_the_last_attempt(monkeypatch):
    monkeypatch.setattr(hub, "HUB_RETRY_BASE_DELAY", 0.01)
    calls = []

    def always_429() -> None:
        calls.append(1)
        raise _hub_error(429)

    with pytest.raises(HfHubHTTPError):
        hub_call(always_429)
    assert len(calls) == HUB_RETRY_ATTEMPTS, "must not retry forever"


def test_honours_retry_after_header(monkeypatch):
    monkeypatch.setattr(hub, "HUB_RETRY_BASE_DELAY", 10.0)  # would dominate if used
    calls = []

    def flaky() -> str:
        calls.append(1)
        if len(calls) < 2:
            raise _hub_error(429, {"Retry-After": "0.05"})
        return "ok"

    start = time.monotonic()
    assert hub_call(flaky) == "ok"
    elapsed = time.monotonic() - start
    assert elapsed < 5.0, f"waited {elapsed:.1f}s — ignored Retry-After"


def test_missing_repo_fails_immediately_over_the_real_network():
    """A 404 is a real answer about the repo: raise at once, no backoff."""
    start = time.monotonic()
    with pytest.raises(Exception) as exc_info:
        hub_call(list_repo_files, MISSING_REPO)
    elapsed = time.monotonic() - start
    assert not is_retryable(exc_info.value)
    assert elapsed < 30, f"took {elapsed:.1f}s — a 404 was retried with backoff"


def test_real_hub_call_passes_through():
    files = hub_call(list_repo_files, REAL_REPO)
    assert "config.json" in files
