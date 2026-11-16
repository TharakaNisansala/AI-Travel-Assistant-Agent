"""Shared test setup: every test starts with empty caches and never really sleeps.

The services cache successful API results (resilience.TTLCache) and wait before
retrying a temporary failure (resilience.sleep). Both would leak between tests
or slow them down, so this autouse fixture resets them for every test.

A second autouse fixture blocks the real network in every test that is not
marked live: a real HTTP request fails like an unplugged cable (httpx
ConnectError) instead of quietly reaching the internet. Tests that use
httpx.MockTransport are unaffected, because MockTransport never opens a socket.
"""

from __future__ import annotations

import httpx
import pytest

from ai_travel_assistant import resilience


@pytest.fixture(autouse=True)
def fresh_caches_and_no_sleep(monkeypatch) -> list[float]:
    """Clear all caches; record retry waits instead of sleeping.

    Tests that care about the waits can request this fixture by name and read
    the list of requested sleep durations.
    """
    resilience.clear_all_caches()
    waits: list[float] = []
    monkeypatch.setattr(resilience, "sleep", waits.append)
    yield waits
    resilience.clear_all_caches()


@pytest.fixture(autouse=True)
def no_real_network(request, monkeypatch) -> None:
    """Fail any real HTTP request made by an offline (non-live) test."""
    if request.node.get_closest_marker("live"):
        return

    def blocked(self, http_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(
            f"offline test tried to reach {http_request.url.host}",
            request=http_request,
        )

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", blocked)
