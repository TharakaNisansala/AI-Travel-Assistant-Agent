"""Shared test setup: every test starts with empty caches and never really sleeps.

The services cache successful API results (resilience.TTLCache) and wait before
retrying a temporary failure (resilience.sleep). Both would leak between tests
or slow them down, so this autouse fixture resets them for every test.
"""

from __future__ import annotations

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
