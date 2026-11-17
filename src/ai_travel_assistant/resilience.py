"""Stage 4: retries and caching shared by every external API client.

What this file demonstrates:
  1. Retrying only TEMPORARY failures: a timeout, a dropped connection, or an
     HTTP status that means "busy, try later" (429, 502, 503, 504). A 400 or a
     404 will fail the same way every time, so retrying it only wastes time.
  2. Honouring the server's Retry-After header, but never blocking the agent for
     long: if the server asks us to wait more than MAX_RETRY_WAIT_S, we give up at
     once and let the service report "busy". A traveller waiting 30 seconds for a
     chat reply is worse than a clear "try again in a minute".
  3. A small in-memory cache with a time-to-live (TTL). Free APIs are shared and
     rate-limited, and one agent turn often asks for the same thing twice (all
     three location tools geocode the same place). Only successes are cached.

This module knows nothing about any one API. Each service keeps its own error
messages and calls send_with_retry() around its httpx request, so the existing
error handling (timeout -> TimeoutError, 429 -> BusyError, ...) still applies to
the final attempt.

The cache lives in process memory only: it is empty after a restart and is not
shared between processes. That is enough for a CLI; a server would use Redis or
similar later.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable, Hashable
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

# Statuses that mean "temporarily unable to serve you": worth one more try.
RETRY_STATUSES = frozenset({429, 502, 503, 504})
# 2 attempts = the first request plus one retry. More retries would make a chat
# turn feel stuck and put extra load on free, shared servers.
MAX_ATTEMPTS = 2
# Wait before a retry when the server gives no Retry-After header.
BACKOFF_S = 1.0
# The longest Retry-After we are willing to wait inside a chat turn.
MAX_RETRY_WAIT_S = 5.0

# Looked up at call time (not imported by name), so tests can replace it with a
# function that records the waits instead of sleeping.
sleep: Callable[[float], None] = time.sleep


def retry_after_seconds(response: httpx.Response) -> float | None:
    """The wait a Retry-After header asks for, in seconds, or None.

    The header is either a number of seconds ("34") or an HTTP date.
    """
    value = response.headers.get("Retry-After")
    if not value:
        return None
    value = value.strip()
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when is None:
        return None
    return max(0.0, when.timestamp() - time.time())


def send_with_retry(
    send: Callable[[], httpx.Response],
    *,
    attempts: int = MAX_ATTEMPTS,
    retry_timeouts: bool = True,
) -> httpx.Response:
    """Call `send` (one HTTP request), retrying temporary failures.

    Returns the last response, which may still be an error status: the caller
    turns it into its own typed error. Re-raises the last httpx exception if
    every attempt failed with one.

    retry_timeouts=False suits slow servers (e.g. Overpass, 30 s timeout), where
    a second timeout would double an already long wait.
    """
    for attempt in range(1, attempts + 1):
        last = attempt == attempts
        try:
            response = send()
        except httpx.TimeoutException:
            if last or not retry_timeouts:
                raise
            sleep(BACKOFF_S * attempt)
            continue
        # Other transport errors: DNS hiccup, connection reset or refused.
        except httpx.TransportError:
            if last:
                raise
            sleep(BACKOFF_S * attempt)
            continue

        if response.status_code not in RETRY_STATUSES or last:
            return response
        wait = retry_after_seconds(response)
        if wait is None:
            wait = BACKOFF_S * attempt
        if wait > MAX_RETRY_WAIT_S:
            # The server wants us gone for a while: report busy now.
            return response
        sleep(wait)
    raise AssertionError("unreachable: the loop always returns or raises")


# --- Caching ------------------------------------------------------------------

_MISSING = object()
_ALL_CACHES: list[TTLCache] = []


class TTLCache:
    """A tiny least-recently-used cache whose entries expire after ttl_s seconds.

    Use get(key) and compare with `TTLCache.MISSING`, so a cached None or empty
    result is still a hit.
    """

    MISSING = _MISSING

    def __init__(
        self,
        name: str,
        ttl_s: float,
        maxsize: int = 256,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.name = name
        self.ttl_s = ttl_s
        self.maxsize = maxsize
        self.clock = clock
        self._entries: OrderedDict[Hashable, tuple[float, Any]] = OrderedDict()
        _ALL_CACHES.append(self)

    def get(self, key: Hashable) -> Any:
        entry = self._entries.get(key)
        if entry is None:
            return _MISSING
        expires_at, value = entry
        if self.clock() >= expires_at:
            del self._entries[key]
            return _MISSING
        self._entries.move_to_end(key)  # recently used: evicted last
        return value

    def set(self, key: Hashable, value: Any) -> None:
        self._entries[key] = (self.clock() + self.ttl_s, value)
        self._entries.move_to_end(key)
        while len(self._entries) > self.maxsize:
            self._entries.popitem(last=False)  # the least recently used

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)


def clear_all_caches() -> None:
    """Empty every cache (used by tests, so no test sees another's results)."""
    for cache in _ALL_CACHES:
        cache.clear()
