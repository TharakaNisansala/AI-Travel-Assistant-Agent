"""Deployment: TTLCache under concurrent use (one thread per web visitor). Offline.

Several tests use a clock that yields the CPU on every call. That forces a
thread switch in the middle of get()/set() - exactly where an unlocked
OrderedDict breaks (two threads deleting the same expired key, or
move_to_end() on a key another thread just evicted, raise KeyError).
"""

from __future__ import annotations

import threading
import time

import httpx

from ai_travel_assistant import weather_service
from ai_travel_assistant.resilience import TTLCache, clear_all_caches

THREADS = 16


class YieldingClock:
    """A settable clock that gives other threads a chance to run on each call."""

    def __init__(self, now: float = 0.0):
        self.now = now

    def __call__(self) -> float:
        time.sleep(0)  # release the GIL: invite a thread switch right here
        return self.now


def run_threads(target, count: int = THREADS) -> list[BaseException]:
    """Run target(i) on `count` threads started together; return their errors."""
    errors: list[BaseException] = []
    start = threading.Barrier(count)

    def worker(i: int) -> None:
        start.wait()
        try:
            target(i)
        except BaseException as exc:  # noqa: BLE001 - collected for the assert
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return errors


def entries_are_consistent(cache: TTLCache) -> bool:
    """Size within maxsize, and every stored key still readable."""
    keys = list(cache._entries)
    return len(keys) <= cache.maxsize and all(
        cache.get(k) is not TTLCache.MISSING for k in keys
    )


def test_concurrent_reads_all_see_the_stored_value():
    cache = TTLCache("reads", ttl_s=60, clock=YieldingClock())
    cache.set("ella", {"temp": 21})
    seen = []
    errors = run_threads(lambda i: seen.extend(cache.get("ella") for _ in range(200)))
    assert not errors
    assert seen == [{"temp": 21}] * (THREADS * 200)


def test_concurrent_writes_keep_the_size_limit_and_a_consistent_order():
    cache = TTLCache("writes", ttl_s=60, maxsize=8, clock=YieldingClock())

    def write(i):
        for n in range(200):
            cache.set((i, n), n)

    assert not run_threads(write)
    assert len(cache) == 8
    assert entries_are_consistent(cache)


def test_concurrent_expiry_of_the_same_key_never_raises():
    clock = YieldingClock()
    cache = TTLCache("expiry", ttl_s=10, clock=clock)

    def read_expired(i):
        for _ in range(50):
            assert cache.get("k") is TTLCache.MISSING

    for _ in range(20):
        clock.now = 0
        cache.set("k", "old")
        clock.now = 10  # every thread now finds the same expired entry
        errors = run_threads(read_expired)
        assert not errors, errors[0]
        assert len(cache) == 0


def test_concurrent_reads_and_evictions_never_raise():
    cache = TTLCache("evict", ttl_s=60, maxsize=4, clock=YieldingClock())

    def churn(i):
        for n in range(300):
            if i % 2:
                cache.set(n % 12, n)  # evicts the least recently used
            else:
                value = cache.get(n % 12)  # may find a key that is being evicted
                assert value is TTLCache.MISSING or isinstance(value, int)

    errors = run_threads(churn)
    assert not errors, errors[0]
    assert entries_are_consistent(cache)


def test_concurrent_clear_with_reads_and_writes_never_raises():
    cache = TTLCache("clear", ttl_s=60, maxsize=16, clock=YieldingClock())

    def mixed(i):
        for n in range(200):
            if i == 0 and n % 20 == 0:
                clear_all_caches()
            elif i % 2:
                cache.set(n % 20, n)
            else:
                cache.get(n % 20)

    assert not run_threads(mixed)
    assert entries_are_consistent(cache)


def test_a_slow_fetch_does_not_block_other_visitors():
    """The lock is never held while a service waits for the network."""
    cache = TTLCache("slow", ttl_s=60)
    cache.set("kandy", "cached")
    fetch_started, fetch_may_finish = threading.Event(), threading.Event()

    def slow_service():
        if cache.get("ella") is TTLCache.MISSING:  # miss...
            fetch_started.set()
            fetch_may_finish.wait(5)  # ...a slow network call, no lock held
            cache.set("ella", "fetched")

    visitor = threading.Thread(target=slow_service)
    visitor.start()
    assert fetch_started.wait(5)
    began = time.perf_counter()
    assert cache.get("kandy") == "cached"  # another visitor, mid-fetch
    cache.set("galle", "x")
    assert time.perf_counter() - began < 0.5
    fetch_may_finish.set()
    visitor.join(5)
    assert cache.get("ella") == "fetched"


def test_a_real_service_does_not_hold_the_cache_lock_during_its_request():
    locked_during_request = []

    def route(request):
        locked_during_request.append(weather_service.GEOCODE_CACHE._lock.locked())
        return httpx.Response(
            200,
            json={"results": [{"name": "Ella", "latitude": 6.9, "longitude": 81.0}]},
        )

    client = weather_service.make_client(httpx.MockTransport(route))
    weather_service.geocode("Ella", client=client)
    assert locked_during_request == [False]


def test_the_services_cache_settings_are_unchanged():
    from ai_travel_assistant import (
        attractions_service,
        currency_service,
        guide_service,
        hotel_service,
        travel_service,
    )

    ttls = {
        weather_service.GEOCODE_CACHE: 24 * 3600,
        weather_service.FORECAST_CACHE: 15 * 60,
        travel_service.ROUTE_CACHE: 24 * 3600,
        guide_service.ARTICLE_CACHE: 24 * 3600,
        hotel_service.WIKITEXT_CACHE: 24 * 3600,
        currency_service.RATES_CACHE: 3600,
        attractions_service.ATTRACTIONS_CACHE: 3600,
    }
    for cache, ttl in ttls.items():
        assert (cache.ttl_s, cache.maxsize) == (ttl, 256), cache.name
