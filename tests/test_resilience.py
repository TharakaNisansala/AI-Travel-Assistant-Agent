"""Stage 4 checks: shared retries and caching (resilience.py), fully offline.

`fresh_caches_and_no_sleep` (tests/conftest.py) replaces resilience.sleep with a
recorder, so these tests see exactly which waits were requested without
actually waiting.
"""

from __future__ import annotations

import httpx
import pytest

from ai_travel_assistant import attractions_service, resilience, weather_service
from ai_travel_assistant.resilience import (
    BACKOFF_S,
    MAX_ATTEMPTS,
    MAX_RETRY_WAIT_S,
    TTLCache,
    retry_after_seconds,
    send_with_retry,
)


class Scripted:
    """A fake `send`: returns (or raises) the prepared outcomes in order."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def __call__(self) -> httpx.Response:
        outcome = self.outcomes[self.calls]
        self.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def ok() -> httpx.Response:
    return httpx.Response(200, json={"ok": True})


def status(code: int, **headers) -> httpx.Response:
    return httpx.Response(code, headers=headers)


# --- send_with_retry ----------------------------------------------------------


def test_two_attempts_is_the_agreed_budget():
    assert MAX_ATTEMPTS == 2


def test_success_is_returned_without_retrying(fresh_caches_and_no_sleep):
    send = Scripted(ok())
    assert send_with_retry(send).status_code == 200
    assert send.calls == 1
    assert fresh_caches_and_no_sleep == []


@pytest.mark.parametrize("code", [429, 502, 503, 504])
def test_busy_statuses_are_retried_once(code, fresh_caches_and_no_sleep):
    send = Scripted(status(code), ok())
    assert send_with_retry(send).status_code == 200
    assert send.calls == 2
    assert fresh_caches_and_no_sleep == [BACKOFF_S]


@pytest.mark.parametrize("code", [400, 404, 500])
def test_permanent_errors_are_not_retried(code):
    # 500 usually means a bug on the server for THIS request: retrying repeats it.
    send = Scripted(status(code))
    assert send_with_retry(send).status_code == code
    assert send.calls == 1


def test_last_busy_response_is_returned_for_the_caller_to_report():
    send = Scripted(status(503), status(503))
    assert send_with_retry(send).status_code == 503
    assert send.calls == 2


def test_retry_after_seconds_is_honoured(fresh_caches_and_no_sleep):
    send = Scripted(status(429, **{"Retry-After": "3"}), ok())
    assert send_with_retry(send).status_code == 200
    assert fresh_caches_and_no_sleep == [3.0]


def test_long_retry_after_gives_up_at_once(fresh_caches_and_no_sleep):
    # Wikimedia really sends "Retry-After: 34": too long for a chat turn.
    long_wait = str(int(MAX_RETRY_WAIT_S) + 29)
    send = Scripted(status(429, **{"Retry-After": long_wait}), ok())
    assert send_with_retry(send).status_code == 429
    assert send.calls == 1
    assert fresh_caches_and_no_sleep == []


def test_timeouts_and_connection_errors_are_retried_once():
    assert send_with_retry(Scripted(httpx.ReadTimeout("slow"), ok())).is_success
    assert send_with_retry(Scripted(httpx.ConnectError("reset"), ok())).is_success


def test_repeated_timeout_is_re_raised_for_the_caller_to_report():
    send = Scripted(httpx.ReadTimeout("slow"), httpx.ReadTimeout("slow again"))
    with pytest.raises(httpx.ReadTimeout, match="slow again"):
        send_with_retry(send)
    assert send.calls == 2


def test_timeouts_can_be_excluded_from_retries():
    send = Scripted(httpx.ReadTimeout("slow"), ok())
    with pytest.raises(httpx.ReadTimeout):
        send_with_retry(send, retry_timeouts=False)
    assert send.calls == 1


def test_non_transport_httpx_errors_are_never_retried():
    send = Scripted(httpx.DecodingError("corrupt"), ok())
    with pytest.raises(httpx.DecodingError):
        send_with_retry(send)
    assert send.calls == 1


def test_retry_after_parsing():
    assert retry_after_seconds(status(429)) is None
    assert retry_after_seconds(status(429, **{"Retry-After": "34"})) == 34.0
    assert retry_after_seconds(status(429, **{"Retry-After": "soon"})) is None
    # An HTTP date in the past means "no need to wait".
    past = "Wed, 21 Oct 2015 07:28:00 GMT"
    assert retry_after_seconds(status(429, **{"Retry-After": past})) == 0.0


# --- TTLCache -----------------------------------------------------------------


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_cache_returns_values_until_they_expire():
    clock = Clock()
    cache = TTLCache("test", ttl_s=10, clock=clock)
    cache.set("k", "v")
    clock.now = 9.9
    assert cache.get("k") == "v"
    clock.now = 10.0
    assert cache.get("k") is TTLCache.MISSING
    assert len(cache) == 0


def test_cached_none_is_still_a_hit():
    cache = TTLCache("test", ttl_s=10)
    cache.set("k", None)
    assert cache.get("k") is None


def test_cache_evicts_the_least_recently_used_entry():
    cache = TTLCache("test", ttl_s=10, maxsize=2)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.get("a")  # "a" is now newer than "b"
    cache.set("c", 3)
    assert cache.get("b") is TTLCache.MISSING
    assert (cache.get("a"), cache.get("c")) == (1, 3)


def test_clear_all_caches_empties_every_cache():
    cache = TTLCache("test", ttl_s=10)
    cache.set("k", "v")
    resilience.clear_all_caches()
    assert len(cache) == 0


# --- The services use them ----------------------------------------------------

ELLA_GEOCODING = {
    "results": [{"name": "Ella", "latitude": 6.8756, "longitude": 81.0463}]
}


def counting_client(handler):
    requests: list[httpx.Request] = []

    def route(request):
        requests.append(request)
        return handler(request)

    return weather_service.make_client(httpx.MockTransport(route)), requests


def test_geocoding_is_cached_across_tools():
    client, requests = counting_client(
        lambda r: httpx.Response(200, json=ELLA_GEOCODING)
    )
    first = weather_service.geocode("Ella", client=client)
    # Same place, different spelling of case and spaces: still one request.
    second = weather_service.geocode("  ella ", client=client)
    assert first == second
    assert len(requests) == 1


def test_geocoding_cache_keeps_countries_apart():
    client, requests = counting_client(
        lambda r: httpx.Response(200, json=ELLA_GEOCODING)
    )
    weather_service.geocode("Ella", "LK", client=client)
    weather_service.geocode("Ella", "GR", client=client)
    assert len(requests) == 2


def test_failures_are_not_cached():
    responses = iter([httpx.Response(400), httpx.Response(200, json=ELLA_GEOCODING)])
    client, requests = counting_client(lambda r: next(responses))
    with pytest.raises(weather_service.WeatherHTTPError):
        weather_service.geocode("Ella", client=client)
    assert weather_service.geocode("Ella", client=client).name == "Ella"
    assert len(requests) == 2


def test_weather_retries_a_busy_forecast_api():
    forecast = {"daily": {"time": ["2026-11-15"]}}
    responses = iter(
        [
            httpx.Response(200, json=ELLA_GEOCODING),
            httpx.Response(503),
            httpx.Response(200, json=forecast),
        ]
    )
    client, requests = counting_client(lambda r: next(responses))
    report = weather_service.get_weather_report("Ella", client=client)
    assert report.daily[0].date == "2026-11-15"
    assert len(requests) == 3


def test_overpass_does_not_retry_a_timeout():
    calls = []

    def handler(request):
        calls.append(request)
        raise httpx.ReadTimeout("slow")

    client = attractions_service.make_client(httpx.MockTransport(handler))
    location = weather_service.Location("Ella", 6.8756, 81.0463)
    with pytest.raises(attractions_service.AttractionsTimeoutError):
        attractions_service.fetch_attractions(location, client=client)
    # One attempt per server, no retry on the same server: the main server,
    # then the fallback.
    assert [str(c.url) for c in calls] == list(attractions_service.OVERPASS_URLS)
