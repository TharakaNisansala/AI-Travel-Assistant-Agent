"""Stage 4 checks: the OSRM travel-time service and its estimate_travel_time tool.

Offline tests use httpx.MockTransport: geocoding GETs get Open-Meteo-shaped
answers, OSRM GETs get synthetic route JSON shaped like the real API.

Live tests call the real Open-Meteo geocoder and OSRM demo server (never
Gemini). Run them ONE AT A TIME by node ID, e.g.:
    uv run pytest -m live tests/test_travel.py::test_live_ella_to_kandy
"""

from __future__ import annotations

import httpx
import pytest
from langchain_core.messages import AIMessage

from ai_travel_assistant import travel_service
from ai_travel_assistant.tool_calling import build_tool_agent_graph, run_tool_agent
from ai_travel_assistant.tools import (
    TRAVEL_SOURCE,
    _duration,
    estimate_travel_time,
    format_travel_estimate,
)
from ai_travel_assistant.travel_service import (
    MAX_SNAP_M,
    OSRM_ROUTE_URL,
    InvalidTravelResponseError,
    NoRouteError,
    TravelBusyError,
    TravelEstimate,
    TravelHTTPError,
    TravelLocationNotFoundError,
    TravelNetworkError,
    TravelServiceError,
    TravelTimeoutError,
    get_travel_estimate,
    make_client,
)
from ai_travel_assistant.weather_service import GEOCODING_URL, Location
from tests.test_attractions import ToolCallingFake, tool_call

# --- Sample data --------------------------------------------------------------

PLACES = {
    "Ella": {"name": "Ella Town", "latitude": 6.8756, "longitude": 81.0463},
    "Kandy": {"name": "Kandy", "latitude": 7.2906, "longitude": 80.6336},
}


def route_json(distance_m=133416.6, duration_s=9575.2, snaps=(36.5, 51.6)) -> dict:
    return {
        "code": "Ok",
        "routes": [{"distance": distance_m, "duration": duration_s, "legs": []}],
        "waypoints": [{"distance": snap, "location": [0, 0]} for snap in snaps],
    }


class FakeAPIs:
    """Answers geocoding and OSRM requests; records them."""

    def __init__(self, route=None):
        self.route = route_json() if route is None else route
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        url = str(request.url.copy_with(query=None))
        if url == GEOCODING_URL:
            place = PLACES.get(request.url.params["name"])
            return httpx.Response(200, json={"results": [place]} if place else {})
        if isinstance(self.route, httpx.Response):
            return self.route
        if callable(self.route):
            return self.route(request)
        return httpx.Response(200, json=self.route)

    def client(self) -> httpx.Client:
        return make_client(transport=httpx.MockTransport(self))

    def osrm_requests(self) -> list[httpx.Request]:
        return [r for r in self.requests if str(r.url).startswith(OSRM_ROUTE_URL)]


def raising(exc: Exception):
    def handler(request):
        raise exc

    return handler


@pytest.fixture
def fake_apis(monkeypatch) -> FakeAPIs:
    api = FakeAPIs()
    monkeypatch.setattr(travel_service, "make_client", api.client)
    return api


# --- 1. The whole flow --------------------------------------------------------


def test_estimate_reads_distance_and_duration():
    estimate = get_travel_estimate("Ella", "Kandy", client=FakeAPIs().client())
    assert estimate.origin.name == "Ella Town"
    assert estimate.destination.name == "Kandy"
    assert estimate.road_distance_km == pytest.approx(133.4166)
    assert estimate.duration_min == pytest.approx(159.59, abs=0.01)
    assert estimate.straight_line_km == pytest.approx(64.8, abs=0.2)


def test_osrm_gets_longitude_first_and_no_geometry():
    api = FakeAPIs()
    get_travel_estimate("Ella", "Kandy", client=api.client())
    (request,) = api.osrm_requests()
    assert str(request.url.copy_with(query=None)) == (
        f"{OSRM_ROUTE_URL}/81.0463,6.8756;80.6336,7.2906"
    )
    assert request.url.params["overview"] == "false"
    assert request.headers["User-Agent"].startswith("ai-travel-assistant/")


def test_routes_are_cached():
    api = FakeAPIs()
    get_travel_estimate("Ella", "Kandy", client=api.client())
    get_travel_estimate("Ella", "Kandy", client=api.client())
    # 2 geocodings + 1 route; the second call was served from the caches.
    assert len(api.requests) == 3


def test_unknown_place_fails_before_routing():
    api = FakeAPIs()
    with pytest.raises(TravelLocationNotFoundError, match="Atlantis"):
        get_travel_estimate("Ella", "Atlantis", client=api.client())
    assert api.osrm_requests() == []


# --- 2. Distrusting a "successful" answer -------------------------------------


def test_far_snapped_point_is_rejected_even_when_osrm_says_ok():
    # The real public server answers Ella -> London with "code": "Ok" after
    # moving a point ~307 km. That route is meaningless.
    api = FakeAPIs(route_json(11_426_878, 481_314, snaps=(307_065, 12)))
    with pytest.raises(NoRouteError, match="307 km"):
        get_travel_estimate("Ella", "Kandy", client=api.client())


def test_snap_just_inside_the_limit_is_accepted():
    api = FakeAPIs(route_json(snaps=(MAX_SNAP_M, 10)))
    assert get_travel_estimate("Ella", "Kandy", client=api.client())


@pytest.mark.parametrize("code", ["NoRoute", "NoSegment"])
def test_osrm_no_route_codes_become_no_route(code):
    body = httpx.Response(400, json={"code": code, "message": "Impossible route"})
    with pytest.raises(NoRouteError):
        get_travel_estimate("Ella", "Kandy", client=FakeAPIs(body).client())


# --- 3. Failures --------------------------------------------------------------


@pytest.mark.parametrize(
    ("route", "error"),
    [
        (raising(httpx.ReadTimeout("slow")), TravelTimeoutError),
        (raising(httpx.ConnectError("down")), TravelNetworkError),
        (httpx.Response(429), TravelBusyError),
        (httpx.Response(504), TravelBusyError),
        (httpx.Response(400, json={"code": "InvalidQuery"}), TravelHTTPError),
        (httpx.Response(500, text="oops"), TravelHTTPError),
        (httpx.Response(200, text="oops"), InvalidTravelResponseError),
        (httpx.Response(200, json=[]), InvalidTravelResponseError),
        (
            httpx.Response(200, json={"code": "Ok", "routes": [], "waypoints": []}),
            InvalidTravelResponseError,
        ),
        (
            httpx.Response(
                200,
                json={**route_json(), "routes": [{"distance": None, "duration": 5}]},
            ),
            InvalidTravelResponseError,
        ),
    ],
    ids=[
        "timeout",
        "offline",
        "429",
        "504",
        "invalid-query",
        "500-html",
        "not-json",
        "not-object",
        "no-waypoints",
        "no-distance",
    ],
)
def test_failures_become_typed_errors(route, error):
    with pytest.raises(error):
        get_travel_estimate("Ella", "Kandy", client=FakeAPIs(route).client())


def test_every_error_is_a_travel_service_error():
    for error in (
        TravelLocationNotFoundError,
        NoRouteError,
        TravelTimeoutError,
        TravelNetworkError,
        TravelHTTPError,
        TravelBusyError,
        InvalidTravelResponseError,
    ):
        assert issubclass(error, TravelServiceError)


# --- 4. The estimate_travel_time tool (still offline) -------------------------


def test_tool_has_name_description_and_arguments():
    assert estimate_travel_time.name == "estimate_travel_time"
    assert set(estimate_travel_time.args) == {"origin", "destination", "country_code"}
    assert "timetable" in estimate_travel_time.description


@pytest.mark.parametrize(
    ("minutes", "text"),
    [(0.2, "1 min"), (45.4, "45 min"), (60, "1 h"), (159.6, "2 h 40 min")],
)
def test_durations_read_naturally(minutes, text):
    assert _duration(minutes) == text


def test_tool_output_is_short_with_units_caveat_and_source(fake_apis):
    text = estimate_travel_time.invoke({"origin": "Ella", "destination": "Kandy"})
    assert text.splitlines() == [
        "Road trip from Ella Town to Kandy:",
        "- Road distance: 133 km (straight line 65 km).",
        "- Driving time: about 2 h 40 min by car without traffic or stops.",
        "This is a free-flow car estimate: real journeys, especially on hill "
        "roads or by bus, tuk-tuk or train, usually take longer. Say so.",
        TRAVEL_SOURCE,
    ]


def test_tool_reports_failures_instead_of_raising(monkeypatch):
    api = FakeAPIs(httpx.Response(503))
    monkeypatch.setattr(travel_service, "make_client", api.client)
    text = estimate_travel_time.invoke({"origin": "Ella", "destination": "Kandy"})
    assert text.startswith("Travel time unavailable from 'Ella' to 'Kandy'")
    assert "busy (HTTP 503)" in text
    assert "do not guess distances or times" in text


def test_format_uses_location_labels():
    place = Location("A", 6.0, 80.0, country="Sri Lanka")
    text = format_travel_estimate(TravelEstimate(place, place, 0.0, 0.0, 0.0))
    assert text.startswith("Road trip from A (Sri Lanka) to A (Sri Lanka):")


# --- 5. The agent using the tool (scripted model, offline) --------------------


def test_agent_binds_and_runs_the_travel_tool(fake_apis):
    model = ToolCallingFake(
        messages=iter(
            [
                tool_call(
                    "estimate_travel_time",
                    {"origin": "Ella", "destination": "Kandy"},
                    "call_t1",
                ),
                AIMessage("About 2 h 40 min by car, longer by bus."),
            ]
        )
    )
    messages = run_tool_agent(
        build_tool_agent_graph(model), "How long from Ella to Kandy?"
    )
    assert estimate_travel_time in model.bound_tools
    result = messages[2]
    assert (result.name, result.status) == ("estimate_travel_time", "success")
    assert "2 h 40 min" in result.content


# --- 6. Live (real APIs, opt-in) ----------------------------------------------

INCONCLUSIVE = "INCONCLUSIVE (infrastructure, not a code failure)"


def live_estimate(origin: str, destination: str, country_code: str | None = "LK"):
    try:
        return get_travel_estimate(origin, destination, country_code)
    except (TravelBusyError, TravelTimeoutError, TravelNetworkError) as exc:
        pytest.skip(f"{INCONCLUSIVE}: {exc}")


@pytest.mark.live
def test_live_ella_to_kandy():
    estimate = live_estimate("Ella", "Kandy")
    # Roughly 130 km and 2.5-3 hours of free-flow driving.
    assert 100 <= estimate.road_distance_km <= 180
    assert 100 <= estimate.duration_min <= 300
    assert estimate.road_distance_km > estimate.straight_line_km


@pytest.mark.live
def test_live_route_across_the_sea_is_rejected():
    with pytest.raises(NoRouteError):
        live_estimate("Ella", "London", None)
