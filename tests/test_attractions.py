"""Stage 4 checks: the Overpass attractions service.

The offline tests never touch the internet. httpx.MockTransport stands in for
the network: our handler receives each request and returns a prepared response
(or raises a timeout or connection error). The Overpass sample below is
synthetic, shaped like real Overpass JSON, with approximate coordinates for
real places around Ella, Sri Lanka.

The live test calls the real geocoding and Overpass APIs (2 HTTP requests, no
Gemini) and runs only with:
    uv run pytest -m live tests/test_attractions.py
"""

from __future__ import annotations

from urllib.parse import parse_qs

import httpx
import pytest

from ai_travel_assistant import attractions_service
from ai_travel_assistant.attractions_service import (
    CATEGORIES,
    OVERPASS_URL,
    AttractionsBusyError,
    AttractionsHTTPError,
    AttractionsLocationNotFoundError,
    AttractionsNetworkError,
    AttractionsServiceError,
    AttractionsTimeoutError,
    InvalidAttractionsResponseError,
    build_query,
    distance_km,
    fetch_attractions,
    get_attractions_report,
    make_client,
    parse_attractions,
)
from ai_travel_assistant.weather_service import GEOCODING_URL, USER_AGENT, Location

# --- Sample data --------------------------------------------------------------

ELLA = Location(
    name="Ella Town",
    latitude=6.8756,
    longitude=81.0463,
    country="Sri Lanka",
    admin1="Uva Province",
    admin2="Badulla District",
)

ELLA_GEOCODING = {
    "results": [
        {
            "name": "Ella Town",
            "latitude": 6.8756,
            "longitude": 81.0463,
            "country": "Sri Lanka",
            "admin1": "Uva Province",
            "admin2": "Badulla District",
        }
    ]
}


def node(id_, lat, lon, **tags):
    return {"type": "node", "id": id_, "lat": lat, "lon": lon, "tags": tags}


def way(id_, lat, lon, **tags):
    """A mapped outline: Overpass gives it a `center` point, not lat/lon."""
    return {
        "type": "way",
        "id": id_,
        "center": {"lat": lat, "lon": lon},
        "tags": tags,
    }


ELLA_OVERPASS = {
    "version": 0.6,
    "generator": "Overpass API",
    "elements": [
        node(1, 6.8424, 81.0516, name="Ravana Falls", natural="waterfall"),
        node(2, 6.8687, 81.0610, name="Little Adam's Peak", tourism="viewpoint"),
        way(3, 6.8768, 81.0608, name="Nine Arches Bridge", tourism="attraction"),
        # The same bridge again, mapped as a point a little further away.
        node(4, 6.8790, 81.0620, name="Nine Arches Bridge", tourism="attraction"),
        # Unnamed viewpoint: noise, must be dropped.
        node(5, 6.8700, 81.0500, tourism="viewpoint"),
        # Local-script name plus an English name: the English one is used.
        way(
            6,
            6.8600,
            81.0450,
            name="රාවණා ගුහාව",
            **{"name:en": "Ravana Cave"},
            historic="archaeological_site",
        ),
        node(
            7,
            6.8650,
            81.0400,
            name="Ella Viharaya",
            amenity="place_of_worship",
            religion="buddhist",
            wikidata="Q1",
        ),
        # No coordinates at all: must be skipped, not crash.
        {"type": "way", "id": 8, "tags": {"name": "Broken Way"}},
    ],
}


# --- Fake network -------------------------------------------------------------


class FakeAPIs:
    """Answers geocoding (GET) and Overpass (POST) requests; records them."""

    def __init__(self, geocoding=ELLA_GEOCODING, overpass=ELLA_OVERPASS):
        self.geocoding = geocoding
        self.overpass = overpass
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        url = str(request.url.copy_with(query=None))
        body = self.geocoding if url == GEOCODING_URL else self.overpass
        if isinstance(body, httpx.Response):
            return body
        return httpx.Response(200, json=body)

    def client(self) -> httpx.Client:
        return make_client(transport=httpx.MockTransport(self))

    def overpass_query(self) -> str:
        """The Overpass QL text of the last POST we received."""
        request = next(r for r in reversed(self.requests) if r.method == "POST")
        return parse_qs(request.content.decode())["data"][0]


def client_for(handler) -> httpx.Client:
    return make_client(transport=httpx.MockTransport(handler))


def raising(exc: Exception):
    def handler(request: httpx.Request) -> httpx.Response:
        raise exc

    return handler


def respond(status: int, **kwargs):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, **kwargs)

    return handler


# --- 1. Query building --------------------------------------------------------


def test_query_searches_around_the_location_with_the_radius():
    query = build_query(ELLA, "all", 5000)

    assert query.startswith("[out:json][timeout:25];")
    assert "(around:5000,6.8756,81.0463)" in query
    assert query.rstrip().endswith("out center tags 100;")


@pytest.mark.parametrize(
    ("category", "must_contain", "must_not_contain"),
    [
        ("landmarks", ['"tourism"="attraction"'], ["waterfall", "museum"]),
        ("nature", ["waterfall|peak", '"tourism"="viewpoint"'], ["museum"]),
        (
            "culture",
            ['"tourism"="museum"', "archaeological_site", '["wikidata"]'],
            ["waterfall"],
        ),
        ("all", ["attraction", "waterfall", "museum", "place_of_worship"], []),
    ],
)
def test_each_category_uses_its_own_tag_filters(
    category, must_contain, must_not_contain
):
    query = build_query(ELLA, category)
    for text in must_contain:
        assert text in query
    for text in must_not_contain:
        assert text not in query


def test_categories_are_exactly_the_four_agreed_ones():
    assert CATEGORIES == ("all", "nature", "culture", "landmarks")


def test_unknown_category_and_bad_radius_are_rejected():
    with pytest.raises(ValueError, match="Unknown category"):
        build_query(ELLA, "nightlife")
    with pytest.raises(ValueError, match="positive"):
        build_query(ELLA, "all", 0)


# --- 2. Parsing ---------------------------------------------------------------


def test_parsing_keeps_named_items_nearest_first():
    found = parse_attractions(ELLA_OVERPASS, ELLA)
    names = [a.name for a in found]

    assert "Broken Way" not in names  # no coordinates
    assert len(names) == len(set(names))  # no duplicates
    distances = [a.distance_km for a in found]
    assert distances == sorted(distances)
    assert len(found) == 5  # 8 elements - unnamed - no coords - duplicate


def test_duplicate_names_keep_the_nearest_copy():
    found = parse_attractions(ELLA_OVERPASS, ELLA)
    bridges = [a for a in found if a.name == "Nine Arches Bridge"]

    assert len(bridges) == 1
    # The way's centre (6.8768, 81.0608) is nearer than the extra node.
    assert bridges[0].latitude == 6.8768


def test_english_name_is_preferred_over_local_script():
    names = [a.name for a in parse_attractions(ELLA_OVERPASS, ELLA)]
    assert "Ravana Cave" in names
    assert "රාවණා ගුහාව" not in names


def test_kinds_are_short_readable_labels():
    kinds = {a.name: a.kind for a in parse_attractions(ELLA_OVERPASS, ELLA)}

    assert kinds["Ravana Falls"] == "waterfall"
    assert kinds["Little Adam's Peak"] == "viewpoint"
    assert kinds["Nine Arches Bridge"] == "attraction"
    assert kinds["Ravana Cave"] == "archaeological site"
    assert kinds["Ella Viharaya"] == "temple"


def test_distance_uses_the_haversine_formula():
    # One degree of latitude is about 111.2 km everywhere on Earth.
    assert distance_km(0, 0, 1, 0) == pytest.approx(111.19, abs=0.01)
    assert distance_km(6.8756, 81.0463, 6.8756, 81.0463) == 0


def test_response_without_elements_list_is_invalid():
    with pytest.raises(InvalidAttractionsResponseError):
        parse_attractions({"version": 0.6}, ELLA)


# --- 3. The whole flow (fake network) -----------------------------------------


def test_fetch_caps_results_but_reports_the_total():
    api = FakeAPIs()
    report = fetch_attractions(ELLA, "all", client=api.client(), limit=3)

    assert len(report.attractions) == 3
    assert report.total_found == 5
    assert report.radius_m == 5000
    assert report.category == "all"


def test_fetch_posts_the_query_with_our_user_agent():
    api = FakeAPIs()
    fetch_attractions(ELLA, "nature", client=api.client())

    request = api.requests[0]
    assert request.method == "POST"
    assert str(request.url) == OVERPASS_URL
    assert request.headers["User-Agent"] == USER_AGENT
    assert "waterfall|peak" in api.overpass_query()


def test_report_geocodes_then_searches():
    api = FakeAPIs()
    report = get_attractions_report("Ella", client=api.client())

    assert [r.method for r in api.requests] == ["GET", "POST"]
    assert report.location.name == "Ella Town"
    assert report.attractions[0].name == "Ella Viharaya"  # nearest in the sample


def test_no_matches_is_an_empty_report_not_an_error():
    api = FakeAPIs(overpass={"elements": []})
    report = get_attractions_report("Ella", client=api.client())

    assert report.attractions == ()
    assert report.total_found == 0


def test_unknown_category_fails_before_any_network_call():
    api = FakeAPIs()
    with pytest.raises(ValueError):
        get_attractions_report("Ella", "nightlife", client=api.client())
    assert api.requests == []


def test_report_opens_its_own_client_when_none_is_given(monkeypatch):
    api = FakeAPIs()
    monkeypatch.setattr(attractions_service, "make_client", api.client)

    report = get_attractions_report("Ella")
    assert report.total_found == 5


# --- 4. Failures --------------------------------------------------------------


def test_unknown_place_becomes_an_attractions_error():
    api = FakeAPIs(geocoding={"generationtime_ms": 0.1})
    with pytest.raises(AttractionsLocationNotFoundError, match="Atlantis"):
        get_attractions_report("Atlantis", client=api.client())


def test_geocoding_outage_becomes_an_attractions_error():
    api = FakeAPIs(geocoding=httpx.Response(500))
    with pytest.raises(AttractionsServiceError, match="Could not locate"):
        get_attractions_report("Ella", client=api.client())


@pytest.mark.parametrize(
    ("handler", "error"),
    [
        (raising(httpx.ReadTimeout("slow")), AttractionsTimeoutError),
        (raising(httpx.ConnectError("offline")), AttractionsNetworkError),
        (respond(429), AttractionsBusyError),
        (respond(504), AttractionsBusyError),
        (respond(400, text="<html>syntax error</html>"), AttractionsHTTPError),
        (respond(200, text="not json"), InvalidAttractionsResponseError),
        (respond(200, json=["a", "list"]), InvalidAttractionsResponseError),
    ],
    ids=["timeout", "offline", "429", "504", "400", "not-json", "not-object"],
)
def test_overpass_failures_become_typed_errors(handler, error):
    with pytest.raises(error):
        fetch_attractions(ELLA, client=client_for(handler))


def test_busy_errors_are_http_errors_and_service_errors():
    # One `except AttractionsServiceError` in the tool catches everything,
    # while a future retry policy can single out AttractionsBusyError.
    assert issubclass(AttractionsBusyError, AttractionsHTTPError)
    assert issubclass(AttractionsHTTPError, AttractionsServiceError)


def test_runtime_error_remark_with_http_200_is_not_treated_as_no_results():
    body = {
        "elements": [],
        "remark": 'runtime error: Query timed out in "query" at line 3 after 25 s.',
    }
    with pytest.raises(AttractionsBusyError, match="could not finish"):
        fetch_attractions(ELLA, client=client_for(respond(200, json=body)))


# --- 5. Live (real APIs, opt-in) ----------------------------------------------


@pytest.mark.live
def test_live_attractions_near_ella():
    report = get_attractions_report("Ella")

    assert report.attractions, "expected at least one attraction near Ella"
    assert all(a.distance_km <= 5.5 for a in report.attractions)
