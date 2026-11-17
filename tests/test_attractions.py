"""Stage 4 checks: the Overpass attractions service and its find_attractions tool.

The offline tests never touch the internet. httpx.MockTransport stands in for
the network: our handler receives each request and returns a prepared response
(or raises a timeout or connection error). The Overpass sample below is
synthetic, shaped like real Overpass JSON, with approximate coordinates for
real places around Ella, Sri Lanka.

The live tests (section 7) call the real Open-Meteo geocoding and Overpass APIs,
never Gemini. Run them ONE AT A TIME by node ID, about 10 seconds apart, e.g.:
    uv run pytest -m live tests/test_attractions.py::test_live_attractions_near_ella
Never run the whole live group (`pytest -m live` with no node ID): it also
includes Gemini tests that spend the small daily quota.
"""

from __future__ import annotations

import re
from urllib.parse import parse_qs

import httpx
import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ValidationError

from ai_travel_assistant import attractions_service, guide_service
from ai_travel_assistant.attractions_service import (
    CATEGORIES,
    OVERPASS_FALLBACK_URL,
    OVERPASS_URL,
    OVERPASS_URLS,
    Attraction,
    AttractionsBusyError,
    AttractionsHTTPError,
    AttractionsLocationNotFoundError,
    AttractionsNetworkError,
    AttractionsReport,
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
from ai_travel_assistant.guide_service import (
    DestinationGuide,
    GuideNetworkError,
    GuideSection,
)
from ai_travel_assistant.tool_calling import (
    SYSTEM_PROMPT,
    build_tool_agent_graph,
    describe_steps,
    final_answer,
    run_tool_agent,
)
from ai_travel_assistant.tools import (
    ATTRACTIONS_SOURCE,
    find_attractions,
    format_attractions_report,
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


# --- 4b. Server fallback ------------------------------------------------------
# The main public Overpass server is often at capacity, so a failure there must
# send the same query to the fallback server before giving up.


def per_server(main, fallback):
    """A handler that answers the main and fallback servers differently."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return {OVERPASS_URL: main, OVERPASS_FALLBACK_URL: fallback}[str(request.url)](
            request
        )

    return handler, requests


def test_two_distinct_servers_are_configured_main_first():
    assert OVERPASS_URLS == (OVERPASS_URL, OVERPASS_FALLBACK_URL)
    assert OVERPASS_URL != OVERPASS_FALLBACK_URL
    assert all(url.startswith("https://") for url in OVERPASS_URLS)


def test_main_server_success_never_asks_the_fallback():
    handler, requests = per_server(respond(200, json=ELLA_OVERPASS), respond(500))
    report = fetch_attractions(ELLA, client=client_for(handler))
    assert report.attractions
    assert [str(r.url) for r in requests] == [OVERPASS_URL]


@pytest.mark.parametrize(
    "main_failure",
    [
        respond(502),
        respond(504),
        respond(400),
        respond(200, text="not json"),
        respond(200, json={"elements": [], "remark": "runtime error: timeout"}),
        raising(httpx.ReadTimeout("slow")),
        raising(httpx.ConnectError("refused")),
    ],
    ids=["502", "504", "400", "not-json", "remark", "timeout", "offline"],
)
def test_any_main_server_failure_falls_back_to_the_second_server(main_failure):
    handler, requests = per_server(main_failure, respond(200, json=ELLA_OVERPASS))
    report = fetch_attractions(ELLA, client=client_for(handler))

    expected = parse_attractions(ELLA_OVERPASS, ELLA)
    assert list(report.attractions) == expected[: len(report.attractions)]
    assert str(requests[-1].url) == OVERPASS_FALLBACK_URL
    # The fallback receives exactly the same query as the main server.
    bodies = {r.content for r in requests}
    assert len(bodies) == 1


def test_when_every_server_fails_the_message_names_each_problem():
    handler, requests = per_server(respond(502), raising(httpx.ReadTimeout("slow")))
    with pytest.raises(AttractionsTimeoutError) as caught:
        fetch_attractions(ELLA, client=client_for(handler))

    message = str(caught.value)
    assert message.startswith("No Overpass server could complete the search.")
    assert "overpass-api.de: The Overpass API returned HTTP 502." in message
    assert "overpass.private.coffee: The Overpass API did not respond" in message
    # Main: first try + one retry of the 502; fallback: one timeout, no retry.
    assert [str(r.url) for r in requests] == [
        OVERPASS_URL,
        OVERPASS_URL,
        OVERPASS_FALLBACK_URL,
    ]


def test_the_error_type_is_that_of_the_last_server():
    # The tool and the live-test policy read the type and message, so the
    # final error must still be a precise AttractionsServiceError subclass.
    handler, _ = per_server(raising(httpx.ReadTimeout("slow")), respond(504))
    with pytest.raises(AttractionsBusyError):
        fetch_attractions(ELLA, client=client_for(handler))


def test_a_fallback_success_is_cached_like_any_other():
    handler, requests = per_server(respond(502), respond(200, json=ELLA_OVERPASS))
    client = client_for(handler)
    first = fetch_attractions(ELLA, client=client)
    count = len(requests)
    assert fetch_attractions(ELLA, client=client) == first
    assert len(requests) == count  # served from the cache


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


# --- 5. The find_attractions tool (still offline) -----------------------------


@pytest.fixture
def fake_apis(monkeypatch) -> FakeAPIs:
    """Route the tool's own HTTP client to a FakeAPIs."""
    api = FakeAPIs()
    monkeypatch.setattr(attractions_service, "make_client", api.client)
    return api


def overpass_fails_with(monkeypatch, handler) -> None:
    """Geocoding succeeds; every Overpass request goes to `handler`."""

    def route(request: httpx.Request) -> httpx.Response:
        if str(request.url.copy_with(query=None)) == GEOCODING_URL:
            return httpx.Response(200, json=ELLA_GEOCODING)
        return handler(request)

    monkeypatch.setattr(attractions_service, "make_client", lambda: client_for(route))


def attraction(name: str, kind: str, km: float) -> Attraction:
    return Attraction(name, kind, 6.87, 81.05, km)


def make_report(*attractions: Attraction, category="all", total=None):
    return AttractionsReport(
        location=ELLA,
        category=category,
        radius_m=5000,
        attractions=attractions,
        total_found=len(attractions) if total is None else total,
    )


class ToolCallingFake(GenericFakeChatModel):
    """A scripted chat model that accepts bind_tools(), so the real graph runs.

    Records the tools it was given and the messages it received on each call,
    so tests can check what the LLM "saw" at every step.
    """

    bound_tools: list = []
    calls: list = []

    def bind_tools(self, tools, **kwargs):
        self.bound_tools = list(tools)
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def tool_call(name: str, args: dict, call_id: str) -> AIMessage:
    """What a tool-calling LLM returns when it wants one tool run."""
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": args, "id": call_id}]
    )


# Schema: what bind_tools() sends to the LLM.


def test_tool_has_name_description_and_arguments():
    assert find_attractions.name == "find_attractions"
    assert "attractions" in find_attractions.description.lower()
    assert set(find_attractions.args) == {"destination", "category", "country_code"}


def test_category_is_an_enum_of_the_service_categories():
    schema = convert_to_openai_tool(find_attractions)["function"]["parameters"]
    category = schema["properties"]["category"]
    assert category["enum"] == list(CATEGORIES)
    assert category["default"] == "all"
    # Only the destination is required; the LLM may leave the others out.
    assert schema["required"] == ["destination"]


def test_docstring_explains_every_category_and_what_is_not_covered():
    description = find_attractions.description
    for name in CATEGORIES:
        assert f'"{name}"' in description
    assert "NOT cover" in description


def test_docstring_numbers_match_the_service_defaults():
    # The docstring is the LLM's only manual; it must not drift from the code.
    radius_km = f"{attractions_service.DEFAULT_RADIUS_M / 1000:g} km"
    assert f"within {radius_km}" in find_attractions.description
    assert f"up to {attractions_service.DEFAULT_LIMIT} " in find_attractions.description


# Success: the tool calls the service and formats the result.


def test_tool_returns_compact_attractions_nearest_first(fake_apis):
    text = find_attractions.invoke({"destination": "Ella"})

    lines = text.splitlines()
    assert lines[0] == (
        "Attractions near Ella Town (Badulla District, Uva Province, Sri Lanka); "
        "category: all; within 5 km; nearest first:"
    )
    # 5 named, de-duplicated attractions from the 8 sample elements.
    assert lines[1:6] == [
        "- Ella Viharaya (temple), 1.4 km",
        "- Nine Arches Bridge (attraction), 1.6 km",
        "- Ravana Cave (archaeological site), 1.7 km",
        "- Little Adam's Peak (viewpoint), 1.8 km",
        "- Ravana Falls (waterfall), 3.7 km",
    ]
    assert lines[-1].startswith("Source: OpenStreetMap")
    assert "straight-line" in text


def test_tool_output_has_no_raw_json_ids_or_coordinates(fake_apis):
    text = find_attractions.invoke({"destination": "Ella"})
    for noise in ("{", '"elements"', "wikidata", "6.8424", "81.0516", "node"):
        assert noise not in text


def test_tool_passes_category_and_country_to_the_service(fake_apis):
    find_attractions.invoke({"destination": "Ella", "category": "nature"})

    geocoding = next(r for r in fake_apis.requests if r.method == "GET")
    assert geocoding.url.params["countryCode"] == "LK"  # the default
    query = fake_apis.overpass_query()
    assert '"natural"~' in query and '"tourism"="viewpoint"' in query
    assert "museum" not in query  # culture filters were not sent


def test_tool_passes_a_non_default_country_to_the_geocoder(fake_apis):
    find_attractions.invoke({"destination": "Ooty", "country_code": "IN"})
    geocoding = next(r for r in fake_apis.requests if r.method == "GET")
    assert geocoding.url.params["name"] == "Ooty"
    assert geocoding.url.params["countryCode"] == "IN"


def test_bad_category_is_rejected_by_the_schema_before_any_request(fake_apis):
    with pytest.raises(ValidationError):
        find_attractions.invoke({"destination": "Ella", "category": "nightlife"})
    assert fake_apis.requests == []


def test_agent_gets_a_fixable_error_for_a_bad_category(fake_apis):
    # Inside the agent, ToolNode turns the schema error into a ToolMessage, so
    # the LLM can correct itself instead of the run crashing.
    bad_call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "find_attractions",
                "args": {"destination": "Ella", "category": "nightlife"},
                "id": "call_1",
            }
        ],
    )
    model = ToolCallingFake(messages=iter([bad_call, AIMessage("Sorry.")]))
    graph = build_tool_agent_graph(model, tools=[find_attractions])

    result = graph.invoke({"messages": [HumanMessage("Nightlife in Ella?")]})

    reply = result["messages"][2]
    assert isinstance(reply, ToolMessage) and reply.status == "error"
    assert "Input should be 'all', 'nature', 'culture' or 'landmarks'" in reply.content
    assert fake_apis.requests == []


# Formatting edge cases (no network at all).


def test_truncated_results_say_how_many_were_found():
    report = make_report(
        attraction("Ravana Cave", "archaeological site", 1.71),
        attraction("Ella Viharaya", "temple", 2.04),
        total=23,
    )
    text = format_attractions_report(report)
    assert "Showing the 2 nearest of 23 found." in text
    assert "- Ravana Cave (archaeological site), 1.7 km" in text


def test_complete_results_have_no_showing_line():
    text = format_attractions_report(make_report(attraction("A", "museum", 1.0)))
    assert "Showing" not in text


def test_output_stays_one_line_per_attraction_plus_three():
    many = [attraction(f"Place {i}", "viewpoint", i / 2) for i in range(8)]
    text = format_attractions_report(make_report(*many, total=40))
    # header + 8 items + "Showing..." + "straight-line" + "Source"
    assert len(text.splitlines()) == 1 + 8 + 3


def test_no_results_for_a_category_suggests_all_and_forbids_inventing():
    text = format_attractions_report(make_report(category="nature"))
    assert text.startswith("No named attractions found in OpenStreetMap within 5 km")
    assert "category: nature" in text
    assert "Try again with category 'all'" in text
    assert "do not invent attractions" in text
    assert text.splitlines()[-1].startswith("Source: OpenStreetMap")


def test_no_results_for_all_does_not_suggest_all_again():
    text = format_attractions_report(make_report(category="all"))
    assert "Try again" not in text
    assert "do not invent attractions" in text


def test_tool_reports_empty_overpass_result_as_none_found(monkeypatch):
    api = FakeAPIs(overpass={"elements": []})
    monkeypatch.setattr(attractions_service, "make_client", api.client)
    text = find_attractions.invoke({"destination": "Ella", "category": "culture"})
    assert text.startswith("No named attractions found")


# Failures come back as text the LLM can act on, never as exceptions.


@pytest.mark.parametrize(
    ("handler", "expected"),
    [
        (raising(httpx.ReadTimeout("slow")), "did not respond within 30"),
        (raising(httpx.ConnectError("offline")), "Could not connect"),
        (respond(429), "busy (HTTP 429)"),
        (respond(200, text="not json"), "not JSON"),
    ],
    ids=["timeout", "offline", "busy", "invalid"],
)
def test_tool_reports_service_failures_instead_of_raising(
    monkeypatch, handler, expected
):
    overpass_fails_with(monkeypatch, handler)
    text = find_attractions.invoke({"destination": "Ella"})
    assert text.startswith("Attractions unavailable for 'Ella':")
    assert expected in text
    assert "do not invent attractions" in text


# --- 5b. Wikivoyage as the final fallback -----------------------------------
# When every Overpass server fails, find_attractions returns the sights named
# in the place's Wikivoyage guide instead, clearly labelled as a fallback.

ELLA_SEE_AND_DO = DestinationGuide(
    title="Ella",
    url="https://en.wikivoyage.org/wiki/Ella",
    topic="see_and_do",
    sections=(
        GuideSection("See", "1 Little Adam's Peak. An easy hike.", False),
        GuideSection("Do", "Visit a tea garden. Rs 3,000 per person.", False),
    ),
    missing=(),
)


def guide_returns(monkeypatch, result) -> list[tuple[str, str]]:
    """Replace the guide service; record each (destination, topic) asked for."""
    calls: list[tuple[str, str]] = []

    def fake_guide(destination, topic="all", **kwargs):
        calls.append((destination, topic))
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(guide_service, "get_destination_guide", fake_guide)
    return calls


def test_overpass_outage_falls_back_to_the_wikivoyage_sights(monkeypatch):
    overpass_fails_with(monkeypatch, respond(504))
    calls = guide_returns(monkeypatch, ELLA_SEE_AND_DO)

    text = find_attractions.invoke({"destination": "Ella", "category": "nature"})

    assert calls == [("Ella", "see_and_do")]
    lines = text.splitlines()
    # The first line still reports the OpenStreetMap failure...
    assert lines[0].startswith("Attractions unavailable for 'Ella':")
    assert "busy (HTTP 504)" in lines[0]
    # ...then the fallback, labelled, with the guide's own source line.
    assert lines[1].startswith("FALLBACK: sights named in the Wikivoyage guide")
    assert "[See] 1 Little Adam's Peak. An easy hike." in text
    assert "Source: Wikivoyage" in text
    assert text.rstrip().endswith("do not invent attractions or distances.")


def test_fallback_has_no_distances_or_openstreetmap_source(monkeypatch):
    overpass_fails_with(monkeypatch, respond(502))
    guide_returns(monkeypatch, ELLA_SEE_AND_DO)
    text = find_attractions.invoke({"destination": "Ella"})
    assert ATTRACTIONS_SOURCE not in text
    assert not re.search(r"\d km\b", text)


def test_when_the_guide_also_fails_the_tool_says_so(monkeypatch):
    overpass_fails_with(monkeypatch, respond(504))
    guide_returns(monkeypatch, GuideNetworkError("Could not connect to Wikivoyage."))

    text = find_attractions.invoke({"destination": "Ella"})

    assert text.startswith("Attractions unavailable for 'Ella':")
    assert "The Wikivoyage fallback also failed: Could not connect" in text
    assert "do not invent attractions" in text
    assert "FALLBACK" not in text


def test_a_guide_without_sights_is_not_presented_as_a_fallback(monkeypatch):
    overpass_fails_with(monkeypatch, respond(504))
    empty = DestinationGuide("Ella", "https://x", "see_and_do", (), ("see_and_do",))
    guide_returns(monkeypatch, empty)

    text = find_attractions.invoke({"destination": "Ella"})

    assert "names no sights either" in text
    assert "FALLBACK" not in text


def test_unknown_place_does_not_try_the_guide(monkeypatch):
    api = FakeAPIs(geocoding={"generationtime_ms": 0.1})
    monkeypatch.setattr(attractions_service, "make_client", api.client)
    calls = guide_returns(monkeypatch, ELLA_SEE_AND_DO)
    find_attractions.invoke({"destination": "Atlantis"})
    assert calls == []


def test_openstreetmap_success_does_not_ask_the_guide(monkeypatch, fake_apis):
    calls = guide_returns(monkeypatch, ELLA_SEE_AND_DO)
    text = find_attractions.invoke({"destination": "Ella"})
    assert calls == []
    assert ATTRACTIONS_SOURCE in text


@pytest.mark.parametrize("place", ["Kandy", "Galle", "Nuwara Eliya", "Sigiriya"])
def test_fallback_asks_the_guide_for_the_place_requested(monkeypatch, place):
    # Generic, not Ella-specific: the guide is asked for whatever place failed.
    overpass_fails_with(monkeypatch, respond(504))
    calls = guide_returns(
        monkeypatch,
        DestinationGuide(
            place,
            "https://x",
            "see_and_do",
            (GuideSection("See", "A sight.", False),),
            (),
        ),
    )
    text = find_attractions.invoke({"destination": place})
    assert calls == [(place, "see_and_do")]
    assert f"Wikivoyage guide for {place} (see and do):" in text


def test_the_offline_network_guard_blocks_real_requests():
    # conftest.no_real_network: offline tests can never reach the internet.
    with pytest.raises(httpx.ConnectError, match="offline test"):
        httpx.get("https://example.com")


def test_tool_explains_an_unknown_destination(monkeypatch):
    api = FakeAPIs(geocoding={"generationtime_ms": 0.1})
    monkeypatch.setattr(attractions_service, "make_client", api.client)
    text = find_attractions.invoke({"destination": "Atlantis"})
    assert text.startswith("Attractions unavailable for 'Atlantis'")
    assert "do not invent attractions" in text
    assert len(api.requests) == 1  # Overpass was never asked


def test_programming_errors_are_not_hidden(monkeypatch):
    # Only expected failures become text; a real bug must still surface.
    def broken(*args, **kwargs):
        raise TypeError("bug")

    monkeypatch.setattr(attractions_service, "get_attractions_report", broken)
    with pytest.raises(TypeError):
        find_attractions.invoke({"destination": "Ella"})


# --- 6. The agent using find_attractions (scripted model, offline) ------------


def test_agent_binds_the_attractions_tool():
    # Membership only: the exact tool list is checked once, in test_tool_calling.
    model = ToolCallingFake(messages=iter([]))
    build_tool_agent_graph(model)
    assert find_attractions in model.bound_tools


def test_agent_trajectory_search_then_attractions_then_answer(fake_apis):
    # The chain SYSTEM_PROMPT describes: choose a place with search_destinations,
    # then call find_attractions for the ONE best place (not every result), then
    # answer. The scripted model follows it; this checks our wiring, not Gemini.
    assert "for the one place that fits best, not for every result" in SYSTEM_PROMPT
    model = ToolCallingFake(
        messages=iter(
            [
                tool_call("search_destinations", {"query": "hiking"}, "call_s1"),
                tool_call(
                    "find_attractions",
                    {"destination": "Ella", "category": "nature"},
                    "call_a1",
                ),
                AIMessage("Go to Ella: Little Adam's Peak is 1.8 km away."),
            ]
        )
    )
    graph = build_tool_agent_graph(model)

    messages = run_tool_agent(graph, "Where can I hike, and what is there to see?")

    # Trajectory: Human -> AI -> Tool -> AI -> Tool -> AI (final answer)
    assert [type(m) for m in messages] == [
        HumanMessage,
        AIMessage,
        ToolMessage,
        AIMessage,
        ToolMessage,
        AIMessage,
    ]
    search_result, attractions_result = messages[2], messages[4]
    assert (search_result.name, search_result.tool_call_id) == (
        "search_destinations",
        "call_s1",
    )
    assert (attractions_result.name, attractions_result.tool_call_id) == (
        "find_attractions",
        "call_a1",
    )
    assert search_result.status == attractions_result.status == "success"
    assert "Ella" in search_result.content
    assert attractions_result.content.startswith("Attractions near Ella Town")
    assert "category: nature" in attractions_result.content

    # Only find_attractions used the (fake) network: geocode, then Overpass.
    assert [r.method for r in fake_apis.requests] == ["GET", "POST"]
    assert '"natural"~' in fake_apis.overpass_query()

    # The LLM ran three times, each time with the system prompt, and each later
    # call ended with the newest tool result (Observe).
    assert len(model.calls) == 3
    assert all(call[0].content == SYSTEM_PROMPT for call in model.calls)
    assert model.calls[1][-1].content == search_result.content
    assert model.calls[2][-1].content == attractions_result.content

    assert describe_steps(messages) == [
        "[human] Where can I hike, and what is there to see?",
        "[llm -> tool call] search_destinations({'query': 'hiking'})",
        f"[tool result] {search_result.content.splitlines()[0]} ...",
        "[llm -> tool call] find_attractions("
        "{'destination': 'Ella', 'category': 'nature'})",
        f"[tool result] {attractions_result.content.splitlines()[0]} ...",
        "[llm -> final answer]",
    ]
    assert final_answer(messages).startswith("Go to Ella")


def test_agent_survives_an_overpass_outage(monkeypatch):
    overpass_fails_with(monkeypatch, respond(504))
    model = ToolCallingFake(
        messages=iter(
            [
                tool_call("find_attractions", {"destination": "Ella"}, "call_a1"),
                AIMessage("Sorry, I could not look up attractions right now."),
            ]
        )
    )

    messages = run_tool_agent(build_tool_agent_graph(model), "What to see in Ella?")

    # The failure is an ordinary tool result (status "success": the tool ran and
    # returned text), unlike a schema error, which ToolNode marks "error".
    tool_message = messages[2]
    assert tool_message.status == "success"
    assert tool_message.content.startswith("Attractions unavailable for 'Ella'")
    assert "busy (HTTP 504)" in tool_message.content
    assert final_answer(messages).startswith("Sorry")


# --- 7. Live (real APIs, opt-in) ----------------------------------------------


@pytest.mark.live
def test_live_attractions_near_ella():
    report = get_attractions_report("Ella")

    assert report.attractions, "expected at least one attraction near Ella"
    assert all(a.distance_km <= 5.5 for a in report.attractions)


# Stage 4 live validation. Policy: an overloaded or slow Overpass server (HTTP
# 429/502/503/504 or a "runtime error" remark, or no answer within
# REQUEST_TIMEOUT) says nothing about our
# code, so those runs are SKIPPED as inconclusive instead of failing. Unexpected
# responses and bugs in our code still fail.

INCONCLUSIVE = "INCONCLUSIVE (infrastructure, not a code failure)"
NATURE_KINDS = {"waterfall", "peak", "viewpoint"}
CULTURE_KINDS = {
    "museum",
    "temple",
    "church",
    "mosque",
    "place of worship",
    "archaeological site",
    "monument",
    "ruins",
    "castle",
    "fort",
}
# One result line of format_attractions_report: "- <name> (<kind>), 1.2 km".
# The kind is the LAST bracket, so names that contain brackets still parse.
RESULT_LINE = re.compile(r"^- (?P<name>.+) \((?P<kind>[^()]+)\), [\d.]+ km$", re.M)


def _overpass_temporarily_unavailable(tool_text: str) -> bool:
    """Whether a find_attractions result reports a busy or slow Overpass server.

    Works on the tool's text and on a service exception's message alike.
    """
    return bool(_OVERPASS_TRANSIENT.search(tool_text))


# Messages of AttractionsBusyError (429/504, runtime-error remark),
# AttractionsTimeoutError, and AttractionsHTTPError for 502/503.
_OVERPASS_TRANSIENT = re.compile(
    r"Overpass API is busy \(HTTP"
    r"|Overpass API could not finish the search"
    r"|Overpass API did not respond within"
    r"|Overpass API returned HTTP 50[23]\."
)


# The two helpers above decide pass, fail or skip in the live tests, so they get
# offline checks of their own.


def test_result_line_reads_name_and_kind_even_with_brackets_in_the_name():
    text = format_attractions_report(
        make_report(
            attraction("Ravana Falls (Ella)", "waterfall", 5.1),
            attraction("Ella Gap", "viewpoint", 0.6),
        )
    )
    matches = [(m["name"], m["kind"]) for m in RESULT_LINE.finditer(text)]
    assert matches == [
        ("Ravana Falls (Ella)", "waterfall"),
        ("Ella Gap", "viewpoint"),
    ]


@pytest.mark.parametrize(
    "handler",
    [
        respond(429),
        respond(504),
        respond(200, json={"elements": [], "remark": "runtime error: timed out"}),
        raising(httpx.ReadTimeout("slow")),
        respond(502),
        respond(503),
    ],
    ids=["429", "504", "runtime-error-remark", "client-timeout", "502", "503"],
)
def test_busy_overpass_results_are_recognised(monkeypatch, handler):
    overpass_fails_with(monkeypatch, handler)
    assert _overpass_temporarily_unavailable(
        find_attractions.invoke({"destination": "Ella"})
    )


@pytest.mark.parametrize(
    "handler",
    [
        respond(400),
        respond(500),
        respond(200, text="not json"),
        raising(httpx.ConnectError("refused")),
    ],
    ids=["400", "500", "not-json", "connect-error"],
)
def test_other_overpass_failures_are_not_called_busy(monkeypatch, handler):
    overpass_fails_with(monkeypatch, handler)
    text = find_attractions.invoke({"destination": "Ella"})
    assert text.startswith("Attractions unavailable")
    assert not _overpass_temporarily_unavailable(text)


@pytest.mark.live
def test_live_find_attractions_tool_nature_near_ella():
    """O2: the tool end to end, no Gemini. 1 geocoding + 1 Overpass request."""
    text = find_attractions.invoke({"destination": "Ella", "category": "nature"})

    if _overpass_temporarily_unavailable(text):
        pytest.skip(f"{INCONCLUSIVE}: {text.splitlines()[0]}")
    assert text.startswith("Attractions near"), text
    assert "category: nature" in text
    kinds = [match["kind"] for match in RESULT_LINE.finditer(text)]
    assert kinds, f"expected at least one nature attraction near Ella:\n{text}"
    assert set(kinds) <= NATURE_KINDS, text
    assert "Distances are straight-line" in text
    assert ATTRACTIONS_SOURCE in text


@pytest.mark.live
def test_live_culture_attractions_near_kandy():
    """O3: the service, no Gemini. 1 geocoding + 1 Overpass request."""
    try:
        report = get_attractions_report("Kandy", "culture")
    except AttractionsServiceError as exc:
        if _overpass_temporarily_unavailable(str(exc)):
            pytest.skip(f"{INCONCLUSIVE}: {exc}")
        raise

    assert report.location.country == "Sri Lanka"
    assert report.attractions, "expected at least one culture attraction in Kandy"
    assert {a.kind for a in report.attractions} <= CULTURE_KINDS, report.attractions
    distances = [a.distance_km for a in report.attractions]
    assert distances == sorted(distances)
    assert len(report.attractions) <= attractions_service.DEFAULT_LIMIT
    assert report.total_found >= len(report.attractions)


@pytest.mark.live
def test_live_unknown_place_fails_before_overpass():
    """O4: 1 geocoding request and NO Overpass request.

    If this ever fails because the geocoder fuzzy-matched the made-up name to a
    real place, pick another nonsense name; the code is not at fault.
    """
    sent: list[str] = []
    with make_client() as client:
        client.event_hooks = {"request": [lambda r: sent.append(str(r.url))]}
        with pytest.raises(AttractionsLocationNotFoundError):
            get_attractions_report("Xyzzyqq Vrrnk", client=client)

    assert len(sent) == 1, sent
    assert sent[0].startswith(GEOCODING_URL)
    assert not any(url.startswith(OVERPASS_URL) for url in sent)
