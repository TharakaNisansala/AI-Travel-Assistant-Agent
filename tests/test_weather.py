"""Stage 4 checks: the Open-Meteo weather service, its tool, and the agent using it.

The offline tests never touch the internet. httpx.MockTransport stands in for
the network: our handler receives each request and returns a prepared response
(or raises a timeout or connection error). The sample JSON below is trimmed
from real Open-Meteo responses for Ella, Sri Lanka.

The live test calls the real Open-Meteo API (2 HTTP requests, no Gemini) and
runs only with:
    uv run pytest -m live tests/test_weather.py
"""

from __future__ import annotations

import httpx
import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ai_travel_assistant import weather_service
from ai_travel_assistant.tool_calling import (
    build_tool_agent_graph,
    describe_steps,
    final_answer,
    run_tool_agent,
)
from ai_travel_assistant.tools import (
    format_weather_report,
    get_weather_forecast,
)
from ai_travel_assistant.weather_service import (
    FORECAST_URL,
    GEOCODING_URL,
    InvalidWeatherResponseError,
    Location,
    LocationNotFoundError,
    WeatherHTTPError,
    WeatherNetworkError,
    WeatherServiceError,
    WeatherTimeoutError,
    fetch_forecast,
    geocode,
    get_weather_report,
    make_client,
    parse_forecast,
)

# --- Sample API responses -----------------------------------------------------

ELLA_GEOCODING = {
    "results": [
        {
            "id": 1247148,
            "name": "Ella Town",
            "latitude": 6.8756,
            "longitude": 81.0463,
            "elevation": 1031.0,
            "country_code": "LK",
            "timezone": "Asia/Colombo",
            "country": "Sri Lanka",
            "admin1": "Uva Province",
            "admin2": "Badulla District",
        }
    ],
    "generationtime_ms": 0.16,
}

NO_RESULTS = {"generationtime_ms": 0.1}  # Open-Meteo omits "results" on no match

ELLA_FORECAST = {
    "latitude": 6.85,
    "longitude": 81.05,
    "timezone": "Asia/Colombo",
    "current_units": {"temperature_2m": "°C", "wind_speed_10m": "km/h"},
    "current": {
        "time": "2026-11-12T19:00",
        "interval": 900,
        "temperature_2m": 19.1,
        "weather_code": 51,
        "wind_speed_10m": 14.4,
        "precipitation": 0.1,
    },
    "daily": {
        "time": ["2026-11-12", "2026-11-13", "2026-11-14"],
        "weather_code": [80, 51, 51],
        "temperature_2m_max": [23.1, 24.6, 23.7],
        "temperature_2m_min": [18.2, 17.6, 17.0],
        "precipitation_probability_max": [100, 47, 89],
        "precipitation_sum": [15.4, 0.2, 0.7],
        "wind_speed_10m_max": [14.8, 14.3, 8.4],
    },
}

ELLA = Location(
    name="Ella Town",
    latitude=6.8756,
    longitude=81.0463,
    country="Sri Lanka",
    admin1="Uva Province",
    admin2="Badulla District",
    elevation=1031.0,
)


# --- Fake network -------------------------------------------------------------


class FakeOpenMeteo:
    """Answers geocoding and forecast requests; records every request."""

    def __init__(self, geocoding=ELLA_GEOCODING, forecast=ELLA_FORECAST):
        self.geocoding = geocoding
        self.forecast = forecast
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        url = str(request.url.copy_with(query=None))
        body = self.geocoding if url == GEOCODING_URL else self.forecast
        if isinstance(body, httpx.Response):
            return body
        return httpx.Response(200, json=body)

    def client(self) -> httpx.Client:
        return make_client(transport=httpx.MockTransport(self))


def client_for(handler) -> httpx.Client:
    return make_client(transport=httpx.MockTransport(handler))


def raising(exc: Exception):
    def handler(request: httpx.Request) -> httpx.Response:
        raise exc

    return handler


@pytest.fixture
def fake_api(monkeypatch) -> FakeOpenMeteo:
    """Route the tool's own HTTP client to a FakeOpenMeteo."""
    api = FakeOpenMeteo()
    monkeypatch.setattr(weather_service, "make_client", api.client)
    return api


# --- 1. Successful location search --------------------------------------------


def test_geocode_returns_the_best_match():
    api = FakeOpenMeteo()
    location = geocode("Ella", client=api.client())

    assert location == ELLA
    assert location.label == "Ella Town (Badulla District, Uva Province, Sri Lanka)"


def test_geocode_sends_name_and_country_filter():
    api = FakeOpenMeteo()
    geocode("  Ella ", "lk", client=api.client())

    params = api.requests[0].url.params
    assert params["name"] == "Ella"  # trimmed
    assert params["countryCode"] == "LK"  # upper-cased
    assert params["count"] == "1"
    assert api.requests[0].headers["User-Agent"] == weather_service.USER_AGENT


def test_geocode_without_country_searches_worldwide():
    api = FakeOpenMeteo()
    geocode("Paris", "", client=api.client())
    assert "countryCode" not in api.requests[0].url.params


# --- 2. Successful weather forecast retrieval ---------------------------------


def test_fetch_forecast_asks_for_current_daily_and_local_timezone():
    api = FakeOpenMeteo()
    fetch_forecast(ELLA, client=api.client())

    params = api.requests[0].url.params
    assert params["latitude"] == "6.8756" and params["longitude"] == "81.0463"
    assert params["timezone"] == "auto"
    assert params["forecast_days"] == "3"
    assert "temperature_2m" in params["current"]
    for field in ("temperature_2m_max", "precipitation_probability_max"):
        assert field in params["daily"]


def test_forecast_json_becomes_typed_values():
    report = fetch_forecast(ELLA, client=FakeOpenMeteo().client())

    assert report.timezone == "Asia/Colombo"
    assert report.current.temperature_c == 19.1
    assert report.current.conditions == "light drizzle"
    assert len(report.daily) == 3
    first = report.daily[0]
    assert first.date == "2026-11-12"
    assert first.conditions == "slight rain showers"
    assert (first.temperature_min_c, first.temperature_max_c) == (18.2, 23.1)
    assert first.precipitation_probability_pct == 100
    assert first.precipitation_sum_mm == 15.4
    assert first.wind_speed_max_kmh == 14.8


def test_get_weather_report_geocodes_then_forecasts():
    api = FakeOpenMeteo()
    report = get_weather_report("Ella", client=api.client())

    assert [str(r.url.copy_with(query=None)) for r in api.requests] == [
        GEOCODING_URL,
        FORECAST_URL,
    ]
    # The forecast uses the coordinates the geocoder found.
    assert api.requests[1].url.params["latitude"] == "6.8756"
    assert report.location.name == "Ella Town"


def test_formatted_report_has_units_and_no_raw_json():
    text = format_weather_report(
        get_weather_report("Ella", client=FakeOpenMeteo().client())
    )

    assert text.startswith("Weather forecast for Ella Town (Badulla District")
    assert "Current conditions (2026-11-12 19:00): 19.1 °C, light drizzle" in text
    assert (
        "- 2026-11-13 (Fri): light drizzle; min 17.6 °C, max 24.6 °C; "
        "chance of precipitation 47%; rainfall 0.2 mm; max wind 14.3 km/h."
    ) in text
    assert "Source: Open-Meteo" in text
    # Unneeded API fields never reach the LLM.
    assert "generationtime" not in text and "interval" not in text
    assert "n/a" not in text


def test_tool_has_name_description_and_schema():
    assert get_weather_forecast.name == "get_weather_forecast"
    assert "weather" in get_weather_forecast.description.lower()
    assert set(get_weather_forecast.args) == {"destination", "country_code"}


def test_tool_returns_the_formatted_forecast(fake_api):
    text = get_weather_forecast.invoke({"destination": "Ella"})
    assert text.startswith("Weather forecast for Ella Town")
    assert fake_api.requests[0].url.params["countryCode"] == "LK"  # default


# --- Missing or incomplete data -----------------------------------------------


def test_null_values_become_none_and_show_as_na():
    forecast = {
        "timezone": "Asia/Colombo",
        "daily": {
            "time": ["2026-11-12", "2026-11-13"],
            "weather_code": [80, None],
            "temperature_2m_max": [23.1],  # shorter than "time"
            # temperature_2m_min and the rest missing entirely
        },
    }  # and no "current" block
    report = parse_forecast(forecast, ELLA)
    assert report.current is None
    assert report.daily[1].weather_code is None
    assert report.daily[1].temperature_max_c is None
    assert report.daily[0].temperature_min_c is None

    text = format_weather_report(report)
    assert "Current conditions: not available." in text
    assert "max 23.1 °C" in text and "min n/a" in text
    assert "were not provided by the weather service" in text


@pytest.mark.parametrize(
    "forecast",
    [
        {"current": ELLA_FORECAST["current"]},  # no daily block
        {"daily": {"time": []}},  # empty daily block
        {"daily": "oops"},  # wrong type
    ],
)
def test_missing_daily_forecast_is_an_invalid_response(forecast):
    with pytest.raises(InvalidWeatherResponseError, match="no daily forecast"):
        parse_forecast(forecast, ELLA)


def test_geocoding_result_without_coordinates_is_invalid():
    api = FakeOpenMeteo(geocoding={"results": [{"name": "Ella"}]})
    with pytest.raises(InvalidWeatherResponseError, match="no coordinates"):
        geocode("Ella", client=api.client())


# --- 3. Unknown destination ---------------------------------------------------


def test_unknown_destination_raises_location_not_found():
    api = FakeOpenMeteo(geocoding=NO_RESULTS)
    with pytest.raises(LocationNotFoundError, match="'Atlantis'.*'LK'"):
        get_weather_report("Atlantis", client=api.client())
    assert len(api.requests) == 1  # no forecast request without coordinates


def test_empty_destination_needs_no_request():
    api = FakeOpenMeteo()
    with pytest.raises(LocationNotFoundError):
        geocode("   ", client=api.client())
    assert api.requests == []


def test_tool_explains_an_unknown_destination(monkeypatch):
    monkeypatch.setattr(
        weather_service, "make_client", FakeOpenMeteo(geocoding=NO_RESULTS).client
    )
    text = get_weather_forecast.invoke({"destination": "Atlantis"})
    assert text.startswith("Weather unavailable for 'Atlantis'")
    assert "No place called 'Atlantis' was found" in text
    assert "do not guess the weather" in text


# --- 4. API timeout -----------------------------------------------------------


def test_timeout_raises_weather_timeout_error():
    client = client_for(raising(httpx.ReadTimeout("timed out")))
    with pytest.raises(WeatherTimeoutError, match="did not respond within 10"):
        get_weather_report("Ella", client=client)


def test_tool_reports_a_timeout_instead_of_raising(monkeypatch):
    handler = raising(httpx.ConnectTimeout("timed out"))
    monkeypatch.setattr(weather_service, "make_client", lambda: client_for(handler))
    text = get_weather_forecast.invoke({"destination": "Ella"})
    assert text.startswith("Weather unavailable for 'Ella'")
    assert "did not respond" in text


# --- 5. Network failure -------------------------------------------------------


def test_connection_error_raises_weather_network_error():
    client = client_for(raising(httpx.ConnectError("Name or service not known")))
    with pytest.raises(WeatherNetworkError, match="Could not connect.*ConnectError"):
        get_weather_report("Ella", client=client)


def test_tool_reports_a_network_failure_instead_of_raising(monkeypatch):
    handler = raising(httpx.ConnectError("offline"))
    monkeypatch.setattr(weather_service, "make_client", lambda: client_for(handler))
    text = get_weather_forecast.invoke({"destination": "Ella"})
    assert "Could not connect" in text and "do not guess" in text


# --- 6. Invalid API responses and HTTP errors ---------------------------------


def test_http_error_includes_open_meteo_reason():
    bad = httpx.Response(400, json={"error": True, "reason": "Latitude out of range"})
    api = FakeOpenMeteo(forecast=bad)
    with pytest.raises(WeatherHTTPError, match="HTTP 400: Latitude out of range"):
        get_weather_report("Ella", client=api.client())


def test_server_error_without_json_body():
    api = FakeOpenMeteo(geocoding=httpx.Response(503, text="Service Unavailable"))
    with pytest.raises(WeatherHTTPError, match="geocoding API returned HTTP 503"):
        get_weather_report("Ella", client=api.client())


def test_non_json_body_is_an_invalid_response():
    api = FakeOpenMeteo(forecast=httpx.Response(200, text="<html>oops</html>"))
    with pytest.raises(InvalidWeatherResponseError, match="not JSON"):
        get_weather_report("Ella", client=api.client())


def test_json_that_is_not_an_object_is_an_invalid_response():
    api = FakeOpenMeteo(geocoding=httpx.Response(200, json=[1, 2, 3]))
    with pytest.raises(InvalidWeatherResponseError, match="not an object"):
        get_weather_report("Ella", client=api.client())


@pytest.mark.parametrize(
    "exc",
    [
        httpx.DecodingError("corrupt gzip body"),
        httpx.TooManyRedirects("redirect loop"),
    ],
)
def test_other_httpx_errors_become_weather_errors(exc):
    # These are not TransportErrors, but must still become WeatherServiceErrors
    # so the tool can report them instead of crashing the agent.
    with pytest.raises(InvalidWeatherResponseError, match="could not be read"):
        get_weather_report("Ella", client=client_for(raising(exc)))


def test_nan_and_infinity_are_treated_as_missing():
    raw = (
        '{"current": {"time": "2026-11-12T19:00", "temperature_2m": NaN,'
        ' "weather_code": NaN},'
        ' "daily": {"time": ["2026-11-12"], "temperature_2m_max": [Infinity]}}'
    )
    api = FakeOpenMeteo(forecast=httpx.Response(200, text=raw))
    report = get_weather_report("Ella", client=api.client())

    assert report.current.temperature_c is None
    assert report.current.weather_code is None
    assert report.daily[0].temperature_max_c is None
    text = format_weather_report(report)
    assert "nan °C" not in text and "inf °C" not in text
    assert "Current conditions (2026-11-12 19:00): n/a, conditions n/a" in text
    assert "max n/a" in text


def test_every_failure_shares_one_base_class():
    # The tool catches WeatherServiceError, so every failure must be one.
    for error in (
        LocationNotFoundError,
        WeatherTimeoutError,
        WeatherNetworkError,
        WeatherHTTPError,
        InvalidWeatherResponseError,
    ):
        assert issubclass(error, WeatherServiceError)


def test_tool_reports_an_invalid_response_instead_of_raising(monkeypatch):
    api = FakeOpenMeteo(forecast={"daily": {}})
    monkeypatch.setattr(weather_service, "make_client", api.client)
    text = get_weather_forecast.invoke({"destination": "Ella"})
    assert text.startswith("Weather unavailable for 'Ella'")
    assert "no daily forecast" in text


# --- 7. The agent invoking the weather tool -----------------------------------


class ScriptedToolModel(GenericFakeChatModel):
    """Returns prepared AIMessages in order and records what it was sent."""

    bound_tools: list = []
    calls: list = []

    def bind_tools(self, tools, **kwargs):
        self.bound_tools = list(tools)
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def weather_call(destination: str = "Ella", call_id: str = "call_w1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "get_weather_forecast",
                "args": {"destination": destination},
                "id": call_id,
            }
        ],
    )


def test_agent_binds_the_weather_tool():
    # Membership only: the exact tool list is checked once, in test_tool_calling.
    model = ScriptedToolModel(messages=iter([]))
    build_tool_agent_graph(model)
    assert get_weather_forecast in model.bound_tools


def test_agent_calls_the_weather_tool_and_answers_from_its_result(fake_api):
    model = ScriptedToolModel(
        messages=iter(
            [
                weather_call("Ella"),
                AIMessage("Expect showers in Ella today, 18-23 °C."),
            ]
        )
    )
    graph = build_tool_agent_graph(model)

    messages = run_tool_agent(graph, "What is the weather forecast for Ella?")

    # Human -> AI (tool call) -> Tool (real tool code, fake network) -> AI
    assert [type(m) for m in messages] == [
        HumanMessage,
        AIMessage,
        ToolMessage,
        AIMessage,
    ]
    tool_message = messages[2]
    assert tool_message.name == "get_weather_forecast"
    assert tool_message.tool_call_id == "call_w1"
    assert "Weather forecast for Ella Town" in tool_message.content
    # The tool really made the two HTTP requests.
    assert len(fake_api.requests) == 2
    # Observe: the LLM's second call ends with the forecast.
    assert model.calls[1][-1].content == tool_message.content
    assert final_answer(messages).startswith("Expect showers")
    assert describe_steps(messages)[1] == (
        "[llm -> tool call] get_weather_forecast({'destination': 'Ella'})"
    )


def test_agent_survives_a_weather_outage(monkeypatch):
    handler = raising(httpx.ConnectError("offline"))
    monkeypatch.setattr(weather_service, "make_client", lambda: client_for(handler))
    model = ScriptedToolModel(
        messages=iter(
            [weather_call("Ella"), AIMessage("Sorry, the forecast is unavailable.")]
        )
    )

    messages = run_tool_agent(build_tool_agent_graph(model), "Weather in Ella?")

    # The failure reaches the LLM as an ordinary tool result; nothing crashes.
    assert messages[2].content.startswith("Weather unavailable for 'Ella'")
    assert final_answer(messages) == "Sorry, the forecast is unavailable."


# --- Live (real Open-Meteo, skipped by default) -------------------------------


@pytest.mark.live
def test_live_open_meteo_forecast_for_ella():
    report = get_weather_report("Ella")
    assert report.location.country == "Sri Lanka"
    assert report.timezone == "Asia/Colombo"
    assert len(report.daily) == 3
    assert "°C" in format_weather_report(report)
