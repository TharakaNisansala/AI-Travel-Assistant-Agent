"""Stage 7 checks: fault injection (FAULT_INJECT), all offline.

The injected failure is an httpx transport inside each service's own client,
so these tests run the real retry, error and "unavailable" code above it.
"""

from __future__ import annotations

import httpx
import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from ai_travel_assistant import (
    attractions_service,
    currency_service,
    faults,
    guide_service,
    tool_calling,
    travel_service,
    weather_service,
)
from ai_travel_assistant.tool_calling import (
    ask_agent,
    build_tool_agent_graph,
    final_answer,
)

# --- Parsing the switch ------------------------------------------------------------


@pytest.mark.parametrize("value", [None, "", "  ", ",,"])
def test_nothing_is_injected_by_default(value):
    assert faults.parse_faults(value) == {}


def test_parse_services_and_modes():
    assert faults.parse_faults(" Weather , guide:BUSY,travel:timeout ") == {
        "weather": "down",
        "guide": "busy",
        "travel": "timeout",
    }


@pytest.mark.parametrize("value", ["wether", "llm", "weather:slow"])
def test_typos_are_errors_not_silently_ignored(value):
    with pytest.raises(faults.FaultConfigError):
        faults.parse_faults(value)


def test_an_empty_mode_means_the_default():
    assert faults.parse_faults("weather:") == {"weather": "down"}


def test_the_switch_is_off_in_tests_unless_set(monkeypatch):
    assert faults.active_faults() == {}
    assert faults.describe_active_faults() is None
    for service in faults.SERVICES:
        assert faults.transport_for(service) is None
    monkeypatch.setenv("FAULT_INJECT", "guide:busy")
    assert faults.describe_active_faults() == (
        "FAULT INJECTION ACTIVE (FAULT_INJECT): guide (busy)"
    )


def test_transport_for_rejects_an_unknown_service():
    with pytest.raises(ValueError):
        faults.transport_for("llm")


# --- The services' clients ---------------------------------------------------------

SERVICES = {
    "weather": weather_service,
    "attractions": attractions_service,
    "guide": guide_service,
    "travel": travel_service,
    "currency": currency_service,
}


@pytest.mark.parametrize("name", list(SERVICES))
def test_each_client_fails_only_when_its_service_is_named(monkeypatch, name):
    request = httpx.Request("GET", "https://example.org")
    other = next(n for n in SERVICES if n != name)
    monkeypatch.setenv("FAULT_INJECT", other)
    with SERVICES[name].make_client() as client:
        assert not isinstance(client._transport, httpx.MockTransport)

    monkeypatch.setenv("FAULT_INJECT", name)
    with SERVICES[name].make_client() as client, pytest.raises(httpx.ConnectError):
        client.send(request)


def test_an_explicit_transport_wins_over_the_switch(monkeypatch):
    monkeypatch.setenv("FAULT_INJECT", "weather")
    ok = httpx.MockTransport(lambda request: httpx.Response(200, json={}))
    with weather_service.make_client(ok) as client:
        response = client.get("https://example.org")
    assert response.status_code == 200


@pytest.mark.parametrize(
    ("mode", "check"),
    [
        ("busy", lambda r: r.status_code == 503),
        ("down", None),
        ("timeout", None),
    ],
)
def test_modes(monkeypatch, mode, check):
    monkeypatch.setenv("FAULT_INJECT", f"currency:{mode}")
    with currency_service.make_client() as client:
        if check is not None:
            assert check(client.get("https://example.org"))
        else:
            expected = httpx.ReadTimeout if mode == "timeout" else httpx.ConnectError
            with pytest.raises(expected):
                client.get("https://example.org")


# --- The tools report "unavailable" through the real code path ----------------------

TOOL_CALLS = {
    "weather": (
        tool_calling.get_weather_forecast,
        {"destination": "Ella"},
        "Weather unavailable",
    ),
    "attractions": (
        tool_calling.find_attractions,
        {"destination": "Ella"},
        "Attractions unavailable",
    ),
    "guide": (
        tool_calling.get_destination_guide,
        {"destination": "Ella"},
        "Guide unavailable",
    ),
    "travel": (
        tool_calling.estimate_travel_time,
        {"origin": "Colombo", "destination": "Ella"},
        "Travel time unavailable",
    ),
    "currency": (
        tool_calling.convert_currency,
        {"amount": 10, "from_currency": "USD", "to_currency": "LKR"},
        "Currency conversion unavailable",
    ),
}


@pytest.mark.parametrize("name", list(TOOL_CALLS))
@pytest.mark.parametrize("mode", faults.MODES)
def test_each_tool_degrades_to_unavailable_text(monkeypatch, name, mode):
    monkeypatch.setenv("FAULT_INJECT", f"{name}:{mode}")
    tool, args, prefix = TOOL_CALLS[name]
    assert tool.invoke(args).startswith(prefix)


def test_a_busy_service_is_retried_before_it_gives_up(
    monkeypatch, fresh_caches_and_no_sleep
):
    monkeypatch.setenv("FAULT_INJECT", "currency:busy")
    text = tool_calling.convert_currency.invoke(
        {"amount": 10, "from_currency": "USD", "to_currency": "LKR"}
    )
    assert text.startswith("Currency conversion unavailable")
    assert fresh_caches_and_no_sleep  # the service waited before its retry


# --- The agent still answers with the weather API down -----------------------------


class Scripted(GenericFakeChatModel):
    calls: list = []

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def test_itinerary_is_still_written_with_the_weather_api_down(monkeypatch):
    monkeypatch.setenv("FAULT_INJECT", "weather")
    monkeypatch.setattr(
        tool_calling.get_destination_guide, "func", lambda **_: "Guide."
    )
    monkeypatch.setattr(tool_calling.find_attractions, "func", lambda **_: "Sights.")
    first = AIMessage(
        "",
        tool_calls=[
            {
                "name": "get_destination_guide",
                "args": {"destination": "Ella"},
                "id": "g",
            },
            {"name": "find_attractions", "args": {"destination": "Ella"}, "id": "a"},
            {
                "name": "get_weather_forecast",
                "args": {"destination": "Ella"},
                "id": "w",
            },
        ],
    )
    model = Scripted(
        messages=iter([first, AIMessage("Day 1 ... Not available: weather.")]), calls=[]
    )
    graph = build_tool_agent_graph(model, checkpointer=InMemorySaver())

    turn = ask_agent(graph, "t1", "Plan 2 days in Ella")

    results = {m.name: m.content for m in model.calls[1] if isinstance(m, ToolMessage)}
    assert results["get_destination_guide"] == "Guide."
    assert results["find_attractions"] == "Sights."
    assert results["get_weather_forecast"].startswith("Weather unavailable for 'Ella'")
    assert final_answer(turn) == "Day 1 ... Not available: weather."


# --- The CLI -----------------------------------------------------------------------


def test_cli_refuses_a_mistyped_switch(monkeypatch, capsys):
    monkeypatch.setenv("FAULT_INJECT", "wether")
    assert tool_calling.main(["Where to hike?"]) == 1
    err = capsys.readouterr().err
    assert "Configuration error: FAULT_INJECT: unknown service 'wether'" in err


def test_cli_warns_when_faults_are_on(monkeypatch, capsys):
    from ai_travel_assistant import llm_basics

    monkeypatch.setenv("FAULT_INJECT", "weather")
    settings = llm_basics.GeminiSettings(model="fake-model", api_key="fake")
    monkeypatch.setattr(llm_basics, "load_settings", lambda: settings)
    model = Scripted(messages=iter([AIMessage("Hi.")]), calls=[])
    monkeypatch.setattr(llm_basics, "build_llm", lambda s, **kw: model)

    assert tool_calling.main(["Hello"]) == 0
    assert "FAULT INJECTION ACTIVE (FAULT_INJECT): weather (down)" in (
        capsys.readouterr().err
    )
