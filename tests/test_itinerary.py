"""Stage 4 deliverable checks: multi-day itineraries for any Sri Lankan destination.

The agent must plan a grounded 3-day trip for Kandy, Galle, Nuwara Eliya,
Sigiriya and Ella using the SAME generic logic, with nothing written for one
place in particular. These offline tests check that from three sides:

  1. The prompt: the itinerary rule names no destination (only the default
     origin), and covers every data source the plan needs.
  2. The code: no source module compares anything with a destination name, so
     there is no hidden `if destination == "Ella"` branch anywhere.
  3. The graph: for each of the five places, a scripted model runs the full
     itinerary trajectory (4 tools in parallel, then travel time + currency,
     then the answer). Fake services record what they were asked for, so we
     can check every request was for that place, and that the results the
     model reads are about that place only. An OpenStreetMap outage takes the
     Wikivoyage fallback for every place, too.

No Gemini calls and no network (conftest blocks real HTTP). The live check of
the real model writing real itineraries is itinerary_check.py.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ai_travel_assistant import (
    attractions_service,
    currency_service,
    guide_service,
    itinerary_check,
    llm_basics,
    travel_service,
    weather_service,
)
from ai_travel_assistant.attractions_service import (
    Attraction,
    AttractionsBusyError,
    AttractionsReport,
)
from ai_travel_assistant.currency_service import Conversion
from ai_travel_assistant.guide_service import DestinationGuide, GuideSection
from ai_travel_assistant.itinerary_check import (
    build_question,
    check_itinerary,
    failed_tools,
    format_report,
    gemini_calls,
    used_guide_fallback,
)
from ai_travel_assistant.tool_calling import (
    DEFAULT_ORIGIN,
    SYSTEM_PROMPT,
    build_tool_agent_graph,
    final_answer,
    run_tool_agent,
)
from ai_travel_assistant.tools import DESTINATIONS
from ai_travel_assistant.travel_service import TravelEstimate
from ai_travel_assistant.weather_service import (
    CurrentWeather,
    DailyForecast,
    Location,
    WeatherReport,
)

PLACES = ("Kandy", "Galle", "Nuwara Eliya", "Sigiriya", "Ella")
SOURCE_DIR = Path(__file__).resolve().parents[1] / "src" / "ai_travel_assistant"

# The tools an itinerary needs, in the order of the first parallel step.
FIRST_STEP_TOOLS = (
    "get_destination_guide",
    "find_attractions",
    "get_weather_forecast",
    "estimate_travel_time",
)


# --- 1. The prompt is generic ------------------------------------------------


def test_prompt_names_no_destination():
    # Every place in the built-in catalogue, plus the five test destinations.
    names = {d.name for d in DESTINATIONS} | set(PLACES)
    named = sorted(n for n in names if n.lower() in SYSTEM_PROMPT.lower())
    assert named == [], f"SYSTEM_PROMPT mentions destinations: {named}"


def test_the_default_origin_is_stated_and_must_be_announced():
    assert DEFAULT_ORIGIN == "Colombo"
    assert f"use {DEFAULT_ORIGIN} and state in the plan that you assumed it" in (
        SYSTEM_PROMPT
    )


def test_itinerary_rule_covers_every_data_source():
    rule = SYSTEM_PROMPT[SYSTEM_PROMPT.index("asks for an itinerary") :]
    for tool_name in (*FIRST_STEP_TOOLS, "convert_currency"):
        assert tool_name in rule, f"{tool_name} missing from the itinerary rule"
    for source in ("OpenStreetMap", "Wikivoyage", "OSRM", "Open-Meteo"):
        assert source in rule
    assert "Not available" in rule
    assert "Wikivoyage fallback" in rule
    assert "only sights, travel times, weather and prices from the tool results" in (
        rule
    )


def test_itinerary_rule_caps_the_extra_calls():
    # The second step is bounded, so one itinerary stays at about 3 Gemini calls.
    assert "at most two nearby places" in SYSTEM_PROMPT
    assert "at most three prices" in SYSTEM_PROMPT


# --- 2. No destination-specific branches in the code -------------------------


def _string_constants(node: ast.AST) -> list[str]:
    """String literals directly inside a comparison operand (incl. tuples/sets)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return [s for element in node.elts for s in _string_constants(element)]
    return []


def destination_comparisons(source: str) -> list[str]:
    """Comparisons such as `x == "Ella"` or `x in ("Kandy", ...)` in a module."""
    places = {p.casefold() for p in PLACES} | {d.name.casefold() for d in DESTINATIONS}
    found = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Compare):
            continue
        for operand in (node.left, *node.comparators):
            for text in _string_constants(operand):
                if text.casefold() in places:
                    found.append(f"line {node.lineno}: {text!r}")
    return found


def test_the_checker_finds_a_hardcoded_destination_branch():
    # Prove the scan works before trusting its "nothing found".
    code = 'if destination == "Ella":\n    pass\nif d in ("Kandy", "x"):\n    pass\n'
    assert destination_comparisons(code) == ["line 1: 'Ella'", "line 3: 'Kandy'"]


@pytest.mark.parametrize(
    "module", sorted(SOURCE_DIR.glob("*.py")), ids=lambda path: path.name
)
def test_no_source_module_branches_on_a_destination(module):
    assert destination_comparisons(module.read_text(encoding="utf-8")) == []


# --- 3. The full itinerary trajectory for each place (scripted model) --------


class ScriptedModel(GenericFakeChatModel):
    """Returns scripted AIMessages in order; accepts bind_tools; records calls."""

    calls: list = []

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def location(name: str) -> Location:
    return Location(name, 7.0, 80.7, "Sri Lanka", "Central Province")


class FakeServices:
    """Stand-ins for the five service functions the tools call.

    Each records the arguments it received and returns a small, valid result
    that names the place it was asked about, so the tools' real formatting
    code runs and the test can see which place each result is about.
    """

    def __init__(self, monkeypatch, *, overpass_down: bool = False):
        self.requests: list[tuple[str, tuple]] = []
        self.overpass_down = overpass_down
        monkeypatch.setattr(weather_service, "get_weather_report", self.weather)
        monkeypatch.setattr(attractions_service, "get_attractions_report", self.sights)
        monkeypatch.setattr(guide_service, "get_destination_guide", self.guide)
        monkeypatch.setattr(travel_service, "get_travel_estimate", self.travel)
        monkeypatch.setattr(currency_service, "convert", self.convert)

    def weather(self, destination, country_code="LK", **kwargs):
        self.requests.append(("weather", (destination,)))
        days = tuple(
            DailyForecast(f"2026-11-{16 + i}", 61, 27.0, 19.0, 70.0, 2.0, 12.0)
            for i in range(3)
        )
        return WeatherReport(
            location(destination),
            "Asia/Colombo",
            CurrentWeather("2026-11-16T12:00", 25.0, 3, 10.0, 0.0),
            days,
        )

    def sights(self, destination, category="all", country_code="LK", **kwargs):
        self.requests.append(("attractions", (destination, category)))
        if self.overpass_down:
            raise AttractionsBusyError("The Overpass API is busy (HTTP 504).")
        sight = Attraction(f"{destination} Viewpoint", "viewpoint", 7.0, 80.7, 1.2)
        return AttractionsReport(location(destination), category, 5000, (sight,), 1)

    def guide(self, destination, topic="all", **kwargs):
        self.requests.append(("guide", (destination, topic)))
        return DestinationGuide(
            title=destination,
            url=f"https://en.wikivoyage.org/wiki/{destination}",
            topic=topic,
            sections=(
                GuideSection("See", f"1 {destination} Old Temple. Entry US$10.", False),
            ),
            missing=(),
        )

    def travel(self, origin, destination, country_code="LK", **kwargs):
        self.requests.append(("travel", (origin, destination)))
        return TravelEstimate(
            location(origin), location(destination), 120.0, 150.0, 95.0
        )

    def convert(self, amount, from_code, to_code, **kwargs):
        self.requests.append(("currency", (amount, from_code, to_code)))
        return Conversion(amount, from_code, to_code, 332.6, amount * 332.6, None)

    def asked_about(self, kind: str) -> list[tuple]:
        return [args for name, args in self.requests if name == kind]


def itinerary_script(place: str) -> list[AIMessage]:
    """What a good model does for "Plan 3 days in <place>", step by step."""
    first = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "get_destination_guide",
                "args": {"destination": place, "topic": "all"},
                "id": "c1",
            },
            {
                "name": "find_attractions",
                "args": {"destination": place, "category": "all"},
                "id": "c2",
            },
            {
                "name": "get_weather_forecast",
                "args": {"destination": place},
                "id": "c3",
            },
            {
                "name": "estimate_travel_time",
                "args": {"origin": DEFAULT_ORIGIN, "destination": place},
                "id": "c4",
            },
        ],
    )
    second = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "convert_currency",
                "args": {"amount": 10, "from_currency": "USD", "to_currency": "LKR"},
                "id": "c5",
            },
        ],
    )
    answer = AIMessage(
        f"Day 1 in {place}: ... (I assumed you start in {DEFAULT_ORIGIN}.) "
        "Not available: hotel prices."
    )
    return [first, second, answer]


def run_itinerary(place: str) -> tuple[list, ScriptedModel]:
    model = ScriptedModel(messages=iter(itinerary_script(place)), calls=[])
    messages = run_tool_agent(
        build_tool_agent_graph(model), f"Plan a 3-day trip to {place}."
    )
    return messages, model


@pytest.mark.parametrize("place", PLACES)
def test_every_service_is_asked_about_the_requested_place(monkeypatch, place):
    services = FakeServices(monkeypatch)
    run_itinerary(place)

    assert services.asked_about("guide") == [(place, "all")]
    assert services.asked_about("attractions") == [(place, "all")]
    assert services.asked_about("weather") == [(place,)]
    assert services.asked_about("travel") == [(DEFAULT_ORIGIN, place)]
    assert services.asked_about("currency") == [(10, "USD", "LKR")]


@pytest.mark.parametrize("place", PLACES)
def test_itinerary_trajectory_is_two_tool_steps_then_the_answer(monkeypatch, place):
    FakeServices(monkeypatch)
    messages, model = run_itinerary(place)

    assert [type(m).__name__ for m in messages] == [
        "HumanMessage",
        "AIMessage",
        *["ToolMessage"] * 4,  # the parallel first step
        "AIMessage",
        "ToolMessage",  # currency
        "AIMessage",  # the plan
    ]
    assert len(model.calls) == 3  # = 3 Gemini requests in real use
    assert all(m.status == "success" for m in messages if isinstance(m, ToolMessage))
    assert final_answer(messages).startswith(f"Day 1 in {place}")


@pytest.mark.parametrize("place", PLACES)
def test_the_model_reads_results_about_that_place_only(monkeypatch, place):
    FakeServices(monkeypatch)
    _, model = run_itinerary(place)

    # What the model saw just before writing the plan.
    results = [m.content for m in model.calls[-1] if isinstance(m, ToolMessage)]
    by_tool = dict(zip((*FIRST_STEP_TOOLS, "convert_currency"), results, strict=True))
    assert f"Wikivoyage guide for {place}" in by_tool["get_destination_guide"]
    assert f"{place} Viewpoint (viewpoint), 1.2 km" in by_tool["find_attractions"]
    assert f"Weather forecast for {place}" in by_tool["get_weather_forecast"]
    assert f"from {DEFAULT_ORIGIN}" in by_tool["estimate_travel_time"]
    assert "332.6" in by_tool["convert_currency"]

    others = [p for p in PLACES if p != place and p not in place]
    seen = "\n".join(results)
    assert not [p for p in others if p in seen], "results leaked another place"


@pytest.mark.parametrize("place", PLACES)
def test_an_openstreetmap_outage_uses_the_guide_for_every_place(monkeypatch, place):
    services = FakeServices(monkeypatch, overpass_down=True)
    _, model = run_itinerary(place)

    sights = next(
        m.content
        for m in model.calls[-1]
        if isinstance(m, ToolMessage) and m.name == "find_attractions"
    )
    assert sights.startswith(f"Attractions unavailable for '{place}'")
    assert "FALLBACK: sights named in the Wikivoyage guide" in sights
    assert f"1 {place} Old Temple" in sights
    # Guide asked twice: once by the model (all), once by the fallback. The
    # tools run in parallel, so the two requests may arrive in either order.
    assert sorted(services.asked_about("guide")) == [
        (place, "all"),
        (place, "see_and_do"),
    ]


def test_the_system_prompt_is_identical_for_every_place(monkeypatch):
    FakeServices(monkeypatch)
    prompts = set()
    for place in PLACES:
        _, model = run_itinerary(place)
        prompts |= {call[0].content for call in model.calls}
    assert prompts == {SYSTEM_PROMPT}


def test_the_request_reaches_the_model_unchanged(monkeypatch):
    FakeServices(monkeypatch)
    _, model = run_itinerary("Nuwara Eliya")
    first_call = model.calls[0]
    assert isinstance(first_call[1], HumanMessage)
    assert first_call[1].content == "Plan a 3-day trip to Nuwara Eliya."


# --- 4. The live checker's own logic (itinerary_check.py, offline) -----------


def test_question_wording_with_and_without_a_start():
    assert build_question("Kandy", 3, None) == "Plan a 3-day trip to Kandy."
    assert build_question("Galle", 2, "Kandy") == (
        "Plan a 2-day trip to Galle, starting from Kandy."
    )


@pytest.mark.parametrize("place", PLACES)
def test_a_good_run_passes_every_automatic_check(monkeypatch, place):
    FakeServices(monkeypatch)
    messages, _ = run_itinerary(place)
    checks = check_itinerary(messages, place, start=None)
    assert [c.label for c in checks if not c.passed] == []
    assert gemini_calls(messages) == 3
    assert failed_tools(messages) == []
    assert not used_guide_fallback(messages)


def test_a_missing_tool_and_a_wrong_place_are_caught(monkeypatch):
    FakeServices(monkeypatch)
    script = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "get_destination_guide",
                    "args": {"destination": "Galle", "topic": "all"},
                    "id": "c1",
                },
            ],
        ),
        AIMessage("A plan without the other tools."),
    ]
    model = ScriptedModel(messages=iter(script), calls=[])
    messages = run_tool_agent(build_tool_agent_graph(model), "Plan 3 days in Kandy.")

    failed = [c.label for c in check_itinerary(messages, "Kandy", None) if not c.passed]
    assert "called find_attractions" in failed
    assert "called get_weather_forecast" in failed
    assert any(
        label.startswith("get_destination_guide asked about Kandy") for label in failed
    )
    assert any(label.startswith("travel time from Colombo") for label in failed)
    assert "plan says it assumed Colombo as the start" in failed
    assert "plan has a 'Not available' list" in failed


def test_a_given_start_replaces_the_default_origin_check(monkeypatch):
    FakeServices(monkeypatch)
    messages, _ = run_itinerary("Galle")  # the script travels from Colombo
    labels = {c.label: c.passed for c in check_itinerary(messages, "Galle", "Kandy")}
    assert not any("assumed" in label for label in labels)
    assert [
        ok for label, ok in labels.items() if label.startswith("travel time from Kandy")
    ] == [False]


@pytest.mark.parametrize(
    "first_line",
    [
        "Weather unavailable for 'X': boom.",
        "Attractions unavailable for 'X': boom.",
        "Guide unavailable for 'X': boom.",
        "Travel time unavailable from 'A' to 'B': boom.",
        "Currency conversion unavailable: boom.",
    ],
)
def test_every_tools_failure_wording_is_recognised(first_line):
    message = ToolMessage(first_line + "\nmore", tool_call_id="c", name="some_tool")
    assert failed_tools([message]) == ["some_tool"]


def test_a_result_mentioning_unavailable_later_is_not_a_failure():
    message = ToolMessage(
        "Wikivoyage guide for X:\nTrains unavailable on Sundays.",
        tool_call_id="c",
        name="get_destination_guide",
    )
    assert failed_tools([message]) == []


def test_the_report_records_the_fallback_and_the_failure(monkeypatch):
    FakeServices(monkeypatch, overpass_down=True)
    messages, _ = run_itinerary("Sigiriya")
    report = format_report(
        "Sigiriya",
        "Plan a 3-day trip to Sigiriya.",
        messages,
        check_itinerary(messages, "Sigiriya", None),
    )
    assert "**Tools that failed:** find_attractions" in report
    assert "**Wikivoyage fallback for attractions:** used" in report
    assert "## Your review" in report
    assert "### find_attractions" in report  # raw results kept for the review
    assert "automatic checks passed" in report


def test_cli_runs_saves_and_exits_zero_on_a_good_plan(monkeypatch, tmp_path, capsys):
    FakeServices(monkeypatch)
    model = ScriptedModel(messages=iter(itinerary_script("Kandy")), calls=[])
    monkeypatch.setattr(
        llm_basics, "load_settings", lambda: SimpleNamespace(api_key="k")
    )
    monkeypatch.setattr(llm_basics, "build_llm", lambda settings, **kwargs: model)

    out_file = tmp_path / "runs" / "kandy.md"
    code = itinerary_check.main(["Kandy", "--save", str(out_file)])

    assert code == 0
    assert "# Itinerary check: Kandy" in capsys.readouterr().out
    assert out_file.read_text(encoding="utf-8").startswith("# Itinerary check: Kandy")


def test_cli_exits_two_when_an_automatic_check_fails(monkeypatch, capsys):
    FakeServices(monkeypatch)
    model = ScriptedModel(messages=iter([AIMessage("Just go.")]), calls=[])
    monkeypatch.setattr(
        llm_basics, "load_settings", lambda: SimpleNamespace(api_key="k")
    )
    monkeypatch.setattr(llm_basics, "build_llm", lambda settings, **kwargs: model)
    assert itinerary_check.main(["Ella"]) == 2


def test_cli_rejects_zero_days():
    with pytest.raises(SystemExit):
        itinerary_check.main(["Kandy", "--days", "0"])
