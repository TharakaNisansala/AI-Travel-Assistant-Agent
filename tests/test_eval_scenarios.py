"""Stage 9, step 1: the scenario dataset and its loader (offline, no API key).

Two kinds of test: the real evals/scenarios.yaml is valid and has the agreed
shape, and broken YAML fails at load time with a message that names the rule.
"""

from __future__ import annotations

import textwrap
from collections import Counter

import pytest

from ai_travel_assistant.evals.scenarios import (
    CATEGORIES,
    CATEGORY_RULES,
    DEFAULT_PATH,
    FAULT_TOOLS,
    ScenarioFileError,
    load_scenarios,
    parse_scenarios,
    select_scenarios,
)
from ai_travel_assistant.faults import SERVICES
from ai_travel_assistant.itinerary_check import REQUIRED_TOOLS

# --- The real dataset -----------------------------------------------------------


@pytest.fixture(scope="module")
def scenarios():
    return load_scenarios()


def test_dataset_has_the_agreed_30_scenarios(scenarios):
    counts = Counter(s.category for s in scenarios)
    assert counts == {
        "understanding": 8,
        "routing": 7,
        "itinerary": 8,
        "fault": 4,
        "multi_turn": 3,
    }
    assert len(scenarios) == 30


def test_routing_covers_every_route(scenarios):
    routes = {s.expect.route for s in scenarios if s.category == "routing"}
    assert routes == {"agent", "clarify", "off_topic"}


def test_understanding_checks_that_unstated_fields_stay_empty(scenarios):
    # At least one case per critical field where the value must stay null.
    nulls = {
        name
        for s in scenarios
        if s.category == "understanding"
        for name, value in s.expect.trip.items()
        if value is None
    }
    assert {"destination", "days"} <= nulls


def test_plans_require_the_itinerary_check_tools(scenarios):
    # The YAML anchor and the Stage 4 constant must not drift apart.
    # A plan that asks for hotels may also require suggest_hotels, nothing else.
    for s in scenarios:
        if s.category in ("itinerary", "fault"):
            required = set(s.expect.required_tools)
            assert set(REQUIRED_TOOLS) <= required, s.id
            assert required - set(REQUIRED_TOOLS) <= {"suggest_hotels"}, s.id


def test_fault_scenarios_each_break_a_different_tool(scenarios):
    failing = [s.failing_tools for s in scenarios if s.category == "fault"]
    assert all(len(tools) == 1 for tools in failing)
    assert len(set(failing)) == len(failing)
    # A failing tool must also be one the run is required to call.
    for s in scenarios:
        if s.category == "fault":
            assert s.failing_tools <= set(s.expect.required_tools), s.id


def test_prompts_are_not_repeated(scenarios):
    first_turns = Counter(s.turns[0] for s in scenarios)
    assert [t for t, n in first_turns.items() if n > 1] == []


def test_every_fault_service_maps_to_a_tool():
    assert set(FAULT_TOOLS) == set(SERVICES)


def test_every_category_has_a_rule():
    assert set(CATEGORY_RULES) == set(CATEGORIES)
    for rule in CATEGORY_RULES.values():
        assert rule.required <= rule.allowed


# --- Filtering ------------------------------------------------------------------


def test_select_by_category_and_id(scenarios):
    routing = select_scenarios(scenarios, categories=["routing"])
    assert {s.category for s in routing} == {"routing"}
    assert len(routing) == 7

    picked = select_scenarios(scenarios, only=["itin-ella-3d", "fault-weather-ella"])
    assert [s.id for s in picked] == ["itin-ella-3d", "fault-weather-ella"]

    both = select_scenarios(
        scenarios, categories=["fault"], only=["fault-weather-ella"]
    )
    assert [s.id for s in both] == ["fault-weather-ella"]


@pytest.mark.parametrize(
    ("categories", "only", "message"),
    [
        (["itinerarys"], None, "unknown categories"),
        (None, ["itin-ella-4d"], "no scenario with ids"),
        # A real id, but outside the chosen category.
        (["routing"], ["itin-ella-3d"], "no scenario with ids"),
    ],
)
def test_select_rejects_unknown_names(scenarios, categories, only, message):
    with pytest.raises(ScenarioFileError, match=message):
        select_scenarios(scenarios, categories=categories, only=only)


def test_load_filters_too():
    assert len(load_scenarios(DEFAULT_PATH, categories=["multi_turn"])) == 3


def test_missing_file_is_a_clear_error(tmp_path):
    with pytest.raises(ScenarioFileError, match="cannot read"):
        load_scenarios(tmp_path / "nope.yaml")


# --- Broken YAML fails at load time ---------------------------------------------


def _yaml(body: str) -> str:
    return "version: 1\nscenarios:\n" + textwrap.indent(textwrap.dedent(body), "  ")


GOOD = """\
- id: itin-ella
  category: itinerary
  turns: ["Plan 2 days in Ella."]
  expect:
    days: 2
    destination: Ella
    required_tools: [find_attractions]
"""


def test_a_minimal_valid_file_parses():
    [scenario] = parse_scenarios(_yaml(GOOD))
    assert scenario.id == "itin-ella"
    assert scenario.failing_tools == frozenset()


@pytest.mark.parametrize(
    ("body", "message"),
    [
        # Typos in field names.
        (GOOD.replace("required_tools", "requird_tools"), "requird_tools"),
        (GOOD.replace("turns:", "turn:"), "turn"),
        # A tool that does not exist.
        (GOOD.replace("find_attractions", "find_atractions"), "unknown tools"),
        (
            GOOD.replace("[find_attractions]", "[find_attractions, find_attractions]"),
            "lists a tool twice",
        ),
        (GOOD.replace("[find_attractions]", "[]"), "required_tools is empty"),
        # Same id twice.
        (GOOD + GOOD, "used more than once"),
        # Bad ids and categories.
        (GOOD.replace("itin-ella", "Itin Ella"), "pattern"),
        (GOOD.replace("category: itinerary", "category: plan"), "category"),
        # Category rules: a required field missing, a field not scored here.
        (GOOD.replace("    days: 2\n", ""), "must set expect fields \\['days'\\]"),
        (GOOD + "    route: agent\n", "does not score expect fields"),
        # Turn counts.
        (
            GOOD.replace('["Plan 2 days in Ella."]', '["Plan 2 days.", "Ella."]'),
            "needs 1 turns",
        ),
        (GOOD.replace('"Plan 2 days in Ella."', '"  "'), "a turn is empty"),
        # Faults only in the fault category, and only real services.
        (GOOD + "  faults: weather\n", "only category 'fault'"),
        (GOOD.replace("itinerary", "fault") + "  faults: wether\n", "unknown service"),
        (GOOD.replace("itinerary", "fault"), "needs `faults`"),
        # Values the schema rejects.
        (GOOD.replace("days: 2", "days: 0"), "days"),
    ],
)
def test_broken_files_fail_with_a_clear_message(body, message):
    with pytest.raises(ScenarioFileError, match=message):
        parse_scenarios(_yaml(body))


@pytest.mark.parametrize(
    ("trip", "message"),
    [
        ("{destinaton: Ella}", "unknown trip fields"),
        ("{budget: cheap}", "budget"),  # not budget / mid-range / luxury
        ("{}", "trip is empty"),
    ],
)
def test_bad_trip_expectations_fail(trip, message):
    body = f"""\
- id: und-x
  category: understanding
  turns: ["Plan a trip."]
  expect:
    trip: {trip}
"""
    with pytest.raises(ScenarioFileError, match=message):
        parse_scenarios(_yaml(body))


def test_multi_turn_routes_must_match_the_turns():
    body = """\
- id: multi-x
  category: multi_turn
  turns: ["Plan me a trip.", "Galle, 2 days."]
  expect: {routes: [clarify], days: 2, destination: Galle}
"""
    with pytest.raises(ScenarioFileError, match="1 entries for 2 turns"):
        parse_scenarios(_yaml(body))


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("scenarios: [", "not valid YAML"),
        ("version: 2\nscenarios: []", "version"),
        ("version: 1\nscenarios: []", "scenarios"),
        # safe_load refuses Python object tags.
        ("!!python/object/apply:os.system ['echo hi']", "not valid YAML"),
    ],
)
def test_bad_files(text, message):
    with pytest.raises(ScenarioFileError, match=message):
        parse_scenarios(text)
