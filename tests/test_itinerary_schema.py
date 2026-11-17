"""Stage 8 checks: the itinerary schemas and their Gemini function declaration.

Offline only. The declaration test uses langchain-google-genai's own converter
(no API call), so a schema Gemini could not accept fails here first.
"""

from __future__ import annotations

import copy

import pytest
from langchain_core.tools import StructuredTool
from langchain_google_genai._function_utils import (
    convert_to_genai_function_declarations,
)
from pydantic import ValidationError

from ai_travel_assistant.schemas import (
    CostItem,
    DayPlan,
    DayPlanDraft,
    Itinerary,
    ItineraryDraft,
    PracticalFact,
)


def activity(name="Little Adam's Peak", **overrides) -> dict:
    return {
        "time_of_day": "morning",
        "name": name,
        "description": "Short hike with wide views.",
        "source": "OpenStreetMap",
        **overrides,
    }


def day(number: int, **overrides) -> dict:
    return {
        "day": number,
        "theme": f"Day {number}",
        "activities": [activity()],
        **overrides,
    }


def draft(days: int = 3, **overrides) -> dict:
    return {
        "destination": "Ella",
        "days": days,
        "start_from": "Colombo",
        "start_assumed": True,
        "summary": "A relaxed hill-country trip.",
        "day_plans": [day(n) for n in range(1, days + 1)],
        **overrides,
    }


# --- Valid drafts -------------------------------------------------------------------


@pytest.mark.parametrize("days", [1, 3, 7])
def test_valid_multi_day_drafts(days):
    itinerary = ItineraryDraft.model_validate(draft(days))
    assert itinerary.days == len(itinerary.day_plans) == days


def test_optional_fields_default_to_empty_or_null():
    itinerary = ItineraryDraft.model_validate(draft(1))
    plan = itinerary.day_plans[0]
    assert plan.forecast_date is None and plan.travel == []
    assert plan.activities[0].estimated_hours is None
    assert plan.activities[0].distance_km is None
    assert itinerary.costs == itinerary.facts == itinerary.tips == []
    assert itinerary.assumptions == itinerary.not_available == []


def test_json_round_trip():
    itinerary = ItineraryDraft.model_validate(draft(2))
    again = ItineraryDraft.model_validate_json(itinerary.model_dump_json())
    assert again == itinerary


def test_the_checked_itinerary_adds_weather_and_warnings():
    data = draft(1)
    data["day_plans"][0]["weather"] = "Sunny; 15.0-24.0 °C"
    checked = Itinerary.model_validate({**data, "warnings": ["checked"]})
    assert isinstance(checked.day_plans[0], DayPlan)
    assert checked.day_plans[0].weather == "Sunny; 15.0-24.0 °C"
    assert checked.warnings == ["checked"]
    # A draft never carries the code-filled fields.
    assert "weather" not in DayPlanDraft.model_fields
    assert "warnings" not in ItineraryDraft.model_fields


# --- Cross-field rules ----------------------------------------------------------------


def test_days_must_match_the_number_of_day_plans():
    with pytest.raises(ValidationError, match="days is 3 but day_plans has 2"):
        ItineraryDraft.model_validate({**draft(2), "days": 3})


@pytest.mark.parametrize(
    "numbers",
    [[1, 1, 3], [1, 3, 2], [2, 3, 4], [0, 1, 2]],
    ids=["duplicate", "out-of-order", "not-from-1", "zero"],
)
def test_day_numbers_must_be_sequential_and_unique(numbers):
    data = draft(3)
    for plan, number in zip(data["day_plans"], numbers, strict=True):
        plan["day"] = number
    with pytest.raises(ValidationError):
        ItineraryDraft.model_validate(data)


def test_a_day_needs_at_least_one_activity():
    data = draft(1)
    data["day_plans"][0]["activities"] = []
    with pytest.raises(ValidationError):
        ItineraryDraft.model_validate(data)


@pytest.mark.parametrize(
    "field",
    ["destination", "days", "start_from", "start_assumed", "summary", "day_plans"],
)
def test_required_top_level_fields(field):
    data = draft(1)
    del data[field]
    with pytest.raises(ValidationError):
        ItineraryDraft.model_validate(data)


@pytest.mark.parametrize(
    ("change", "path"),
    [
        ({"time_of_day": "night"}, "time_of_day"),
        ({"source": "Google Maps"}, "source"),
        ({"estimated_hours": 0}, "estimated_hours"),
        ({"estimated_hours": 30}, "estimated_hours"),
        ({"distance_km": -1}, "distance_km"),
        ({"name": ""}, "name"),
    ],
)
def test_activity_value_rules(change, path):
    data = draft(1)
    data["day_plans"][0]["activities"][0].update(change)
    with pytest.raises(ValidationError, match=path):
        ItineraryDraft.model_validate(data)


def test_a_malformed_forecast_date_becomes_null_instead_of_an_error():
    ok = DayPlanDraft.model_validate(day(1, forecast_date=" 2026-11-18 "))
    assert ok.forecast_date == "2026-11-18"
    assert (
        DayPlanDraft.model_validate(day(1, forecast_date="Tuesday")).forecast_date
        is None
    )


# --- Costs and facts ------------------------------------------------------------------


def test_a_cost_needs_its_quoted_amount():
    assert (
        CostItem(item="Ticket", quoted_currency="LKR", amount_lkr=3000).amount_usd
        is None
    )
    assert CostItem(
        item="Ticket", quoted_currency="USD", amount_usd=10, amount_lkr=None
    )
    with pytest.raises(ValidationError, match="quoted amount"):
        CostItem(item="Ticket", quoted_currency="LKR", amount_usd=10)
    with pytest.raises(ValidationError):
        CostItem(item="Ticket", quoted_currency="LKR", amount_lkr=-5)
    with pytest.raises(ValidationError):
        CostItem(item="Ticket", quoted_currency="EUR", amount_lkr=5)


def test_a_fact_needs_a_quote_and_a_known_source():
    PracticalFact(
        text="Trains run to Kandy.", quote="trains to Kandy", source="Wikivoyage"
    )
    with pytest.raises(ValidationError):
        PracticalFact(text="Trains run.", quote="", source="Wikivoyage")
    with pytest.raises(ValidationError):
        PracticalFact(text="Trains run.", quote="x", source="TripAdvisor")


def test_validation_does_not_change_the_input():
    data = draft(2)
    before = copy.deepcopy(data)
    ItineraryDraft.model_validate(data)
    assert data == before


# --- The Gemini function declaration (offline conversion) ----------------------------


def declaration():
    tool = StructuredTool.from_function(
        func=lambda **_: None,
        name="submit_itinerary",
        description="Submit the finished itinerary.",
        args_schema=ItineraryDraft,
    )
    [genai_tool] = convert_to_genai_function_declarations([tool])
    [function] = genai_tool.function_declarations
    return function


def depth(schema) -> int:
    """Levels of nested objects below this one (arrays do not add a level)."""
    if schema is None:
        return 0
    if schema.items is not None:
        return depth(schema.items)
    children = (schema.properties or {}).values()
    nested = [depth(child) for child in children]
    return 1 + max(nested, default=0) if schema.properties else 0


def test_the_schema_converts_to_a_gemini_function_declaration():
    function = declaration()
    params = function.parameters
    assert function.name == "submit_itinerary"
    assert list(params.properties) == list(ItineraryDraft.model_fields)
    assert set(params.required) == {
        "destination",
        "days",
        "start_from",
        "start_assumed",
        "summary",
        "day_plans",
    }


def test_nested_models_are_inlined_with_their_enums_and_required_fields():
    params = declaration().parameters
    day_schema = params.properties["day_plans"].items
    activity_schema = day_schema.properties["activities"].items
    cost_schema = params.properties["costs"].items
    assert set(day_schema.required) == {"day", "theme", "activities"}
    assert set(activity_schema.required) == {
        "time_of_day",
        "name",
        "description",
        "source",
    }
    assert activity_schema.properties["source"].enum == [
        "OpenStreetMap",
        "Wikivoyage",
        "Built-in guide",
    ]
    assert cost_schema.properties["quoted_currency"].enum == ["LKR", "USD"]


def test_optional_values_are_simply_not_required():
    """Optional fields become plain types outside `required`; Gemini omits them."""
    day_schema = declaration().parameters.properties["day_plans"].items
    activity_schema = day_schema.properties["activities"].items
    for name in ("estimated_hours", "distance_km"):
        assert name not in activity_schema.required
        assert activity_schema.properties[name].any_of is None
    assert "forecast_date" not in day_schema.required


def test_the_declaration_stays_within_three_levels():
    assert depth(declaration().parameters) <= 3
