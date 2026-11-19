# ruff: noqa: E501 - tool-text fixtures keep their real line lengths.
"""Stage 8 checks: submit_itinerary, validation, salvage, rendering and compose.

Offline only: compose_itinerary() is a pure function, so these tests feed it
an AIMessage with a submit call and a thread of tool results.
"""

from __future__ import annotations

from datetime import date

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import ValidationError

from ai_travel_assistant import itinerary_output
from ai_travel_assistant.itinerary_output import (
    ACCEPTED_RESULT,
    ITINERARY_NAME,
    NOT_RUN_WITH_SUBMIT,
    REPAIR_PREFIX,
    SALVAGE_WARNING,
    UNUSABLE_REPLY,
    compose_itinerary,
    make_compose_node,
    render_itinerary_markdown,
    repairs_this_turn,
    route_after_compose,
    salvage_draft,
    submit_itinerary,
    summarize_validation_error,
)
from ai_travel_assistant.schemas import Itinerary, ItineraryDraft

TODAY = date(2026, 11, 18)

OSM_ELLA = """Attractions near Ella, Badulla, Sri Lanka; category: all; within 5 km; nearest first:
- Little Adam's Peak (peak), 1.8 km
- Nine Arch Bridge (viewpoint), 2.1 km
Source: OpenStreetMap via the Overpass API (© OpenStreetMap contributors)."""
GUIDE_ELLA = """Wikivoyage guide for Ella (all topics):
[See] Ella Rock is a popular half-day hike. Entry to the Ravana Cave costs Rs 500 for foreigners.
[Get in] Trains from Kandy stop at Ella station."""
WEATHER_ELLA = """Weather forecast for Ella, Badulla, Sri Lanka, at lat 6.87, lon 81.05.
Daily forecast (2 days):
- 2026-11-18 (Wed): Light rain; min 15.0 °C, max 22.0 °C; chance of precipitation 60%; rainfall 3.2 mm; max wind 12.0 km/h.
- 2026-11-19 (Thu): Partly cloudy; min 14.5 °C, max 24.0 °C; chance of precipitation 20%; rainfall 0.0 mm; max wind 9.0 km/h."""
ROUTE = """Road trip from Colombo to Ella:
- Road distance: 212 km (straight line 150 km).
- Driving time: about 5 h 40 min by car without traffic or stops."""
CONVERT = "500.00 LKR = 1.67 USD"


def tool_step() -> list:
    items = [
        ("find_attractions", {"destination": "Ella"}, OSM_ELLA),
        ("get_destination_guide", {"destination": "Ella"}, GUIDE_ELLA),
        ("get_weather_forecast", {"destination": "Ella"}, WEATHER_ELLA),
        ("estimate_travel_time", {"origin": "Colombo", "destination": "Ella"}, ROUTE),
        (
            "convert_currency",
            {"amount": 500, "from_currency": "LKR", "to_currency": "USD"},
            CONVERT,
        ),
    ]
    calls = [{"name": n, "args": a, "id": f"t{i}"} for i, (n, a, _) in enumerate(items)]
    return [
        AIMessage("", tool_calls=calls),
        *(
            ToolMessage(t, tool_call_id=c["id"], name=c["name"])
            for c, (_, _, t) in zip(calls, items, strict=True)
        ),
    ]


def plan_args(**overrides) -> dict:
    args = {
        "destination": "Ella",
        "days": 2,
        "start_from": "Colombo",
        "start_assumed": True,
        "summary": "Two easy days in the hills.",
        "day_plans": [
            {
                "day": 1,
                "theme": "Arrival and views",
                "forecast_date": "2026-11-18",
                "travel": [{"from_place": "Colombo", "to_place": "Ella"}],
                "activities": [
                    {
                        "time_of_day": "afternoon",
                        "name": "Nine Arch Bridge",
                        "description": "Watch a train cross.",
                        "source": "OpenStreetMap",
                        "estimated_hours": 1.5,
                    }
                ],
            },
            {
                "day": 2,
                "theme": "Hikes",
                "forecast_date": "2026-11-19",
                "activities": [
                    {
                        "time_of_day": "morning",
                        "name": "Little Adam's Peak",
                        "description": "A short climb.",
                        "source": "OpenStreetMap",
                    },
                    {
                        "time_of_day": "afternoon",
                        "name": "Ella Rock",
                        "description": "A longer hike.",
                        "source": "Wikivoyage",
                        "estimated_hours": 4,
                    },
                ],
            },
        ],
        "costs": [
            {"item": "Ravana Cave entry", "quoted_currency": "LKR", "amount_lkr": 500}
        ],
        "facts": [
            {
                "text": "Trains from Kandy stop at Ella.",
                "quote": "Trains from Kandy stop at Ella station",
                "source": "Wikivoyage",
            }
        ],
        "tips": ["Carry rain gear."],
        "not_available": ["Hotel prices"],
    }
    args.update(overrides)
    return args


def submit(args: dict, call_id: str = "s1", *others: dict) -> AIMessage:
    return AIMessage(
        "",
        tool_calls=[*others, {"name": "submit_itinerary", "args": args, "id": call_id}],
    )


def thread_with(last: AIMessage, *before) -> list:
    return [HumanMessage("Plan 2 days in Ella"), *tool_step(), *before, last]


def compose(last: AIMessage, *before, **kwargs):
    return compose_itinerary(last, thread_with(last, *before), today=TODAY, **kwargs)


# --- The tool -----------------------------------------------------------------------------


def test_submit_tool_takes_the_draft_schema_and_is_never_run():
    assert submit_itinerary.name == "submit_itinerary"
    assert submit_itinerary.args_schema is ItineraryDraft
    with pytest.raises(RuntimeError):
        submit_itinerary.func()


# --- Validation messages ------------------------------------------------------------------


def test_validation_summary_lists_paths_and_messages_but_never_values():
    bad = plan_args(days="SECRET-VALUE", destination="")
    with pytest.raises(ValidationError) as caught:
        ItineraryDraft.model_validate(bad)
    text = summarize_validation_error(caught.value)
    assert "- days:" in text and "- destination:" in text
    assert "SECRET-VALUE" not in text


def test_validation_summary_is_limited():
    bad = plan_args(day_plans=[{"day": "x"}] * 10)
    with pytest.raises(ValidationError) as caught:
        ItineraryDraft.model_validate(bad)
    text = summarize_validation_error(caught.value, limit=3)
    assert len(text.splitlines()) == 4
    assert text.splitlines()[-1].startswith("- and ")


def test_repairs_are_counted_only_in_the_current_turn():
    old = ToolMessage("x", tool_call_id="a", name="submit_itinerary", status="error")
    now = ToolMessage("y", tool_call_id="b", name="submit_itinerary", status="error")
    ok = ToolMessage("z", tool_call_id="c", name="submit_itinerary")
    messages = [HumanMessage("1"), old, HumanMessage("2"), ok, now]
    assert repairs_this_turn(messages) == 1


# --- Salvage --------------------------------------------------------------------------------


def test_salvage_keeps_valid_days_renumbered_and_drops_broken_parts():
    args = plan_args()
    args["day_plans"].insert(
        0, {"day": 1, "theme": "Broken", "activities": [{"name": "?"}]}
    )
    args["day_plans"][2]["activities"].append({"time_of_day": "never"})
    args["costs"].append({"item": "Bad", "quoted_currency": "LKR"})
    args["tips"] = ["Ok tip", 7, ""]
    args["days"] = 99

    draft = salvage_draft(args)

    assert [p.day for p in draft.day_plans] == [1, 2]
    assert [p.theme for p in draft.day_plans] == ["Arrival and views", "Hikes"]
    assert len(draft.day_plans[1].activities) == 2
    assert draft.days == 2
    assert [c.item for c in draft.costs] == ["Ravana Cave entry"]
    assert draft.tips == ["Ok tip"]


def test_salvage_takes_the_destination_from_the_trip_request():
    args = plan_args(destination=None, start_from=None)
    draft = salvage_draft(args, {"destination": "Ella"})
    assert draft.destination == "Ella"
    assert (draft.start_from, draft.start_assumed) == ("Not stated", False)


@pytest.mark.parametrize(
    "args",
    [
        None,
        "text",
        {"destination": "Ella"},
        plan_args(day_plans=[{"activities": []}]),
        plan_args(destination=""),
    ],
)
def test_salvage_gives_up_when_nothing_usable_is_left(args):
    assert salvage_draft(args) is None


# --- Rendering ----------------------------------------------------------------------------


def checked(**overrides) -> Itinerary:
    result = compose(submit(plan_args(**overrides)))
    return Itinerary.model_validate(result.itinerary)


def test_the_rendered_plan():
    text = render_itinerary_markdown(checked())
    assert text.splitlines()[:5] == [
        "# 2-day trip to Ella",
        "",
        "Two easy days in the hills.",
        "",
        "**Starting point:** Colombo (assumed)",
    ]
    assert "## Day 1: Arrival and views" in text
    assert "*Weather:* 2026-11-18: Light rain; min 15.0 °C, max 22.0 °C" in text
    assert (
        "*Travel:* Colombo to Ella: 212 km, about 5 h 40 min by car without traffic (OSRM)"
        in text
    )
    assert (
        "- **Afternoon: Nine Arch Bridge** (about 1.5 h; 2.1 km straight-line) - Watch a train cross. _(OpenStreetMap)_"
        in text
    )
    assert (
        "- **Afternoon: Ella Rock** (about 4 h) - A longer hike. _(Wikivoyage)_" in text
    )
    # Guide prices are estimates, per person and for the group, then a budget.
    assert "## Other costs (estimates from Wikivoyage, may be out of date)" in text
    assert "- **Ravana Cave entry**" in text
    assert "  - Price: LKR 500 (USD 1.67) (shared) x 1" in text
    assert "  - For the group: LKR 500 (USD 1.67)" in text
    assert "## Trip budget" in text
    assert "- TOTAL estimated budget for the group: LKR 500 (USD 1.67)" in text
    assert "  - Food and drinks" in text  # named as not included, never guessed
    assert "- Trains from Kandy stop at Ella. _(Wikivoyage)_" in text
    assert "_General advice, not checked against live data._" in text
    assert "- Hotel prices" in text
    assert text.endswith(
        "_Sources: OpenStreetMap, OSRM, Open-Meteo, Wikivoyage, ExchangeRate-API._"
    )


def test_empty_sections_are_left_out_and_nothing_says_none():
    text = render_itinerary_markdown(
        checked(
            costs=[],
            facts=[],
            tips=[],
            not_available=[],
            start_assumed=False,
            assumptions=[],
            summary="",
        )
    )
    for heading in (
        "## Costs",
        "## Practical information",
        "## Suggestions",
        "## Not available",
        "## Assumptions",
        "## Please note",
    ):
        assert heading not in text
    assert "None" not in text
    assert "(assumed)" not in text


def test_a_day_without_forecast_or_route_says_so():
    args = plan_args()
    args["day_plans"][0]["forecast_date"] = None
    args["day_plans"][0]["travel"] = [{"from_place": "Ella", "to_place": "Haputale"}]
    text = render_itinerary_markdown(checked(day_plans=args["day_plans"]))
    assert "*Weather:* No forecast for this day." in text
    assert "*Travel:* Ella to Haputale: distance and time not available" in text


# --- Compose ---------------------------------------------------------------------------------


def test_a_valid_submission_is_accepted_and_rendered():
    last = submit(plan_args())
    result = compose(last)

    accepted, reply = result.messages
    assert (accepted.tool_call_id, accepted.content, accepted.status) == (
        "s1",
        ACCEPTED_RESULT,
        "success",
    )
    assert reply.name == ITINERARY_NAME and reply.content.startswith(
        "# 2-day trip to Ella"
    )
    assert result.repair is False
    assert result.itinerary["destination"] == "Ella"
    assert result.itinerary["day_plans"][0]["weather"].startswith("2026-11-18")


def test_other_calls_next_to_the_submit_are_closed_in_call_order():
    weather = {
        "name": "get_weather_forecast",
        "args": {"destination": "Ella"},
        "id": "w9",
    }
    result = compose(submit(plan_args(), "s1", weather))
    first, second, _ = result.messages
    assert (first.tool_call_id, first.content) == ("w9", NOT_RUN_WITH_SUBMIT)
    assert second.tool_call_id == "s1"


def test_an_invalid_first_submission_goes_back_for_one_repair():
    result = compose(submit(plan_args(days=3)))
    [errors] = result.messages
    assert result.repair is True and result.itinerary is None
    assert errors.status == "error" and errors.name == "submit_itinerary"
    assert errors.content.startswith(REPAIR_PREFIX)
    assert "days is 3 but day_plans has 2" in errors.content


def test_a_second_invalid_submission_is_salvaged_without_another_repair():
    earlier = [
        submit(plan_args(days=3), "s0"),
        ToolMessage(
            "errors", tool_call_id="s0", name="submit_itinerary", status="error"
        ),
    ]
    result = compose(submit(plan_args(days=5), "s1"), *earlier)
    assert result.repair is False
    assert result.itinerary["days"] == 2
    assert result.itinerary["warnings"][0] == SALVAGE_WARNING
    assert result.messages[-1].content.startswith("# 2-day trip to Ella")


def test_repair_can_be_switched_off():
    result = compose(submit(plan_args(days=3)), allow_repair=False)
    assert result.repair is False
    assert result.itinerary["warnings"][0] == SALVAGE_WARNING


def test_an_unusable_submission_gets_a_fixed_reply():
    result = compose(submit({"days": "many"}), allow_repair=False)
    failed, reply = result.messages
    assert failed.status == "error" and failed.tool_call_id == "s1"
    assert reply.content == UNUSABLE_REPLY and reply.name == ITINERARY_NAME
    assert result.itinerary is None


def test_a_bug_while_rendering_never_crashes_the_turn(monkeypatch):
    def broken(_):
        raise KeyError("bug")

    monkeypatch.setattr(itinerary_output, "render_itinerary_markdown", broken)
    result = compose(submit(plan_args()))
    assert result.messages[-1].content == UNUSABLE_REPLY


def test_grounding_is_applied():
    result = compose(
        submit(
            plan_args(
                costs=[{"item": "Made up", "quoted_currency": "USD", "amount_usd": 99}]
            )
        )
    )
    assert result.itinerary["costs"] == []
    assert (
        "Price of Made up (not quoted in the Ella guide)"
        in result.itinerary["not_available"]
    )


# --- The node and its router -------------------------------------------------------------


def test_compose_node_saves_the_itinerary_only_when_it_is_accepted():
    node = make_compose_node(today=lambda: TODAY)
    good = submit(plan_args())
    update = node(
        {
            "messages": thread_with(good),
            "trip_request": {"destination": "Ella", "days": 2},
        }
    )
    assert update["itinerary"]["days"] == 2
    assert route_after_compose({"messages": update["messages"]}) == "done"

    bad = submit(plan_args(days=4))
    update = node({"messages": thread_with(bad)})
    assert "itinerary" not in update
    assert route_after_compose({"messages": update["messages"]}) == "repair"


def test_compose_node_passes_the_trip_request_to_the_checks():
    node = make_compose_node(today=lambda: TODAY)
    update = node(
        {"messages": thread_with(submit(plan_args())), "trip_request": {"days": 3}}
    )
    assert "You asked for 3 days; this plan has 2." in update["itinerary"]["warnings"]
