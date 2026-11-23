# ruff: noqa: E501 - tool-text fixtures keep their real line lengths.
"""Stage 8 checks: grounding a submitted itinerary in the tool results, offline.

The tool texts below follow the exact formats in tools.py. Each test thread
also holds results for ANOTHER place (Kandy) or another route, to prove that a
value is accepted only from the source and destination it belongs to.
"""

from __future__ import annotations

import copy
from datetime import date

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ai_travel_assistant.itinerary_grounding import (
    FAILED_TOOL_NOTES,
    collect_evidence,
    contains_phrase,
    ground_itinerary,
    has_unverified_detail,
    normalize,
    parse_minutes,
    same_place,
    scrub_unverified,
    sources_for,
)
from ai_travel_assistant.schemas import ItineraryDraft

TODAY = date(2026, 11, 18)

OSM_ELLA = """Attractions near Ella, Badulla, Sri Lanka; category: all; within 5 km; nearest first:
- Little Adam's Peak (peak), 1.8 km
- Nine Arch Bridge (viewpoint), 2.1 km
- Ravana Falls (waterfall), 4.6 km
Distances are straight-line, not travel distance.
Source: OpenStreetMap via the Overpass API (© OpenStreetMap contributors)."""

OSM_KANDY = """Attractions near Kandy, Central Province, Sri Lanka; category: all; within 5 km; nearest first:
- Temple of the Tooth (temple), 0.5 km
- Ravana Falls (waterfall), 9.9 km
Distances are straight-line, not travel distance.
Source: OpenStreetMap via the Overpass API (© OpenStreetMap contributors)."""

GUIDE_ELLA = """Wikivoyage guide for Ella (all topics):
[Understand] Ella is a small hill town known for walks and tea estates.
[Get in] Trains from Kandy stop at Ella station. A tuk-tuk from the station costs Rs 300.
[See] Ella Rock is a popular half-day hike. Entry to the Ravana Cave costs Rs 500 for foreigners.
[Stay safe] Leeches are common on trails after rain.
Full article: https://en.wikivoyage.org/wiki/Ella
Source: Wikivoyage, community-written (CC BY-SA 4.0); prices, timetables and transport details may be out of date."""

GUIDE_KANDY = """Wikivoyage guide for Kandy (all topics):
[See] The Temple of the Tooth charges Rs 2,000 for foreigners. Udawattakele forest is next to the lake.
Source: Wikivoyage, community-written (CC BY-SA 4.0)."""

WEATHER_ELLA = """Weather forecast for Ella, Badulla, Sri Lanka, at lat 6.87, lon 81.05, elevation 1041 m.
Daily forecast (3 days):
- 2026-11-18 (Wed): Light rain; min 15.0 °C, max 22.0 °C; chance of precipitation 60%; rainfall 3.2 mm; max wind 12.0 km/h.
- 2026-11-19 (Thu): Partly cloudy; min 14.5 °C, max 24.0 °C; chance of precipitation 20%; rainfall 0.0 mm; max wind 9.0 km/h.
- 2026-11-20 (Fri): Overcast; min 15.5 °C, max 23.0 °C; chance of precipitation 35%; rainfall 0.4 mm; max wind 10.0 km/h.
Source: Open-Meteo live forecast API (open-meteo.com)."""

WEATHER_KANDY = """Weather forecast for Kandy, Central Province, Sri Lanka, at lat 7.29, lon 80.63.
Daily forecast (1 days):
- 2026-11-21 (Sat): Sunny; min 20.0 °C, max 30.0 °C; chance of precipitation 5%; rainfall 0.0 mm; max wind 8.0 km/h.
Source: Open-Meteo live forecast API (open-meteo.com)."""

ROUTE_COLOMBO_ELLA = """Road trip from Colombo, Western Province, Sri Lanka to Ella, Badulla, Sri Lanka:
- Road distance: 212 km (straight line 150 km).
- Driving time: about 5 h 40 min by car without traffic or stops.
This is a free-flow car estimate: real journeys, especially on hill roads or by bus, tuk-tuk or train, usually take longer. Say so.
Source: OSRM routing on OpenStreetMap roads (© OpenStreetMap contributors)."""

ROUTE_COLOMBO_KANDY = """Road trip from Colombo, Western Province, Sri Lanka to Kandy, Central Province, Sri Lanka:
- Road distance: 115 km (straight line 94 km).
- Driving time: about 3 h 5 min by car without traffic or stops.
Source: OSRM routing on OpenStreetMap roads (© OpenStreetMap contributors)."""

CONVERT_500 = """500.00 LKR = 1.67 USD
Rate used: 1 LKR = 0.00334 USD (updated 2026-11-18).
Source: Rates By Exchange Rate API (https://www.exchangerate-api.com)."""

CONVERT_2000 = """2,000.00 LKR = 6.68 USD
Rate used: 1 LKR = 0.00334 USD."""


_STEP = iter(range(10**6))


def results(*items: tuple[str, dict, str]) -> list:
    """One agent step: an AIMessage with these calls, then their results."""
    step = next(_STEP)
    calls = [
        {"name": name, "args": args, "id": f"s{step}-{index}-{name}"}
        for index, (name, args, _) in enumerate(items)
    ]
    messages: list = [AIMessage("", tool_calls=calls)]
    for call, (_, _, text) in zip(calls, items, strict=True):
        status = "error" if text.startswith("Tool failed") else "success"
        messages.append(
            ToolMessage(text, tool_call_id=call["id"], name=call["name"], status=status)
        )
    return messages


def full_thread(**replace: str) -> list:
    """A thread with Ella data, plus Kandy data from an earlier turn."""
    texts = {
        "osm": OSM_ELLA,
        "guide": GUIDE_ELLA,
        "weather": WEATHER_ELLA,
        "route": ROUTE_COLOMBO_ELLA,
        "convert": CONVERT_500,
        **replace,
    }
    return [
        HumanMessage("Plan 2 days in Kandy"),
        *results(
            ("find_attractions", {"destination": "Kandy"}, OSM_KANDY),
            ("get_destination_guide", {"destination": "Kandy"}, GUIDE_KANDY),
            ("get_weather_forecast", {"destination": "Kandy"}, WEATHER_KANDY),
            (
                "estimate_travel_time",
                {"origin": "Colombo", "destination": "Kandy"},
                ROUTE_COLOMBO_KANDY,
            ),
            (
                "convert_currency",
                {"amount": 2000, "from_currency": "LKR", "to_currency": "USD"},
                CONVERT_2000,
            ),
        ),
        AIMessage("Here is your Kandy plan."),
        HumanMessage("Now plan 3 days in Ella"),
        *results(
            ("find_attractions", {"destination": "Ella"}, texts["osm"]),
            ("get_destination_guide", {"destination": "Ella"}, texts["guide"]),
            ("get_weather_forecast", {"destination": "Ella"}, texts["weather"]),
            (
                "estimate_travel_time",
                {"origin": "Colombo", "destination": "Ella"},
                texts["route"],
            ),
        ),
        *results(
            (
                "convert_currency",
                {"amount": 500, "from_currency": "LKR", "to_currency": "USD"},
                texts["convert"],
            ),
        ),
    ]


def activity(name: str, **overrides) -> dict:
    return {
        "time_of_day": "morning",
        "name": name,
        "description": "A good walk.",
        "source": "OpenStreetMap",
        **overrides,
    }


def draft(**overrides) -> ItineraryDraft:
    data = {
        "destination": "Ella",
        "days": 1,
        "start_from": "Colombo",
        "start_assumed": True,
        "summary": "A short hill-country trip.",
        "day_plans": [
            {
                "day": 1,
                "theme": "Hikes",
                "forecast_date": "2026-11-18",
                "activities": [activity("Little Adam's Peak", distance_km=1.8)],
            }
        ],
    }
    data.update(overrides)
    return ItineraryDraft.model_validate(data)


def one_day(*activities: dict, **day_overrides) -> list[dict]:
    return [{"day": 1, "theme": "Day", "activities": list(activities), **day_overrides}]


def ground(itinerary: ItineraryDraft, messages=None, **kwargs):
    return ground_itinerary(
        itinerary,
        full_thread() if messages is None else messages,
        today=TODAY,
        **kwargs,
    )


# --- Helpers --------------------------------------------------------------------------


def test_normalize_and_phrase_matching():
    assert normalize("Ravana's  Fálls!") == "ravana s falls"
    assert contains_phrase("Visit Ravana Falls today", "ravana falls")
    assert not contains_phrase("Ellaborate plans", "Ella")  # whole words only
    assert not contains_phrase("anything", "")


@pytest.mark.parametrize(
    ("a", "b", "same"),
    [
        ("Ella", "ella", True),
        ("Ella, Sri Lanka", "Ella", True),
        ("Nuwara Eliya", "Nuwara Eliya town", True),
        ("Ella", "Kandy", False),
        ("Ella", "Ellawala", False),
        ("", "Ella", False),
        (None, "Ella", False),
    ],
)
def test_same_place(a, b, same):
    assert same_place(a, b) is same


def test_parse_minutes():
    assert parse_minutes("5 h 40 min") == 340
    assert parse_minutes("45 min") == 45
    assert parse_minutes("3 h") == 180
    assert parse_minutes("soon") is None


def test_unverified_details_are_spotted_and_scrubbed():
    for risky in [
        "Costs Rs 500.",
        "About $20.",
        "1,500 LKR each.",
        "Opens at 8am.",
        "Open daily until 5.",
        "The train departs often.",
        "Meet at 09:30.",
    ]:
        assert has_unverified_detail(risky), risky
    assert not has_unverified_detail("Bring good shoes and water.")
    assert scrub_unverified("Great views. Entry is Rs 500. Go early.") == (
        "Great views. Go early.",
        1,
    )


def test_collect_evidence_pairs_results_with_calls_and_marks_failures():
    messages = [
        *results(
            (
                "get_weather_forecast",
                {"destination": "Ella"},
                "Weather unavailable for 'Ella': down.",
            ),
            (
                "find_attractions",
                {"destination": "Ella"},
                "Attractions unavailable for 'Ella': x\nFALLBACK: sights named in the Wikivoyage guide instead.",
            ),
            (
                "get_destination_guide",
                {"destination": "Ella"},
                "Tool failed with an internal error (KeyError).",
            ),
            ("submit_itinerary", {"destination": "Ella"}, "Itinerary shown."),
        ),
        ToolMessage("orphan", tool_call_id="nobody", name="x"),
    ]
    evidence = collect_evidence(messages)
    assert [(e.tool, e.ok, e.fallback) for e in evidence] == [
        ("get_weather_forecast", False, False),
        ("find_attractions", False, True),
        ("get_destination_guide", False, False),
    ]
    assert evidence[0].args == {"destination": "Ella"}


def test_sources_keep_only_the_destination_they_belong_to():
    src = sources_for("Ella", collect_evidence(full_thread()))
    assert set(src.osm_sights) == {
        "little adam s peak",
        "nine arch bridge",
        "ravana falls",
    }
    assert src.osm_sights["ravana falls"] == ("Ravana Falls", 4.6)  # not Kandy's 9.9
    assert "Temple of the Tooth" not in src.wikivoyage_text
    assert set(src.forecast) == {"2026-11-18", "2026-11-19", "2026-11-20"}
    assert src.guide_prices["LKR"] == [300.0, 500.0]  # not Kandy's 2,000


@pytest.mark.parametrize(
    ("sleep", "currency", "prices"),
    [
        ("Guesthouses charge Rs 3,000-5,000 a night.", "LKR", [3000.0, 5000.0]),
        ("Rooms cost Rs. 2,500 – Rs 4,000.", "LKR", [2500.0, 4000.0]),
        ("Doubles for 6,000 - 9,000 rupees.", "LKR", [6000.0, 9000.0]),
        ("Hotels from US$20-40 a night.", "USD", [20.0, 40.0]),
        ("Boutique villas $80–$120.", "USD", [80.0, 120.0]),
    ],
)
def test_both_ends_of_a_price_range_count_as_quoted(sleep, currency, prices):
    guide = GUIDE_ELLA.replace("[Stay safe]", f"[Sleep] {sleep}\n[Stay safe]")
    src = sources_for("Ella", collect_evidence(full_thread(guide=guide)))
    assert set(prices) <= set(src.guide_prices[currency])


def test_an_accommodation_price_from_the_guide_is_kept():
    guide = GUIDE_ELLA.replace(
        "[Stay safe]",
        "[Sleep] Guesthouses charge Rs 3,000-5,000 a night.\n[Stay safe]",
    )
    room = cost("Guesthouse double room per night (top of range)", "LKR", lkr=5000)
    checked = ground(draft(costs=[room]), full_thread(guide=guide))
    assert [c.amount_lkr for c in checked.costs] == [5000]


# --- Activities -----------------------------------------------------------------------


def test_an_osm_sight_gets_its_own_listed_distance():
    checked = ground(
        draft(day_plans=one_day(activity("Ravana Falls", distance_km=1.8)))
    )
    [act] = checked.day_plans[0].activities
    assert act.distance_km == 4.6  # the distance listed for Ravana Falls in Ella
    assert act.source == "OpenStreetMap"
    assert checked.warnings == []


def test_a_sight_from_another_destination_is_flagged_without_a_distance():
    checked = ground(
        draft(day_plans=one_day(activity("Temple of the Tooth", distance_km=0.5)))
    )
    [act] = checked.day_plans[0].activities
    assert act.distance_km is None
    assert any("Temple of the Tooth: not found" in w for w in checked.warnings)


def test_a_wikivoyage_sight_has_no_distance():
    checked = ground(
        draft(
            day_plans=one_day(
                activity("Ella Rock", distance_km=3.0, source="OpenStreetMap")
            )
        )
    )
    [act] = checked.day_plans[0].activities
    assert (act.source, act.distance_km) == ("Wikivoyage", None)
    assert checked.warnings == []


def test_risky_sentences_are_removed_from_descriptions():
    checked = ground(
        draft(
            day_plans=one_day(
                activity(
                    "Ella Rock", description="A steep hike. The trail opens at 6am."
                )
            )
        )
    )
    assert checked.day_plans[0].activities[0].description == "A steep hike."
    assert any("Ella Rock: details" in w for w in checked.warnings)


def test_the_osm_fallback_gives_wikivoyage_sights_without_distances():
    fallback = (
        "Attractions unavailable for 'Ella': Overpass down\n"
        "FALLBACK: sights named in the Wikivoyage guide instead (no distances).\n"
        "Wikivoyage guide for Ella (see and do):\n[See] Visit the Ella Spice Garden."
    )
    checked = ground(
        draft(day_plans=one_day(activity("Ella Spice Garden", distance_km=1.0))),
        full_thread(osm=fallback),
    )
    [act] = checked.day_plans[0].activities
    assert (act.source, act.distance_km) == ("Wikivoyage", None)
    assert FAILED_TOOL_NOTES["find_attractions"] in checked.not_available


# --- Travel legs ----------------------------------------------------------------------


def leg(a: str, b: str, km=None, minutes=None) -> dict:
    return {
        "from_place": a,
        "to_place": b,
        "road_distance_km": km,
        "driving_minutes": minutes,
    }


def test_a_leg_takes_the_osrm_figures_for_its_own_two_places():
    checked = ground(
        draft(
            day_plans=one_day(
                activity("Ella Rock"), travel=[leg("Colombo", "Ella", 200, 300)]
            )
        )
    )
    [travel] = checked.day_plans[0].travel
    assert (travel.road_distance_km, travel.driving_minutes) == (212, 340)


def test_a_leg_in_the_other_direction_matches_too():
    checked = ground(
        draft(day_plans=one_day(activity("Ella Rock"), travel=[leg("Ella", "Colombo")]))
    )
    assert checked.day_plans[0].travel[0].road_distance_km == 212


def test_a_leg_without_its_own_route_is_not_given_another_routes_numbers():
    # Kandy's route figures (115 km) are in the thread, but not for Ella -> Kandy.
    checked = ground(
        draft(
            day_plans=one_day(
                activity("Ella Rock"), travel=[leg("Ella", "Kandy", 115, 185)]
            )
        )
    )
    [travel] = checked.day_plans[0].travel
    assert (travel.road_distance_km, travel.driving_minutes) == (None, None)
    assert "Road distance and driving time, Ella to Kandy" in checked.not_available


# --- Costs ----------------------------------------------------------------------------


def cost(item: str, currency: str, lkr=None, usd=None) -> dict:
    return {
        "item": item,
        "quoted_currency": currency,
        "amount_lkr": lkr,
        "amount_usd": usd,
    }


def test_a_guide_price_keeps_its_amount_and_takes_the_matching_conversion():
    checked = ground(draft(costs=[cost("Ravana Cave entry", "LKR", lkr=500, usd=9.99)]))
    [item] = checked.costs
    assert (item.amount_lkr, item.amount_usd) == (500, 1.67)


def test_a_price_quoted_only_for_another_destination_is_removed():
    checked = ground(draft(costs=[cost("Temple ticket", "LKR", lkr=2000, usd=6.68)]))
    assert checked.costs == []
    assert (
        "Price of Temple ticket (not quoted in the Ella guide)" in checked.not_available
    )


def test_a_price_in_the_wrong_currency_is_removed():
    checked = ground(draft(costs=[cost("Ravana Cave entry", "USD", usd=500)]))
    assert checked.costs == []


def test_a_guide_price_without_a_conversion_keeps_only_the_quote():
    checked = ground(
        draft(costs=[cost("Tuk-tuk from the station", "LKR", lkr=300, usd=1.0)])
    )
    [item] = checked.costs
    assert (item.amount_lkr, item.amount_usd) == (300, None)
    assert "USD value of Tuk-tuk from the station (no conversion retrieved)" in (
        checked.not_available
    )


# --- Weather --------------------------------------------------------------------------


def days_with_dates(*dates) -> list[dict]:
    return [
        {
            "day": n,
            "theme": f"Day {n}",
            "forecast_date": when,
            "activities": [activity("Ella Rock")],
        }
        for n, when in enumerate(dates, start=1)
    ]


def test_each_day_gets_the_forecast_line_of_its_own_date():
    checked = ground(
        draft(days=2, day_plans=days_with_dates("2026-11-18", "2026-11-19"))
    )
    first, second = (d.weather for d in checked.day_plans)
    assert first.startswith("2026-11-18: Light rain; min 15.0 °C")
    assert second.startswith("2026-11-19: Partly cloudy")
    assert first.endswith("(Open-Meteo)")


def test_no_weather_for_dates_outside_the_forecast_past_dates_or_other_places():
    checked = ground(
        draft(
            days=4,
            day_plans=days_with_dates("2026-11-17", "2026-11-21", None, "2026-11-25"),
        ),
    )
    # 11-17 is in the past; 11-21 is only in KANDY's forecast; 11-25 is not covered.
    assert [d.weather for d in checked.day_plans] == [None, None, None, None]


def test_a_date_used_twice_gets_the_weather_once():
    checked = ground(
        draft(days=2, day_plans=days_with_dates("2026-11-18", "2026-11-18"))
    )
    assert checked.day_plans[1].weather is None


def test_a_failed_forecast_is_listed_and_nothing_is_filled_in():
    checked = ground(
        draft(),
        full_thread(
            weather="Weather unavailable for 'Ella': service down. Do not guess."
        ),
    )
    assert checked.day_plans[0].weather is None
    assert FAILED_TOOL_NOTES["get_weather_forecast"] in checked.not_available


def test_a_forecast_that_was_never_requested_is_listed():
    thread = [
        m for m in full_thread() if "Weather forecast for Ella" not in str(m.content)
    ]
    checked = ground(draft(), thread)
    assert checked.day_plans[0].weather is None
    assert "Weather forecast (not retrieved)" in checked.not_available


# --- Facts and suggestions ------------------------------------------------------------


def fact(text: str, quote: str, source="Wikivoyage") -> dict:
    return {"text": text, "quote": quote, "source": source}


def test_a_fact_backed_by_this_destinations_guide_is_kept():
    checked = ground(
        draft(
            facts=[
                fact(
                    "Trains from Kandy stop at Ella.",
                    "Trains from Kandy stop at Ella station",
                )
            ]
        )
    )
    assert [f.text for f in checked.facts] == ["Trains from Kandy stop at Ella."]


def test_a_fact_quoted_from_another_destination_is_left_out():
    checked = ground(
        draft(
            facts=[
                fact(
                    "A forest is next to the lake.",
                    "Udawattakele forest is next to the lake",
                )
            ]
        )
    )
    assert checked.facts == []
    assert any("do not confirm" in w for w in checked.warnings)


def test_a_fact_with_numbers_not_in_its_quote_is_shown_as_the_quote():
    checked = ground(
        draft(
            facts=[
                fact(
                    "The tuk-tuk costs Rs 250.",
                    "A tuk-tuk from the station costs Rs 300",
                )
            ]
        )
    )
    assert checked.facts[0].text == "A tuk-tuk from the station costs Rs 300"


def test_an_osrm_fact_must_quote_a_route_for_this_destination():
    ok = ground(
        draft(
            facts=[
                fact("Driving takes a while.", "Driving time: about 5 h 40 min", "OSRM")
            ]
        )
    )
    assert len(ok.facts) == 1
    other = ground(
        draft(
            facts=[fact("Driving takes 3 h.", "Driving time: about 3 h 5 min", "OSRM")]
        )
    )
    assert other.facts == []


def test_suggestions_with_prices_or_times_are_left_out():
    checked = ground(
        draft(tips=["Carry water.", "Tuk-tuks cost Rs 300.", "Start at 6am."])
    )
    assert checked.tips == ["Carry water."]
    assert any("suggestions" in w for w in checked.warnings)


def test_the_summary_loses_unconfirmed_details():
    checked = ground(draft(summary="A green trip. Entry is $5 everywhere."))
    assert checked.summary == "A green trip."


# --- The whole plan -------------------------------------------------------------------


def test_a_fully_grounded_plan_has_no_warnings():
    checked = ground(draft(), trip_request={"destination": "Ella", "days": 1})
    assert checked.warnings == []
    assert checked.day_plans[0].weather.startswith("2026-11-18")


def test_trip_request_mismatches_are_warned_about():
    checked = ground(draft(), trip_request={"destination": "Galle", "days": 3})
    assert "You asked for 3 days; this plan has 1." in checked.warnings
    assert "You asked about Galle; this plan is for Ella." in checked.warnings


def test_the_assumed_start_is_listed_once():
    assert ground(draft()).assumptions == ["The trip starts from Colombo (assumed)."]
    already = ground(draft(assumptions=["Starting from Colombo by car."]))
    assert already.assumptions == ["Starting from Colombo by car."]


def test_a_plan_with_no_tool_results_for_its_destination_is_warned_about():
    checked = ground(
        draft(destination="Galle", day_plans=one_day(activity("Galle Fort")))
    )
    assert checked.warnings[0].startswith("No tool results were found for Galle")
    assert checked.day_plans[0].activities[0].distance_km is None


def test_partial_failures_keep_the_other_results():
    checked = ground(
        draft(costs=[cost("Ravana Cave entry", "LKR", lkr=500)]),
        full_thread(
            weather="Weather unavailable for 'Ella': down.",
            route="Travel time unavailable from 'Colombo' to 'Ella': down.",
        ),
    )
    assert checked.day_plans[0].activities[0].distance_km == 1.8  # OSM still used
    assert checked.costs[0].amount_usd == 1.67  # conversion still used
    assert FAILED_TOOL_NOTES["estimate_travel_time"] in checked.not_available
    assert FAILED_TOOL_NOTES["get_weather_forecast"] in checked.not_available


def test_model_listed_gaps_are_kept_without_duplicates():
    checked = ground(
        draft(not_available=["Hotel prices", "weather forecast (not retrieved)"]),
        [m for m in full_thread() if "Weather forecast for Ella" not in str(m.content)],
    )
    assert checked.not_available == ["Hotel prices", "weather forecast (not retrieved)"]


def test_grounding_changes_neither_the_draft_nor_the_messages():
    itinerary = draft(
        costs=[cost("Temple ticket", "LKR", lkr=2000)], tips=["Rs 5 tip."]
    )
    messages = full_thread()
    before = (itinerary.model_copy(deep=True), copy.deepcopy(messages))
    ground(itinerary, messages)
    assert (itinerary, messages) == before


def test_a_call_id_reused_in_a_later_turn_never_pairs_with_the_wrong_call():
    def step(destination: str, text: str) -> list:
        call = {
            "name": "find_attractions",
            "args": {"destination": destination},
            "id": "same-id",
        }
        return [
            AIMessage("", tool_calls=[call]),
            ToolMessage(text, tool_call_id="same-id"),
        ]

    evidence = collect_evidence([*step("Kandy", OSM_KANDY), *step("Ella", OSM_ELLA)])
    assert [(e.args["destination"], e.text) for e in evidence] == [
        ("Kandy", OSM_KANDY),
        ("Ella", OSM_ELLA),
    ]
