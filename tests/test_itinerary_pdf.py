"""Stage 10: the itinerary PDF holds the whole plan (offline: no key, no network)."""

from __future__ import annotations

import io
from datetime import date

import pytest

pytest.importorskip("fpdf")  # the ui extra
pypdf = pytest.importorskip("pypdf")

from ai_travel_assistant.itinerary_pdf import itinerary_pdf, pdf_file_name  # noqa: E402

DAYS = [
    {
        "day": n,
        "theme": f"Theme {n}",
        "forecast_date": f"2026-11-2{n}",
        "weather": f"2026-11-2{n}: light rain; min 18 °C, max 27 °C",
        "travel": [
            {
                "from_place": "Colombo",
                "to_place": "Kandy",
                "road_distance_km": 115.0,
                "driving_minutes": 190,
            }
        ]
        if n == 1
        else [],
        "activities": [
            {
                "time_of_day": when,
                "name": f"Place {n}-{when}",
                "description": f"Visit number {n} in the {when}.",
                "estimated_hours": 2.0,
                "distance_km": 1.5,
                "source": "OpenStreetMap",
            }
            for when in ("evening", "morning", "afternoon")  # out of order on purpose
        ],
    }
    for n in (1, 2, 3)
]
PLAN = {
    "destination": "Kandy",
    "days": 3,
    "start_from": "Colombo",
    "start_assumed": True,
    "summary": "Three days of temples, lakes and tea.",
    "day_plans": DAYS,
    "costs": [
        {
            "item": "Temple of the Tooth entry",
            "quoted_currency": "LKR",
            "amount_lkr": 2000.0,
            "amount_usd": 6.6,
        }
    ],
    "facts": [
        {
            "text": "Dress modestly at the temple.",
            "quote": "cover shoulders",
            "source": "Wikivoyage",
        }
    ],
    "tips": ["Start early to beat the heat."],
    "assumptions": ["Travelling by car."],
    "warnings": ["Opening hours were not confirmed."],
    "not_available": ["Exact train times."],
}


def pdf_text(data: bytes) -> str:
    reader = pypdf.PdfReader(io.BytesIO(data))
    # Join wrapped lines so phrases split over two lines still match.
    return " ".join(" ".join(page.extract_text().split()) for page in reader.pages)


def test_it_is_a_real_pdf():
    data = itinerary_pdf(PLAN)
    assert data.startswith(b"%PDF-")
    assert len(pypdf.PdfReader(io.BytesIO(data)).pages) >= 1


def test_every_day_and_every_activity_is_in_the_pdf():
    text = pdf_text(itinerary_pdf(PLAN))
    assert "3-day trip to Kandy" in text
    assert "Colombo (assumed)" in text
    assert "Three days of temples, lakes and tea." in text
    for day in DAYS:
        assert f"Day {day['day']} · {day['theme']}" in text
        assert day["weather"] in text
        for act in day["activities"]:
            assert act["name"] in text
            assert act["description"] in text
    assert "Colombo → Kandy · 115 km · about 3 h 10 min by road" in text


def test_costs_notes_and_checks_are_in_the_pdf():
    text = pdf_text(itinerary_pdf(PLAN, today=date(2026, 11, 19)))
    assert "Temple of the Tooth entry" in text
    assert "LKR 2,000" in text and "USD 6.60" in text
    assert "Dress modestly at the temple. (Wikivoyage)" in text
    assert "Start early to beat the heat. (suggestion, not checked)" in text
    assert "Assumed: Travelling by car." in text
    assert "Check: Opening hours were not confirmed." in text
    assert "Not available: Exact train times." in text
    assert "Made on 2026-11-19" in text


def test_activities_follow_the_time_of_day():
    text = pdf_text(itinerary_pdf(PLAN))
    morning = text.index("Place 1-morning")
    assert morning < text.index("Place 1-afternoon") < text.index("Place 1-evening")


def test_model_characters_do_not_break_the_pdf():
    # Groq writes U+2011 hyphens; models also use curly quotes, dashes and emoji.
    plan = {
        **PLAN,
        "summary": "A two‑day trip — temples’ calm 🌴 and “tea”.",
        "day_plans": [{**DAYS[0], "theme": "Lake walk 🚶"}],
    }
    text = pdf_text(itinerary_pdf(plan))
    assert "temples’ calm" in text
    assert "🌴" not in text  # dropped: the font has no emoji


def test_a_minimal_plan_still_makes_a_pdf():
    plan = {
        "destination": "Ella",
        "start_from": "Ella",
        "summary": "",
        "day_plans": [
            {
                "day": 1,
                "theme": "Rest",
                "activities": [
                    {
                        "time_of_day": "morning",
                        "name": "Nine Arch Bridge",
                        "description": "",
                        "source": "Wikivoyage",
                    }
                ],
            }
        ],
    }
    assert "Nine Arch Bridge" in pdf_text(itinerary_pdf(plan))


def test_file_name_is_safe():
    assert pdf_file_name(PLAN) == "kandy-3-day-plan.pdf"
    assert pdf_file_name({"destination": "Nuwara Eliya / Hatton!"}) == (
        "nuwara-eliya-hatton-plan.pdf"
    )
    assert pdf_file_name({}) == "trip-plan.pdf"
