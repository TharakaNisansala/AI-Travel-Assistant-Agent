"""Stage 10 polish: the HTML pieces of the web page (pure functions, no Streamlit)."""

from __future__ import annotations

from ai_travel_assistant import web_ui

DAY = {
    "day": 1,
    "theme": "Hill views",
    "weather": "2026-11-18: fog; min 15 °C, max 25 °C",
    "travel": [
        {
            "from_place": "Colombo",
            "to_place": "Ella",
            "road_distance_km": 196.0,
            "driving_minutes": 312,
        }
    ],
    "activities": [
        {
            "time_of_day": "morning",
            "name": "Little Adam's Peak",
            "description": "A short hike.",
            "estimated_hours": 2.5,
            "distance_km": 1.4,
            "source": "OpenStreetMap",
        }
    ],
}
PLAN = {
    "destination": "Ella",
    "days": 1,
    "start_from": "Colombo",
    "start_assumed": True,
    "summary": "One green day.",
    "day_plans": [DAY],
    "costs": [
        {
            "item": "Ravana Cave entry",
            "quoted_currency": "LKR",
            "amount_lkr": 500.0,
            "amount_usd": 1.7,
        }
    ],
}


def test_hero_shows_name_sources_and_model():
    html = web_ui.hero_html("Cloudflare · glm-4.7-flash")
    assert web_ui.APP_NAME in html
    assert all(source in html for source in web_ui.DATA_SOURCES)
    assert "Cloudflare · glm-4.7-flash" in html


def test_hero_without_a_model_has_no_badge():
    assert "cc-badge" not in web_ui.hero_html(None)


def test_plan_header_counts_days_and_activities():
    html = web_ui.plan_header_html(PLAN)
    assert "1-day trip to Ella" in html
    assert "<b>1</b> days" in html and "<b>1</b> activities" in html
    assert "Colombo (assumed)" in html


def test_day_meta_shows_weather_and_travel():
    html = web_ui.day_meta_html(DAY)
    assert "fog" in html
    assert "Colombo → Ella · 196 km · 5 h 12 min" in html


def test_day_meta_leaves_out_unknown_travel_figures():
    day = {**DAY, "weather": None, "travel": [{"from_place": "A", "to_place": "B"}]}
    assert web_ui.day_meta_html(day) == '<span class="cc-meta">🚗 A → B</span>'


def test_timeline_shows_each_activity_with_its_tags():
    html = web_ui.timeline_html(DAY["activities"])
    assert "Little Adam&#x27;s Peak" in html  # escaped apostrophe
    assert "about 2.5 h" in html and "1.4 km away" in html
    assert "OpenStreetMap" in html


def test_costs_table_formats_both_currencies():
    html = web_ui.costs_html(PLAN["costs"])
    assert "LKR 500" in html and "USD 1.70" in html


def test_model_text_is_escaped_never_run_as_html():
    hostile = "<script>alert(1)</script>"
    day = {
        **DAY,
        "weather": hostile,
        "travel": [{"from_place": hostile, "to_place": "B"}],
        "activities": [
            {**DAY["activities"][0], "name": hostile, "description": hostile}
        ],
    }
    plan = {**PLAN, "destination": hostile, "summary": hostile, "day_plans": [day]}
    for html in (
        web_ui.plan_header_html(plan),
        web_ui.day_meta_html(day),
        web_ui.timeline_html(day["activities"]),
        web_ui.costs_html([{**PLAN["costs"][0], "item": hostile}]),
        web_ui.hero_html(hostile),
    ):
        assert "<script>" not in html
        assert "&lt;script&gt;" in html


def test_day_tab_label():
    assert web_ui.day_tab_label(DAY) == "Day 1 · Hill views"
