"""sri_lanka_places: the fixed province and district lists."""

from __future__ import annotations

from ai_travel_assistant.sri_lanka_places import (
    DISTRICTS,
    PROVINCES,
    province_of,
    weather_question,
)


def test_nine_provinces_and_twenty_five_districts():
    assert len(PROVINCES) == 9
    assert len(DISTRICTS) == 25
    assert len(set(DISTRICTS)) == 25  # no district in two provinces


def test_a_district_knows_its_province():
    assert province_of("Badulla") == "Uva"
    assert province_of(" nuwara eliya ") == "Central"
    assert province_of("Ella") is None  # a town, not a district


def test_the_question_names_only_the_district():
    assert weather_question("Monaragala") == "What's the weather in Monaragala?"
