"""Hotel suggestions: hotel_service, the suggest_hotels tool, the stay budget.

Offline: Wikivoyage wikitext for Kandy and Ella is saved in tests/fixtures
(read 2026-11-19); HTTP goes through httpx.MockTransport.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import httpx
import pytest
from langchain_core.messages import ToolMessage

from ai_travel_assistant import currency_service, guide_service, hotel_service, tools
from ai_travel_assistant.currency_service import Conversion
from ai_travel_assistant.hotel_service import (
    Listing,
    Stays,
    choose,
    fetch_stays,
    parse_listings,
    parse_price,
)
from ai_travel_assistant.itinerary_budget import (
    accommodation_estimate,
    budget_markdown,
    build_costs,
    hotel_results,
)
from ai_travel_assistant.schemas import Itinerary
from tests.test_itinerary_graph import plan

FIXTURES = Path(__file__).parent / "fixtures"
TODAY = date(2026, 11, 19)
RATE = 300.0


def wikitext(place: str) -> str:
    data = json.loads(
        (FIXTURES / f"wikivoyage_{place}_wikitext.json").read_text(encoding="utf-8")
    )
    return data["parse"]["wikitext"]


KANDY = parse_listings(wikitext("kandy"))
ELLA = parse_listings(wikitext("ella"))


def named(listings, name) -> Listing:
    [found] = [x for x in listings if x.name == name]
    return found


# --- Parsing the guide's listings ---------------------------------------------


def test_every_listing_in_the_sleep_section_is_read_with_its_level():
    assert len(KANDY) == 28
    levels = {x.tier for x in KANDY}
    assert levels == {"budget", "mid-range", "splurge", "other"}
    assert named(KANDY, "Burmese Rest").tier == "budget"
    assert named(KANDY, "Queen's Hotel").tier == "mid-range"
    assert named(KANDY, "Cinnamon Citadel Kandy").tier == "splurge"


def test_a_listing_keeps_its_price_url_and_year():
    house = named(KANDY, "Kandy City House")
    assert house.price_text == "From Rs 2000"
    assert house.listed_year == 2018  # from lastedit
    assert (house.currency, house.low, house.high) == ("LKR", 2000.0, 2000.0)
    cottage = named(KANDY, "Kandy Cottage")
    assert cottage.url == "http://www.kandycottage.com"
    assert cottage.basis == "person"  # "US$22-40 per person"
    assert (cottage.currency, cottage.low, cottage.high) == ("USD", 22.0, 40.0)


def test_a_year_written_in_the_price_counts_as_the_listing_year():
    bungalow = named(ELLA, "Sunnyside Holiday Bungalow")
    assert bungalow.price_text == "Rs 3000/3500 (2014)"
    assert bungalow.listed_year == 2014
    assert (bungalow.low, bungalow.high) == (3000.0, 3500.0)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Room from Rs 4,500", ("LKR", 4500.0, 4500.0, "room")),
        ("Rs 1,300-1,700", ("LKR", 1300.0, 1700.0, "room")),
        ("Double 2500Rs", ("LKR", 2500.0, 2500.0, "room")),
        ("US$53-60(2014)", ("USD", 53.0, 60.0, "room")),
        ("From USD 29 (2019)", ("USD", 29.0, 29.0, "room")),
        ("US$5-10 person", ("USD", 5.0, 10.0, "person")),
        ("Ask at the desk", (None, None, None, "room")),
        ("", (None, None, None, "room")),
    ],
)
def test_parse_price(text, expected):
    assert parse_price(text) == expected


def test_no_sleep_section_means_no_listings():
    assert parse_listings("==See==\n* {{see | name=Fort }}\n") == []


def test_listing_templates_of_other_types_are_skipped():
    text = (
        "==Sleep==\n* {{listing | type=eat | name=Cafe }}\n"
        "* {{listing | type=sleep | name=Lake Inn | price=Rs 5000 }}\n"
    )
    assert [x.name for x in parse_listings(text)] == ["Lake Inn"]


def test_nested_templates_and_links_do_not_split_fields():
    text = (
        "==Sleep==\n===Budget===\n* {{sleep | name=[[Kandy|Lake]] View "
        "| phone={{phone|+94 81 222}} | price=Rs 3000 | lastedit=2025-01-02 }}\n"
    )
    [stay] = parse_listings(text)
    assert stay.name == "Lake View"
    assert (stay.tier, stay.low, stay.listed_year) == ("budget", 3000.0, 2025)


# --- Choosing by level or amount --------------------------------------------------


def test_a_level_picks_its_listings_newest_first():
    choice = choose(KANDY, tier="splurge", max_lkr=None, lkr_per_usd=RATE)
    assert {x.tier for x in choice.fits} == {"splurge"}
    years = [x.listed_year or 0 for x in choice.fits]
    assert years == sorted(years, reverse=True)
    assert len(choice.fits) <= hotel_service.MAX_LISTINGS


def test_an_amount_keeps_readable_prices_that_fit_and_lists_the_rest():
    choice = choose(KANDY, tier="budget", max_lkr=2500, lkr_per_usd=RATE)
    assert {x.name for x in choice.fits} == {"Burmese Rest", "Kandy City House"}
    # Per-person prices cannot be compared with a room budget.
    assert "Kandy Cottage" in {x.name for x in choice.unpriced}


def test_usd_prices_are_compared_in_lkr():
    choice = choose(KANDY, tier="mid-range", max_lkr=6000, lkr_per_usd=RATE)
    names = {x.name for x in choice.fits}
    assert "Heavens Holiday Resort" in names  # US$15 = LKR 4,500
    assert "Riverdale Hotel" not in names  # US$42 = LKR 12,600


def test_a_missing_level_falls_back_to_every_listing_and_says_so():
    choice = choose(ELLA, tier="splurge", max_lkr=None, lkr_per_usd=RATE)
    assert "lists no luxury places" in choice.note
    assert len(choice.fits) == len(ELLA)


# --- Fetching ---------------------------------------------------------------------


def test_fetch_reads_the_wikitext_once_and_caches_it():
    hotel_service.WIKITEXT_CACHE.clear()
    calls = []

    def handle(request):
        calls.append(request)
        body = (FIXTURES / "wikivoyage_kandy_wikitext.json").read_text(encoding="utf-8")
        return httpx.Response(200, text=body)

    with guide_service.make_client(httpx.MockTransport(handle)) as client:
        first = fetch_stays("Kandy", client=client)
        second = fetch_stays("kandy", client=client)
    assert len(calls) == 1 and first is second
    assert calls[0].url.params["prop"] == "wikitext"
    assert first.article_url == "https://en.wikivoyage.org/wiki/Kandy"
    assert len(first.listings) == 28
    hotel_service.WIKITEXT_CACHE.clear()


# --- The tool -----------------------------------------------------------------------


@pytest.fixture
def offline(monkeypatch):
    stays = Stays("Kandy", "Kandy", "https://en.wikivoyage.org/wiki/Kandy", KANDY)
    monkeypatch.setattr(hotel_service, "fetch_stays", lambda name, **kw: stays)

    def convert(amount, source, target, **kwargs):
        return Conversion(amount, source, target, RATE, amount * RATE, None)

    monkeypatch.setattr(currency_service, "convert", convert)
    monkeypatch.setattr(
        tools, "date", type("D", (), {"today": staticmethod(lambda: TODAY)})
    )


def ask(trip: dict) -> tuple[str, dict]:
    message = tools.suggest_hotels.invoke(
        {
            "type": "tool_call",
            "id": "h1",
            "name": "suggest_hotels",
            "args": {"destination": "Kandy", "state": {"trip_request": trip}},
        }
    )
    return message.content, message.artifact


@pytest.mark.usefixtures("offline")
def test_the_tool_lists_the_level_with_years_and_says_it_cannot_book():
    text, artifact = ask({"budget": "luxury"})
    assert text.startswith("Places to stay in Kandy (luxury places)")
    assert "- Cinnamon Citadel Kandy (Luxury): From US$82 (2021) (listed 2021" in text
    assert "listed 2014, may have changed" in text
    assert "cannot book rooms or check availability" in text
    assert artifact["level"] == "splurge"
    assert {x["tier"] for x in artifact["listings"]} == {"splurge"}


@pytest.mark.usefixtures("offline")
def test_the_tool_reads_a_usd_room_budget_in_lkr():
    text, artifact = ask({"room_budget": 10, "room_budget_currency": "USD"})
    assert artifact["max_lkr"] == 3000.0
    assert "up to LKR 3,000 a room a night" in text
    assert "1 USD = 300.00 LKR" in text


def test_the_tool_reports_a_guide_outage(monkeypatch):
    def down(name, **kwargs):
        raise guide_service.GuideNetworkError("offline")

    monkeypatch.setattr(hotel_service, "fetch_stays", down)
    text, artifact = ask({})
    assert "Hotel suggestions unavailable" in text and artifact == {}


def test_the_llm_sees_only_the_destination():
    schema = tools.suggest_hotels.tool_call_schema.model_json_schema()
    assert list(schema["properties"]) == ["destination"]


# --- In the itinerary: where to stay, and the stay in the budget ------------------


def artifact_for(listings, **extra) -> dict:
    return {
        "destination": "Ella",
        "title": "Ella",
        "article_url": "https://en.wikivoyage.org/wiki/Ella",
        "level": "mid-range",
        "max_lkr": None,
        "lkr_per_usd": RATE,
        "checked_on": TODAY.isoformat(),
        "note": "",
        "listings": [x.to_json() for x in listings],
        "unpriced": [],
        **extra,
    }


RECENT = Listing("Lake Inn", "mid-range", "Rs 8,000", listed_year=2025,
                 currency="LKR", low=6000.0, high=8000.0)  # fmt: skip
OLD = Listing("Old Lodge", "mid-range", "Rs 3,000 (2014)", listed_year=2014,
              currency="LKR", low=3000.0, high=3000.0)  # fmt: skip


def test_only_recent_prices_make_the_estimate():
    money, note = accommodation_estimate(
        artifact_for([RECENT, OLD]), days=3, travellers=3, today=TODAY
    )
    # 3 people -> 2 rooms; 3 days -> 2 nights; the highest recent price.
    assert money.lkr == 8000.0 * 2 * 2
    assert money.usd == pytest.approx(32000 / RATE)
    assert "2 rooms (assuming 2 people per room) x 2 nights" in note
    assert "Lake Inn" in note and "upper estimate" in note


def test_old_prices_alone_make_no_estimate():
    money, note = accommodation_estimate(
        artifact_for([OLD]), days=3, travellers=2, today=TODAY
    )
    assert money is None
    assert "no price listed in the last 3 years" in note


def test_a_day_trip_has_no_stay():
    money, note = accommodation_estimate(
        artifact_for([RECENT]), days=1, travellers=2, today=TODAY
    )
    assert money is None and "one-day trip" in note


def test_a_per_person_price_counts_people_not_rooms():
    per_person = Listing(
        "Hostel",
        "budget",
        "US$10 per person",
        listed_year=2025,
        currency="USD",
        low=10.0,
        high=10.0,
        basis="person",
    )
    money, note = accommodation_estimate(
        artifact_for([per_person]), days=2, travellers=3, today=TODAY
    )
    assert money.usd == 30.0 and money.lkr == 9000.0
    assert note.startswith("3 people x 1 night")


def stays_message(art) -> ToolMessage:
    return ToolMessage("text", tool_call_id="h1", name="suggest_hotels", artifact=art)


def test_hotel_results_match_the_plan_destination():
    ella = stays_message(artifact_for([RECENT]))
    kandy = stays_message(
        {**artifact_for([OLD]), "destination": "Kandy", "title": "Kandy"}
    )
    assert hotel_results([ella, kandy], "Ella")["listings"][0]["name"] == "Lake Inn"
    assert hotel_results([kandy], "Ella") is None


def test_the_plan_shows_where_to_stay_and_adds_the_stay_to_the_budget():
    trip = {"travellers": 2, "citizenship": "local", "children": 0}
    itinerary = Itinerary.model_validate(plan(costs=[]))  # 2 days in Ella
    money = build_costs(
        itinerary, [stays_message(artifact_for([RECENT, OLD]))], trip, today=TODAY
    )
    stays, budget = money["stays"], money["budget"]
    assert [s.name for s in stays.items] == ["Lake Inn", "Old Lodge"]
    assert [s.recent for s in stays.items] == [True, False]
    assert budget.accommodation_lkr == 8000.0  # 1 room x 1 night
    assert budget.total_lkr == 8000.0
    text = "\n".join(
        budget_markdown(itinerary.model_copy(update=money).model_dump(mode="json"))
    )
    assert "## Where to stay (mid-range places)" in text
    assert "- **Lake Inn (Mid-range)**" in text
    assert "  - Price: Rs 3,000 (2014) (listed 2014, may have changed)" in text
    assert "not booked and availability not checked" in text
    assert "- **Accommodation (upper estimate): LKR 8,000" in text


def test_without_hotels_the_budget_says_accommodation_is_not_included():
    trip = {"travellers": 1, "citizenship": "local", "wants_hotels": False}
    itinerary = Itinerary.model_validate(plan())
    budget = build_costs(itinerary, [], trip, today=TODAY)["budget"]
    assert budget.not_included[0] == "Accommodation (you said you have your own)"
    assert budget.accommodation_lkr is None
