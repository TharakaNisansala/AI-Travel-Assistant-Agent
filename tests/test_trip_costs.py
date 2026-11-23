"""trip_costs and the get_attraction_prices tool (offline).

Prices: the saved CCF page (foreign prices only) plus hand-made "curated"
records, so local, SAARC, child and vehicle rules can all be exercised.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from ai_travel_assistant import currency_service, pricing_service, tools
from ai_travel_assistant.currency_service import Conversion, CurrencyNetworkError
from ai_travel_assistant.party import group_breakdown
from ai_travel_assistant.pricing_service import PriceBook, PriceRecord, parse_ccf
from ai_travel_assistant.trip_costs import (
    ZERO,
    Money,
    attraction_cost,
    record_money,
    summarise,
)

TODAY = date(2026, 11, 19)
CCF = parse_ccf(
    (Path(__file__).parent / "fixtures" / "ccf_ticket_prices.html").read_text(
        encoding="utf-8"
    ),
    TODAY,
)
RATE = 300.0  # LKR per USD, for amounts a source gives in one currency only


def curated(**changes) -> PriceRecord:
    base = {
        "attraction_id": "example-gardens",
        "attraction_name": "Example Gardens",
        "nationality": "local",
        "age_group": "adult",
        "amount": 300.0,
        "currency": "LKR",
        "ticket_type": "entrance",
        "unit": "person",
        "source_url": "https://example.gov.lk/fees",
        "source_authority": "Example Authority",
        "last_verified": TODAY,
        "status": "official",
    }
    return PriceRecord(**{**base, **changes})


GARDENS = [
    curated(),
    curated(age_group="child", min_age=3, max_age=12, amount=100.0),
    curated(nationality="saarc", currency="USD", amount=5.0),
    curated(nationality="non_saarc", currency="USD", amount=10.0),
    curated(ticket_type="vehicle", unit="vehicle", amount=500.0),
]
BOOK = PriceBook([*GARDENS, *CCF])

MIXED = {
    "party": [
        {"nationality": "Sri Lankan", "age_group": "adult", "count": 2},
        {"nationality": "Indian", "age_group": "adult", "count": 1},
        {"nationality": "German", "age_group": "adult", "count": 1},
    ]
}


def cost(name: str, trip: dict, rate: float | None = RATE):
    return attraction_cost(
        BOOK.lookup(name), group_breakdown(trip), today=TODAY, lkr_per_usd=rate
    )


# --- Money ------------------------------------------------------------------------


def test_a_source_s_own_lkr_figure_is_used_as_given():
    sigiriya = next(r for r in CCF if r.attraction_id == "sigiriya")
    # CCF converts at 336; today's rate (300) must not change its LKR figure.
    assert record_money(sigiriya, RATE) == Money(11760.0, 35.0)


def test_the_other_currency_comes_from_the_rate_or_is_left_out():
    assert record_money(curated(amount=3000.0), RATE) == Money(3000.0, 10.0)
    assert record_money(curated(amount=3000.0), None) == Money(3000.0, None)
    usd_only = curated(currency="USD", amount=10.0)
    assert record_money(usd_only, RATE) == Money(3000.0, 10.0)


def test_a_missing_side_stays_missing_when_added():
    assert (Money(100.0, None) + Money(50.0, 2.0)) == Money(150.0, None)
    assert Money(10.0, 1.0).times(3) == Money(30.0, 3.0)


# --- One attraction, a mixed group --------------------------------------------------


def test_each_category_pays_its_own_price():
    gardens = cost("Example Gardens", MIXED)
    lines = {(g.nationality, g.age_group): g for g in gardens.group}
    assert lines[("local", "adult")].subtotal == Money(600.0, 2.0)
    assert lines[("saarc", "adult")].subtotal == Money(1500.0, 5.0)
    assert lines[("non_saarc", "adult")].subtotal == Money(3000.0, 10.0)
    assert lines[("local", "adult")].each == Money(300.0, 1.0)
    assert gardens.verified_total == Money(5100.0, 17.0)
    assert gardens.unverified == []


def test_ccf_prices_only_the_non_saarc_visitor_and_lists_the_rest():
    sigiriya = cost("Sigiriya", MIXED)
    lines = {g.nationality: g for g in sigiriya.group}
    assert lines["non_saarc"].subtotal == Money(11760.0, 35.0)
    assert lines["local"].subtotal is None  # never the foreign price
    assert lines["saarc"].subtotal is None
    assert sigiriya.verified_total == Money(11760.0, 35.0)
    assert [g.nationality for g in sigiriya.unverified] == ["local", "saarc"]


def test_every_category_is_shown_even_if_nobody_in_the_group_has_it():
    sigiriya = cost("Sigiriya", MIXED)
    shown = [(p.nationality, p.age_group, p.verified) for p in sigiriya.categories]
    assert shown == [
        ("local", "adult", False),
        ("local", "child", False),
        ("saarc", "adult", False),
        ("saarc", "child", False),
        ("non_saarc", "adult", True),
        ("non_saarc", "child", False),
    ]


def test_a_child_with_a_published_age_rule_gets_the_child_price():
    trip = {"citizenship": "local", "travellers": 3, "children": 1}
    gardens = cost("Example Gardens", trip)
    child = next(g for g in gardens.group if g.age_group == "child")
    assert child.subtotal == Money(100.0, 100 / RATE)
    assert child.estimate is None


def test_a_child_without_an_age_rule_gets_a_labelled_upper_estimate():
    trip = {"citizenship": "non_saarc", "travellers": 3, "children": 1}
    sigiriya = cost("Sigiriya", trip)
    child = next(g for g in sigiriya.group if g.age_group == "child")
    assert child.subtotal is None  # CCF's half ticket has no age rule
    assert child.estimate == Money(11760.0, 35.0)  # the adult price, as a cap
    assert "upper estimate" in child.estimate_note
    assert sigiriya.verified_total == Money(23520.0, 70.0)  # adults only
    assert sigiriya.estimated_total == Money(11760.0, 35.0)


def test_vehicle_charges_are_listed_not_added():
    gardens = cost("Example Gardens", {"citizenship": "local", "travellers": 1})
    assert [e.ticket_type for e in gardens.extras] == ["vehicle"]
    assert gardens.verified_total == Money(300.0, 1.0)


def test_an_unclear_citizenship_is_never_priced():
    trip = {"party": [{"nationality": "dual", "age_group": "adult", "count": 1}]}
    [line] = cost("Example Gardens", trip).group
    assert line.subtotal is None and line.estimate is None
    assert "Citizenship not clear" in line.price.note


def test_without_a_rate_the_usd_total_is_unknown_not_zero():
    gardens = cost("Example Gardens", MIXED, rate=None)
    assert gardens.verified_total.lkr is None  # USD prices cannot become LKR
    assert gardens.verified_total.usd is None  # LKR prices cannot become USD


# --- Several attractions ----------------------------------------------------------


def test_the_summary_adds_verified_fees_and_lists_the_gaps():
    costs = [cost(n, MIXED) for n in ("Example Gardens", "Sigiriya", "Ella Rock")]
    summary = summarise(costs, travellers=4)
    assert summary.verified == Money(5100.0 + 11760.0, 17.0 + 35.0)
    assert summary.estimated == ZERO
    assert summary.unverified == [
        "Sigiriya: 2 Sri Lankan adults",
        "Sigiriya: 1 SAARC adult",
    ]
    assert summary.not_found == ["Ella Rock"]


# --- The tool -----------------------------------------------------------------------


@pytest.fixture
def offline(monkeypatch):
    monkeypatch.setattr(pricing_service, "load_price_book", lambda: BOOK)

    def convert(amount, source, target, **kwargs):
        return Conversion(amount, source, target, RATE, amount * RATE, None)

    monkeypatch.setattr(currency_service, "convert", convert)


def call(attractions, trip=None) -> str:
    return tools.get_attraction_prices.invoke(
        {"attractions": attractions, "state": {"trip_request": trip or {}}}
    )


@pytest.mark.usefixtures("offline")
def test_the_tool_shows_every_category_the_group_and_the_totals():
    text = call(["Sigiriya Rock Fortress", "Example Gardens"], MIXED)
    assert "Sigiriya:" in text
    assert "- Sri Lankan adult: Not verified. No official Sri Lankan price" in text
    assert "- SAARC adult: Not verified." in text
    assert "- Non-SAARC adult: LKR 11,760 (USD 35) · Confidence: Medium" in text
    assert "Source: Central Cultural Fund (CCF), Sri Lanka (official), " in text
    assert "https://ccf.gov.lk/is/ (source updated 2026-10-14" in text
    assert "- 1 Non-SAARC adult x LKR 11,760 (USD 35) = LKR 11,760 (USD 35)" in text
    assert "- 2 Sri Lankan adults: not verified" in text
    assert "Separate vehicle (not in the entrance total)" in text
    assert (
        "Group: 2 Sri Lankan adults, 1 SAARC adult, 1 non-SAARC foreign adult." in text
    )
    assert "Verified entrance fees for the group: LKR 16,860 (USD 52)" in text
    assert (
        "Not verified, could change the budget: Sigiriya: 2 Sri Lankan adults" in text
    )
    assert "1 USD = 300.00 LKR" in text
    assert "Never fill in a 'Not verified' price" in text


@pytest.mark.usefixtures("offline")
def test_the_tool_without_a_known_group_gives_prices_but_no_total():
    text = call(["Sigiriya"])
    assert "Non-SAARC adult: LKR 11,760" in text
    assert "no group total was worked out" in text
    assert "For this group" not in text


@pytest.mark.usefixtures("offline")
def test_an_unknown_attraction_requires_confirmation():
    text = call(["Ella Rock"], MIXED)
    assert "Ella Rock: no price found in any official source" in text
    assert "Requires confirmation: Ella Rock" in text


def test_the_tool_survives_an_exchange_rate_outage(monkeypatch):
    monkeypatch.setattr(pricing_service, "load_price_book", lambda: BOOK)

    def down(*args, **kwargs):
        raise CurrencyNetworkError("offline")

    monkeypatch.setattr(currency_service, "convert", down)
    text = call(["Example Gardens"], {"citizenship": "local", "travellers": 1})
    assert "Exchange rate unavailable" in text
    assert "- 1 Sri Lankan adult x LKR 300 = LKR 300" in text


def test_the_llm_never_sees_the_state_argument():
    schema = tools.get_attraction_prices.tool_call_schema.model_json_schema()
    assert list(schema["properties"]) == ["attractions"]
