"""itinerary_budget: entrance fees, per-person and group costs, the budget (offline)."""

from __future__ import annotations

from datetime import date

import pytest
from langchain_core.messages import ToolMessage

from ai_travel_assistant import currency_service, pricing_service
from ai_travel_assistant.currency_service import Conversion
from ai_travel_assistant.itinerary_budget import (
    NOT_INCLUDED,
    budget_markdown,
    budget_sections,
    build_costs,
    cost_line,
    pricing_results,
)
from ai_travel_assistant.pricing_service import PriceBook, PriceRecord
from ai_travel_assistant.schemas import CostItem, Itinerary, PartyMember
from tests.test_clarification import FakeUnderstander, understood
from tests.test_itinerary_graph import (
    TODAY,
    build,
    gather,
    plan,
    scripted,
    state,
    step,
    submit,
    tools,  # noqa: F401 - pytest fixture
)

RATE = 300.0


def peak(**changes) -> PriceRecord:
    base = {
        "attraction_id": "little-adam-s-peak",
        "attraction_name": "Little Adam's Peak",
        "nationality": "local",
        "age_group": "adult",
        "amount": 300.0,
        "currency": "LKR",
        "ticket_type": "entrance",
        "unit": "person",
        "source_url": "https://example.gov.lk/peak",
        "source_authority": "Example Authority",
        "last_verified": date(2026, 11, 1),
        "status": "official",
    }
    return PriceRecord(**{**base, **changes})


PEAK = [
    peak(),
    peak(nationality="non_saarc", currency="USD", amount=10.0),
]


def artifact(*entries, rate=RATE) -> dict:
    return {
        "checked_on": "2026-11-18",
        "lkr_per_usd": rate,
        "rate_note": "Other-currency amounts use 1 USD = 300.00 LKR.",
        "problems": [],
        "attractions": list(entries),
    }


def entry(query, records=(), name=None, aid=None) -> dict:
    return {
        "query": query,
        "attraction_id": aid,
        "name": name,
        "records": [r.to_json() for r in records],
    }


def priced(art: dict, call_id="p1") -> ToolMessage:
    return ToolMessage(
        "text", tool_call_id=call_id, name="get_attraction_prices", artifact=art
    )


PEAK_ENTRY = entry(
    "Little Adam's Peak", PEAK, "Little Adam's Peak", "little-adam-s-peak"
)
TRIP = {
    "travellers": 3,
    "party": [
        {"nationality": "Sri Lankan", "age_group": "adult", "count": 2},
        {"nationality": "German", "age_group": "adult", "count": 1},
    ],
}


def itinerary(**overrides) -> Itinerary:
    return Itinerary.model_validate(plan(**overrides))


# --- Reading the tool's artifacts -----


def test_a_later_lookup_replaces_an_earlier_one():
    old = entry("Little Adam's Peak", [peak(amount=100.0)], "Little Adam's Peak", "x")
    new = entry("Little Adam's Peak", [peak(amount=300.0)], "Little Adam's Peak", "x")
    found = pricing_results([priced(artifact(old)), priced(artifact(new), "p2")])
    [only] = found.attractions
    assert only["records"][0]["amount"] == 300.0
    assert found.lkr_per_usd == RATE


def test_other_tool_messages_are_ignored():
    other = ToolMessage("x", tool_call_id="c", name="convert_currency", artifact={})
    assert pricing_results([other]).attractions == []


# --- Building the money fields -----


def test_only_attractions_in_the_plan_are_priced():
    elsewhere = entry("Sigiriya", [peak(attraction_id="sigiriya")], "Sigiriya", "s")
    money = build_costs(
        itinerary(), [priced(artifact(PEAK_ENTRY, elsewhere))], TRIP, today=TODAY
    )
    assert [f.name for f in money["entrance_fees"]] == ["Little Adam's Peak"]


def test_each_category_is_priced_and_the_budget_adds_up():
    money = build_costs(
        itinerary(costs=[]), [priced(artifact(PEAK_ENTRY))], TRIP, today=TODAY
    )
    [fees] = money["entrance_fees"]
    assert [(g.people, g.status) for g in fees.group] == [
        ("2 Sri Lankan adults", "verified"),
        ("1 Non-SAARC adult", "verified"),
    ]
    assert (
        fees.group[0].text == "2 Sri Lankan adults x LKR 300 (USD 1) = LKR 600 (USD 2)"
    )
    assert (fees.verified_lkr, fees.verified_usd) == (3600.0, 12.0)
    budget = money["budget"]
    assert (budget.total_lkr, budget.total_usd) == (3600.0, 12.0)
    assert budget.per_person_lkr is None  # a mixed group pays different prices
    assert budget.group == "2 Sri Lankan adults, 1 non-SAARC foreign adult"
    assert budget.not_included == [
        "Accommodation (no hotel suggestions were made)",
        *NOT_INCLUDED,
    ]


def test_an_unpublished_category_is_listed_not_guessed():
    saarc_trip = {
        "party": [{"nationality": "Indian", "age_group": "adult", "count": 2}]
    }
    money = build_costs(
        itinerary(costs=[]), [priced(artifact(PEAK_ENTRY))], saarc_trip, today=TODAY
    )
    budget = money["budget"]
    assert budget.total_lkr == 0.0
    assert budget.not_verified == ["Little Adam's Peak: 2 SAARC adults"]


def test_an_attraction_without_a_price_requires_confirmation():
    missing = entry("Little Adam's Peak")
    money = build_costs(
        itinerary(costs=[]), [priced(artifact(missing))], TRIP, today=TODAY
    )
    assert money["budget"].requires_confirmation == ["Little Adam's Peak"]


def test_a_uniform_group_gets_a_per_person_figure():
    local = {"citizenship": "local", "travellers": 2, "children": 0}
    money = build_costs(
        itinerary(costs=[]), [priced(artifact(PEAK_ENTRY))], local, today=TODAY
    )
    assert money["budget"].per_person_lkr == 300.0


def test_nothing_to_price_means_no_budget():
    money = build_costs(itinerary(costs=[]), [], TRIP, today=TODAY)
    assert money == {
        "entrance_fees": [],
        "cost_lines": [],
        "stays": None,
        "budget": None,
    }


# --- Guide costs: per person and for the group -----


def guide_cost(**changes) -> CostItem:
    base = {"item": "Guesthouse room", "quoted_currency": "LKR", "amount_lkr": 5000}
    return CostItem(**{**base, **changes})


def test_a_shared_cost_is_split_per_person():
    line = cost_line(guide_cost(per="group", quantity=4, amount_usd=16.0), 3)
    assert (line.group_lkr, line.group_usd) == (20000.0, 64.0)
    assert line.per_person_lkr == pytest.approx(20000 / 3)


def test_a_per_person_cost_is_multiplied_by_the_group():
    line = cost_line(guide_cost(item="Boat trip", per="person", amount_lkr=2000), 3)
    assert (line.per_person_lkr, line.group_lkr) == (2000.0, 6000.0)
    assert line.group_usd is None  # no conversion was retrieved


def test_without_a_head_count_the_group_side_is_unknown():
    line = cost_line(guide_cost(per="person"), None)
    assert line.group_lkr is None and line.per_person_lkr == 5000.0


def test_guide_costs_join_the_total_as_estimates():
    room = {
        "item": "Guesthouse room",
        "quoted_currency": "LKR",
        "amount_lkr": 5000,
        "quantity": 2,
        "quantity_note": "2 rooms x 1 night, assuming 2 people per room",
    }
    money = build_costs(
        itinerary(costs=[room]), [priced(artifact(PEAK_ENTRY))], TRIP, today=TODAY
    )
    budget = money["budget"]
    assert budget.other_estimated_lkr == 10000.0
    assert budget.total_lkr == 3600.0 + 10000.0
    assert budget.total_usd is None  # the room has no USD figure: never guessed


# --- The shared layout -----


def saved(**changes) -> dict:
    checked = itinerary()
    money = build_costs(checked, [priced(artifact(PEAK_ENTRY))], TRIP, today=TODAY)
    return checked.model_copy(update=money).model_dump(mode="json") | changes


def test_budget_sections_cover_fees_and_the_total():
    titles = [title for title, _ in budget_sections(saved())]
    assert titles[0] == "Entrance fees (official sources)"
    assert titles[-1] == "Trip budget"
    text = "\n".join(budget_markdown(saved()))
    assert "- **Little Adam's Peak**" in text
    assert "  - Sri Lankan adult: LKR 300 (USD 1) · Confidence: High" in text
    assert "  - SAARC adult: Not verified. No official price" in text
    assert "  - Your group: 1 Non-SAARC adult x LKR 3,000 (USD 10)" in text
    assert (
        "  - Source: Example Authority (official), https://example.gov.lk/peak" in text
    )
    assert "- TOTAL estimated budget for the group:" in text
    assert "- Per person: differs by visitor category" in text
    assert "  - Food and drinks" in text


def test_an_old_itinerary_without_a_budget_has_no_sections():
    assert budget_sections(itinerary().model_dump(mode="json")) == []


def test_the_pdf_shows_the_budget():
    pypdf = pytest.importorskip("pypdf")
    import io

    from ai_travel_assistant.itinerary_pdf import itinerary_pdf

    reader = pypdf.PdfReader(io.BytesIO(itinerary_pdf(saved(), today=TODAY)))
    text = " ".join(page.extract_text() for page in reader.pages)
    assert "Entrance fees (official sources)" in text
    assert "Trip budget" in text
    assert "Not included" in text


# --- The whole graph: tool artifact to checked itinerary -----


@pytest.fixture
def prices(monkeypatch):
    monkeypatch.setattr(pricing_service, "load_price_book", lambda: PriceBook(PEAK))

    def convert(amount, source, target, **kwargs):
        return Conversion(amount, source, target, RATE, amount * RATE, None)

    monkeypatch.setattr(currency_service, "convert", convert)


@pytest.mark.usefixtures("tools", "prices")
def test_the_plan_gets_the_group_fees_from_the_tool_artifact():
    party = [
        PartyMember(nationality="Sri Lankan", count=2),
        PartyMember(nationality="German"),
    ]
    understander = FakeUnderstander(
        {
            "Plan 2 days in Ella": understood(
                "plan",
                destination="Ella",
                days=2,
                origin="Colombo",
                party=party,
                wants_hotels=False,
            )
        }
    )
    lookup = step(("get_attraction_prices", {"attractions": ["Little Adam's Peak"]}))
    model = scripted(gather(), lookup, submit(plan(costs=[])))
    graph = build(model, understander=understander)

    from ai_travel_assistant.tool_calling import ask_agent, final_answer

    turn = ask_agent(graph, "t1", "Plan 2 days in Ella")

    [result] = [m for m in turn if getattr(m, "name", "") == "get_attraction_prices"]
    assert result.artifact["attractions"][0]["name"] == "Little Adam's Peak"
    assert "lkr_per_usd" not in result.content  # the LLM sees text only
    saved_plan = state(graph)["itinerary"]
    [fees] = saved_plan["entrance_fees"]
    assert [g["people"] for g in fees["group"]] == [
        "2 Sri Lankan adults",
        "1 Non-SAARC adult",
    ]
    assert saved_plan["budget"]["total_lkr"] == 3600.0
    reply = final_answer(turn)
    assert "## Entrance fees (official sources)" in reply
    assert "TOTAL estimated budget for the group: LKR 3,600 (USD 12)" in reply
