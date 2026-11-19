"""The money part of a checked itinerary: entrance fees, other costs, the budget.

Runs inside compose_answer, after grounding, with no LLM call and no network:
everything comes from what the tools already returned in this thread.

What this file demonstrates:
  1. Tool artifacts. get_attraction_prices returns text for the LLM AND an
     artifact (the raw price records), which LangGraph keeps on the
     ToolMessage. The itinerary recomputes the fees from those records for
     the group in the trip state, so the plan's figures never pass through
     the model and always match the current group.
  2. Only what the plan visits. A priced attraction is included when an
     activity in the plan names it ("Sigiriya Rock Fortress" -> Sigiriya).
  3. Per person and for the group. Each guide cost says whether its price is
     per person or shared, and how many the trip needs; code multiplies.
  4. One honest total. The budget adds verified entrance fees and labelled
     estimates, lists what is not verified, and names what no tool covers
     (food, transport fares, guides), instead of guessing those.
  5. One layout for three screens. budget_sections() returns titled lists of
     lines; the Markdown reply, the web cards and the PDF only lay them out.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from langchain_core.messages import AnyMessage, ToolMessage

from ai_travel_assistant.hotel_service import RECENT_YEARS, TIER_LABELS, Listing
from ai_travel_assistant.party import describe_group, group_breakdown
from ai_travel_assistant.pricing_service import (
    AttractionPrices,
    PriceRecord,
    name_words,
)
from ai_travel_assistant.schemas import (
    AttractionFees,
    Budget,
    CostItem,
    CostLine,
    FeeCategory,
    FeeLine,
    Itinerary,
    StayItem,
    StaySuggestions,
)
from ai_travel_assistant.trip_costs import (
    CATEGORY_LABELS,
    ZERO,
    AttractionCost,
    Money,
    attraction_cost,
    category_text,
    entrance_conditions,
    extra_text,
    format_money,
    group_line_text,
    source_lines,
    summarise,
)

PRICES_TOOL = "get_attraction_prices"

# No tool in this app has reliable prices for these, so the total leaves
# them out and says so.
NOT_INCLUDED = (
    "Food and drinks",
    "Transport fares (buses, trains, tuk-tuks, a driver)",
    "Guides, tips and personal spending",
)


# --- Reading the pricing tool's artifacts -----


@dataclass
class PricingResults:
    attractions: list[dict] = field(default_factory=list)  # latest per name
    lkr_per_usd: float | None = None
    rate_note: str = ""
    problems: list[str] = field(default_factory=list)


def pricing_results(messages: Sequence[AnyMessage]) -> PricingResults:
    """Every attraction priced in this thread; a later lookup replaces an earlier."""
    found = PricingResults()
    latest: dict[str, dict] = {}
    for message in messages:
        if not (
            isinstance(message, ToolMessage)
            and message.name == PRICES_TOOL
            and isinstance(message.artifact, Mapping)
            and message.artifact.get("attractions")
        ):
            continue
        artifact = message.artifact
        for entry in artifact["attractions"]:
            key = (entry.get("attraction_id") or entry.get("query") or "").lower()
            latest[key] = dict(entry)
        found.lkr_per_usd = artifact.get("lkr_per_usd")
        found.rate_note = artifact.get("rate_note") or ""
        found.problems = list(artifact.get("problems") or [])
    found.attractions = list(latest.values())
    return found


def _in_plan(entry: Mapping, itinerary: Itinerary) -> bool:
    """True if an activity names this attraction (all of its words, or vice versa)."""
    keys = [set(name_words(entry.get("name") or "")), set(name_words(entry["query"]))]
    for plan in itinerary.day_plans:
        for activity in plan.activities:
            words = set(name_words(activity.name))
            if any(key and key <= words for key in keys):
                return True
            if words and words <= keys[1]:
                return True
    return False


# --- Places to stay ------------------------------------------------------------

HOTELS_TOOL = "suggest_hotels"
PEOPLE_PER_ROOM = 2


def hotel_results(
    messages: Sequence[AnyMessage], destination: str
) -> dict[str, Any] | None:
    """The latest suggest_hotels artifact for this destination, or None."""
    wanted = set(name_words(destination))
    found = None
    for message in messages:
        if not (
            isinstance(message, ToolMessage)
            and message.name == HOTELS_TOOL
            and isinstance(message.artifact, Mapping)
            and "listings" in message.artifact
        ):
            continue
        art = message.artifact
        names = set(name_words(art.get("title") or "")) | set(
            name_words(art.get("destination") or "")
        )
        if wanted and (wanted <= names or names <= wanted):
            found = dict(art)
    return found


def stay_suggestions(art: Mapping[str, Any], today: date) -> StaySuggestions:
    listings = [Listing.from_json(x) for x in art.get("listings") or []]
    wanted = []
    if art.get("level"):
        wanted.append(f"{TIER_LABELS[art['level']].lower()} places")
    if art.get("max_lkr"):
        wanted.append(f"up to LKR {art['max_lkr']:,.0f} a room a night")
    return StaySuggestions(
        source_url=art.get("article_url") or "",
        wanted=", ".join(wanted),
        items=[
            StayItem(
                name=x.name,
                level=TIER_LABELS[x.tier],
                price_text=x.price_text,
                listed_year=x.listed_year,
                recent=x.is_recent(today),
                url=x.url,
                description=x.description,
            )
            for x in listings
        ],
        unpriced=[x["name"] for x in art.get("unpriced") or []],
        note=art.get("note") or "",
    )


def accommodation_estimate(
    art: Mapping[str, Any],
    *,
    days: int,
    travellers: int | None,
    today: date,
) -> tuple[Money | None, str]:
    """An upper estimate for the stay from recent prices, or (None, reason).

    Rooms are the travellers two to a room (stated as an assumption); a price
    per person is multiplied by the travellers instead. Only listings from
    the last RECENT_YEARS years count.
    """
    nights = max(days - 1, 0)
    if nights == 0:
        return None, "a one-day trip has no night's stay"
    rate = art.get("lkr_per_usd")
    rooms = math.ceil(travellers / PEOPLE_PER_ROOM) if travellers else 1
    best: tuple[Money, Listing, int] | None = None
    for listing in (Listing.from_json(x) for x in art.get("listings") or []):
        if not listing.is_recent(today) or listing.high is None:
            continue
        if listing.basis == "person":
            if not travellers:
                continue
            units = travellers
        else:
            units = rooms
        total = listing.high * units * nights
        if listing.currency == "LKR":
            money = Money(total, total / rate if rate else None)
        else:
            money = Money(total * rate if rate else None, total)
        key = money.lkr if money.lkr is not None else (money.usd or 0) * 1000
        if best is None or key > (best[0].lkr or (best[0].usd or 0) * 1000):
            best = (money, listing, units)
    if best is None:
        return None, (
            f"no price listed in the last {RECENT_YEARS} years for the suggested places"
        )
    money, listing, units = best
    who = (
        f"{units} {'person' if units == 1 else 'people'}"
        if listing.basis == "person"
        else f"{units} room{'s' if units != 1 else ''} "
        f"(assuming {PEOPLE_PER_ROOM} people per room)"
    )
    note = (
        f"{who} x {nights} night{'s' if nights != 1 else ''}, at the highest "
        f"recently listed price ({listing.name}: {listing.price_text}, listed "
        f"{listing.listed_year}). An upper estimate; book and check prices with "
        "the hotel."
    )
    return money, note


# --- Building the money fields -----


def _fees(cost: AttractionCost, lkr_per_usd: float | None) -> AttractionFees:
    if not cost.found:
        return AttractionFees(query=cost.query, name=None)
    status = {True: "verified", False: "not_verified"}
    lines = []
    for g in cost.group:
        kind = status[g.subtotal is not None]
        if g.subtotal is None and g.estimate is not None:
            kind = "estimate"
        money = g.subtotal or g.estimate
        lines.append(
            FeeLine(
                people=g.people,
                status=kind,  # type: ignore[arg-type]
                text=group_line_text(g),
                total_lkr=money.lkr if money else None,
                total_usd=money.usd if money else None,
            )
        )
    verified = cost.verified_total
    return AttractionFees(
        query=cost.query,
        name=cost.name,
        categories=[
            FeeCategory(
                label=CATEGORY_LABELS[(p.nationality, p.age_group)],
                text=category_text(p, lkr_per_usd),
                verified=p.verified,
            )
            for p in cost.categories
        ],
        group=lines,
        sources=source_lines(cost.records),
        conditions=entrance_conditions(cost.records),
        extras=[extra_text(r, lkr_per_usd) for r in cost.extras],
        verified_lkr=verified.lkr,
        verified_usd=verified.usd,
    )


def cost_line(cost: CostItem, travellers: int | None) -> CostLine:
    """A guide cost per person and for the group (None where it cannot be known)."""
    each = Money(cost.amount_lkr, cost.amount_usd)
    if cost.per == "person":
        per_person = each.times(cost.quantity)
        group = per_person.times(travellers) if travellers else None
    else:
        group = each.times(cost.quantity)
        per_person = (
            Money(
                None if group.lkr is None else group.lkr / travellers,
                None if group.usd is None else group.usd / travellers,
            )
            if travellers
            else None
        )
    return CostLine(
        item=cost.item,
        per=cost.per,
        quantity=cost.quantity,
        quantity_note=cost.quantity_note,
        each_lkr=each.lkr,
        each_usd=each.usd,
        group_lkr=group.lkr if group else None,
        group_usd=group.usd if group else None,
        per_person_lkr=per_person.lkr if per_person else None,
        per_person_usd=per_person.usd if per_person else None,
    )


def build_costs(
    itinerary: Itinerary,
    messages: Sequence[AnyMessage],
    trip_request: Mapping | None,
    *,
    today: date,
) -> dict[str, Any]:
    """The entrance_fees, cost_lines, stays and budget of a checked itinerary."""
    trip = dict(trip_request or {})
    group = group_breakdown(trip)
    travellers = trip.get("travellers") or None
    results = pricing_results(messages)
    costs = []
    for entry in results.attractions:
        if not _in_plan(entry, itinerary):
            continue
        records = [PriceRecord.from_json(r) for r in entry.get("records") or []]
        found = AttractionPrices(
            entry["query"], entry.get("attraction_id"), entry.get("name"), records
        )
        costs.append(
            attraction_cost(found, group, today=today, lkr_per_usd=results.lkr_per_usd)
        )
    fees = [_fees(c, results.lkr_per_usd) for c in costs]
    lines = [cost_line(c, travellers) for c in itinerary.costs]
    hotels = hotel_results(messages, itinerary.destination)
    stays = stay_suggestions(hotels, today) if hotels else None
    if not fees and not lines and stays is None:
        return {"entrance_fees": [], "cost_lines": [], "stays": None, "budget": None}

    not_included = list(NOT_INCLUDED)
    accommodation, stay_note = None, ""
    if hotels:
        accommodation, stay_note = accommodation_estimate(
            hotels, days=itinerary.days, travellers=travellers, today=today
        )
        if accommodation is None:
            not_included.insert(0, f"Accommodation ({stay_note})")
    elif trip.get("wants_hotels") is False:
        not_included.insert(0, "Accommodation (you said you have your own)")
    else:
        not_included.insert(0, "Accommodation (no hotel suggestions were made)")

    summary = summarise(costs, travellers or 0)
    other = ZERO
    for line in lines:
        other = other + Money(line.group_lkr, line.group_usd)
    total = summary.verified + summary.estimated + other
    if accommodation is not None:
        total = total + accommodation
    uniform = len(group) == 1 and travellers
    rate_note = " ".join(
        part for part in (results.rate_note, *results.problems) if part
    )
    budget = Budget(
        group=describe_group(trip),
        travellers=travellers,
        entrance_verified_lkr=summary.verified.lkr,
        entrance_verified_usd=summary.verified.usd,
        entrance_estimated_lkr=summary.estimated.lkr,
        entrance_estimated_usd=summary.estimated.usd,
        other_estimated_lkr=other.lkr,
        other_estimated_usd=other.usd,
        accommodation_lkr=accommodation.lkr if accommodation else None,
        accommodation_usd=accommodation.usd if accommodation else None,
        accommodation_note=stay_note if accommodation else "",
        total_lkr=total.lkr,
        total_usd=total.usd,
        per_person_lkr=total.lkr / travellers
        if uniform and total.lkr is not None
        else None,
        per_person_usd=total.usd / travellers
        if uniform and total.usd is not None
        else None,
        not_verified=summary.unverified,
        requires_confirmation=summary.not_found,
        not_included=not_included,
        rate_note=rate_note,
    )
    return {
        "entrance_fees": fees,
        "cost_lines": lines,
        "stays": stays,
        "budget": budget,
    }


# --- One layout for every screen -----


@dataclass
class Item:
    text: str
    details: list[str] = field(default_factory=list)


Section = tuple[str, list[Item]]


def _money(lkr: float | None, usd: float | None) -> str:
    return format_money(Money(lkr, usd))


def _is_zero(lkr: float | None, usd: float | None) -> bool:
    return (lkr or 0) == 0 and (usd or 0) == 0 and lkr is not None


def budget_sections(plan: Mapping[str, Any]) -> list[Section]:
    """The cost sections of a saved itinerary (a dict), as titled lines.

    Old itineraries without a budget get an empty list; their renderers keep
    the plain 'costs' table.
    """
    sections: list[Section] = []
    fees = plan.get("entrance_fees") or []
    if fees:
        items = []
        for fee in fees:
            if not fee.get("name"):
                items.append(
                    Item(
                        f"{fee['query']}: no official price found. Requires "
                        "confirmation before you go."
                    )
                )
                continue
            details = [f"{c['label']}: {c['text']}" for c in fee.get("categories", [])]
            details += [f"Your group: {g['text']}" for g in fee.get("group", [])]
            if fee.get("group"):
                subtotal = _money(fee.get("verified_lkr"), fee.get("verified_usd"))
                details.append(f"Verified subtotal for your group: {subtotal}")
            details += fee.get("extras") or []
            details += fee.get("sources") or []
            if fee.get("conditions"):
                details.append(f"Conditions: {fee['conditions']}")
            items.append(Item(f"{fee['name']}", details))
        sections.append(("Entrance fees (official sources)", items))

    lines = plan.get("cost_lines") or []
    if lines:
        items = []
        for line in lines:
            unit = "per person" if line["per"] == "person" else "shared"
            details = [
                f"Price: {_money(line.get('each_lkr'), line.get('each_usd'))} "
                f"({unit}) x {line['quantity']}"
            ]
            if line.get("quantity_note"):
                details.append(f"Quantity: {line['quantity_note']}")
            details.append(
                "Per person: "
                + _money(line.get("per_person_lkr"), line.get("per_person_usd"))
            )
            details.append(
                "For the group: " + _money(line.get("group_lkr"), line.get("group_usd"))
            )
            items.append(Item(line["item"], details))
        sections.append(
            (
                "Other costs (estimates from Wikivoyage, may be out of date)",
                items,
            )
        )

    stays = plan.get("stays")
    if stays:
        items = []
        for stay in stays.get("items") or []:
            year = stay.get("listed_year")
            if year is None:
                when = "date of listing unknown"
            elif stay.get("recent"):
                when = f"listed {year}"
            else:
                when = f"listed {year}, may have changed"
            details = [f"Price: {stay.get('price_text') or 'not listed'} ({when})"]
            if stay.get("description"):
                details.append(stay["description"])
            if stay.get("url"):
                details.append(f"Website: {stay['url']}")
            items.append(Item(f"{stay['name']} ({stay['level']})", details))
        if stays.get("unpriced"):
            items.append(
                Item(
                    "Price not listed (not checked against your budget)",
                    stays["unpriced"],
                )
            )
        if not stays.get("items"):
            items.append(Item("The guide lists no places that fit."))
        if stays.get("note"):
            items.append(Item(stays["note"]))
        items.append(
            Item(
                "From Wikivoyage: not booked and availability not checked; "
                f"contact the hotel. {stays.get('source_url', '')}".strip()
            )
        )
        wanted = f" ({stays['wanted']})" if stays.get("wanted") else ""
        sections.append((f"Where to stay{wanted}", items))

    budget = plan.get("budget")
    if budget:
        items = []
        if budget.get("group"):
            items.append(Item(f"Group: {budget['group']}"))
        items.append(
            Item(
                "Verified entrance fees: "
                + _money(
                    budget.get("entrance_verified_lkr"),
                    budget.get("entrance_verified_usd"),
                )
            )
        )
        estimated = (
            budget.get("entrance_estimated_lkr"),
            budget.get("entrance_estimated_usd"),
        )
        if not _is_zero(*estimated):
            items.append(
                Item(
                    f"Estimated entrance fees (children, adult price as a cap): "
                    f"{_money(*estimated)}"
                )
            )
        other = (budget.get("other_estimated_lkr"), budget.get("other_estimated_usd"))
        if lines:
            items.append(Item(f"Other estimated costs: {_money(*other)}"))
        stay_money = (budget.get("accommodation_lkr"), budget.get("accommodation_usd"))
        if stay_money != (None, None):
            items.append(
                Item(
                    f"Accommodation (upper estimate): {_money(*stay_money)}",
                    [budget.get("accommodation_note") or ""],
                )
            )
        items.append(
            Item(
                "TOTAL estimated budget for the group: "
                + _money(budget.get("total_lkr"), budget.get("total_usd"))
            )
        )
        if (
            budget.get("per_person_lkr") is not None
            or budget.get("per_person_usd") is not None
        ):
            items.append(
                Item(
                    "Per person: "
                    + _money(budget.get("per_person_lkr"), budget.get("per_person_usd"))
                )
            )
        elif budget.get("travellers") and (budget.get("travellers") or 0) > 1:
            items.append(
                Item(
                    "Per person: differs by visitor category; see the entrance "
                    "fee lines above."
                )
            )
        if budget.get("not_verified"):
            items.append(
                Item("Not verified (could change the budget)", budget["not_verified"])
            )
        if budget.get("requires_confirmation"):
            items.append(Item("Requires confirmation", budget["requires_confirmation"]))
        if budget.get("not_included"):
            items.append(
                Item("Not included (no reliable source here)", budget["not_included"])
            )
        if budget.get("rate_note"):
            items.append(Item(budget["rate_note"]))
        sections.append(("Trip budget", items))
    return sections


def budget_markdown(plan: Mapping[str, Any]) -> list[str]:
    """The cost sections as Markdown lines (empty if there is no budget)."""
    lines: list[str] = []
    for title, items in budget_sections(plan):
        lines += ["", f"## {title}"]
        for item in items:
            lines.append(f"- **{item.text}**" if item.details else f"- {item.text}")
            lines.extend(f"  - {detail}" for detail in item.details)
    return lines
