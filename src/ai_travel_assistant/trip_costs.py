"""Entrance fees for a group: per visitor category, per person and in total.

Pure arithmetic over pricing_service results: no HTTP and no LLM, so every
sum the traveller sees is computed by code and can be tested.

What this file demonstrates:
  1. One line per visitor category in the group (party.group_breakdown):
     "1 non-SAARC adult x USD 35 = USD 35". A category whose price is not
     verified is listed as such and adds nothing to the verified total.
  2. Verified and estimated money never mix. The verified total holds only
     verified category prices. The one estimate allowed is an upper bound for
     children when only the adult price of the same category is verified
     (child rules are not published), and it is labelled as an estimate.
  3. Both currencies. A source's own LKR figure (CCF converts at its own
     rate) is used as given; otherwise the other currency comes from one
     exchange rate passed in by the caller. With no rate, that side is None,
     never a guess.
  4. Activities and vehicle or parking charges are listed separately and not
     added to the entrance total: they are optional or depend on the vehicle.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta

from ai_travel_assistant.party import GroupRow
from ai_travel_assistant.pricing_service import (
    AttractionPrices,
    CategoryPrice,
    PriceRecord,
    price_for,
)

# Every category shown for an attraction, even when nobody in the group has it.
DISPLAY_CATEGORIES = (
    ("local", "adult"),
    ("local", "child"),
    ("saarc", "adult"),
    ("saarc", "child"),
    ("non_saarc", "adult"),
    ("non_saarc", "child"),
)
CATEGORY_LABELS = {
    ("local", "adult"): "Sri Lankan adult",
    ("local", "child"): "Sri Lankan child",
    ("saarc", "adult"): "SAARC adult",
    ("saarc", "child"): "SAARC child",
    ("non_saarc", "adult"): "Non-SAARC adult",
    ("non_saarc", "child"): "Non-SAARC child",
    ("unknown", "adult"): "Adult (citizenship not clear)",
    ("unknown", "child"): "Child (citizenship not clear)",
}


@dataclass(frozen=True)
class Money:
    """An amount in both currencies; None where it could not be worked out."""

    lkr: float | None
    usd: float | None

    def __add__(self, other: Money) -> Money:
        return Money(_add(self.lkr, other.lkr), _add(self.usd, other.usd))

    def times(self, n: int) -> Money:
        return Money(
            None if self.lkr is None else self.lkr * n,
            None if self.usd is None else self.usd * n,
        )


ZERO = Money(0.0, 0.0)


def _add(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else a + b


def record_money(record: PriceRecord, lkr_per_usd: float | None) -> Money:
    """One unit of a price in LKR and USD.

    A source's own LKR equivalent wins over today's rate, so the traveller
    sees the figure the ticket office uses.
    """
    if record.currency == "USD":
        lkr = record.lkr_equivalent
        if lkr is None and lkr_per_usd:
            lkr = record.amount * lkr_per_usd
        return Money(lkr, record.amount)
    usd = record.amount / lkr_per_usd if lkr_per_usd else None
    return Money(record.amount, usd)


@dataclass(frozen=True)
class GroupLine:
    """What the travellers of one category pay at one attraction."""

    nationality: str
    age_group: str
    count: int
    price: CategoryPrice
    each: Money | None = None  # verified price per person
    subtotal: Money | None = None  # verified: each x count
    estimate: Money | None = None  # labelled estimate (children, adult price)
    estimate_note: str = ""

    @property
    def label(self) -> str:
        return CATEGORY_LABELS[(self.nationality, self.age_group)]

    @property
    def people(self) -> str:
        """'1 SAARC adult', '2 Sri Lankan adults', '3 Non-SAARC children'."""
        if self.count == 1:
            return f"1 {self.label}"
        plural = self.label.replace("child", "children").replace("adult", "adults")
        return f"{self.count} {plural}"


@dataclass
class AttractionCost:
    query: str
    name: str | None
    records: list[PriceRecord]
    categories: list[CategoryPrice]
    group: list[GroupLine]
    extras: list[PriceRecord]  # activities, vehicles, parking: listed only

    @property
    def found(self) -> bool:
        return self.name is not None

    @property
    def verified_total(self) -> Money:
        total = ZERO
        for line in self.group:
            if line.subtotal is not None:
                total = total + line.subtotal
        return total

    @property
    def estimated_total(self) -> Money:
        total = ZERO
        for line in self.group:
            if line.estimate is not None:
                total = total + line.estimate
        return total

    @property
    def unverified(self) -> list[GroupLine]:
        return [line for line in self.group if line.subtotal is None]


def attraction_cost(
    found: AttractionPrices,
    group: Sequence[GroupRow],
    *,
    today: date,
    lkr_per_usd: float | None,
    max_age: timedelta | None = None,
) -> AttractionCost:
    """Prices for every category, and the group's lines, at one attraction."""
    records = found.records

    def price(nationality: str, age: str) -> CategoryPrice:
        return price_for(records, nationality, age, today=today, max_age=max_age)

    categories = [price(n, a) for n, a in DISPLAY_CATEGORIES]
    lines = []
    for nationality, age, count in group:
        if nationality == "unknown":
            missing = CategoryPrice(
                nationality,
                age,
                None,
                None,
                "Citizenship not clear: ask the traveller.",
            )
            lines.append(GroupLine(nationality, age, count, missing))
            continue
        chosen = price(nationality, age)
        if chosen.record is not None and chosen.record.unit == "person":
            each = record_money(chosen.record, lkr_per_usd)
            lines.append(
                GroupLine(nationality, age, count, chosen, each, each.times(count))
            )
            continue
        estimate, note = None, ""
        if age == "child":
            adult = price(nationality, "adult")
            if adult.record is not None and adult.record.unit == "person":
                estimate = record_money(adult.record, lkr_per_usd).times(count)
                note = (
                    "child rate not verified; the adult price is used as an "
                    "upper estimate"
                )
        lines.append(
            GroupLine(nationality, age, count, chosen, None, None, estimate, note)
        )
    extras = [r for r in records if r.ticket_type != "entrance"]
    return AttractionCost(
        found.query, found.attraction_name, records, categories, lines, extras
    )


@dataclass
class EntranceSummary:
    """The group's entrance fees over every attraction looked up."""

    attractions: list[AttractionCost]
    travellers: int
    verified: Money = field(default=ZERO)
    estimated: Money = field(default=ZERO)
    unverified: list[str] = field(default_factory=list)
    not_found: list[str] = field(default_factory=list)
    # No per-person average: in a mixed group nobody pays it. Each GroupLine
    # carries the real per-person price of its category.


def summarise(costs: Iterable[AttractionCost], travellers: int) -> EntranceSummary:
    costs = list(costs)
    summary = EntranceSummary(costs, travellers)
    for cost in costs:
        if not cost.found:
            summary.not_found.append(cost.query)
            continue
        summary.verified = summary.verified + cost.verified_total
        summary.estimated = summary.estimated + cost.estimated_total
        for line in cost.unverified:
            summary.unverified.append(f"{cost.name}: {line.people}")
    return summary
