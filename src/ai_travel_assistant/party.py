"""Who is travelling: the group's ticket categories, decided by code, not the LLM.

Many Sri Lankan sites charge Sri Lankan citizens, citizens of other SAARC
countries and other foreigners different prices, and children differently
from adults. A plan for a mixed group needs the count in each category.

What this file demonstrates:
  1. The LLM only EXTRACTS what the traveller said ("2 Sri Lankans, 1 Indian,
     1 German child"): a nationality word, adult or child, and a count.
     The pricing category (local, saarc, non_saarc) is then decided here by
     a fixed list, so "Indian" can never be priced as a Sri Lankan.
  2. Citizenship, not residence. A German living in Colombo is non_saarc.
     Anything the code cannot place for sure (a dual citizen, "Asian", no
     nationality) is "unknown", and the agent asks instead of assuming the
     cheaper rate.
  3. One pure function, group_breakdown(), turns a trip into counts per
     (category, age group). The prompt note and the cost calculation both
     use it, so they always agree.

The trip is the plain dict request_understanding keeps in the graph state.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping
from typing import Literal

NationalityGroup = Literal["local", "saarc", "non_saarc", "unknown"]
AgeGroup = Literal["adult", "child"]

# What the citizenship quick replies and the understander may store for a
# group that is all one kind ("mixed" means: ask for the party list).
Citizenship = Literal["local", "saarc", "non_saarc", "mixed"]

# SAARC members other than Sri Lanka (South Asian Association for Regional
# Cooperation), as names and demonyms in lower case.
_SAARC = {
    "afghanistan", "afghan", "afghani",
    "bangladesh", "bangladeshi",
    "bhutan", "bhutanese",
    "india", "indian",
    "maldives", "maldivian",
    "nepal", "nepali", "nepalese",
    "pakistan", "pakistani",
}  # fmt: skip
_LOCAL = {"sri lanka", "sri lankan", "srilanka", "srilankan", "lankan", "local"}
# Words the quick replies and the understander use for a whole category.
_CATEGORY_WORDS = {
    "saarc": "saarc",
    "saarc country": "saarc",
    "non-saarc": "non_saarc",
    "non saarc": "non_saarc",
}
# A nationality that does not settle the price on its own: "foreign" or
# "tourist" could still be a SAARC citizen, so the agent asks.
_UNSURE = re.compile(
    r"\b(dual|unknown|not sure|mixed|asian|foreign(er)?s?|tourists?|"
    r"international|abroad|visitors?)\b"
)

LABELS = {
    "local": "Sri Lankan",
    "saarc": "SAARC",
    "non_saarc": "non-SAARC foreign",
    "unknown": "citizenship not clear",
}


def nationality_group(nationality: str | None) -> NationalityGroup:
    """The pricing category for a nationality the traveller named.

    Any named country outside SAARC is non_saarc. Dual citizens and vague
    answers are unknown: each site has its own rules for them.
    """
    text = re.sub(r"\s+", " ", (nationality or "").strip().lower())
    text = re.sub(r"\b(citizens?|nationals?|people|passport)\b", "", text).strip()
    if not text or _UNSURE.search(text):
        return "unknown"
    if text in _LOCAL:
        return "local"
    if text in _SAARC:
        return "saarc"
    if text in _CATEGORY_WORDS:
        return _CATEGORY_WORDS[text]  # type: ignore[return-value]
    return "non_saarc"


def _members(trip: Mapping) -> list[Mapping]:
    return [m for m in trip.get("party") or [] if (m or {}).get("count")]


GroupRow = tuple[NationalityGroup, AgeGroup, int]


def group_breakdown(trip: Mapping | None) -> list[GroupRow]:
    """Counts per (category, age group), largest first; [] if not known yet.

    An explicit party list wins. Otherwise a single citizenship for the whole
    group is split into adults and children with the counts given.
    """
    trip = trip or {}
    counts: Counter[tuple[NationalityGroup, AgeGroup]] = Counter()
    members = _members(trip)
    if members:
        for member in members:
            age = "child" if member.get("age_group") == "child" else "adult"
            counts[(nationality_group(member.get("nationality")), age)] += int(
                member["count"]
            )
    else:
        group, total = trip.get("citizenship"), trip.get("travellers")
        if group not in ("local", "saarc", "non_saarc") or not total:
            return []
        children = min(int(trip.get("children") or 0), int(total))
        if total - children:
            counts[(group, "adult")] = int(total) - children
        if children:
            counts[(group, "child")] = children
    order = list(LABELS)
    return sorted(
        ((group, age, n) for (group, age), n in counts.items()),
        key=lambda row: (order.index(row[0]), row[1] != "adult"),
    )


def complete_party(trip: Mapping) -> dict:
    """Make travellers and children agree with an explicit party list."""
    trip = dict(trip)
    members = _members(trip)
    if members:
        trip["travellers"] = sum(int(m["count"]) for m in members)
        trip["children"] = sum(
            int(m["count"]) for m in members if m.get("age_group") == "child"
        )
    elif trip.get("travellers") == 1 and trip.get("children") is None:
        trip["children"] = 0  # "just me": the person planning is an adult
    return trip


def missing_group_fields(trip: Mapping | None) -> list[str]:
    """Which group questions are still open, in the order they are asked.

    citizenship: nobody said who is travelling.
    party: a mixed group (or someone's citizenship is unclear) must be listed.
    children: a group of two or more with no word on children.
    """
    trip = trip or {}
    members = _members(trip)
    if members:
        unclear = any(g == "unknown" for g, _, _ in group_breakdown(trip))
        return ["party"] if unclear else []
    citizenship = trip.get("citizenship")
    if citizenship is None:
        return ["citizenship"]
    if citizenship == "mixed":
        return ["party"]
    travellers = trip.get("travellers")
    if travellers and travellers > 1 and trip.get("children") is None:
        return ["children"]
    return []


def describe_group(trip: Mapping | None) -> str:
    """'2 Sri Lankan adults, 1 SAARC adult' ('' when not known)."""
    parts = []
    for group, age, count in group_breakdown(trip):
        noun = age if count == 1 else ("children" if age == "child" else "adults")
        parts.append(f"{count} {LABELS[group]} {noun}")
    return ", ".join(parts)
