"""Places to stay from Wikivoyage's structured listings, by budget level.

Wikivoyage writes each hotel in its "Sleep" section as a template:
    {{sleep | name=Kandy City House | url= | price=From Rs 2000 |
             lastedit=2018-05-14 | content=Family run little guest house. }}
under sub-headings such as ===Budget===, ===Mid-range=== and ===Splurge===.
This service reads those templates from the article's wikitext, so hotel
names, prices and dates come from the guide, never from the LLM.

What this file demonstrates:
  1. Parsing a template language by hand: nested {{...}} and [[...|...]]
     must not split a listing's fields, so a small bracket-depth scanner is
     used instead of one regex.
  2. Honest dates. Each listing keeps the year it was last edited (or the year
     written in its price). Prices older than RECENT_YEARS are shown with
     their year but never added to a budget.
  3. Budget levels and amounts. A level (budget, mid-range, luxury) picks the
     matching sub-section; a typed nightly amount keeps listings whose price
     can be read and fits, and lists the rest as "price not listed".
  4. No booking. Wikivoyage has no availability or booking, and neither does
     this app: the traveller is told to check with the hotel.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any, Literal

import httpx

from ai_travel_assistant.guide_service import (
    GuideServiceError,
    _get_json,
    make_client,
)
from ai_travel_assistant.resilience import TTLCache

Tier = Literal["budget", "mid-range", "splurge", "other"]
Basis = Literal["room", "person"]

# Wikivoyage's sub-heading -> our level. Anything else ("Outside the city",
# "Camping") is kept as "other" and shown only when no level was asked for.
_TIERS = {
    "budget": "budget",
    "cheap": "budget",
    "backpackers": "budget",
    "mid-range": "mid-range",
    "midrange": "mid-range",
    "mid range": "mid-range",
    "splurge": "splurge",
    "luxury": "splurge",
    "upscale": "splurge",
    "high-end": "splurge",
}
# The trip's budget field (schemas.TripRequest.budget) -> a Wikivoyage level.
BUDGET_TO_TIER: dict[str, Tier] = {
    "budget": "budget",
    "mid-range": "mid-range",
    "luxury": "splurge",
}
TIER_LABELS = {
    "budget": "Budget",
    "mid-range": "Mid-range",
    "splurge": "Luxury",
    "other": "Other",
}

# A price older than this many years is shown, never added to a budget.
RECENT_YEARS = 3
MAX_LISTINGS = 6

WIKITEXT_CACHE = TTLCache("wikivoyage wikitext", ttl_s=24 * 3600)


@dataclass(frozen=True)
class Listing:
    """One place to stay, as Wikivoyage lists it."""

    name: str
    tier: Tier
    price_text: str = ""
    url: str = ""
    address: str = ""
    description: str = ""
    listed_year: int | None = None  # lastedit year, or a year in the price
    currency: str | None = None  # "LKR" or "USD" if the price can be read
    low: float | None = None
    high: float | None = None
    basis: Basis = "room"

    def is_recent(self, today: date) -> bool:
        return (
            self.listed_year is not None
            and today.year - self.listed_year <= RECENT_YEARS
        )

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Listing:
        return cls(**data)


# --- Wikitext -----------------------------------------------------------------


def _templates(text: str, names: Iterable[str]) -> list[tuple[int, str]]:
    """(position, body) of every {{name ...}} template, nesting respected."""
    wanted = tuple(n.lower() for n in names)
    found = []
    i = 0
    while True:
        start = text.find("{{", i)
        if start < 0:
            return found
        head = text[start + 2 : start + 40].lstrip().lower()
        depth, j = 0, start
        while j < len(text) - 1:
            pair = text[j : j + 2]
            if pair == "{{":
                depth += 1
                j += 2
                continue
            if pair == "}}":
                depth -= 1
                j += 2
                if depth == 0:
                    break
                continue
            j += 1
        if head.startswith(wanted):
            found.append((start, text[start + 2 : j - 2]))
        i = j if j > start else start + 2


def _fields(body: str) -> dict[str, str]:
    """name=value pairs of a template body; | inside {{ }} or [[ ]] is kept."""
    parts, depth, current = [], 0, []
    k = 0
    while k < len(body):
        two = body[k : k + 2]
        if two in ("{{", "[["):
            depth += 1
            current.append(two)
            k += 2
            continue
        if two in ("}}", "]]"):
            depth -= 1
            current.append(two)
            k += 2
            continue
        if body[k] == "|" and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(body[k])
        k += 1
    parts.append("".join(current))
    fields = {}
    for part in parts[1:]:
        if "=" in part:
            key, value = part.split("=", 1)
            fields[key.strip().lower()] = _plain(value)
    return fields


def _plain(value: str) -> str:
    """Wiki markup to plain text: [[a|b]] -> b, {{...}} and '' removed."""
    value = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", value)
    value = re.sub(r"\{\{[^{}]*\}\}", "", value)
    value = re.sub(r"<[^>]+>", "", value).replace("''", "")
    return re.sub(r"\s+", " ", value).strip()


def _sleep_section(wikitext: str) -> str:
    match = re.search(r"^==\s*Sleep\s*==\s*$", wikitext, re.M | re.I)
    if not match:
        return ""
    rest = wikitext[match.end() :]
    end = re.search(r"^==[^=].*?==\s*$", rest, re.M)
    return rest[: end.start()] if end else rest


def _tier_at(section: str, position: int) -> Tier:
    tier: Tier = "other"
    for heading in re.finditer(r"^===\s*([^=]+?)\s*===\s*$", section, re.M):
        if heading.start() > position:
            break
        tier = _TIERS.get(heading.group(1).strip().lower(), "other")  # type: ignore[assignment]
    return tier


# --- Prices -------------------------------------------------------------------

_NUM = r"(\d[\d,]*(?:\.\d+)?)"
_RANGE = rf"{_NUM}(?:\s*(?:-|–|/|to)\s*(?:Rs\.?|LKR|US\$|USD|\$)?\s*{_NUM})?"
_LKR = re.compile(
    rf"(?:\bRs\.?|\bLKR)\s*{_RANGE}|{_RANGE}\s*(?:Rs\b|LKR\b|rupees)", re.I
)
_USD = re.compile(
    rf"(?:US\$|\bUSD|\$)\s*{_RANGE}|{_RANGE}\s*(?:USD\b|US dollars)", re.I
)
_YEAR = re.compile(r"\b(20\d\d)\b")


def _amount(text: str) -> float:
    return float(text.replace(",", ""))


def parse_price(text: str) -> tuple[str | None, float | None, float | None, Basis]:
    """('LKR', 3000, 3500, 'room') from 'Rs 3000/3500 (2014)'; Nones if unreadable."""
    basis: Basis = (
        "person"
        if re.search(r"\b(per person|person|pp|per head)\b", text, re.I)
        else "room"
    )
    for currency, pattern in (("LKR", _LKR), ("USD", _USD)):
        match = pattern.search(text)
        if not match:
            continue
        numbers = [_amount(n) for n in match.groups() if n]
        numbers = [n for n in numbers if not 1990 <= n <= 2100 or currency == "LKR"]
        if numbers:
            return currency, min(numbers), max(numbers), basis
    return None, None, None, basis


def parse_listings(wikitext: str) -> list[Listing]:
    """Every {{sleep}} (or {{listing | type=sleep}}) in the Sleep section."""
    section = _sleep_section(wikitext)
    listings = []
    for position, body in _templates(section, ("sleep", "listing")):
        fields = _fields(body)
        if (
            body.lstrip().lower().startswith("listing")
            and fields.get("type") != "sleep"
        ):
            continue
        name = fields.get("name", "")
        if not name:
            continue
        price = fields.get("price", "")
        year = None
        if fields.get("lastedit"):
            found = _YEAR.search(fields["lastedit"])
            year = int(found.group(1)) if found else None
        if year is None and _YEAR.search(price):
            year = int(_YEAR.search(price).group(1))  # type: ignore[union-attr]
        currency, low, high, basis = parse_price(price)
        listings.append(
            Listing(
                name=name,
                tier=_tier_at(section, position),
                price_text=price,
                url=fields.get("url", ""),
                address=fields.get("address", ""),
                description=fields.get("content", "")[:200],
                listed_year=year,
                currency=currency,
                low=low,
                high=high,
                basis=basis,
            )
        )
    return listings


# --- Fetching -----------------------------------------------------------------


@dataclass(frozen=True)
class Stays:
    destination: str
    title: str
    article_url: str
    listings: list[Listing]


def fetch_stays(destination: str, *, client: httpx.Client | None = None) -> Stays:
    """The article's place-to-stay listings (1 request, cached a day).

    Raises guide_service.GuideServiceError subclasses, as the guide does.
    """
    name = destination.strip()
    key = name.lower()
    cached = WIKITEXT_CACHE.get(key)
    if cached is not TTLCache.MISSING:
        return cached
    params = {
        "action": "parse",
        "page": name,
        "prop": "wikitext",
        "format": "json",
        "formatversion": 2,
        "redirects": 1,
    }
    if client is not None:
        data = _get_json(client, params)
    else:
        with make_client() as own:
            data = _get_json(own, params)
    parse = data.get("parse") or {}
    title = parse.get("title") or name
    stays = Stays(
        destination=name,
        title=title,
        article_url="https://en.wikivoyage.org/wiki/" + title.replace(" ", "_"),
        listings=parse_listings(parse.get("wikitext") or ""),
    )
    WIKITEXT_CACHE.set(key, stays)
    return stays


# --- Choosing ------------------------------------------------------------------


@dataclass
class Choice:
    """The listings that fit, and those shown without a readable price."""

    fits: list[Listing]
    unpriced: list[Listing]
    note: str = ""


def _nightly_lkr(listing: Listing, lkr_per_usd: float | None) -> float | None:
    """The low end of a listing's price per room per night, in LKR."""
    if listing.low is None or listing.basis != "room":
        return None
    if listing.currency == "LKR":
        return listing.low
    return listing.low * lkr_per_usd if lkr_per_usd else None


def choose(
    listings: list[Listing],
    *,
    tier: Tier | None,
    max_lkr: float | None,
    lkr_per_usd: float | None,
) -> Choice:
    """Listings for a level and/or a nightly amount; most recent first."""
    pool = [x for x in listings if tier is None or x.tier == tier]
    note = ""
    if tier is not None and not pool:
        note = (
            f"The guide lists no {TIER_LABELS[tier].lower()} places here; "
            "showing the others it lists."
        )
        pool = list(listings)
    if max_lkr is None:
        fits, unpriced = pool, []
    else:
        fits, unpriced = [], []
        for listing in pool:
            nightly = _nightly_lkr(listing, lkr_per_usd)
            if nightly is None:
                unpriced.append(listing)
            elif nightly <= max_lkr:
                fits.append(listing)
    newest = sorted(fits, key=lambda x: -(x.listed_year or 0))
    return Choice(newest[:MAX_LISTINGS], unpriced[:3], note)


__all__ = [
    "BUDGET_TO_TIER",
    "Choice",
    "GuideServiceError",
    "Listing",
    "Stays",
    "TIER_LABELS",
    "choose",
    "fetch_stays",
    "parse_listings",
    "parse_price",
]
