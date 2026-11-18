"""Stage 8: checking a submitted itinerary against the tool results it came from.

What this file demonstrates:
  1. Grounding is SOURCE-SPECIFIC and DESTINATION-AWARE. Every tool result is
     paired with the call that produced it (tool name and arguments), so a
     check knows WHICH place and WHICH tool a value came from. A number or a
     name that appears in some unrelated earlier result does not count:
       - an OpenStreetMap sight must be in find_attractions for THIS destination,
         and its distance must be the one listed for THAT sight;
       - a travel leg must match the estimate_travel_time call for THOSE two
         places;
       - a price must be quoted, in that currency, in THIS destination's guide,
         and its conversion must be a convert_currency result for THAT amount;
       - the weather of a day comes from the forecast line for THAT date, for
         THIS destination, and the date must not be in the past.
     On a "modify" turn the earlier results are still in the thread; they are
     reused only if they belong to the current destination.
  2. Facts versus suggestions. Factual details (transport, opening hours,
     safety, prices) must come as PracticalFacts with a verbatim quote that is
     found in that source. Free-text suggestions and descriptions lose any
     sentence with an unconfirmed price, clock time or schedule.
  3. Never filling a gap. A value that cannot be confirmed is removed, set to
     null, or replaced by the tool's own value; what is missing goes to
     not_available. Place names that cannot be found are kept but flagged,
     because a name may be spelled a little differently.

Pure functions: the messages are read, never changed. No LLM calls.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date

from langchain_core.messages import AIMessage, AnyMessage, ToolMessage

from ai_travel_assistant.schemas import (
    Activity,
    CostItem,
    DayPlan,
    Itinerary,
    ItineraryDraft,
    PracticalFact,
    TravelLeg,
)

SUBMIT_TOOL = "submit_itinerary"

# How a tool's text starts when it could not provide data (see tools.py and
# robustness.py). Such results are evidence of a FAILURE, never of a value.
UNAVAILABLE_PREFIXES = (
    "Weather unavailable",
    "Attractions unavailable",
    "Guide unavailable",
    "Travel time unavailable",
    "Currency conversion unavailable",
    "Tool failed",
    "Not run",
)
FALLBACK_MARKER = "FALLBACK: sights named in the Wikivoyage guide"

# What to list in not_available when a tool failed for the destination.
FAILED_TOOL_NOTES = {
    "get_weather_forecast": "Weather forecast (the weather service was unavailable)",
    "find_attractions": "Nearby sights from OpenStreetMap (the service failed)",
    "get_destination_guide": "Travel guide details (the guide was unavailable)",
    "estimate_travel_time": "Some road distances and driving times (routing failed)",
    "convert_currency": "Some currency conversions (the exchange service failed)",
}


# --- Evidence: each tool result with the call that produced it ------------------------


@dataclass(frozen=True)
class Evidence:
    tool: str
    args: dict
    text: str
    ok: bool  # the tool returned data, not an "unavailable" message
    fallback: bool = False  # find_attractions answered from the Wikivoyage guide


def _as_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block if isinstance(block, str) else str(block.get("text", ""))
            for block in content
            if isinstance(block, (str, dict))
        )
    return str(content)


def collect_evidence(messages: Sequence[AnyMessage]) -> list[Evidence]:
    """Pair every tool result in the thread with its call (oldest first).

    A result belongs to the calls of the AIMessage just before it, so a call
    id reused in a later turn can never pair a result with the wrong call
    (and so with the wrong destination).
    """
    calls: dict[str, dict] = {}
    evidence = []
    for message in messages:
        if isinstance(message, AIMessage):
            if message.tool_calls:
                calls = {call["id"]: call for call in message.tool_calls}
            continue
        if not isinstance(message, ToolMessage):
            continue
        call = calls.get(message.tool_call_id)
        if call is None or call["name"] == SUBMIT_TOOL:
            continue
        text = _as_text(message.content)
        fallback = call["name"] == "find_attractions" and FALLBACK_MARKER in text
        ok = (
            message.status != "error"
            and not text.startswith(UNAVAILABLE_PREFIXES)
            and not fallback
        )
        evidence.append(Evidence(call["name"], dict(call["args"]), text, ok, fallback))
    return evidence


# --- Text helpers ---------------------------------------------------------------------


def normalize(text: str) -> str:
    """Lowercase, no accents, punctuation as spaces ("Ravana's" -> "ravana s")."""
    decomposed = unicodedata.normalize("NFKD", text)
    plain = "".join(c for c in decomposed if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^0-9a-z]+", " ", plain.casefold()).split())


def contains_phrase(haystack: str, phrase: str) -> bool:
    """True if the normalized phrase appears as whole words in the haystack."""
    needle = normalize(phrase)
    return bool(needle) and f" {needle} " in f" {normalize(haystack)} "


_COUNTRY_SUFFIX = re.compile(r"\s+(?:sri lanka|lk)$")


def same_place(a: object, b: object) -> bool:
    """'Ella' matches 'Ella', 'ella, Sri Lanka' and 'Ella town'; not 'Kandy'."""
    first = _COUNTRY_SUFFIX.sub("", normalize(str(a or "")))
    second = _COUNTRY_SUFFIX.sub("", normalize(str(b or "")))
    if not first or not second:
        return False
    return (
        first == second
        or first.startswith(second + " ")
        or second.startswith(first + " ")
    )


def close(a: float, b: float, *, rel: float, absolute: float) -> bool:
    return abs(a - b) <= max(absolute, rel * max(abs(a), abs(b)))


def _number(text: str) -> float:
    return float(text.replace(",", ""))


# Details that free text must not state unless a quoted fact backs them.
_UNVERIFIED_DETAIL = re.compile(
    r"(?:\b(?:rs\.?|lkr|usd|us\$|rupees?)\s*\d"
    r"|\$\s*\d"
    r"|\d[\d,]*(?:\.\d+)?\s*(?:lkr|usd|rupees?|rs)\b"
    r"|\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b"
    r"|\b\d{1,2}:\d{2}\b"
    r"|\b(?:open|opens|closed|closes)\b[^.]*\b(?:daily|monday|tuesday|wednesday|"
    r"thursday|friday|saturday|sunday|weekends?|weekdays?|\d)"
    r"|\b(?:departs?|departures?|leaves at|runs every"
    r"|every \d+ (?:min|minutes|hours?)))",
    re.IGNORECASE,
)


def has_unverified_detail(text: str) -> bool:
    return bool(_UNVERIFIED_DETAIL.search(text))


def scrub_unverified(text: str) -> tuple[str, int]:
    """Drop sentences with prices, clock times or schedules: (text, removed)."""
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]
    kept = [s for s in sentences if not has_unverified_detail(s)]
    return " ".join(kept), len(sentences) - len(kept)


# --- What the tools said about one destination ----------------------------------------

_OSM_LINE = re.compile(
    r"^- (?P<name>.+) \((?P<kind>[^()]*)\), (?P<km>\d+(?:\.\d+)?) km$", re.M
)
_WEATHER_LINE = re.compile(
    r"^- (?P<date>\d{4}-\d{2}-\d{2})(?: \(\w+\))?: (?P<rest>.+)$", re.M
)
_ROAD_KM = re.compile(r"Road distance: (?P<km>[\d,]+(?:\.\d+)?) km")
_DRIVE_TIME = re.compile(r"Driving time: about (?P<time>.+?) by car")
_CONVERSION = re.compile(
    r"^(?P<amount>[\d,]+(?:\.\d+)?) (?P<src>[A-Z]{3}) = "
    r"(?P<out>[\d,]+(?:\.\d+)?) (?P<dst>[A-Z]{3})$",
    re.M,
)
_NUMBER = r"(\d[\d,]*(?:\.\d+)?)"
_PRICE_PATTERNS = {
    "LKR": (
        re.compile(rf"(?:\bLKR|\bRs\.?|\brupees?)\s*{_NUMBER}", re.I),
        re.compile(rf"{_NUMBER}\s*(?:LKR|rupees?|Rs)\b", re.I),
    ),
    "USD": (
        re.compile(rf"(?:\bUSD|US\$|\$)\s*{_NUMBER}", re.I),
        re.compile(rf"{_NUMBER}\s*(?:USD|US dollars?)\b", re.I),
    ),
}


def parse_minutes(text: str) -> float | None:
    """'2 h 40 min' -> 160; '45 min' -> 45; '3 h' -> 180."""
    hours = re.search(r"(\d+)\s*h\b", text)
    minutes = re.search(r"(\d+)\s*min\b", text)
    if not hours and not minutes:
        return None
    return (int(hours.group(1)) * 60 if hours else 0) + (
        int(minutes.group(1)) if minutes else 0
    )


@dataclass
class Route:
    origin: str
    destination: str
    km: float | None
    minutes: float | None
    text: str


@dataclass
class DestinationSources:
    """Everything the tools returned that is relevant to one destination."""

    destination: str
    osm_sights: dict[str, tuple[str, float]] = field(default_factory=dict)
    osm_text: str = ""
    wikivoyage_text: str = ""
    builtin_text: str = ""
    forecast: dict[str, str] = field(default_factory=dict)
    weather_text: str = ""
    routes: list[Route] = field(default_factory=list)
    conversions: list[tuple[float, str, float, str]] = field(default_factory=list)
    guide_prices: dict[str, list[float]] = field(default_factory=dict)
    called: set[str] = field(default_factory=set)
    failed: set[str] = field(default_factory=set)

    def route_between(self, a: str, b: str) -> Route | None:
        """The latest OSRM result for these two places (either direction)."""
        for route in reversed(self.routes):
            if (same_place(route.origin, a) and same_place(route.destination, b)) or (
                same_place(route.origin, b) and same_place(route.destination, a)
            ):
                return route
        return None


def _about_destination(item: Evidence, destination: str) -> bool:
    if item.tool == "estimate_travel_time":
        return same_place(item.args.get("origin"), destination) or same_place(
            item.args.get("destination"), destination
        )
    return same_place(item.args.get("destination"), destination)


def sources_for(destination: str, evidence: Iterable[Evidence]) -> DestinationSources:
    """Sort the evidence by source, keeping only what concerns this destination.

    Exceptions, because they are not tied to one destination: OSRM routes are
    kept for any pair of places (each leg is matched by its own two ends), and
    currency conversions are kept because they are matched by exact amount.
    """
    found = DestinationSources(destination)
    wikivoyage, osm, weather = [], [], []
    for item in evidence:
        if item.tool == "search_destinations":
            if item.ok:
                found.builtin_text += "\n" + item.text
            continue
        if item.tool == "convert_currency":
            # Not tied to a destination, so it never counts as "called for
            # this place"; a failure is still worth listing.
            if item.ok:
                for m in _CONVERSION.finditer(item.text):
                    found.conversions.append(
                        (_number(m["amount"]), m["src"], _number(m["out"]), m["dst"])
                    )
            else:
                found.failed.add(item.tool)
            continue
        if item.tool == "estimate_travel_time":
            if item.ok:
                km, time = _ROAD_KM.search(item.text), _DRIVE_TIME.search(item.text)
                found.routes.append(
                    Route(
                        str(item.args.get("origin", "")),
                        str(item.args.get("destination", "")),
                        _number(km["km"]) if km else None,
                        parse_minutes(time["time"]) if time else None,
                        item.text,
                    )
                )
            if _about_destination(item, destination):
                found.called.add(item.tool)
                if not item.ok:
                    found.failed.add(item.tool)
            continue
        if not _about_destination(item, destination):
            continue  # another place: never evidence for this one
        found.called.add(item.tool)
        if item.tool == "find_attractions":
            if item.ok:
                osm.append(item.text)
                for m in _OSM_LINE.finditer(item.text):
                    found.osm_sights[normalize(m["name"])] = (m["name"], float(m["km"]))
            else:
                found.failed.add(item.tool)
                if item.fallback:
                    wikivoyage.append(item.text)
        elif item.tool == "get_destination_guide":
            if item.ok:
                wikivoyage.append(item.text)
            else:
                found.failed.add(item.tool)
        elif item.tool == "get_weather_forecast":
            if item.ok:
                weather.append(item.text)
                for m in _WEATHER_LINE.finditer(item.text):
                    found.forecast[m["date"]] = m["rest"].rstrip(".")
            else:
                found.failed.add(item.tool)
    found.wikivoyage_text = "\n".join(wikivoyage)
    found.osm_text = "\n".join(osm)
    found.weather_text = "\n".join(weather)
    for currency, patterns in _PRICE_PATTERNS.items():
        amounts = {
            _number(m.group(1))
            for pattern in patterns
            for m in pattern.finditer(found.wikivoyage_text)
        }
        found.guide_prices[currency] = sorted(amounts)
    return found


# --- The checks -----------------------------------------------------------------------


@dataclass
class _Notes:
    warnings: list[str] = field(default_factory=list)
    not_available: list[str] = field(default_factory=list)

    def warn(self, text: str) -> None:
        if text not in self.warnings:
            self.warnings.append(text)

    def missing(self, text: str) -> None:
        if normalize(text) not in {normalize(t) for t in self.not_available}:
            self.not_available.append(text)


def place_source(name: str, src: DestinationSources) -> str | None:
    """The tool result that names this place for src's destination, or None.

    Shared by the compose check (below) and the Stage 9 grounding scorer, so
    both use one rule for "this place came from a tool".
    """
    if normalize(name) in src.osm_sights:
        return "OpenStreetMap"
    if contains_phrase(src.wikivoyage_text, name):
        return "Wikivoyage"
    if contains_phrase(src.builtin_text, name):
        return "Built-in guide"
    if contains_phrase(src.osm_text, name):
        return "OpenStreetMap"
    return None


def _check_activity(
    activity: Activity, src: DestinationSources, notes: _Notes
) -> Activity:
    data = activity.model_dump()
    description, removed = scrub_unverified(activity.description)
    data["description"] = description
    if removed:
        notes.warn(
            f"{activity.name}: details with prices, times or schedules were left "
            "out of the description because no tool result confirms them."
        )
    osm = src.osm_sights.get(normalize(activity.name))
    if osm is not None:
        # Named by OpenStreetMap: use the distance listed for THIS sight.
        data["source"], data["distance_km"] = "OpenStreetMap", osm[1]
        return Activity.model_validate(data)
    data["distance_km"] = None  # only OpenStreetMap gives distances
    source = place_source(activity.name, src)
    if source is not None:
        data["source"] = source
    else:
        notes.warn(
            f"{activity.name}: not found in the tool results for {src.destination}; "
            "check it before you go."
        )
    return Activity.model_validate(data)


def _check_leg(leg: TravelLeg, src: DestinationSources, notes: _Notes) -> TravelLeg:
    route = src.route_between(leg.from_place, leg.to_place)
    if route is None or (route.km is None and route.minutes is None):
        if leg.road_distance_km is not None or leg.driving_minutes is not None:
            notes.missing(
                f"Road distance and driving time, {leg.from_place} to {leg.to_place}"
            )
        return leg.model_copy(
            update={"road_distance_km": None, "driving_minutes": None}
        )
    # The OSRM figures for THIS pair of places replace whatever was submitted.
    return leg.model_copy(
        update={"road_distance_km": route.km, "driving_minutes": route.minutes}
    )


def _check_cost(
    cost: CostItem, src: DestinationSources, notes: _Notes
) -> CostItem | None:
    quoted_currency = cost.quoted_currency
    other_currency = "USD" if quoted_currency == "LKR" else "LKR"
    quoted = cost.amount_lkr if quoted_currency == "LKR" else cost.amount_usd
    assert quoted is not None  # the schema guarantees it
    in_guide = any(
        close(quoted, price, rel=0.005, absolute=0.01)
        for price in src.guide_prices.get(quoted_currency, [])
    )
    if not in_guide:
        notes.missing(
            f"Price of {cost.item} (not quoted in the {src.destination} guide)"
        )
        return None
    converted = next(
        (
            out
            for amount, source, out, target in reversed(src.conversions)
            if source == quoted_currency
            and target == other_currency
            and close(amount, quoted, rel=0.001, absolute=0.01)
        ),
        None,
    )
    if converted is None:
        notes.missing(
            f"{other_currency} value of {cost.item} (no conversion retrieved)"
        )
    key = f"amount_{other_currency.lower()}"
    return cost.model_copy(update={key: converted})


_FACT_TEXTS = {
    "Wikivoyage": lambda src: src.wikivoyage_text,
    "OpenStreetMap": lambda src: src.osm_text,
    "Open-Meteo": lambda src: src.weather_text,
    "OSRM": lambda src: "\n".join(
        r.text
        for r in src.routes
        if same_place(r.origin, src.destination)
        or same_place(r.destination, src.destination)
    ),
}


def _check_fact(
    fact: PracticalFact, src: DestinationSources, notes: _Notes
) -> PracticalFact | None:
    text = _FACT_TEXTS[fact.source](src)
    if not contains_phrase(text, fact.quote):
        notes.warn(
            f"Left out a detail the {fact.source} results do not confirm: "
            f"{fact.text[:80]}"
        )
        return None
    # Numbers in the reworded fact must be in the quote; otherwise show the quote.
    quote_numbers = set(re.findall(r"\d+(?:\.\d+)?", fact.quote.replace(",", "")))
    text_numbers = set(re.findall(r"\d+(?:\.\d+)?", fact.text.replace(",", "")))
    if not text_numbers <= quote_numbers:
        return fact.model_copy(update={"text": fact.quote})
    return fact


def _check_weather(
    plans: Sequence, src: DestinationSources, today: date, notes: _Notes
) -> list[str | None]:
    """The weather text of each day, taken from the forecast line of its date."""
    used: set[str] = set()
    weather: list[str | None] = []
    for plan in plans:
        line = None
        when = plan.forecast_date
        if when and when not in used and when >= today.isoformat():
            line = src.forecast.get(when)
        if line is not None:
            used.add(when)
            weather.append(f"{when}: {line} (Open-Meteo)")
        else:
            weather.append(None)
    if "get_weather_forecast" in src.failed and not src.forecast:
        notes.missing(FAILED_TOOL_NOTES["get_weather_forecast"])
    elif "get_weather_forecast" not in src.called:
        notes.missing("Weather forecast (not retrieved)")
    return weather


def ground_itinerary(
    draft: ItineraryDraft,
    messages: Sequence[AnyMessage],
    *,
    today: date,
    trip_request: dict | None = None,
) -> Itinerary:
    """Check every factual value of a draft against the tool results.

    Returns the checked Itinerary: unconfirmed prices removed, travel figures
    and distances taken from their tool, weather filled from the forecast,
    unbacked facts and risky sentences left out, and every gap listed in
    not_available or warnings. The draft and the messages are not changed.
    """
    src = sources_for(draft.destination, collect_evidence(messages))
    notes = _Notes(not_available=list(draft.not_available))

    if not src.called:
        notes.warn(
            f"No tool results were found for {draft.destination}; the details "
            "below are not confirmed."
        )
    if trip_request:
        wanted_days = trip_request.get("days")
        if isinstance(wanted_days, int) and wanted_days != draft.days:
            notes.warn(f"You asked for {wanted_days} days; this plan has {draft.days}.")
        wanted_place = trip_request.get("destination")
        if wanted_place and not same_place(wanted_place, draft.destination):
            notes.warn(
                f"You asked about {wanted_place}; this plan is for {draft.destination}."
            )

    weather = _check_weather(draft.day_plans, src, today, notes)
    day_plans = [
        DayPlan(
            **plan.model_dump(exclude={"activities", "travel"}),
            activities=[_check_activity(a, src, notes) for a in plan.activities],
            travel=[_check_leg(leg, src, notes) for leg in plan.travel],
            weather=weather[index],
        )
        for index, plan in enumerate(draft.day_plans)
    ]
    costs = [
        c for c in (_check_cost(c, src, notes) for c in draft.costs) if c is not None
    ]
    facts = [
        f for f in (_check_fact(f, src, notes) for f in draft.facts) if f is not None
    ]

    tips = [tip for tip in draft.tips if not has_unverified_detail(tip)]
    if len(tips) < len(draft.tips):
        notes.warn(
            "Left out suggestions that state prices, times or schedules no tool "
            "result confirms."
        )
    summary, removed = scrub_unverified(draft.summary)
    if removed:
        notes.warn("Left out unconfirmed prices, times or schedules from the summary.")

    assumptions = list(draft.assumptions)
    if draft.start_assumed and not any(
        contains_phrase(a, draft.start_from) for a in assumptions
    ):
        assumptions.insert(0, f"The trip starts from {draft.start_from} (assumed).")

    for tool in (
        "find_attractions",
        "get_destination_guide",
        "estimate_travel_time",
        "convert_currency",
    ):
        if tool in src.failed:
            notes.missing(FAILED_TOOL_NOTES[tool])

    return Itinerary(
        **draft.model_dump(
            exclude={
                "day_plans",
                "costs",
                "facts",
                "tips",
                "summary",
                "assumptions",
                "not_available",
            }
        ),
        summary=summary,
        day_plans=day_plans,
        costs=costs,
        facts=facts,
        tips=tips,
        assumptions=assumptions,
        not_available=notes.not_available,
        warnings=notes.warnings,
    )
