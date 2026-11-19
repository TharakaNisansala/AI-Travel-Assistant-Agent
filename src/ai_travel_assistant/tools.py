"""Stage 3 and 4: tools the LLM can call.

What this file demonstrates:
  1. A plain Python search function over a small, built-in dataset.
  2. Turning it into a LangChain tool with the @tool decorator. The function
     name becomes the tool name, the docstring becomes the description the LLM
     reads, and the type hints become the argument schema.

The dataset is a short hand-written guide to well-known Sri Lankan places. It is
NOT live data: no web search and no external API. The tool says so in its output,
so the LLM (and the traveller) never mistake it for live search results.

Try the tool on its own (offline, no Gemini call):
    uv run python -c "from ai_travel_assistant.tools import search_destinations; \
print(search_destinations.invoke({'query': 'hiking'}))"

Stage 4 adds get_weather_forecast, the first tool backed by a real external API
(Open-Meteo). The HTTP work lives in weather_service.py; this file only wraps it
as a tool and formats the result for the LLM. Try it (2 HTTP requests, no Gemini):
    uv run python -c "from ai_travel_assistant.tools import get_weather_forecast; \
print(get_weather_forecast.invoke({'destination': 'Ella'}))"

find_attractions wraps attractions_service.py (OpenStreetMap via Overpass) the
same way. Its `category` argument is a Literal, so the LLM sees it as an enum of
four values. Try it (2 HTTP requests, no Gemini):
    uv run python -c "from ai_travel_assistant.tools import find_attractions; \
print(find_attractions.invoke({'destination': 'Ella', 'category': 'nature'}))"

Three more tools complete the Stage 4 suite, each wrapping its own service:
  - get_destination_guide (guide_service.py, Wikivoyage): background prose,
    trimmed by `topic` so a 30,000-character article costs ~1,000 tokens.
  - estimate_travel_time (travel_service.py, OSRM): road distance and driving
    time between two places.
  - convert_currency (currency_service.py, ExchangeRate-API): the arithmetic is
    done in code, never by the LLM.
Try them (no Gemini):
    uv run python -c "from ai_travel_assistant.tools import get_destination_guide; \
print(get_destination_guide.invoke({'destination': 'Ella', 'topic': 'see_and_do'}))"
    uv run python -c "from ai_travel_assistant.tools import estimate_travel_time; \
print(estimate_travel_time.invoke({'origin': 'Ella', 'destination': 'Kandy'}))"
    uv run python -c "from ai_travel_assistant.tools import convert_currency; \
print(convert_currency.invoke({'amount': 100, 'from_currency': 'USD', \
'to_currency': 'LKR'}))"
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from langchain_core.tools import tool

from ai_travel_assistant import (
    attractions_service,
    currency_service,
    guide_service,
    travel_service,
    weather_service,
)
from ai_travel_assistant.attractions_service import (
    AttractionsLocationNotFoundError,
    AttractionsReport,
    AttractionsServiceError,
    Category,
)
from ai_travel_assistant.currency_service import Conversion, CurrencyServiceError
from ai_travel_assistant.guide_service import (
    DestinationGuide,
    GuideAmbiguousError,
    GuideServiceError,
    Topic,
)
from ai_travel_assistant.travel_service import TravelEstimate, TravelServiceError
from ai_travel_assistant.weather_service import WeatherReport, WeatherServiceError

# --- The offline dataset ------------------------------------------------------


@dataclass(frozen=True)
class Destination:
    name: str
    location: str  # province and district
    activities: tuple[str, ...]
    # Extra search words that are not activities, e.g. "beach", "mountains".
    tags: tuple[str, ...] = ()


DESTINATIONS: tuple[Destination, ...] = (
    Destination(
        name="Ella",
        location="Uva Province, Badulla District",
        activities=(
            "hiking Little Adam's Peak and Ella Rock",
            "Nine Arch Bridge",
            "tea plantation visits",
            "scenic train ride",
        ),
        tags=("hills", "mountains", "views", "nature"),
    ),
    Destination(
        name="Nuwara Eliya",
        location="Central Province, Nuwara Eliya District",
        activities=(
            "hiking to World's End in Horton Plains National Park",
            "tea factory tours",
            "Gregory Lake",
        ),
        tags=("hills", "mountains", "cool climate", "tea", "nature"),
    ),
    Destination(
        name="Adam's Peak (Sri Pada)",
        location="Central and Sabaragamuwa Provinces, near Dalhousie",
        activities=(
            "overnight pilgrimage hike to the summit",
            "sunrise views from the top",
        ),
        tags=("mountains", "pilgrimage", "climbing", "trekking"),
    ),
    Destination(
        name="Sigiriya",
        location="Central Province, Matale District",
        activities=(
            "climbing Sigiriya Lion Rock",
            "ancient frescoes and water gardens",
            "Pidurangala Rock hike",
        ),
        tags=("history", "ancient", "unesco", "culture", "climbing"),
    ),
    Destination(
        name="Kandy",
        location="Central Province, Kandy District",
        activities=(
            "Temple of the Sacred Tooth Relic",
            "Peradeniya Royal Botanical Gardens",
            "Kandyan cultural dance shows",
        ),
        tags=("culture", "temples", "history", "unesco", "gardens"),
    ),
    Destination(
        name="Anuradhapura",
        location="North Central Province, Anuradhapura District",
        activities=(
            "ancient stupas such as Ruwanwelisaya",
            "Jaya Sri Maha Bodhi sacred fig tree",
            "cycling around the ruins",
        ),
        tags=("history", "ancient", "unesco", "temples", "culture", "buddhist"),
    ),
    Destination(
        name="Galle",
        location="Southern Province, Galle District",
        activities=(
            "walking the ramparts of Galle Fort",
            "colonial-era streets, cafes and museums",
            "nearby beaches such as Unawatuna",
        ),
        tags=("history", "unesco", "beach", "coast", "culture"),
    ),
    Destination(
        name="Mirissa",
        location="Southern Province, Matara District",
        activities=("whale watching", "surfing", "beach relaxation"),
        tags=("beach", "coast", "ocean", "sea"),
    ),
    Destination(
        name="Arugam Bay",
        location="Eastern Province, Ampara District",
        activities=("surfing", "lagoon safaris", "beach relaxation"),
        tags=("beach", "coast", "surf", "ocean"),
    ),
    Destination(
        name="Yala National Park",
        location="Southern and Uva Provinces, Hambantota and Monaragala Districts",
        activities=("jeep safaris", "leopard and elephant spotting", "bird watching"),
        tags=("wildlife", "safari", "nature", "animals"),
    ),
)


# --- Searching ----------------------------------------------------------------

# Words that say nothing about WHAT the traveller wants, so they must not count
# as matches. "sri" and "lanka" are here because every place is in Sri Lanka.
_STOPWORDS = frozenset(
    {
        "a", "an", "and", "any", "are", "best", "can", "do", "find", "for",
        "go", "good", "i", "in", "is", "like", "looking", "me", "my", "of",
        "on", "or", "place", "places", "please", "recommend", "show", "some",
        "sri", "lanka", "suggest", "that", "the", "there", "to", "visit",
        "want", "what", "where", "which", "with", "you",
    }
)  # fmt: skip


def _stem(word: str) -> str:
    """Cut common endings so different forms of a word match each other.

    hiking/hike -> "hik", beaches/beach -> "beach", temples/temple -> "templ".
    Deliberately tiny: good enough for a short, fixed dataset.
    """
    if word.endswith("ing") and len(word) > 5:
        word = word[:-3]
    elif word.endswith("es") and len(word) > 4:
        word = word[:-2]
    elif word.endswith("s") and len(word) > 3:
        word = word[:-1]
    if word.endswith("e") and len(word) > 3:
        word = word[:-1]
    return word


def _keywords(text: str) -> set[str]:
    """Lowercase words of the text, without stopwords, stemmed."""
    words = re.findall(r"[a-z]+", text.lower())
    return {_stem(word) for word in words if word not in _STOPWORDS}


def _searchable_text(destination: Destination) -> str:
    return " ".join(
        (destination.name, destination.location, *destination.activities)
        + destination.tags
    )


def find_destinations(query: str, limit: int = 5) -> list[Destination]:
    """Return the places that best match the query, best match first.

    A place scores one point for each query keyword found in its name, location,
    activities or tags. Places with no points are left out.
    """
    wanted = _keywords(query)
    if not wanted:
        return []
    scored = []
    for destination in DESTINATIONS:
        score = len(wanted & _keywords(_searchable_text(destination)))
        if score:
            scored.append((score, destination))
    # sorted() is stable: equal scores keep the dataset's order.
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [destination for _, destination in scored[:limit]]


def format_destinations(query: str, matches: list[Destination]) -> str:
    """Turn search results into the text the LLM will read."""
    source = "Source: the assistant's built-in offline guide (not live data)."
    if not matches:
        return (
            f"No destinations in the offline guide matched '{query}'. "
            "Try a broader activity such as hiking, beaches, wildlife, history "
            f"or tea.\n{source}"
        )
    lines = [f"Found {len(matches)} destination(s) for '{query}':"]
    for destination in matches:
        lines.append(
            f"- {destination.name} ({destination.location}): "
            f"{'; '.join(destination.activities)}"
        )
    lines.append(source)
    return "\n".join(lines)


# --- The tool -----------------------------------------------------------------
# @tool reads three things from the function below and sends them to the LLM:
#   name        -> "search_destinations"
#   description -> the docstring (this is how the LLM knows WHEN to use the tool)
#   args schema -> {"query": string}, built from the type hints
# The LLM never runs this code. It only asks for it to be run (a tool call);
# LangGraph's ToolNode runs it and sends the returned text back to the LLM.


@tool
def search_destinations(query: str) -> str:
    """Search a built-in guide of Sri Lankan destinations.

    Use this when the traveller asks where to go in Sri Lanka or which places
    suit an activity or interest (for example hiking, beaches, surfing,
    wildlife, history, temples or tea plantations). Returns each matching
    place's name, location and activities, or a message saying nothing matched.

    Args:
        query: What the traveller is looking for, in plain words,
            e.g. "hiking" or "beaches for surfing".
    """
    return format_destinations(query, find_destinations(query))


# --- Weather (Stage 4: a real external API) -----------------------------------


def _show(value: float | None, unit: str, digits: int = 1) -> str:
    """e.g. 19.1 -> "19.1 °C"; None -> "n/a" (Open-Meteo sent no value)."""
    return "n/a" if value is None else f"{value:.{digits}f} {unit}"


def _weekday(iso_date: str) -> str:
    """ "2026-11-13" -> "Fri"; returns "" if the date cannot be parsed."""
    try:
        return date.fromisoformat(iso_date).strftime("%a")
    except ValueError:
        return ""


def format_weather_report(report: WeatherReport) -> str:
    """Turn a WeatherReport into the short text the LLM will read.

    Only the useful fields, always with units, and "n/a" for missing values,
    so the LLM never has to guess what a number means or invent a missing one.
    """
    loc = report.location
    header = (
        f"Weather forecast for {loc.label}, "
        f"at lat {loc.latitude:.2f}, lon {loc.longitude:.2f}"
        + (f", elevation {loc.elevation:.0f} m" if loc.elevation is not None else "")
        + "."
    )
    lines = [header]
    if report.timezone:
        lines.append(f"Times are local ({report.timezone}).")

    current = report.current
    if current is None:
        lines.append("Current conditions: not available.")
    else:
        lines.append(
            f"Current conditions ({current.time.replace('T', ' ')}): "
            f"{_show(current.temperature_c, '°C')}, "
            f"{current.conditions or 'conditions n/a'}, "
            f"wind {_show(current.wind_speed_kmh, 'km/h')}, "
            f"precipitation {_show(current.precipitation_mm, 'mm')}."
        )

    lines.append(f"Daily forecast ({len(report.daily)} days):")
    for day in report.daily:
        weekday = _weekday(day.date)
        when = f"{day.date} ({weekday})" if weekday else day.date
        rain_chance = (
            "n/a"
            if day.precipitation_probability_pct is None
            else f"{day.precipitation_probability_pct:.0f}%"
        )
        lines.append(
            f"- {when}: {day.conditions or 'conditions n/a'}; "
            f"min {_show(day.temperature_min_c, '°C')}, "
            f"max {_show(day.temperature_max_c, '°C')}; "
            f"chance of precipitation {rain_chance}; "
            f"rainfall {_show(day.precipitation_sum_mm, 'mm')}; "
            f"max wind {_show(day.wind_speed_max_kmh, 'km/h')}."
        )

    if any("n/a" in line for line in lines):
        lines.append("Values shown as n/a were not provided by the weather service.")
    lines.append("Source: Open-Meteo live forecast API (open-meteo.com).")
    return "\n".join(lines)


# Errors are RETURNED as text, not raised: the LLM reads the message and can
# tell the traveller what went wrong, and the rest of the agent keeps running.


@tool
def get_weather_forecast(destination: str, country_code: str = "LK") -> str:
    """Get the live weather forecast for a destination from the Open-Meteo API.

    Use this whenever the traveller asks about the weather, temperature, rain,
    wind or what to pack for a place. Returns the current conditions and a
    3-day daily forecast (conditions, min/max temperature in °C, chance of
    precipitation, rainfall in mm, max wind in km/h), or a message explaining
    why the forecast is unavailable.

    Args:
        destination: Only the place name, e.g. "Ella" or "Nuwara Eliya".
        country_code: Two-letter ISO country code of the place. Defaults to
            "LK" (Sri Lanka); use another code such as "IN" for places
            outside Sri Lanka.
    """
    try:
        report = weather_service.get_weather_report(destination, country_code)
    except WeatherServiceError as exc:
        return (
            f"Weather unavailable for '{destination}': {exc} "
            "Tell the traveller the forecast could not be retrieved; "
            "do not guess the weather."
        )
    return format_weather_report(report)


# --- Attractions (Stage 4: OpenStreetMap via Overpass) ------------------------

ATTRACTIONS_SOURCE = (
    "Source: OpenStreetMap via the Overpass API (© OpenStreetMap contributors)."
)


def format_attractions_report(report: AttractionsReport) -> str:
    """Turn an AttractionsReport into the short text the LLM will read.

    One line per attraction (name, kind, distance), plus the search scope, so
    the LLM knows what was searched and how much was left out. No raw JSON,
    OSM ids or tags: they cost context and add nothing to the answer.
    """
    place = report.location.label
    radius = f"{report.radius_m / 1000:g} km"
    shown = len(report.attractions)

    if shown == 0:
        lines = [
            f"No named attractions found in OpenStreetMap within {radius} of "
            f"{place} (category: {report.category})."
        ]
        if report.category != "all":
            lines.append("Try again with category 'all' for a wider search.")
        lines.append("Tell the traveller none were found; do not invent attractions.")
        lines.append(ATTRACTIONS_SOURCE)
        return "\n".join(lines)

    lines = [
        f"Attractions near {place}; category: {report.category}; "
        f"within {radius}; nearest first:"
    ]
    for attraction in report.attractions:
        lines.append(
            f"- {attraction.name} ({attraction.kind}), {attraction.distance_km:.1f} km"
        )
    if report.total_found > shown:
        lines.append(f"Showing the {shown} nearest of {report.total_found} found.")
    lines.append("Distances are straight-line, not travel distance.")
    lines.append(ATTRACTIONS_SOURCE)
    return "\n".join(lines)


# `category` is typed with the service's Literal, so the schema sent to the LLM
# is an enum of exactly the categories the service supports. A value outside it
# is rejected by the schema before this function runs, and ToolNode returns that
# error to the LLM. Expected runtime failures (unknown place, timeout, Overpass
# busy) are caught here and returned as text, like get_weather_forecast.


@tool
def find_attractions(
    destination: str, category: Category = "all", country_code: str = "LK"
) -> str:
    """Find real attractions near a destination, from OpenStreetMap.

    Use this when the traveller asks what to see or do in or around a place,
    or when planning sightseeing for a trip. Searches within 5 km of the place
    and returns up to 8 named attractions, nearest first, each with its kind
    (e.g. waterfall, viewpoint, temple, museum) and straight-line distance in
    km, or a message explaining why none could be found. If OpenStreetMap
    cannot be reached, it returns the sights named in the place's Wikivoyage
    guide instead (without distances) and says so. It does NOT cover hotels
    (use get_destination_guide with topic "accommodation"), restaurants, shops
    or hiking routes.

    Args:
        destination: Only the place name, e.g. "Ella" or "Kandy".
        category: What kind of attractions to find:
            "nature" - waterfalls, peaks and viewpoints;
            "culture" - museums, historic sites and notable places of worship;
            "landmarks" - places mapped as tourist attractions;
            "all" - every category above (default).
        country_code: Two-letter ISO country code of the place. Defaults to
            "LK" (Sri Lanka); use another code such as "IN" for places
            outside Sri Lanka.
    """
    try:
        report = attractions_service.get_attractions_report(
            destination, category, country_code
        )
    except AttractionsLocationNotFoundError as exc:
        # The place itself is unknown: a guide lookup would not help.
        return (
            f"Attractions unavailable for '{destination}': {exc} "
            "Tell the traveller the attractions could not be retrieved; "
            "do not invent attractions."
        )
    except AttractionsServiceError as exc:
        # Every Overpass server failed: fall back to the Wikivoyage guide.
        return attractions_from_guide(destination, exc)
    return format_attractions_report(report)


def attractions_from_guide(destination: str, overpass_error: Exception) -> str:
    """The final fallback when OpenStreetMap could not be reached.

    Returns the "see and do" part of the place's Wikivoyage guide, clearly
    labelled as a fallback: the sights are named but have no distances or
    categories. The first line always starts "Attractions unavailable for",
    so the agent and the tests can still tell that OpenStreetMap failed.
    """
    unavailable = f"Attractions unavailable for '{destination}': {overpass_error}"
    try:
        guide = guide_service.get_destination_guide(destination, "see_and_do")
    except GuideServiceError as guide_error:
        return (
            f"{unavailable} The Wikivoyage fallback also failed: {guide_error} "
            "Tell the traveller the attractions could not be retrieved; "
            "do not invent attractions."
        )
    if not guide.sections:
        return (
            f"{unavailable} The Wikivoyage guide for {guide.title} names no "
            "sights either. Tell the traveller the attractions could not be "
            "retrieved; do not invent attractions."
        )
    return "\n".join(
        [
            unavailable,
            "FALLBACK: sights named in the Wikivoyage guide instead "
            "(no distances, not filtered by category).",
            format_destination_guide(guide),
            "Mention only sights named above, say they come from Wikivoyage "
            "because OpenStreetMap could not be reached, and do not invent "
            "attractions or distances.",
        ]
    )


# --- Destination guide (Stage 4: Wikivoyage) ----------------------------------

GUIDE_SOURCE = (
    "Source: Wikivoyage, community-written (CC BY-SA 4.0); prices, timetables "
    "and transport details may be out of date."
)
TOPIC_LABELS = {
    "overview": "overview",
    "getting_there": "getting there",
    "getting_around": "getting around",
    "see_and_do": "see and do",
    "accommodation": "accommodation",
    "safety": "safety and health",
}


def format_destination_guide(guide: DestinationGuide) -> str:
    """Turn a DestinationGuide into the text the LLM will read.

    One short paragraph per section, a note on what the article lacks (so the
    LLM does not fill the gap from memory), and the source with its caveat.
    """
    topic = "all topics" if guide.topic == "all" else TOPIC_LABELS[guide.topic]
    lines = [f"Wikivoyage guide for {guide.title} ({topic}):"]
    if not guide.sections:
        lines.append("The article has nothing on this topic.")
    for section in guide.sections:
        lines.append(f"[{section.title}] {section.text}")
    if guide.missing:
        labels = ", ".join(TOPIC_LABELS[t] for t in guide.missing)
        lines.append(f"Not covered by this article: {labels}. Do not fill this in.")
    if any(section.truncated for section in guide.sections):
        lines.append(
            f"Sections ending in ... were shortened; full article: {guide.url}"
        )
    else:
        lines.append(f"Full article: {guide.url}")
    lines.append(GUIDE_SOURCE)
    return "\n".join(lines)


@tool
def get_destination_guide(destination: str, topic: Topic = "all") -> str:
    """Get travel-guide background on a destination from Wikivoyage.

    Use this when the traveller asks what a place is like, how to get there,
    how to get around, practical or safety tips, or wants background while
    planning a trip. Returns short excerpts of the place's Wikivoyage article
    for the chosen topic, or a message explaining why no guide was found
    (if the name matches several places, the message lists the article names
    to retry with). Guide text can be out of date and has no live data: use
    get_weather_forecast for weather and find_attractions for a list of
    nearby sights with distances.

    Args:
        destination: Only the place name as Wikivoyage titles it, e.g. "Ella",
            "Kandy" or "Nuwara Eliya".
        topic: Which part of the guide to return:
            "overview" - what the place is like;
            "getting_there" - how to reach it (train, bus, air);
            "getting_around" - local transport;
            "see_and_do" - sights and activities;
            "accommodation" - where to stay (guesthouses, hotels, price ranges);
            "safety" - safety and health tips;
            "all" - a shorter excerpt of every topic above (default).
    """
    try:
        guide = guide_service.get_destination_guide(destination, topic)
    except GuideAmbiguousError as exc:
        return (
            f"Guide unavailable for '{destination}': {exc} "
            "If one of these is the place the traveller means, call this tool "
            "again with that exact name; otherwise ask the traveller which one."
        )
    except GuideServiceError as exc:
        return (
            f"Guide unavailable for '{destination}': {exc} "
            "Tell the traveller the guide could not be retrieved; "
            "do not make up guide information."
        )
    return format_destination_guide(guide)


# --- Travel time (Stage 4: OSRM) ----------------------------------------------

TRAVEL_SOURCE = (
    "Source: OSRM routing on OpenStreetMap roads (© OpenStreetMap contributors)."
)


def _duration(minutes: float) -> str:
    """159.6 -> "2 h 40 min"; 45.2 -> "45 min"."""
    total = max(1, round(minutes))
    hours, mins = divmod(total, 60)
    if hours == 0:
        return f"{mins} min"
    return f"{hours} h {mins} min" if mins else f"{hours} h"


def format_travel_estimate(estimate: TravelEstimate) -> str:
    """Turn a TravelEstimate into the short text the LLM will read."""
    return "\n".join(
        [
            f"Road trip from {estimate.origin.label} to {estimate.destination.label}:",
            f"- Road distance: {estimate.road_distance_km:.0f} km "
            f"(straight line {estimate.straight_line_km:.0f} km).",
            f"- Driving time: about {_duration(estimate.duration_min)} by car "
            "without traffic or stops.",
            "This is a free-flow car estimate: real journeys, especially on hill "
            "roads or by bus, tuk-tuk or train, usually take longer. Say so.",
            TRAVEL_SOURCE,
        ]
    )


@tool
def estimate_travel_time(
    origin: str, destination: str, country_code: str = "LK"
) -> str:
    """Estimate the road distance and driving time between two places (OSRM).

    Use this when the traveller asks how far apart two places are, how long it
    takes to get from one to the other, or when ordering stops in a multi-place
    trip. Returns the road distance in km and a car driving time without
    traffic, or a message explaining why no route was found. It does NOT give
    train or bus timetables (get_destination_guide's getting_there topic may
    describe those).

    Args:
        origin: Only the starting place name, e.g. "Ella".
        destination: Only the destination place name, e.g. "Kandy".
        country_code: Two-letter ISO country code of both places. Defaults to
            "LK" (Sri Lanka).
    """
    try:
        estimate = travel_service.get_travel_estimate(origin, destination, country_code)
    except TravelServiceError as exc:
        return (
            f"Travel time unavailable from '{origin}' to '{destination}': {exc} "
            "Tell the traveller the estimate could not be retrieved; "
            "do not guess distances or times."
        )
    return format_travel_estimate(estimate)


# --- Currency (Stage 4: ExchangeRate-API) -------------------------------------

CURRENCY_SOURCE = (
    "Source: Rates By Exchange Rate API (https://www.exchangerate-api.com)."
)


def _money(value: float) -> str:
    """33255.98 -> "33,255.98"; tiny values keep enough digits to be useful."""
    if value != 0 and abs(value) < 0.01:
        return f"{value:.6f}"
    return f"{value:,.2f}"


def format_conversion(conversion: Conversion) -> str:
    """Turn a Conversion into the short text the LLM will read."""
    c = conversion
    lines = [
        f"{_money(c.amount)} {c.from_code} = {_money(c.converted)} {c.to_code}",
        f"Rate used: 1 {c.from_code} = {c.rate:.6g} {c.to_code}"
        + (f" (updated {c.updated_utc})." if c.updated_utc else "."),
    ]
    if c.from_code != c.to_code:
        lines.append(
            "Indicative mid-market rate: banks, cards and money changers "
            "charge a margin, so the traveller will get somewhat less."
        )
        lines.append(CURRENCY_SOURCE)
    return "\n".join(lines)


@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert an amount of money between currencies at today's exchange rate.

    Use this whenever the traveller asks what a price is worth in another
    currency, how much money to bring, or about exchange rates, e.g. US dollars
    to Sri Lankan rupees. Returns the converted amount and the rate used
    (updated daily), or a message explaining why it failed. Always use this
    instead of a remembered exchange rate.

    Args:
        amount: The amount to convert, e.g. 100 or 2500.50.
        from_currency: Three-letter ISO 4217 code of the amount, e.g. "USD".
        to_currency: Three-letter ISO 4217 code to convert to, e.g. "LKR".
    """
    try:
        conversion = currency_service.convert(amount, from_currency, to_currency)
    except CurrencyServiceError as exc:
        return (
            f"Currency conversion unavailable: {exc} "
            "Tell the traveller the conversion could not be done; "
            "do not use a remembered exchange rate."
        )
    return format_conversion(conversion)
