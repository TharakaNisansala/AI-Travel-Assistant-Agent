"""Stage 4: a real external API client for OpenStreetMap attractions (no LangChain).

What this file demonstrates:
  1. Querying OpenStreetMap through the Overpass API
     (https://overpass-api.de/api/interpreter) with Overpass QL: tag filters,
     a radius search around a point, and `out center` for mapped outlines.
  2. Reusing the Open-Meteo geocoder from weather_service, so every tool agrees
     on where a place is.
  3. Filtering in code, not in the LLM: drop unnamed items, remove duplicates,
     sort by distance and cap the count. The LLM only ever sees a few short,
     relevant results, which keeps its context small and its answers grounded.
  4. Turning every failure into one typed exception family,
     AttractionsServiceError, with AttractionsBusyError marking the failures a
     later retry policy may safely retry (server overloaded or timed out).

This is the "services" layer from docs/PROJECT_PLAN.md (section 3.6): plain
Python plus httpx. The @tool wrapper in tools.py calls it and formats the result.

OpenStreetMap data is © OpenStreetMap contributors (ODbL). The public Overpass
server is free and needs no key; keep queries small and send a User-Agent.

Try it on its own (2 HTTP requests: geocoding + Overpass, no Gemini call):
    uv run python -c "from ai_travel_assistant.attractions_service import \
get_attractions_report; print(get_attractions_report('Ella'))"
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, get_args

import httpx

from ai_travel_assistant import weather_service
from ai_travel_assistant.weather_service import (
    USER_AGENT,
    Location,
    LocationNotFoundError,
    WeatherServiceError,
)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

DEFAULT_RADIUS_M = 5000
DEFAULT_LIMIT = 8
# The most elements we ask Overpass for. Overpass does not sort by distance, so
# we fetch a generous batch, then sort and cut it down to `limit` in Python.
SERVER_RESULT_CAP = 100
# Overpass stops the query itself after this many seconds and says so...
QUERY_TIMEOUT_S = 25
# ...so the client waits a little longer, to receive that message instead of
# cutting the connection first.
REQUEST_TIMEOUT = 30.0

# --- Categories ---------------------------------------------------------------
# A fixed list (not free text), so the tool can expose it to the LLM as an enum.

Category = Literal["all", "nature", "culture", "landmarks"]
CATEGORIES: tuple[str, ...] = get_args(Category)

# Overpass QL tag filters per category. Each string filters one `nwr` search;
# the searches inside a category are combined as a union.
CATEGORY_FILTERS: dict[str, tuple[str, ...]] = {
    "landmarks": ('["tourism"="attraction"]',),
    "nature": (
        '["natural"~"^(waterfall|peak)$"]',
        '["tourism"="viewpoint"]',
    ),
    "culture": (
        '["tourism"="museum"]',
        '["historic"~"^(archaeological_site|monument|ruins|castle|fort)$"]',
        # Only places of worship linked to Wikidata: a cheap notability signal
        # that keeps out every small village shrine.
        '["amenity"="place_of_worship"]["wikidata"]',
    ),
}
CATEGORY_FILTERS["all"] = (
    CATEGORY_FILTERS["landmarks"]
    + CATEGORY_FILTERS["nature"]
    + CATEGORY_FILTERS["culture"]
)


# --- Errors -------------------------------------------------------------------


class AttractionsServiceError(Exception):
    """Any failure to find attractions. The message is safe to show."""


class AttractionsLocationNotFoundError(AttractionsServiceError):
    """The destination could not be geocoded."""


class AttractionsTimeoutError(AttractionsServiceError):
    """The Overpass API did not answer within REQUEST_TIMEOUT seconds."""


class AttractionsNetworkError(AttractionsServiceError):
    """The Overpass API could not be reached (no internet, DNS, refused, ...)."""


class AttractionsHTTPError(AttractionsServiceError):
    """The Overpass API answered with an HTTP error status."""


class AttractionsBusyError(AttractionsHTTPError):
    """Overpass is overloaded (429/504) or ran out of time or memory.

    These failures are temporary, so a retry later may succeed.
    """


class InvalidAttractionsResponseError(AttractionsServiceError):
    """The Overpass API answered, but not with the JSON we expected."""


# --- Typed results ------------------------------------------------------------


@dataclass(frozen=True)
class Attraction:
    name: str
    kind: str  # e.g. "waterfall", "viewpoint", "temple", "museum"
    latitude: float
    longitude: float
    distance_km: float  # straight-line distance from the destination


@dataclass(frozen=True)
class AttractionsReport:
    location: Location
    category: str
    radius_m: int
    attractions: tuple[Attraction, ...]  # nearest first, at most `limit`
    total_found: int  # named, de-duplicated matches before the limit


# --- Query building -----------------------------------------------------------


def build_query(
    location: Location, category: str = "all", radius_m: int = DEFAULT_RADIUS_M
) -> str:
    """The Overpass QL query for one category around a location."""
    if category not in CATEGORY_FILTERS:
        raise ValueError(
            f"Unknown category {category!r}; expected one of {', '.join(CATEGORIES)}."
        )
    if radius_m <= 0:
        raise ValueError("radius_m must be positive.")

    around = f"(around:{int(radius_m)},{location.latitude},{location.longitude})"
    searches = "\n".join(
        f"  nwr{tag_filter}{around};" for tag_filter in CATEGORY_FILTERS[category]
    )
    # `out center` gives mapped outlines (ways/relations) one centre point, so
    # every result has a single latitude and longitude.
    return (
        f"[out:json][timeout:{QUERY_TIMEOUT_S}];\n"
        f"(\n{searches}\n);\n"
        f"out center tags {SERVER_RESULT_CAP};"
    )


# --- HTTP ---------------------------------------------------------------------


def make_client(transport: httpx.BaseTransport | None = None) -> httpx.Client:
    """An httpx client with our timeout and User-Agent.

    `transport` lets tests plug in httpx.MockTransport, so no real network is used.
    """
    return httpx.Client(
        timeout=REQUEST_TIMEOUT,
        headers={"User-Agent": USER_AGENT},
        transport=transport,
    )


def _post_query(client: httpx.Client, query: str) -> dict:
    """POST an Overpass query and return its JSON object, or raise."""
    try:
        response = client.post(OVERPASS_URL, data={"data": query})
    # TimeoutException is a kind of TransportError, so it must be caught first.
    except httpx.TimeoutException as exc:
        raise AttractionsTimeoutError(
            f"The Overpass API did not respond within {REQUEST_TIMEOUT:g} seconds."
        ) from exc
    except httpx.TransportError as exc:
        raise AttractionsNetworkError(
            f"Could not connect to the Overpass API ({type(exc).__name__}). "
            "Check the internet connection."
        ) from exc
    except httpx.HTTPError as exc:
        raise InvalidAttractionsResponseError(
            f"The Overpass API response could not be read ({type(exc).__name__})."
        ) from exc

    if response.status_code in (429, 504):
        # 429: too many requests from us; 504: the server is overloaded.
        raise AttractionsBusyError(
            f"The Overpass API is busy (HTTP {response.status_code}). "
            "Try again in a minute."
        )
    if response.is_error:
        # Overpass explains errors in an HTML page, which is not worth parsing.
        raise AttractionsHTTPError(
            f"The Overpass API returned HTTP {response.status_code}."
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise InvalidAttractionsResponseError(
            "The Overpass API returned a response that is not JSON."
        ) from exc
    if not isinstance(data, dict):
        raise InvalidAttractionsResponseError(
            "The Overpass API returned unexpected JSON (not an object)."
        )

    # Overpass can answer HTTP 200 and still have failed: it then adds a
    # "remark" such as "runtime error: Query timed out ...". Treating that as
    # "nothing found" would mislead the traveller, so it is an error.
    remark = data.get("remark")
    if isinstance(remark, str) and "error" in remark.lower():
        raise AttractionsBusyError(
            "The Overpass API could not finish the search "
            f"({remark.strip()[:120]}). Try again in a minute."
        )
    return data


# --- Parsing ------------------------------------------------------------------


def _coordinates(element: dict) -> tuple[float, float] | None:
    """A node's own lat/lon, or the centre point of a way or relation."""
    source = element if "lat" in element else element.get("center")
    if not isinstance(source, dict):
        return None
    latitude = weather_service._number(source.get("lat"))
    longitude = weather_service._number(source.get("lon"))
    if latitude is None or longitude is None:
        return None
    return latitude, longitude


def _name(tags: dict) -> str | None:
    """The English name if mapped, else the default name (may be local script)."""
    for key in ("name:en", "name"):
        value = tags.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


_RELIGION_KINDS = {
    "buddhist": "temple",
    "hindu": "temple",
    "christian": "church",
    "muslim": "mosque",
}


def _kind(tags: dict) -> str:
    """A short human-readable type, from the most specific tag we know."""
    natural = tags.get("natural")
    if natural in ("waterfall", "peak"):
        return natural
    tourism = tags.get("tourism")
    if tourism in ("viewpoint", "museum"):
        return tourism
    if tags.get("amenity") == "place_of_worship":
        return _RELIGION_KINDS.get(tags.get("religion"), "place of worship")
    historic = tags.get("historic")
    if isinstance(historic, str) and historic:
        return historic.replace("_", " ")
    return "attraction"


def distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Straight-line (great-circle) distance in km, using the haversine formula."""
    earth_radius_km = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return 2 * earth_radius_km * math.asin(math.sqrt(a))


def parse_attractions(data: dict, origin: Location) -> list[Attraction]:
    """All named, de-duplicated attractions in an Overpass response, nearest first.

    Elements without a name or coordinates are skipped. When the same name
    appears more than once (e.g. mapped as both a point and an outline), only
    the nearest copy is kept.
    """
    elements = data.get("elements")
    if not isinstance(elements, list):
        raise InvalidAttractionsResponseError(
            "The Overpass API response has no list of elements."
        )

    nearest_by_name: dict[str, Attraction] = {}
    for element in elements:
        if not isinstance(element, dict):
            continue
        tags = element.get("tags")
        if not isinstance(tags, dict):
            continue
        name = _name(tags)
        point = _coordinates(element)
        if name is None or point is None:
            continue
        attraction = Attraction(
            name=name,
            kind=_kind(tags),
            latitude=point[0],
            longitude=point[1],
            distance_km=distance_km(origin.latitude, origin.longitude, *point),
        )
        key = name.casefold()
        kept = nearest_by_name.get(key)
        if kept is None or attraction.distance_km < kept.distance_km:
            nearest_by_name[key] = attraction

    return sorted(nearest_by_name.values(), key=lambda a: (a.distance_km, a.name))


# --- The whole flow -----------------------------------------------------------


def fetch_attractions(
    location: Location,
    category: str = "all",
    *,
    client: httpx.Client,
    radius_m: int = DEFAULT_RADIUS_M,
    limit: int = DEFAULT_LIMIT,
) -> AttractionsReport:
    """Search Overpass around an already-geocoded location (1 HTTP request)."""
    query = build_query(location, category, radius_m)
    data = _post_query(client, query)
    found = parse_attractions(data, location)
    return AttractionsReport(
        location=location,
        category=category,
        radius_m=radius_m,
        attractions=tuple(found[:limit]),
        total_found=len(found),
    )


def _geocode(destination: str, country_code: str | None, client: httpx.Client):
    """weather_service.geocode, with its errors translated into ours.

    Callers of this service then only need to catch AttractionsServiceError.
    """
    try:
        return weather_service.geocode(destination, country_code, client=client)
    except LocationNotFoundError as exc:
        raise AttractionsLocationNotFoundError(str(exc)) from exc
    except WeatherServiceError as exc:
        raise AttractionsServiceError(
            f"Could not locate '{destination}': {exc}"
        ) from exc


def get_attractions_report(
    destination: str,
    category: str = "all",
    country_code: str | None = "LK",
    *,
    client: httpx.Client | None = None,
    radius_m: int = DEFAULT_RADIUS_M,
    limit: int = DEFAULT_LIMIT,
) -> AttractionsReport:
    """Geocode a destination, then find attractions around it (2 HTTP requests).

    Raises ValueError for an unknown category (a programming error), and an
    AttractionsServiceError subclass for every expected failure.
    """
    # Check the category before any network call, so a bad value costs nothing.
    if category not in CATEGORY_FILTERS:
        raise ValueError(
            f"Unknown category {category!r}; expected one of {', '.join(CATEGORIES)}."
        )
    if client is not None:
        location = _geocode(destination, country_code, client)
        return fetch_attractions(
            location, category, client=client, radius_m=radius_m, limit=limit
        )
    with make_client() as own_client:
        location = _geocode(destination, country_code, own_client)
        return fetch_attractions(
            location, category, client=own_client, radius_m=radius_m, limit=limit
        )
