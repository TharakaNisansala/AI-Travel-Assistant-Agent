"""Stage 4: a real external API client for road travel times via OSRM (no LangChain).

What this file demonstrates:
  1. Routing between two places with the OSRM Route API
     (https://router.project-osrm.org/route/v1/driving/...) on OpenStreetMap
     roads. Note the coordinate order: OSRM wants LONGITUDE first.
  2. Reusing the shared geocoder (weather_service.geocode), which is cached, so
     "Ella to Kandy" after a weather question about Ella geocodes only Kandy.
  3. NOT trusting a "successful" answer blindly. OSRM always snaps each point to
     the nearest road it knows, however far away. Asked to route Ella to London,
     the public server answers "code": "Ok" with a 5-day drive, because it
     snapped London to a road in northern Sri Lanka 300 km away. So we check
     the snap distance and reject such routes instead of reporting nonsense.
  4. Being honest about what the number means: OSRM gives free-flow driving
     time (no traffic, no stops). Real journeys on Sri Lankan hill roads, and
     by bus or train, usually take longer. The tool output says so.

The public OSRM demo server is free and needs no key, but asks for at most one
request per second and is car-only. Route data © OpenStreetMap contributors.

Try it on its own (2 geocoding requests + 1 OSRM request, no Gemini call):
    uv run python -c "from ai_travel_assistant.travel_service import \
get_travel_estimate; print(get_travel_estimate('Ella', 'Kandy'))"
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from ai_travel_assistant import faults, weather_service
from ai_travel_assistant.attractions_service import distance_km
from ai_travel_assistant.resilience import TTLCache, send_with_retry
from ai_travel_assistant.weather_service import (
    USER_AGENT,
    Location,
    LocationNotFoundError,
    WeatherServiceError,
)

OSRM_ROUTE_URL = "https://router.project-osrm.org/route/v1/driving"
REQUEST_TIMEOUT = 15.0
# If OSRM had to move a point more than this to reach a road, the route is not
# really between the places the traveller named.
MAX_SNAP_M = 2000.0
# Road distance never changes from minute to minute: keep routes for a day.
ROUTE_CACHE = TTLCache("osrm routes", ttl_s=24 * 3600)


# --- Errors -------------------------------------------------------------------


class TravelServiceError(Exception):
    """Any failure to estimate a travel time. The message is safe to show."""


class TravelLocationNotFoundError(TravelServiceError):
    """One of the places could not be geocoded."""


class NoRouteError(TravelServiceError):
    """OSRM found no road route between the two places."""


class TravelTimeoutError(TravelServiceError):
    """OSRM did not answer within REQUEST_TIMEOUT seconds."""


class TravelNetworkError(TravelServiceError):
    """OSRM could not be reached (no internet, DNS failure, refused, ...)."""


class TravelHTTPError(TravelServiceError):
    """OSRM answered with an HTTP error status."""


class TravelBusyError(TravelHTTPError):
    """OSRM is rate-limiting us or overloaded (429/502/503/504). Temporary."""


class InvalidTravelResponseError(TravelServiceError):
    """OSRM answered, but not with the JSON we expected."""


# --- Typed results ------------------------------------------------------------


@dataclass(frozen=True)
class TravelEstimate:
    origin: Location
    destination: Location
    road_distance_km: float
    duration_min: float  # free-flow driving time, no traffic or stops
    straight_line_km: float


# --- HTTP ---------------------------------------------------------------------


def make_client(transport: httpx.BaseTransport | None = None) -> httpx.Client:
    """An httpx client with our timeout and User-Agent (required by OSRM's policy)."""
    return httpx.Client(
        timeout=REQUEST_TIMEOUT,
        headers={"User-Agent": USER_AGENT},
        # Stage 7: fails on purpose only if FAULT_INJECT names travel.
        transport=transport or faults.transport_for("travel"),
    )


def _get_route_json(client: httpx.Client, url: str) -> dict:
    """GET an OSRM route URL and return its JSON object, or raise."""
    try:
        response = send_with_retry(
            lambda: client.get(url, params={"overview": "false"})
        )
    except httpx.TimeoutException as exc:
        raise TravelTimeoutError(
            f"The OSRM routing API did not respond within {REQUEST_TIMEOUT:g} seconds."
        ) from exc
    except httpx.TransportError as exc:
        raise TravelNetworkError(
            f"Could not connect to the OSRM routing API ({type(exc).__name__}). "
            "Check the internet connection."
        ) from exc
    except httpx.HTTPError as exc:
        raise InvalidTravelResponseError(
            f"The OSRM routing API response could not be read ({type(exc).__name__})."
        ) from exc

    if response.status_code in (429, 502, 503, 504):
        raise TravelBusyError(
            f"The OSRM routing API is busy (HTTP {response.status_code}). "
            "Try again in a minute."
        )

    try:
        data = response.json()
    except ValueError as exc:
        if response.is_error:
            raise TravelHTTPError(
                f"The OSRM routing API returned HTTP {response.status_code}."
            ) from exc
        raise InvalidTravelResponseError(
            "The OSRM routing API returned a response that is not JSON."
        ) from exc
    if not isinstance(data, dict):
        raise InvalidTravelResponseError(
            "The OSRM routing API returned unexpected JSON (not an object)."
        )
    # OSRM explains failures as {"code": "NoRoute", "message": "..."}, sometimes
    # with HTTP 400, so the code is checked before the HTTP status.
    code = data.get("code")
    if code in ("NoRoute", "NoSegment"):
        raise NoRouteError("No road route was found between these places.")
    if response.is_error or code != "Ok":
        detail = data.get("message") or code or "no reason given"
        raise TravelHTTPError(
            f"The OSRM routing API returned HTTP {response.status_code}: {detail}."
        )
    return data


# --- Parsing ------------------------------------------------------------------


def parse_route(data: dict, origin: Location, destination: Location) -> TravelEstimate:
    """Check the snapped points and read distance and duration from the route."""
    waypoints = data.get("waypoints")
    if not isinstance(waypoints, list) or len(waypoints) != 2:
        raise InvalidTravelResponseError(
            "The OSRM response has no start and end point."
        )
    for place, waypoint in zip((origin, destination), waypoints, strict=True):
        snap = weather_service._number(
            waypoint.get("distance") if isinstance(waypoint, dict) else None
        )
        if snap is None:
            raise InvalidTravelResponseError("The OSRM response has a malformed point.")
        if snap > MAX_SNAP_M:
            raise NoRouteError(
                f"No connected road route was found between {origin.name} and "
                f"{destination.name}: the router had to move {place.name} "
                f"{snap / 1000:.0f} km to reach its road network, so a travel time "
                "would not be meaningful."
            )

    routes = data.get("routes")
    route = routes[0] if isinstance(routes, list) and routes else None
    if not isinstance(route, dict):
        raise InvalidTravelResponseError("The OSRM response has no route.")
    distance_m = weather_service._number(route.get("distance"))
    duration_s = weather_service._number(route.get("duration"))
    if distance_m is None or duration_s is None:
        raise InvalidTravelResponseError("The OSRM route has no distance or duration.")

    return TravelEstimate(
        origin=origin,
        destination=destination,
        road_distance_km=distance_m / 1000,
        duration_min=duration_s / 60,
        straight_line_km=distance_km(
            origin.latitude,
            origin.longitude,
            destination.latitude,
            destination.longitude,
        ),
    )


# --- The whole flow -----------------------------------------------------------


def fetch_route(
    origin: Location, destination: Location, *, client: httpx.Client
) -> TravelEstimate:
    """Route between two already-geocoded places (1 HTTP request, cached a day)."""
    cache_key = (origin, destination)
    cached = ROUTE_CACHE.get(cache_key)
    if cached is not TTLCache.MISSING:
        return cached
    # Longitude first: "lon,lat;lon,lat".
    points = (
        f"{origin.longitude},{origin.latitude};"
        f"{destination.longitude},{destination.latitude}"
    )
    data = _get_route_json(client, f"{OSRM_ROUTE_URL}/{points}")
    estimate = parse_route(data, origin, destination)
    ROUTE_CACHE.set(cache_key, estimate)
    return estimate


def _geocode(name: str, country_code: str | None, client: httpx.Client) -> Location:
    """weather_service.geocode, with its errors translated into ours."""
    try:
        return weather_service.geocode(name, country_code, client=client)
    except LocationNotFoundError as exc:
        raise TravelLocationNotFoundError(str(exc)) from exc
    except WeatherServiceError as exc:
        raise TravelServiceError(f"Could not locate '{name}': {exc}") from exc


def get_travel_estimate(
    origin: str,
    destination: str,
    country_code: str | None = "LK",
    *,
    client: httpx.Client | None = None,
) -> TravelEstimate:
    """Geocode both places, then route between them (up to 3 HTTP requests).

    Raises a TravelServiceError subclass for every expected failure.
    """
    if client is not None:
        return _estimate(origin, destination, country_code, client)
    with make_client() as own_client:
        return _estimate(origin, destination, country_code, own_client)


def _estimate(
    origin: str, destination: str, country_code: str | None, client: httpx.Client
) -> TravelEstimate:
    start = _geocode(origin, country_code, client)
    end = _geocode(destination, country_code, client)
    return fetch_route(start, end, client=client)
