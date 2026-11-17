"""Stage 4: a real external API client for Open-Meteo weather (no LangChain here).

What this file demonstrates:
  1. Geocoding: turning a place name into coordinates with the Open-Meteo
     Geocoding API (https://geocoding-api.open-meteo.com/v1/search).
  2. Forecasting: calling the Open-Meteo Forecast API
     (https://api.open-meteo.com/v1/forecast) with `current`, `daily` and
     `timezone=auto`.
  3. Turning a large JSON response into small, typed dataclasses, so only the
     fields we need ever reach the LLM.
  4. Turning every failure (timeout, no network, HTTP error, bad JSON, missing
     data, unknown place) into one typed exception family, WeatherServiceError.
  5. Retrying temporary failures once and caching results (see resilience.py).

This is the "services" layer from docs/PROJECT_PLAN.md (section 3.6): plain
Python plus httpx, with no LangChain imports. The @tool wrapper in tools.py
calls it and formats the result. A future MCP server can reuse this file as is.

Open-Meteo is free for non-commercial use and needs no API key.

Try it on its own (2 HTTP requests, no Gemini call):
    uv run python -c "from ai_travel_assistant.weather_service import \
get_weather_report; print(get_weather_report('Ella'))"
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import httpx

from ai_travel_assistant import faults
from ai_travel_assistant.resilience import TTLCache, send_with_retry

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

FORECAST_DAYS = 3
# Seconds to wait for each request before giving up.
REQUEST_TIMEOUT = 10.0
# A plain, generic client name. Polite to free APIs, and contains no personal data.
USER_AGENT = "ai-travel-assistant/0.1 (learning project)"

# Place coordinates do not change, so geocoding results are kept for a day. A
# forecast changes, but not within minutes. All three location tools geocode
# through here, so one agent turn about Ella geocodes Ella only once.
GEOCODE_CACHE = TTLCache("open-meteo geocoding", ttl_s=24 * 3600)
FORECAST_CACHE = TTLCache("open-meteo forecast", ttl_s=15 * 60)

# The fields we ask Open-Meteo for. Asking only for what we use keeps the
# response (and the text the LLM reads) small.
CURRENT_FIELDS = ("temperature_2m", "weather_code", "wind_speed_10m", "precipitation")
DAILY_FIELDS = (
    "weather_code",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_probability_max",
    "precipitation_sum",
    "wind_speed_10m_max",
)

# WMO weather interpretation codes, as listed in the Open-Meteo docs.
WEATHER_CODES: dict[int, str] = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    56: "light freezing drizzle",
    57: "dense freezing drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "heavy freezing rain",
    71: "slight snowfall",
    73: "moderate snowfall",
    75: "heavy snowfall",
    77: "snow grains",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    85: "slight snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with slight hail",
    99: "thunderstorm with heavy hail",
}


def describe_weather_code(code: int | None) -> str | None:
    """A readable description of a WMO weather code, or None if unknown."""
    if code is None:
        return None
    return WEATHER_CODES.get(code, f"unknown conditions (WMO code {code})")


# --- Errors -------------------------------------------------------------------
# One base class, so the tool can catch every weather failure in one place,
# and one subclass per kind of failure, so tests (and later, retries) can tell
# them apart.


class WeatherServiceError(Exception):
    """Any failure to get a weather report. The message is safe to show."""


class LocationNotFoundError(WeatherServiceError):
    """The geocoding API found no place with that name."""


class WeatherTimeoutError(WeatherServiceError):
    """The API did not answer within REQUEST_TIMEOUT seconds."""


class WeatherNetworkError(WeatherServiceError):
    """The API could not be reached (no internet, DNS failure, refused, ...)."""


class WeatherHTTPError(WeatherServiceError):
    """The API answered with an HTTP error status (4xx or 5xx)."""


class InvalidWeatherResponseError(WeatherServiceError):
    """The API answered, but not with the JSON we expected."""


# --- Typed results ------------------------------------------------------------
# Values are None when Open-Meteo did not provide them (it sends null for
# fields it has no data for).


@dataclass(frozen=True)
class Location:
    name: str
    latitude: float
    longitude: float
    country: str | None = None
    admin1: str | None = None  # province / state
    admin2: str | None = None  # district / county
    elevation: float | None = None  # metres

    @property
    def label(self) -> str:
        """e.g. "Ella Town (Badulla District, Uva Province, Sri Lanka)"."""
        parts = [p for p in (self.admin2, self.admin1, self.country) if p]
        return f"{self.name} ({', '.join(parts)})" if parts else self.name


@dataclass(frozen=True)
class CurrentWeather:
    time: str  # local time, ISO 8601, e.g. "2026-11-12T19:00"
    temperature_c: float | None
    weather_code: int | None
    wind_speed_kmh: float | None
    precipitation_mm: float | None

    @property
    def conditions(self) -> str | None:
        return describe_weather_code(self.weather_code)


@dataclass(frozen=True)
class DailyForecast:
    date: str  # local date, ISO 8601, e.g. "2026-11-13"
    weather_code: int | None
    temperature_max_c: float | None
    temperature_min_c: float | None
    precipitation_probability_pct: float | None
    precipitation_sum_mm: float | None
    wind_speed_max_kmh: float | None

    @property
    def conditions(self) -> str | None:
        return describe_weather_code(self.weather_code)


@dataclass(frozen=True)
class WeatherReport:
    location: Location
    timezone: str | None
    current: CurrentWeather | None  # None if the API sent no current block
    daily: tuple[DailyForecast, ...]


# --- HTTP ---------------------------------------------------------------------


def make_client(transport: httpx.BaseTransport | None = None) -> httpx.Client:
    """An httpx client with our timeout and User-Agent.

    `transport` lets tests plug in httpx.MockTransport, so no real network is used.
    """
    return httpx.Client(
        timeout=REQUEST_TIMEOUT,
        headers={"User-Agent": USER_AGENT},
        # Stage 7: fails on purpose only if FAULT_INJECT names weather.
        transport=transport or faults.transport_for("weather"),
    )


def _get_json(client: httpx.Client, url: str, params: dict[str, Any]) -> dict:
    """GET a URL and return its JSON object, or raise a WeatherServiceError."""
    api = "geocoding" if url == GEOCODING_URL else "forecast"
    try:
        # Retries timeouts, dropped connections and 429/502/503/504 once.
        response = send_with_retry(lambda: client.get(url, params=params))
    # TimeoutException is a kind of TransportError, so it must be caught first.
    except httpx.TimeoutException as exc:
        raise WeatherTimeoutError(
            f"The Open-Meteo {api} API did not respond within "
            f"{REQUEST_TIMEOUT:g} seconds."
        ) from exc
    except httpx.TransportError as exc:
        raise WeatherNetworkError(
            f"Could not connect to the Open-Meteo {api} API "
            f"({type(exc).__name__}). Check the internet connection."
        ) from exc
    # The remaining httpx errors (e.g. a corrupt compressed body, too many
    # redirects) are not TransportErrors; without this they would escape the
    # tool and stop the whole agent run.
    except httpx.HTTPError as exc:
        raise InvalidWeatherResponseError(
            f"The Open-Meteo {api} API response could not be read "
            f"({type(exc).__name__})."
        ) from exc

    if response.is_error:
        # Open-Meteo explains bad requests as {"error": true, "reason": "..."}.
        reason = ""
        try:
            body = response.json()
            if isinstance(body, dict) and body.get("reason"):
                reason = f": {body['reason']}"
        except ValueError:
            pass
        raise WeatherHTTPError(
            f"The Open-Meteo {api} API returned HTTP {response.status_code}{reason}."
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise InvalidWeatherResponseError(
            f"The Open-Meteo {api} API returned a response that is not JSON."
        ) from exc
    if not isinstance(data, dict):
        raise InvalidWeatherResponseError(
            f"The Open-Meteo {api} API returned unexpected JSON (not an object)."
        )
    if data.get("error"):
        raise WeatherHTTPError(
            f"The Open-Meteo {api} API reported an error: "
            f"{data.get('reason', 'no reason given')}."
        )
    return data


# --- JSON helpers -------------------------------------------------------------


def _number(value: Any) -> float | None:
    """A float if the value is a real number, else None (null, text, bool ...).

    NaN and infinity count as missing: Python's json module accepts them, and
    they must never reach the LLM as "nan °C".
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _code(value: Any) -> int | None:
    number = _number(value)
    return int(number) if number is not None else None


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _item(values: Any, index: int) -> Any:
    """values[index] if values is a long-enough list, else None."""
    if isinstance(values, list) and index < len(values):
        return values[index]
    return None


# --- Geocoding ----------------------------------------------------------------


def geocode(
    destination: str, country_code: str | None = "LK", *, client: httpx.Client
) -> Location:
    """Find the coordinates of a place by name (best match first).

    country_code is an ISO 3166-1 alpha-2 code such as "LK" (Sri Lanka). It stops
    "Ella" from resolving to a namesake in another country. Pass None or "" to
    search worldwide.
    """
    name = destination.strip()
    if not name:
        raise LocationNotFoundError("No destination name was given.")

    params: dict[str, Any] = {
        "name": name,
        "count": 1,
        "language": "en",
        "format": "json",
    }
    code = (country_code or "").strip().upper()
    if code:
        params["countryCode"] = code

    cache_key = (name.casefold(), code)
    cached = GEOCODE_CACHE.get(cache_key)
    if cached is not TTLCache.MISSING:
        return cached

    data = _get_json(client, GEOCODING_URL, params)
    # When nothing matches, Open-Meteo leaves the "results" key out entirely.
    results = data.get("results")
    if not results:
        where = f" in country '{code}'" if code else ""
        raise LocationNotFoundError(f"No place called '{name}' was found{where}.")
    if not isinstance(results, list) or not isinstance(results[0], dict):
        raise InvalidWeatherResponseError(
            "The Open-Meteo geocoding API returned results in an unexpected format."
        )

    best = results[0]
    latitude, longitude = _number(best.get("latitude")), _number(best.get("longitude"))
    if latitude is None or longitude is None:
        raise InvalidWeatherResponseError(
            f"The geocoding result for '{name}' has no coordinates."
        )
    location = Location(
        name=_text(best.get("name")) or name,
        latitude=latitude,
        longitude=longitude,
        country=_text(best.get("country")),
        admin1=_text(best.get("admin1")),
        admin2=_text(best.get("admin2")),
        elevation=_number(best.get("elevation")),
    )
    GEOCODE_CACHE.set(cache_key, location)
    return location


# --- Forecast -----------------------------------------------------------------


def fetch_forecast(
    location: Location, *, client: httpx.Client, days: int = FORECAST_DAYS
) -> WeatherReport:
    """Get current weather and a daily forecast for a location."""
    cache_key = (location, days)
    cached = FORECAST_CACHE.get(cache_key)
    if cached is not TTLCache.MISSING:
        return cached
    params = {
        "latitude": location.latitude,
        "longitude": location.longitude,
        "current": ",".join(CURRENT_FIELDS),
        "daily": ",".join(DAILY_FIELDS),
        # Report times and day boundaries in the destination's local time zone.
        "timezone": "auto",
        "forecast_days": days,
        # These are Open-Meteo's defaults, but we state them so the units we
        # show the LLM can never silently change.
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
    }
    data = _get_json(client, FORECAST_URL, params)
    report = parse_forecast(data, location)
    FORECAST_CACHE.set(cache_key, report)
    return report


def parse_forecast(data: dict, location: Location) -> WeatherReport:
    """Pick the fields we need out of the forecast JSON.

    A missing daily forecast is an error (it is the main thing we asked for).
    A missing current block, or null values, are allowed and become None.
    """
    daily = data.get("daily")
    dates = daily.get("time") if isinstance(daily, dict) else None
    if not isinstance(dates, list) or not dates:
        raise InvalidWeatherResponseError(
            "The Open-Meteo forecast response has no daily forecast."
        )

    forecasts = tuple(
        DailyForecast(
            date=str(date),
            weather_code=_code(_item(daily.get("weather_code"), i)),
            temperature_max_c=_number(_item(daily.get("temperature_2m_max"), i)),
            temperature_min_c=_number(_item(daily.get("temperature_2m_min"), i)),
            precipitation_probability_pct=_number(
                _item(daily.get("precipitation_probability_max"), i)
            ),
            precipitation_sum_mm=_number(_item(daily.get("precipitation_sum"), i)),
            wind_speed_max_kmh=_number(_item(daily.get("wind_speed_10m_max"), i)),
        )
        for i, date in enumerate(dates)
    )

    current_block = data.get("current")
    current = None
    if isinstance(current_block, dict) and _text(current_block.get("time")):
        current = CurrentWeather(
            time=current_block["time"],
            temperature_c=_number(current_block.get("temperature_2m")),
            weather_code=_code(current_block.get("weather_code")),
            wind_speed_kmh=_number(current_block.get("wind_speed_10m")),
            precipitation_mm=_number(current_block.get("precipitation")),
        )

    return WeatherReport(
        location=location,
        timezone=_text(data.get("timezone")),
        current=current,
        daily=forecasts,
    )


# --- The whole flow -----------------------------------------------------------


def get_weather_report(
    destination: str,
    country_code: str | None = "LK",
    *,
    client: httpx.Client | None = None,
    days: int = FORECAST_DAYS,
) -> WeatherReport:
    """Geocode a destination, then fetch its forecast (2 HTTP requests).

    Raises a WeatherServiceError subclass for every expected failure.
    """
    if client is not None:
        location = geocode(destination, country_code, client=client)
        return fetch_forecast(location, client=client, days=days)
    with make_client() as own_client:
        location = geocode(destination, country_code, client=own_client)
        return fetch_forecast(location, client=own_client, days=days)
