"""Stage 4: a real external API client for currency conversion (no LangChain).

What this file demonstrates:
  1. Calling ExchangeRate-API's open access endpoint
     (https://open.er-api.com/v6/latest/USD): no key, about 160 currencies
     including the Sri Lankan rupee (LKR), updated once a day.
  2. Doing the arithmetic in code, not in the LLM. LLMs are unreliable at
     multiplying 332.559829 by 1,250, and their remembered exchange rates are
     out of date. The tool returns the finished number and the rate it used.
  3. An API that reports errors inside an HTTP 200 response
     ({"result": "error", "error-type": "unsupported-code"}), so success must
     be read from the body, not just the status.
  4. Caching by base currency: rates change once a day, and one conversion
     fetches the rates for every currency, so later conversions from the same
     currency need no request at all.

Rates are indicative mid-market rates, not what a bank or money changer pays.
ExchangeRate-API's terms require attribution, which the tool output includes.

Try it on its own (1 HTTP request, no Gemini call):
    uv run python -c "from ai_travel_assistant.currency_service import \
convert; print(convert(100, 'USD', 'LKR'))"
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import httpx

from ai_travel_assistant import weather_service
from ai_travel_assistant.resilience import TTLCache, send_with_retry
from ai_travel_assistant.weather_service import USER_AGENT

RATES_URL = "https://open.er-api.com/v6/latest"
REQUEST_TIMEOUT = 10.0
# The provider updates once a day; an hour keeps us well within its fair use.
RATES_CACHE = TTLCache("exchange rates", ttl_s=3600)

_CODE = re.compile(r"^[A-Z]{3}$")


# --- Errors -------------------------------------------------------------------


class CurrencyServiceError(Exception):
    """Any failure to convert an amount. The message is safe to show."""


class UnsupportedCurrencyError(CurrencyServiceError):
    """A currency code is malformed or not offered by the provider."""


class InvalidAmountError(CurrencyServiceError):
    """The amount is negative, not a number, or infinite."""


class CurrencyTimeoutError(CurrencyServiceError):
    """The rates API did not answer within REQUEST_TIMEOUT seconds."""


class CurrencyNetworkError(CurrencyServiceError):
    """The rates API could not be reached (no internet, DNS failure, ...)."""


class CurrencyHTTPError(CurrencyServiceError):
    """The rates API answered with an HTTP error or an error result."""


class CurrencyBusyError(CurrencyHTTPError):
    """The rates API is rate-limiting us or overloaded (429/502/503/504)."""


class InvalidCurrencyResponseError(CurrencyServiceError):
    """The rates API answered, but not with the JSON we expected."""


# --- Typed results ------------------------------------------------------------


@dataclass(frozen=True)
class Rates:
    base: str
    rates: dict[str, float]  # currency code -> units per 1 `base`
    updated_utc: str | None  # e.g. "Sun, 15 Nov 2026 00:02:32 +0000"


@dataclass(frozen=True)
class Conversion:
    amount: float
    from_code: str
    to_code: str
    rate: float  # units of to_code per 1 from_code
    converted: float
    updated_utc: str | None


# --- Validation ---------------------------------------------------------------


def normalise_code(code: str) -> str:
    """ "usd " -> "USD"; raises UnsupportedCurrencyError if it is not 3 letters."""
    cleaned = (code or "").strip().upper()
    if not _CODE.match(cleaned):
        raise UnsupportedCurrencyError(
            f"'{code}' is not a three-letter ISO 4217 currency code such as USD or LKR."
        )
    return cleaned


def check_amount(amount: float) -> float:
    if isinstance(amount, bool) or not isinstance(amount, (int, float)):
        raise InvalidAmountError("The amount must be a number.")
    if not math.isfinite(amount) or amount < 0:
        raise InvalidAmountError("The amount must be a finite number, zero or more.")
    return float(amount)


# --- HTTP ---------------------------------------------------------------------


def make_client(transport: httpx.BaseTransport | None = None) -> httpx.Client:
    """An httpx client with our timeout and User-Agent."""
    return httpx.Client(
        timeout=REQUEST_TIMEOUT,
        headers={"User-Agent": USER_AGENT},
        transport=transport,
    )


def _get_json(client: httpx.Client, url: str) -> dict:
    try:
        response = send_with_retry(lambda: client.get(url))
    except httpx.TimeoutException as exc:
        raise CurrencyTimeoutError(
            f"The exchange rate API did not respond within {REQUEST_TIMEOUT:g} seconds."
        ) from exc
    except httpx.TransportError as exc:
        raise CurrencyNetworkError(
            f"Could not connect to the exchange rate API ({type(exc).__name__}). "
            "Check the internet connection."
        ) from exc
    except httpx.HTTPError as exc:
        raise InvalidCurrencyResponseError(
            f"The exchange rate API response could not be read ({type(exc).__name__})."
        ) from exc

    if response.status_code in (429, 502, 503, 504):
        raise CurrencyBusyError(
            f"The exchange rate API is busy (HTTP {response.status_code}). "
            "Try again in a minute."
        )
    if response.is_error:
        raise CurrencyHTTPError(
            f"The exchange rate API returned HTTP {response.status_code}."
        )
    try:
        data = response.json()
    except ValueError as exc:
        raise InvalidCurrencyResponseError(
            "The exchange rate API returned a response that is not JSON."
        ) from exc
    if not isinstance(data, dict):
        raise InvalidCurrencyResponseError(
            "The exchange rate API returned unexpected JSON (not an object)."
        )
    return data


# --- Parsing ------------------------------------------------------------------


def parse_rates(data: dict, base: str) -> Rates:
    """Read the rates out of the response, treating {"result": "error"} as failure."""
    if data.get("result") == "error":
        kind = data.get("error-type")
        if kind == "unsupported-code":
            raise UnsupportedCurrencyError(
                f"The exchange rate service does not offer the currency '{base}'."
            )
        raise CurrencyHTTPError(
            f"The exchange rate API reported an error: {kind or 'no reason given'}."
        )
    raw = data.get("rates")
    if data.get("result") != "success" or not isinstance(raw, dict) or not raw:
        raise InvalidCurrencyResponseError(
            "The exchange rate API response has no rates."
        )
    rates = {}
    for code, value in raw.items():
        number = weather_service._number(value)
        if isinstance(code, str) and number is not None and number > 0:
            rates[code.upper()] = number
    updated = data.get("time_last_update_utc")
    return Rates(
        base=base,
        rates=rates,
        updated_utc=updated if isinstance(updated, str) else None,
    )


# --- The whole flow -----------------------------------------------------------


def fetch_rates(base: str, *, client: httpx.Client) -> Rates:
    """All rates from one base currency (1 HTTP request, cached an hour)."""
    cached = RATES_CACHE.get(base)
    if cached is not TTLCache.MISSING:
        return cached
    rates = parse_rates(_get_json(client, f"{RATES_URL}/{base}"), base)
    RATES_CACHE.set(base, rates)
    return rates


def convert(
    amount: float,
    from_code: str,
    to_code: str,
    *,
    client: httpx.Client | None = None,
) -> Conversion:
    """Convert an amount between two currencies at today's indicative rate.

    Raises a CurrencyServiceError subclass for every expected failure. Bad
    codes and amounts are rejected before any network request.
    """
    amount = check_amount(amount)
    source, target = normalise_code(from_code), normalise_code(to_code)
    if source == target:
        return Conversion(amount, source, target, 1.0, amount, None)

    if client is not None:
        rates = fetch_rates(source, client=client)
    else:
        with make_client() as own_client:
            rates = fetch_rates(source, client=own_client)

    rate = rates.rates.get(target)
    if rate is None:
        raise UnsupportedCurrencyError(
            f"The exchange rate service does not offer the currency '{target}'."
        )
    return Conversion(amount, source, target, rate, amount * rate, rates.updated_utc)
