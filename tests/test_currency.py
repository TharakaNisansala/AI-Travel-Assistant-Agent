"""Stage 4 checks: the currency service and its convert_currency tool.

Offline tests use httpx.MockTransport with responses shaped like the real
ExchangeRate-API open access endpoint.

Live tests call the real API (never Gemini), 1 request each. Run them ONE AT A
TIME by node ID, e.g.:
    uv run pytest -m live tests/test_currency.py::test_live_usd_to_lkr
"""

from __future__ import annotations

import math

import httpx
import pytest
from langchain_core.messages import AIMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

from ai_travel_assistant import currency_service
from ai_travel_assistant.currency_service import (
    RATES_URL,
    Conversion,
    CurrencyBusyError,
    CurrencyHTTPError,
    CurrencyNetworkError,
    CurrencyServiceError,
    CurrencyTimeoutError,
    InvalidAmountError,
    InvalidCurrencyResponseError,
    UnsupportedCurrencyError,
    convert,
    make_client,
)
from ai_travel_assistant.tool_calling import build_tool_agent_graph, run_tool_agent
from ai_travel_assistant.tools import (
    CURRENCY_SOURCE,
    _money,
    convert_currency,
    format_conversion,
)
from tests.test_attractions import ToolCallingFake, tool_call

USD_RATES = {
    "result": "success",
    "time_last_update_utc": "Sun, 15 Nov 2026 00:02:32 +0000",
    "base_code": "USD",
    "rates": {"USD": 1, "LKR": 332.559829, "EUR": 0.905984, "BAD": "x"},
}
UNSUPPORTED = {"result": "error", "error-type": "unsupported-code"}


class FakeRates:
    def __init__(self, body=None):
        self.body = USD_RATES if body is None else body
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if isinstance(self.body, httpx.Response):
            return self.body
        if callable(self.body):
            return self.body(request)
        return httpx.Response(200, json=self.body)

    def client(self) -> httpx.Client:
        return make_client(transport=httpx.MockTransport(self))


def raising(exc: Exception):
    def handler(request):
        raise exc

    return handler


# --- 1. Converting ------------------------------------------------------------


def test_conversion_is_computed_in_code():
    api = FakeRates()
    result = convert(1250, "USD", "LKR", client=api.client())
    assert result == Conversion(
        amount=1250.0,
        from_code="USD",
        to_code="LKR",
        rate=332.559829,
        converted=pytest.approx(415_699.786),
        updated_utc="Sun, 15 Nov 2026 00:02:32 +0000",
    )
    assert str(api.requests[0].url) == f"{RATES_URL}/USD"


def test_codes_are_normalised():
    result = convert(1, " usd", "lkr ", client=FakeRates().client())
    assert (result.from_code, result.to_code) == ("USD", "LKR")


def test_same_currency_needs_no_request():
    api = FakeRates()
    result = convert(42, "LKR", "lkr", client=api.client())
    assert (result.rate, result.converted) == (1.0, 42.0)
    assert api.requests == []


def test_rates_are_cached_per_base_currency():
    api = FakeRates()
    convert(1, "USD", "LKR", client=api.client())
    convert(5, "USD", "EUR", client=api.client())
    assert len(api.requests) == 1


def test_bad_rate_values_are_dropped():
    with pytest.raises(UnsupportedCurrencyError, match="'BAD'"):
        convert(1, "USD", "BAD", client=FakeRates().client())


# --- 2. Validation before any request -----------------------------------------


@pytest.mark.parametrize("code", ["US", "DOLLARS", "U$D", "", "rupees"])
def test_malformed_codes_are_rejected_before_any_request(code):
    api = FakeRates()
    with pytest.raises(UnsupportedCurrencyError, match="ISO 4217"):
        convert(1, code, "LKR", client=api.client())
    assert api.requests == []


@pytest.mark.parametrize("amount", [-1, math.nan, math.inf, True, "10"])
def test_bad_amounts_are_rejected_before_any_request(amount):
    api = FakeRates()
    with pytest.raises(InvalidAmountError):
        convert(amount, "USD", "LKR", client=api.client())
    assert api.requests == []


def test_zero_is_a_valid_amount():
    assert convert(0, "USD", "LKR", client=FakeRates().client()).converted == 0


# --- 3. Failures --------------------------------------------------------------


@pytest.mark.parametrize(
    ("body", "error"),
    [
        (UNSUPPORTED, UnsupportedCurrencyError),
        ({"result": "error", "error-type": "malformed-request"}, CurrencyHTTPError),
        (raising(httpx.ReadTimeout("slow")), CurrencyTimeoutError),
        (raising(httpx.ConnectError("down")), CurrencyNetworkError),
        (httpx.Response(429), CurrencyBusyError),
        (httpx.Response(502), CurrencyBusyError),
        (httpx.Response(404), CurrencyHTTPError),
        (httpx.Response(200, text="oops"), InvalidCurrencyResponseError),
        (httpx.Response(200, json=[1]), InvalidCurrencyResponseError),
        ({"result": "success", "rates": {}}, InvalidCurrencyResponseError),
    ],
    ids=[
        "unsupported-base",
        "other-error-result",
        "timeout",
        "offline",
        "429",
        "502",
        "404",
        "not-json",
        "not-object",
        "no-rates",
    ],
)
def test_failures_become_typed_errors(body, error):
    with pytest.raises(error):
        convert(1, "XYZ", "LKR", client=FakeRates(body).client())


def test_error_results_with_http_200_are_not_cached():
    responses = iter([UNSUPPORTED, USD_RATES])
    api = FakeRates(lambda request: httpx.Response(200, json=next(responses)))
    with pytest.raises(UnsupportedCurrencyError):
        convert(1, "USD", "LKR", client=api.client())
    assert convert(1, "USD", "LKR", client=api.client()).rate == 332.559829


def test_every_error_is_a_currency_service_error():
    for error in (
        UnsupportedCurrencyError,
        InvalidAmountError,
        CurrencyTimeoutError,
        CurrencyNetworkError,
        CurrencyHTTPError,
        CurrencyBusyError,
        InvalidCurrencyResponseError,
    ):
        assert issubclass(error, CurrencyServiceError)


# --- 4. The convert_currency tool (still offline) -----------------------------


@pytest.fixture
def fake_rates(monkeypatch) -> FakeRates:
    api = FakeRates()
    monkeypatch.setattr(currency_service, "make_client", api.client)
    return api


def test_tool_schema_has_a_number_and_two_codes():
    schema = convert_to_openai_tool(convert_currency)["function"]["parameters"]
    assert schema["properties"]["amount"]["type"] == "number"
    assert schema["required"] == ["amount", "from_currency", "to_currency"]
    assert "remembered exchange rate" in convert_currency.description


@pytest.mark.parametrize(
    ("value", "text"),
    [(33255.9829, "33,255.98"), (0, "0.00"), (0.002717, "0.002717")],
)
def test_money_formatting(value, text):
    assert _money(value) == text


def test_tool_returns_result_rate_caveat_and_attribution(fake_rates):
    text = convert_currency.invoke(
        {"amount": 100, "from_currency": "USD", "to_currency": "LKR"}
    )
    assert text.splitlines() == [
        "100.00 USD = 33,255.98 LKR",
        "Rate used: 1 USD = 332.56 LKR (updated Sun, 15 Nov 2026 00:02:32 +0000).",
        "Indicative mid-market rate: banks, cards and money changers charge a "
        "margin, so the traveller will get somewhat less.",
        CURRENCY_SOURCE,
    ]


def test_same_currency_output_has_no_source_line():
    text = format_conversion(Conversion(5, "LKR", "LKR", 1.0, 5, None))
    assert text.splitlines() == ["5.00 LKR = 5.00 LKR", "Rate used: 1 LKR = 1 LKR."]


def test_tool_reports_failures_instead_of_raising(fake_rates):
    text = convert_currency.invoke(
        {"amount": 10, "from_currency": "dollars", "to_currency": "LKR"}
    )
    assert text.startswith("Currency conversion unavailable:")
    assert "ISO 4217" in text
    assert "do not use a remembered exchange rate" in text
    assert fake_rates.requests == []


# --- 5. The agent using the tool (scripted model, offline) --------------------


def test_agent_binds_and_runs_the_currency_tool(fake_rates):
    model = ToolCallingFake(
        messages=iter(
            [
                tool_call(
                    "convert_currency",
                    {"amount": 50, "from_currency": "USD", "to_currency": "LKR"},
                    "call_c1",
                ),
                AIMessage("50 USD is about 16,628 LKR."),
            ]
        )
    )
    messages = run_tool_agent(
        build_tool_agent_graph(model), "What is 50 dollars in rupees?"
    )
    assert convert_currency in model.bound_tools
    result = messages[2]
    assert (result.name, result.status) == ("convert_currency", "success")
    assert result.content.startswith("50.00 USD = 16,627.99 LKR")


# --- 6. Live (real API, opt-in) -----------------------------------------------

INCONCLUSIVE = "INCONCLUSIVE (infrastructure, not a code failure)"


@pytest.mark.live
def test_live_usd_to_lkr():
    try:
        result = convert(100, "USD", "LKR")
    except (CurrencyBusyError, CurrencyTimeoutError, CurrencyNetworkError) as exc:
        pytest.skip(f"{INCONCLUSIVE}: {exc}")
    # A wide sanity band: the rupee has traded between 180 and 400 per dollar.
    assert 100 * 150 < result.converted < 100 * 500
    assert result.updated_utc
