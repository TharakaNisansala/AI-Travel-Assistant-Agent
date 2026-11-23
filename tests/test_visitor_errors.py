"""Deployment: plain, safe error messages for public visitors (offline)."""

from __future__ import annotations

import httpx
import pytest

from ai_travel_assistant import robustness
from ai_travel_assistant.provider_pool import AllProvidersUnavailable
from ai_travel_assistant.robustness import (
    VISITOR_BUSY,
    VISITOR_NOT_CONFIGURED,
    VISITOR_QUOTA,
    VISITOR_UNAVAILABLE,
    VISITOR_UNEXPECTED,
    describe_error_for_visitor,
    describe_llm_error,
    error_category,
)
from tests.test_provider_pool import ApiError

# Text a provider error might carry that a visitor must never see.
SECRET = "AIzaSyFAKE-KEY-123"
LEAKY = (
    f"key={SECRET} GEMINI_API_KEY LLM_PROVIDER .env "
    'File "/mount/src/app/provider.py", line 12 Traceback'
)
FORBIDDEN = (
    SECRET,
    "GEMINI_API_KEY",
    "LLM_PROVIDER",
    ".env",
    "Traceback",
    "provider.py",
    "ApiError",
    "RuntimeError",
)

CASES = [
    # (error, category, expected message or None for a rate-limit message)
    (
        AllProvidersUnavailable([("gemini", "quota"), ("groq", "quota")]),
        "quota",
        VISITOR_QUOTA,
    ),
    (ApiError(429, f"GenerateRequestsPerDay {LEAKY}"), "quota", VISITOR_QUOTA),
    (AllProvidersUnavailable([("gemini", "rate_limit")], 40), "rate_limit", None),
    (ApiError(429, f"per minute {LEAKY}"), "rate_limit", None),
    (AllProvidersUnavailable([("gemini", "busy")]), "busy", VISITOR_BUSY),
    (ApiError(503, LEAKY), "busy", VISITOR_BUSY),
    (httpx.ConnectTimeout(LEAKY), "busy", VISITOR_BUSY),
    (TimeoutError(LEAKY), "busy", VISITOR_BUSY),
    (
        AllProvidersUnavailable([("gemini", "rejected")]),
        "unavailable",
        VISITOR_UNAVAILABLE,
    ),
    (ApiError(401, LEAKY), "unavailable", VISITOR_UNAVAILABLE),
    (ApiError(404, LEAKY), "unavailable", VISITOR_UNAVAILABLE),
    (RuntimeError(LEAKY), "unexpected", VISITOR_UNEXPECTED),
]


@pytest.mark.parametrize(("error", "category", "message"), CASES)
def test_each_failure_gets_its_category_and_plain_message(error, category, message):
    assert error_category(error) == category
    text = describe_error_for_visitor(error)
    if message is None:
        assert text.startswith("The free AI service is getting too many requests")
    else:
        assert text == message


@pytest.mark.parametrize(("error", "category", "message"), CASES)
def test_no_visitor_message_leaks_internal_details(error, category, message):
    text = describe_error_for_visitor(error)
    for forbidden in FORBIDDEN:
        assert forbidden not in text


@pytest.mark.parametrize(
    ("seconds", "wait"),
    [
        (40, "about 40 seconds"),
        (3, "about 5 seconds"),
        (70, "about a minute"),
        (180, "about 3 minutes"),
        (None, "a minute"),
        (5000, "a minute"),  # a daily-sized wait is not promised as a time
    ],
)
def test_the_rate_limit_message_says_how_long_to_wait(seconds, wait):
    error = AllProvidersUnavailable([("gemini", "rate_limit")], seconds)
    assert f"Please wait {wait} and try again." in describe_error_for_visitor(error)


def test_every_visitor_message_is_free_of_developer_terms():
    messages = [
        getattr(robustness, name)
        for name in dir(robustness)
        if name.startswith("VISITOR_")
    ]
    assert len(messages) == 6
    for text in messages:
        for term in (".env", "LLM_PROVIDER", "API key", "Gemini", "Groq", "Cloudflare"):
            assert term not in text, (term, text)
    assert VISITOR_NOT_CONFIGURED.startswith("Live AI is not available")


def test_the_developer_messages_are_unchanged():
    """The CLI and local page keep their hints for the developer."""
    quota = ApiError(429, "GenerateRequestsPerDay")
    assert "LLM_PROVIDER" in describe_llm_error(quota)
    assert describe_llm_error(RuntimeError("x")) == (
        "Something went wrong (RuntimeError). Please try again."
    )
