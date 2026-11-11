"""Stage 0 smoke test: confirm we can reach Gemini through LangChain.

Run with:
    uv run python smoke_gemini.py

Reads GEMINI_API_KEY and GEMINI_MODEL from the environment (or a local .env file),
sends one short prompt, and prints the reply. The API key is never printed.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import httpx  # transport used by the google-genai SDK; needed to spot network errors
from dotenv import load_dotenv
from google.genai import errors as genai_errors
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import (
    GoogleAuthenticationError,
    GoogleInvalidRequestError,
    GoogleModelNotFoundError,
    GooglePermissionDeniedError,
    GoogleRateLimitError,
)

TEST_PROMPT = "In one short sentence, name one famous place to visit in Sri Lanka."
MODELS_URL = "https://aistudio.google.com/"

# Per-attempt timeout. The google-genai SDK also sends this to Google as the
# X-Server-Timeout header, so too small a value causes 504 DEADLINE_EXCEEDED.
REQUEST_TIMEOUT_SECONDS = 120
# The SDK counts *attempts* (the first request included), so 3 = 1 try + 2 retries.
# Retries use the SDK's exponential backoff with jitter (~1s, ~2s) and apply only to
# transient failures: 408/429/500/502/503/504 and network errors.
MAX_ATTEMPTS = 3


class ConfigError(Exception):
    """Raised when required environment variables are missing."""


@dataclass(frozen=True)
class GeminiSettings:
    model: str
    # repr=False keeps the key out of any accidental print/log of this object.
    api_key: str = field(repr=False)


def load_settings() -> GeminiSettings:
    """Read and validate GEMINI_API_KEY and GEMINI_MODEL."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model = os.getenv("GEMINI_MODEL", "").strip()

    problems = []
    if not api_key:
        problems.append(
            "GEMINI_API_KEY is not set. Create a key in Google AI Studio "
            f"({MODELS_URL}) and put it in your .env file."
        )
    if not model:
        problems.append(
            "GEMINI_MODEL is not set. Pick a currently supported *Flash* model ID "
            f"from Google AI Studio ({MODELS_URL}) and put it in your .env file."
        )
    if problems:
        raise ConfigError("\n".join(f"  - {p}" for p in problems))

    return GeminiSettings(model=model, api_key=api_key)


def _redact(text: str, secret: str) -> str:
    """Belt-and-braces: strip the key from any error text before printing it."""
    return text.replace(secret, "***REDACTED***") if secret else text


def run(settings: GeminiSettings) -> str:
    """Send the test prompt and return the model's text reply."""
    llm = ChatGoogleGenerativeAI(
        model=settings.model,
        # Passed explicitly so a stray GOOGLE_API_KEY env var can't take precedence.
        api_key=settings.api_key,
        temperature=0,
        max_retries=MAX_ATTEMPTS,  # library default is 6 attempts; keep it small
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response = llm.invoke(TEST_PROMPT)
    return response.text


def main() -> int:
    load_dotenv()  # existing environment variables win over .env values

    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"Configuration error:\n{exc}", file=sys.stderr)
        return 2

    print(f"Model:  {settings.model}")
    print(f"Prompt: {TEST_PROMPT}")

    try:
        reply = run(settings)
    except Exception as exc:  # noqa: BLE001 - we classify, then report safely
        print(f"\nGemini call failed: {_describe(exc, settings)}", file=sys.stderr)
        return 1

    print(f"Reply:  {reply.strip()}")
    print("\nSmoke test passed: Gemini is reachable.")
    return 0


def _describe(exc: Exception, settings: GeminiSettings) -> str:
    """Turn an exception into a clear, key-free message."""
    detail = _redact(str(exc), settings.api_key)

    if isinstance(exc, (GoogleAuthenticationError, GooglePermissionDeniedError)):
        return f"authentication/permission problem. Check GEMINI_API_KEY.\n{detail}"
    # The Gemini API reports a bad key as HTTP 400 "API key not valid".
    if isinstance(exc, GoogleInvalidRequestError) and "api key" in detail.lower():
        return f"the API key was rejected. Check GEMINI_API_KEY.\n{detail}"
    if isinstance(exc, GoogleModelNotFoundError):
        return (
            f"model '{settings.model}' was not found. Set GEMINI_MODEL to a "
            f"currently supported Flash model from {MODELS_URL}.\n{detail}"
        )
    if isinstance(exc, GoogleInvalidRequestError):
        return f"the request was rejected (check GEMINI_MODEL).\n{detail}"
    if isinstance(exc, GoogleRateLimitError):
        return f"rate limit / quota exceeded. Wait a minute and retry.\n{detail}"
    if isinstance(exc, (httpx.TransportError, ConnectionError, TimeoutError)):
        return f"network error. Check your internet connection.\n{detail}"
    if isinstance(exc, genai_errors.APIError):
        return f"Gemini API error.\n{detail}"
    return f"unexpected error ({type(exc).__name__}).\n{detail}"


if __name__ == "__main__":
    sys.exit(main())
