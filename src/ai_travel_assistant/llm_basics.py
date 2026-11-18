"""Stage 1 - Lesson 1: LLM fundamentals with LangChain + Gemini.

What this file demonstrates:
  1. Initializing a chat model (ChatGoogleGenerativeAI).
  2. Sending a plain-text prompt with .invoke().
  3. Reading the response object and its .content.
  4. Setting temperature.
  5. Running different prompts (three Sri Lanka travel examples).

Run from the project root:
    uv run python -m ai_travel_assistant.llm_basics            # all examples
    uv run python -m ai_travel_assistant.llm_basics ella       # one example
    uv run python -m ai_travel_assistant.llm_basics sigiriya --temperature 0.2

No LangGraph, agents, tools or memory here: just one prompt in, one reply out.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Settings -----------------------------------------------------------------

# Which company's API runs the model. Set LLM_PROVIDER in .env; Gemini is the
# default, so an .env without it keeps working as before.
PROVIDERS: dict[str, tuple[str, str]] = {
    # provider: (API key variable, model ID variable)
    "gemini": ("GEMINI_API_KEY", "GEMINI_MODEL"),
    "groq": ("GROQ_API_KEY", "GROQ_MODEL"),
}
DEFAULT_PROVIDER = "gemini"


class ConfigError(Exception):
    """Raised when LLM_PROVIDER is unknown or its API key or model is missing."""


@dataclass(frozen=True)
class LLMSettings:
    model: str
    # repr=False: if this object is ever printed or logged, the key is not shown.
    api_key: str = field(repr=False)
    provider: str = DEFAULT_PROVIDER


# The old name, kept so existing code and tests still import it.
GeminiSettings = LLMSettings


def load_settings() -> LLMSettings:
    """Read the provider, model ID and API key from the environment (or .env)."""
    # load_dotenv() copies values from .env into os.environ. Variables that are
    # already set in your shell take priority over the file.
    load_dotenv()
    provider = os.getenv("LLM_PROVIDER", "").strip().lower() or DEFAULT_PROVIDER
    if provider not in PROVIDERS:
        raise ConfigError(
            f"Unknown LLM_PROVIDER '{provider}'. Use one of: {', '.join(PROVIDERS)}."
        )
    key_var, model_var = PROVIDERS[provider]
    api_key = os.getenv(key_var, "").strip()
    model = os.getenv(model_var, "").strip()

    missing = [
        name for name, value in ((key_var, api_key), (model_var, model)) if not value
    ]
    if missing:
        raise ConfigError(
            f"Missing {', '.join(missing)} (LLM_PROVIDER={provider}). "
            "Copy .env.example to .env and fill it in."
        )
    return LLMSettings(model=model, api_key=api_key, provider=provider)


# --- 1. Initializing an LLM ---------------------------------------------------


def build_llm(
    settings: LLMSettings, temperature: float = 1.0, max_retries: int = 3
) -> BaseChatModel:
    """Create a LangChain chat model for the configured provider.

    Creating the object does NOT call the API. It only stores the configuration.
    The network request happens later, when we call .invoke().

    Both classes share LangChain's BaseChatModel interface (invoke, stream,
    bind_tools, with_structured_output), so the rest of the project does not
    care which provider is behind it.

    max_retries=0 turns the client's own retries off, for callers that retry
    in one place themselves (Stage 7: the agent graph's RetryPolicy).
    """
    if settings.provider == "groq":
        # Imported here so Gemini-only setups never load the Groq client.
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=settings.model,  # a Groq model ID that supports tool use
            api_key=settings.api_key,
            temperature=temperature,
            max_retries=max_retries,
            timeout=120,
        )
    return ChatGoogleGenerativeAI(
        model=settings.model,  # which Gemini model to use, e.g. gemini-3.5-flash
        api_key=settings.api_key,  # passed explicitly, never hardcoded
        # Temperature controls randomness when picking the next token:
        #   low (0-0.3)  -> focused and repeatable: good for facts
        #   high (~1.0)  -> more varied wording and ideas: good for suggestions
        # Google recommends keeping Gemini 3.x models at the default of 1.0. Lower
        # values work, but they can sometimes cause repetitive output.
        temperature=temperature,
        max_retries=max_retries,  # retry transient failures (rate limits, 5xx)
        timeout=120,  # seconds before we give up on one request
    )


# --- 2 & 3. Sending a prompt and reading the response -------------------------


def content_to_text(content: str | list) -> str:
    """Turn AIMessage.content into a plain string.

    .content comes in two shapes:
      - a plain str (many models and simple replies), or
      - a list of "content blocks", e.g.
          [{"type": "text", "text": "Hello!", "extras": {"signature": "..."}}]
        Gemini 3.x models return this shape. The signature is an opaque token
        that belongs to the model's hidden reasoning, so we skip it.
    We keep only the text blocks and join them.
    """
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


def ask(llm: BaseChatModel, prompt: str) -> AIMessage:
    """Send one prompt and return the model's full response message."""
    # .invoke() accepts a plain string. LangChain wraps it in a HumanMessage,
    # sends it to Gemini over HTTPS, waits for the reply and returns an AIMessage.
    return llm.invoke(prompt)


# --- 5. Three practical examples ---------------------------------------------
# Each example pairs a prompt with a temperature that suits it.

EXAMPLES: dict[str, tuple[str, float]] = {
    "sri_lanka": (
        "In 3 short bullet points, what makes Sri Lanka a good travel destination?",
        0.3,  # factual -> keep it focused
    ),
    "ella": (
        "I'm visiting Ella, Sri Lanka for 2 days and I love hiking and views. "
        "Recommend 3 things to do, one line each.",
        1.0,  # recommendations -> allow more variety
    ),
    "sigiriya": (
        "Explain Sigiriya to a first-time visitor in under 80 words.",
        0.3,
    ),
}


def run_example(settings: GeminiSettings, name: str, temperature: float | None) -> None:
    prompt, default_temp = EXAMPLES[name]
    temp = default_temp if temperature is None else temperature
    llm = build_llm(settings, temperature=temp)

    response = ask(llm, prompt)

    print(f"\n=== {name}  (temperature={temp}) ===")
    print(f"Prompt: {prompt}\n")
    # response.content: the raw payload (a str, or a list of blocks for Gemini 3).
    print(f"type(response.content) -> {type(response.content).__name__}")
    # Our helper pulls out the readable text.
    print(content_to_text(response.content).strip())
    # Metadata: token usage tells you the cost of the call. "reasoning" tokens
    # are the model's hidden thinking, and they are billed too.
    if response.usage_metadata:
        usage = response.usage_metadata
        print(f"\n[tokens] in={usage['input_tokens']} out={usage['output_tokens']}")


# --- 8. Safe error reporting --------------------------------------------------


def safe_error_message(exc: Exception, api_key: str) -> str:
    """Describe an error without ever leaking the API key."""
    text = str(exc)
    if api_key:
        text = text.replace(api_key, "***REDACTED***")
    return f"{type(exc).__name__}: {text}"


# --- 9. Command-line entry point ---------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LLM basics with Gemini.")
    parser.add_argument("example", nargs="?", default="all", choices=[*EXAMPLES, "all"])
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="override each example's temperature (0.0-2.0)",
    )
    args = parser.parse_args(argv)

    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    names = list(EXAMPLES) if args.example == "all" else [args.example]
    print(f"Model: {settings.model}")
    for name in names:
        try:
            run_example(settings, name, args.temperature)
        except Exception as exc:  # noqa: BLE001 - report any API failure safely
            print(
                f"\n[{name}] Gemini call failed: "
                f"{safe_error_message(exc, settings.api_key)}",
                file=sys.stderr,
            )
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
