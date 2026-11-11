"""Stage 1 - Lesson 3: structured output with Pydantic.

What this file demonstrates:
  1. Turning free text into a validated TripRequest object (see schemas.py).
  2. llm.with_structured_output(Schema): the model returns a Pydantic object,
     not text.
  3. Using null/None for missing information instead of letting the model guess.
  4. Finding missing critical fields, which will later decide whether the agent
     asks a clarification question.

Run from the project root:
    uv run python -m ai_travel_assistant.structured_output
        # runs all the sample prompts (one Gemini call each)
    uv run python -m ai_travel_assistant.structured_output "3 days in Ella, love hiking"
        # extracts from your own text
    uv run python -m ai_travel_assistant.structured_output --schema
        # prints the JSON schema sent to Gemini (no API call)

No LangGraph, tools or memory here.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ai_travel_assistant.llm_basics import (
    ConfigError,
    build_llm,
    load_settings,
    safe_error_message,
)
from ai_travel_assistant.schemas import TripRequest

EXTRACTION_INSTRUCTIONS = """\
You extract trip requirements for Sri Lanka travel from the traveller's message.

Rules:
- Use only what the traveller actually said. Do not guess or add suggestions.
- Use null for any field the traveller did not mention.
- Convert spoken durations to days: "a weekend" = 2, "a week" = 7,
  "a fortnight" = 14, "N nights" = N + 1.
- "my family of four" means 4 travellers; "we two" or "my partner and I" means 2.
"""

# Sample prompts with the destination and days we expect. The live test uses
# these to measure accuracy, and the CLI runs them when you give no text.
SAMPLE_PROMPTS: list[tuple[str, str | None, int | None]] = [
    ("3 days in Ella, love hiking", "Ella", 3),
    (
        "My partner and I want a weekend in Kandy in April for temples and culture.",
        "Kandy",
        2,
    ),
    ("Planning a week in Galle on a tight budget, beaches and food.", "Galle", 7),
    (
        "5 day luxury trip to Nuwara Eliya for my family of four, relaxed pace.",
        "Nuwara Eliya",
        5,
    ),
    ("Sigiriya for 2 days. I'm a history buff.", "Sigiriya", 2),
    ("I want to see leopards in Yala, 4 days in July.", "Yala", 4),
    (
        "10 days in Trincomalee, snorkelling and whale watching, mid-range budget.",
        "Trincomalee",
        10,
    ),
    ("Thinking of Jaffna for six days, I love food and local culture.", "Jaffna", 6),
    ("Surfing trip to Arugam Bay for a fortnight!", "Arugam Bay", 14),
    ("Where should I go in Sri Lanka?", None, None),  # nothing to extract
]

# The Gemini free tier allows about 5 requests per minute for Flash models.
# Waiting 13 s between calls keeps the 10 samples under that limit.
FREE_TIER_DELAY_SECONDS = 13.0


def extract_trip_request(llm: BaseChatModel, text: str) -> TripRequest:
    """Turn a traveller's message into a validated TripRequest."""
    # with_structured_output() wraps the model. The wrapper sends TripRequest's
    # JSON schema to Gemini, which must reply with JSON that matches it. The
    # wrapper then parses that JSON into a TripRequest. If the JSON breaks the
    # schema's rules, it raises an error instead of returning bad data.
    extractor = llm.with_structured_output(TripRequest)
    result = extractor.invoke(
        [SystemMessage(EXTRACTION_INSTRUCTIONS), HumanMessage(text)]
    )
    if not isinstance(result, TripRequest):  # guards against a None reply
        raise ValueError(f"Expected a TripRequest, got {type(result).__name__}")
    return result


def show_result(text: str, trip: TripRequest) -> None:
    print(f"\nInput:  {text}")
    # model_dump_json() turns the object back into JSON, so you can see every field.
    print(f"Output: {trip.model_dump_json(indent=2)}")
    missing = trip.missing_critical_fields()
    print(f"Missing critical fields: {missing or 'none'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract a TripRequest from text.")
    parser.add_argument("text", nargs="?", help="a travel request (default: samples)")
    parser.add_argument(
        "--schema",
        action="store_true",
        help="print the JSON schema sent to Gemini and exit (no API call)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=FREE_TIER_DELAY_SECONDS,
        help="seconds to wait between sample calls (free-tier rate limit)",
    )
    args = parser.parse_args(argv)

    if args.schema:
        print(json.dumps(TripRequest.model_json_schema(), indent=2))
        return 0

    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    llm = build_llm(settings)  # default 1.0: Google's advice for Gemini 3.x
    texts = [args.text] if args.text else [prompt for prompt, _, _ in SAMPLE_PROMPTS]
    print(f"Model: {settings.model}")

    for i, text in enumerate(texts):
        if i > 0:
            time.sleep(args.delay)  # stay under the requests-per-minute limit
        try:
            trip = extract_trip_request(llm, text)
        except Exception as exc:  # noqa: BLE001 - report API or parsing failures
            print(
                f"\nExtraction failed for {text!r}: "
                f"{safe_error_message(exc, settings.api_key)}",
                file=sys.stderr,
            )
            return 1
        show_result(text, trip)
    return 0


if __name__ == "__main__":
    sys.exit(main())
