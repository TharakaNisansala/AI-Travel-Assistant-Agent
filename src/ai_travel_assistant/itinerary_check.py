"""Stage 4 deliverable: ask the agent for a real 3-day itinerary and check it.

What this file demonstrates:
  1. Running the Stage 3/4 ReAct agent on a realistic multi-tool request.
  2. Checking an agent's TRAJECTORY automatically (which tools it called, with
     which arguments, which ones failed) rather than only reading its answer.
  3. Separating what code can check from what a human must judge: the script
     checks tool coverage and arguments; you then review the plan itself
     against the printed checklist (is every sight, time and price really in
     the tool results?).

The logic is generic: the place name comes only from the command line, so the
same checks run unchanged for Kandy, Galle, Nuwara Eliya, Sigiriya or Ella.

Run from the project root (usually 3-4 Gemini calls; keep runs paced, the free
tier allows 5 requests a minute and 20 a day):
    uv run python -m ai_travel_assistant.itinerary_check Kandy
    uv run python -m ai_travel_assistant.itinerary_check "Nuwara Eliya" --days 2
    uv run python -m ai_travel_assistant.itinerary_check Galle --start Kandy
    uv run python -m ai_travel_assistant.itinerary_check Kandy --save run.md
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from langchain_core.messages import AIMessage, AnyMessage, ToolMessage

from ai_travel_assistant.tool_calling import (
    DEFAULT_ORIGIN,
    build_tool_agent_graph,
    describe_steps,
    final_answer,
    run_tool_agent,
)

# The first-step tools every itinerary needs (see SYSTEM_PROMPT).
REQUIRED_TOOLS = (
    "get_destination_guide",
    "find_attractions",
    "get_weather_forecast",
    "estimate_travel_time",
)
# How each tool's result starts when its data could not be retrieved.
UNAVAILABLE = re.compile(
    r"^(Weather|Attractions|Guide|Travel time|Currency conversion) unavailable\b"
)
FALLBACK_MARKER = "FALLBACK: sights named in the Wikivoyage guide"

# What only a person can judge. Printed after every run.
REVIEW_CHECKLIST = (
    "Every sight in the plan appears in the find_attractions or guide results.",
    "Travel times quoted match the estimate_travel_time results (car, no traffic).",
    "Weather is given only for forecast days; other days say no forecast.",
    "Every price comes from the guide, with LKR/USD from convert_currency.",
    "Each fact names its source (OpenStreetMap, Wikivoyage, OSRM, Open-Meteo, rates).",
    "Failed tools and missing data are listed under 'Not available', not invented.",
    "The days are realistic: not too many sights a day, travel time accounted for.",
)


def build_question(place: str, days: int, start: str | None) -> str:
    """The traveller's request. No start means the agent must assume one."""
    question = f"Plan a {days}-day trip to {place}"
    if start:
        question += f", starting from {start}"
    return question + "."


@dataclass(frozen=True)
class Check:
    passed: bool
    label: str


def tool_calls(messages: Sequence[AnyMessage]) -> list[dict]:
    """Every tool call the model made, in order."""
    return [
        call
        for message in messages
        if isinstance(message, AIMessage)
        for call in message.tool_calls
    ]


def gemini_calls(messages: Sequence[AnyMessage]) -> int:
    """Model requests in the run: one per AIMessage."""
    return sum(isinstance(m, AIMessage) for m in messages)


def failed_tools(messages: Sequence[AnyMessage]) -> list[str]:
    """Names of tools whose result says their data was unavailable."""
    return [
        message.name or "?"
        for message in messages
        if isinstance(message, ToolMessage) and UNAVAILABLE.match(str(message.content))
    ]


def used_guide_fallback(messages: Sequence[AnyMessage]) -> bool:
    return any(
        isinstance(m, ToolMessage) and FALLBACK_MARKER in str(m.content)
        for m in messages
    )


def check_itinerary(
    messages: Sequence[AnyMessage], place: str, start: str | None
) -> list[Check]:
    """The automatic checks on one run. Generic: only `place` varies."""
    calls = tool_calls(messages)
    called = {call["name"] for call in calls}
    answer = final_answer(messages) if messages else ""
    origin = start or DEFAULT_ORIGIN
    place_key = place.casefold()

    checks = [Check(name in called, f"called {name}") for name in REQUIRED_TOOLS]
    # Every place-level tool was asked about the requested place.
    for name in ("get_destination_guide", "find_attractions", "get_weather_forecast"):
        args = [c["args"].get("destination", "") for c in calls if c["name"] == name]
        if args:
            checks.append(
                Check(
                    all(place_key in a.casefold() for a in args),
                    f"{name} asked about {place} (got {args})",
                )
            )
    # The journey there starts from the given (or default) origin.
    journeys = [
        (c["args"].get("origin", ""), c["args"].get("destination", ""))
        for c in calls
        if c["name"] == "estimate_travel_time"
    ]
    checks.append(
        Check(
            any(
                origin.casefold() in o.casefold() and place_key in d.casefold()
                for o, d in journeys
            ),
            f"travel time from {origin} to {place} (got {journeys})",
        )
    )
    if start is None:
        checks.append(
            Check(
                DEFAULT_ORIGIN.casefold() in answer.casefold()
                and "assum" in answer.casefold(),
                f"plan says it assumed {DEFAULT_ORIGIN} as the start",
            )
        )
    checks.append(
        Check("not available" in answer.casefold(), "plan has a 'Not available' list")
    )
    checks.append(Check(bool(answer), "the run ended with a written plan"))
    return checks


def format_report(
    place: str, question: str, messages: Sequence[AnyMessage], checks: list[Check]
) -> str:
    """Everything needed to review one run, as Markdown."""
    failures = failed_tools(messages)
    passed = sum(c.passed for c in checks)
    lines = [
        f"# Itinerary check: {place}",
        "",
        f"**Request:** {question}",
        f"**Gemini calls:** {gemini_calls(messages)}",
        f"**Tools that failed:** {', '.join(failures) if failures else 'none'}",
        "**Wikivoyage fallback for attractions:** "
        + ("used" if used_guide_fallback(messages) else "not needed"),
        "",
        "## Steps",
        "```",
        *describe_steps(messages),
        "```",
        "",
        "## Automatic checks",
        *(f"- [{'x' if c.passed else ' '}] {c.label}" for c in checks),
        "",
        f"**Result:** {passed}/{len(checks)} automatic checks passed",
        "",
        "## Plan",
        "",
        final_answer(messages) if messages else "(no answer)",
        "",
        "## Your review (compare the plan with the tool results)",
        *(f"- [ ] {item}" for item in REVIEW_CHECKLIST),
        "",
        "## Tool results the plan must be based on",
    ]
    for message in messages:
        if isinstance(message, ToolMessage):
            lines += ["", f"### {message.name}", "```", str(message.content), "```"]
    return "\n".join(lines) + "\n"


# --- CLI ----------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Ask the agent for an itinerary and check it (about 3-4 Gemini calls)."
        )
    )
    parser.add_argument("place", help='e.g. "Kandy" or "Nuwara Eliya"')
    parser.add_argument("--days", type=int, default=3, help="trip length (default 3)")
    parser.add_argument(
        "--start",
        help=f"where the traveller starts (default: unstated, so {DEFAULT_ORIGIN})",
    )
    parser.add_argument("--save", type=Path, help="also write the report to this file")
    args = parser.parse_args(argv)
    if args.days < 1:
        parser.error("--days must be at least 1")

    from ai_travel_assistant.llm_basics import (
        ConfigError,
        build_llm,
        load_settings,
        safe_error_message,
    )

    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    question = build_question(args.place, args.days, args.start)
    graph = build_tool_agent_graph(build_llm(settings, temperature=1.0))
    try:
        messages = run_tool_agent(graph, question)
    except Exception as exc:  # e.g. Gemini 429: a clean message, no traceback
        print(
            f"Request failed: {safe_error_message(exc, settings.api_key)}",
            file=sys.stderr,
        )
        return 1

    checks = check_itinerary(messages, args.place, args.start)
    report = format_report(args.place, question, messages, checks)
    print(report)
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(report, encoding="utf-8")
        print(f"Saved to {args.save}")
    return 0 if all(c.passed for c in checks) else 2


if __name__ == "__main__":
    raise SystemExit(main())
