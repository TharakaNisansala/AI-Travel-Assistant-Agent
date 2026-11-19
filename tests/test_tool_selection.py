"""Stage 4 checks: does the LLM pick the right tool for each kind of question?

With seven tools, some of them overlapping (guide "see_and_do" vs
find_attractions), tool choice depends on names, docstrings and SYSTEM_PROMPT.
This file holds a fixed set of 17 prompts, each with the tool(s) the agent's
FIRST step should call, and scores Gemini against it.

Only the first step is checked: one Gemini request per prompt, no tools run.
"Where should I hike, and what's the weather there?" first needs
search_destinations; the weather call comes a step later, so it is not part of
that case.

The offline tests check the dataset itself and the scoring. The live test
spends 17 Gemini requests (most of the 20/day free quota) and takes about 4
minutes, because requests are paced to stay under 5 per minute. Run it ALONE,
on a day with quota to spare:
    uv run pytest -m live -s tests/test_tool_selection.py::test_live_tool_selection
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ai_travel_assistant import llm_basics
from ai_travel_assistant.tool_calling import SYSTEM_PROMPT, TOOLS
from tests.test_tool_calling import INCONCLUSIVE, no_retry_llm

TOOL_NAMES = {tool.name for tool in TOOLS}


@dataclass(frozen=True)
class Case:
    prompt: str
    # Tools the first step MUST call (empty: it must answer without tools).
    required: frozenset[str]
    # Extra tools that are acceptable alongside the required ones.
    also_ok: frozenset[str] = field(default_factory=frozenset)


def case(prompt: str, *required: str, also_ok: tuple[str, ...] = ()) -> Case:
    return Case(prompt, frozenset(required), frozenset(also_ok))


CASES: tuple[Case, ...] = (
    # Choosing where to go.
    case("Where in Sri Lanka can I go surfing?", "search_destinations"),
    case("Suggest a good place for a wildlife safari.", "search_destinations"),
    # Weather.
    case("What's the weather in Nuwara Eliya this weekend?", "get_weather_forecast"),
    case(
        "Will it rain in Galle in the next few days? What should I pack?",
        "get_weather_forecast",
    ),
    # Sights at a named place.
    case("What waterfalls and viewpoints are near Ella?", "find_attractions"),
    case(
        "Which temples and museums can I visit in Kandy?",
        "find_attractions",
        also_ok=("get_destination_guide",),
    ),
    # Guide background.
    case(
        "How do I get to Ella from Colombo by train?",
        "get_destination_guide",
        also_ok=("estimate_travel_time",),
    ),
    case("What is Sigiriya like? Give me some background.", "get_destination_guide"),
    case("Any safety or health tips for visiting Mirissa?", "get_destination_guide"),
    # Distances.
    case("How far is Kandy from Nuwara Eliya by road?", "estimate_travel_time"),
    case(
        "How long does it take to drive from Galle to Mirissa?", "estimate_travel_time"
    ),
    # Money.
    case("How much is 100 US dollars in Sri Lankan rupees?", "convert_currency"),
    case(
        "A tuk-tuk ride costs 2500 LKR. How much is that in euros?",
        "convert_currency",
    ),
    # Ticket prices: only from the pricing tool, never the guide or memory.
    case(
        "How much is the entrance ticket for Sigiriya?",
        "get_attraction_prices",
        also_ok=("get_destination_guide",),
    ),
    case(
        "What do foreign tourists pay to enter Polonnaruwa and Anuradhapura?",
        "get_attraction_prices",
    ),
    # Several tools at once.
    case(
        "I'm visiting Kandy next week. What should I see, and what will the "
        "weather be like?",
        "find_attractions",
        "get_weather_forecast",
        also_ok=("get_destination_guide",),
    ),
    # No tool needed.
    case("Hi! What kinds of things can you help me with?"),
)

# 15 of 17 correct (about 88%). Below that, docstrings or SYSTEM_PROMPT need work.
PASS_MARK = 15
# Free tier: 5 requests per minute. 13 s apart keeps a safe margin.
SECONDS_BETWEEN_REQUESTS = 13


def is_correct(expected: Case, called: set[str]) -> bool:
    """All required tools called, and nothing outside required + also_ok."""
    return expected.required <= called <= expected.required | expected.also_ok


def called_tools(message: AIMessage) -> set[str]:
    return {call["name"] for call in message.tool_calls}


# --- Offline: the dataset and the scoring -------------------------------------


def test_there_are_fifteen_cases_with_unique_prompts():
    assert len(CASES) == 17
    assert len({c.prompt for c in CASES}) == len(CASES)


def test_every_expected_tool_is_registered():
    for c in CASES:
        assert c.required | c.also_ok <= TOOL_NAMES, c.prompt
        assert not c.required & c.also_ok, c.prompt


def test_every_tool_is_required_by_at_least_two_cases():
    for name in TOOL_NAMES:
        count = sum(name in c.required for c in CASES)
        assert count >= 2, f"{name} is required by only {count} case(s)"


def test_the_set_includes_a_no_tool_case_and_a_multi_tool_case():
    assert any(not c.required for c in CASES)
    assert any(len(c.required) >= 2 for c in CASES)


def test_pass_mark_leaves_room_for_two_misses():
    assert len(CASES) - 2 == PASS_MARK


@pytest.mark.parametrize(
    ("called", "correct"),
    [
        ({"find_attractions", "get_weather_forecast"}, True),
        ({"find_attractions", "get_weather_forecast", "get_destination_guide"}, True),
        ({"find_attractions"}, False),  # a required tool is missing
        ({"find_attractions", "get_weather_forecast", "convert_currency"}, False),
    ],
)
def test_scoring_requires_all_and_allows_only_listed_extras(called, correct):
    expected = case(
        "Kandy?",
        "find_attractions",
        "get_weather_forecast",
        also_ok=("get_destination_guide",),
    )
    assert is_correct(expected, called) is correct


def test_no_tool_case_fails_if_any_tool_is_called():
    expected = case("Hi!")
    assert is_correct(expected, set())
    assert not is_correct(expected, {"search_destinations"})


# --- Live: score Gemini (17 requests) -----------------------------------------


@pytest.mark.live
def test_live_tool_selection():
    from langchain_google_genai.chat_models import GoogleAPIError, GoogleRateLimitError

    settings = llm_basics.load_settings()
    model = no_retry_llm(settings).bind_tools(list(TOOLS))

    rows: list[tuple[str, Case, set[str] | None]] = []
    for index, expected in enumerate(CASES):
        if index:
            time.sleep(SECONDS_BETWEEN_REQUESTS)
        try:
            reply = model.invoke(
                [SystemMessage(SYSTEM_PROMPT), HumanMessage(expected.prompt)]
            )
        except GoogleRateLimitError as exc:
            reason = llm_basics.safe_error_message(exc, settings.api_key)
            pytest.exit(f"STOP: Gemini 429. Do not re-run live tests today. {reason}")
        except GoogleAPIError as exc:
            if exc.code != 503:
                raise
            rows.append(("SKIP", expected, None))  # Gemini overloaded: no verdict
            continue
        called = called_tools(reply)
        rows.append(
            ("ok" if is_correct(expected, called) else "MISS", expected, called)
        )

    print("\nTool-selection scorecard (first step only):")
    for verdict, expected, called in rows:
        want = ", ".join(sorted(expected.required)) or "(no tool)"
        got = "-" if called is None else ", ".join(sorted(called)) or "(no tool)"
        print(f"  [{verdict:4}] {expected.prompt}\n         want {want} | got {got}")

    decided = [row for row in rows if row[0] != "SKIP"]
    correct = sum(row[0] == "ok" for row in decided)
    print(f"Score: {correct}/{len(decided)} (pass mark {PASS_MARK}/{len(CASES)})")

    if len(decided) < PASS_MARK:
        pytest.skip(f"{INCONCLUSIVE}: only {len(decided)} cases got an answer")
    misses = len(decided) - correct
    assert misses <= len(CASES) - PASS_MARK, f"{misses} misses; see scorecard above"
