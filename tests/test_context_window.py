"""Stage 5 checks: what the LLM sees of a long conversation (offline)."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately

from ai_travel_assistant.context_window import (
    compact_earlier_turns,
    current_turn_start,
    history_for_llm,
)


def tool_turn(n: int, size: int = 200) -> list:
    """One full turn: question, tool call, a bulky tool result, final answer."""
    return [
        HumanMessage(f"Question {n}"),
        AIMessage(
            "",
            tool_calls=[{"name": "get_weather_forecast", "args": {}, "id": f"c{n}"}],
        ),
        ToolMessage("x" * size, tool_call_id=f"c{n}"),
        AIMessage(f"Answer {n}"),
    ]


def contents(messages) -> list[str]:
    return [m.content for m in messages]


def test_current_turn_start_finds_the_latest_human_message():
    messages = [*tool_turn(1), *tool_turn(2)]
    assert current_turn_start(messages) == 4
    assert current_turn_start([]) == 0


def test_compact_keeps_questions_and_final_answers_only():
    compacted = compact_earlier_turns([*tool_turn(1), *tool_turn(2)])
    assert contents(compacted) == ["Question 1", "Answer 1", "Question 2", "Answer 2"]


def test_current_turn_is_kept_whole_with_its_tool_results():
    current = tool_turn(3, size=50_000)  # far bigger than the budget
    seen = history_for_llm([*tool_turn(1), *current], max_tokens=100)

    assert seen[-4:] == current


def test_old_tool_results_are_not_sent():
    seen = history_for_llm([*tool_turn(1), *tool_turn(2)])

    assert not any(isinstance(m, ToolMessage) for m in seen[:-4])
    assert contents(seen[:2]) == ["Question 1", "Answer 1"]


def test_long_history_keeps_the_newest_turns_within_budget():
    earlier = [m for n in range(50) for m in tool_turn(n)]
    current = tool_turn(99)

    seen = history_for_llm([*earlier, *current], max_tokens=60)
    kept = seen[:-4]

    assert count_tokens_approximately(kept) <= 60
    assert kept, "some recent history should still fit"
    assert contents(kept)[-2:] == ["Question 49", "Answer 49"]
    assert isinstance(kept[0], HumanMessage)


def test_every_tool_call_sent_has_its_result():
    seen = history_for_llm([m for n in range(10) for m in tool_turn(n)], 50)

    call_ids = {
        call["id"] for m in seen if isinstance(m, AIMessage) for call in m.tool_calls
    }
    result_ids = {m.tool_call_id for m in seen if isinstance(m, ToolMessage)}
    assert call_ids == result_ids


def test_none_turns_trimming_off():
    messages = [*tool_turn(1), *tool_turn(2)]
    assert history_for_llm(messages, max_tokens=None) == messages


def test_first_turn_is_unchanged():
    messages = [HumanMessage("Hi")]
    assert history_for_llm(messages) == messages
