"""Stage 5 checks: short-term memory (checkpointer + thread_id) for the agent.

Offline only: a scripted fake model plays the LLM, so no Gemini calls. We check
what the model RECEIVES on a follow-up turn: with the same thread_id it must see
the earlier turn; with a new thread_id it must not.
"""

from __future__ import annotations

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver

from ai_travel_assistant.tool_calling import (
    ask_agent,
    build_tool_agent_graph,
    final_answer,
)


class RecordingModel(GenericFakeChatModel):
    """Returns scripted replies and records the messages of every call."""

    calls: list = []

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def recording(*replies: str) -> RecordingModel:
    return RecordingModel(messages=iter(AIMessage(r) for r in replies), calls=[])


def texts(messages) -> list[str]:
    return [m.content for m in messages if not isinstance(m, SystemMessage)]


def test_follow_up_on_same_thread_sees_earlier_turn():
    model = recording("Day 1: Temple. Day 2: Hike.", "Day 2 is now a garden walk.")
    graph = build_tool_agent_graph(model, checkpointer=InMemorySaver())

    ask_agent(graph, "t1", "Plan 2 days in a hill town.")
    turn = ask_agent(graph, "t1", "Make day 2 easier.")

    assert texts(model.calls[1]) == [
        "Plan 2 days in a hill town.",
        "Day 1: Temple. Day 2: Hike.",
        "Make day 2 easier.",
    ]
    assert final_answer(turn) == "Day 2 is now a garden walk."


def test_ask_agent_returns_only_this_turns_messages():
    model = recording("First answer.", "Second answer.")
    graph = build_tool_agent_graph(model, checkpointer=InMemorySaver())

    ask_agent(graph, "t1", "Question one")
    turn = ask_agent(graph, "t1", "Question two")

    assert isinstance(turn[0], HumanMessage)
    assert texts(turn) == ["Question two", "Second answer."]


def test_new_thread_starts_empty():
    model = recording("Answer A.", "Answer B.")
    graph = build_tool_agent_graph(model, checkpointer=InMemorySaver())

    ask_agent(graph, "thread-a", "Plan Galle.")
    ask_agent(graph, "thread-b", "Plan Kandy.")

    assert texts(model.calls[1]) == ["Plan Kandy."]


def test_system_prompt_is_not_saved_in_the_thread():
    model = recording("One.", "Two.")
    graph = build_tool_agent_graph(model, checkpointer=InMemorySaver())

    ask_agent(graph, "t1", "Hi")
    ask_agent(graph, "t1", "Again")

    saved = graph.get_state({"configurable": {"thread_id": "t1"}}).values["messages"]
    assert not any(isinstance(m, SystemMessage) for m in saved)
    # The prompt is still sent once per call, at the front.
    assert sum(isinstance(m, SystemMessage) for m in model.calls[1]) == 1
