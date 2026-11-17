"""Stage 5 checks: short-term memory (checkpointer + thread_id) for the agent.

Offline only: a scripted fake model plays the LLM, so no Gemini calls. We check
what the model RECEIVES on a follow-up turn: with the same thread_id it must see
the earlier turn; with a new thread_id it must not.
"""

from __future__ import annotations

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver

from ai_travel_assistant import tool_calling
from ai_travel_assistant.tool_calling import (
    ask_agent,
    build_tool_agent_graph,
    final_answer,
    run_agent_chat,
    saved_message_count,
    sqlite_checkpointer,
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


# --- The terminal chat loop ---------------------------------------------------


def fake_terminal(*lines: str):
    """read() returns the given lines in order; write() records the output."""
    replies = iter(lines)
    shown: list[str] = []
    return (lambda prompt: next(replies)), shown.append, shown


def test_chat_keeps_context_between_messages():
    model = recording("Plan: two days.", "Day 2 made easier.")
    graph = build_tool_agent_graph(model, checkpointer=InMemorySaver())
    read, write, shown = fake_terminal("Plan 2 days.", "Make day 2 easier.", "/quit")

    run_agent_chat(graph, read, write)

    assert texts(model.calls[1])[:2] == ["Plan 2 days.", "Plan: two days."]
    assert shown[-3:-1] == ["AI: Plan: two days.", "AI: Day 2 made easier."]
    assert shown[-1].startswith("Bye! Reopen this conversation with --thread ")


def test_new_command_starts_a_fresh_thread():
    model = recording("Answer one.", "Answer two.")
    graph = build_tool_agent_graph(model, checkpointer=InMemorySaver())
    read, write, shown = fake_terminal("First trip", "/new", "Second trip", "/exit")

    run_agent_chat(graph, read, write)

    assert texts(model.calls[1]) == ["Second trip"]
    assert sum(line.startswith("New conversation: ") for line in shown) == 2


def test_blank_lines_are_skipped_without_calling_the_model():
    model = recording("Only answer.")
    graph = build_tool_agent_graph(model, checkpointer=InMemorySaver())
    read, write, _ = fake_terminal("", "   ", "Hello", "/quit")

    run_agent_chat(graph, read, write)

    assert len(model.calls) == 1


def test_show_steps_prints_the_turns_steps():
    model = recording("Hi there.")
    graph = build_tool_agent_graph(model, checkpointer=InMemorySaver())
    read, write, shown = fake_terminal("Hello", "/quit")

    run_agent_chat(graph, read, write, show_steps=True)

    assert "[human] Hello\n[llm -> final answer]" in shown


class FailingOnceModel(RecordingModel):
    """Raises on the first call (like a Gemini 503), then answers normally."""

    failed: bool = False

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if not self.failed:
            self.failed = True
            raise RuntimeError("503 UNAVAILABLE")
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def test_failed_turn_is_reported_and_the_chat_goes_on():
    model = FailingOnceModel(messages=iter([AIMessage("Works now.")]), calls=[])
    graph = build_tool_agent_graph(model, checkpointer=InMemorySaver())
    read, write, shown = fake_terminal("Try 1", "Try 2", "/quit")

    run_agent_chat(graph, read, write, describe_error=str)

    assert "Request failed: 503 UNAVAILABLE" in shown
    assert "AI: Works now." in shown


def test_cli_needs_a_question_or_chat():
    with pytest.raises(SystemExit):
        tool_calling.main([])
    with pytest.raises(SystemExit):
        tool_calling.main(["Where to hike?", "--chat"])


# --- SQLite: conversations survive a restart ----------------------------------


def test_sqlite_thread_survives_a_new_graph_and_connection(tmp_path):
    db = tmp_path / "chat.sqlite"
    with sqlite_checkpointer(db) as saver:
        first = build_tool_agent_graph(recording("Plan saved."), checkpointer=saver)
        ask_agent(first, "trip-1", "Plan Galle.")

    # A "restart": a new connection and a new graph on the same file.
    model = recording("Day 2 is lighter.")
    with sqlite_checkpointer(db) as saver:
        second = build_tool_agent_graph(model, checkpointer=saver)
        ask_agent(second, "trip-1", "Make day 2 lighter.")

    assert texts(model.calls[0]) == [
        "Plan Galle.",
        "Plan saved.",
        "Make day 2 lighter.",
    ]


def test_sqlite_checkpointer_creates_the_folder(tmp_path):
    db = tmp_path / "nested" / "chat.sqlite"
    with sqlite_checkpointer(db) as saver:
        graph = build_tool_agent_graph(recording("Hi."), checkpointer=saver)
        ask_agent(graph, "t1", "Hello")
    assert db.exists()


def test_saved_message_count(tmp_path):
    with sqlite_checkpointer(tmp_path / "chat.sqlite") as saver:
        graph = build_tool_agent_graph(recording("Hi."), checkpointer=saver)
        assert saved_message_count(graph, "t1") == 0
        ask_agent(graph, "t1", "Hello")
        assert saved_message_count(graph, "t1") == 2


def test_chat_continues_a_saved_thread(tmp_path):
    db = tmp_path / "chat.sqlite"
    with sqlite_checkpointer(db) as saver:
        graph = build_tool_agent_graph(recording("Plan saved."), checkpointer=saver)
        ask_agent(graph, "trip-1", "Plan Galle.")

    model = recording("Updated plan.")
    read, write, shown = fake_terminal("Add a beach day.", "/quit")
    with sqlite_checkpointer(db) as saver:
        graph = build_tool_agent_graph(model, checkpointer=saver)
        run_agent_chat(graph, read, write, thread_id="trip-1")

    assert shown[0] == "Continuing conversation trip-1 (2 saved messages)."
    assert texts(model.calls[0])[:2] == ["Plan Galle.", "Plan saved."]
    assert shown[-1] == "Bye! Reopen this conversation with --thread trip-1"


def test_chat_with_an_unknown_thread_starts_it(tmp_path):
    read, write, shown = fake_terminal("/quit")
    with sqlite_checkpointer(tmp_path / "chat.sqlite") as saver:
        graph = build_tool_agent_graph(recording(), checkpointer=saver)
        run_agent_chat(graph, read, write, thread_id="nope")
    assert shown[0] == "No saved messages for nope; starting it now."


def test_cli_thread_needs_chat():
    with pytest.raises(SystemExit):
        tool_calling.main(["Where to hike?", "--thread", "abc"])
