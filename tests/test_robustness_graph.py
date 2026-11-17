"""Stage 7 checks: failures and recovery inside the agent graph, all offline.

Scripted fake models play Gemini (and can raise like Gemini does); tool
functions are replaced with stand-ins that succeed or raise. No API calls.
"""

from __future__ import annotations

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from ai_travel_assistant import tool_calling
from ai_travel_assistant.tool_calling import (
    ask_agent,
    build_tool_agent_graph,
    final_answer,
)


class ScriptedModel(GenericFakeChatModel):
    """Returns scripted replies, records every call, and raises on chosen calls.

    fail_on maps a call number (0-based) to the exception that call raises.
    """

    calls: list = []
    fail_on: dict = {}

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        number = len(self.calls)
        self.calls.append(list(messages))
        if number in self.fail_on:
            raise self.fail_on[number]
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def scripted(*replies, fail_on=None) -> ScriptedModel:
    messages = [r if isinstance(r, AIMessage) else AIMessage(r) for r in replies]
    return ScriptedModel(messages=iter(messages), calls=[], fail_on=fail_on or {})


def tool_step(*names: str, prefix: str = "c") -> AIMessage:
    return AIMessage(
        "",
        tool_calls=[
            {"name": name, "args": {"destination": "Ella"}, "id": f"{prefix}{i}"}
            for i, name in enumerate(names)
        ],
    )


def saved(graph, thread_id="t1") -> list:
    return graph.get_state({"configurable": {"thread_id": thread_id}}).values[
        "messages"
    ]


def kinds(messages) -> list[str]:
    return [type(m).__name__ for m in messages if not isinstance(m, SystemMessage)]


@pytest.fixture
def tools_ok(monkeypatch):
    """Weather and attractions answer with fixed text (no network)."""
    monkeypatch.setattr(
        tool_calling.get_weather_forecast, "func", lambda **_: "Forecast."
    )
    monkeypatch.setattr(tool_calling.find_attractions, "func", lambda **_: "Sights.")


# --- Tool errors -----------------------------------------------------------------


def test_a_failing_tool_keeps_the_other_parallel_results(tools_ok, monkeypatch):
    def broken(**_):
        raise KeyError("daily")

    monkeypatch.setattr(tool_calling.get_weather_forecast, "func", broken)
    model = scripted(tool_step("get_weather_forecast", "find_attractions"), "Plan.")
    graph = build_tool_agent_graph(model, checkpointer=InMemorySaver())

    turn = ask_agent(graph, "t1", "Plan Ella")

    results = {m.name: m for m in turn if isinstance(m, ToolMessage)}
    assert results["find_attractions"].content == "Sights."
    assert results["find_attractions"].status == "success"
    assert results["get_weather_forecast"].status == "error"
    assert "KeyError" in results["get_weather_forecast"].content
    assert final_answer(turn) == "Plan."
    # The model saw both results before answering.
    assert sum(isinstance(m, ToolMessage) for m in model.calls[1]) == 2


def test_a_failing_tool_leaves_no_orphaned_call(monkeypatch):
    def broken(**_):
        raise ValueError("formatter bug")

    monkeypatch.setattr(tool_calling.get_weather_forecast, "func", broken)
    graph = build_tool_agent_graph(
        scripted(tool_step("get_weather_forecast"), "No weather today."),
        checkpointer=InMemorySaver(),
    )

    ask_agent(graph, "t1", "Weather?")

    assert kinds(saved(graph)) == [
        "HumanMessage",
        "AIMessage",
        "ToolMessage",
        "AIMessage",
    ]
