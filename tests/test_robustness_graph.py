"""Stage 7 checks: failures and recovery inside the agent graph, all offline.

Scripted fake models play Gemini (and can raise like Gemini does); tool
functions are replaced with stand-ins that succeed or raise. No API calls.
"""

from __future__ import annotations

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
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


# --- Repairing the history after a failure ------------------------------------


def test_a_gemini_failure_after_the_tools_leaves_a_valid_next_turn(tools_ok):
    from google.genai import errors as genai_errors

    busy = genai_errors.ServerError(503, {"error": {"code": 503, "message": "x"}})
    model = scripted(tool_step("get_weather_forecast"), "Hello!", fail_on={1: busy})
    graph = build_tool_agent_graph(model, checkpointer=InMemorySaver())

    with pytest.raises(genai_errors.ServerError):
        ask_agent(graph, "t1", "Weather?")
    turn = ask_agent(graph, "t1", "Hello again")

    assert final_answer(turn) == "Hello!"
    from ai_travel_assistant.robustness import NO_ANSWER_NOTE

    seen = model.calls[-1]
    assert kinds(seen) == ["HumanMessage", "AIMessage", "HumanMessage"]
    assert seen[2].content == NO_ANSWER_NOTE


def test_an_orphaned_tool_call_in_the_thread_is_closed_for_the_model():
    from ai_travel_assistant.robustness import NOT_RUN_RESULT

    model = scripted("Hello!")
    # Trimming off, so earlier turns are sent whole, orphan included.
    graph = build_tool_agent_graph(
        model, checkpointer=InMemorySaver(), max_history_tokens=None
    )
    config = {"configurable": {"thread_id": "t1"}}
    # A thread left by a crash between "call the tool" and "tool answered".
    graph.update_state(
        config,
        {"messages": [HumanMessage("Weather?"), tool_step("get_weather_forecast")]},
    )

    ask_agent(graph, "t1", "Hello again")

    seen = model.calls[0]
    assert kinds(seen) == ["HumanMessage", "AIMessage", "ToolMessage", "HumanMessage"]
    assert seen[3].content == NOT_RUN_RESULT
    # Only the model's view was repaired; the saved thread is unchanged.
    assert kinds(saved(graph)) == [
        "HumanMessage",
        "AIMessage",
        "HumanMessage",
        "AIMessage",
    ]


# --- The tool-round cap ---------------------------------------------------------


def endless_tools(rounds: int, wrap_up="Partial plan from what I found.", **kw):
    """A model that asks for weather `rounds` times, then gives `wrap_up`."""
    steps = [tool_step("get_weather_forecast", prefix=f"r{i}-") for i in range(rounds)]
    return scripted(*steps, wrap_up, **kw)


def test_a_model_that_keeps_calling_tools_is_stopped_at_the_cap(tools_ok):
    from ai_travel_assistant.robustness import (
        LOOP_CAP_RESULT,
        MAX_TOOL_ROUNDS,
        WRAP_UP_INSTRUCTION,
    )

    model = endless_tools(MAX_TOOL_ROUNDS + 1)
    graph = build_tool_agent_graph(model, checkpointer=InMemorySaver())

    turn = ask_agent(graph, "t1", "Weather?")

    assert final_answer(turn) == "Partial plan from what I found."
    # 4 tool rounds + the 5th request + the wrap-up = 6 model calls.
    assert len(model.calls) == MAX_TOOL_ROUNDS + 2
    real = [m for m in turn if isinstance(m, ToolMessage) and m.content == "Forecast."]
    assert len(real) == MAX_TOOL_ROUNDS
    # The stopped call has a result, so nothing is orphaned.
    assert turn[-2].content == LOOP_CAP_RESULT
    assert kinds(turn)[-3:] == ["AIMessage", "ToolMessage", "AIMessage"]
    assert WRAP_UP_INSTRUCTION in model.calls[-1][0].content


def test_a_failed_wrap_up_gives_the_fixed_reply(tools_ok):
    from ai_travel_assistant.robustness import FALLBACK_REPLY, MAX_TOOL_ROUNDS

    model = endless_tools(
        MAX_TOOL_ROUNDS + 1, fail_on={MAX_TOOL_ROUNDS + 1: RuntimeError("503")}
    )
    graph = build_tool_agent_graph(model, checkpointer=InMemorySaver())

    turn = ask_agent(graph, "t1", "Weather?")

    assert final_answer(turn) == FALLBACK_REPLY


def test_a_wrap_up_that_asks_for_tools_again_gives_the_fixed_reply(tools_ok):
    from ai_travel_assistant.robustness import FALLBACK_REPLY, MAX_TOOL_ROUNDS

    model = endless_tools(MAX_TOOL_ROUNDS + 2)  # the wrap-up is one more tool call
    graph = build_tool_agent_graph(model, checkpointer=InMemorySaver())

    turn = ask_agent(graph, "t1", "Weather?")

    assert final_answer(turn) == FALLBACK_REPLY
    assert not turn[-1].tool_calls


def test_the_cap_counts_each_turn_separately(tools_ok):
    model = scripted(
        *(tool_step("get_weather_forecast", prefix=f"a{i}-") for i in range(3)),
        "First answer.",
        *(tool_step("get_weather_forecast", prefix=f"b{i}-") for i in range(3)),
        "Second answer.",
    )
    graph = build_tool_agent_graph(model, checkpointer=InMemorySaver())

    ask_agent(graph, "t1", "One")
    turn = ask_agent(graph, "t1", "Two")

    assert final_answer(turn) == "Second answer."  # 3 rounds each: under the cap


def test_the_cap_is_configurable(tools_ok):
    from ai_travel_assistant.robustness import LOOP_CAP_RESULT

    model = endless_tools(2, wrap_up="Done early.")
    graph = build_tool_agent_graph(
        model, checkpointer=InMemorySaver(), max_tool_rounds=1
    )

    turn = ask_agent(graph, "t1", "Weather?")

    assert final_answer(turn) == "Done early."
    assert turn[-2].content == LOOP_CAP_RESULT


def test_preferences_are_still_saved_after_the_fallback(tools_ok):
    from langgraph.store.memory import InMemoryStore

    from ai_travel_assistant.long_term_memory import load_preferences
    from ai_travel_assistant.robustness import MAX_TOOL_ROUNDS

    store = InMemoryStore()
    graph = build_tool_agent_graph(
        endless_tools(MAX_TOOL_ROUNDS + 1),
        checkpointer=InMemorySaver(),
        store=store,
        preference_extractor=lambda text, known: ["vegetarian"],
    )

    ask_agent(graph, "t1", "I'm vegetarian, what is the weather?")

    assert load_preferences(store, "local") == ["vegetarian"]


# --- RetryPolicy on the llm node -------------------------------------------------


def gemini_error(code: int, message: str = "Try later."):
    from google.genai import errors as genai_errors

    cls = genai_errors.ServerError if code >= 500 else genai_errors.ClientError
    return cls(code, {"error": {"code": code, "message": message}})


def retrying_graph(model, **kwargs):
    from ai_travel_assistant.robustness import llm_retry_policy

    return build_tool_agent_graph(
        model,
        checkpointer=InMemorySaver(),
        llm_retry=llm_retry_policy(initial_interval=0),  # no real sleeping
        **kwargs,
    )


def test_one_transient_failure_is_retried_once():
    model = scripted("Answer.", fail_on={0: gemini_error(503)})
    turn = ask_agent(retrying_graph(model), "t1", "Hi")
    assert final_answer(turn) == "Answer."
    assert len(model.calls) == 2


def test_two_transient_failures_fail_the_turn_after_two_attempts():
    model = scripted("Never.", fail_on={0: gemini_error(504), 1: gemini_error(503)})
    with pytest.raises(Exception, match="503"):
        ask_agent(retrying_graph(model), "t1", "Hi")
    assert len(model.calls) == 2


@pytest.mark.parametrize("code", [429, 400, 401])
def test_rate_limits_and_bad_requests_are_never_retried(code):
    model = scripted("Never.", fail_on={0: gemini_error(code)})
    with pytest.raises(Exception, match=str(code)):
        ask_agent(retrying_graph(model), "t1", "Hi")
    assert len(model.calls) == 1


def test_a_retry_does_not_run_the_tools_again(monkeypatch):
    runs = []
    monkeypatch.setattr(
        tool_calling.get_weather_forecast,
        "func",
        lambda **_: runs.append(1) or "Forecast.",
    )
    model = scripted(
        tool_step("get_weather_forecast"), "Sunny.", fail_on={1: gemini_error(503)}
    )

    turn = ask_agent(retrying_graph(model), "t1", "Weather?")

    assert final_answer(turn) == "Sunny."
    assert len(runs) == 1
    assert len(model.calls) == 3


def test_without_a_retry_policy_a_failure_reaches_the_caller():
    model = scripted("Answer.", fail_on={0: gemini_error(503)})
    graph = build_tool_agent_graph(model, checkpointer=InMemorySaver())
    with pytest.raises(Exception, match="503"):
        ask_agent(graph, "t1", "Hi")
    assert len(model.calls) == 1


# --- Stage 6 understanding under failures ---------------------------------------


DAILY_QUOTA_TEXT = "Quota exceeded. quotaId: GenerateRequestsPerDayPerProjectPerModel"


def understanding_graph(model, understander, **kwargs):
    return build_tool_agent_graph(
        model, checkpointer=InMemorySaver(), understander=understander, **kwargs
    )


def test_daily_quota_in_the_understander_ends_the_turn_without_an_agent_call():
    def out_of_quota(text, **context):
        raise gemini_error(429, DAILY_QUOTA_TEXT)

    model = scripted("Never.")
    graph = understanding_graph(model, out_of_quota)

    with pytest.raises(Exception, match="429"):
        ask_agent(graph, "t1", "Plan me a trip")
    assert model.calls == []


@pytest.mark.parametrize(
    "error",
    [lambda: gemini_error(503), lambda: gemini_error(429, "Per minute limit")],
    ids=["503", "per-minute-429"],
)
def test_other_understander_errors_still_fall_back_to_the_agent(error):
    def failing(text, **context):
        raise error()

    model = scripted("Answer anyway.")
    turn = ask_agent(understanding_graph(model, failing), "t1", "Plan me a trip")

    assert final_answer(turn) == "Answer anyway."
    assert len(model.calls) == 1


def plan_understander(text, **context):
    from ai_travel_assistant.schemas import RequestUnderstanding, TripRequest

    answers = {
        "Plan me a trip": RequestUnderstanding(intent="plan"),
        "Galle, 2 days": RequestUnderstanding(
            intent="plan", trip=TripRequest(destination="Galle", days=2)
        ),
        "Hello again": RequestUnderstanding(intent="question"),
    }
    return answers[text]


def test_an_agent_failure_after_a_resume_leaves_the_thread_usable():
    from ai_travel_assistant.tool_calling import pending_question

    model = scripted("Hello!", fail_on={0: gemini_error(503)})
    graph = understanding_graph(model, plan_understander)

    ask_agent(graph, "t1", "Plan me a trip")  # pauses with a question
    with pytest.raises(Exception, match="503"):
        ask_agent(graph, "t1", "Galle, 2 days")  # resumes, then the agent fails

    assert pending_question(graph, "t1") is None  # not stuck on the question
    turn = ask_agent(graph, "t1", "Hello again")
    assert final_answer(turn) == "Hello!"
    # The trip from the answered question was kept for the next turn.
    state = graph.get_state({"configurable": {"thread_id": "t1"}}).values
    assert state["trip_request"]["destination"] == "Galle"


def test_a_retry_after_a_resume_does_not_ask_or_understand_again():
    from ai_travel_assistant.robustness import llm_retry_policy

    calls = []

    def counting(text, **context):
        calls.append(text)
        return plan_understander(text, **context)

    model = scripted("Day 1 in Galle ...", fail_on={0: gemini_error(503)})
    graph = understanding_graph(
        model, counting, llm_retry=llm_retry_policy(initial_interval=0)
    )

    ask_agent(graph, "t1", "Plan me a trip")
    turn = ask_agent(graph, "t1", "Galle, 2 days")

    assert final_answer(turn) == "Day 1 in Galle ..."
    assert calls == ["Plan me a trip", "Galle, 2 days"]
    assert len(model.calls) == 2


def test_a_failure_on_one_thread_leaves_a_paused_thread_alone():
    from ai_travel_assistant.tool_calling import pending_question

    model = scripted("Never.", fail_on={0: gemini_error(503)})
    graph = understanding_graph(model, plan_understander)

    ask_agent(graph, "paused", "Plan me a trip")
    with pytest.raises(Exception, match="503"):
        ask_agent(graph, "other", "Galle, 2 days")

    assert pending_question(graph, "paused") is not None
