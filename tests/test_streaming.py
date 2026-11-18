"""Stage 10: streaming a turn as UI events (offline, no API key).

graph.stream(stream_mode="messages") makes LangChain call the model in
STREAMING mode, so the fake must implement _stream. StreamingScripted plays
the scripted replies word by word, with tool calls in the last chunk, which is
how Gemini streams them too. Other Stage 10 tests import it from here.
"""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGenerationChunk
from langgraph.checkpoint.memory import InMemorySaver

from ai_travel_assistant import tool_calling
from ai_travel_assistant.streaming import (
    FAILED_STEP,
    NODE_STEPS,
    Step,
    Token,
    TurnDone,
    steps_from_update,
    stream_turn,
    tool_step,
    visible_history,
)
from ai_travel_assistant.tool_calling import ask_agent, build_tool_agent_graph
from tests.test_clarification import FakeUnderstander, understood
from tests.test_itinerary_graph import (
    TODAY,
    Scripted,
    gather,
    plan,
    submit,
    tools,  # noqa: F401 - pytest fixture
)


class StreamingScripted(Scripted):
    """The scripted fake, streamed: words as chunks, tool calls at the end."""

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        message = self._generate(messages, stop=stop, **kwargs).generations[0].message
        words = message.content.split(" ") if message.content else [""]
        for index, word in enumerate(words):
            text = word if index == 0 else " " + word
            calls = []
            if index == len(words) - 1:
                calls = [
                    {
                        "name": c["name"],
                        "args": json.dumps(c["args"]),
                        "id": c["id"],
                        "index": i,
                    }
                    for i, c in enumerate(message.tool_calls)
                ]
            chunk = AIMessageChunk(content=text, id=message.id, tool_call_chunks=calls)
            if run_manager and text:
                run_manager.on_llm_new_token(
                    text, chunk=ChatGenerationChunk(message=chunk)
                )
            yield ChatGenerationChunk(message=chunk)


def streaming(*replies, fail_on=None) -> StreamingScripted:
    messages = [r if isinstance(r, AIMessage) else AIMessage(r) for r in replies]
    return StreamingScripted(messages=iter(messages), calls=[], fail_on=fail_on or {})


ANSWERS = {
    "Plan 2 days in Ella.": understood("plan", destination="Ella", days=2),
    "What is Ella like?": understood("question"),
    "Plan me a trip.": understood("plan"),
    "Ella, 2 days.": understood("plan", destination="Ella", days=2),
    "Tell me a joke.": understood("off_topic"),
}


def build(model, understander=True):
    return build_tool_agent_graph(
        model,
        checkpointer=InMemorySaver(),
        understander=FakeUnderstander(ANSWERS) if understander else None,
        structured_itinerary=True,
        today=lambda: TODAY,
    )


def events(graph, text, thread="t1"):
    return list(stream_turn(graph, thread, text))


# --- Whole turns ----------------------------------------------------------------


@pytest.mark.usefixtures("tools")
def test_an_itinerary_turn_streams_progress_then_the_checked_plan():
    out = events(build(streaming(gather(), submit(plan()))), "Plan 2 days in Ella.")
    steps = [e.text for e in out if isinstance(e, Step)]
    assert steps == [
        NODE_STEPS["understand_request"],
        "📖 Reading the travel guide for Ella",
        "📍 Finding sights near Ella",
        "🔎 Checking the weather in Ella",
        "🚗 Estimating the drive from Colombo to Ella",
        "📝 Writing your itinerary",
        NODE_STEPS["compose_answer"],
    ]
    assert not any(isinstance(e, Token) for e in out)  # rendered, not typed
    done = out[-1]
    assert isinstance(done, TurnDone) and done.kind == "itinerary"
    assert done.answer.startswith("# 2-day trip to Ella")
    assert done.itinerary["days"] == 2


def test_a_prose_answer_streams_token_by_token():
    out = events(build(streaming("Ella is green and calm.")), "What is Ella like?")
    tokens = [e.text for e in out if isinstance(e, Token)]
    assert tokens == ["Ella", " is", " green", " and", " calm."]
    done = out[-1]
    assert (done.kind, done.answer) == ("answer", "Ella is green and calm.")
    assert done.itinerary is None


def test_the_understanders_output_is_never_streamed_as_tokens():
    # The understander here is a fake, but a real one's JSON comes from node
    # understand_request, which is not in TOKEN_NODES.
    out = events(build(streaming("Hi!")), "What is Ella like?")
    assert "".join(e.text for e in out if isinstance(e, Token)) == "Hi!"


@pytest.mark.usefixtures("tools")
def test_a_question_pauses_the_stream_and_the_reply_resumes_it():
    graph = build(streaming(gather(), submit(plan())))
    first = events(graph, "Plan me a trip.")
    done = first[-1]
    assert done.kind == "clarification"
    assert done.question and done.question == done.answer

    second = events(graph, "Ella, 2 days.")
    assert second[-1].kind == "itinerary"


def test_off_topic_gets_the_fixed_reply():
    model = streaming()
    done = events(build(model), "Tell me a joke.")[-1]
    assert done.kind == "off_topic"
    assert model.calls == []


@pytest.mark.usefixtures("tools")
def test_streaming_leaves_the_same_state_as_ask_agent():
    streamed, plain = (
        build(streaming(gather(), submit(plan()))),
        build(streaming(gather(), submit(plan()))),
    )
    events(streamed, "Plan 2 days in Ella.", thread="a")
    ask_agent(plain, "a", "Plan 2 days in Ella.")
    config = {"configurable": {"thread_id": "a"}}
    one, two = streamed.get_state(config).values, plain.get_state(config).values
    assert one["itinerary"] == two["itinerary"]
    assert [type(m) for m in one["messages"]] == [type(m) for m in two["messages"]]


def test_errors_propagate_to_the_caller():
    graph = build(streaming(fail_on={0: RuntimeError("boom")}), understander=False)
    with pytest.raises(RuntimeError, match="boom"):
        events(graph, "Hello")


def test_without_an_understander_there_is_no_understood_step():
    out = events(build(streaming("Hello!"), understander=False), "Hi")
    assert [e for e in out if isinstance(e, Step)] == []
    assert out[-1].answer == "Hello!"


# --- Pieces ---------------------------------------------------------------------


def test_tool_steps_fill_in_the_arguments_and_survive_missing_ones():
    assert (
        tool_step(
            "estimate_travel_time", {"origin": "Kandy", "destination": "Ella"}
        ).text
        == "🚗 Estimating the drive from Kandy to Ella"
    )
    assert tool_step("get_weather_forecast", {}).text == "🔎 Checking the weather in ?"
    assert tool_step("brand_new_tool", {}).text == "🔧 Running brand_new_tool"


def test_failed_tool_results_become_a_warning_step():
    update = {
        "messages": [
            ToolMessage(
                "Weather unavailable for 'Ella': down",
                tool_call_id="1",
                name="get_weather_forecast",
            ),
            ToolMessage(
                "Attractions near Ella: ...", tool_call_id="2", name="find_attractions"
            ),
            ToolMessage(
                "boom", tool_call_id="3", name="convert_currency", status="error"
            ),
        ]
    }
    assert steps_from_update("tools", update) == [
        Step(FAILED_STEP.format(tool="get_weather_forecast")),
        Step(FAILED_STEP.format(tool="convert_currency")),
    ]


def test_nodes_without_a_step_and_odd_updates_are_quiet():
    assert steps_from_update("save_memory", {"messages": []}) == []
    assert steps_from_update("llm", None) == []
    assert steps_from_update("load_memory", {"user_prefs": ["vegetarian"]}) == []


def test_visible_history_shows_only_what_a_chat_window_should():
    messages = [
        HumanMessage("Plan 2 days in Ella."),
        AIMessage("", tool_calls=[{"name": "find_attractions", "args": {}, "id": "1"}]),
        ToolMessage("Attractions near Ella", tool_call_id="1"),
        AIMessage("# 2-day trip to Ella", name="itinerary"),
        HumanMessage("Thanks"),
        AIMessage([{"type": "text", "text": "You're welcome!"}]),
        AIMessage("   "),
    ]
    assert visible_history(messages) == [
        ("user", "Plan 2 days in Ella."),
        ("assistant", "# 2-day trip to Ella"),
        ("user", "Thanks"),
        ("assistant", "You're welcome!"),
    ]


def test_tool_steps_cover_every_agent_tool():
    names = {t.name for t in tool_calling.TOOLS}
    from ai_travel_assistant.streaming import TOOL_STEPS

    assert names <= set(TOOL_STEPS)
