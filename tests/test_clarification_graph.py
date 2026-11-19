"""Stage 6 checks: interrupt and resume in the agent graph, all offline.

A dictionary-lookup understander plays the understanding LLM and a scripted
fake model plays the agent. We check where the run pauses, what is saved while
it waits, and that resuming continues the same run with the right state.
"""

from __future__ import annotations

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from ai_travel_assistant import tool_calling
from ai_travel_assistant.long_term_memory import load_preferences
from ai_travel_assistant.request_understanding import (
    CLARIFICATION_NAME,
    MAX_CLARIFICATIONS,
    OFF_TOPIC_REPLY,
    UNDERSTAND_REQUEST,
)
from ai_travel_assistant.schemas import PartyMember, RequestUnderstanding, TripRequest
from ai_travel_assistant.tool_calling import (
    ask_agent,
    build_tool_agent_graph,
    describe_steps,
    final_answer,
    pending_question,
    run_agent_chat,
    sqlite_checkpointer,
)

ASK_PLACE = "Where in Sri Lanka would you like to go?"
ASK_GROUP = "How many people are travelling, including you?"
ASK_CITIZENSHIP = "What citizenship do the travellers have?"
# Replies that give every detail the agent asks for, so the plan goes ahead.
GALLE = "Galle, 2 days, from Colombo; two British adults"
ELLA = "Plan 3 days in Ella from Kandy, just me; I'm Sri Lankan"


class RecordingModel(GenericFakeChatModel):
    """The agent: returns scripted replies and records every call."""

    calls: list = []

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def agent(*replies: str) -> RecordingModel:
    return RecordingModel(messages=iter(AIMessage(r) for r in replies), calls=[])


def understood(intent, **trip) -> RequestUnderstanding:
    return RequestUnderstanding(intent=intent, trip=TripRequest(**trip))


class FakeUnderstander:
    """Looks the message up in a table and records every call."""

    def __init__(self, **answers: RequestUnderstanding):
        self.answers = {k.replace("_", " "): v for k, v in answers.items()}
        self.calls: list[tuple[str, dict]] = []

    def add(self, text: str, answer: RequestUnderstanding) -> FakeUnderstander:
        self.answers[text] = answer
        return self

    def __call__(self, text, **context):
        self.calls.append((text, context))
        return self.answers[text]


def standard_understander() -> FakeUnderstander:
    return (
        FakeUnderstander()
        .add("Plan me a trip", understood("plan"))
        .add(
            GALLE,
            understood(
                "plan",
                destination="Galle",
                days=2,
                origin="Colombo",
                wants_hotels=False,
                travellers=2,
                party=[PartyMember(nationality="British", count=2)],
            ),
        )
        .add("Galle", understood("plan", destination="Galle"))
        .add("3", understood("question", days=3))
        .add("no idea", understood("question"))
        .add(
            ELLA,
            understood(
                "plan",
                destination="Ella",
                days=3,
                origin="Kandy",
                wants_hotels=False,
                travellers=1,
                nationality="Sri Lankan",
            ),
        )
        .add("Where can I surf?", understood("question"))
        .add("Write my Python homework", understood("off_topic"))
        .add("Make day 2 easier", understood("modify"))
    )


def build(model, understander=None, checkpointer=None, **kwargs):
    return build_tool_agent_graph(
        model,
        checkpointer=checkpointer or InMemorySaver(),
        understander=understander or standard_understander(),
        **kwargs,
    )


def saved(graph, thread_id="t1") -> dict:
    return graph.get_state({"configurable": {"thread_id": thread_id}}).values


def system_prompt(call) -> str:
    return next(m.content for m in call if isinstance(m, SystemMessage))


# --- Backward compatibility ---------------------------------------------------


def test_without_an_understander_the_graph_is_unchanged():
    graph = build_tool_agent_graph(agent("Hi."), checkpointer=InMemorySaver())
    assert UNDERSTAND_REQUEST not in graph.get_graph().nodes


def test_an_understander_needs_a_checkpointer():
    with pytest.raises(ValueError):
        build_tool_agent_graph(agent(), understander=standard_understander())


# --- Pausing ------------------------------------------------------------------


def test_a_vague_plan_pauses_with_a_question_and_no_agent_call():
    model = agent()
    graph = build(model)

    turn = ask_agent(graph, "t1", "Plan me a trip")

    assert final_answer(turn) == ASK_PLACE
    assert turn[-1].name == CLARIFICATION_NAME
    assert pending_question(graph, "t1") == ASK_PLACE
    assert model.calls == []
    # The question is already in the saved history while the run waits.
    assert [m.content for m in saved(graph)["messages"]] == [
        "Plan me a trip",
        ASK_PLACE,
    ]


def test_a_complete_plan_never_pauses():
    model = agent("Day 1 in Ella ...")
    graph = build(model)

    turn = ask_agent(graph, "t1", ELLA)

    assert pending_question(graph, "t1") is None
    assert final_answer(turn) == "Day 1 in Ella ..."
    assert "destination=Ella, days=3" in system_prompt(model.calls[0])


def test_a_question_without_a_place_goes_straight_to_the_agent():
    model = agent("Try Arugam Bay.")
    graph = build(model)

    ask_agent(graph, "t1", "Where can I surf?")

    assert pending_question(graph, "t1") is None
    assert len(model.calls) == 1


def test_off_topic_gets_a_fixed_reply_without_the_agent():
    model = agent()
    graph = build(model)

    turn = ask_agent(graph, "t1", "Write my Python homework")

    assert final_answer(turn) == OFF_TOPIC_REPLY
    assert model.calls == []
    assert pending_question(graph, "t1") is None


# --- Resuming -----------------------------------------------------------------


def test_the_reply_resumes_the_run_with_the_right_state():
    model = agent("Day 1 in Galle ...")
    understander = standard_understander()
    graph = build(model, understander)

    ask_agent(graph, "t1", "Plan me a trip")
    turn = ask_agent(graph, "t1", GALLE)

    # The turn is the reply and the answer: the run resumed, it did not restart.
    assert [m.content for m in turn] == [GALLE, "Day 1 in Galle ..."]
    assert understander.calls[1] == (
        GALLE,
        {"clarification_context": ASK_PLACE},
    )
    state = saved(graph)
    assert state["trip_request"]["destination"] == "Galle"
    assert state["trip_request"]["days"] == 2
    assert state["missing_fields"] == []
    assert state["clarification_context"] is None
    assert pending_question(graph, "t1") is None
    # The agent sees the whole exchange and the merged trip.
    assert "destination=Galle, days=2" in system_prompt(model.calls[0])
    assert [m.content for m in model.calls[0][1:]] == [
        "Plan me a trip",
        ASK_PLACE,
        GALLE,
    ]


def test_two_rounds_ask_only_for_what_is_still_missing():
    model = agent("Day 1 in Galle ...")
    graph = build(model)

    ask_agent(graph, "t1", "Plan me a trip")
    replies = [
        ask_agent(graph, "t1", text)
        for text in (
            "Galle",
            "3",
            "4 people",
            "All from a SAARC country",
            "1 child",
            "The airport (Katunayake)",
            "Yes, suggest hotels",
            "Mid-range",
        )
    ]

    questions = [final_answer(turn) for turn in replies]
    assert questions[0] == "How many days would you like to spend in Galle?"
    assert questions[1].startswith(ASK_GROUP)
    assert questions[2].startswith(ASK_CITIZENSHIP)
    assert questions[3].startswith("How many of the 4 travellers are children?")
    assert questions[4] == "Where will you start your trip to Galle from?"
    assert questions[5].startswith("Would you like hotel suggestions")
    assert "can't book" in questions[5]
    assert questions[6].startswith("What's your budget for a room per night?")
    assert questions[7] == "Day 1 in Galle ..."
    trip = saved(graph)["trip_request"]
    assert (trip["days"], trip["travellers"], trip["children"]) == (3, 4, 1)
    assert (trip["citizenship"], trip["origin"]) == ("saarc", "Katunayake")
    assert (trip["wants_hotels"], trip["budget"]) == (True, "mid-range")


def test_resuming_runs_no_node_twice():
    """Idempotency: one understander call per traveller message, one question each."""
    understander = standard_understander()
    graph = build(agent("Plan."), understander)

    ask_agent(graph, "t1", "Plan me a trip")
    ask_agent(graph, "t1", "Galle")
    ask_agent(graph, "t1", "3")
    ask_agent(graph, "t1", "Just me")  # one traveller: no children question
    ask_agent(graph, "t1", "All from other countries")
    ask_agent(graph, "t1", "Colombo")

    # Every reply is a quick reply ("Galle" and "Colombo" are buttons, "3" a
    # bare day count), so only the first message needs the understander.
    assert [text for text, _ in understander.calls] == ["Plan me a trip"]
    kinds = [type(m).__name__ for m in saved(graph)["messages"]]
    assert kinds == ["HumanMessage", "AIMessage"] * 6


def test_after_the_round_limit_the_agent_answers_with_assumptions():
    model = agent("Here are some ideas ...")
    graph = build(model)

    ask_agent(graph, "t1", "Plan me a trip")
    for _ in range(MAX_CLARIFICATIONS - 1):
        ask_agent(graph, "t1", "no idea")
    turn = ask_agent(graph, "t1", "no idea")

    assert final_answer(turn) == "Here are some ideas ..."
    assert pending_question(graph, "t1") is None
    assert "Still unknown after asking: destination, days" in system_prompt(
        model.calls[0]
    )


def test_a_new_message_after_the_limit_counts_rounds_again():
    graph = build(agent("Here are some ideas ..."))
    ask_agent(graph, "t1", "Plan me a trip")
    for _ in range(MAX_CLARIFICATIONS):
        ask_agent(graph, "t1", "no idea")  # limit reached, the agent answered

    turn = ask_agent(graph, "t1", "Plan me a trip")

    assert final_answer(turn) == ASK_PLACE
    assert pending_question(graph, "t1") == ASK_PLACE


def test_a_vague_request_after_a_plan_keeps_the_saved_trip():
    model = agent("Day 1 in Galle ...", "Another Galle plan ...")
    graph = build(model)
    ask_agent(graph, "t1", "Plan me a trip")
    ask_agent(graph, "t1", GALLE)

    turn = ask_agent(graph, "t1", "Plan me a trip")

    assert pending_question(graph, "t1") is None
    assert final_answer(turn) == "Another Galle plan ..."


def test_a_follow_up_edit_after_a_plan_is_not_questioned():
    model = agent("Day 1 in Galle ...", "Day 2 is easier now.")
    graph = build(model)
    ask_agent(graph, "t1", "Plan me a trip")
    ask_agent(graph, "t1", GALLE)

    turn = ask_agent(graph, "t1", "Make day 2 easier")

    assert final_answer(turn) == "Day 2 is easier now."
    assert saved(graph)["trip_request"]["destination"] == "Galle"


def test_a_failing_understander_lets_the_agent_answer():
    def broken(text, **context):
        raise RuntimeError("429 RESOURCE_EXHAUSTED")

    model = agent("Answer anyway.")
    graph = build(model, broken)

    turn = ask_agent(graph, "t1", "Plan me a trip")

    assert final_answer(turn) == "Answer anyway."
    assert pending_question(graph, "t1") is None


def test_paused_threads_are_isolated():
    model = agent("Day 1 in Ella ...")
    graph = build(model)

    ask_agent(graph, "a", "Plan me a trip")
    ask_agent(graph, "b", ELLA)

    assert pending_question(graph, "a") == ASK_PLACE
    assert pending_question(graph, "b") is None
    assert "Plan me a trip" not in [m.content for m in model.calls[0]]


def test_steps_show_the_clarification():
    graph = build(agent())
    turn = ask_agent(graph, "t1", "Plan me a trip")
    assert describe_steps(turn) == [
        "[human] Plan me a trip",
        "[clarify -> waiting for the traveller]",
    ]


# --- Persistence: a paused question survives a restart ------------------------


def test_a_paused_thread_resumes_after_a_restart(tmp_path):
    db = tmp_path / "chat.sqlite"
    with sqlite_checkpointer(db) as saver:
        first = build(agent(), checkpointer=saver)
        ask_agent(first, "trip-1", "Plan me a trip")

    # A "restart": a new connection and a new graph on the same file.
    model = agent("Day 1 in Galle ...")
    with sqlite_checkpointer(db) as saver:
        second = build(model, checkpointer=saver)
        assert pending_question(second, "trip-1") == ASK_PLACE
        turn = ask_agent(second, "trip-1", GALLE)
        state = saved(second, "trip-1")

    assert final_answer(turn) == "Day 1 in Galle ..."
    assert state["trip_request"]["destination"] == "Galle"


# --- Long-term memory still works around clarification ------------------------


def test_preferences_reach_the_agent_after_a_resume():
    store = InMemoryStore()
    store.put(("users", "local", "preferences"), "vegetarian", {"text": "vegetarian"})
    model = agent("Day 1 in Galle ...")
    graph = build(model, store=store)

    ask_agent(graph, "t1", "Plan me a trip")
    ask_agent(graph, "t1", GALLE)

    prompt = system_prompt(model.calls[0])
    assert "vegetarian" in prompt
    assert "destination=Galle" in prompt


def test_a_preference_said_as_a_reply_is_still_saved():
    store = InMemoryStore()
    understander = standard_understander().add("I'm vegetarian", understood("question"))
    graph = build(
        agent("Noted."),
        understander,
        store=store,
        preference_extractor=lambda text, known: ["vegetarian"],
    )

    ask_agent(graph, "t1", "I'm vegetarian")

    assert load_preferences(store, "local") == ["vegetarian"]


def test_off_topic_still_passes_the_preference_gate():
    store = InMemoryStore()
    understander = FakeUnderstander().add(
        "I'm vegetarian, also do my homework", understood("off_topic")
    )
    graph = build(
        agent(),
        understander,
        store=store,
        preference_extractor=lambda text, known: ["vegetarian"],
    )

    ask_agent(graph, "t1", "I'm vegetarian, also do my homework")

    assert load_preferences(store, "local") == ["vegetarian"]


def test_no_pending_question_on_an_unknown_thread():
    graph = build(agent())
    assert pending_question(graph, "nope") is None


# --- The terminal chat ----------------------------------------------------------


def fake_terminal(*lines: str):
    """read() returns the given lines in order; write() records the output."""
    replies = iter(lines)
    shown: list[str] = []
    return (lambda prompt: next(replies)), shown.append, shown


def test_chat_asks_then_answers():
    graph = build(agent("Day 1 in Galle ..."))
    read, write, shown = fake_terminal("Plan me a trip", GALLE, "/quit")

    run_agent_chat(graph, read, write)

    assert f"AI: {ASK_PLACE}" in shown
    assert "AI: Day 1 in Galle ..." in shown
    assert shown.index(f"AI: {ASK_PLACE}") < shown.index("AI: Day 1 in Galle ...")


def test_chat_reopened_on_a_paused_thread_shows_the_question(tmp_path):
    db = tmp_path / "chat.sqlite"
    with sqlite_checkpointer(db) as saver:
        ask_agent(build(agent(), checkpointer=saver), "trip-1", "Plan me a trip")

    read, write, shown = fake_terminal(GALLE, "/quit")
    with sqlite_checkpointer(db) as saver:
        graph = build(agent("Day 1 in Galle ..."), checkpointer=saver)
        run_agent_chat(graph, read, write, thread_id="trip-1")

    assert shown[0] == "Continuing conversation trip-1 (2 saved messages)."
    assert shown[1] == f"AI (waiting for your answer): {ASK_PLACE}"
    assert "AI: Day 1 in Galle ..." in shown


def test_new_while_paused_starts_fresh_and_keeps_the_old_question():
    graph = build(agent("Day 1 in Ella ..."))
    read, write, shown = fake_terminal("Plan me a trip", "/new", ELLA, "/quit")

    run_agent_chat(graph, read, write, thread_id="old")

    assert "AI: Day 1 in Ella ..." in shown
    assert pending_question(graph, "old") == ASK_PLACE


def test_cli_no_clarify_needs_chat():
    with pytest.raises(SystemExit):
        tool_calling.main(["Where to hike?", "--no-clarify"])
