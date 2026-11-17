"""Stage 5 checks: long-term preference memory (gate, extractor, Store helpers).

Offline only: the Store is InMemoryStore or a SQLite file in tmp_path, and the
LLM extractor runs on a fake model, so no Gemini calls.
"""

from __future__ import annotations

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from ai_travel_assistant import tool_calling
from ai_travel_assistant.long_term_memory import (
    TravellerPreferences,
    clear_preferences,
    load_preferences,
    looks_like_preference,
    make_llm_preference_extractor,
    preference_key,
    preferences_namespace,
    save_preferences,
    sqlite_store,
)
from ai_travel_assistant.tool_calling import (
    DEFAULT_USER,
    LOAD_MEMORY,
    SAVE_MEMORY,
    SYSTEM_PROMPT,
    ask_agent,
    build_tool_agent_graph,
    final_answer,
    run_agent_chat,
    system_prompt_for,
)

# --- The gate -----------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "I'm vegetarian, so keep that in mind.",
        "I am allergic to peanuts",
        "I hate early mornings",
        "I prefer quiet places over crowds",
        "I don't like long bus rides",
        "I'm travelling with two kids",
        "I am on a tight budget",
        "My partner uses a wheelchair",
        "Please remember that I avoid spicy food",
        "Make it halal-friendly please",
    ],
)
def test_gate_spots_preference_statements(text):
    assert looks_like_preference(text)


@pytest.mark.parametrize(
    "text",
    [
        "Plan a 3-day trip to Galle.",
        "I'm planning a trip to Kandy",
        "I want 3 days in Ella",
        "What is the weather in Nuwara Eliya?",
        "How far is Sigiriya from Colombo?",
        "Make day 2 easier.",
    ],
)
def test_gate_ignores_plain_trip_requests(text):
    assert not looks_like_preference(text)


# --- The extractor ------------------------------------------------------------


class StructuredFake(GenericFakeChatModel):
    """with_structured_output() returns a runnable that gives a fixed object."""

    result: object = None
    seen: list = []

    def with_structured_output(self, schema, **kwargs):
        from langchain_core.runnables import RunnableLambda

        def answer(messages):
            self.seen.append(messages)
            return self.result

        return RunnableLambda(answer)


def test_extractor_returns_the_models_preferences_and_sends_known_ones():
    model = StructuredFake(
        messages=iter([]),
        result=TravellerPreferences(preferences=["vegetarian"]),
        seen=[],
    )
    extract = make_llm_preference_extractor(model)

    assert extract("I'm vegetarian", ["avoids long drives"]) == ["vegetarian"]
    human = model.seen[0][1].content
    assert "Already known: avoids long drives" in human
    assert "Message: I'm vegetarian" in human


def test_extractor_rejects_a_missing_reply():
    model = StructuredFake(messages=iter([]), result=None, seen=[])
    with pytest.raises(ValueError):
        make_llm_preference_extractor(model)("I'm vegan", [])


# --- Store helpers ------------------------------------------------------------


def test_save_and_load_keep_order_and_skip_duplicates():
    store = InMemoryStore()
    added = save_preferences(store, "u1", ["vegetarian", "avoids long drives"])
    again = save_preferences(store, "u1", ["Vegetarian ", "likes waterfalls"])

    assert added == ["vegetarian", "avoids long drives"]
    assert again == ["likes waterfalls"]
    assert load_preferences(store, "u1") == [
        "vegetarian",
        "avoids long drives",
        "likes waterfalls",
    ]


def test_travellers_do_not_share_preferences():
    store = InMemoryStore()
    save_preferences(store, "alice", ["vegan"])
    assert load_preferences(store, "bob") == []


def test_blank_preferences_are_ignored():
    store = InMemoryStore()
    assert save_preferences(store, "u1", ["", "  ", "!!"]) == []


def test_clear_preferences():
    store = InMemoryStore()
    save_preferences(store, "u1", ["vegan", "early riser"])
    assert clear_preferences(store, "u1") == 2
    assert load_preferences(store, "u1") == []


def test_namespace_and_key():
    assert preferences_namespace("u1") == ("users", "u1", "preferences")
    assert preference_key("Avoids LONG drives!") == "avoids-long-drives"


def test_sqlite_store_keeps_preferences_after_reopening(tmp_path):
    db = tmp_path / "nested" / "memory.sqlite"
    with sqlite_store(db) as store:
        save_preferences(store, "u1", ["vegetarian"])
    with sqlite_store(db) as store:
        assert load_preferences(store, "u1") == ["vegetarian"]


# --- In the agent graph: load_memory and save_memory --------------------------


class RecordingModel(GenericFakeChatModel):
    """Scripted replies; records the messages of every call."""

    calls: list = []

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def recording(*replies: str) -> RecordingModel:
    return RecordingModel(messages=iter(AIMessage(r) for r in replies), calls=[])


class FakeExtractor:
    """Plays the LLM extractor: returns fixed preferences, counts its calls."""

    def __init__(self, *found: str, error: Exception | None = None):
        self.found = list(found)
        self.error = error
        self.calls: list[tuple[str, list[str]]] = []

    def __call__(self, text, known):
        self.calls.append((text, list(known)))
        if self.error:
            raise self.error
        return self.found


def memory_graph(model, extractor, store=None):
    return build_tool_agent_graph(
        model,
        checkpointer=InMemorySaver(),
        store=store if store is not None else InMemoryStore(),
        preference_extractor=extractor,
    )


def test_preference_from_one_thread_shapes_another():
    model = recording("Noted, vegetarian food it is.", "Galle plan, veg options.")
    extractor = FakeExtractor("vegetarian")
    graph = memory_graph(model, extractor)

    ask_agent(graph, "thread-a", "I'm vegetarian.", user_id="u1")
    ask_agent(graph, "thread-b", "Plan 2 days in Galle.", user_id="u1")

    second_prompt = model.calls[1][0]
    assert isinstance(second_prompt, SystemMessage)
    assert "Known traveller preferences" in second_prompt.content
    assert "vegetarian" in second_prompt.content
    # Thread B's history holds only its own message.
    assert [m.content for m in model.calls[1][1:]] == ["Plan 2 days in Galle."]


def test_plain_trip_request_never_calls_the_extractor():
    extractor = FakeExtractor("should not be saved")
    store = InMemoryStore()
    graph = memory_graph(recording("Here is your plan."), extractor, store)

    ask_agent(graph, "t1", "Plan a 3-day trip to Kandy.", user_id="u1")

    assert extractor.calls == []
    assert load_preferences(store, "u1") == []


def test_other_travellers_do_not_see_the_preference():
    model = recording("Noted.", "Plan.")
    graph = memory_graph(model, FakeExtractor("vegan"))

    ask_agent(graph, "t1", "I'm vegan.", user_id="alice")
    ask_agent(graph, "t2", "Plan Ella.", user_id="bob")

    assert "Known traveller preferences" not in model.calls[1][0].content


def test_extractor_sees_known_preferences():
    store = InMemoryStore()
    save_preferences(store, "u1", ["vegetarian"])
    extractor = FakeExtractor()
    graph = memory_graph(recording("Ok."), extractor, store)

    ask_agent(graph, "t1", "I hate early mornings.", user_id="u1")

    assert extractor.calls == [("I hate early mornings.", ["vegetarian"])]


def test_extractor_failure_keeps_the_answer():
    graph = memory_graph(
        recording("Still answered."), FakeExtractor(error=RuntimeError("429"))
    )

    turn = ask_agent(graph, "t1", "I'm vegetarian.", user_id="u1")

    assert final_answer(turn) == "Still answered."


def test_without_a_store_the_graph_has_no_memory_nodes():
    graph = build_tool_agent_graph(recording("Hi."))
    assert LOAD_MEMORY not in graph.nodes
    assert SAVE_MEMORY not in graph.nodes


def test_store_without_extractor_only_loads():
    graph = build_tool_agent_graph(
        recording("Hi."), checkpointer=InMemorySaver(), store=InMemoryStore()
    )
    assert LOAD_MEMORY in graph.nodes
    assert SAVE_MEMORY not in graph.nodes


def test_system_prompt_for_without_preferences_is_unchanged():
    assert system_prompt_for([]) == SYSTEM_PROMPT


# --- Chat commands: /prefs, /forget, --user -----------------------------------


def fake_terminal(*lines: str):
    replies = iter(lines)
    shown: list[str] = []
    return (lambda prompt: next(replies)), shown.append, shown


def test_chat_saves_for_the_given_user_and_lists_with_prefs():
    store = InMemoryStore()
    graph = memory_graph(recording("Noted."), FakeExtractor("vegetarian"), store)
    read, write, shown = fake_terminal("I'm vegetarian.", "/prefs", "/quit")

    run_agent_chat(graph, read, write, user_id="tharaka")

    assert load_preferences(store, "tharaka") == ["vegetarian"]
    assert load_preferences(store, DEFAULT_USER) == []
    assert "Saved preferences: vegetarian" in shown


def test_prefs_with_nothing_saved():
    graph = memory_graph(recording(), FakeExtractor())
    read, write, shown = fake_terminal("/prefs", "/quit")

    run_agent_chat(graph, read, write)

    assert "No saved preferences yet." in shown


def test_forget_clears_preferences_without_calling_the_model():
    store = InMemoryStore()
    save_preferences(store, "u1", ["vegan", "early riser"])
    model = recording()
    graph = memory_graph(model, FakeExtractor(), store)
    read, write, shown = fake_terminal("/forget", "/quit")

    run_agent_chat(graph, read, write, user_id="u1")

    assert "Forgot 2 saved preferences." in shown
    assert load_preferences(store, "u1") == []
    assert model.calls == []


def test_prefs_without_a_store_says_memory_is_off():
    graph = build_tool_agent_graph(recording(), checkpointer=InMemorySaver())
    read, write, shown = fake_terminal("/prefs", "/quit")

    run_agent_chat(graph, read, write)

    assert "Long-term memory is off for this chat." in shown


def test_cli_user_needs_chat():
    with pytest.raises(SystemExit):
        tool_calling.main(["Where to hike?", "--user", "someone"])
