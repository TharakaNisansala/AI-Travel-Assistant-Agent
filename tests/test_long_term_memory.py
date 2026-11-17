"""Stage 5 checks: long-term preference memory (gate, extractor, Store helpers).

Offline only: the Store is InMemoryStore or a SQLite file in tmp_path, and the
LLM extractor runs on a fake model, so no Gemini calls.
"""

from __future__ import annotations

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langgraph.store.memory import InMemoryStore

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
