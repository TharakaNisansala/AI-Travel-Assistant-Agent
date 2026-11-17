"""Stage 6 checks: understanding requests and asking for missing details.

Offline only. A dictionary-lookup understander plays the extraction LLM and a
scripted fake model plays the agent, so no Gemini calls are made.
"""

from __future__ import annotations

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import ValidationError

from ai_travel_assistant import request_understanding as ru
from ai_travel_assistant.request_understanding import (
    MAX_CLARIFICATIONS,
    agent_answered_before,
    ask_clarification_node,
    clarification_question,
    make_understand_request_node,
    merge_trip,
    missing_critical,
    route_after_understanding,
    trip_prompt_note,
)
from ai_travel_assistant.schemas import RequestUnderstanding, TripRequest

# --- Helpers --------------------------------------------------------------------


class FakeUnderstander:
    """Looks the message up in a table and records every call."""

    def __init__(self, answers: dict[str, RequestUnderstanding]):
        self.answers = answers
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, text, **context):
        self.calls.append((text, context))
        return self.answers[text]


def understood(intent, **trip) -> RequestUnderstanding:
    return RequestUnderstanding(intent=intent, trip=TripRequest(**trip))


# --- Schema -------------------------------------------------------------------


def test_understanding_defaults_to_an_empty_trip():
    understanding = RequestUnderstanding(intent="question")
    assert understanding.trip == TripRequest()


def test_understanding_rejects_an_unknown_intent():
    with pytest.raises(ValidationError):
        RequestUnderstanding(intent="booking")


def test_understanding_requires_an_intent():
    assert RequestUnderstanding.model_json_schema()["required"] == ["intent"]


# --- Merging trip details -------------------------------------------------------


def test_merge_starts_from_the_first_message():
    merged = merge_trip(None, TripRequest(destination="Galle"))
    assert merged["destination"] == "Galle"
    assert merged["days"] is None


def test_merge_keeps_saved_values_the_message_leaves_out():
    merged = merge_trip({"destination": "Galle", "days": None}, TripRequest(days=2))
    assert (merged["destination"], merged["days"]) == ("Galle", 2)


def test_merge_replaces_a_value_the_message_changes():
    merged = merge_trip({"destination": "Galle", "days": 2}, TripRequest(days=5))
    assert merged["days"] == 5


def test_merge_adds_interests_without_duplicates():
    known = {"destination": "Ella", "interests": ["hiking"]}
    merged = merge_trip(known, TripRequest(interests=["hiking", "tea plantations"]))
    assert merged["interests"] == ["hiking", "tea plantations"]


def test_a_new_destination_starts_a_new_trip():
    known = {"destination": "Ella", "days": 3, "interests": ["hiking"]}
    merged = merge_trip(known, TripRequest(destination="Galle"))
    assert merged["destination"] == "Galle"
    assert merged["days"] is None
    assert merged["interests"] == []


def test_same_destination_in_other_case_is_the_same_trip():
    known = {"destination": "Ella", "days": 3}
    merged = merge_trip(known, TripRequest(destination="ella"))
    assert merged["days"] == 3


def test_missing_critical_treats_blank_as_missing():
    assert missing_critical(None) == ["destination", "days"]
    assert missing_critical({"destination": "  ", "days": 2}) == ["destination"]
    assert missing_critical({"destination": "Galle", "days": 2}) == []


# --- The router -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("state", "route"),
    [
        ({"intent": "off_topic"}, "off_topic"),
        ({"intent": "plan", "missing_fields": ["destination", "days"]}, "clarify"),
        ({"intent": "plan", "missing_fields": ["days"]}, "clarify"),
        ({"intent": "plan", "missing_fields": []}, "agent"),
        ({"intent": "question", "missing_fields": []}, "agent"),
        ({"intent": "modify", "missing_fields": []}, "agent"),
        ({"intent": None}, "agent"),  # understanding failed
        ({}, "agent"),
        (
            {
                "intent": "plan",
                "missing_fields": ["days"],
                "clarification_rounds": MAX_CLARIFICATIONS,
            },
            "agent",
        ),
    ],
)
def test_route_after_understanding(state, route):
    assert route_after_understanding(state) == route


# --- Questions and prompt notes -------------------------------------------------


def test_question_asks_only_for_what_is_missing():
    assert clarification_question(["destination", "days"], {}) == (
        "Where in Sri Lanka would you like to go, and for how many days?"
    )
    assert clarification_question(["destination"], {"days": 3}) == (
        "Where in Sri Lanka would you like to spend your 3 days?"
    )
    assert clarification_question(["days"], {"destination": "Galle"}) == (
        "How many days would you like to spend in Galle?"
    )


def test_ask_clarification_writes_a_tagged_question_and_counts_it():
    update = ask_clarification_node(
        {"missing_fields": ["days"], "trip_request": {"destination": "Galle"}}
    )
    [question] = update["messages"]
    assert question.name == ru.CLARIFICATION_NAME
    assert update["clarification_rounds"] == 1


def test_trip_prompt_note():
    assert trip_prompt_note(None, []) == ""
    note = trip_prompt_note({"destination": "Galle", "days": 2, "budget": None}, [])
    assert "destination=Galle, days=2" in note
    assert "budget" not in note
    assert "assume 3 days" in trip_prompt_note({"destination": "Galle"}, ["days"])


def test_agent_answered_before_ignores_our_own_replies():
    asked = [
        HumanMessage("Plan me a trip"),
        AIMessage("Where?", name=ru.CLARIFICATION_NAME),
        HumanMessage("Galle"),
    ]
    assert not agent_answered_before(asked)
    planned = [HumanMessage("Plan Galle"), AIMessage("Day 1 ..."), HumanMessage("More")]
    assert agent_answered_before(planned)


# --- The understand_request node ------------------------------------------------


def test_understand_records_intent_trip_and_missing_fields():
    understander = FakeUnderstander(
        {"Plan Galle": understood("plan", destination="Galle")}
    )
    update = make_understand_request_node(understander)(
        {"messages": [HumanMessage("Plan Galle")]}
    )
    assert update["intent"] == "plan"
    assert update["trip_request"]["destination"] == "Galle"
    assert update["missing_fields"] == ["days"]
    assert update["clarification_rounds"] == 0


def test_a_question_does_not_change_the_trip():
    understander = FakeUnderstander(
        {"Weather in Ella?": understood("question", destination="Ella")}
    )
    update = make_understand_request_node(understander)(
        {
            "messages": [HumanMessage("Weather in Ella?")],
            "trip_request": {"destination": "Galle"},
        }
    )
    assert "trip_request" not in update
    assert update["missing_fields"] == []


def test_a_reply_is_read_with_its_question_and_known_details():
    understander = FakeUnderstander({"3": understood("question", days=3)})
    update = make_understand_request_node(understander)(
        {
            "messages": [
                HumanMessage("Plan Galle"),
                AIMessage("How many days?", name=ru.CLARIFICATION_NAME),
                HumanMessage("3"),
            ],
            "clarification_context": "How many days?",
            "clarification_rounds": 1,
            "trip_request": {"destination": "Galle"},
        }
    )
    assert understander.calls == [
        (
            "3",
            {
                "clarification_context": "How many days?",
                "existing_data": {"destination": "Galle"},
            },
        )
    ]
    assert update["intent"] == "plan"  # a reply to our question is planning
    assert update["trip_request"]["days"] == 3
    assert update["missing_fields"] == []
    assert update["clarification_context"] is None
    assert update["clarification_rounds"] == 1


def test_an_off_topic_reply_to_a_question_stays_off_topic():
    understander = FakeUnderstander({"Do my homework": understood("off_topic")})
    update = make_understand_request_node(understander)(
        {
            "messages": [HumanMessage("Do my homework")],
            "clarification_context": "Where?",
        }
    )
    assert update["intent"] == "off_topic"


def test_a_fresh_message_resets_the_round_count():
    understander = FakeUnderstander({"Hi": understood("question")})
    update = make_understand_request_node(understander)(
        {"messages": [HumanMessage("Hi")], "clarification_rounds": 2}
    )
    assert update["clarification_rounds"] == 0


def test_modify_without_any_plan_is_a_new_plan():
    understander = FakeUnderstander({"Make it longer": understood("modify")})
    update = make_understand_request_node(understander)(
        {"messages": [HumanMessage("Make it longer")]}
    )
    assert update["intent"] == "plan"
    assert update["missing_fields"] == ["destination", "days"]


def test_modify_after_a_plan_tells_the_understander_and_skips_clarifying():
    understander = FakeUnderstander({"Make day 2 easier": understood("modify")})
    update = make_understand_request_node(understander)(
        {
            "messages": [
                HumanMessage("Plan 2 days in Galle"),
                AIMessage("Day 1 ..."),
                HumanMessage("Make day 2 easier"),
            ],
            "trip_request": {"destination": "Galle", "days": 2},
        }
    )
    assert understander.calls[0][1]["has_plan"] is True
    assert update["intent"] == "modify"
    assert update["missing_fields"] == []


def test_a_failing_understander_hands_the_turn_to_the_agent():
    def broken(text, **context):
        raise RuntimeError("429 RESOURCE_EXHAUSTED")

    update = make_understand_request_node(broken)(
        {"messages": [HumanMessage("Plan a trip")]}
    )
    assert update["intent"] is None
    assert route_after_understanding(update) == "agent"


def test_a_blank_message_makes_no_understander_call():
    understander = FakeUnderstander({})
    update = make_understand_request_node(understander)(
        {"messages": [HumanMessage("  ")]}
    )
    assert understander.calls == []
    assert route_after_understanding(update) == "agent"


# --- The Gemini understander (structured output, faked) -------------------------


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


def test_llm_understander_sends_context_and_returns_the_result():
    model = StructuredFake(
        messages=iter([]), result=understood("plan", days=3), seen=[]
    )
    understand = ru.make_llm_request_understander(model)

    result = understand(
        "3",
        clarification_context="How many days?",
        existing_data={"destination": "Galle"},
        has_plan=True,
    )

    assert result.trip.days == 3
    system, human = model.seen[0]
    assert "off_topic" in system.content
    assert "N nights" in system.content  # Stage 1 extraction rules are reused
    assert human.content.startswith("The assistant already gave a plan")
    assert "Question the assistant just asked: How many days?" in human.content
    assert "Traveller's latest reply: 3" in human.content


def test_llm_understander_sends_a_plain_first_message_unchanged():
    model = StructuredFake(messages=iter([]), result=understood("plan"), seen=[])
    ru.make_llm_request_understander(model)("Plan me a trip")
    assert model.seen[0][1].content == "Plan me a trip"


def test_llm_understander_rejects_a_missing_reply():
    model = StructuredFake(messages=iter([]), result=None, seen=[])
    with pytest.raises(ValueError):
        ru.make_llm_request_understander(model)("Plan me a trip")
