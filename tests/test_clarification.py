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
from ai_travel_assistant.schemas import PartyMember, RequestUnderstanding, TripRequest

# --- Helpers --------------------------------------------------------------------


# The details asked after the destination and days. Fakes in other test
# files add these to a plan request so that it is complete and runs in one turn;
# the questions themselves are tested here and in test_clarification_graph.
COMPLETE = {
    "origin": "Colombo",
    "wants_hotels": False,
    "travellers": 2,
    "nationality": "British",  # the code maps it to non_saarc
    "children": 0,
}


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


FULL_TRIP = {
    "destination": "Galle",
    "days": 2,
    "travellers": 2,
    "citizenship": "non_saarc",
    "children": 0,
    "origin": "Colombo",
    "wants_hotels": False,
}
ALL_FIELDS = [
    "destination",
    "days",
    "travellers",
    "citizenship",
    "origin",
    "wants_hotels",
]


def test_missing_critical_treats_blank_as_missing():
    # children is asked only once the group size is known (not of one person).
    assert missing_critical(None) == ALL_FIELDS
    assert missing_critical({**FULL_TRIP, "destination": "  "}) == ["destination"]
    assert missing_critical(FULL_TRIP) == []


def test_missing_fields_come_in_the_order_they_are_asked():
    assert missing_critical({"days": 2, "origin": "Kandy"}) == [
        "destination",
        "travellers",
        "citizenship",
        "wants_hotels",
    ]
    # A room budget is asked only of someone who wants hotel suggestions.
    hotels = {**FULL_TRIP, "wants_hotels": True}
    assert missing_critical(hotels) == ["budget"]
    assert missing_critical({**hotels, "budget": "luxury"}) == []
    assert missing_critical({**hotels, "room_budget": 10000}) == []
    group_only = {k: v for k, v in FULL_TRIP.items() if k != "children"}
    assert missing_critical(group_only) == ["children"]
    assert missing_critical({**FULL_TRIP, "citizenship": "mixed"}) == ["party"]


def test_a_listed_party_answers_every_group_question():
    trip = {
        **FULL_TRIP,
        "citizenship": None,
        "children": None,
        "party": [{"nationality": "Sri Lankan", "age_group": "adult", "count": 2}],
    }
    assert missing_critical(trip) == []


def test_a_new_destination_keeps_only_the_group():
    merged = merge_trip(FULL_TRIP, TripRequest(destination="Ella"))
    assert merged["destination"] == "Ella"
    # About the people: kept.
    assert (merged["travellers"], merged["citizenship"], merged["children"]) == (
        2,
        "non_saarc",
        0,
    )
    # About the old trip: dropped.
    assert merged["origin"] is None
    assert merged["days"] is None
    said = merge_trip(FULL_TRIP, TripRequest(destination="Ella", citizenship="local"))
    assert said["citizenship"] == "local"


def test_a_single_citizenship_replaces_an_earlier_party_list():
    listed = {**FULL_TRIP, "party": [{"nationality": "Indian", "count": 2}]}
    merged = merge_trip(listed, TripRequest(citizenship="local"))
    assert merged["party"] == []
    assert merged["citizenship"] == "local"


# --- Quick replies ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "text", "value"),
    [
        ("destination", "Kandy", "Kandy"),
        ("destination", "  nuwara   ELIYA ", "Nuwara Eliya"),
        ("days", "3 days", 3),
        ("days", "1 day", 1),
        ("days", "4", 4),  # a bare number is a day count
        ("travellers", "Just me", 1),
        ("travellers", "4 people", 4),
        ("travellers", "8", 8),
        ("citizenship", "All Sri Lankan citizens", "local"),
        ("citizenship", "All from a SAARC country", "saarc"),
        ("citizenship", "all from other countries", "non_saarc"),
        ("citizenship", "A mixed group", "mixed"),
        ("children", "No, all adults", 0),
        ("children", "0", 0),
        ("children", "2 children", 2),
        ("origin", "The airport (Katunayake)", "Katunayake"),
        ("origin", "Colombo", "Colombo"),
    ],
)
def test_a_quick_reply_is_read_without_the_llm(field, text, value):
    assert ru.quick_reply_value(field, text) == value


@pytest.mark.parametrize(
    ("field", "text"),
    [
        (None, "Kandy"),  # no question was asked
        ("destination", "Kandy and Ella"),
        ("days", "0"),
        ("travellers", "0"),
        ("days", "about a week"),
        ("citizenship", "We're from Germany"),  # typed: the understander reads it
        ("citizenship", "Sri Lankan / I live here"),  # residence is not citizenship
        ("party", "2 Sri Lankans"),  # a list is always read by the understander
        ("destination", "Colombo"),  # a start place, not a destination button
    ],
)
def test_anything_else_is_not_a_quick_reply(field, text):
    assert ru.quick_reply_value(field, text) is None


def test_every_asked_field_but_the_party_list_has_quick_replies():
    from ai_travel_assistant.schemas import PLAN_FIELDS

    assert set(ru.QUICK_REPLIES) == set(PLAN_FIELDS) - {"party"}
    for field, options in ru.QUICK_REPLIES.items():
        for label, value in options:
            assert ru.quick_reply_value(field, label) == value


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


def test_question_asks_for_the_first_missing_field_only():
    assert clarification_question(["destination", "days"], {}) == (
        "Where in Sri Lanka would you like to go?"
    )
    assert clarification_question(["destination"], {"days": 3}) == (
        "Where in Sri Lanka would you like to spend your 3 days?"
    )
    assert clarification_question(["days", "origin"], {"destination": "Galle"}) == (
        "How many days would you like to spend in Galle?"
    )
    assert clarification_question(["travellers"], FULL_TRIP).startswith(
        "How many people are travelling, including you?"
    )
    citizenship = clarification_question(["citizenship", "origin"], FULL_TRIP)
    assert "Sri Lankan, SAARC and other foreign" in citizenship
    assert "citizenship counts, not where you live" in citizenship
    assert "e.g. '2 Sri Lankan adults" in clarification_question(["party"], FULL_TRIP)
    assert clarification_question(["children"], FULL_TRIP).startswith(
        "How many of the 2 travellers are children?"
    )
    assert clarification_question(["origin"], {"destination": "Jaffna"}) == (
        "Where will you start your trip to Jaffna from?"
    )


def test_ask_clarification_writes_a_tagged_question_and_counts_it():
    update = ask_clarification_node(
        {"missing_fields": ["days", "origin"], "trip_request": {"destination": "Galle"}}
    )
    [question] = update["messages"]
    assert question.name == ru.CLARIFICATION_NAME
    assert update["clarification_rounds"] == 1
    assert update["clarification_field"] == "days"


def test_trip_prompt_note():
    assert trip_prompt_note(None, []) == ""
    note = trip_prompt_note({"destination": "Galle", "days": 2, "budget": None}, [])
    assert "destination=Galle, days=2" in note
    assert "budget" not in note
    assert "assume 3 days" in trip_prompt_note({"destination": "Galle"}, ["days"])


def test_the_prompt_note_names_each_category_and_asks_for_group_totals():
    mixed = {
        **FULL_TRIP,
        "party": [
            {"nationality": "Sri Lankan", "age_group": "adult", "count": 2},
            {"nationality": "Indian", "age_group": "adult", "count": 1},
            {"nationality": "German", "age_group": "adult", "count": 1},
        ],
    }
    note = trip_prompt_note(mixed, [])
    assert "2 Sri Lankan adults, 1 SAARC adult, 1 non-SAARC foreign adult" in note
    assert "never one category's price for another" in note
    assert "per person and for the whole group" in note
    assert "LKR and USD" in note and "give no visa details" in note
    assert "party=" not in note  # the raw list is never shown


def test_an_all_local_group_gets_lkr_only():
    note = trip_prompt_note({**FULL_TRIP, "citizenship": "local"}, [])
    assert "LKR only" in note
    assert "do not call convert_currency" in note
    assert "Skip visa advice" in note


def test_an_unknown_group_is_stated_as_an_assumption():
    note = trip_prompt_note({"destination": "Galle"}, ["citizenship"])
    assert "Sri Lankan, SAARC and other foreign visitors may pay different" in note


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
    assert update["missing_fields"] == [
        "days",
        "travellers",
        "citizenship",
        "origin",
        "wants_hotels",
    ]
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
    assert update["missing_fields"] == [
        "travellers",
        "citizenship",
        "origin",
        "wants_hotels",
    ]
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
    assert update["missing_fields"] == [
        "destination",
        "days",
        "travellers",
        "citizenship",
        "origin",
        "wants_hotels",
    ]


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


# Seen live (2026-11-24): a weather answer counted as "a plan was given", the
# understander was told so and returned "modify", and every question was skipped.
WEATHER_THEN_PLAN = [
    HumanMessage("What is the weather like in Ella?"),
    AIMessage("Ella: 22-27 °C, with afternoon showers."),
    HumanMessage("Plan 2 days in Kandy"),
]


def test_a_new_plan_after_a_weather_answer_is_clarified():
    understander = FakeUnderstander(
        {"Plan 2 days in Kandy": understood("plan", destination="Kandy", days=2)}
    )
    update = make_understand_request_node(understander)({"messages": WEATHER_THEN_PLAN})
    assert "has_plan" not in understander.calls[0][1]  # no plan was given
    assert update["intent"] == "plan"
    assert update["missing_fields"][0] == "travellers"
    assert route_after_understanding(update) == "clarify"


def test_modify_after_a_weather_answer_is_treated_as_a_new_plan():
    understander = FakeUnderstander(
        {"Plan 2 days in Kandy": understood("modify", destination="Kandy", days=2)}
    )
    update = make_understand_request_node(understander)({"messages": WEATHER_THEN_PLAN})
    assert update["intent"] == "plan"  # nothing to modify yet
    assert update["trip_request"]["destination"] == "Kandy"
    assert update["missing_fields"]
    assert route_after_understanding(update) == "clarify"


def test_only_a_real_plan_counts_as_a_plan_given():
    from ai_travel_assistant.itinerary_output import ITINERARY_NAME

    weather = {"messages": WEATHER_THEN_PLAN}
    assert agent_answered_before(WEATHER_THEN_PLAN)  # an answer, but not a plan
    assert not ru.plan_given_before(weather)
    # A checked itinerary in the state counts, even with a question after it.
    assert ru.plan_given_before({**weather, "itinerary": {"destination": "Ella"}})
    # So does a rendered itinerary reply in the thread.
    rendered = [
        HumanMessage("Plan 2 days in Ella"),
        AIMessage("# 2-day trip to Ella", name=ITINERARY_NAME),
        HumanMessage("Make day 2 easier"),
    ]
    assert ru.plan_given_before({"messages": rendered})
    # A prose plan (no structured itinerary) counts once a trip is planned.
    assert ru.plan_given_before(
        {**weather, "trip_request": {"destination": "Ella", "days": 2}}
    )
    # Clarification questions are not plans.
    asked = [
        HumanMessage("Plan me a trip"),
        AIMessage("Where?", name=ru.CLARIFICATION_NAME),
        HumanMessage("Galle"),
    ]
    assert not ru.plan_given_before(
        {"messages": asked, "trip_request": {"destination": "Galle"}}
    )


def test_a_failing_understander_is_logged_without_its_error_text(caplog):
    def broken(text, **context):
        raise RuntimeError("bad reply; key=secret-123 user text: Plan a trip")

    with caplog.at_level("WARNING", logger=ru.__name__):
        update = make_understand_request_node(broken)(
            {"messages": [HumanMessage("Plan a trip")]}
        )
    assert update["intent"] is None
    [record] = [r for r in caplog.records if r.name == ru.__name__]
    logged = record.getMessage()
    assert "unexpected" in logged and "RuntimeError" in logged
    assert "secret-123" not in logged and "Plan a trip" not in logged


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


# --- Quick replies and the group in the node -----------------------------------


def asked(field: str, reply: str, trip: dict, **state) -> dict:
    return {
        "messages": [
            HumanMessage("Plan me a trip"),
            AIMessage("The question", name=ru.CLARIFICATION_NAME),
            HumanMessage(reply),
        ],
        "clarification_context": "The question",
        "clarification_field": field,
        "clarification_rounds": 1,
        "trip_request": trip,
        **state,
    }


def test_a_clicked_quick_reply_makes_no_understander_call():
    understander = FakeUnderstander({})
    update = make_understand_request_node(understander)(
        asked(
            "citizenship",
            "All from a SAARC country",
            {"destination": "Galle", "travellers": 3},
        )
    )
    assert understander.calls == []
    assert update["intent"] == "plan"
    assert update["trip_request"]["citizenship"] == "saarc"
    assert update["trip_request"]["destination"] == "Galle"
    assert update["missing_fields"] == ["days", "children", "origin", "wants_hotels"]
    assert update["clarification_field"] is None
    assert update["clarification_rounds"] == 1


def test_a_typed_mixed_group_goes_to_the_understander():
    party = [
        PartyMember(nationality="Sri Lankan", count=2),
        PartyMember(nationality="German", age_group="child"),
    ]
    understander = FakeUnderstander(
        {"2 Sri Lankans and a German kid": understood("plan", party=party)}
    )
    update = make_understand_request_node(understander)(
        asked(
            "party",
            "2 Sri Lankans and a German kid",
            {**FULL_TRIP, "citizenship": "mixed", "children": None},
        )
    )
    assert [text for text, _ in understander.calls] == [
        "2 Sri Lankans and a German kid"
    ]
    trip = update["trip_request"]
    assert (trip["travellers"], trip["children"]) == (3, 1)  # from the list
    assert update["missing_fields"] == []


def test_more_children_than_travellers_is_not_taken_as_a_click():
    understander = FakeUnderstander({"3": understood("question")})
    make_understand_request_node(understander)(
        asked("children", "3", {**FULL_TRIP, "children": None})
    )
    assert len(understander.calls) == 1  # 3 of 2 travellers: read it properly


def test_a_button_label_on_a_fresh_message_is_not_a_quick_reply():
    understander = FakeUnderstander({"Kandy": understood("question")})
    update = make_understand_request_node(understander)(
        {"messages": [HumanMessage("Kandy")], "clarification_field": "destination"}
    )
    assert len(understander.calls) == 1  # no question pending: read it properly
    assert update["intent"] == "question"


# --- The ticket category is decided by code, not the LLM ---------------------------


@pytest.mark.parametrize(
    ("said", "category"),
    [
        ({"nationality": "Maldivian"}, "saarc"),  # the live bug: LLM said non_saarc
        ({"nationality": "Maldivian", "citizenship": "non_saarc"}, "saarc"),
        ({"nationality": "Sri Lankan"}, "local"),
        ({"nationality": "German"}, "non_saarc"),
        ({"citizenship": "local"}, None),  # a category with no nationality: ask
        ({"nationality": "tourist"}, None),  # unclear: ask
        ({"citizenship": "mixed"}, "mixed"),
        # "mixed" that contradicts the rest is dropped (seen live).
        ({"citizenship": "mixed", "nationality": "Sri Lankan"}, "local"),
        ({"citizenship": "mixed", "travellers": 1}, None),
        (
            {
                "citizenship": "mixed",
                "party": [PartyMember(nationality="Sri Lankan")],
            },
            None,  # the party list itself decides
        ),
    ],
)
def test_the_code_maps_a_nationality_to_its_ticket_category(said, category):
    trip = ru.code_decides_citizenship(TripRequest(**said))
    assert trip.citizenship == category


def test_a_typed_nationality_reaches_the_trip_as_its_category():
    understander = FakeUnderstander(
        {
            "just me, I'm Maldivian": understood(
                "plan", travellers=1, nationality="Maldivian", citizenship="non_saarc"
            )
        }
    )
    update = make_understand_request_node(understander)(
        asked("citizenship", "just me, I'm Maldivian", {"destination": "Galle"})
    )
    assert update["trip_request"]["citizenship"] == "saarc"
