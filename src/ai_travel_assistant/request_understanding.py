"""Stage 6: understanding the request, and asking when a critical detail is missing.

What this file demonstrates:
  1. Intent classification with structured output. One LLM call reads the
     traveller's latest message and returns a RequestUnderstanding: what the
     message is for (plan, modify, question, off_topic) and the trip details it
     states. Routing then reads that from state; it never calls the LLM.
  2. Merging requirements across turns. "Plan me a trip" then "Galle" then "3"
     builds one TripRequest; a missing value never erases a saved one.
  3. Asking versus assuming. Only a PLAN request with a missing PLAN_FIELDS
     value (destination, days, travellers, who is travelling, origin) makes
     the agent ask, one field per question. A question without a place
     ("Where can I surf?") goes to the agent, which can search for places.
  4. Human-in-the-loop with interrupt(). ask_clarification writes the question
     into the conversation; wait_for_reply calls interrupt() and the run stops.
     The caller resumes it with Command(resume=reply). On resume LangGraph
     re-runs wait_for_reply FROM THE TOP, so it does nothing but wait: no LLM
     call and no state change happen before interrupt().
  5. Guardrails: at most MAX_CLARIFICATIONS questions in a row, then the agent
     takes over and states its assumptions; if the understander fails (e.g. a
     Gemini 429) the turn goes straight to the agent, as before Stage 6.
  6. Quick replies. Each question comes with QUICK_REPLIES buttons. A reply
     that is exactly one of them is read by a dictionary lookup, not the LLM,
     so a click costs no quota. Typed answers still go to the understander.
  7. Who is travelling (count, citizenships, children) decides the ticket
     category of each person; party.py turns it into counts per category.
     It is about the group, not the trip, so a new destination keeps it.

The nodes work on the agent's state (tool_calling.AgentState). The graph wiring
lives in tool_calling.build_tool_agent_graph(..., understander=...).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, Protocol

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.types import interrupt

from ai_travel_assistant.party import (
    complete_party,
    describe_group,
    group_breakdown,
    missing_group_fields,
    nationality_group,
)
from ai_travel_assistant.robustness import error_category, is_quota_exhausted
from ai_travel_assistant.schemas import (
    PLAN_FIELDS,
    RequestUnderstanding,
    TripRequest,
)

logger = logging.getLogger(__name__)

# Node names, used by tool_calling to wire the graph.
UNDERSTAND_REQUEST = "understand_request"
ASK_CLARIFICATION = "ask_clarification"
WAIT_FOR_REPLY = "wait_for_reply"
RESPOND_OFF_TOPIC = "respond_off_topic"

# What route_after_understanding returns; the graph maps these to nodes.
Route = Literal["agent", "clarify", "off_topic"]
TO_AGENT: Route = "agent"
TO_CLARIFY: Route = "clarify"
TO_OFF_TOPIC: Route = "off_topic"

# Questions asked in a row before the agent takes over with assumptions. One
# per PLAN_FIELDS value, so "Plan me a trip" can be completed by clicks alone.
MAX_CLARIFICATIONS = len(PLAN_FIELDS)

# The buttons shown under each question: (label, value saved in the trip).
# The label is also the text of the traveller's reply when it is clicked.
QUICK_REPLIES: dict[str, tuple[tuple[str, Any], ...]] = {
    "destination": tuple(
        (place, place)
        for place in (
            "Kandy",
            "Ella",
            "Galle",
            "Sigiriya",
            "Nuwara Eliya",
            "Mirissa",
            "Trincomalee",
            "Jaffna",
        )
    ),
    "days": tuple((f"{n} day{'s' if n > 1 else ''}", n) for n in (1, 2, 3, 5, 7)),
    "travellers": (
        ("Just me", 1),
        *((f"{n} people", n) for n in range(2, 7)),
    ),
    # Citizenship, not residence: a foreigner living in Sri Lanka is foreign.
    "citizenship": (
        ("All Sri Lankan citizens", "local"),
        ("All from a SAARC country", "saarc"),
        ("All from other countries", "non_saarc"),
        ("A mixed group", "mixed"),
    ),
    "children": (
        ("No, all adults", 0),
        ("1 child", 1),
        ("2 children", 2),
        ("3 children", 3),
    ),
    # "party" has no buttons: a mixed group is typed ("2 Sri Lankans, 1 Indian").
    # Suggestions only: this app cannot book or check availability.
    "wants_hotels": (
        ("Yes, suggest hotels", True),
        ("No thanks", False),
    ),
    "budget": (
        ("Budget", "budget"),
        ("Mid-range", "mid-range"),
        ("Luxury", "luxury"),
    ),
    "origin": (
        ("Colombo", "Colombo"),
        ("The airport (Katunayake)", "Katunayake"),
        ("Kandy", "Kandy"),
        ("Galle", "Galle"),
        ("Negombo", "Negombo"),
    ),
}

# Names that mark replies written by these nodes rather than by the agent, so
# a later turn can tell "a plan was given" from "we only asked a question".
CLARIFICATION_NAME = "clarification"
OFF_TOPIC_NAME = "off_topic"

OFF_TOPIC_REPLY = (
    "I can only help with travel in Sri Lanka: places to visit, itineraries, "
    "weather, getting around and costs. What trip can I help you plan?"
)


# --- The understander -----------------------------------------------------------


class RequestUnderstander(Protocol):
    """Anything that turns the latest message into a RequestUnderstanding.

    In the app this calls Gemini; in the tests it is a dictionary lookup.
    The keyword arguments help it read short replies and follow-ups:
      clarification_context: the question the message answers, if any
      existing_data: critical trip details already known
      has_plan: True when the agent already answered earlier in this thread
    """

    def __call__(
        self,
        text: str,
        *,
        clarification_context: str | None = None,
        existing_data: dict[str, object] | None = None,
        has_plan: bool = False,
    ) -> RequestUnderstanding: ...


UNDERSTANDING_INSTRUCTIONS = """\
You read one message from a traveller talking to a Sri Lanka travel assistant.
Return its intent and the trip details it states.

Intent:
- plan: asks for a trip plan or itinerary, or answers a question the assistant
  asked about the trip.
- modify: asks to change a plan the assistant already gave ("make day 2
  easier", "add a beach day"). Only when a plan was already given.
- question: any other travel question (weather, distances, what to see, where
  to go), a greeting, or a statement about the traveller such as a diet or
  pace preference.
- off_topic: clearly not about travel (homework, coding, maths, news).

"""


def make_llm_request_understander(llm: BaseChatModel) -> RequestUnderstander:
    """An understander that asks the LLM (one structured-output call)."""
    # Imported here so tool_calling (and the offline tests) never load the
    # Gemini setup code that structured_output imports.
    from ai_travel_assistant.structured_output import (
        EXTRACTION_INSTRUCTIONS,
        build_extraction_message,
    )

    structured = llm.with_structured_output(RequestUnderstanding)
    instructions = (
        UNDERSTANDING_INSTRUCTIONS + "Trip details:\n" + EXTRACTION_INSTRUCTIONS
    )

    def understand(
        text: str,
        *,
        clarification_context: str | None = None,
        existing_data: dict[str, object] | None = None,
        has_plan: bool = False,
    ) -> RequestUnderstanding:
        message = build_extraction_message(text, clarification_context, existing_data)
        if has_plan:
            if message == text:
                message = f"Traveller's latest reply: {text}"
            message = (
                "The assistant already gave a plan earlier in this conversation.\n"
                + message
            )
        result = structured.invoke([SystemMessage(instructions), HumanMessage(message)])
        if not isinstance(result, RequestUnderstanding):
            raise ValueError(
                f"Expected a RequestUnderstanding, got {type(result).__name__}"
            )
        return result

    return understand


# --- Merging trip details across turns -----------------------------------------


def _is_missing(value: object) -> bool:
    """None, a blank string or an empty list counts as missing."""
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, list):
        return not value
    return value is None


# Details about the travellers rather than the trip: a new destination keeps
# them, so "Now plan Ella" does not ask the group size again.
PERSON_FIELDS = ("travellers", "children", "nationality", "citizenship", "party")

# Fields where a bare number ("3") answers the question without the LLM.
_NUMBER_FIELDS = ("days", "travellers", "children")


def merge_trip(known: Mapping[str, object] | None, new: TripRequest) -> dict:
    """Merge the details of the latest message into the saved ones.

    - A value the message gives replaces the saved one ("make it 5 days").
    - A value the message leaves out never erases a saved one.
    - Interests add up ("also tea plantations").
    - A different destination starts a new trip, so the old trip's interests
      or dates do not leak into it. Only PERSON_FIELDS carry over.
    Returns a plain dict, which the checkpointer saves without custom classes.
    """
    fresh = new.model_dump()
    saved = dict(known or {})
    old_place, new_place = saved.get("destination"), fresh.get("destination")
    if (
        not _is_missing(old_place)
        and not _is_missing(new_place)
        and str(old_place).strip().lower() != str(new_place).strip().lower()
    ):
        for name in PERSON_FIELDS:
            if _is_missing(fresh.get(name)) and not _is_missing(saved.get(name)):
                fresh[name] = saved[name]
        return fresh
    merged = {**TripRequest().model_dump(), **saved}
    # A new one-word citizenship replaces an earlier list of the travellers.
    if not _is_missing(fresh.get("citizenship")) and _is_missing(fresh.get("party")):
        merged["party"] = []
    for name, value in fresh.items():
        if _is_missing(value):
            continue
        if name == "interests":
            merged[name] = [
                *merged.get(name, []),
                *(v for v in value if v not in merged.get(name, [])),
            ]
        else:
            merged[name] = value
    return merged


def missing_critical(trip: Mapping[str, object] | None) -> list[str]:
    """The PLAN_FIELDS still missing from a trip, in the order they are asked.

    citizenship, party and children are open only when party.py says so
    (e.g. children is not asked of one traveller, party only of a mixed group).
    """
    trip = trip or {}
    open_group = set(missing_group_fields(trip))
    if (
        trip.get("wants_hotels") is True
        and _is_missing(trip.get("budget"))
        and _is_missing(trip.get("room_budget"))
    ):
        open_group.add("budget")
    return [
        name
        for name in PLAN_FIELDS
        if (name in open_group)
        or (name not in CONDITIONAL_FIELDS and _is_missing(trip.get(name)))
    ]


# Asked only when another answer makes them needed (see missing_critical).
GROUP_FIELDS = ("citizenship", "party", "children")
CONDITIONAL_FIELDS = (*GROUP_FIELDS, "budget")


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def quick_reply_value(field: str | None, text: str) -> Any:
    """The value of a quick reply for `field`, or None if `text` is not one.

    Exact button labels match (any case or spacing); for days and travellers
    a bare number such as "3" matches too. Anything else goes to the
    understander.
    """
    if not field:
        return None
    wanted = _squash(text)
    for label, value in QUICK_REPLIES.get(field, ()):
        if _squash(label) == wanted:
            return value
    if field in _NUMBER_FIELDS and re.fullmatch(r"\d{1,2}", wanted):
        number = int(wanted)
        return number if number or field == "children" else None
    return None


# --- Helpers on the message history --------------------------------------------


def _text(message: AnyMessage) -> str:
    from ai_travel_assistant.llm_basics import content_to_text

    return content_to_text(message.content).strip()


def _latest_human_index(messages: Sequence[AnyMessage]) -> int | None:
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            return index
    return None


def agent_answered_before(messages: Sequence[AnyMessage]) -> bool:
    """True if the agent gave a real answer before the latest traveller message.

    Clarification questions and off-topic replies do not count.
    """
    end = _latest_human_index(messages) or 0
    return any(
        isinstance(m, AIMessage)
        and not m.tool_calls
        and m.name not in (CLARIFICATION_NAME, OFF_TOPIC_NAME)
        for m in messages[:end]
    )


def plan_given_before(state: Mapping) -> bool:
    """True if a real plan was given before the latest traveller message.

    A real plan is a checked itinerary (saved in the state, or a reply named
    ITINERARY_NAME), or, for agents without structured itineraries, an answer
    given while a trip was already being planned (a destination is known).
    Other answers never count: a weather question saves no trip, so "Plan 2
    days in Kandy" after it is a new plan to clarify, not a change (seen live
    on 2026-11-24, when "modify" skipped every question).
    """
    # Imported here: itinerary_output loads the budget and grounding code.
    from ai_travel_assistant.itinerary_output import ITINERARY_NAME

    if state.get("itinerary"):
        return True
    messages = state.get("messages", [])
    end = _latest_human_index(messages) or 0
    if any(
        isinstance(m, AIMessage) and m.name == ITINERARY_NAME for m in messages[:end]
    ):
        return True
    trip = state.get("trip_request") or {}
    return not _is_missing(trip.get("destination")) and agent_answered_before(messages)


def describe_trip(trip: Mapping[str, object] | None) -> str:
    """The known trip details as one short line, e.g. 'destination=Galle, days=2'."""
    parts = []
    for name, value in (trip or {}).items():
        if _is_missing(value):
            continue
        if name == "party":
            continue  # shown as describe_group() text instead
        if isinstance(value, list):
            value = ", ".join(str(v) for v in value)
        parts.append(f"{name}={value}")
    return ", ".join(parts)


# --- Nodes --------------------------------------------------------------------


def make_understand_request_node(
    understander: RequestUnderstander,
) -> Callable[[Mapping], dict]:
    """Build the node that classifies the latest message and merges its details."""

    def understand_request_node(state: Mapping) -> dict:
        messages = state.get("messages", [])
        index = _latest_human_index(messages)
        text = _text(messages[index]) if index is not None else ""
        question = state.get("clarification_context")
        resumed = bool(question)
        known = state.get("trip_request") or {}

        update: dict = {
            # The question belongs to this one reply; clear it once used.
            "clarification_context": None,
            "clarification_field": None,
            # A fresh message starts the count again; a reply continues it.
            "clarification_rounds": (
                state.get("clarification_rounds", 0) if resumed else 0
            ),
        }

        if not text:
            # Nothing to read: no LLM call. A blank reply to a question asks
            # again (the round limit still applies); otherwise the agent answers.
            if resumed:
                return {
                    **update,
                    "intent": "plan",
                    "missing_fields": missing_critical(known),
                }
            return {**update, "intent": None, "missing_fields": []}

        # A clicked button (or "3" after "How many days?"): no LLM call.
        field = state.get("clarification_field") if resumed else None
        value = quick_reply_value(field, text)
        if (
            field == "children"
            and value is not None
            and value > (known.get("travellers") or 0)
        ):
            value = None  # more children than travellers: let the LLM read it
        if value is not None:
            trip = {**TripRequest().model_dump(), **known, field: value}
            if field == "citizenship":
                trip["party"] = []
            return _plan_update(update, trip, "plan")

        context: dict = {}
        if question:
            context["clarification_context"] = question
        existing = {
            name: known[name]
            for name in PLAN_FIELDS
            if not _is_missing(known.get(name))
        }
        if existing:
            context["existing_data"] = existing
        has_plan = plan_given_before(state)
        if has_plan:
            context["has_plan"] = True

        try:
            understanding = understander(text, **context)
        except Exception as exc:
            # Stage 7: with the DAILY quota used up, the agent's call would
            # fail too, so end the turn now instead of spending another call.
            if is_quota_exhausted(exc):
                raise
            # Anything else (a 503, a bad reply): the agent answers as before.
            # Category and type only: provider error text can echo the request.
            logger.warning(
                "Request understanding failed: %s (%s); the agent answers "
                "without clarifying",
                error_category(exc),
                type(exc).__name__,
            )
            return {**update, "intent": None, "missing_fields": []}

        intent = understanding.intent
        # A reply to our question is part of planning, unless it is off-topic.
        if resumed and intent != "off_topic":
            intent = "plan"
        # Nothing to modify yet: treat it as a new plan.
        if intent == "modify" and not has_plan and not existing:
            intent = "plan"

        if intent in ("plan", "modify"):
            trip = merge_trip(known, code_decides_citizenship(understanding.trip))
            return _plan_update(update, trip, intent)
        return {**update, "intent": intent, "missing_fields": []}

    return understand_request_node


def code_decides_citizenship(trip: TripRequest) -> TripRequest:
    """Keep the LLM to words; the ticket category comes from party.py.

    The LLM may only say "mixed". A category it chose itself is dropped, and
    a named nationality ("Maldivian") is mapped by the fixed SAARC list. An
    unclear nationality ("tourist") sets nothing, so the agent asks.
    "mixed" is dropped when it contradicts the rest: a shared nationality, a
    listed party, or a single traveller (seen live: "just me; I'm Sri
    Lankan" came back as mixed).
    """
    mixed = (
        trip.citizenship == "mixed"
        and not trip.nationality
        and not trip.party
        and trip.travellers != 1
    )
    citizenship = "mixed" if mixed else None
    if trip.nationality and citizenship is None:
        group = nationality_group(trip.nationality)
        citizenship = None if group == "unknown" else group
    return trip.model_copy(update={"citizenship": citizenship})


def _plan_update(update: dict, trip: dict, intent: str) -> dict:
    """Finish a plan or modify turn: record the trip and what is still missing."""
    trip = complete_party(trip)
    return {
        **update,
        "intent": intent,
        "trip_request": trip,
        "missing_fields": missing_critical(trip) if intent == "plan" else [],
    }


def route_after_understanding(state: Mapping) -> Route:
    """Pick the next step from the understanding (pure Python, no LLM)."""
    intent = state.get("intent")
    if intent == "off_topic":
        return TO_OFF_TOPIC
    if (
        intent == "plan"
        and state.get("missing_fields")
        and state.get("clarification_rounds", 0) < MAX_CLARIFICATIONS
    ):
        return TO_CLARIFY
    return TO_AGENT


def clarification_question(missing: Sequence[str], trip: Mapping | None) -> str:
    """Ask for the first missing field only, using what is already known."""
    trip = trip or {}
    field = missing[0] if missing else "destination"
    place, days = trip.get("destination"), trip.get("days")
    if field == "destination":
        if days:
            return f"Where in Sri Lanka would you like to spend your {days} days?"
        return "Where in Sri Lanka would you like to go?"
    if field == "days":
        return f"How many days would you like to spend in {place}?"
    if field == "travellers":
        return (
            "How many people are travelling, including you? Prices are per "
            "person, so this sets the cost for your group."
        )
    if field == "citizenship":
        return (
            "What citizenship do the travellers have? Many sites charge Sri "
            "Lankan, SAARC and other foreign citizens different ticket prices "
            "(citizenship counts, not where you live)."
        )
    if field == "party":
        return (
            "Please list who is travelling, with each person's citizenship and "
            "whether they are a child, e.g. '2 Sri Lankan adults, 1 Indian "
            "adult, 1 German child'. For a dual citizen, say which passports "
            "they hold."
        )
    if field == "children":
        group = trip.get("travellers")
        return (
            f"How many of the {group} travellers are children? "
            "Child ticket rules differ from site to site."
        )
    if field == "wants_hotels":
        return (
            "Would you like hotel suggestions for your stay? I can't book rooms "
            "or check availability, but I can list places from the travel "
            "guide with their last listed prices."
        )
    if field == "budget":
        return (
            "What's your budget for a room per night? Pick a level, or type an "
            "amount, e.g. 'LKR 10,000'."
        )
    return f"Where will you start your trip to {place} from?"


def ask_clarification_node(state: Mapping) -> dict:
    """Write the clarification question into the conversation (no LLM call).

    clarification_field records which field it asks for, so the page can show
    that field's QUICK_REPLIES and a clicked one skips the understander.
    """
    missing = state.get("missing_fields", [])
    question = clarification_question(missing, state.get("trip_request"))
    return {
        "messages": [AIMessage(question, name=CLARIFICATION_NAME)],
        "clarification_rounds": state.get("clarification_rounds", 0) + 1,
        "clarification_field": missing[0] if missing else None,
    }


def wait_for_reply_node(state: Mapping) -> dict:
    """Pause the run until the traveller answers the question.

    interrupt() saves the state and stops the run; its value (the question) is
    what the caller shows. Command(resume=reply) re-runs this node from the top
    and interrupt() then returns the reply. Nothing happens before interrupt(),
    so re-running it is safe.
    """
    question = next(
        (
            _text(m)
            for m in reversed(state.get("messages", []))
            if isinstance(m, AIMessage) and m.name == CLARIFICATION_NAME
        ),
        "",
    )
    reply = interrupt(question)
    return {
        "messages": [HumanMessage(reply if isinstance(reply, str) else str(reply))],
        "clarification_context": question or None,
    }


def respond_off_topic_node(state: Mapping) -> dict:
    """A fixed, polite redirect (no LLM call)."""
    return {"messages": [AIMessage(OFF_TOPIC_REPLY, name=OFF_TOPIC_NAME)]}


def trip_prompt_note(trip: Mapping | None, missing: Sequence[str]) -> str:
    """What the agent's system prompt adds about the current trip ('' if none)."""
    known = describe_trip(trip)
    if not known and not missing:
        return ""
    note = ""
    if known:
        note += (
            f" Current trip request, gathered from the conversation: {known}. "
            "Plan for these details."
        )
    group = describe_group(trip)
    if group:
        note += (
            f" Who is travelling: {group}. Ticket prices depend on citizenship: "
            "use each category's own price and never one category's price for "
            "another; if a category's price is not given, say it is not "
            "verified. Give each cost per person and for the whole group, and "
            "end the plan with the total for the group."
        )
    wants_hotels = (trip or {}).get("wants_hotels")
    if wants_hotels is True:
        note += (
            " The traveller wants hotel suggestions: when gathering data for "
            "the plan, also call suggest_hotels for the destination. Name only "
            "places it returns, and say you cannot book or check availability."
        )
    elif wants_hotels is False:
        note += " The traveller does not want hotel suggestions; do not add any."
    categories = {category for category, _, _ in group_breakdown(trip)}
    if categories == {"local"}:
        note += (
            " Everyone is a Sri Lankan citizen: give prices in LKR only and do "
            "not call convert_currency. Skip visa advice."
        )
    elif categories - {"local"}:
        note += (
            " Some travellers are foreign citizens: give their prices in LKR and "
            "USD. Add one line asking them to check the current visa (ETA) rules "
            "on the official Sri Lanka immigration website; give no visa "
            "details yourself."
        )
    if missing:
        note += (
            f" Still unknown after asking: {', '.join(missing)}. If the "
            "destination is unknown, suggest places with search_destinations "
            "instead of choosing one; if the number of days is unknown, assume "
            "3 days. If the group's citizenship is unknown, show prices in LKR "
            "and USD, and say that Sri Lankan, SAARC and other foreign "
            "visitors may pay different fees. State every assumption."
        )
    return note
