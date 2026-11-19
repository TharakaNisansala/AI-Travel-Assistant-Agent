"""Stage 6: understanding the request, and asking when a critical detail is missing.

What this file demonstrates:
  1. Intent classification with structured output. One LLM call reads the
     traveller's latest message and returns a RequestUnderstanding: what the
     message is for (plan, modify, question, off_topic) and the trip details it
     states. Routing then reads that from state; it never calls the LLM.
  2. Merging requirements across turns. "Plan me a trip" then "Galle" then "3"
     builds one TripRequest; a missing value never erases a saved one.
  3. Asking versus assuming. Only a PLAN request with a missing PLAN_FIELDS
     value (destination, days, traveller_type, origin) makes the agent ask,
     one field per question. A question without a place ("Where can I
     surf?") goes to the agent, which can search for places.
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
  7. The traveller type (local or foreign) is a fact about the person, not
     the trip: it is kept when the destination changes, saved to the Store
     when the graph has one, and filled in from it on later conversations.

The nodes work on the agent's state (tool_calling.AgentState). The graph wiring
lives in tool_calling.build_tool_agent_graph(..., understander=...).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, Protocol

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.types import interrupt

from ai_travel_assistant.long_term_memory import save_traveller_type
from ai_travel_assistant.robustness import is_quota_exhausted
from ai_travel_assistant.schemas import (
    PLAN_FIELDS,
    RequestUnderstanding,
    TripRequest,
)

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
# per PLAN_FIELDS value, so "Plan me a trip" can be completed by four clicks.
MAX_CLARIFICATIONS = 4

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
    "traveller_type": (
        ("Sri Lankan / I live here", "local"),
        ("Visiting from abroad", "foreign"),
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


# Details about the traveller rather than the trip: a new destination keeps them.
PERSON_FIELDS = ("traveller_type",)


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
    """The PLAN_FIELDS still missing from a trip, in the order they are asked."""
    trip = trip or {}
    return [name for name in PLAN_FIELDS if _is_missing(trip.get(name))]


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def quick_reply_value(field: str | None, text: str) -> Any:
    """The value of a quick reply for `field`, or None if `text` is not one.

    Exact button labels match (any case or spacing); for days a bare number
    such as "3" matches too. Anything else goes to the understander.
    """
    if not field:
        return None
    wanted = _squash(text)
    for label, value in QUICK_REPLIES.get(field, ()):
        if _squash(label) == wanted:
            return value
    if field == "days" and re.fullmatch(r"[1-9]\d?", wanted):
        return int(wanted)
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


def describe_trip(trip: Mapping[str, object] | None) -> str:
    """The known trip details as one short line, e.g. 'destination=Galle, days=2'."""
    parts = []
    for name, value in (trip or {}).items():
        if _is_missing(value):
            continue
        if isinstance(value, list):
            value = ", ".join(str(v) for v in value)
        parts.append(f"{name}={value}")
    return ", ".join(parts)


# --- Nodes --------------------------------------------------------------------


def make_understand_request_node(
    understander: RequestUnderstander,
) -> Callable[[Mapping], dict]:
    """Build the node that classifies the latest message and merges its details."""

    def understand_request_node(state: Mapping, runtime: Any = None) -> dict:
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
        if value is not None:
            trip = {**TripRequest().model_dump(), **known, field: value}
            return _plan_update(update, trip, "plan", state, runtime)

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
        has_plan = agent_answered_before(messages)
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
            return {**update, "intent": None, "missing_fields": []}

        intent = understanding.intent
        # A reply to our question is part of planning, unless it is off-topic.
        if resumed and intent != "off_topic":
            intent = "plan"
        # Nothing to modify yet: treat it as a new plan.
        if intent == "modify" and not has_plan and not existing:
            intent = "plan"

        if intent in ("plan", "modify"):
            trip = merge_trip(known, understanding.trip)
            return _plan_update(update, trip, intent, state, runtime)
        return {**update, "intent": intent, "missing_fields": []}

    return understand_request_node


def _plan_update(
    update: dict, trip: dict, intent: str, state: Mapping, runtime: Any
) -> dict:
    """Finish a plan or modify turn: fill in and save the traveller type,
    then record the trip and what is still missing."""
    saved_type = state.get("saved_traveller_type")
    if _is_missing(trip.get("traveller_type")) and saved_type:
        trip = {**trip, "traveller_type": saved_type}
    new_type = trip.get("traveller_type")
    store = getattr(runtime, "store", None)
    if new_type and new_type != saved_type and store is not None:
        context = getattr(runtime, "context", None)
        user_id = getattr(context, "user_id", None) or "local"
        save_traveller_type(store, user_id, new_type)
        update["saved_traveller_type"] = new_type
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
    if field == "traveller_type":
        return (
            "Are you Sri Lankan or living in Sri Lanka, or visiting from abroad? "
            "Entry fees and prices are often different for each."
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
    kind = (trip or {}).get("traveller_type")
    if kind == "local":
        note += (
            " The traveller is Sri Lankan or lives in Sri Lanka: give prices in "
            "LKR only and do not call convert_currency. Where the guide lists "
            "separate local and foreigner prices, use the local one. Skip visa "
            "advice."
        )
    elif kind == "foreign":
        note += (
            " The traveller is visiting from abroad: give prices in LKR and USD. "
            "Where the guide lists separate foreigner prices, use those. Add one "
            "line asking them to check the current visa (ETA) rules on the "
            "official Sri Lanka immigration website; give no visa details "
            "yourself."
        )
    if missing:
        note += (
            f" Still unknown after asking: {', '.join(missing)}. If the "
            "destination is unknown, suggest places with search_destinations "
            "instead of choosing one; if the number of days is unknown, assume "
            "3 days. If the traveller type is unknown, show prices in LKR and "
            "USD and say that locals and foreigners often pay different fees. "
            "State every assumption."
        )
    return note
