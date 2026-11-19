"""Stage 3: a tool-calling agent (the ReAct loop) with LangGraph and Gemini.

What this file demonstrates:
  1. bind_tools(): telling the LLM which tools exist.
  2. MessagesState: a state that is just the growing list of chat messages.
  3. An LLM node that either answers or asks for a tool to be run (a tool call).
  4. ToolNode: runs the requested tools and adds their results as ToolMessages.
  5. tools_condition: routes to the tools if the LLM asked for one, else ends.
  6. The loop tools -> llm, so the LLM sees the tool result and can answer.

Flow:

    START -> llm --(AIMessage has tool_calls?)--> tools -> llm   (loop)
                 \\--(no tool_calls: final answer)--> END

Reason (llm decides) -> Act (tools runs) -> Observe (llm reads the result) ...
until the LLM replies without a tool call.

The model is passed in, so the same graph runs with Gemini in the app and with a
scripted fake model in the offline tests.

Run from the project root (usually 2 Gemini calls: one to choose the tool, one
to write the answer):
    uv run python -m ai_travel_assistant.tool_calling "Find places for hiking"
    uv run python -m ai_travel_assistant.tool_calling --steps "Where can I surf?"
    uv run python -m ai_travel_assistant.tool_calling --steps \\
        "What is the weather forecast for Ella?"
    uv run python -m ai_travel_assistant.tool_calling --steps \\
        "What nature spots can I see around Ella?"

Stage 4 adds get_weather_forecast, which calls the real Open-Meteo API (2 HTTP
requests, no API key). The graph does not change: a new tool only needs to be
added to TOOLS, and Gemini decides from its docstring when to call it.
find_attractions (OpenStreetMap via Overpass, 2 HTTP requests) is added the
same way. With three tools, SYSTEM_PROMPT says how they relate: search to choose
a place, attractions for what to see at a named place, weather for forecasts.

The rest of the Stage 4 suite is registered the same way:
get_destination_guide (Wikivoyage background), estimate_travel_time (OSRM road
distance and driving time) and convert_currency (exchange rates). Six tools now
overlap in places (guide "see_and_do" vs find_attractions), so SYSTEM_PROMPT
spells out which tool answers which kind of question.

For multi-day itineraries, SYSTEM_PROMPT adds a generic plan: gather guide,
attractions, weather and travel time in one step; optionally a few travel
times and currency conversions in a second; then write the plan day by day
from tool results only, with sources, a stated default origin (DEFAULT_ORIGIN)
and a "Not available" list. Usually 3-4 Gemini calls per itinerary.

Stage 5 adds short-term memory. build_tool_agent_graph(model, checkpointer=...)
compiles the same loop with a checkpointer, which saves the message history
after every step under a thread_id. ask_agent() sends one message on a thread:
LangGraph loads that thread's saved history first and the add_messages reducer
appends the new message, so a follow-up ("make day 2 easier") sees the earlier
plan. Without a checkpointer every run starts a fresh conversation, as before.
run_agent_chat() is the terminal loop around ask_agent, with /new for a fresh
thread (each message makes the usual 2-4 Gemini calls):
    uv run python -m ai_travel_assistant.tool_calling --chat
    uv run python -m ai_travel_assistant.tool_calling --chat --steps
The chat saves its checkpoints in a SQLite file (DEFAULT_CHAT_DB), so a
conversation survives a restart; reopen it with the id printed at the start:
    uv run python -m ai_travel_assistant.tool_calling --chat --thread <id>
The chat also has long-term memory (long_term_memory.py): load_memory reads the
traveller's saved preferences before the LLM, and save_memory stores new ones
after the answer, but only spends a Gemini call when a message looks like a
preference ("I'm vegetarian"). /prefs and /forget show or clear them; --user
picks the traveller:
    uv run python -m ai_travel_assistant.tool_calling --chat --user tharaka

Stage 6 adds request understanding (request_understanding.py). In --chat every
message is first read by one extra Gemini call: what it is for (plan, modify,
question, off_topic) and which trip details it gives. A plan without a
destination or number of days makes the agent ask, and the chat pauses with
interrupt() until the traveller answers; off-topic requests get a fixed reply
without calling the agent. --no-clarify turns this off to save quota (the
graph is then the Stage 5 one):
    uv run python -m ai_travel_assistant.tool_calling --chat --no-clarify

Stage 7 makes the loop robust (robustness.py, faults.py). A tool that raises
becomes an error result for that call only; the LLM sees a repaired history
(tool calls left without results by a crash are closed); route_after_agent
replaces tools_condition and stops a turn after MAX_TOOL_ROUNDS tool rounds,
ending it with one wrap-up call. In --chat, transient Gemini errors are
retried once by the llm node's RetryPolicy (the client itself does not retry,
and 429s are never retried), and failures are shown as short, safe messages.
Try an outage without breaking anything real (PowerShell):
    $env:FAULT_INJECT = "weather"
    uv run python -m ai_travel_assistant.tool_calling --chat

Stage 8 adds structured itineraries (itinerary_output.py,
itinerary_grounding.py). In --chat and one-shot mode the agent finishes an
itinerary by calling submit_itinerary; compose_answer validates it, checks
every factual value against the tool results for that destination, and
renders Markdown, all without an extra Gemini call. An invalid submission
is sent back once, then salvaged. /itinerary (chat) or --json (one-shot)
prints the checked plan as JSON:
    uv run python -m ai_travel_assistant.tool_calling --json "Plan 2 days in Ella"
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore
from langgraph.types import Command, RetryPolicy

from ai_travel_assistant import faults
from ai_travel_assistant.context_window import MAX_HISTORY_TOKENS, history_for_llm
from ai_travel_assistant.itinerary_grounding import SUBMIT_TOOL
from ai_travel_assistant.itinerary_output import (
    ITINERARY_NAME,
    ITINERARY_RULE,
    compose_itinerary,
    has_submit_call,
    make_compose_node,
    route_after_compose,
    submit_itinerary,
)
from ai_travel_assistant.long_term_memory import (
    PreferenceExtractor,
    clear_preferences,
    load_preferences,
    looks_like_preference,
    make_llm_preference_extractor,
    save_preferences,
    sqlite_store,
)
from ai_travel_assistant.request_understanding import (
    ASK_CLARIFICATION,
    CLARIFICATION_NAME,
    OFF_TOPIC_NAME,
    RESPOND_OFF_TOPIC,
    TO_AGENT,
    TO_CLARIFY,
    TO_OFF_TOPIC,
    UNDERSTAND_REQUEST,
    WAIT_FOR_REPLY,
    RequestUnderstander,
    ask_clarification_node,
    make_llm_request_understander,
    make_understand_request_node,
    respond_off_topic_node,
    route_after_understanding,
    trip_prompt_note,
    wait_for_reply_node,
)
from ai_travel_assistant.robustness import (
    FALLBACK_REPLY,
    MAX_TOOL_ROUNDS,
    TO_COMPOSE,
    TO_DONE,
    TO_FALLBACK,
    TO_TOOLS,
    WRAP_UP_INSTRUCTION,
    close_pending_calls,
    describe_llm_error,
    llm_retry_policy,
    make_route_after_agent,
    repair_history,
    tool_error_message,
)
from ai_travel_assistant.tools import (
    convert_currency,
    estimate_travel_time,
    find_attractions,
    get_destination_guide,
    get_weather_forecast,
    search_destinations,
)

if TYPE_CHECKING:  # only for the annotation; llm_basics loads Gemini code
    from ai_travel_assistant.llm_basics import GeminiSettings

# The tools this agent may use. bind_tools() sends all of them to the LLM, and
# ToolNode can run any of them (and only them).
TOOLS: tuple[BaseTool, ...] = (
    search_destinations,
    get_weather_forecast,
    find_attractions,
    get_destination_guide,
    estimate_travel_time,
    convert_currency,
)

# Node names. "tools" is the name tools_condition returns, so the tool node must
# be called exactly that (or we would need a mapping in add_conditional_edges).
LLM = "llm"
TOOLS_NODE = "tools"

# Where a trip starts when the traveller does not say: the main entry point to
# Sri Lanka (international airport and rail hub). The agent must say it assumed
# this. It is the only place name in SYSTEM_PROMPT; destinations never appear.
DEFAULT_ORIGIN = "Colombo"

# Each tool's docstring says what that tool does; this prompt says how the
# tools relate, so the LLM can choose between overlapping ones.
SYSTEM_PROMPT = (
    "You are a friendly Sri Lanka travel assistant. Choose tools like this. "
    # Choosing WHERE to go.
    "When the traveller has not chosen a place yet and asks where to go or which "
    "places suit an activity, call search_destinations. Recommend only places it "
    "returned, and mention that they come from the built-in guide. If it finds "
    "nothing, say so and suggest a broader search. "
    # Any place in Sri Lanka, not only the built-in guide's.
    "The built-in guide is only for choosing where to go: when the traveller "
    "names any place in Sri Lanka, even one it does not list, use the other "
    "tools for that place. If the tools say a place was not found in Sri "
    "Lanka, ask the traveller to check the spelling or name the nearest town, "
    "and do not plan for it. "
    "If they also want to know what to see, call find_attractions for the one "
    "place that fits best, not for every result. "
    # WHAT to see at a named place.
    "When the traveller names a place and asks what to see or do there, call "
    "find_attractions with just the place name and a category matching their "
    "interests: nature for waterfalls, peaks or viewpoints; culture for temples, "
    "museums or history; landmarks for well-known sights; all when unsure or "
    "mixed. Mention only attractions it returned, and note that distances are "
    "straight-line. If it finds nothing for a specific category, you may retry "
    "once with category all. It does not cover hotels, restaurants or hiking "
    "routes; say so instead of inventing them. "
    # Weather.
    "When the traveller asks about the weather, rain, temperature or what to "
    "pack, call get_weather_forecast with just the place name and base your "
    "answer only on the forecast it returns, using its units. "
    # Background from the travel guide.
    "When the traveller asks what a named place is like, how to get there or "
    "around, where to stay, or for practical or safety tips, call "
    "get_destination_guide with the place name and the matching topic "
    "(overview, getting_there, getting_around, see_and_do, accommodation, "
    "safety; all when planning a whole visit). Name only places to stay that "
    "the guide mentions, never invent hotels, and say you cannot check "
    "availability or book. "
    "Mention that the guide is from Wikivoyage and may be out of date. For a "
    "list of nearby sights with distances, prefer find_attractions. "
    # Distances between places.
    "When the traveller asks how far apart two places are or how long the "
    "journey takes, call estimate_travel_time with the two place names, and "
    "say the time is a car estimate without traffic. "
    # Money.
    "When the traveller asks about exchange rates or what an amount is worth "
    "in another currency, call convert_currency with ISO codes such as USD or "
    "LKR, and quote its result; never use a remembered rate or do the "
    "conversion yourself. "
    # Combining tools and handling failures.
    "When planning a visit to a named place, you may call several of these "
    "tools in the same step. If a tool says its data is "
    "unavailable, do not call it again; tell the traveller plainly and never "
    "guess the weather or invent places. "
    # Multi-day itineraries. Generic on purpose: no place names, so the same
    # rule works for every destination. Colombo is only the default ORIGIN.
    "When the traveller asks for an itinerary or a plan of one or more days "
    "at a named place, first gather real data in one step: "
    "get_destination_guide (topic all), find_attractions (category all) and "
    "get_weather_forecast for that place, and estimate_travel_time from the "
    "traveller's starting point to it. If they did not say where they start, "
    f"use {DEFAULT_ORIGIN} and state in the plan that you assumed it. "
    "In the next step you may call estimate_travel_time for at most two "
    "nearby places the guide suggests visiting from there, and "
    "convert_currency for at most three prices the guide quotes (one of them "
    "a place-to-stay price, if the guide gives one), so the plan shows each "
    "in both LKR and USD. "
    "Then write the plan day by day, using only sights, travel times, "
    "weather and prices from the tool results, and say where each comes "
    "from (OpenStreetMap, Wikivoyage, OSRM, Open-Meteo or the exchange rate "
    "service). If find_attractions returns a Wikivoyage fallback, use only "
    "the sights it names. Give weather only for the days the forecast "
    "covers and say no forecast is available for the others. End with a "
    "short 'Not available' list of what the tools could not provide, such "
    "as room availability, food prices, travel times between sights inside "
    "a town, or "
    "any tool that failed. "
    "Keep answers short; an itinerary may be longer but stays to the point."
)


# --- State and context ----------------------------------------------------------

# Who is chatting when nobody says: one local traveller on this machine.
DEFAULT_USER = "local"

# Stage 7: where a turn goes when it hits the tool-round cap.
FALLBACK_RESPONSE = "fallback_response"
# Stage 8: where a submit_itinerary call is checked and rendered (no LLM).
COMPOSE_ANSWER = "compose_answer"

# Node names for long-term memory (only added when the graph gets a Store).
LOAD_MEMORY = "load_memory"
SAVE_MEMORY = "save_memory"


class AgentState(MessagesState):
    """The messages, plus the traveller's saved preferences for this turn.

    The other keys are used only when the graph has an understander (Stage 6);
    nodes read them with .get(), so threads saved before Stage 6 still load.
    """

    user_prefs: list[str]
    trip_request: dict  # merged trip details (a TripRequest as a plain dict)
    intent: str | None  # plan, modify, question, off_topic; None if unknown
    missing_fields: list[str]  # PLAN_FIELDS still missing for a plan
    clarification_context: str | None  # the question the latest reply answers
    clarification_field: str | None  # the field that question asks for
    clarification_rounds: int  # questions asked in a row this turn
    itinerary: dict  # Stage 8: the last checked itinerary (JSON-ready data)


@dataclass
class AgentContext:
    """Fixed for a whole run, so it is runtime CONTEXT, not state.

    Passed as graph.invoke(..., context=AgentContext(user_id=...)); nodes read
    it as runtime.context. In a real app this would come from the login.
    """

    user_id: str = DEFAULT_USER


def system_prompt_for(
    prefs: Sequence[str],
    trip: dict | None = None,
    missing: Sequence[str] = (),
    structured: bool = False,
) -> str:
    """SYSTEM_PROMPT, plus saved preferences and the current trip, if any.

    The trip details come from understand_request (Stage 6). Without them the
    prompt is exactly the Stage 5 one. structured=True (Stage 8) adds the rule
    to finish itineraries with submit_itinerary.
    """
    prompt = SYSTEM_PROMPT
    if prefs:
        prompt += (
            " Known traveller preferences from earlier "
            f"conversations: {'; '.join(prefs)}. Respect them in every plan, and "
            "mention briefly when one changes a suggestion."
        )
    prompt += trip_prompt_note(trip, missing)
    return prompt + ITINERARY_RULE if structured else prompt


def _user_id(runtime: Runtime[AgentContext]) -> str:
    return runtime.context.user_id if runtime.context else DEFAULT_USER


# --- Nodes --------------------------------------------------------------------


def make_llm_node(
    llm_with_tools: BaseChatModel,
    max_history_tokens: int | None = MAX_HISTORY_TOKENS,
    structured_itinerary: bool = False,
) -> Callable[[AgentState], dict]:
    """Build the llm node around a model that already has the tools bound."""

    def llm_node(state: AgentState) -> dict:
        # The system prompt is added on every call but never stored in the
        # state, so the message history holds only the real conversation.
        prompt = system_prompt_for(
            state.get("user_prefs", []),
            state.get("trip_request"),
            state.get("missing_fields", []),
            structured=structured_itinerary,
        )
        # Only what the LLM sees is trimmed; the state keeps every message.
        # Stage 7: repair_history() then closes tool calls left without a
        # result by an earlier failure, which the model API would reject.
        # It builds a new list; the saved thread is never changed.
        history = repair_history(history_for_llm(state["messages"], max_history_tokens))
        messages = [SystemMessage(prompt), *history]
        response = llm_with_tools.invoke(messages)
        # MessagesState uses the add_messages reducer: returning a list APPENDS
        # it to the history instead of replacing the history.
        return {"messages": [response]}

    return llm_node


def make_fallback_node(
    llm_with_tools: BaseChatModel,
    max_history_tokens: int | None = MAX_HISTORY_TOKENS,
    structured_itinerary: bool = False,
    today: Callable[[], date] = date.today,
) -> Callable[[AgentState], dict]:
    """Build the node that ends a turn that hit the tool-round cap.

    1. The tool calls the cap stopped get "not run" results, so no call is
       left without a result in the saved thread.
    2. One wrap-up LLM call answers from the results gathered so far.
    3. If that call fails, or asks for tools again, a fixed reply is used.
    Stage 8: a wrap-up that submits an itinerary is checked and rendered by
    the same compose step (no repair: the turn has no rounds left).
    """

    def fallback_node(state: AgentState) -> dict:
        messages = state["messages"]
        last = messages[-1] if messages else None
        closed = (
            close_pending_calls(last)
            if isinstance(last, AIMessage) and last.tool_calls
            else []
        )
        prompt = system_prompt_for(
            state.get("user_prefs", []),
            state.get("trip_request"),
            state.get("missing_fields", []),
            structured=structured_itinerary,
        )
        history = repair_history(
            history_for_llm([*messages, *closed], max_history_tokens)
        )
        try:
            reply = llm_with_tools.invoke(
                [SystemMessage(f"{prompt} {WRAP_UP_INSTRUCTION}"), *history]
            )
        except Exception:  # the wrap-up is best effort; never crash the turn
            reply = None
        if structured_itinerary and has_submit_call(reply):
            result = compose_itinerary(
                reply,
                [*messages, *closed, reply],
                today=today(),
                trip_request=state.get("trip_request"),
                allow_repair=False,
            )
            update: dict = {"messages": [*closed, reply, *result.messages]}
            if result.itinerary is not None:
                update["itinerary"] = result.itinerary
            return update
        if not isinstance(reply, AIMessage) or reply.tool_calls or not reply.content:
            reply = AIMessage(FALLBACK_REPLY)
        return {"messages": [*closed, reply]}

    return fallback_node


def load_memory_node(state: AgentState, runtime: Runtime[AgentContext]) -> dict:
    """Read the traveller's preferences from the Store (no LLM call)."""
    return {"user_prefs": load_preferences(runtime.store, _user_id(runtime))}


def make_save_memory_node(
    extractor: PreferenceExtractor,
) -> Callable[[AgentState, Runtime[AgentContext]], dict]:
    """Build the node that stores new durable preferences after a turn."""

    def save_memory_node(state: AgentState, runtime: Runtime[AgentContext]) -> dict:
        from ai_travel_assistant.llm_basics import content_to_text

        last_human = next(
            (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            None,
        )
        if last_human is None:
            return {}
        text = content_to_text(last_human.content)
        # The gate: most messages are trip requests, and those cost no call.
        if not looks_like_preference(text):
            return {}
        known = state.get("user_prefs", [])
        try:
            found = extractor(text, known)
        except Exception:  # e.g. a Gemini 429: keep the answer, skip saving
            return {}
        added = save_preferences(runtime.store, _user_id(runtime), found)
        return {"user_prefs": [*known, *added]} if added else {}

    return save_memory_node


# --- Graph --------------------------------------------------------------------


def build_tool_agent_graph(
    model: BaseChatModel,
    tools: Sequence[BaseTool] = TOOLS,
    checkpointer: BaseCheckpointSaver | None = None,
    store: BaseStore | None = None,
    preference_extractor: PreferenceExtractor | None = None,
    max_history_tokens: int | None = MAX_HISTORY_TOKENS,
    understander: RequestUnderstander | None = None,
    max_tool_rounds: int = MAX_TOOL_ROUNDS,
    llm_retry: RetryPolicy | None = None,
    structured_itinerary: bool = False,
    today: Callable[[], date] = date.today,
    persistence_provided: bool = False,
) -> CompiledStateGraph:
    """Bind the tools to the model and wire the ReAct loop.

    bind_tools() does not call the API. It returns a wrapped model that sends
    the tools' names, descriptions and argument schemas with every request.

    With a checkpointer, every invoke must name a thread_id (see ask_agent),
    and each thread keeps its own message history between invokes.

    With a store, the graph gets long-term memory:
        START -> load_memory -> llm <-> tools -> save_memory -> END
    load_memory reads the traveller's preferences; save_memory (added only
    with a preference_extractor) stores new ones after the final answer.

    Every LLM call sees the current turn whole, plus earlier turns compacted
    and trimmed to max_history_tokens (None: send everything).

    With an understander (Stage 6), every message is understood first:
        ... -> understand_request --off_topic--> respond_off_topic -> (save) END
                                  --clarify--> ask_clarification -> wait_for_reply
                                  --agent----> llm <-> tools -> ...
    wait_for_reply pauses with interrupt() and loops back to understand_request
    with the reply. Interrupts need a saved state, so an understander requires
    a checkpointer. Without an understander the graph is the Stage 5 one.

    Stage 7: the LLM may run tools max_tool_rounds times per turn; asking once
    more sends the turn to fallback_response (one wrap-up call). A tool that
    raises becomes an error result for that call only. llm_retry (e.g.
    robustness.llm_retry_policy()) retries the llm node on transient Gemini
    errors; it is the ONLY retry layer for Gemini, so build the model with
    max_retries=0 when passing it. None: a failed call fails the turn.

    Stage 8, structured_itinerary=True: the model also gets submit_itinerary
    (the final-answer tool; never run by the tools node). A call to it goes to
    compose_answer, which validates, grounds and renders the plan without an
    LLM call; invalid output goes back to the llm once, then is salvaged.
    today is the clock used to drop past forecast dates (tests fix it).
    Without it the graph and the prompt are the Stage 7 ones.
    """
    # Stage 10: a server such as LangGraph Studio attaches its own checkpointer
    # after the graph is built (persistence_provided=True).
    if understander is not None and checkpointer is None and not persistence_provided:
        raise ValueError("an understander needs a checkpointer to pause and resume")
    bound = [*tools, submit_itinerary] if structured_itinerary else list(tools)
    llm_with_tools = model.bind_tools(bound)

    builder = StateGraph(AgentState, context_schema=AgentContext)
    # Only the llm node is retried: it just reads the state and calls the
    # model, so running it again is safe. Tools retry inside their services.
    builder.add_node(
        LLM,
        make_llm_node(llm_with_tools, max_history_tokens, structured_itinerary),
        retry_policy=llm_retry,
    )
    # ToolNode looks at the last AIMessage, runs every tool call in it (several
    # calls run in parallel), and returns one ToolMessage per call. Stage 7:
    # a tool that raises gives an error ToolMessage for THAT call only (the
    # other calls keep their results), instead of crashing the whole turn.
    builder.add_node(
        TOOLS_NODE, ToolNode(list(tools), handle_tool_errors=tool_error_message)
    )

    # After an answer the turn ends; with save_memory it goes there first.
    after_answer = END
    if store is not None and preference_extractor is not None:
        builder.add_node(SAVE_MEMORY, make_save_memory_node(preference_extractor))
        builder.add_edge(SAVE_MEMORY, END)
        after_answer = SAVE_MEMORY

    first = LLM
    if understander is not None:
        builder.add_node(UNDERSTAND_REQUEST, make_understand_request_node(understander))
        builder.add_node(ASK_CLARIFICATION, ask_clarification_node)
        builder.add_node(WAIT_FOR_REPLY, wait_for_reply_node)
        builder.add_node(RESPOND_OFF_TOPIC, respond_off_topic_node)
        builder.add_conditional_edges(
            UNDERSTAND_REQUEST,
            route_after_understanding,
            {
                TO_AGENT: LLM,
                TO_CLARIFY: ASK_CLARIFICATION,
                TO_OFF_TOPIC: RESPOND_OFF_TOPIC,
            },
        )
        builder.add_edge(ASK_CLARIFICATION, WAIT_FOR_REPLY)
        builder.add_edge(WAIT_FOR_REPLY, UNDERSTAND_REQUEST)
        # Off-topic still passes save_memory: its free keyword gate catches a
        # preference ("I'm vegetarian") the understander might misread.
        builder.add_edge(RESPOND_OFF_TOPIC, after_answer)
        first = UNDERSTAND_REQUEST

    if store is None:
        builder.add_edge(START, first)
    else:
        builder.add_node(LOAD_MEMORY, load_memory_node)
        builder.add_edge(START, LOAD_MEMORY)
        builder.add_edge(LOAD_MEMORY, first)
    # Like tools_condition (tool calls -> tools, else done), plus the cap.
    builder.add_node(
        FALLBACK_RESPONSE,
        make_fallback_node(
            llm_with_tools, max_history_tokens, structured_itinerary, today
        ),
    )
    routes = {
        TO_TOOLS: TOOLS_NODE,
        TO_FALLBACK: FALLBACK_RESPONSE,
        TO_DONE: after_answer,
    }
    if structured_itinerary:
        # The submit call is never run by the tools node: compose_answer checks
        # it and either answers or sends the field errors back to the llm once.
        builder.add_node(COMPOSE_ANSWER, make_compose_node(today))
        builder.add_conditional_edges(
            COMPOSE_ANSWER, route_after_compose, {"repair": LLM, "done": after_answer}
        )
        routes[TO_COMPOSE] = COMPOSE_ANSWER
    builder.add_conditional_edges(
        LLM,
        make_route_after_agent(
            max_tool_rounds, final_tool=SUBMIT_TOOL if structured_itinerary else None
        ),
        routes,
    )
    builder.add_edge(FALLBACK_RESPONSE, after_answer)
    # The loop: after a tool runs, the LLM reads its result and decides again.
    builder.add_edge(TOOLS_NODE, LLM)
    return builder.compile(checkpointer=checkpointer, store=store)


def run_tool_agent(graph: CompiledStateGraph, question: str) -> list[AnyMessage]:
    """Ask one question and return the full message history of the run."""
    state = graph.invoke({"messages": [HumanMessage(question)]})
    return state["messages"]


def ask_agent(
    graph: CompiledStateGraph,
    thread_id: str,
    text: str,
    user_id: str = DEFAULT_USER,
) -> list[AnyMessage]:
    """Send one message on a thread and return only this turn's messages.

    The graph must be built with a checkpointer. The saved history of the
    thread is loaded first, so the LLM sees the earlier turns; the returned
    list starts at the new HumanMessage (tool calls, results, final answer).
    user_id picks whose long-term preferences are loaded and saved.

    If the thread is paused at a clarification question (Stage 6), the text is
    the answer: the paused run is resumed with Command(resume=text). Sending it
    as a new message instead would start a new run and drop the question.
    The turn then ends with the agent's answer, or with the next question.
    """
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    snapshot = graph.get_state(config)
    before = len(snapshot.values.get("messages", []))
    graph_input = (
        Command(resume=text)
        if snapshot.interrupts
        else {"messages": [HumanMessage(text)]}
    )
    graph.invoke(graph_input, config, context=AgentContext(user_id=user_id))
    # Read the saved state: invoke()'s return value may also carry LangGraph's
    # internal "__interrupt__" entry when the run paused.
    return graph.get_state(config).values["messages"][before:]


def pending_question(graph: CompiledStateGraph, thread_id: str) -> str | None:
    """The clarification question a thread is paused at, or None."""
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    interrupts = graph.get_state(config).interrupts
    return str(interrupts[0].value) if interrupts else None


def final_answer(messages: Sequence[AnyMessage]) -> str:
    """The text of the last message: the LLM's final reply."""
    # Imported here to keep the module light; content_to_text handles both plain
    # strings and Gemini 3's list-of-blocks content.
    from ai_travel_assistant.llm_basics import content_to_text

    return content_to_text(messages[-1].content).strip()


def describe_steps(messages: Sequence[AnyMessage]) -> list[str]:
    """One readable line per message, to show the Reason/Act/Observe steps."""
    from ai_travel_assistant.llm_basics import content_to_text

    lines = []
    for message in messages:
        if isinstance(message, HumanMessage):
            lines.append(f"[human] {content_to_text(message.content)}")
        elif isinstance(message, AIMessage) and message.tool_calls:
            for call in message.tool_calls:
                if call["name"] == SUBMIT_TOOL:  # Stage 8: too big to print
                    args = call["args"]
                    lines.append(
                        f"[llm -> submit_itinerary] {args.get('days')} days in "
                        f"{args.get('destination')}"
                    )
                else:
                    lines.append(f"[llm -> tool call] {call['name']}({call['args']})")
        elif isinstance(message, AIMessage) and message.name == ITINERARY_NAME:
            lines.append("[compose -> itinerary checked and shown]")
        elif isinstance(message, AIMessage) and message.name == CLARIFICATION_NAME:
            lines.append("[clarify -> waiting for the traveller]")
        elif isinstance(message, AIMessage) and message.name == OFF_TOPIC_NAME:
            lines.append("[off-topic -> fixed reply]")
        elif isinstance(message, AIMessage):
            lines.append("[llm -> final answer]")
        else:  # ToolMessage
            first_line = content_to_text(message.content).splitlines()[0]
            lines.append(f"[tool result] {first_line} ...")
    return lines


# --- Multi-turn chat ----------------------------------------------------------

NEW_COMMAND = "/new"
QUIT_COMMANDS = frozenset({"/quit", "/exit"})
PREFS_COMMAND = "/prefs"
FORGET_COMMAND = "/forget"
ITINERARY_COMMAND = "/itinerary"
NO_ITINERARY = "No itinerary in this conversation yet."


def itinerary_json(graph: CompiledStateGraph, thread_id: str) -> str:
    """The thread's last checked itinerary as JSON (Stage 8), without the LLM."""
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    itinerary = graph.get_state(config).values.get("itinerary")
    if not itinerary:
        return NO_ITINERARY
    return json.dumps(itinerary, indent=2, ensure_ascii=False)


# Where --chat keeps its conversations. Git-ignored: it holds real chat text.
DEFAULT_CHAT_DB = Path(".data/chat.sqlite")


def new_thread_id() -> str:
    """A random, unique id for a new conversation."""
    return uuid.uuid4().hex


@contextmanager
def sqlite_checkpointer(path: Path) -> Iterator[SqliteSaver]:
    """Open a SQLite checkpointer on a file, and close the file afterwards.

    Same checkpoints as InMemorySaver, but written to disk: a new process that
    opens the same file and uses the same thread_id gets the old messages.
    check_same_thread=False because LangGraph may run nodes on worker threads.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    try:
        yield SqliteSaver(conn)
    finally:
        conn.close()


def saved_message_count(graph: CompiledStateGraph, thread_id: str) -> int:
    """How many messages the checkpointer holds for this thread (0 if new)."""
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    return len(graph.get_state(config).values.get("messages", []))


def preference_command(store: BaseStore | None, user_id: str, command: str) -> str:
    """Answer /prefs or /forget for this traveller."""
    if store is None:
        return "Long-term memory is off for this chat."
    if command == FORGET_COMMAND:
        return f"Forgot {clear_preferences(store, user_id)} saved preferences."
    prefs = load_preferences(store, user_id)
    if not prefs:
        return "No saved preferences yet."
    return "Saved preferences: " + "; ".join(prefs)


@dataclass(frozen=True)
class CommandResult:
    """What a chat command did: a reply, maybe a new thread, maybe quit."""

    reply: str
    thread_id: str | None = None  # set when the command started a new thread
    quit: bool = False


def chat_command(
    graph: CompiledStateGraph, thread_id: str, user_id: str, text: str
) -> CommandResult | None:
    """Handle /quit, /new, /prefs, /forget and /itinerary (no LLM call).

    None: the text is not a command and goes to the agent. Shared by the plain
    chat below and the rich chat (chat_ui.py), so both accept the same commands.
    """
    command = text.strip().lower()
    if command in QUIT_COMMANDS:
        return CommandResult(
            f"Bye! Reopen this conversation with --thread {thread_id}", quit=True
        )
    if command == NEW_COMMAND:
        new = new_thread_id()
        return CommandResult(f"New conversation: {new}", thread_id=new)
    if command in (PREFS_COMMAND, FORGET_COMMAND):
        return CommandResult(preference_command(graph.store, user_id, command))
    if command == ITINERARY_COMMAND:
        return CommandResult(itinerary_json(graph, thread_id))
    return None


CHAT_HELP = (
    f"Type {NEW_COMMAND} for a new conversation, {PREFS_COMMAND} or "
    f"{FORGET_COMMAND} for saved preferences, {ITINERARY_COMMAND} for the "
    "last itinerary as JSON, /quit to stop."
)


def open_chat(
    graph: CompiledStateGraph, thread_id: str | None
) -> tuple[str, list[str]]:
    """Pick the thread to chat on and the lines to show first (no LLM call)."""
    lines = []
    if thread_id is None:
        thread_id = new_thread_id()
        lines.append(f"New conversation: {thread_id}")
    else:
        count = saved_message_count(graph, thread_id)
        if count:
            lines.append(
                f"Continuing conversation {thread_id} ({count} saved messages)."
            )
            question = pending_question(graph, thread_id)
            if question:
                lines.append(f"AI (waiting for your answer): {question}")
        else:
            lines.append(f"No saved messages for {thread_id}; starting it now.")
    lines.append(CHAT_HELP)
    return thread_id, lines


def run_agent_chat(
    graph: CompiledStateGraph,
    read: Callable[[str], str] = input,
    write: Callable[[str], None] = print,
    *,
    thread_id: str | None = None,
    user_id: str = DEFAULT_USER,
    show_steps: bool = False,
    describe_error: Callable[[Exception], str] = lambda exc: type(exc).__name__,
) -> None:
    """Chat with the agent in the terminal, one thread at a time.

    The graph must be built with a checkpointer. Every message goes to the
    current thread, so follow-ups see the earlier turns. Pass thread_id to
    continue a saved conversation; otherwise a new one starts. /new starts a
    fresh thread (the old one stays saved), /quit or /exit ends the chat. A
    failed turn (e.g. a Gemini 503) is reported and the chat goes on.

    If the graph has a Store, /prefs lists the traveller's saved preferences
    and /forget deletes them (neither calls the LLM).

    If the graph asks clarification questions (Stage 6), the next message
    answers the question (ask_agent resumes the paused run). A reopened thread
    that is waiting for an answer shows its question again. /new while a
    question waits leaves that thread paused; --thread can answer it later.

    read and write default to input() and print(); tests pass fakes. Ctrl+C
    or end of input (EOFError) is left to the caller.
    """
    thread_id, lines = open_chat(graph, thread_id)
    for line in lines:
        write(line)
    while True:
        text = read("You: ").strip()
        if not text:
            continue
        command = chat_command(graph, thread_id, user_id, text)
        if command is not None:
            write(command.reply)
            if command.quit:
                return
            thread_id = command.thread_id or thread_id
            continue
        try:
            turn = ask_agent(graph, thread_id, text, user_id=user_id)
        except Exception as exc:  # keep chatting after one failed turn
            write(f"Request failed: {describe_error(exc)}")
            continue
        if show_steps:
            write("\n".join(describe_steps(turn)))
        write(f"AI: {final_answer(turn)}")


# --- The chat agent, as every interface builds it ------------------------------


def build_chat_graph(
    settings: GeminiSettings,
    checkpointer: BaseCheckpointSaver | None,
    store: BaseStore | None,
    *,
    clarify: bool = True,
    persistence_provided: bool = False,
) -> CompiledStateGraph:
    """The full agent of --chat: the CLI, the web page and Studio all use it.

    Stage 7: one retry layer. The llm node retries a transient error once
    (llm_retry), never a 429. The client does not retry on Gemini; on Groq it
    waits out short tokens-per-minute 429s (see agent_client_retries). The
    understander and the extractor are not retried: they fall back (to the
    agent, or to not saving) instead. Both run at temperature 0 and the
    extractor only when its gate matches. Stage 8: itineraries come back
    structured, checked and rendered (no extra call).

    persistence_provided: LangGraph Studio attaches its own checkpointer and
    store to the graph, so it is built without them (see studio.py).
    """
    from ai_travel_assistant.llm_basics import agent_client_retries, build_llm

    retries = agent_client_retries(settings)
    llm = build_llm(settings, temperature=1.0, max_retries=retries)
    precise_llm = build_llm(settings, temperature=0, max_retries=retries)
    return build_tool_agent_graph(
        llm,
        checkpointer=checkpointer,
        store=store,
        preference_extractor=make_llm_preference_extractor(precise_llm),
        # Stage 6: one extra call per message reads its intent and details.
        understander=make_llm_request_understander(precise_llm) if clarify else None,
        llm_retry=llm_retry_policy(),
        structured_itinerary=True,
        persistence_provided=persistence_provided,
    )


# --- CLI ----------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ask the tool-calling travel agent (usually 2 Gemini calls)."
    )
    parser.add_argument(
        "question", nargs="?", help='e.g. "Find places in Sri Lanka for hiking"'
    )
    parser.add_argument(
        "--steps", action="store_true", help="also print each step of the loop"
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="multi-turn chat with memory (each message makes Gemini calls)",
    )
    parser.add_argument(
        "--thread", metavar="ID", help="with --chat: continue a saved conversation"
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_CHAT_DB,
        help=f"with --chat: SQLite file for conversations (default: {DEFAULT_CHAT_DB})",
    )
    parser.add_argument(
        "--user",
        default=DEFAULT_USER,
        help=f"with --chat: whose preferences to use (default: {DEFAULT_USER})",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="with a question: print the structured itinerary as JSON (no --chat)",
    )
    parser.add_argument(
        "--no-clarify",
        action="store_true",
        help=(
            "with --chat: skip request understanding and clarification "
            "questions (saves one Gemini call per message)"
        ),
    )
    parser.add_argument(
        "--plain",
        action="store_true",
        help="with --chat: plain text, no live progress or streaming (Stage 7 look)",
    )
    args = parser.parse_args(argv)
    if args.plain and not args.chat:
        parser.error("--plain is used together with --chat")
    if args.steps and args.chat and not args.plain:
        parser.error(
            "--chat always shows the steps; --steps is for --plain or one question"
        )
    if args.question is None and not args.chat:
        parser.error("give a question, or use --chat")
    if args.question is not None and args.chat:
        parser.error("--chat reads its questions from the terminal")
    if args.thread is not None and not args.chat:
        parser.error("--thread is used together with --chat")
    if args.user != DEFAULT_USER and not args.chat:
        parser.error("--user is used together with --chat")
    if args.no_clarify and not args.chat:
        parser.error("--no-clarify is used together with --chat")
    if args.json and args.chat:
        parser.error("--json is for one question; in --chat type /itinerary")

    # Stage 7: refuse a mistyped FAULT_INJECT, and say loudly when one is on.
    try:
        fault_warning = faults.describe_active_faults()
    except faults.FaultConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1
    if fault_warning:
        print(fault_warning, file=sys.stderr)

    # Imported here so the offline tests never load the Gemini setup code.
    from ai_travel_assistant.llm_basics import (
        ConfigError,
        build_llm,
        load_settings,
    )

    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    if args.chat:
        # Conversations (checkpointer) and preferences (Store) live in the same
        # SQLite file, so both survive a restart.
        with (
            sqlite_checkpointer(args.db) as checkpointer,
            sqlite_store(args.db) as store,
        ):
            graph = build_chat_graph(
                settings, checkpointer, store, clarify=not args.no_clarify
            )
            try:
                if args.plain:
                    run_agent_chat(
                        graph,
                        thread_id=args.thread,
                        user_id=args.user,
                        show_steps=args.steps,
                        # A short, safe message: never the raw error text.
                        describe_error=describe_llm_error,
                    )
                else:
                    # Stage 10: live progress and streamed replies.
                    from ai_travel_assistant.chat_ui import run_rich_chat

                    run_rich_chat(
                        graph,
                        thread_id=args.thread,
                        user_id=args.user,
                        describe_error=describe_llm_error,
                    )
            except (KeyboardInterrupt, EOFError):  # Ctrl+C / Ctrl+Z ends the chat
                print("\nBye!")
        return 0

    # One-shot mode keeps its Stage 3/4 retries (the client's own), and gets
    # Stage 8's structured itineraries.
    graph = build_tool_agent_graph(
        build_llm(settings, temperature=1.0), structured_itinerary=True
    )
    try:
        state = graph.invoke({"messages": [HumanMessage(args.question)]})
    except Exception as exc:  # show a clean message instead of a traceback
        print(f"Request failed: {describe_llm_error(exc)}", file=sys.stderr)
        return 1
    messages = state["messages"]

    if args.steps:
        print("\n".join(describe_steps(messages)))
        print()
    if args.json:
        itinerary = state.get("itinerary")
        if itinerary:
            print(json.dumps(itinerary, indent=2, ensure_ascii=False))
            return 0
        print(NO_ITINERARY, file=sys.stderr)
    print(final_answer(messages))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
