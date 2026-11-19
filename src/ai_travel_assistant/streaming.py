"""Stage 10: streaming one chat turn as UI events.

What this file demonstrates:
  1. LangGraph STREAM MODES, two at once:
       - "updates": after each node finishes, what it wrote to the state. Used
         for progress ("🔎 Checking the weather in Ella").
       - "messages": messages as they are made, including the LLM's reply
         token by token (AIMessageChunk). Used for the typing effect.
     With a list of modes, graph.stream yields (mode, chunk) pairs.
  2. A PRESENTATION LAYER. The raw stream knows about nodes and chunks; a UI
     should not. stream_turn turns it into three small events (Step, Token,
     TurnDone) that the rich CLI and the Streamlit page both render. A web or
     mobile frontend talking to a backend would get the same kind of events.
  3. Interrupts in a stream. A clarification question (Stage 6) ends the
     stream early with an "__interrupt__" update; the next message resumes the
     paused run with Command(resume=...), exactly as ask_agent does.

Only the agent's own replies stream as tokens (nodes llm and
fallback_response). The understander's structured output is not text for the
traveller, so it is never shown. A checked itinerary is not streamed either:
compose_answer renders it in one piece, without an LLM call.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    AnyMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from ai_travel_assistant.itinerary_grounding import SUBMIT_TOOL, UNAVAILABLE_PREFIXES
from ai_travel_assistant.itinerary_output import ITINERARY_NAME
from ai_travel_assistant.request_understanding import (
    CLARIFICATION_NAME,
    OFF_TOPIC_NAME,
    UNDERSTAND_REQUEST,
)
from ai_travel_assistant.tool_calling import (
    COMPOSE_ANSWER,
    DEFAULT_USER,
    FALLBACK_RESPONSE,
    LLM,
    TOOLS_NODE,
    AgentContext,
    final_answer,
)

# Nodes whose LLM output is the reply the traveller reads.
TOKEN_NODES = frozenset({LLM, FALLBACK_RESPONSE})

# One line per tool call, filled from the call's own arguments.
TOOL_STEPS = {
    "search_destinations": "🗺️  Searching for places that match",
    "get_weather_forecast": "🔎 Checking the weather in {destination}",
    "find_attractions": "📍 Finding sights near {destination}",
    "get_destination_guide": "📖 Reading the travel guide for {destination}",
    "estimate_travel_time": "🚗 Estimating the drive from {origin} to {destination}",
    "convert_currency": "💱 Converting {amount} {from_currency} to {to_currency}",
    "get_attraction_prices": "🎟️  Checking official ticket prices",
    "suggest_hotels": "🏨 Finding places to stay in {destination}",
    SUBMIT_TOOL: "📝 Writing your itinerary",
}
NODE_STEPS = {
    UNDERSTAND_REQUEST: "🧭 Understood your request",
    COMPOSE_ANSWER: "✅ Checked the itinerary against the tool results",
    FALLBACK_RESPONSE: "⏱️  Tool limit reached; writing the best answer so far",
}
FAILED_STEP = "⚠️  {tool} returned no data; the answer will say what is missing"

Kind = Literal["answer", "itinerary", "clarification", "off_topic"]


@dataclass(frozen=True)
class Step:
    """A line of progress: a node finished or a tool is about to run."""

    text: str


@dataclass(frozen=True)
class Token:
    """A piece of the agent's reply, as the LLM writes it."""

    text: str


@dataclass(frozen=True)
class TurnDone:
    """The turn is over. `answer` is the full reply, also when it streamed."""

    answer: str
    kind: Kind
    question: str | None = None  # set when the turn paused to ask
    itinerary: dict | None = None  # the checked plan, when this turn made one


Event = Step | Token | TurnDone


def _format_step(template: str, args: Mapping) -> str:
    class Missing(dict):
        def __missing__(self, key: str) -> str:
            return "?"

    return template.format_map(Missing(args))


def tool_step(name: str, args: Mapping) -> Step:
    template = TOOL_STEPS.get(name, "🔧 Running " + name)
    return Step(_format_step(template, args))


def steps_from_update(node: str, update: object) -> list[Step]:
    """The progress lines one node's state update stands for."""
    if not isinstance(update, Mapping):
        return []
    messages = update.get("messages") or []
    if node == LLM:
        return [
            tool_step(call["name"], call["args"])
            for message in messages
            if isinstance(message, AIMessage)
            for call in message.tool_calls
        ]
    if node == TOOLS_NODE:
        return [
            Step(FAILED_STEP.format(tool=message.name or "A tool"))
            for message in messages
            if isinstance(message, ToolMessage)
            and (
                message.status == "error"
                or message.text.startswith(UNAVAILABLE_PREFIXES)
            )
        ]
    if node in NODE_STEPS:
        return [Step(NODE_STEPS[node])]
    return []


def turn_kind(messages: Sequence[AnyMessage]) -> Kind:
    last = messages[-1] if messages else None
    name = getattr(last, "name", None)
    if name == ITINERARY_NAME:
        return "itinerary"
    if name == CLARIFICATION_NAME:
        return "clarification"
    if name == OFF_TOPIC_NAME:
        return "off_topic"
    return "answer"


def stream_turn(
    graph: CompiledStateGraph,
    thread_id: str,
    text: str,
    user_id: str = DEFAULT_USER,
) -> Iterator[Event]:
    """Send one message and yield Steps and Tokens, then one TurnDone.

    Same contract as ask_agent (it resumes a paused question), but streamed.
    Exceptions (e.g. a Gemini 503 after the retry) propagate to the caller.
    """
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    snapshot = graph.get_state(config)
    before = len(snapshot.values.get("messages", []))
    graph_input = (
        Command(resume=text)
        if snapshot.interrupts
        else {"messages": [HumanMessage(text)]}
    )
    for mode, chunk in graph.stream(
        graph_input,
        config,
        context=AgentContext(user_id=user_id),
        stream_mode=["updates", "messages"],
    ):
        if mode == "messages":
            message, metadata = chunk
            if (
                isinstance(message, AIMessageChunk)
                and metadata.get("langgraph_node") in TOKEN_NODES
                and message.text
            ):
                yield Token(message.text)
        elif mode == "updates":
            for node, update in chunk.items():
                if node != "__interrupt__":  # the question is read below
                    yield from steps_from_update(node, update)

    after = graph.get_state(config)
    new = after.values.get("messages", [])[before:]
    kind = turn_kind(new)
    yield TurnDone(
        answer=final_answer(new) if new else "",
        kind=kind,
        question=str(after.interrupts[0].value) if after.interrupts else None,
        itinerary=after.values.get("itinerary") if kind == "itinerary" else None,
    )


def visible_history(messages: Sequence[AnyMessage]) -> list[tuple[str, str]]:
    """(role, text) pairs a chat window shows: the traveller and the replies.

    Tool calls, tool results and empty steps are left out; the itinerary,
    questions and fixed replies are AI messages like any other answer.
    """
    from ai_travel_assistant.llm_basics import content_to_text

    shown = []
    for message in messages:
        if isinstance(message, HumanMessage):
            shown.append(("user", content_to_text(message.content)))
        elif isinstance(message, AIMessage) and not message.tool_calls:
            text = content_to_text(message.content).strip()
            if text:
                shown.append(("assistant", text))
    return shown
