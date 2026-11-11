"""Stage 2: a multi-node LangGraph workflow for trip requests, now multi-turn.

What this file demonstrates:
  1. A TypedDict State shared by every node.
  2. Four separate nodes, each doing one small job:
       trip_input_node        -> pull destination and days out of the user's text
       validation_node        -> work out which required fields are missing
       ask_clarification_node -> ask the traveller for the missing details
       response_node          -> confirm the trip when nothing is missing
  3. A conditional edge: after validation, a router function picks the next node.
  4. Compiling the graph and invoking it.
  5. Multi-turn conversations: a checkpointer saves the state per thread_id,
     interrupt() pauses the graph to wait for the traveller's reply, and
     Command(resume=reply) continues it.
  6. Context-aware replies: the reply travels together with the question it
     answers, so a short answer like "3" is read as days after "How many
     days...?" instead of being guessed at on its own.

Single-turn flow (build_trip_graph(extractor), no checkpointer):

    START -> trip_input -> validation --(missing fields?)--> ask_clarification -> END
                                      \\--(all present)---> response ----------> END

Multi-turn flow (build_trip_graph(extractor, checkpointer=InMemorySaver())):

    START -> trip_input -> validation --(missing fields?)--> ask_clarification
                 ^                    \\--(all present)---> response -> END   |
                 |                                                          v
                 +------------------------------------------- wait_for_reply
                                                       (interrupt: pause here)

Extraction is passed in as a function (see TripExtractor), so the same graph
runs with Gemini in the app and with fixed offline data in the tests.

Run from the project root (one Gemini call):
    uv run python -m ai_travel_assistant.trip_workflow "3 days in Ella, love hiking"

Chat until the trip is complete (one Gemini call per message):
    uv run python -m ai_travel_assistant.trip_workflow --chat "I want to visit Kandy"

Draw the graph into docs/ as a Mermaid diagram (offline, no Gemini call):
    uv run python -m ai_travel_assistant.trip_workflow --draw

Also save it as a PNG image (sends the diagram text to the mermaid.ink service):
    uv run python -m ai_travel_assistant.trip_workflow --draw --png

No tools, long-term memory, or multiple agents here. Conversation memory lives
only in RAM (InMemorySaver), so it is lost when the process exits.
"""

from __future__ import annotations

import argparse
import sys
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Literal, Protocol, TypedDict

from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.graph_mermaid import draw_mermaid_png
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, interrupt

from ai_travel_assistant.schemas import CRITICAL_FIELDS, TripRequest

# --- State --------------------------------------------------------------------


class TripState(TypedDict, total=False):
    """The data that flows through the graph.

    total=False means every key is optional. That lets us start the graph with
    only {"user_input": ...}; the nodes fill in the other keys as they run.
    """

    user_input: str  # the traveller's latest message (first request or a reply)
    destination: str | None  # e.g. "Ella", or None if not given
    days: int | None  # e.g. 3, or None if not given
    missing_fields: list[str]  # e.g. ["days"]; empty when the request is complete
    response: str  # the final message for the traveller
    # The clarification question that user_input answers, e.g. "How many days
    # would you like to spend in Kandy?". Set by wait_for_reply together with
    # the reply, cleared by trip_input once used. None for a fresh message.
    clarification_context: str | None


class TripExtractor(Protocol):
    """Anything that turns the traveller's message into a TripRequest.

    In the app this calls Gemini; in the tests it is a dictionary lookup.

    The keyword arguments are context for reading short replies, and are passed
    only when there is something to pass:
      clarification_context: the question the message answers, e.g.
          "How many days would you like to spend in Kandy?"
      existing_data: details already known, e.g. {"destination": "Kandy"}

    The extractor should return only what the message itself states (read in
    that context), not repeat existing_data. A first message with nothing known
    is called as extractor(text), so a text-only function still works for it.
    """

    def __call__(
        self,
        user_input: str,
        *,
        clarification_context: str | None = None,
        existing_data: dict[str, object] | None = None,
    ) -> TripRequest: ...


# Node names, kept in one place so the edges and tests can't drift apart.
TRIP_INPUT = "trip_input"
VALIDATION = "validation"
ASK_CLARIFICATION = "ask_clarification"
RESPONSE = "response"
WAIT_FOR_REPLY = "wait_for_reply"


# --- Nodes --------------------------------------------------------------------
# A node takes the current state and returns a dict containing only the keys it
# wants to change. LangGraph merges that dict into the state.


def make_trip_input_node(extractor: TripExtractor) -> Callable[[TripState], dict]:
    """Build the trip_input node around the given extractor.

    A node receives only the state, so we use a closure: the inner function
    "remembers" the extractor it was created with.
    """

    def trip_input_node(state: TripState) -> dict:
        text = state.get("user_input", "")
        question = state.get("clarification_context")
        # The question belongs to this one reply. Clear it, so a later message
        # (e.g. a new request after the trip is confirmed) is never read as an
        # answer to an old question.
        update: dict = {"clarification_context": None} if question else {}
        if not text.strip():
            return update  # nothing to extract: skip the API call, keep the state

        # Give the extractor the context it needs to read a short reply: "3"
        # means days only because the question asked for days.
        context: dict = {}
        if question:
            context["clarification_context"] = question
        known = {
            name: state[name]
            for name in CRITICAL_FIELDS
            if not _is_missing(state.get(name))
        }
        if known:
            context["existing_data"] = known
        trip = extractor(text, **context)

        # Merge: return only the fields this message actually gave. A follow-up
        # like "3 days" extracts destination=None, which must not erase the
        # destination from an earlier turn. A new value (e.g. "Actually, make
        # it 5 days") replaces the old one.
        update.update(
            {
                name: getattr(trip, name)
                for name in CRITICAL_FIELDS
                if not _is_missing(getattr(trip, name))
            }
        )
        return update

    return trip_input_node


def _is_missing(value: object) -> bool:
    """None or a blank string counts as missing."""
    return value is None or (isinstance(value, str) and not value.strip())


def validation_node(state: TripState) -> dict:
    """Record which required fields (destination, days) are still missing."""
    missing = [name for name in CRITICAL_FIELDS if _is_missing(state.get(name))]
    return {"missing_fields": missing}


def ask_clarification_node(state: TripState) -> dict:
    """Ask the traveller only for the details we don't have yet."""
    missing = state.get("missing_fields", [])
    destination = state.get("destination")
    days = state.get("days")

    if "destination" in missing and "days" in missing:
        question = "Where in Sri Lanka would you like to go, and for how many days?"
    elif "destination" in missing:
        question = f"Where in Sri Lanka would you like to spend your {days} days?"
    else:  # only days is missing
        question = f"How many days would you like to spend in {destination}?"
    return {"response": question}


def wait_for_reply_node(state: TripState) -> dict:
    """Pause the graph until the traveller answers the clarification question.

    interrupt() saves the state and stops the run; the caller shows the question
    (the interrupt's value) to the traveller. When the caller sends
    Command(resume=reply), LangGraph runs this node again FROM THE TOP, and this
    time interrupt() returns the reply. That is why this node does nothing but
    wait: any work placed before interrupt() (like a Gemini call) would run twice.

    The reply is returned together with the question it answers (the value the
    traveller was shown), so trip_input can read "3" as an answer to "How many
    days...?". Pairing them here means every round uses its own latest question.
    """
    question = state.get("response", "")
    reply = interrupt(question)
    # Only text can be extracted. Anything else counts as an empty reply, which
    # makes the graph ask the same question again.
    return {
        "user_input": reply if isinstance(reply, str) else "",
        "clarification_context": question or None,
    }


def response_node(state: TripState) -> dict:
    """Confirm a complete trip request."""
    return {
        "response": (
            f"Great! I've noted a {state['days']}-day trip to {state['destination']}."
        )
    }


# --- Router (used by the conditional edge) ------------------------------------


def route_after_validation(
    state: TripState,
) -> Literal["ask_clarification", "response"]:
    """Decide where to go after validation.

    A router does not change the state. It only returns the name of the next node.
    """
    if state.get("missing_fields"):
        return ASK_CLARIFICATION
    return RESPONSE


# --- Graph --------------------------------------------------------------------


def build_trip_graph(
    extractor: TripExtractor, checkpointer: BaseCheckpointSaver | None = None
) -> CompiledStateGraph:
    """Wire the nodes and edges together and compile the graph.

    Without a checkpointer the graph is single-turn: it asks its question and
    ends. With one it is multi-turn: after asking, it pauses in wait_for_reply
    and loops back to trip_input with the traveller's answer. A paused run can
    only be resumed from a saved checkpoint, so the loop is added only when there
    is a checkpointer to save to.
    """
    builder = StateGraph(TripState)

    builder.add_node(TRIP_INPUT, make_trip_input_node(extractor))
    builder.add_node(VALIDATION, validation_node)
    builder.add_node(ASK_CLARIFICATION, ask_clarification_node)
    builder.add_node(RESPONSE, response_node)

    # Fixed edges: these steps always run in this order.
    builder.add_edge(START, TRIP_INPUT)
    builder.add_edge(TRIP_INPUT, VALIDATION)

    # Conditional edge: after validation, call the router to pick the next node.
    # The dict maps each value the router can return to a node name.
    builder.add_conditional_edges(
        VALIDATION,
        route_after_validation,
        {ASK_CLARIFICATION: ASK_CLARIFICATION, RESPONSE: RESPONSE},
    )

    if checkpointer is None:
        builder.add_edge(ASK_CLARIFICATION, END)
    else:
        # The loop: ask -> wait for the answer -> extract from the answer -> ...
        builder.add_node(WAIT_FOR_REPLY, wait_for_reply_node)
        builder.add_edge(ASK_CLARIFICATION, WAIT_FOR_REPLY)
        builder.add_edge(WAIT_FOR_REPLY, TRIP_INPUT)
    builder.add_edge(RESPONSE, END)

    # compile() checks the wiring (e.g. no edge to an unknown node) and returns
    # a runnable graph with .invoke(). The checkpointer saves the state after
    # every step, keyed by the thread_id in the config.
    return builder.compile(checkpointer=checkpointer)


def run_trip_workflow(extractor: TripExtractor, user_input: str) -> TripState:
    """Build the graph, run it once, and return the final state."""
    graph = build_trip_graph(extractor)
    return graph.invoke({"user_input": user_input})


# --- Multi-turn conversations --------------------------------------------------
# A thread_id names one conversation. The checkpointer keeps a separate saved
# state for each thread_id, so two travellers never see each other's trips.


def new_thread_id() -> str:
    """A random, unique id for a new conversation."""
    return uuid.uuid4().hex


def thread_config(thread_id: str) -> RunnableConfig:
    """The config that tells the graph which conversation to load and save."""
    return {"configurable": {"thread_id": thread_id}}


def is_awaiting_reply(graph: CompiledStateGraph, thread_id: str) -> bool:
    """True when this conversation is paused at a clarification question."""
    return bool(graph.get_state(thread_config(thread_id)).interrupts)


def send_message(graph: CompiledStateGraph, thread_id: str, text: str) -> TripState:
    """Send one traveller message to a conversation and return its new state.

    The graph must be built with a checkpointer. The reply for the traveller is
    always state["response"]: a clarification question while details are
    missing, otherwise the confirmation.

    - If the conversation is paused at a question, the message is the answer:
      resume the paused run with Command(resume=text).
    - Otherwise (a new thread, or a finished trip) start a new run. The saved
      destination and days are kept, so "Actually, make it 5 days" after a
      confirmation updates the existing trip.

    Checking first matters: resuming a thread that is not paused does nothing
    and silently returns the old state.
    """
    config = thread_config(thread_id)
    if is_awaiting_reply(graph, thread_id):
        graph.invoke(Command(resume=text), config)
    else:
        graph.invoke({"user_input": text}, config)
    # Read the saved state instead of invoke()'s return value, which may also
    # carry LangGraph's internal "__interrupt__" entry.
    return graph.get_state(config).values


def run_chat(
    extractor: TripExtractor,
    first_message: str | None = None,
    read: Callable[[str], str] = input,
    write: Callable[[str], None] = print,
) -> TripState:
    """Chat in the terminal until the trip has a destination and days.

    read and write default to input() and print(); tests pass fakes.
    """
    graph = build_trip_graph(extractor, checkpointer=InMemorySaver())
    thread_id = new_thread_id()
    text = first_message if first_message is not None else read("You: ")
    while True:
        state = send_message(graph, thread_id, text)
        write(f"AI: {state['response']}")
        if not is_awaiting_reply(graph, thread_id):
            return state
        text = read("You: ")


# --- Visualization (offline) --------------------------------------------------

DEFAULT_DIAGRAM_PATH = Path("docs/trip_workflow_graph.md")


def _drawable_graph():
    """Return a description of the graph's nodes and edges, ready to draw.

    The graph's shape does not depend on the extractor, and drawing never runs a
    node, so a placeholder extractor is enough. No Gemini call is made.
    """

    def placeholder_extractor(text: str, **context: object) -> TripRequest:
        return TripRequest()

    # Draw the multi-turn version, which includes the wait_for_reply loop.
    graph = build_trip_graph(placeholder_extractor, checkpointer=InMemorySaver())
    return graph.get_graph()


def render_mermaid() -> str:
    """Return the graph as Mermaid diagram text (made locally, no network)."""
    return _drawable_graph().draw_mermaid()


# LangGraph draws only nodes and edges, with no edge labels and no per-node
# colours. The annotated diagram adds those on top of the generated text. Each
# label is attached to an edge LangGraph actually drew; if the graph changes and
# that edge disappears, render_annotated_mermaid() fails instead of drawing a
# wrong picture.
EDGE_LABELS: tuple[tuple[str, str, str], ...] = (
    (VALIDATION, ASK_CLARIFICATION, "missing destination or days"),
    (VALIDATION, RESPONSE, "all details present"),
    (WAIT_FOR_REPLY, TRIP_INPUT, "reply: extract, merge, validate again"),
)

# Nodes added or changed for multi-turn conversations. The others (validation,
# ask_clarification, response) are unchanged from the single-turn graph.
NEW_NODES = (WAIT_FOR_REPLY,)
MODIFIED_NODES = (TRIP_INPUT,)

# Things that are not graph nodes but take part in every multi-turn run: the
# traveller (who answers the interrupt) and the checkpointer (which saves and
# reloads the state for a thread_id).
_MEMORY_LABEL = (
    "InMemorySaver checkpointer<br/>state saved after every step<br/>"
    "one saved state per thread_id"
)
_MERMAID_EXTRAS = f"""\
\ttraveller{{{{"Traveller<br/>send_message(graph, thread_id, text)"}}}}:::outside
\tmemory[("{_MEMORY_LABEL}")]:::memory
\t{WAIT_FOR_REPLY} -.->|"interrupt(question): run pauses"| traveller;
\ttraveller -.->|"Command(resume=reply), same thread_id"| {WAIT_FOR_REPLY};
\t{WAIT_FOR_REPLY} <-.->|"pause saved / reloaded on resume"| memory;
\tclassDef new fill:#c8f0d4,stroke:#1a7f4b,stroke-width:2px
\tclassDef modified fill:#fff0b3,stroke:#a67c00,stroke-width:2px
\tclassDef outside fill:#ffffff,stroke:#888888,stroke-dasharray:4 3
\tclassDef memory fill:#dbeafe,stroke:#1d4ed8,stroke-width:2px
\tclass {",".join(NEW_NODES)} new;
\tclass {",".join(MODIFIED_NODES)} modified;
"""


def render_annotated_mermaid() -> str:
    """The generated diagram plus edge labels, colours, the traveller and memory.

    Every node and edge of the graph still comes from LangGraph's own drawing;
    this only labels edges and adds the parts LangGraph cannot draw.
    """
    lines = render_mermaid().rstrip().splitlines()
    for source, target, label in EDGE_LABELS:
        for arrow in ("-->", "-.->"):
            edge = f"\t{source} {arrow} {target};"
            if edge in lines:
                lines[lines.index(edge)] = f'\t{source} {arrow}|"{label}"| {target};'
                break
        else:
            raise ValueError(f"No edge {source} -> {target} in the compiled graph")
    return "\n".join(lines) + "\n" + _MERMAID_EXTRAS


def render_png(path: Path) -> Path:
    """Save the annotated diagram as a PNG image.

    Uses LangGraph's default method: the Mermaid text (node names, edge labels
    and styling only, no keys or user data) is sent to the mermaid.ink web
    service, which returns the image. Needs an internet connection.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    draw_mermaid_png(render_annotated_mermaid(), output_file_path=str(path))
    return path


def write_diagram_doc(
    path: Path = DEFAULT_DIAGRAM_PATH, png_path: Path | None = None
) -> Path:
    """Write a Markdown page with the Mermaid diagram (and the PNG, if given)."""
    command = "uv run python -m ai_travel_assistant.trip_workflow --draw"
    image = ""
    if png_path is not None:
        command += " --png"
        # Link the image relative to the Markdown file so it works on GitHub.
        relative = png_path.resolve().relative_to(path.parent.resolve())
        image = f"![Trip workflow graph]({relative.as_posix()})\n\n"

    content = (
        "# Trip workflow graph\n\n"
        "Generated from the compiled LangGraph in "
        "`src/ai_travel_assistant/trip_workflow.py`. Do not edit by hand; "
        f"regenerate with:\n\n```\n{command}\n```\n\n"
        "This is the multi-turn graph (built with a checkpointer). Without a "
        "checkpointer, `ask_clarification` goes straight to `__end__` instead "
        "and there is no `wait_for_reply`.\n\n"
        "| Look | Meaning |\n"
        "|---|---|\n"
        "| Solid arrow | Fixed edge |\n"
        "| Dashed arrow from `validation` | Conditional edge, chosen by "
        "`route_after_validation` |\n"
        "| Lilac node | Unchanged from the single-turn graph |\n"
        "| Yellow node | Changed: `trip_input` reads each reply together with "
        "the question it answers, merges new details into the saved ones and "
        "skips blank replies |\n"
        "| Green node | New: `wait_for_reply` calls `interrupt()` |\n"
        "| Dashed white box | The traveller, outside the graph |\n"
        "| Blue cylinder | The checkpointer (conversation memory) |\n\n"
        "How one conversation runs:\n\n"
        "1. `send_message` starts a run for a `thread_id`. `trip_input` "
        "extracts details and `validation` lists what is missing.\n"
        "2. Missing details: `ask_clarification` writes the question, then "
        "`wait_for_reply` calls `interrupt(question)`. The run stops and the "
        "checkpointer keeps the state under that `thread_id`.\n"
        "3. The traveller answers. `send_message` sees the thread is paused "
        "and sends `Command(resume=reply)` with the same `thread_id`. The "
        "state is reloaded, `wait_for_reply` re-runs and `interrupt()` returns "
        "the reply. `wait_for_reply` returns the reply together with the "
        "question it answers, saved as `clarification_context`.\n"
        "4. Back to `trip_input` with the reply. The extractor receives the "
        "reply, that question and the details already known, so a short answer "
        "like `3` is read as days after *How many days...?*. New details are "
        "merged into the saved ones (a missing value never erases a saved one), "
        "the question is cleared, and the result is validated again. Steps 2-4 "
        "repeat until nothing is missing; each round uses its own latest "
        "question.\n"
        "5. All details present: `response` confirms the trip and the run "
        "ends.\n\n"
        f"{image}"
        f"```mermaid\n{render_annotated_mermaid().strip()}\n```\n\n"
        "<details>\n<summary>Raw graph, exactly as LangGraph draws it</summary>"
        "\n\n"
        f"```mermaid\n{render_mermaid().strip()}\n```\n\n"
        "</details>\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


# --- CLI ----------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the trip-request workflow.")
    parser.add_argument(
        "text", nargs="?", help='Trip request, e.g. "3 days in Ella" (one Gemini call)'
    )
    parser.add_argument(
        "--draw",
        nargs="?",
        const=DEFAULT_DIAGRAM_PATH,
        type=Path,
        metavar="PATH",
        help=f"write the graph diagram (default: {DEFAULT_DIAGRAM_PATH}); offline",
    )
    parser.add_argument(
        "--png",
        action="store_true",
        help="with --draw: also save a PNG next to it (uses the mermaid.ink service)",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="keep asking until destination and days are known (one call per reply)",
    )
    args = parser.parse_args(argv)

    if args.png and args.draw is None:
        parser.error("--png is used together with --draw")
    if args.draw is not None:
        png_path = None
        if args.png:
            try:
                png_path = render_png(args.draw.with_suffix(".png"))
            except Exception as exc:  # e.g. no internet, or mermaid.ink is down
                print(f"PNG failed: {type(exc).__name__}: {exc}", file=sys.stderr)
                return 1
            print(f"Image written to {png_path}")
        print(f"Diagram written to {write_diagram_doc(args.draw, png_path)}")
        return 0
    if args.text is None and not args.chat:
        parser.error("give a trip request, or use --chat or --draw")

    # Imported here so the offline tests never load the Gemini setup code.
    from ai_travel_assistant.llm_basics import (
        ConfigError,
        build_llm,
        load_settings,
        safe_error_message,
    )
    from ai_travel_assistant.structured_output import extract_trip_request

    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    # The real extractor: Stage 1's structured output, called by trip_input_node.
    llm = build_llm(settings, temperature=0)

    def gemini_extractor(
        text: str,
        *,
        clarification_context: str | None = None,
        existing_data: dict[str, object] | None = None,
    ) -> TripRequest:
        return extract_trip_request(
            llm,
            text,
            clarification_context=clarification_context,
            existing_data=existing_data,
        )

    try:
        if args.chat:
            run_chat(gemini_extractor, args.text)
            return 0
        state = run_trip_workflow(gemini_extractor, args.text)
    except (KeyboardInterrupt, EOFError):  # Ctrl+C / Ctrl+Z ends the chat
        print("\nBye!")
        return 0
    except Exception as exc:  # show a clean message instead of a traceback
        message = safe_error_message(exc, settings.api_key)
        print(f"Request failed: {message}", file=sys.stderr)
        return 1

    print(f"destination:    {state.get('destination')}")
    print(f"days:           {state.get('days')}")
    print(f"missing_fields: {state.get('missing_fields')}")
    print(f"response:       {state.get('response')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
