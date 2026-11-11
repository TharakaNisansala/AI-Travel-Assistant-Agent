"""Stage 2: a multi-node LangGraph workflow for trip requests.

What this file demonstrates:
  1. A TypedDict State shared by every node.
  2. Four separate nodes, each doing one small job:
       trip_input_node        -> pull destination and days out of the user's text
       validation_node        -> work out which required fields are missing
       ask_clarification_node -> ask the traveller for the missing details
       response_node          -> confirm the trip when nothing is missing
  3. A conditional edge: after validation, a router function picks the next node.
  4. Compiling the graph and invoking it.

The flow:

    START -> trip_input -> validation --(missing fields?)--> ask_clarification -> END
                                      \\--(all present)---> response ----------> END

Extraction is passed in as a plain function (text -> TripRequest), so the same
graph runs with Gemini in the app and with fixed offline data in the tests.

Run from the project root (one Gemini call):
    uv run python -m ai_travel_assistant.trip_workflow "3 days in Ella, love hiking"

Draw the graph into docs/ as a Mermaid diagram (offline, no Gemini call):
    uv run python -m ai_travel_assistant.trip_workflow --draw

Also save it as a PNG image (sends the diagram text to the mermaid.ink service):
    uv run python -m ai_travel_assistant.trip_workflow --draw --png

No tools, memory, or multiple agents here.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ai_travel_assistant.schemas import CRITICAL_FIELDS, TripRequest

# --- State --------------------------------------------------------------------


class TripState(TypedDict, total=False):
    """The data that flows through the graph.

    total=False means every key is optional. That lets us start the graph with
    only {"user_input": ...}; the nodes fill in the other keys as they run.
    """

    user_input: str  # the traveller's raw message
    destination: str | None  # e.g. "Ella", or None if not given
    days: int | None  # e.g. 3, or None if not given
    missing_fields: list[str]  # e.g. ["days"]; empty when the request is complete
    response: str  # the final message for the traveller


# Any function that turns text into a TripRequest. In the app this calls Gemini.
# In the tests it is a dictionary lookup.
TripExtractor = Callable[[str], TripRequest]

# Node names, kept in one place so the edges and tests can't drift apart.
TRIP_INPUT = "trip_input"
VALIDATION = "validation"
ASK_CLARIFICATION = "ask_clarification"
RESPONSE = "response"


# --- Nodes --------------------------------------------------------------------
# A node takes the current state and returns a dict containing only the keys it
# wants to change. LangGraph merges that dict into the state.


def make_trip_input_node(extractor: TripExtractor) -> Callable[[TripState], dict]:
    """Build the trip_input node around the given extractor.

    A node receives only the state, so we use a closure: the inner function
    "remembers" the extractor it was created with.
    """

    def trip_input_node(state: TripState) -> dict:
        trip = extractor(state["user_input"])
        return {"destination": trip.destination, "days": trip.days}

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


def build_trip_graph(extractor: TripExtractor) -> CompiledStateGraph:
    """Wire the nodes and edges together and compile the graph."""
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

    builder.add_edge(ASK_CLARIFICATION, END)
    builder.add_edge(RESPONSE, END)

    # compile() checks the wiring (e.g. no edge to an unknown node) and returns
    # a runnable graph with .invoke().
    return builder.compile()


def run_trip_workflow(extractor: TripExtractor, user_input: str) -> TripState:
    """Build the graph, run it once, and return the final state."""
    graph = build_trip_graph(extractor)
    return graph.invoke({"user_input": user_input})


# --- Visualization (offline) --------------------------------------------------

DEFAULT_DIAGRAM_PATH = Path("docs/trip_workflow_graph.md")


def _drawable_graph():
    """Return a description of the graph's nodes and edges, ready to draw.

    The graph's shape does not depend on the extractor, and drawing never runs a
    node, so a placeholder extractor is enough. No Gemini call is made.
    """

    def placeholder_extractor(text: str) -> TripRequest:
        return TripRequest()

    return build_trip_graph(placeholder_extractor).get_graph()


def render_mermaid() -> str:
    """Return the graph as Mermaid diagram text (made locally, no network)."""
    return _drawable_graph().draw_mermaid()


def render_png(path: Path) -> Path:
    """Save the graph as a PNG image.

    Uses LangGraph's default method: the Mermaid text (node names and edges only,
    no keys or user data) is sent to the mermaid.ink web service, which returns
    the image. Needs an internet connection.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    _drawable_graph().draw_mermaid_png(output_file_path=str(path))
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
        "Solid arrows are fixed edges. Dashed arrows are the conditional edge: "
        "`route_after_validation` picks `ask_clarification` when destination or "
        "days is missing, and `response` otherwise.\n\n"
        f"{image}"
        f"```mermaid\n{render_mermaid().strip()}\n```\n"
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
    if args.text is None:
        parser.error("give a trip request, or use --draw")

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

    def gemini_extractor(text: str) -> TripRequest:
        return extract_trip_request(llm, text)

    try:
        state = run_trip_workflow(gemini_extractor, args.text)
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
