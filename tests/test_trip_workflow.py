"""Stage 2 checks: the trip-request LangGraph workflow.

All tests are offline. Instead of Gemini, the graph gets a fake extractor that
looks up a fixed TripRequest for each message, so no API key or network is used.
"""

import pytest

from ai_travel_assistant import trip_workflow
from ai_travel_assistant.schemas import TripRequest
from ai_travel_assistant.trip_workflow import (
    ASK_CLARIFICATION,
    RESPONSE,
    ask_clarification_node,
    build_trip_graph,
    make_trip_input_node,
    response_node,
    route_after_validation,
    run_trip_workflow,
    validation_node,
)

# --- Offline test data --------------------------------------------------------
# Each message maps to the TripRequest that Gemini would be expected to return.
OFFLINE_TRIPS: dict[str, TripRequest] = {
    "3 days in Ella, love hiking": TripRequest(
        destination="Ella", days=3, interests=["hiking"]
    ),
    "A week in Galle please": TripRequest(destination="Galle", days=7),
    "I want to visit Kandy": TripRequest(destination="Kandy"),
    "I have 5 days free": TripRequest(days=5),
    "Where should I go in Sri Lanka?": TripRequest(),
}


def fake_extractor(text: str) -> TripRequest:
    """Stand-in for Gemini: returns the prepared answer for a known message."""
    return OFFLINE_TRIPS[text]


# --- Individual nodes ---------------------------------------------------------


def test_trip_input_node_copies_destination_and_days():
    node = make_trip_input_node(fake_extractor)
    update = node({"user_input": "3 days in Ella, love hiking"})
    # A node returns only the keys it changes.
    assert update == {"destination": "Ella", "days": 3}


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ({"destination": "Ella", "days": 3}, []),
        ({"destination": "Kandy", "days": None}, ["days"]),
        ({"destination": None, "days": 5}, ["destination"]),
        ({"destination": None, "days": None}, ["destination", "days"]),
        ({"destination": "   ", "days": 2}, ["destination"]),  # blank = missing
        ({}, ["destination", "days"]),  # keys not set yet
    ],
)
def test_validation_node_finds_missing_fields(state, expected):
    assert validation_node(state) == {"missing_fields": expected}


def test_router_picks_clarification_when_fields_are_missing():
    assert route_after_validation({"missing_fields": ["days"]}) == ASK_CLARIFICATION


def test_router_picks_response_when_nothing_is_missing():
    assert route_after_validation({"missing_fields": []}) == RESPONSE


def test_response_node_confirms_the_trip():
    update = response_node({"destination": "Ella", "days": 3})
    assert update == {"response": "Great! I've noted a 3-day trip to Ella."}


@pytest.mark.parametrize(
    ("state", "expected_text"),
    [
        (
            {
                "destination": None,
                "days": None,
                "missing_fields": ["destination", "days"],
            },
            "Where in Sri Lanka would you like to go, and for how many days?",
        ),
        (
            {"destination": None, "days": 5, "missing_fields": ["destination"]},
            "Where in Sri Lanka would you like to spend your 5 days?",
        ),
        (
            {"destination": "Kandy", "days": None, "missing_fields": ["days"]},
            "How many days would you like to spend in Kandy?",
        ),
    ],
)
def test_clarification_node_asks_only_for_missing_details(state, expected_text):
    assert ask_clarification_node(state) == {"response": expected_text}


# --- Whole graph --------------------------------------------------------------


def test_graph_contains_all_four_nodes():
    graph = build_trip_graph(fake_extractor)
    nodes = set(graph.get_graph().nodes)
    assert {"trip_input", "validation", "ask_clarification", "response"} <= nodes


def test_complete_request_gets_a_confirmation():
    state = run_trip_workflow(fake_extractor, "3 days in Ella, love hiking")
    assert state == {
        "user_input": "3 days in Ella, love hiking",
        "destination": "Ella",
        "days": 3,
        "missing_fields": [],
        "response": "Great! I've noted a 3-day trip to Ella.",
    }


@pytest.mark.parametrize(
    ("user_input", "missing", "expected_text"),
    [
        (
            "I want to visit Kandy",
            ["days"],
            "How many days would you like to spend in Kandy?",
        ),
        (
            "I have 5 days free",
            ["destination"],
            "Where in Sri Lanka would you like to spend your 5 days?",
        ),
        (
            "Where should I go in Sri Lanka?",
            ["destination", "days"],
            "Where in Sri Lanka would you like to go, and for how many days?",
        ),
    ],
)
def test_incomplete_request_gets_a_clarification(user_input, missing, expected_text):
    state = run_trip_workflow(fake_extractor, user_input)
    assert state["missing_fields"] == missing
    assert state["response"] == expected_text


def test_only_one_branch_runs():
    # stream_mode="updates" yields one item per node that ran, in order.
    graph = build_trip_graph(fake_extractor)
    complete = [
        name
        for step in graph.stream({"user_input": "A week in Galle please"})
        for name in step
    ]
    incomplete = [
        name
        for step in graph.stream({"user_input": "I want to visit Kandy"})
        for name in step
    ]
    assert complete == ["trip_input", "validation", "response"]
    assert incomplete == ["trip_input", "validation", "ask_clarification"]


def test_extractor_receives_the_user_input():
    seen: list[str] = []

    def recording_extractor(text: str) -> TripRequest:
        seen.append(text)
        return TripRequest(destination="Ella", days=2)

    run_trip_workflow(recording_extractor, "2 days in Ella")
    assert seen == ["2 days in Ella"]


def test_module_does_not_load_gemini_on_import():
    # The Gemini setup is imported inside main(), so building and running the
    # graph never needs an API key.
    assert not hasattr(trip_workflow, "build_llm")


# --- Visualization ------------------------------------------------------------


def test_mermaid_shows_fixed_and_conditional_edges():
    mermaid = trip_workflow.render_mermaid()
    assert "trip_input --> validation;" in mermaid  # fixed edge (solid)
    assert "validation -.-> ask_clarification;" in mermaid  # conditional (dashed)
    assert "validation -.-> response;" in mermaid


def test_draw_flag_writes_diagram_without_gemini(tmp_path, monkeypatch):
    # If --draw ever tried to set up Gemini, this would fail the test.
    def no_gemini():
        raise AssertionError("--draw must not load Gemini settings")

    monkeypatch.setattr("ai_travel_assistant.llm_basics.load_settings", no_gemini)
    out = tmp_path / "graph.md"

    assert trip_workflow.main(["--draw", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "```mermaid" in text
    assert "validation -.-> response;" in text


def test_png_flag_links_the_image_without_network(tmp_path, monkeypatch):
    # Replace the mermaid.ink call with a fake that writes dummy bytes offline.
    def fake_render_png(path):
        path.write_bytes(b"fake png")
        return path

    monkeypatch.setattr(trip_workflow, "render_png", fake_render_png)
    out = tmp_path / "graph.md"

    assert trip_workflow.main(["--draw", str(out), "--png"]) == 0
    assert (tmp_path / "graph.png").read_bytes() == b"fake png"
    text = out.read_text(encoding="utf-8")
    assert "![Trip workflow graph](graph.png)" in text
    assert "--draw --png" in text


def test_png_flag_needs_draw():
    with pytest.raises(SystemExit):
        trip_workflow.main(["--png"])
