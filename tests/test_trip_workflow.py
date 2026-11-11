"""Stage 2 checks: the trip-request LangGraph workflow, single- and multi-turn.

All tests are offline. Instead of Gemini, the graph gets a fake extractor that
looks up a fixed TripRequest for each message, so no API key or network is used.
"""

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from ai_travel_assistant import trip_workflow
from ai_travel_assistant.schemas import TripRequest
from ai_travel_assistant.trip_workflow import (
    ASK_CLARIFICATION,
    RESPONSE,
    WAIT_FOR_REPLY,
    ask_clarification_node,
    build_trip_graph,
    is_awaiting_reply,
    make_trip_input_node,
    new_thread_id,
    response_node,
    route_after_validation,
    run_chat,
    run_trip_workflow,
    send_message,
    thread_config,
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


def test_mermaid_shows_the_clarification_loop():
    mermaid = trip_workflow.render_mermaid()
    assert "ask_clarification --> wait_for_reply;" in mermaid
    assert "wait_for_reply --> trip_input;" in mermaid


def _graph_edges(mermaid: str) -> set[tuple[str, str]]:
    """(source, target) for every edge line between two graph nodes."""
    nodes = set(trip_workflow._drawable_graph().nodes)
    edges = set()
    for line in mermaid.splitlines():
        parts = line.strip().rstrip(";").split()
        if len(parts) >= 3 and parts[0] in nodes and parts[-1] in nodes:
            edges.add((parts[0], parts[-1]))
    return edges


def test_annotated_diagram_has_exactly_the_compiled_graph_edges():
    compiled = {(e.source, e.target) for e in trip_workflow._drawable_graph().edges}
    annotated = trip_workflow.render_annotated_mermaid()
    # Labels and extras must not add, drop or invent graph edges.
    assert _graph_edges(annotated) == compiled


def test_annotated_diagram_labels_routing_and_marks_new_parts():
    mermaid = trip_workflow.render_annotated_mermaid()
    assert 'validation -.->|"missing destination or days"| ask_clarification;' in (
        mermaid
    )
    assert 'validation -.->|"all details present"| response;' in mermaid
    assert "Command(resume=reply), same thread_id" in mermaid
    assert "InMemorySaver checkpointer" in mermaid
    assert "class wait_for_reply new;" in mermaid
    assert "class trip_input modified;" in mermaid


def test_annotated_diagram_fails_if_a_labelled_edge_disappears(monkeypatch):
    real = trip_workflow.render_mermaid()
    without_edge = real.replace("\tvalidation -.-> response;\n", "")
    monkeypatch.setattr(trip_workflow, "render_mermaid", lambda: without_edge)
    with pytest.raises(ValueError, match="validation -> response"):
        trip_workflow.render_annotated_mermaid()


def test_png_is_drawn_from_the_annotated_diagram(tmp_path, monkeypatch):
    sent: list[str] = []

    def fake_draw(mermaid_syntax, output_file_path):  # stands in for mermaid.ink
        sent.append(mermaid_syntax)

    monkeypatch.setattr(trip_workflow, "draw_mermaid_png", fake_draw)
    trip_workflow.render_png(tmp_path / "graph.png")
    assert sent == [trip_workflow.render_annotated_mermaid()]


def test_diagram_doc_has_annotated_and_raw_diagrams(tmp_path):
    text = trip_workflow.write_diagram_doc(tmp_path / "graph.md").read_text(
        encoding="utf-8"
    )
    assert text.count("```mermaid") == 2
    assert trip_workflow.render_annotated_mermaid().strip() in text
    assert trip_workflow.render_mermaid().strip() in text


# --- Multi-turn conversations (checkpointer + interrupt + resume) -------------
# Follow-up replies get their own prepared answers. Each fills only the field
# the traveller mentioned, as Gemini would for a short reply.
FOLLOW_UPS: dict[str, TripRequest] = {
    "I want to visit Kandy.": TripRequest(destination="Kandy"),
    "I want to travel for 5 days.": TripRequest(days=5),
    "3 days.": TripRequest(days=3),
    "Ella.": TripRequest(destination="Ella"),
    "I prefer hiking": TripRequest(interests=["hiking"]),
    "Actually, make it 5 days": TripRequest(days=5),
    "Make it Galle instead": TripRequest(destination="Galle"),
}


class CountingExtractor:
    """Fake extractor that records every message it is asked about.

    The record shows how many times the real API would have been called.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, text: str) -> TripRequest:
        self.calls.append(text)
        return {**OFFLINE_TRIPS, **FOLLOW_UPS}[text]


@pytest.fixture
def extractor() -> CountingExtractor:
    return CountingExtractor()


@pytest.fixture
def chat_graph(extractor):
    """A multi-turn graph with fresh in-memory conversation storage."""
    return build_trip_graph(extractor, checkpointer=InMemorySaver())


def test_wait_for_reply_node_is_only_in_the_multi_turn_graph(chat_graph):
    single_turn = build_trip_graph(fake_extractor)
    assert WAIT_FOR_REPLY not in single_turn.get_graph().nodes
    assert WAIT_FOR_REPLY in chat_graph.get_graph().nodes


def test_multi_turn_complete_request_finishes_in_one_turn(chat_graph, extractor):
    state = send_message(chat_graph, "t1", "3 days in Ella, love hiking")
    assert state["response"] == "Great! I've noted a 3-day trip to Ella."
    assert state["missing_fields"] == []
    assert not is_awaiting_reply(chat_graph, "t1")
    assert extractor.calls == ["3 days in Ella, love hiking"]


@pytest.mark.parametrize(
    ("first_message", "question"),
    [
        ("I want to visit Kandy.", "How many days would you like to spend in Kandy?"),
        (
            "I want to travel for 5 days.",
            "Where in Sri Lanka would you like to spend your 5 days?",
        ),
        (
            "Where should I go in Sri Lanka?",
            "Where in Sri Lanka would you like to go, and for how many days?",
        ),
    ],
)
def test_incomplete_request_pauses_with_a_question(chat_graph, first_message, question):
    state = send_message(chat_graph, "t1", first_message)
    assert state["response"] == question
    assert is_awaiting_reply(chat_graph, "t1")
    # The run stopped inside wait_for_reply, and the interrupt carries the
    # question for the app to show.
    snapshot = chat_graph.get_state(thread_config("t1"))
    assert snapshot.next == (WAIT_FOR_REPLY,)
    assert [i.value for i in snapshot.interrupts] == [question]


def test_resume_with_missing_days(chat_graph, extractor):
    send_message(chat_graph, "t1", "I want to visit Kandy.")
    state = send_message(chat_graph, "t1", "3 days.")
    assert state["destination"] == "Kandy"  # kept from the first turn
    assert state["days"] == 3
    assert state["response"] == "Great! I've noted a 3-day trip to Kandy."
    assert not is_awaiting_reply(chat_graph, "t1")
    # One extraction per message: resuming did not repeat the first call.
    assert extractor.calls == ["I want to visit Kandy.", "3 days."]


def test_resume_with_missing_destination(chat_graph):
    send_message(chat_graph, "t1", "I want to travel for 5 days.")
    state = send_message(chat_graph, "t1", "Ella.")
    assert (state["destination"], state["days"]) == ("Ella", 5)
    assert state["response"] == "Great! I've noted a 5-day trip to Ella."


def test_resume_with_command_directly(chat_graph):
    # The raw LangGraph calls that send_message wraps.
    config = thread_config("t1")
    result = chat_graph.invoke({"user_input": "I want to visit Kandy."}, config)
    assert "__interrupt__" in result
    result = chat_graph.invoke(Command(resume="3 days."), config)
    assert result["response"] == "Great! I've noted a 3-day trip to Kandy."


def test_multiple_clarification_rounds(chat_graph, extractor):
    state = send_message(chat_graph, "t1", "Where should I go in Sri Lanka?")
    assert state["missing_fields"] == ["destination", "days"]

    state = send_message(chat_graph, "t1", "Ella.")
    assert state["missing_fields"] == ["days"]
    assert state["response"] == "How many days would you like to spend in Ella?"
    assert is_awaiting_reply(chat_graph, "t1")

    state = send_message(chat_graph, "t1", "3 days.")
    assert state["response"] == "Great! I've noted a 3-day trip to Ella."
    assert extractor.calls == ["Where should I go in Sri Lanka?", "Ella.", "3 days."]


def test_unhelpful_reply_keeps_details_and_asks_again(chat_graph):
    send_message(chat_graph, "t1", "I want to visit Kandy.")
    state = send_message(chat_graph, "t1", "I prefer hiking")  # no days given
    assert state["destination"] == "Kandy"  # not overwritten with None
    assert state["response"] == "How many days would you like to spend in Kandy?"
    assert is_awaiting_reply(chat_graph, "t1")


@pytest.mark.parametrize("reply", ["", "   "])
def test_blank_reply_asks_again_without_calling_the_extractor(
    chat_graph, extractor, reply
):
    send_message(chat_graph, "t1", "I want to visit Kandy.")
    state = send_message(chat_graph, "t1", reply)
    assert state["response"] == "How many days would you like to spend in Kandy?"
    assert extractor.calls == ["I want to visit Kandy."]


def test_non_text_resume_value_asks_again(chat_graph, extractor):
    config = thread_config("t1")
    chat_graph.invoke({"user_input": "I want to visit Kandy."}, config)
    chat_graph.invoke(Command(resume={"days": 3}), config)  # not a string
    assert is_awaiting_reply(chat_graph, "t1")
    assert extractor.calls == ["I want to visit Kandy."]


def test_correction_replaces_only_the_mentioned_field(chat_graph):
    send_message(chat_graph, "t1", "I want to visit Kandy.")
    send_message(chat_graph, "t1", "3 days.")
    state = send_message(chat_graph, "t1", "Actually, make it 5 days")
    assert (state["destination"], state["days"]) == ("Kandy", 5)
    assert state["response"] == "Great! I've noted a 5-day trip to Kandy."


def test_completed_conversation_treats_new_message_as_an_update(chat_graph):
    send_message(chat_graph, "t1", "3 days in Ella, love hiking")
    # Finished threads get a new run on the same saved trip, not a no-op
    # "resume" of a run that is not paused.
    state = send_message(chat_graph, "t1", "Make it Galle instead")
    assert (state["destination"], state["days"]) == ("Galle", 3)
    assert state["response"] == "Great! I've noted a 3-day trip to Galle."
    assert not is_awaiting_reply(chat_graph, "t1")


def test_threads_are_separate_conversations(chat_graph):
    send_message(chat_graph, "kandy", "I want to visit Kandy.")
    send_message(chat_graph, "five-days", "I want to travel for 5 days.")

    kandy = send_message(chat_graph, "kandy", "3 days.")
    ella = send_message(chat_graph, "five-days", "Ella.")

    assert (kandy["destination"], kandy["days"]) == ("Kandy", 3)
    assert (ella["destination"], ella["days"]) == ("Ella", 5)


def test_no_state_leaks_into_a_new_thread(chat_graph):
    send_message(chat_graph, "first", "3 days in Ella, love hiking")
    state = send_message(chat_graph, "second", "Where should I go in Sri Lanka?")
    assert state.get("destination") is None
    assert state.get("days") is None
    assert state["missing_fields"] == ["destination", "days"]


def test_same_thread_id_restores_state_in_a_new_graph(extractor):
    # Two graph objects sharing one checkpointer act like an app restart that
    # kept its storage: the thread continues where it stopped.
    memory = InMemorySaver()
    first = build_trip_graph(extractor, checkpointer=memory)
    send_message(first, "t1", "I want to travel for 5 days.")
    second = build_trip_graph(extractor, checkpointer=memory)
    state = send_message(second, "t1", "Ella.")
    assert state["response"] == "Great! I've noted a 5-day trip to Ella."


def test_new_thread_ids_are_unique():
    assert new_thread_id() != new_thread_id()


def test_run_chat_asks_until_the_trip_is_complete(extractor):
    replies = iter(["Ella.", "I prefer hiking", "3 days."])
    shown: list[str] = []
    state = run_chat(
        extractor,
        "Where should I go in Sri Lanka?",
        read=lambda prompt: next(replies),
        write=shown.append,
    )
    assert shown == [
        "AI: Where in Sri Lanka would you like to go, and for how many days?",
        "AI: How many days would you like to spend in Ella?",
        "AI: How many days would you like to spend in Ella?",
        "AI: Great! I've noted a 3-day trip to Ella.",
    ]
    assert (state["destination"], state["days"]) == ("Ella", 3)


def test_run_chat_reads_the_first_message_when_none_is_given(extractor):
    replies = iter(["3 days in Ella, love hiking"])
    shown: list[str] = []
    run_chat(extractor, read=lambda prompt: next(replies), write=shown.append)
    assert shown == ["AI: Great! I've noted a 3-day trip to Ella."]


def test_cli_needs_text_unless_chatting():
    with pytest.raises(SystemExit):
        trip_workflow.main([])
