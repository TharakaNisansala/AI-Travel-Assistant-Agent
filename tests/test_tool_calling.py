"""Stage 3 checks: the search_destinations tool and the tool-calling agent.

The offline tests never call Gemini. A scripted fake model plays the LLM: it
returns prepared AIMessages (first a tool call, then a final answer), so we can
check the whole Reason -> Act -> Observe loop without a network or API key.

The live test calls Gemini (about 2 requests) and runs only with:
    uv run pytest -m live
"""

from __future__ import annotations

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from ai_travel_assistant import llm_basics, tool_calling
from ai_travel_assistant.tool_calling import (
    LLM,
    SYSTEM_PROMPT,
    TOOLS,
    TOOLS_NODE,
    build_tool_agent_graph,
    describe_steps,
    final_answer,
    run_tool_agent,
)
from ai_travel_assistant.tools import (
    DESTINATIONS,
    find_destinations,
    search_destinations,
)

# --- A fake model that supports bind_tools ------------------------------------


class ScriptedToolModel(GenericFakeChatModel):
    """Returns the given AIMessages in order, like a scripted LLM.

    LangChain's fake models do not implement bind_tools(), so we add a version
    that just records which tools were bound. We also record the messages the
    model receives on each call, to check what the LLM "saw".
    """

    bound_tools: list = []
    calls: list = []

    def bind_tools(self, tools, **kwargs):
        self.bound_tools = list(tools)
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def scripted(*responses: AIMessage) -> ScriptedToolModel:
    return ScriptedToolModel(messages=iter(responses))


def tool_call_message(query: str, call_id: str = "call_1") -> AIMessage:
    """What a tool-calling LLM returns when it wants search_destinations run."""
    return AIMessage(
        content="",
        tool_calls=[
            {"name": "search_destinations", "args": {"query": query}, "id": call_id}
        ],
    )


# --- The tool on its own ------------------------------------------------------


def test_tool_has_name_description_and_schema():
    # These three are what bind_tools() sends to the LLM.
    assert search_destinations.name == "search_destinations"
    assert "Sri Lankan destinations" in search_destinations.description
    assert set(search_destinations.args) == {"query"}


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Find places in Sri Lanka for hiking", "Ella"),
        ("hike", "Ella"),  # a different word form still matches
        ("beaches for surfing", "Arugam Bay"),
        ("whale watching", "Mirissa"),
        ("leopards and safari", "Yala National Park"),
        ("ancient history", "Sigiriya"),
        ("Kandy", "Kandy"),  # a place name works too
    ],
)
def test_find_destinations_returns_matching_places(query, expected):
    names = [d.name for d in find_destinations(query)]
    assert expected in names


def test_best_match_comes_first():
    # Mirissa matches "whale", "watching" and "beach" (3 points); Galle and
    # Arugam Bay match only "beach", Yala only "watching" (1 point each).
    names = [d.name for d in find_destinations("whale watching on the beach")]
    assert names[0] == "Mirissa"


def test_sri_lanka_alone_matches_nothing():
    # Every place is in Sri Lanka, so these words must not match everything.
    assert find_destinations("places in Sri Lanka") == []


def test_tool_output_lists_name_location_and_activities():
    text = search_destinations.invoke({"query": "hiking"})
    assert "Ella (Uva Province, Badulla District)" in text
    assert "Little Adam's Peak" in text
    assert "offline guide (not live data)" in text


@pytest.mark.parametrize("query", ["skiing on glaciers", "", "   "])
def test_tool_reports_when_nothing_matches(query):
    text = search_destinations.invoke({"query": query})
    assert text.startswith("No destinations in the offline guide matched")


def test_every_destination_has_name_location_and_activities():
    for destination in DESTINATIONS:
        assert destination.name and destination.location and destination.activities


# --- Routing and tool execution, piece by piece -------------------------------


def test_tools_condition_routes_tool_calls_to_the_tools_node():
    state = {"messages": [HumanMessage("hiking?"), tool_call_message("hiking")]}
    assert tools_condition(state) == TOOLS_NODE


def test_tools_condition_ends_on_a_plain_answer():
    state = {"messages": [HumanMessage("hi"), AIMessage("Hello!")]}
    assert tools_condition(state) == END


def test_tool_node_runs_the_call_and_returns_a_tool_message():
    # ToolNode needs the runtime a graph provides, so run it in a one-node graph.
    builder = StateGraph(MessagesState)
    builder.add_node(TOOLS_NODE, ToolNode([search_destinations]))
    builder.add_edge(START, TOOLS_NODE)
    graph = builder.compile()

    result = graph.invoke({"messages": [tool_call_message("hiking", "call_42")]})
    message = result["messages"][-1]
    assert isinstance(message, ToolMessage)
    # tool_call_id links the result to the call that asked for it.
    assert message.tool_call_id == "call_42"
    assert message.name == "search_destinations"
    assert "Ella" in message.content


# --- The whole graph with a scripted model ------------------------------------


def test_graph_has_llm_and_tools_nodes():
    graph = build_tool_agent_graph(scripted())
    assert {LLM, TOOLS_NODE} <= set(graph.get_graph().nodes)


def test_build_binds_every_tool_in_order():
    # The one exact check of the tool list; feature tests only check that
    # their own tool is bound, so adding a tool changes just this test.
    model = scripted()
    build_tool_agent_graph(model)
    assert model.bound_tools[0] is search_destinations
    assert [t.name for t in model.bound_tools] == [
        "search_destinations",
        "get_weather_forecast",
        "find_attractions",
    ]


def test_tool_names_are_unique():
    names = [t.name for t in TOOLS]
    assert len(names) == len(set(names))


def test_system_prompt_gives_guidance_for_every_tool():
    # A registered tool with no selection guidance is easy to miss when adding
    # a tool; this catches it.
    for tool in TOOLS:
        assert tool.name in SYSTEM_PROMPT, f"{tool.name} is not in SYSTEM_PROMPT"


def test_graph_can_run_every_registered_tool():
    # ToolNode only runs tools it was given: it must hold exactly TOOLS.
    graph = build_tool_agent_graph(scripted())
    tool_node = graph.nodes[TOOLS_NODE].bound
    assert set(tool_node.tools_by_name) == {t.name for t in TOOLS}


def test_tool_call_then_final_answer():
    model = scripted(
        tool_call_message("hiking"),
        AIMessage("Try Ella, Nuwara Eliya or Adam's Peak (from the built-in guide)."),
    )
    graph = build_tool_agent_graph(model)

    messages = run_tool_agent(graph, "Find places in Sri Lanka for hiking")

    # Human -> AI (tool call) -> Tool (result) -> AI (final answer)
    assert [type(m) for m in messages] == [
        HumanMessage,
        AIMessage,
        ToolMessage,
        AIMessage,
    ]
    assert messages[2].tool_call_id == "call_1"
    assert "Ella" in messages[2].content
    assert final_answer(messages).startswith("Try Ella")
    assert len(model.calls) == 2  # the LLM ran twice: decide, then answer


def test_llm_sees_the_tool_result_on_its_second_call():
    model = scripted(tool_call_message("hiking"), AIMessage("Ella is great."))
    run_tool_agent(build_tool_agent_graph(model), "hiking?")

    first, second = model.calls
    # The system prompt is sent every time but is not stored in the history.
    assert isinstance(first[0], SystemMessage)
    assert first[0].content == SYSTEM_PROMPT
    # Observe: the second call ends with the tool's result.
    assert isinstance(second[-1], ToolMessage)
    assert "Ella" in second[-1].content


def test_direct_answer_skips_the_tools():
    model = scripted(AIMessage("Ayubowan! How can I help?"))
    messages = run_tool_agent(build_tool_agent_graph(model), "Hello")
    assert [type(m) for m in messages] == [HumanMessage, AIMessage]
    assert len(model.calls) == 1


def test_no_match_result_reaches_the_llm():
    model = scripted(
        tool_call_message("skiing"),
        AIMessage("Sorry, the guide has no skiing spots."),
    )
    messages = run_tool_agent(build_tool_agent_graph(model), "Where can I ski?")
    assert messages[2].content.startswith("No destinations")
    assert "no skiing" in final_answer(messages)


def test_describe_steps_shows_the_loop():
    model = scripted(tool_call_message("hiking"), AIMessage("Ella."))
    messages = run_tool_agent(build_tool_agent_graph(model), "hiking?")
    steps = describe_steps(messages)
    assert steps[0] == "[human] hiking?"
    assert steps[1] == "[llm -> tool call] search_destinations({'query': 'hiking'})"
    assert steps[2].startswith("[tool result] Found")
    assert steps[3] == "[llm -> final answer]"


# --- CLI (with the model and settings replaced) -------------------------------


def test_cli_prints_the_final_answer(monkeypatch, capsys):
    settings = llm_basics.GeminiSettings(model="fake-model", api_key="fake-key")
    model = scripted(tool_call_message("hiking"), AIMessage("Go to Ella."))
    monkeypatch.setattr(llm_basics, "load_settings", lambda: settings)
    monkeypatch.setattr(llm_basics, "build_llm", lambda s, temperature: model)

    assert tool_calling.main(["--steps", "Find places for hiking"]) == 0
    out = capsys.readouterr().out
    assert "[llm -> tool call] search_destinations" in out
    assert out.strip().endswith("Go to Ella.")


def test_cli_reports_missing_configuration(monkeypatch, capsys):
    def no_settings():
        raise llm_basics.ConfigError("Missing GEMINI_API_KEY.")

    monkeypatch.setattr(llm_basics, "load_settings", no_settings)
    assert tool_calling.main(["hiking"]) == 1
    assert "Configuration error" in capsys.readouterr().err


# --- Live (real Gemini, skipped by default) -----------------------------------


@pytest.mark.live
def test_live_agent_calls_the_search_tool():
    settings = llm_basics.load_settings()
    graph = build_tool_agent_graph(llm_basics.build_llm(settings))
    messages = run_tool_agent(graph, "Find places in Sri Lanka for hiking")
    tool_calls = [
        call for m in messages if isinstance(m, AIMessage) for call in m.tool_calls
    ]
    assert any(call["name"] == "search_destinations" for call in tool_calls)
    assert any(isinstance(m, ToolMessage) for m in messages)
    assert final_answer(messages)
