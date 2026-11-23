"""Stage 3 checks: the search_destinations tool and the tool-calling agent.

The offline tests never call Gemini. A scripted fake model plays the LLM: it
returns prepared AIMessages (first a tool call, then a final answer), so we can
check the whole Reason -> Act -> Observe loop without a network or API key.

The live tests call Gemini (2-3 requests each). Run them ONE AT A TIME by node
ID, at least 60 seconds apart, e.g. (one command, shown on two lines):
    uv run pytest -m live
        tests/test_tool_calling.py::test_live_g1_agent_finds_nature_attractions_in_ella
Never run the whole live group (`pytest -m live` with no node ID): it includes
a 10-request extraction test and would spend most of the daily quota.
"""

from __future__ import annotations

import re

import httpx
import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from ai_travel_assistant import (
    attractions_service,
    llm_basics,
    tool_calling,
    weather_service,
)
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
    find_attractions,
    find_destinations,
    get_weather_forecast,
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
        "get_destination_guide",
        "estimate_travel_time",
        "convert_currency",
        "get_attraction_prices",
        "suggest_hotels",
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


# --- Stage 4 live validation: the agent with real tools (G1, G2) --------------
# Policy for these tests:
#   * One attempt per Gemini request. build_llm() sets max_retries=3, which the
#     Google SDK treats as 3 ATTEMPTS (1 request + 2 automatic retries on
#     408/429/5xx). The tests use a copy with max_retries=1: no retries.
#   * Gemini 429 (rate limit / quota): stop the whole pytest session at once.
#   * Skipped as inconclusive (infrastructure, not our code): Gemini 503; a busy
#     or slow Overpass server (HTTP 429/502/503/504, a runtime-error remark, or
#     a client timeout); Open-Meteo timeouts,
#     connection failures, HTTP 429 and HTTP 5xx.
#   * Still genuine failures: any other Gemini error, invalid or unreadable
#     responses, other HTTP 4xx, and every bug in our own code.
#   * Assertions check the trajectory (which tools, which arguments) and that
#     the answer is grounded in tool output, never the exact wording.

INCONCLUSIVE = "INCONCLUSIVE (infrastructure, not a code failure)"
# Tool error texts (the tools return failures as text, not exceptions) that
# come from temporary infrastructure problems. The wording is produced by
# attractions_service and weather_service; the offline tests below generate it
# with the real code, so a wording change there breaks them instead of
# silently turning a skip into a failure (or the other way round).
_TRANSIENT_TOOL_ERRORS = re.compile(
    r"Overpass API is busy \(HTTP"
    r"|Overpass API could not finish the search"
    r"|Overpass API did not respond within"
    r"|Overpass API returned HTTP 50[23]\."
    r"|Open-Meteo (geocoding|forecast) API did not respond within"
    r"|Could not connect to the Open-Meteo (geocoding|forecast) API"
    r"|Open-Meteo (geocoding|forecast) API returned HTTP (429|5\d\d)"
)
# One result line of format_attractions_report: "- <name> (<kind>), 1.2 km".
ATTRACTION_LINE = re.compile(r"^- (?P<name>.+) \([^()]+\), [\d.]+ km$", re.M)
# Each Gemini request adds one AIMessage. Two is the minimum (tool call, then
# answer); one retry with category "all" makes three. More means a loop.
MAX_GEMINI_REQUESTS = 3


def no_retry_llm(settings: llm_basics.GeminiSettings):
    """The app's model, but with the SDK's automatic retries switched off.

    max_retries=1 means "1 attempt"; 0 would mean "use the SDK default" (5).
    """
    return llm_basics.build_llm(settings).model_copy(update={"max_retries": 1})


def test_no_retry_llm_sends_a_single_attempt():
    # Offline: building a request config makes no network call.
    settings = llm_basics.GeminiSettings(model="gemini-test", api_key="fake-key")
    request = no_retry_llm(settings)._prepare_request([HumanMessage("hi")])
    assert request["config"].http_options.retry_options.attempts == 1


def run_with_gemini_policy(graph, question: str, api_key: str) -> list:
    """Run the agent once, applying the policy above to Gemini errors."""
    from langchain_google_genai.chat_models import GoogleAPIError, GoogleRateLimitError

    try:
        return run_tool_agent(graph, question)
    except GoogleRateLimitError as exc:
        reason = llm_basics.safe_error_message(exc, api_key)
        pytest.exit(f"STOP: Gemini 429. Do not re-run live tests today. {reason}")
    except GoogleAPIError as exc:
        if exc.code == 503:
            reason = llm_basics.safe_error_message(exc, api_key)
            pytest.skip(f"{INCONCLUSIVE}: Gemini 503. {reason}")
        raise


def run_live_agent(question: str) -> list:
    """Run the real agent (real Gemini, one attempt per request) once."""
    settings = llm_basics.load_settings()
    graph = build_tool_agent_graph(no_retry_llm(settings))
    return run_with_gemini_policy(graph, question, settings.api_key)


def skip_if_infrastructure_failed(tool_text: str) -> None:
    """Skip the test as inconclusive if a tool failed for a temporary reason."""
    failed = tool_text.startswith(("Attractions unavailable", "Weather unavailable"))
    if failed and _TRANSIENT_TOOL_ERRORS.search(tool_text):
        pytest.skip(f"{INCONCLUSIVE}: {tool_text.splitlines()[0]}")


# --- Offline checks of the live-test policy (no network, no Gemini) -----------

FAKE_KEY = "fake-key-1234"


class RaisingGraph:
    """Stands in for the compiled graph: invoke() raises the given error."""

    def __init__(self, error: Exception):
        self.error = error

    def invoke(self, state):
        raise self.error


def gemini_server_error(code: int):
    from langchain_google_genai.chat_models import GoogleAPIError

    return GoogleAPIError(code=code, response_json={"error": {"message": "x"}})


def gemini_client_error(code: int):
    """A Gemini client error other than 429, e.g. 400 invalid request."""
    from langchain_google_genai.chat_models import GoogleInvalidRequestError

    return GoogleInvalidRequestError(f"{code} INVALID_ARGUMENT")


def test_gemini_429_stops_the_whole_session():
    from langchain_google_genai.chat_models import GoogleRateLimitError

    error = GoogleRateLimitError(f"429 RESOURCE_EXHAUSTED for key {FAKE_KEY}")
    with pytest.raises(pytest.exit.Exception) as stopped:
        run_with_gemini_policy(RaisingGraph(error), "hi", FAKE_KEY)
    assert str(stopped.value.msg).startswith("STOP: Gemini 429")
    assert FAKE_KEY not in str(stopped.value.msg)


def test_gemini_503_is_inconclusive():
    with pytest.raises(pytest.skip.Exception) as skipped:
        run_with_gemini_policy(RaisingGraph(gemini_server_error(503)), "hi", FAKE_KEY)
    assert str(skipped.value.msg).startswith(f"{INCONCLUSIVE}: Gemini 503")


@pytest.mark.parametrize(
    "make_error",
    [
        lambda: gemini_server_error(500),
        lambda: gemini_client_error(400),
        lambda: ValueError("a bug in our code"),
    ],
    ids=["gemini-500", "gemini-400", "our-bug"],
)
def test_other_errors_remain_failures(make_error):
    error = make_error()
    with pytest.raises(type(error)):
        run_with_gemini_policy(RaisingGraph(error), "hi", FAKE_KEY)


def tool_text_when(monkeypatch, tool, fail_url: str, handler) -> str:
    """Run a real tool offline; requests to `fail_url` go to `handler`.

    Geocoding otherwise succeeds, so a forecast or Overpass failure can be
    produced too. Uses the services' real error handling and wording.
    """

    # Failing "Overpass" means every Overpass server fails, the main one and
    # the fallback, since the tool only reports a failure when all of them do.
    failing = (
        set(attractions_service.OVERPASS_URLS) if fail_url == OVERPASS else {fail_url}
    )

    def route(request: httpx.Request) -> httpx.Response:
        url = str(request.url.copy_with(query=None))
        if url in failing:
            return handler(request)
        if url == GEO:
            return httpx.Response(200, json=KANDY_GEOCODING)
        raise AssertionError(f"unexpected request to {url}")

    def mock_client():
        return httpx.Client(transport=httpx.MockTransport(route))

    monkeypatch.setattr(weather_service, "make_client", mock_client)
    monkeypatch.setattr(attractions_service, "make_client", mock_client)
    return tool.invoke({"destination": "Kandy"})


KANDY_GEOCODING = {
    "results": [
        {"name": "Kandy", "latitude": 7.2906, "longitude": 80.6337, "country": "LK"}
    ]
}


def status(code: int, **kwargs):
    return lambda request: httpx.Response(code, **kwargs)


def raises(error: Exception):
    def handler(request):
        raise error

    return handler


GEO = weather_service.GEOCODING_URL
FORECAST = weather_service.FORECAST_URL
OVERPASS = attractions_service.OVERPASS_URL

# (tool, failing URL, handler) for every temporary infrastructure problem.
INFRASTRUCTURE_FAILURES = {
    "weather-forecast-503": ("weather", FORECAST, status(503)),
    "weather-forecast-500": ("weather", FORECAST, status(500)),
    "weather-forecast-429": ("weather", FORECAST, status(429)),
    "weather-geocoding-429": ("weather", GEO, status(429)),
    "attractions-geocoding-429": ("attractions", GEO, status(429)),
    "weather-forecast-timeout": (
        "weather",
        FORECAST,
        raises(httpx.ReadTimeout("slow")),
    ),
    "weather-geocoding-connect": (
        "weather",
        GEO,
        raises(httpx.ConnectError("down")),
    ),
    "attractions-geocoding-timeout": (
        "attractions",
        GEO,
        raises(httpx.ConnectTimeout("slow")),
    ),
    "overpass-429": ("attractions", OVERPASS, status(429)),
    "overpass-504": ("attractions", OVERPASS, status(504)),
    "overpass-remark": (
        "attractions",
        OVERPASS,
        status(200, json={"elements": [], "remark": "runtime error: timeout"}),
    ),
    "overpass-timeout": (
        "attractions",
        OVERPASS,
        raises(httpx.ReadTimeout("slow")),
    ),
    "overpass-502": ("attractions", OVERPASS, status(502)),
    "overpass-503": ("attractions", OVERPASS, status(503)),
}
# ...and for problems that must stay genuine failures.
REAL_FAILURES = {
    "weather-forecast-not-json": ("weather", FORECAST, status(200, text="oops")),
    "weather-forecast-unreadable": (
        "weather",
        FORECAST,
        raises(httpx.DecodingError("corrupt body")),
    ),
    "weather-forecast-400": (
        "weather",
        FORECAST,
        status(400, json={"error": True, "reason": "bad"}),
    ),
    "overpass-not-json": ("attractions", OVERPASS, status(200, text="oops")),
    "overpass-400": ("attractions", OVERPASS, status(400)),
    "overpass-500": ("attractions", OVERPASS, status(500)),
    "overpass-connect-error": (
        "attractions",
        OVERPASS,
        raises(httpx.ConnectError("refused")),
    ),
}


def _tool(name: str):
    return {"weather": get_weather_forecast, "attractions": find_attractions}[name]


@pytest.mark.parametrize(
    "case", INFRASTRUCTURE_FAILURES.values(), ids=INFRASTRUCTURE_FAILURES.keys()
)
def test_infrastructure_failures_are_inconclusive(monkeypatch, case):
    tool, url, handler = case
    text = tool_text_when(monkeypatch, _tool(tool), url, handler)
    with pytest.raises(pytest.skip.Exception) as skipped:
        skip_if_infrastructure_failed(text)
    assert str(skipped.value.msg).startswith(INCONCLUSIVE)


@pytest.mark.parametrize("case", REAL_FAILURES.values(), ids=REAL_FAILURES.keys())
def test_real_failures_are_not_skipped(monkeypatch, case):
    tool, url, handler = case
    text = tool_text_when(monkeypatch, _tool(tool), url, handler)
    assert "unavailable" in text  # the tool did fail...
    try:
        skip_if_infrastructure_failed(text)
    except pytest.skip.Exception:
        # Re-raising would only mark this test "skipped" and hide the bug.
        pytest.fail(f"a genuine failure was classed as inconclusive: {text}")


def tool_calls_named(messages: list, name: str) -> list[dict]:
    return [
        call
        for m in messages
        if isinstance(m, AIMessage)
        for call in m.tool_calls
        if call["name"] == name
    ]


def tool_results_named(messages: list, name: str) -> list[str]:
    return [
        m.content for m in messages if isinstance(m, ToolMessage) and m.name == name
    ]


def attraction_names(messages: list) -> list[str]:
    """Every attraction name the find_attractions tool really returned.

    Skips the test as inconclusive if Overpass or the geocoder had a temporary
    problem, since then there are no names to check the answer against.
    """
    names = []
    for text in tool_results_named(messages, "find_attractions"):
        skip_if_infrastructure_failed(text)
        names += [match["name"] for match in ATTRACTION_LINE.finditer(text)]
    assert names, "find_attractions returned no attractions to check against"
    return names


def _plain(text: str) -> str:
    """Lowercase with straight apostrophes, so "Adam’s" matches "Adam's"."""
    return text.replace("’", "'").casefold()


def mentions_any(answer: str, names: list[str]) -> bool:
    return any(_plain(name) in _plain(answer) for name in names)


def gemini_requests(messages: list) -> int:
    return sum(isinstance(m, AIMessage) for m in messages)


@pytest.mark.live
def test_live_g1_agent_finds_nature_attractions_in_ella():
    """G1: 2-3 Gemini requests, 1 geocoding + 1-2 Overpass requests."""
    messages = run_live_agent(
        "What can I see in Ella? I like waterfalls and viewpoints."
    )

    calls = tool_calls_named(messages, "find_attractions")
    assert calls, f"agent did not call find_attractions: {describe_steps(messages)}"
    assert "ella" in calls[0]["args"]["destination"].lower()
    assert calls[0]["args"].get("category", "all") in {"nature", "all"}
    assert gemini_requests(messages) <= MAX_GEMINI_REQUESTS, describe_steps(messages)

    names = attraction_names(messages)
    answer = final_answer(messages)
    assert mentions_any(answer, names), f"answer names no tool result:\n{answer}"


@pytest.mark.live
def test_live_g2_agent_combines_attractions_and_weather_for_kandy():
    """G2: 2-3 Gemini requests, 2 geocoding + 1 Overpass + 1 forecast request."""
    messages = run_live_agent(
        "I'm visiting Kandy this weekend. What should I see, and what's the "
        "weather like?"
    )

    steps = describe_steps(messages)
    attraction_calls = tool_calls_named(messages, "find_attractions")
    weather_calls = tool_calls_named(messages, "get_weather_forecast")
    assert attraction_calls, f"agent did not call find_attractions: {steps}"
    assert weather_calls, f"agent did not call get_weather_forecast: {steps}"
    assert "kandy" in attraction_calls[0]["args"]["destination"].lower()
    assert "kandy" in weather_calls[0]["args"]["destination"].lower()
    assert gemini_requests(messages) <= MAX_GEMINI_REQUESTS, steps

    weather = tool_results_named(messages, "get_weather_forecast")
    skip_if_infrastructure_failed(weather[0])
    assert weather[0].startswith("Weather forecast for"), weather[0]

    names = attraction_names(messages)
    answer = final_answer(messages)
    assert mentions_any(answer, names), f"answer names no tool result:\n{answer}"
    assert re.search(r"\d\s*°\s*C|degrees", answer, re.I), answer


def test_cli_help_is_provider_neutral(capsys):
    from ai_travel_assistant import tool_calling

    with pytest.raises(SystemExit):
        tool_calling.main(["--help"])
    out = " ".join(capsys.readouterr().out.split())  # undo argparse wrapping
    assert "usually 2 LLM calls" in out
    assert "each message makes LLM calls" in out
    assert "saves one LLM call per message" in out
    assert "Gemini" not in out
