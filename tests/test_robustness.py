"""Stage 7 checks: error classification, retries, history repair and the loop cap.

Offline only. Errors are built from the real google.genai error classes (no
API call) or simple stand-ins, so the classifier is tested against what Gemini
actually raises.
"""

from __future__ import annotations

import copy

import httpx
import pytest
from google.genai import errors as genai_errors
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolInvocationError

from ai_travel_assistant.robustness import (
    BUSY_MESSAGE,
    FALLBACK_REPLY,
    LLM_MAX_ATTEMPTS,
    LOOP_CAP_RESULT,
    MAX_TOOL_ROUNDS,
    NO_ANSWER_NOTE,
    NOT_RUN_RESULT,
    QUOTA_MESSAGE,
    RATE_LIMIT_MESSAGE,
    REJECTED_MESSAGE,
    close_pending_calls,
    describe_llm_error,
    error_status,
    is_quota_exhausted,
    is_rate_limited,
    is_transient_llm_error,
    llm_retry_policy,
    make_route_after_agent,
    repair_history,
    tool_error_message,
    tool_rounds_this_turn,
)

FAKE_KEY = "AIzaFAKE-key-for-tests-0123456789"


def server_error(code: int, status: str = "UNAVAILABLE") -> genai_errors.ServerError:
    return genai_errors.ServerError(
        code, {"error": {"code": code, "message": "Try later.", "status": status}}
    )


def client_error(code: int, message: str, status: str) -> genai_errors.ClientError:
    return genai_errors.ClientError(
        code, {"error": {"code": code, "message": message, "status": status}}
    )


DAILY_QUOTA = client_error(
    429,
    "Quota exceeded for metric: generate_content_free_tier_requests, limit: 20. "
    "quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier",
    "RESOURCE_EXHAUSTED",
)
PER_MINUTE = client_error(
    429,
    "Quota exceeded. quotaId: GenerateRequestsPerMinutePerProjectPerModel-FreeTier",
    "RESOURCE_EXHAUSTED",
)


class CodedError(Exception):
    """Like LangChain's Google errors: an exception with an int .code."""

    def __init__(self, code: int, text: str = "failed"):
        super().__init__(text)
        self.code = code


# --- Classification -------------------------------------------------------------


def test_error_status_reads_code_then_status_code_then_text():
    assert error_status(server_error(503)) == 503
    assert error_status(CodedError(504)) == 504
    response = httpx.Response(502, request=httpx.Request("GET", "https://x"))
    assert (
        error_status(
            httpx.HTTPStatusError("bad", request=response.request, response=response)
        )
        is None
    )
    assert error_status(RuntimeError("503 UNAVAILABLE")) == 503
    assert error_status(RuntimeError("no status here")) is None


@pytest.mark.parametrize(
    "exc",
    [
        server_error(500, "INTERNAL"),
        server_error(502, "BAD_GATEWAY"),
        server_error(503),
        server_error(504, "DEADLINE_EXCEEDED"),
        CodedError(503),
        httpx.ReadTimeout("slow"),
        httpx.ConnectError("refused"),
        TimeoutError("slow"),
        ConnectionError("reset"),
    ],
)
def test_transient_errors_are_retried(exc):
    assert is_transient_llm_error(exc)


@pytest.mark.parametrize(
    "exc",
    [
        DAILY_QUOTA,
        PER_MINUTE,
        client_error(400, "Bad schema", "INVALID_ARGUMENT"),
        client_error(401, "API key not valid", "UNAUTHENTICATED"),
        ValueError("parse error"),
        KeyError("x"),
    ],
)
def test_other_errors_are_never_retried(exc):
    assert not is_transient_llm_error(exc)


def test_429s_are_rate_limits_and_only_the_daily_one_is_quota():
    assert is_rate_limited(DAILY_QUOTA) and is_quota_exhausted(DAILY_QUOTA)
    assert is_rate_limited(PER_MINUTE) and not is_quota_exhausted(PER_MINUTE)
    assert not is_quota_exhausted(server_error(503))


def test_retry_policy_has_two_attempts_and_uses_the_classifier():
    policy = llm_retry_policy()
    assert policy.max_attempts == LLM_MAX_ATTEMPTS == 2
    assert policy.retry_on is is_transient_llm_error
    quick = llm_retry_policy(initial_interval=0)
    assert quick.initial_interval == 0 and quick.jitter is False


# --- Safe messages ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("exc", "message"),
    [
        (DAILY_QUOTA, QUOTA_MESSAGE),
        (PER_MINUTE, RATE_LIMIT_MESSAGE),
        (server_error(503), BUSY_MESSAGE),
        (server_error(504, "DEADLINE_EXCEEDED"), BUSY_MESSAGE),
        (httpx.ConnectError("refused"), BUSY_MESSAGE),
        (client_error(400, "Bad", "INVALID_ARGUMENT"), REJECTED_MESSAGE),
        (client_error(401, "API key not valid", "UNAUTHENTICATED"), REJECTED_MESSAGE),
    ],
)
def test_describe_llm_error(exc, message):
    assert describe_llm_error(exc) == message


def test_describe_llm_error_never_shows_the_error_text():
    leaky = RuntimeError(
        f"failed calling https://api?key={FAKE_KEY}\nTraceback (most recent)"
    )
    text = describe_llm_error(leaky)
    assert FAKE_KEY not in text
    assert "Traceback" not in text
    assert text == "Something went wrong (RuntimeError). Please try again."


# --- Tool errors ----------------------------------------------------------------------


def test_tool_error_message_hides_details_and_names_the_type():
    text = tool_error_message(KeyError(f"secret {FAKE_KEY}"))
    assert "KeyError" in text
    assert FAKE_KEY not in text
    assert "Do not call it again" in text


def test_tool_error_message_keeps_the_argument_error_for_the_llm():
    exc = ToolInvocationError("get_weather_forecast", ValueError("bad"), {"x": 1})
    assert tool_error_message(exc) == exc.message


# --- repair_history ------------------------------------------------------------------


def call(call_id: str, name: str = "get_weather_forecast") -> dict:
    return {"name": name, "args": {"destination": "Ella"}, "id": call_id}


def result(call_id: str, text: str = "Forecast.") -> ToolMessage:
    return ToolMessage(text, tool_call_id=call_id, name="get_weather_forecast")


def kinds(messages) -> list[str]:
    return [type(m).__name__ for m in messages]


def test_a_clean_history_is_unchanged():
    history = [
        HumanMessage("Weather?"),
        AIMessage("", tool_calls=[call("a"), call("b")]),
        result("a"),
        result("b"),
        AIMessage("Sunny."),
        HumanMessage("Thanks"),
    ]
    repaired = repair_history(history)
    assert repaired == history
    assert repaired is not history


def test_an_orphaned_tool_call_gets_a_placeholder_result():
    history = [
        HumanMessage("Weather?"),
        AIMessage("", tool_calls=[call("a")]),  # the tool crashed here
        HumanMessage("Hello again"),
    ]
    repaired = repair_history(history)
    assert kinds(repaired) == [
        "HumanMessage",
        "AIMessage",
        "ToolMessage",
        "HumanMessage",
    ]
    placeholder = repaired[2]
    assert placeholder.tool_call_id == "a"
    assert placeholder.content == NOT_RUN_RESULT
    assert placeholder.status == "error"


def test_a_partly_answered_parallel_step_keeps_its_real_results():
    history = [
        HumanMessage("Plan Ella"),
        AIMessage("", tool_calls=[call("a"), call("b", "find_attractions")]),
        result("a", "Real forecast."),
        HumanMessage("Hello again"),
    ]
    repaired = repair_history(history)
    assert [m.content for m in repaired if isinstance(m, ToolMessage)] == [
        "Real forecast.",
        NOT_RUN_RESULT,
    ]
    assert repaired[3].tool_call_id == "b"
    assert repaired[3].name == "find_attractions"


def test_an_orphaned_call_at_the_very_end_is_closed_too():
    repaired = repair_history(
        [HumanMessage("Hi"), AIMessage("", tool_calls=[call("a")])]
    )
    assert kinds(repaired) == ["HumanMessage", "AIMessage", "ToolMessage"]


def test_a_tool_result_without_its_call_is_dropped():
    history = [HumanMessage("Hi"), result("ghost"), AIMessage("Hello.")]
    assert kinds(repair_history(history)) == ["HumanMessage", "AIMessage"]


def test_two_traveller_messages_in_a_row_get_a_note():
    history = [HumanMessage("Plan Galle"), HumanMessage("Hello?")]
    repaired = repair_history(history)
    assert kinds(repaired) == ["HumanMessage", "AIMessage", "HumanMessage"]
    assert repaired[1].content == NO_ANSWER_NOTE


def test_repair_never_changes_the_input():
    history = [
        SystemMessage("sys"),
        HumanMessage("Weather?"),
        AIMessage("", tool_calls=[call("a")]),
        result("ghost"),
        HumanMessage("Again"),
        HumanMessage("Hello?"),
    ]
    before = copy.deepcopy(history)
    repair_history(history)
    assert history == before


# --- The loop cap ---------------------------------------------------------------------


def tool_round(i: int) -> list:
    return [AIMessage("", tool_calls=[call(f"c{i}")]), result(f"c{i}")]


def test_tool_rounds_count_only_the_current_turn():
    earlier = [HumanMessage("Old"), *tool_round(1), AIMessage("Old answer.")]
    now = [HumanMessage("New"), *tool_round(2), *tool_round(3)]
    assert tool_rounds_this_turn(earlier + now) == 2
    assert tool_rounds_this_turn([]) == 0


def test_route_after_agent_without_tool_calls_is_done():
    route = make_route_after_agent()
    assert route({"messages": [HumanMessage("Hi"), AIMessage("Hello.")]}) == "done"
    assert route({"messages": []}) == "done"


def test_route_after_agent_runs_tools_up_to_the_cap_then_falls_back():
    route = make_route_after_agent()
    messages: list = [HumanMessage("Plan")]
    for i in range(1, MAX_TOOL_ROUNDS + 1):
        messages.append(AIMessage("", tool_calls=[call(f"c{i}")]))
        assert route({"messages": messages}) == "tools", i
        messages.append(result(f"c{i}"))
    messages.append(AIMessage("", tool_calls=[call("one-too-many")]))
    assert route({"messages": messages}) == "fallback"


def test_route_after_agent_cap_is_configurable():
    route = make_route_after_agent(max_tool_rounds=1)
    messages = [
        HumanMessage("Plan"),
        *tool_round(1),
        AIMessage("", tool_calls=[call("x")]),
    ]
    assert route({"messages": messages}) == "fallback"


def test_close_pending_calls_answers_every_stopped_call():
    last = AIMessage("", tool_calls=[call("a"), call("b", "find_attractions")])
    results = close_pending_calls(last)
    assert [(m.tool_call_id, m.name, m.content, m.status) for m in results] == [
        ("a", "get_weather_forecast", LOOP_CAP_RESULT, "error"),
        ("b", "find_attractions", LOOP_CAP_RESULT, "error"),
    ]


def test_fallback_reply_is_plain_text():
    assert "Traceback" not in FALLBACK_REPLY and len(FALLBACK_REPLY) < 300


# --- Stage 8: the final-answer tool ------------------------------------------------


def submit_call(call_id: str = "s1") -> dict:
    return {"name": "submit_itinerary", "args": {}, "id": call_id}


def test_a_final_tool_call_goes_to_compose():
    route = make_route_after_agent(final_tool="submit_itinerary")
    messages = [HumanMessage("Plan"), AIMessage("", tool_calls=[submit_call()])]
    assert route({"messages": messages}) == "compose"


def test_a_final_tool_call_beside_other_calls_still_goes_to_compose():
    route = make_route_after_agent(final_tool="submit_itinerary")
    last = AIMessage("", tool_calls=[call("a"), submit_call()])
    assert route({"messages": [HumanMessage("Plan"), last]}) == "compose"


def test_the_final_tool_is_accepted_even_past_the_cap():
    route = make_route_after_agent(max_tool_rounds=1, final_tool="submit_itinerary")
    messages = [
        HumanMessage("Plan"),
        *tool_round(1),
        *tool_round(2),
        AIMessage("", tool_calls=[submit_call()]),
    ]
    assert route({"messages": messages}) == "compose"


def test_without_a_final_tool_the_router_is_unchanged():
    route = make_route_after_agent()
    messages = [HumanMessage("Plan"), AIMessage("", tool_calls=[submit_call()])]
    assert route({"messages": messages}) == "tools"


# --- Groq errors (LLM_PROVIDER=groq) ------------------------------------------------


def groq_status_error(code: int, message: str):
    """A real Groq SDK error, as langchain-groq raises it (has .status_code)."""
    import groq

    classes = {429: groq.RateLimitError, 400: groq.BadRequestError}
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(code, request=request)
    error_class = classes.get(code, groq.InternalServerError)
    return error_class(message, response=response, body=None)


def groq_timeout():
    import groq

    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return groq.APITimeoutError(request=request)


def groq_connection_error():
    import groq

    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return groq.APIConnectionError(request=request)


GROQ_DAILY = groq_status_error(
    429,
    "Rate limit reached for model `m` on requests per day (RPD): Limit 1000",
)
GROQ_PER_MINUTE = groq_status_error(
    429,
    "Rate limit reached for model `m` on requests per minute (RPM): Limit 30",
)


@pytest.mark.parametrize(
    ("exc", "message"),
    [
        (GROQ_DAILY, QUOTA_MESSAGE),
        (GROQ_PER_MINUTE, RATE_LIMIT_MESSAGE),
        (groq_status_error(503, "Service unavailable"), BUSY_MESSAGE),
        (groq_timeout(), BUSY_MESSAGE),
        (groq_connection_error(), BUSY_MESSAGE),
        (groq_status_error(400, "tool_use_failed"), REJECTED_MESSAGE),
    ],
)
def test_groq_errors_get_the_same_messages(exc, message):
    assert describe_llm_error(exc) == message


@pytest.mark.parametrize(
    "exc", [groq_status_error(503, "x"), groq_timeout(), groq_connection_error()]
)
def test_groq_transient_errors_are_retried(exc):
    assert is_transient_llm_error(exc)


def test_groq_429s_are_not_retried():
    assert not is_transient_llm_error(GROQ_PER_MINUTE)
    assert is_quota_exhausted(GROQ_DAILY)
    assert not is_quota_exhausted(GROQ_PER_MINUTE)


# --- Cloudflare Workers AI (LLM_PROVIDER=cloudflare) -------------------------------
# langchain-openai raises the openai SDK's errors. The account ID is part of the
# URL, so these fakes put a made-up one there and check it never leaks.

CF_ACCOUNT = "fake-account-id-0001"
CF_TOKEN = "fake-cf-token-0002"
CF_URL = (
    f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT}/ai/v1/chat/completions"
)


def cloudflare_status_error(code: int, message: str):
    """A real openai SDK error, as ChatOpenAI raises it for Workers AI."""
    import openai

    classes = {
        429: openai.RateLimitError,
        400: openai.BadRequestError,
        401: openai.AuthenticationError,
    }
    response = httpx.Response(code, request=httpx.Request("POST", CF_URL))
    error_class = classes.get(code, openai.InternalServerError)
    return error_class(message, response=response, body=None)


def cloudflare_timeout():
    import openai

    return openai.APITimeoutError(request=httpx.Request("POST", CF_URL))


def cloudflare_connection_error():
    import openai

    return openai.APIConnectionError(request=httpx.Request("POST", CF_URL))


@pytest.mark.parametrize(
    ("exc", "message"),
    [
        (cloudflare_status_error(429, "Too many requests"), RATE_LIMIT_MESSAGE),
        (cloudflare_status_error(503, "Service unavailable"), BUSY_MESSAGE),
        (cloudflare_status_error(500, "Internal error"), BUSY_MESSAGE),
        (cloudflare_timeout(), BUSY_MESSAGE),
        (cloudflare_connection_error(), BUSY_MESSAGE),
        (
            cloudflare_status_error(400, "model does not support tools"),
            REJECTED_MESSAGE,
        ),
        (cloudflare_status_error(401, "Authentication error"), REJECTED_MESSAGE),
    ],
)
def test_cloudflare_errors_get_the_same_messages(exc, message):
    assert describe_llm_error(exc) == message


@pytest.mark.parametrize(
    "exc",
    [
        cloudflare_status_error(503, "x"),
        cloudflare_timeout(),
        cloudflare_connection_error(),
    ],
)
def test_cloudflare_transient_errors_are_retried(exc):
    assert is_transient_llm_error(exc)


def test_cloudflare_429_is_not_retried():
    assert not is_transient_llm_error(cloudflare_status_error(429, "Too many requests"))


@pytest.mark.parametrize(
    "exc",
    [
        cloudflare_status_error(401, f"Invalid token {CF_TOKEN} for {CF_URL}"),
        cloudflare_status_error(503, f"Upstream failed for account {CF_ACCOUNT}"),
        cloudflare_timeout(),
    ],
)
def test_cloudflare_error_messages_never_show_the_token_or_account(exc):
    message = describe_llm_error(exc)
    assert CF_TOKEN not in message
    assert CF_ACCOUNT not in message
