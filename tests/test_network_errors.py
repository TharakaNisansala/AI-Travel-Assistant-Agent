"""Network errors from every HTTP library the providers use count as temporary.

google-genai 3.x (installed on Python 3.11+, so on the deployed 3.12) uses
httpx2, whose errors mirror httpx's but are separate classes. Before the fix
a Gemini timeout or refused connection there was "not a provider error": the
pool stopped instead of falling over to Cloudflare and Groq. Offline.
"""

from __future__ import annotations

import socket

import httpx
import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from ai_travel_assistant import tool_calling
from ai_travel_assistant.llm_basics import LLMSettings, build_llm
from ai_travel_assistant.provider_pool import (
    BUSY_COOLDOWN_S,
    AllProvidersUnavailable,
    ProviderHealth,
    ProviderPool,
    classify,
)
from ai_travel_assistant.robustness import (
    VISITOR_BUSY,
    describe_error_for_visitor,
    error_category,
    is_transient_llm_error,
    llm_retry_policy,
)
from ai_travel_assistant.streaming import TurnDone, stream_turn
from tests.test_provider_pool import (
    AnsweringChatModel,
    ApiError,
    FailingChatModel,
    FakeProvider,
    pool,
)

httpx2 = pytest.importorskip("httpx2")
httpcore = pytest.importorskip("httpcore")


def httpx2_errors():
    request = httpx2.Request("POST", "https://example.invalid/v1")
    return [
        httpx2.ConnectError("refused", request=request),
        httpx2.ConnectTimeout("connect timed out", request=request),
        httpx2.ReadTimeout("read timed out", request=request),
        httpx2.WriteTimeout("write timed out", request=request),
        httpx2.PoolTimeout("pool timed out", request=request),
        httpx2.ReadError("connection reset", request=request),
        httpx2.RemoteProtocolError("server hung up", request=request),
    ]


TRANSIENT = [
    *httpx2_errors(),
    httpcore.ConnectError("refused"),
    httpcore.ReadTimeout("read timed out"),
    httpx.ConnectError("refused"),  # unchanged: Gemini on Python 3.10
    httpx.ReadTimeout("read timed out"),
]


@pytest.mark.parametrize(
    "error", TRANSIENT, ids=lambda e: type(e).__module__ + "." + type(e).__name__
)
def test_network_errors_from_every_http_library_are_temporary(error):
    assert is_transient_llm_error(error)
    assert classify(error) == "busy"  # the pool falls over and pauses 30 s
    assert error_category(error) == "busy"
    assert describe_error_for_visitor(error) == VISITOR_BUSY
    assert llm_retry_policy(0).retry_on(error)  # the CLI's one retry, as for httpx


def test_httpcore2_errors_count_too_if_installed():
    httpcore2 = pytest.importorskip("httpcore2")
    assert is_transient_llm_error(httpcore2.ConnectError("refused"))
    assert is_transient_llm_error(httpcore2.ReadTimeout("read timed out"))


def _status_error():
    request = httpx2.Request("POST", "https://example.invalid/v1")
    response = httpx2.Response(401, request=request)
    return httpx2.HTTPStatusError("401", request=request, response=response)


@pytest.mark.parametrize(
    "error",
    [
        httpx2.DecodingError("bad gzip"),
        httpx2.TooManyRedirects("loop"),
        httpx2.InvalidURL("not a url"),
        _status_error(),
        ValueError("structured output did not parse"),
    ],
    ids=lambda e: type(e).__name__,
)
def test_other_httpx2_errors_are_not_temporary(error):
    assert not is_transient_llm_error(error)


def test_a_look_alike_class_outside_the_http_libraries_does_not_count():
    class TransportError(Exception):  # same name, wrong package
        pass

    class NetworkError(TransportError):
        pass

    assert not is_transient_llm_error(NetworkError("x"))
    assert classify(NetworkError("x")) is None


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        (ApiError(401), "rejected"),
        (ApiError(403), "rejected"),
        (ApiError(404), "rejected"),
        (ApiError(429, "GenerateRequestsPerDay"), "quota"),
        (ApiError(429, "per minute"), "rate_limit"),
        (ApiError(400), "bad_request"),
    ],
)
def test_auth_quota_and_bad_request_handling_is_unchanged(error, kind):
    assert classify(error) == kind


# --- Fallback through the pool ------------------------------------------------------


def test_a_gemini_httpx2_error_falls_over_to_cloudflare_then_groq():
    request = httpx2.Request("POST", "https://example.invalid/v1")
    gemini = FakeProvider(httpx2.ConnectError("refused", request=request))
    cloudflare = FakeProvider(httpx2.ReadTimeout("slow", request=request))
    groq = FakeProvider("from groq")
    p = pool(gemini, cloudflare, groq)
    assert p.invoke("hi") == "from groq"
    assert [x.calls for x in (gemini, cloudflare, groq)] == [1, 1, 1]
    paused = p.health.snapshot()
    assert paused["gemini"][0] == paused["cloudflare"][0] == "busy"
    assert paused["gemini"][1] == pytest.approx(BUSY_COOLDOWN_S)
    assert p.invoke("hi") == "from groq"  # paused providers are skipped
    assert [x.calls for x in (gemini, cloudflare, groq)] == [1, 1, 2]


def test_every_provider_with_network_errors_gives_the_busy_message():
    request = httpx2.Request("POST", "https://example.invalid/v1")
    p = pool(
        *(FakeProvider(httpx2.ConnectError("x", request=request)) for _ in range(3))
    )
    with pytest.raises(AllProvidersUnavailable) as info:
        p.invoke("hi")
    assert info.value.kind == "busy"
    assert describe_error_for_visitor(info.value) == VISITOR_BUSY


def test_a_turn_falls_over_inside_the_graph_with_one_call_each():
    """No duplicate retries: the pool error is final, the graph does not repeat it."""
    request = httpx2.Request("POST", "https://example.invalid/v1")
    gemini = FailingChatModel(
        messages=iter([]), error=httpx2.ConnectError("refused", request=request)
    )
    cloudflare = AnsweringChatModel(messages=iter([AIMessage("Ella is lovely.")]))
    graph = tool_calling.build_tool_agent_graph(
        pool(gemini, cloudflare),
        checkpointer=InMemorySaver(),
        llm_retry=llm_retry_policy(0),
    )
    events = list(stream_turn(graph, "t1", "Where to hike?"))
    assert isinstance(events[-1], TurnDone)
    assert events[-1].answer == "Ella is lovely."
    assert (gemini.calls, cloudflare.calls) == (1, 1)


@pytest.fixture
def no_network(monkeypatch):
    """Block every outgoing connection; loopback stays open (asyncio needs it)."""
    loopback = ("127.0.0.1", "::1", "localhost")
    real_connect, real_getaddrinfo = socket.socket.connect, socket.getaddrinfo

    def connect(sock, address):
        host = address[0] if isinstance(address, tuple) else address
        if host in loopback:
            return real_connect(sock, address)
        raise ConnectionRefusedError("network blocked by the test")

    def getaddrinfo(host, *args, **kwargs):  # never ask a real DNS server
        return real_getaddrinfo(
            "127.0.0.1" if host not in loopback else host, *args, **kwargs
        )

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)


def test_the_real_gemini_client_falls_over_when_the_network_fails(no_network):
    """The actual google-genai client: httpx on Python 3.10, httpx2 on 3.11+."""
    gemini = build_llm(LLMSettings(model="gemini-test", api_key="fake"), 0, 0)
    cloudflare = FakeProvider("from cloudflare")
    p = ProviderPool([("gemini", gemini), ("cloudflare", cloudflare)], ProviderHealth())
    assert p.invoke("hi") == "from cloudflare"
    assert p.health.snapshot()["gemini"][0] == "busy"
