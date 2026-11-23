"""Deployment: the provider pool (fallback, cooldowns, call counts). Offline."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable
from langgraph.checkpoint.memory import InMemorySaver

from ai_travel_assistant import llm_basics, provider_pool, tool_calling
from ai_travel_assistant.provider_pool import (
    AllProvidersUnavailable,
    PoolSettings,
    ProviderHealth,
    ProviderPool,
    classify,
    cooldown_seconds,
    fallback_order,
    load_pool_settings,
    retry_after_hint,
)
from ai_travel_assistant.robustness import (
    BUSY_MESSAGE,
    QUOTA_MESSAGE,
    RATE_LIMIT_MESSAGE,
    describe_llm_error,
    is_transient_llm_error,
    llm_retry_policy,
)
from ai_travel_assistant.streaming import Token, TurnDone, stream_turn

# --- Fakes --------------------------------------------------------------------


class ApiError(Exception):
    """Looks like a provider SDK error: a status code and maybe headers."""

    def __init__(self, status: int, text: str = "", headers: dict | None = None):
        super().__init__(f"Error code: {status} - {text}")
        self.status_code = status
        self.response = SimpleNamespace(headers=headers or {})


QUOTA_429 = ApiError(429, "Quota exceeded: GenerateRequestsPerDay limit")
MINUTE_429 = ApiError(429, "rate_limit_exceeded per minute", {"retry-after": "12"})
UNAVAILABLE_503 = ApiError(503, "UNAVAILABLE")


class FakeProvider(Runnable):
    """Answers or raises from a script, and counts its calls."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0
        self.bound = None
        self.schema = None

    def invoke(self, input, config=None, **kwargs):
        self.calls += 1
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def bind_tools(self, tools, **kwargs):
        self.bound = list(tools)
        return self

    def with_structured_output(self, schema, **kwargs):
        self.schema = schema
        return self


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def pool(*providers, clock=None):
    health = ProviderHealth(clock or Clock())
    names = ["gemini", "cloudflare", "groq"]
    return ProviderPool(list(zip(names, providers, strict=False)), health)


# --- Fallback -----------------------------------------------------------------


def test_the_first_provider_answers_and_the_others_are_not_called():
    gemini, cloudflare = FakeProvider("from gemini"), FakeProvider("from cloudflare")
    assert pool(gemini, cloudflare).invoke("hi") == "from gemini"
    assert (gemini.calls, cloudflare.calls) == (1, 0)


@pytest.mark.parametrize(
    "error",
    [QUOTA_429, MINUTE_429, UNAVAILABLE_503, TimeoutError("slow"), ApiError(401)],
)
def test_a_provider_error_falls_over_to_the_next_provider(error):
    gemini, cloudflare = FakeProvider(error), FakeProvider("from cloudflare")
    assert pool(gemini, cloudflare).invoke("hi") == "from cloudflare"
    assert (gemini.calls, cloudflare.calls) == (1, 1)


def test_a_used_up_quota_is_skipped_by_later_requests_without_a_call():
    gemini, cloudflare = FakeProvider(QUOTA_429), FakeProvider("ok")
    p = pool(gemini, cloudflare)
    for _ in range(5):
        assert p.invoke("hi") == "ok"
    assert gemini.calls == 1  # not hit again while paused
    assert cloudflare.calls == 5


def test_a_paused_provider_is_tried_again_after_its_cooldown():
    clock = Clock()
    gemini, cloudflare = FakeProvider(UNAVAILABLE_503, "back"), FakeProvider("ok")
    p = pool(gemini, cloudflare, clock=clock)
    assert p.invoke("hi") == "ok"
    clock.now += provider_pool.BUSY_COOLDOWN_S - 1
    assert p.invoke("hi") == "ok"
    assert gemini.calls == 1
    clock.now += 2
    assert p.invoke("hi") == "back"
    assert gemini.calls == 2


def test_every_provider_failing_tries_each_once_then_raises():
    providers = [FakeProvider(UNAVAILABLE_503) for _ in range(3)]
    with pytest.raises(AllProvidersUnavailable) as info:
        pool(*providers).invoke("hi")
    assert [p.calls for p in providers] == [1, 1, 1]
    assert info.value.outcomes == (
        ("gemini", "busy"),
        ("cloudflare", "busy"),
        ("groq", "busy"),
    )
    assert isinstance(info.value.__cause__, ApiError)


def test_when_all_are_paused_no_api_call_is_made():
    providers = [FakeProvider(QUOTA_429) for _ in range(3)]
    p = pool(*providers)
    with pytest.raises(AllProvidersUnavailable):
        p.invoke("hi")
    with pytest.raises(AllProvidersUnavailable) as info:
        p.invoke("hi")
    assert [x.calls for x in providers] == [1, 1, 1]
    assert info.value.kind == "quota"
    assert info.value.retry_after_s == pytest.approx(provider_pool.QUOTA_COOLDOWN_S)


def test_a_non_provider_error_is_raised_without_trying_the_next_provider():
    gemini, cloudflare = FakeProvider(ValueError("bad output")), FakeProvider("ok")
    with pytest.raises(ValueError):
        pool(gemini, cloudflare).invoke("hi")
    assert cloudflare.calls == 0


def test_a_bad_request_falls_over_but_does_not_pause_the_provider():
    gemini, cloudflare = FakeProvider(ApiError(400), "fine"), FakeProvider("ok")
    p = pool(gemini, cloudflare)
    assert p.invoke("hi") == "ok"
    assert p.invoke("hi") == "fine"  # gemini was not paused
    assert gemini.calls == 2


def test_bind_tools_and_structured_output_keep_the_order_and_the_health():
    gemini, cloudflare = FakeProvider(QUOTA_429), FakeProvider("ok")
    p = pool(gemini, cloudflare)
    bound = p.bind_tools(["tool"])
    structured = p.with_structured_output(dict)
    assert bound.providers == structured.providers == ["gemini", "cloudflare"]
    assert gemini.bound == ["tool"] and cloudflare.schema is dict
    assert bound.invoke("hi") == "ok"
    assert structured.invoke("hi") == "ok"
    assert gemini.calls == 1  # the structured call saw the shared pause


def test_concurrent_requests_share_the_health_table_safely():
    gemini, cloudflare = FakeProvider(QUOTA_429), FakeProvider("ok")
    p = pool(gemini, cloudflare)
    results, errors = [], []

    def worker():
        try:
            results.append(p.invoke("hi"))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert results == ["ok"] * 20
    assert gemini.calls <= 20  # never more than one call per request


# --- Classifying and cooldowns --------------------------------------------------


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        (QUOTA_429, "quota"),
        (ApiError(429, "daily free allocation of neurons used"), "quota"),
        (MINUTE_429, "rate_limit"),
        (ApiError(413, "rate_limit_exceeded: tokens per minute"), "rate_limit"),
        (UNAVAILABLE_503, "busy"),
        (ApiError(500), "busy"),
        (TimeoutError(), "busy"),
        (ApiError(401), "rejected"),
        (ApiError(404), "rejected"),
        (ApiError(400), "bad_request"),
        (ValueError("parse"), None),
    ],
)
def test_classify(error, kind):
    assert classify(error) == kind


def test_retry_after_comes_from_the_header_or_the_message():
    assert retry_after_hint(MINUTE_429) == 12.0
    assert retry_after_hint(Exception("Please retry in 34.5s.")) == 34.5
    assert retry_after_hint(Exception("{'retryDelay': '20s'}")) == 20.0
    assert retry_after_hint(Exception("nothing here")) is None


def test_cooldowns_are_bounded():
    assert cooldown_seconds("rate_limit", 12) == 12
    assert cooldown_seconds("rate_limit", None) == provider_pool.RATE_LIMIT_COOLDOWN_S
    assert cooldown_seconds("rate_limit", 0) == provider_pool.MIN_RATE_LIMIT_COOLDOWN_S
    assert (
        cooldown_seconds("rate_limit", 99_999)
        == provider_pool.MAX_RATE_LIMIT_COOLDOWN_S
    )
    # A daily quota's short retryDelay must not unpause it early.
    assert cooldown_seconds("quota", 20) == provider_pool.QUOTA_COOLDOWN_S
    assert cooldown_seconds("bad_request", None) == 0


@pytest.mark.parametrize(
    ("outcomes", "message"),
    [
        ([("gemini", "quota"), ("groq", "quota")], QUOTA_MESSAGE),
        ([("gemini", "quota"), ("groq", "rate_limit")], RATE_LIMIT_MESSAGE),
        ([("gemini", "quota"), ("groq", "busy")], BUSY_MESSAGE),
    ],
)
def test_the_traveller_sees_one_safe_message(outcomes, message):
    assert describe_llm_error(AllProvidersUnavailable(outcomes)) == message


def test_the_pool_error_is_never_retried_by_the_graph():
    assert not is_transient_llm_error(AllProvidersUnavailable([("gemini", "busy")]))


# --- Settings -----------------------------------------------------------------

ENV_NAMES = (
    "LLM_PROVIDER",
    "LLM_FALLBACK_ORDER",
    "GEMINI_API_KEY",
    "GEMINI_MODEL",
    "GROQ_API_KEY",
    "GROQ_MODEL",
    "CLOUDFLARE_API_TOKEN",
    "CLOUDFLARE_MODEL",
    "CLOUDFLARE_ACCOUNT_ID",
)


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setattr(llm_basics, "load_dotenv", lambda: None)
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_default_order_is_gemini_cloudflare_groq():
    assert fallback_order(None) == ("gemini", "cloudflare", "groq")
    assert fallback_order(" Groq, gemini ,groq,") == ("groq", "gemini")


def test_unknown_provider_in_the_order_is_a_config_error():
    with pytest.raises(llm_basics.ConfigError, match="openai"):
        fallback_order("gemini,openai")


def test_only_providers_with_credentials_join_the_pool(env):
    env.setenv("GEMINI_API_KEY", "g-secret")
    env.setenv("GEMINI_MODEL", "flash")
    env.setenv("GROQ_API_KEY", "groq-secret")  # no GROQ_MODEL: left out
    settings = load_pool_settings()
    assert settings.providers == ["gemini"]
    assert settings.skipped == ("cloudflare", "groq")
    assert "g-secret" not in repr(settings)


def test_the_order_comes_from_llm_fallback_order(env):
    env.setenv("LLM_FALLBACK_ORDER", "groq,gemini")
    for name, value in [
        ("GEMINI_API_KEY", "g"),
        ("GEMINI_MODEL", "flash"),
        ("GROQ_API_KEY", "q"),
        ("GROQ_MODEL", "oss"),
    ]:
        env.setenv(name, value)
    assert load_pool_settings().providers == ["groq", "gemini"]


def test_no_configured_provider_is_a_config_error(env):
    with pytest.raises(llm_basics.ConfigError, match="No AI provider"):
        load_pool_settings()


def test_the_pool_builds_one_client_per_provider_without_calling_apis(env):
    gemini = llm_basics.LLMSettings(model="flash", api_key="g")
    groq = llm_basics.LLMSettings(model="oss", api_key="q", provider="groq")
    p = provider_pool.build_provider_pool(
        PoolSettings(members=(gemini, groq)), temperature=0, health=ProviderHealth()
    )
    assert p.providers == ["gemini", "groq"]
    retries = [model.max_retries for _, model in p.members]
    assert retries == [0, provider_pool.POOL_CLIENT_RETRIES["groq"]]


# --- Inside the real agent graph -------------------------------------------------


class FailingChatModel(GenericFakeChatModel):
    """A chat model whose every call raises the given error."""

    error: Exception = UNAVAILABLE_503
    calls: int = 0

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls += 1
        raise self.error


class AnsweringChatModel(GenericFakeChatModel):
    calls: int = 0

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls += 1
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def agent(*models):
    p = pool(*models)
    return tool_calling.build_tool_agent_graph(
        p, checkpointer=InMemorySaver(), llm_retry=llm_retry_policy(0)
    )


def test_a_turn_falls_over_inside_the_graph_and_streams_the_answer():
    failing = FailingChatModel(messages=iter([]), error=QUOTA_429)
    answering = AnsweringChatModel(
        messages=iter([AIMessage("Ella is lovely."), AIMessage("Kandy too.")])
    )
    graph = agent(failing, answering)
    events = list(stream_turn(graph, "t1", "Where to hike?"))
    assert isinstance(events[-1], TurnDone)
    assert events[-1].answer == "Ella is lovely."
    # The answering provider's tokens still reach the page (LangGraph callbacks).
    assert "".join(e.text for e in events if isinstance(e, Token)) == "Ella is lovely."
    events = list(stream_turn(graph, "t1", "And culture?"))
    assert events[-1].answer == "Kandy too."
    assert (failing.calls, answering.calls) == (1, 2)


def test_a_turn_where_every_provider_fails_makes_one_call_each():
    """The graph's RetryPolicy must not repeat the pool: 2 providers = 2 calls."""
    first = FailingChatModel(messages=iter([]))
    second = FailingChatModel(messages=iter([]))
    graph = agent(first, second)
    config = {"configurable": {"thread_id": "t1"}}
    with pytest.raises(AllProvidersUnavailable):
        graph.invoke({"messages": [HumanMessage("hi")]}, config)
    assert (first.calls, second.calls) == (1, 1)


def test_build_chat_graph_accepts_pool_settings_without_calling_apis():
    gemini = llm_basics.LLMSettings(model="flash", api_key="g")
    cloudflare = llm_basics.LLMSettings(
        model="@cf/m", api_key="c", provider="cloudflare", account_id="acc"
    )
    graph = tool_calling.build_chat_graph(
        PoolSettings(members=(gemini, cloudflare)), InMemorySaver(), None
    )
    assert graph is not None
