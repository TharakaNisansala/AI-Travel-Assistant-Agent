"""Deployment: usage limits for the public web page (offline)."""

from __future__ import annotations

import threading
import time
from datetime import date, timedelta

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from ai_travel_assistant import llm_basics, tool_calling
from ai_travel_assistant.robustness import is_transient_llm_error
from ai_travel_assistant.usage_limits import (
    DAILY_LIMIT_MESSAGE,
    LIMIT_VARIABLES,
    DailyCallBudget,
    DailyLimitReached,
    ServerBusy,
    UsageGuard,
    UsageLimits,
)
from tests.test_provider_pool import (
    QUOTA_429,
    AnsweringChatModel,
    FailingChatModel,
    FakeProvider,
    pool,
)

# --- Settings -----------------------------------------------------------------


@pytest.fixture
def env(monkeypatch):
    for variable, _ in LIMIT_VARIABLES.values():
        monkeypatch.delenv(variable, raising=False)
    return monkeypatch


def test_the_approved_defaults(env):
    limits = UsageLimits.from_env()
    assert limits == UsageLimits(
        max_session_messages=15,
        max_daily_llm_calls=150,
        max_concurrent_requests=2,
        max_message_chars=1000,
        max_history_tokens=8000,
    )


def test_every_limit_comes_from_its_variable(env):
    env.setenv("TRAVEL_MAX_MESSAGES_PER_SESSION", "5")
    env.setenv("TRAVEL_MAX_DAILY_LLM_CALLS", " 40 ")
    env.setenv("TRAVEL_MAX_CONCURRENT_REQUESTS", "1")
    env.setenv("TRAVEL_MAX_MESSAGE_CHARS", "300")
    env.setenv("TRAVEL_MAX_HISTORY_TOKENS", "4000")
    assert UsageLimits.from_env() == UsageLimits(5, 40, 1, 300, 4000)


@pytest.mark.parametrize("value", ["0", "-3", "ten", "1.5"])
def test_a_bad_limit_is_a_config_error_naming_the_variable(env, value):
    env.setenv("TRAVEL_MAX_DAILY_LLM_CALLS", value)
    with pytest.raises(llm_basics.ConfigError, match="TRAVEL_MAX_DAILY_LLM_CALLS"):
        UsageLimits.from_env()


# --- The daily budget ----------------------------------------------------------


def test_the_budget_allows_exactly_its_limit():
    budget = DailyCallBudget(3)
    assert [budget.try_consume() for _ in range(5)] == [True] * 3 + [False] * 2
    assert (budget.used(), budget.remaining()) == (3, 0)


def test_the_budget_resets_on_a_new_utc_day():
    day = [date(2026, 11, 23)]
    budget = DailyCallBudget(1, today=lambda: day[0])
    assert budget.try_consume()
    assert not budget.try_consume()
    day[0] += timedelta(days=1)
    assert budget.remaining() == 1
    assert budget.try_consume()


def test_the_budget_is_exact_under_concurrent_use():
    budget = DailyCallBudget(30)
    granted = []

    def worker():
        for _ in range(5):
            granted.append(budget.try_consume())

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert granted.count(True) == 30
    assert budget.used() == 30


# --- Checks before a message is sent ----------------------------------------------


def guard(**limits) -> UsageGuard:
    return UsageGuard(UsageLimits(**limits), slot_wait_s=0.05)


def test_a_normal_message_is_allowed():
    assert guard().refusal("Plan 3 days in Ella", messages_sent=0) is None


def test_a_message_at_the_length_limit_is_allowed_and_one_over_is_not():
    g = guard(max_message_chars=10)
    assert g.refusal("x" * 10, 0) is None
    refusal = g.refusal("x" * 11, 0)
    assert "11 characters" in refusal and "limit is 10" in refusal


def test_the_session_limit_counts_messages_already_sent():
    g = guard(max_session_messages=2)
    assert g.refusal("hi", messages_sent=1) is None
    assert "limit of 2 messages" in g.refusal("hi", messages_sent=2)


def test_a_used_up_daily_budget_refuses_before_any_call():
    g = guard(max_daily_llm_calls=1)
    assert g.budget.try_consume()
    assert g.refusal("hi", 0) == DAILY_LIMIT_MESSAGE


def test_the_refusal_check_changes_no_counter():
    g = guard(max_daily_llm_calls=5)
    for _ in range(10):
        g.refusal("hi", 0)
    assert g.budget.used() == 0


# --- Simultaneous requests ---------------------------------------------------------


def test_a_third_request_waits_then_reports_busy():
    g = guard(max_concurrent_requests=2)
    with (
        g.request_slot(),
        g.request_slot(),
        pytest.raises(ServerBusy),
        g.request_slot(),
    ):
        pass
    with g.request_slot():  # all slots were given back
        pass


def test_a_slot_is_released_when_the_turn_fails():
    g = guard(max_concurrent_requests=1)
    with pytest.raises(RuntimeError), g.request_slot():
        raise RuntimeError("turn failed")
    with g.request_slot():
        pass


def test_no_more_than_the_limit_run_at_once():
    g = UsageGuard(UsageLimits(max_concurrent_requests=2), slot_wait_s=5)
    running, peak, lock = [0], [0], threading.Lock()

    def turn():
        with g.request_slot():
            with lock:
                running[0] += 1
                peak[0] = max(peak[0], running[0])
            time.sleep(0.02)
            with lock:
                running[0] -= 1

    threads = [threading.Thread(target=turn) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert peak[0] == 2


# --- The budget inside the provider pool ---------------------------------------------


def test_the_pool_counts_every_provider_call_including_fallbacks():
    budget = DailyCallBudget(10)
    gemini, cloudflare = FakeProvider(QUOTA_429), FakeProvider("ok")
    p = pool(gemini, cloudflare)
    p.budget = budget
    p.invoke("hi")  # gemini fails, cloudflare answers: two real calls
    p.invoke("hi")  # gemini is paused (no call), cloudflare answers
    assert budget.used() == 3


def test_a_used_up_budget_stops_the_pool_without_calling_a_provider():
    budget = DailyCallBudget(1)
    gemini = FakeProvider("ok")
    p = pool(gemini)
    p.budget = budget
    p.invoke("hi")
    with pytest.raises(DailyLimitReached):
        p.invoke("hi")
    assert gemini.calls == 1


def test_bound_and_structured_runnables_share_the_budget():
    budget = DailyCallBudget(2)
    p = pool(FakeProvider("ok"))
    p.budget = budget
    p.bind_tools([]).invoke("hi")
    p.with_structured_output(dict).invoke("hi")
    with pytest.raises(DailyLimitReached):
        p.invoke("hi")


def test_the_daily_limit_is_never_retried_by_the_graph():
    assert not is_transient_llm_error(DailyLimitReached())


def test_a_graph_turn_stops_at_the_budget_and_makes_no_extra_call():
    budget = DailyCallBudget(1)
    answering = AnsweringChatModel(messages=iter([]))
    failing = FailingChatModel(messages=iter([]))
    p = pool(failing, answering)
    p.budget = budget
    graph = tool_calling.build_tool_agent_graph(p, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t"}}
    with pytest.raises(DailyLimitReached):
        graph.invoke({"messages": [HumanMessage("hi")]}, config)
    assert (failing.calls, answering.calls) == (1, 0)


def test_build_chat_graph_passes_the_history_limit_and_budget(monkeypatch):
    seen = {}

    def capture(model, **kwargs):
        seen.update(kwargs, model=model)
        return "graph"

    monkeypatch.setattr(tool_calling, "build_tool_agent_graph", capture)
    from ai_travel_assistant.provider_pool import PoolSettings

    settings = PoolSettings(members=(llm_basics.LLMSettings("flash", "g"),))
    budget = DailyCallBudget(5)
    tool_calling.build_chat_graph(
        settings, None, None, max_history_tokens=1234, call_budget=budget
    )
    assert seen["max_history_tokens"] == 1234
    assert seen["model"].budget is budget


def test_the_cli_graph_keeps_the_default_history_limit(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        tool_calling, "build_tool_agent_graph", lambda model, **kw: seen.update(kw)
    )
    tool_calling.build_chat_graph(llm_basics.LLMSettings("flash", "g"), None, None)
    assert seen["max_history_tokens"] == tool_calling.MAX_HISTORY_TOKENS
