"""Deployment: public visitor ids and thread cleanup (offline)."""

from __future__ import annotations

import threading

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from ai_travel_assistant import llm_basics, public_sessions, tool_calling
from ai_travel_assistant.public_sessions import (
    ThreadJanitor,
    new_visitor_id,
    thread_idle_seconds,
)
from tests.test_provider_pool import AnsweringChatModel


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class Deleted(list):
    """A delete function that records what it was asked to delete."""

    def __call__(self, thread_id):
        self.append(thread_id)


def janitor(idle_s=60, **kwargs):
    clock = Clock()
    return ThreadJanitor(idle_s, sweep_every_s=0, clock=clock, **kwargs), clock


def test_visitor_ids_are_random_and_never_the_cli_default():
    ids = {new_visitor_id() for _ in range(100)}
    assert len(ids) == 100
    assert tool_calling.DEFAULT_USER not in ids


def test_idle_threads_are_deleted_and_active_ones_kept():
    j, clock = janitor(idle_s=60)
    j.touch("old")
    clock.now = 50
    j.touch("recent")
    clock.now = 70
    deleted = Deleted()
    assert j.sweep(deleted) == ["old"]
    assert deleted == ["old"]
    assert j.tracked() == 1


def test_a_thread_used_again_is_not_swept():
    j, clock = janitor(idle_s=60)
    j.touch("t")
    clock.now = 59
    j.touch("t")
    clock.now = 100
    assert j.sweep(Deleted()) == []


def test_sweeps_are_throttled_unless_forced():
    clock = Clock()
    j = ThreadJanitor(10, sweep_every_s=300, clock=clock)
    j.touch("t")
    clock.now = 100
    assert j.sweep(Deleted()) == []  # too soon after the last sweep
    assert j.sweep(Deleted(), force=True) == ["t"]


def test_the_oldest_threads_above_the_cap_are_deleted():
    j, clock = janitor(idle_s=10_000, max_threads=2)
    for i, name in enumerate(["a", "b", "c", "d"]):
        clock.now = i
        j.touch(name)
    assert sorted(j.sweep(Deleted())) == ["a", "b"]
    assert j.tracked() == 2


def test_only_tracked_threads_are_ever_deleted():
    j, _ = janitor()
    deleted = Deleted()
    assert not j.release("a-cli-thread", deleted)
    assert deleted == []
    j.touch("public")
    assert j.release("public", deleted)
    assert deleted == ["public"]
    assert not j.release("public", deleted)  # only once


def test_a_failing_delete_never_raises():
    j, clock = janitor(idle_s=1)
    j.touch("t")
    clock.now = 5

    def broken(_):
        raise RuntimeError("disk full")

    assert j.sweep(broken) == ["t"]
    assert j.tracked() == 0


def test_touching_from_many_threads_is_safe():
    j, _ = janitor()
    workers = [
        threading.Thread(target=lambda n=n: [j.touch(f"t{n}-{i}") for i in range(50)])
        for n in range(10)
    ]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    assert j.tracked() == 500


def test_cleanup_deletes_one_conversation_and_keeps_another():
    """With a real checkpointer: deleting a thread removes only its messages."""
    model = AnsweringChatModel(messages=iter([AIMessage("A."), AIMessage("B.")]))
    checkpointer = InMemorySaver()
    graph = tool_calling.build_tool_agent_graph(model, checkpointer=checkpointer)
    for thread in ("alice", "bob"):
        config = {"configurable": {"thread_id": thread}}
        graph.invoke({"messages": [HumanMessage("hi")]}, config)
    j, _ = janitor()
    j.touch("alice")
    j.touch("bob")
    j.release("alice", checkpointer.delete_thread)

    def messages(thread):
        config = {"configurable": {"thread_id": thread}}
        return graph.get_state(config).values.get("messages", [])

    assert messages("alice") == []
    assert [m.content for m in messages("bob")] == ["hi", "B."]


@pytest.mark.parametrize(("raw", "seconds"), [("", 7200), ("30", 1800)])
def test_idle_time_comes_from_the_environment(monkeypatch, raw, seconds):
    monkeypatch.setenv(public_sessions.THREAD_IDLE_ENV, raw)
    assert thread_idle_seconds() == seconds


@pytest.mark.parametrize("raw", ["0", "-1", "soon"])
def test_a_bad_idle_time_is_a_config_error(monkeypatch, raw):
    monkeypatch.setenv(public_sessions.THREAD_IDLE_ENV, raw)
    with pytest.raises(llm_basics.ConfigError, match="TRAVEL_THREAD_IDLE_MINUTES"):
        thread_idle_seconds()


def test_the_shared_janitor_is_one_per_process(monkeypatch):
    monkeypatch.delenv(public_sessions.THREAD_IDLE_ENV, raising=False)
    public_sessions.reset_shared_janitor()
    try:
        assert public_sessions.shared_janitor() is public_sessions.shared_janitor()
    finally:
        public_sessions.reset_shared_janitor()
