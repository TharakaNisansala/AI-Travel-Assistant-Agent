"""Stage 6 live checks: real LLM understanding and the clarification flow.

Opt-in only (pytest -m live). Uses LLM_PROVIDER, so Gemini or Groq. Mind the
Gemini free tier (5 requests a minute, 20 a day): the deliverable test makes
about 5 calls, the accuracy set 6.
Run one at a time, e.g.:
    uv run pytest -m live tests/test_clarification_live.py -k deliverable
"""

from __future__ import annotations

import time

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from ai_travel_assistant.request_understanding import (
    CLARIFICATION_NAME,
    make_llm_request_understander,
    make_understand_request_node,
    route_after_understanding,
)
from ai_travel_assistant.structured_output import FREE_TIER_DELAY_SECONDS
from ai_travel_assistant.tool_calling import (
    ask_agent,
    build_tool_agent_graph,
    final_answer,
    pending_question,
)

# (message, expected route). "clarify" only for plans missing a PLAN_FIELDS
# detail (place, days, group, citizenship, start, hotels), so the one "agent"
# plan states all of them, like the eval's route-agent-full-plan.
ROUTING_SET = (
    ("Plan me a trip", "clarify"),
    ("Can you prepare an itinerary for Kandy?", "clarify"),  # days missing
    ("I want a 4 day holiday, plan it for me", "clarify"),  # place missing
    (
        "Plan 3 days in Ella, I love hiking. Just me; I'm a Sri Lankan citizen "
        "starting from Colombo. No hotel needed.",
        "agent",
    ),
    ("Where can I go surfing in Sri Lanka?", "agent"),  # question, no plan
    ("Write my Python homework for me", "off_topic"),
)
# The reply to "Plan me a trip" answers every question at once, so the next
# turn plans (as in the eval's multi-clarify-then-plan).
FULL_REPLY = (
    "Galle, 2 days, just me; I'm a Sri Lankan citizen starting from Colombo; "
    "no hotel needed."
)


def _llm(temperature: float):
    from ai_travel_assistant.llm_basics import build_llm, load_settings

    return build_llm(load_settings(), temperature=temperature)


@pytest.mark.live
def test_live_deliverable_plan_me_a_trip_asks_then_continues():
    graph = build_tool_agent_graph(
        _llm(1.0),
        checkpointer=InMemorySaver(),
        understander=make_llm_request_understander(_llm(0)),
    )

    first = ask_agent(graph, "live", "Plan me a trip")
    assert first[-1].name == CLARIFICATION_NAME
    question = pending_question(graph, "live")
    assert question and "Where" in question

    time.sleep(FREE_TIER_DELAY_SECONDS)
    second = ask_agent(graph, "live", FULL_REPLY)

    state = graph.get_state({"configurable": {"thread_id": "live"}}).values
    assert pending_question(graph, "live") is None
    assert state["trip_request"]["destination"].lower() == "galle"
    assert state["trip_request"]["days"] == 2
    answer = final_answer(second)
    assert "Galle" in answer
    print(f"\nQuestion: {question}\nPlan (start): {answer[:400]}")


@pytest.mark.live
def test_live_routing_set():
    node = make_understand_request_node(make_llm_request_understander(_llm(0)))
    from langchain_core.messages import HumanMessage

    wrong = []
    for i, (text, expected) in enumerate(ROUTING_SET):
        if i > 0:
            time.sleep(FREE_TIER_DELAY_SECONDS)
        update = node({"messages": [HumanMessage(text)]})
        got = route_after_understanding(update)
        if got != expected:
            wrong.append((text, expected, got, update.get("intent")))
    assert not wrong, f"misrouted: {wrong}"
