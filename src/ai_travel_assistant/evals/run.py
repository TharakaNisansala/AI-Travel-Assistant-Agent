"""Stage 9, step 4: the eval runner (the harness), live with Gemini.

What this file demonstrates:
  1. An evaluation HARNESS. For each scenario it builds the same graph as
     `--chat` (understander, structured itineraries, the Stage 7 retry), sends
     the turns, and turns the graph state after every turn into a RunRecord.
     It judges nothing; scorers read the saved records later.
  2. Isolated scenarios. Each one gets a fresh in-memory thread, no long-term
     store (one scenario's preferences must not leak into the next) and empty
     service caches (a cached Ella forecast would hide a weather fault).
  3. Observing the run with CALLBACKS. A LangChain callback handler sees every
     chat-model call in the graph, the understander's included. It counts the
     calls and paces them for the free tier (5 a minute); the pacing sleeps
     are left out of the timings.
  4. Infrastructure versus agent failures. A Gemini 429 or 5xx says nothing
     about the agent, so it STOPS the run and the scenario is not saved (run
     it again later with the same --run-id). Any other exception is the
     agent's crash: it is saved, and crash_free scores it.
  5. Cheap scenarios stay cheap. Understanding and routing stop right before
     the agent's first call: the route and the extracted trip are known by
     then, so each costs one Gemini call instead of five.

Run from the project root (default budget: 18 calls, under the 20/day quota):
    uv run python -m ai_travel_assistant.evals.run --dry-run
    uv run python -m ai_travel_assistant.evals.run --category routing
    uv run python -m ai_travel_assistant.evals.run --run-id 2026-11-18T1530
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, RetryPolicy

from ai_travel_assistant import faults
from ai_travel_assistant.evals.records import (
    ErrorRecord,
    RunRecord,
    RunStore,
    TurnRecord,
    new_run_id,
    utc_now,
)
from ai_travel_assistant.evals.scenarios import (
    CATEGORIES,
    Scenario,
    ScenarioFileError,
    load_scenarios,
)
from ai_travel_assistant.request_understanding import (
    CLARIFICATION_NAME,
    OFF_TOPIC_NAME,
    RequestUnderstander,
)
from ai_travel_assistant.resilience import clear_all_caches
from ai_travel_assistant.robustness import is_rate_limited, is_transient_llm_error
from ai_travel_assistant.tool_calling import (
    LLM,
    TOOLS,
    AgentContext,
    build_tool_agent_graph,
)

# Rough Gemini calls per scenario, for --dry-run and the budget (see the YAML).
ESTIMATED_CALLS = {
    "understanding": 1,
    "routing": 1,
    "itinerary": 5,
    "fault": 5,
    "multi_turn": 10,
}
# Stop before the turn's agent call: the understander alone decides these.
STOP_BEFORE_AGENT = frozenset({"understanding", "routing"})
DEFAULT_MAX_CALLS = 18  # leaves 2 of the 20 daily calls for a retry or a check
FREE_TIER_INTERVAL_S = 13.0  # 5 requests a minute, with a margin


class InfrastructureStop(RuntimeError):
    """Gemini refused or failed (429, 5xx): stop the run, save nothing."""


# --- Counting and pacing calls --------------------------------------------------


class CallPacer(BaseCallbackHandler):
    """Counts chat-model calls and keeps them min_interval apart.

    on_chat_model_start runs just before each request is sent, so sleeping
    there spaces the requests themselves. Retries count too: they use quota.
    """

    def __init__(
        self,
        min_interval: float = 0.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.min_interval = min_interval
        self.sleep = sleep
        self.clock = clock
        self.calls = 0
        self.slept = 0.0
        self._last: float | None = None

    def on_chat_model_start(
        self, serialized: Any, messages: Any, **kwargs: Any
    ) -> None:
        if self._last is not None:
            wait = self.min_interval - (self.clock() - self._last)
            if wait > 0:
                self.sleep(wait)
                self.slept += wait
        self._last = self.clock()
        self.calls += 1


@contextmanager
def fault_injection(value: str | None) -> Iterator[None]:
    """Set FAULT_INJECT for one scenario, then restore what was there."""
    old = os.environ.get(faults.ENV_VAR)
    if value:
        os.environ[faults.ENV_VAR] = value
    else:
        os.environ.pop(faults.ENV_VAR, None)
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(faults.ENV_VAR, None)
        else:
            os.environ[faults.ENV_VAR] = old


# --- One scenario ---------------------------------------------------------------


@dataclass
class Agent:
    """What the runner needs to build the --chat graph for each scenario."""

    model: BaseChatModel
    understander: RequestUnderstander
    model_name: str
    tools: Sequence[BaseTool] = TOOLS
    llm_retry: RetryPolicy | None = None
    today: Callable[[], date] = date.today
    pacer: CallPacer = field(default_factory=CallPacer)


def turn_route(snapshot: Any, messages: Sequence[AnyMessage]) -> str:
    """The route a turn took, from what it left behind (not from intent).

    A clarification question or a pause means clarify; the fixed off-topic
    reply means off_topic; anything else reached the agent.
    """
    names = {m.name for m in messages if isinstance(m, AIMessage)}
    if CLARIFICATION_NAME in names or snapshot.interrupts:
        return "clarify"
    if OFF_TOPIC_NAME in names:
        return "off_topic"
    return "agent"


def run_scenario(
    scenario: Scenario,
    agent: Agent,
    run_id: str,
    clock: Callable[[], float] = time.monotonic,
) -> RunRecord:
    """Run every turn of one scenario and return its record (not saved).

    Raises InfrastructureStop on a Gemini 429/5xx; any other error ends the
    scenario and is kept in the record.
    """
    clear_all_caches()
    graph = build_tool_agent_graph(
        agent.model,
        tools=agent.tools,
        checkpointer=InMemorySaver(),
        understander=agent.understander,
        llm_retry=agent.llm_retry,
        structured_itinerary=True,
        today=agent.today,
    )
    config: RunnableConfig = {
        "configurable": {"thread_id": scenario.id},
        "callbacks": [agent.pacer],
    }
    stop_before = [LLM] if scenario.category in STOP_BEFORE_AGENT else None
    started_at = utc_now()
    turns: list[TurnRecord] = []
    error: ErrorRecord | None = None

    with fault_injection(scenario.faults):
        for index, text in enumerate(scenario.turns):
            before = graph.get_state(config)
            count_before = len(before.values.get("messages", []))
            graph_input = (
                Command(resume=text)
                if before.interrupts
                else {"messages": [HumanMessage(text)]}
            )
            calls, slept, t0 = agent.pacer.calls, agent.pacer.slept, clock()
            try:
                graph.invoke(
                    graph_input,
                    config,
                    context=AgentContext(),
                    interrupt_before=stop_before,
                )
            except Exception as exc:
                if is_rate_limited(exc) or is_transient_llm_error(exc):
                    raise InfrastructureStop(
                        f"{scenario.id}, turn {index + 1}: the model "
                        f"({agent.model_name}) is unavailable or rate-limited "
                        f"({type(exc).__name__}); the scenario was not saved"
                    ) from exc
                error = ErrorRecord.from_exception(exc, turn=index)
                break
            seconds = max(0.0, clock() - t0 - (agent.pacer.slept - slept))
            snapshot = graph.get_state(config)
            values = snapshot.values
            new = values.get("messages", [])[count_before:]
            turns.append(
                TurnRecord(
                    text=text,
                    route=turn_route(snapshot, new),
                    intent=values.get("intent"),
                    trip_request=values.get("trip_request") or None,
                    missing_fields=list(values.get("missing_fields") or []),
                    question=str(snapshot.interrupts[0].value)
                    if snapshot.interrupts
                    else None,
                    itinerary=values.get("itinerary") or None,
                    message_count=len(values.get("messages", [])),
                    llm_calls=agent.pacer.calls - calls,
                    seconds=round(seconds, 3),
                )
            )

    messages = graph.get_state(config).values.get("messages", [])
    return RunRecord(
        run_id=run_id,
        scenario_id=scenario.id,
        category=scenario.category,
        started_at=started_at,
        model=agent.model_name,
        faults=scenario.faults,
        turns=turns,
        message_data=RunRecord.dump_messages(messages),
        error=error,
    )


# --- A whole run ----------------------------------------------------------------


@dataclass
class RunSummary:
    saved: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)  # already saved earlier
    left: list[str] = field(default_factory=list)  # not started: budget or stop
    calls: int = 0
    stopped: str | None = None  # why the run stopped early, if it did


def estimate_calls(scenarios: Sequence[Scenario]) -> int:
    return sum(ESTIMATED_CALLS[s.category] for s in scenarios)


def run_all(
    scenarios: Sequence[Scenario],
    agent: Agent,
    store: RunStore,
    max_calls: int = DEFAULT_MAX_CALLS,
    report: Callable[[str], None] = print,
) -> RunSummary:
    """Run the scenarios not saved yet, within a call budget.

    A scenario starts only if its estimated cost fits in what is left of the
    budget, so a run never starts a 10-call conversation with 3 calls left.
    """
    summary = RunSummary()
    todo = [s for s in scenarios if not store.has(s.id)]
    summary.skipped = [s.id for s in scenarios if store.has(s.id)]
    start_calls = agent.pacer.calls
    for position, scenario in enumerate(todo):
        used = agent.pacer.calls - start_calls
        if used + ESTIMATED_CALLS[scenario.category] > max_calls:
            summary.left = [s.id for s in todo[position:]]
            summary.stopped = f"call budget: {used} of {max_calls} used"
            break
        try:
            record = run_scenario(scenario, agent, store.run_id)
        except InfrastructureStop as stop:
            summary.left = [s.id for s in todo[position:]]
            summary.stopped = str(stop)
            break
        store.save(record)
        summary.saved.append(scenario.id)
        outcome = f"crashed: {record.error.type}" if record.error else "ok"
        report(
            f"{scenario.id}: {outcome}, {record.llm_calls} calls, {record.seconds:.1f}s"
        )
    summary.calls = agent.pacer.calls - start_calls
    return summary


# --- CLI ------------------------------------------------------------------------


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Stage 9 eval scenarios against the LLM set by LLM_PROVIDER "
            "and save records."
        )
    )
    parser.add_argument(
        "--category",
        action="append",
        choices=CATEGORIES,
        help="only this category (repeatable)",
    )
    parser.add_argument(
        "--only", action="append", metavar="ID", help="only this scenario (repeatable)"
    )
    parser.add_argument(
        "--run-id", help="continue this run (skips scenarios already saved)"
    )
    parser.add_argument(
        "--max-calls",
        type=int,
        default=DEFAULT_MAX_CALLS,
        help=(
            f"LLM call budget for this invocation (default {DEFAULT_MAX_CALLS}, "
            "sized for Gemini's free tier of about 20 a day)"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show what would run and its estimated calls; call nothing",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if os.environ.get(faults.ENV_VAR):
        print(
            f"{faults.ENV_VAR} is set in this shell; it would break every scenario. "
            f"Remove it (Remove-Item Env:{faults.ENV_VAR}); fault scenarios set "
            "their own.",
            file=sys.stderr,
        )
        return 1
    try:
        scenarios = load_scenarios(categories=args.category, only=args.only)
    except ScenarioFileError as exc:
        print(f"Scenario error: {exc}", file=sys.stderr)
        return 1

    store = RunStore(args.run_id or new_run_id())
    todo = [s for s in scenarios if not store.has(s.id)]
    print(
        f"Run {store.run_id}: {len(scenarios)} selected, {len(todo)} to run "
        f"(~{estimate_calls(todo)} LLM calls), budget {args.max_calls}."
    )
    if args.dry_run:
        for s in todo:
            print(f"  {s.id} ({s.category}, ~{ESTIMATED_CALLS[s.category]} calls)")
        return 0

    # Imported here so the offline tests never load the Gemini setup code.
    from ai_travel_assistant.llm_basics import (
        ConfigError,
        agent_client_retries,
        build_llm,
        load_settings,
    )
    from ai_travel_assistant.request_understanding import (
        make_llm_request_understander,
    )
    from ai_travel_assistant.robustness import llm_retry_policy

    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1
    # The same models and retry layer as --chat (see tool_calling.main).
    retries = agent_client_retries(settings)
    agent = Agent(
        model=build_llm(settings, temperature=1.0, max_retries=retries),
        understander=make_llm_request_understander(
            build_llm(settings, temperature=0, max_retries=retries)
        ),
        model_name=settings.model,
        llm_retry=llm_retry_policy(),
        pacer=CallPacer(min_interval=FREE_TIER_INTERVAL_S),
    )
    summary = run_all(scenarios, agent, store, max_calls=args.max_calls)
    print(
        f"Saved {len(summary.saved)}, already saved {len(summary.skipped)}, "
        f"left {len(summary.left)}; {summary.calls} LLM calls."
    )
    if summary.stopped:
        print(f"Stopped: {summary.stopped}")
    if summary.left:
        print(f"Continue later with: --run-id {store.run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
