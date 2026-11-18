"""Stage 9, step 3: scorers, pure functions from (scenario, record) to scores.

What this file demonstrates:
  1. DETERMINISTIC scorers. Each one checks one property with code: the same
     record always gets the same score, and no scorer calls an LLM. The
     optional LLM judge (judge.py) is kept apart, because it is neither free
     nor deterministic.
  2. Pass, fail and SKIP. A scorer that does not apply to a scenario says
     "skip" (grounding means nothing for a routing question). Skips are left
     out of the rates; otherwise a metric would look better or worse just
     because of which scenarios happen to be in the dataset.
  3. Skipping must never HIDE a failure. When a plan was expected but there is
     no structured plan (the model wrote prose, or the run crashed), the plan
     metrics FAIL rather than skip. Otherwise an agent that never produced a
     structured plan would score 100% on day count.
  4. Ratio metrics. Extraction accuracy and grounding are rates over FIELDS and
     PLACES, not over scenarios (§7.3): a Score carries hits/total, and the
     scorecard divides the sums.

The rule for "this place came from a tool" is the Stage 8 compose rule
(itinerary_grounding.place_source), so the eval and the agent never disagree.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

from langchain_core.messages import AIMessage
from pydantic import ValidationError

from ai_travel_assistant.evals.records import RunRecord
from ai_travel_assistant.evals.scenarios import Scenario
from ai_travel_assistant.itinerary_grounding import (
    FAILED_TOOL_NOTES,
    SUBMIT_TOOL,
    collect_evidence,
    normalize,
    place_source,
    same_place,
    sources_for,
)
from ai_travel_assistant.schemas import Itinerary

Status = Literal["pass", "fail", "skip"]
PLAN_CATEGORIES = frozenset({"itinerary", "fault", "multi_turn"})
NO_PLAN = "no structured plan (prose answer or crash)"


@dataclass(frozen=True)
class Score:
    metric: str
    status: Status
    detail: str = ""
    # Ratio metrics only: how many fields/places/turns were right, of how many.
    hits: int = 0
    total: int = 0


def _pass(metric: str, detail: str = "", hits: int = 0, total: int = 0) -> Score:
    return Score(metric, "pass", detail, hits, total)


def _fail(metric: str, detail: str, hits: int = 0, total: int = 0) -> Score:
    return Score(metric, "fail", detail, hits, total)


def _skip(metric: str) -> Score:
    return Score(metric, "skip")


def _ratio(metric: str, hits: int, total: int, wrong: list[str]) -> Score:
    if hits == total:
        return _pass(metric, f"{hits}/{total}", hits, total)
    return _fail(metric, f"{hits}/{total}; wrong: {'; '.join(wrong)}", hits, total)


# --- Expected routes ------------------------------------------------------------


def expected_routes(scenario: Scenario) -> list[str | None]:
    """The route each turn should take (None: not checked).

    Routing and multi-turn scenarios say so; a plan scenario's single message
    must go to the agent, so it counts for the false-clarify rate too.
    """
    if scenario.expect.routes is not None:
        return list(scenario.expect.routes)
    if scenario.expect.route is not None:
        return [scenario.expect.route]
    if scenario.category in ("itinerary", "fault"):
        return ["agent"]
    return [None] * len(scenario.turns)


def _actual_routes(record: RunRecord) -> list[str | None]:
    return [turn.route for turn in record.turns]


# --- The scorers ----------------------------------------------------------------


def score_extraction(scenario: Scenario, record: RunRecord) -> Score:
    """Field-level accuracy of the understood trip (§7.3: >= 90%)."""
    metric = "extraction"
    if scenario.expect.trip is None:
        return _skip(metric)
    expected = scenario.expect.trip
    if not record.turns:
        return _fail(metric, "no turn ran", 0, len(expected))
    got = record.turns[-1].trip_request or {}
    wrong = [
        f"{name}: expected {want!r}, got {got.get(name)!r}"
        for name, want in expected.items()
        if not trip_field_matches(name, want, got.get(name))
    ]
    return _ratio(metric, len(expected) - len(wrong), len(expected), wrong)


def _empty(value: object) -> bool:
    return value is None or value == "" or value == []


def _loose(a: str, b: str) -> bool:
    """'tea' ~ 'tea plantations'; 'March' ~ 'in March'."""
    a, b = normalize(a), normalize(b)
    return bool(a and b) and (f" {a} " in f" {b} " or f" {b} " in f" {a} ")


def trip_field_matches(name: str, want: object, got: object) -> bool:
    if want is None:
        return _empty(got)  # not stated, so it must not be invented
    if _empty(got):
        return False
    if name == "destination":
        return same_place(want, got)
    if name == "travel_dates":
        return _loose(str(want), str(got))
    if name == "interests":
        have = [str(g) for g in got] if isinstance(got, list) else [str(got)]
        return all(any(_loose(str(w), h) for h in have) for w in want)
    return want == got


def score_route(scenario: Scenario, record: RunRecord) -> Score:
    """Every checked turn took the expected route (routing and multi-turn)."""
    metric = "route"
    if scenario.expect.route is None and scenario.expect.routes is None:
        return _skip(metric)
    want, got = expected_routes(scenario), _actual_routes(record)
    if got[: len(want)] == want:
        return _pass(metric, " -> ".join(map(str, got)))
    return _fail(metric, f"expected {want}, got {got}")


def _route_rate(metric: str, scenario: Scenario, record: RunRecord, route: str):
    """Hits/total over the turns expected to take `route` (skip if none)."""
    got = _actual_routes(record)
    turns = [i for i, want in enumerate(expected_routes(scenario)) if want == route]
    if not turns:
        return _skip(metric)
    wrong = []
    for i in turns:
        actual = got[i] if i < len(got) else "nothing (the turn did not run)"
        if actual != route:
            wrong.append(f"turn {i + 1} went to {actual}")
    return _ratio(metric, len(turns) - len(wrong), len(turns), wrong)


def score_clarify_when_needed(scenario: Scenario, record: RunRecord) -> Score:
    """Turns that should ask a question did ask one (§7.3: >= 90%)."""
    return _route_rate("clarify_when_needed", scenario, record, "clarify")


def score_no_false_clarify(scenario: Scenario, record: RunRecord) -> Score:
    """Turns that should go to the agent were not stopped by a question.

    The rate of this metric is 1 - the false-clarify rate (§7.3: <= 5%).
    A turn that went off-topic instead also counts as wrong here.
    """
    return _route_rate("no_false_clarify", scenario, record, "agent")


def score_required_tools(scenario: Scenario, record: RunRecord) -> Score:
    """Every required tool was called at least once (§7.3: >= 90%)."""
    metric = "required_tools"
    required = scenario.expect.required_tools
    if not required:
        return _skip(metric)
    called = set(record.tool_names())
    missing = [tool for tool in required if tool not in called]
    if missing:
        return _fail(metric, f"not called: {missing}")
    return _pass(metric, f"all {len(required)} called")


def score_crash_free(scenario: Scenario, record: RunRecord) -> Score:
    """The run finished every turn without an exception (§7.3: 100%)."""
    metric = "crash_free"
    if record.error is not None:
        e = record.error
        return _fail(metric, f"turn {e.turn + 1}: {e.type}: {e.message}")
    if len(record.turns) != len(scenario.turns):
        return _fail(metric, f"{len(record.turns)} of {len(scenario.turns)} turns ran")
    return _pass(metric)


def _last_turn_submitted(record: RunRecord) -> bool:
    if not record.turns:
        return False
    return any(
        isinstance(m, AIMessage) and any(c["name"] == SUBMIT_TOOL for c in m.tool_calls)
        for m in record.turn_messages(len(record.turns) - 1)
    )


def score_structured(scenario: Scenario, record: RunRecord) -> Score:
    """The last turn ended with submit_itinerary, not prose (tracked, no target).

    Checked on the last turn's messages, because state["itinerary"] keeps an
    earlier turn's plan: a prose reply to "make it 3 days" leaves it in place.
    """
    metric = "structured"
    if scenario.category not in PLAN_CATEGORIES:
        return _skip(metric)
    if _last_turn_submitted(record) and record.final_itinerary is not None:
        return _pass(metric)
    return _fail(metric, "the last turn answered in prose")


def _plan(record: RunRecord) -> tuple[Itinerary | None, str]:
    data = record.final_itinerary
    if data is None:
        return None, NO_PLAN
    try:
        return Itinerary.model_validate(data), ""
    except ValidationError as error:
        return None, f"invalid itinerary: {error.error_count()} errors"


def score_schema_valid(scenario: Scenario, record: RunRecord) -> Score:
    """The final plan validates as an Itinerary (§7.3: 100%)."""
    metric = "schema_valid"
    if scenario.category not in PLAN_CATEGORIES:
        return _skip(metric)
    plan, why = _plan(record)
    return _pass(metric) if plan else _fail(metric, why)


def score_days(scenario: Scenario, record: RunRecord) -> Score:
    """The plan has the requested number of days, numbered 1..n."""
    metric = "days"
    want = scenario.expect.days
    if want is None:
        return _skip(metric)
    plan, why = _plan(record)
    if plan is None:
        return _fail(metric, why)
    numbers = [day.day for day in plan.day_plans]
    if plan.days == want and numbers == list(range(1, want + 1)):
        return _pass(metric, f"{want} days")
    return _fail(metric, f"expected {want} days, got {plan.days} ({numbers})")


def score_destination(scenario: Scenario, record: RunRecord) -> Score:
    """The plan is for the requested place."""
    metric = "destination"
    want = scenario.expect.destination
    if want is None:
        return _skip(metric)
    plan, why = _plan(record)
    if plan is None:
        return _fail(metric, why)
    if same_place(want, plan.destination):
        return _pass(metric, plan.destination)
    return _fail(metric, f"expected {want}, got {plan.destination}")


def score_grounding(scenario: Scenario, record: RunRecord) -> Score:
    """Places named in the plan appear in this destination's tool results.

    Rate over places (§7.3: >= 95%). The whole thread is evidence, as in
    compose: a modify turn may reuse the earlier turn's results for the same
    place. No plan: skip here; schema_valid and structured already fail it.
    """
    metric = "grounding"
    if scenario.category not in PLAN_CATEGORIES:
        return _skip(metric)
    plan, _ = _plan(record)
    if plan is None:
        return _skip(metric)
    names = [a.name for day in plan.day_plans for a in day.activities]
    if not names:
        return _fail(metric, "the plan names no places")
    src = sources_for(plan.destination, collect_evidence(record.messages()))
    unfound = [name for name in names if place_source(name, src) is None]
    return _ratio(metric, len(names) - len(unfound), len(names), unfound)


def score_failures_reported(scenario: Scenario, record: RunRecord) -> Score:
    """Each broken tool really failed, and the plan says what is missing."""
    metric = "failures_reported"
    failing = scenario.failing_tools
    if not failing:
        return _skip(metric)
    evidence = collect_evidence(record.messages())
    problems = []
    for tool in sorted(failing):
        results = [e for e in evidence if e.tool == tool]
        if not results:
            problems.append(f"{tool} was not called")
        elif any(e.ok for e in results):
            problems.append(f"{tool} returned data although its fault was on")
    plan, _ = _plan(record)
    if plan is not None:
        listed = {normalize(item) for item in plan.not_available}
        for tool in sorted(failing):
            if normalize(FAILED_TOOL_NOTES[tool]) not in listed:
                problems.append(f"not_available does not list {tool}")
    else:
        answer = final_text(record).casefold()
        if "not available" not in answer and "unavailable" not in answer:
            problems.append("the prose answer does not say what is unavailable")
    if problems:
        return _fail(metric, "; ".join(problems))
    return _pass(metric, ", ".join(sorted(failing)))


def final_text(record: RunRecord) -> str:
    for message in reversed(record.messages()):
        if isinstance(message, AIMessage) and not message.tool_calls:
            return message.text
    return ""


def score_plan_replaced(scenario: Scenario, record: RunRecord) -> Score:
    """The last turn replaced the earlier plan (or kept it, if so expected)."""
    metric = "plan_replaced"
    want = scenario.expect.plan_replaced
    if want is None:
        return _skip(metric)
    if len(record.turns) < 2:
        return _fail(metric, "the follow-up turn did not run")
    before = next(
        (t.itinerary for t in reversed(record.turns[:-1]) if t.itinerary), None
    )
    after = record.turns[-1].itinerary
    if before is None:
        return _fail(metric, "no plan before the follow-up")
    replaced = after is not None and after != before
    if replaced == want:
        return _pass(metric, "replaced" if replaced else "kept")
    return _fail(metric, "the plan was kept" if want else "the plan changed")


Scorer = Callable[[Scenario, RunRecord], Score]

SCORERS: Mapping[str, Scorer] = {
    "extraction": score_extraction,
    "route": score_route,
    "clarify_when_needed": score_clarify_when_needed,
    "no_false_clarify": score_no_false_clarify,
    "required_tools": score_required_tools,
    "crash_free": score_crash_free,
    "structured": score_structured,
    "schema_valid": score_schema_valid,
    "days": score_days,
    "destination": score_destination,
    "grounding": score_grounding,
    "failures_reported": score_failures_reported,
    "plan_replaced": score_plan_replaced,
}


def score_record(scenario: Scenario, record: RunRecord) -> list[Score]:
    """Run every scorer. The record must belong to the scenario."""
    if record.scenario_id != scenario.id:
        raise ValueError(
            f"record {record.scenario_id!r} is not for scenario {scenario.id!r}"
        )
    return [scorer(scenario, record) for scorer in SCORERS.values()]
