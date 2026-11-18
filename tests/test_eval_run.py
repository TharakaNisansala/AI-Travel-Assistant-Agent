"""Stage 9, step 4: the eval runner, end to end with fakes (offline, no API key).

The scripted fake model plays Gemini, FakeUnderstander plays the Stage 6
understander, and the tool functions return fixed texts (all reused from the
Stage 6 and Stage 8 tests). The runner's records then go through the real
scorers, so these tests cover runner -> record -> score in one piece.
"""

from __future__ import annotations

import os

import pytest
from google.genai.errors import ClientError

from ai_travel_assistant import faults, tool_calling
from ai_travel_assistant.evals import run as runner
from ai_travel_assistant.evals.records import RunStore
from ai_travel_assistant.evals.run import (
    Agent,
    CallPacer,
    InfrastructureStop,
    fault_injection,
    run_all,
    run_scenario,
)
from ai_travel_assistant.evals.scenarios import Scenario
from ai_travel_assistant.evals.scorers import score_record
from tests.test_clarification import FakeUnderstander, understood
from tests.test_itinerary_graph import (
    TODAY,
    gather,
    plan,
    scripted,
    submit,
    tools,  # noqa: F401 - pytest fixture
)

PLAN_TOOLS = [
    "get_destination_guide",
    "find_attractions",
    "get_weather_forecast",
    "estimate_travel_time",
]


def scenario(sid, category, turns, faults=None, **expect):
    data = {"id": sid, "category": category, "turns": turns, "expect": expect}
    if faults:
        data["faults"] = faults
    return Scenario.model_validate(data)


ITIN = scenario(
    "itin-ella",
    "itinerary",
    ["Plan 2 days in Ella."],
    days=2,
    destination="Ella",
    required_tools=PLAN_TOOLS,
)

ANSWERS = {
    "Plan 2 days in Ella.": understood("plan", destination="Ella", days=2),
    "Plan 2 days in Jaffna.": understood("plan", destination="Jaffna", days=2),
    "Help me plan a holiday.": understood("plan"),
    "Tell me a joke.": understood("off_topic"),
    "Plan me a trip.": understood("plan"),
    "Ella, 2 days.": understood("plan", destination="Ella", days=2),
    "Make day 2 easier.": understood("modify"),
}


def agent(model, **kwargs):
    kwargs.setdefault("understander", FakeUnderstander(ANSWERS))
    return Agent(model=model, model_name="fake", today=lambda: TODAY, **kwargs)


@pytest.fixture(autouse=True)
def no_shell_faults(monkeypatch):
    monkeypatch.delenv(faults.ENV_VAR, raising=False)


# --- Plans, end to end through the scorers --------------------------------------


@pytest.mark.usefixtures("tools")
def test_an_itinerary_run_is_recorded_and_scores_clean():
    model = scripted(gather(), submit(plan()))
    record = run_scenario(ITIN, agent(model), "run-1")

    [turn] = record.turns
    assert turn.route == "agent"
    assert turn.intent == "plan"
    assert turn.trip_request["destination"] == "Ella"
    assert turn.itinerary["days"] == 2
    assert turn.llm_calls == 2 == len(model.calls)
    assert turn.message_count == len(record.message_data)
    assert record.model == "fake" and record.error is None

    failed = [s for s in score_record(ITIN, record) if s.status == "fail"]
    assert failed == []


@pytest.mark.usefixtures("tools")
def test_a_prose_answer_is_recorded_as_such_and_fails_structured():
    record = run_scenario(ITIN, agent(scripted(gather(), "Day 1: walk.")), "run-1")
    assert record.final_itinerary is None
    scores = {s.metric: s.status for s in score_record(ITIN, record)}
    assert scores["structured"] == scores["days"] == "fail"
    assert scores["required_tools"] == "pass"


@pytest.mark.usefixtures("tools")
def test_a_follow_up_replaces_the_plan():
    scn = scenario(
        "multi-ella",
        "multi_turn",
        ["Plan 2 days in Ella.", "Make day 2 easier."],
        routes=["agent", "agent"],
        days=2,
        destination="Ella",
        plan_replaced=True,
    )
    model = scripted(gather(), submit(plan()), submit(plan(easier=True)))
    record = run_scenario(scn, agent(model), "run-1")
    assert [t.intent for t in record.turns] == ["plan", "modify"]
    scores = {s.metric: s for s in score_record(scn, record)}
    assert scores["plan_replaced"].status == "pass"
    assert scores["route"].status == "pass"
    # The second turn's messages start after the first turn's.
    assert record.turn_messages(1)[0].content == "Make day 2 easier."


@pytest.mark.usefixtures("tools")
def test_clarify_then_plan_resumes_the_paused_turn():
    scn = scenario(
        "multi-clarify",
        "multi_turn",
        ["Plan me a trip.", "Ella, 2 days."],
        routes=["clarify", "agent"],
        days=2,
        destination="Ella",
    )
    record = run_scenario(scn, agent(scripted(gather(), submit(plan()))), "run-1")
    first, second = record.turns
    assert first.route == "clarify" and first.question
    assert first.llm_calls == 0  # the question costs no agent call
    assert second.route == "agent"
    assert all(s.status != "fail" for s in score_record(scn, record))


# --- Cheap scenarios stop before the agent --------------------------------------


@pytest.mark.parametrize(
    ("text", "route"),
    [
        ("Plan 2 days in Jaffna.", "agent"),
        ("Help me plan a holiday.", "clarify"),
        ("Tell me a joke.", "off_topic"),
    ],
)
def test_routing_stops_before_the_agents_first_call(text, route):
    model = scripted()  # no replies: any agent call would fail the test
    scn = scenario("route-x", "routing", [text], route=route)
    record = run_scenario(scn, agent(model), "run-1")
    assert model.calls == []
    assert record.turns[0].route == route
    assert record.error is None
    assert all(s.status != "fail" for s in score_record(scn, record))


def test_understanding_records_the_extracted_trip():
    scn = scenario(
        "und-x",
        "understanding",
        ["Plan 2 days in Jaffna."],
        trip={"destination": "Jaffna", "days": 2, "budget": None},
    )
    model = scripted()
    record = run_scenario(scn, agent(model), "run-1")
    assert model.calls == []
    assert record.turns[0].trip_request["destination"] == "Jaffna"
    [extraction] = [s for s in score_record(scn, record) if s.metric == "extraction"]
    assert (extraction.status, extraction.hits) == ("pass", 3)


# --- Failures: the agent's crash is saved, Gemini's outage stops the run --------


def quota_error():
    return ClientError(429, {"error": {"message": "Quota exceeded per day"}})


@pytest.mark.usefixtures("tools")
def test_an_agent_crash_is_kept_in_the_record():
    model = scripted(fail_on={0: ValueError("a bug in a node")})
    record = run_scenario(ITIN, agent(model), "run-1")
    assert record.turns == []
    assert record.error.type == "ValueError"
    assert record.error.turn == 0
    scores = {s.metric: s.status for s in score_record(ITIN, record)}
    assert scores["crash_free"] == "fail"


@pytest.mark.usefixtures("tools")
def test_a_gemini_429_stops_the_scenario_without_a_record():
    model = scripted(fail_on={0: quota_error()})
    with pytest.raises(InfrastructureStop, match="itin-ella, turn 1"):
        run_scenario(ITIN, agent(model), "run-1")


# --- Faults and isolation -------------------------------------------------------


def test_fault_injection_sets_and_restores_the_variable(monkeypatch):
    with fault_injection("weather"):
        assert os.environ[faults.ENV_VAR] == "weather"
    assert faults.ENV_VAR not in os.environ

    monkeypatch.setenv(faults.ENV_VAR, "guide")
    with fault_injection(None):
        assert faults.ENV_VAR not in os.environ
    assert os.environ[faults.ENV_VAR] == "guide"


@pytest.mark.usefixtures("tools")
def test_a_fault_scenario_runs_with_its_fault_and_is_scored(monkeypatch):
    seen = []

    def weather(destination, **_):
        seen.append(os.environ.get(faults.ENV_VAR))
        return f"Weather unavailable for '{destination}': the service is down."

    monkeypatch.setattr(tool_calling.get_weather_forecast, "func", weather)
    scn = scenario(
        "fault-x",
        "fault",
        ["Plan 2 days in Ella."],
        faults="weather",
        days=2,
        destination="Ella",
        required_tools=PLAN_TOOLS,
    )
    record = run_scenario(scn, agent(scripted(gather(), submit(plan()))), "run-1")
    assert seen == ["weather"]
    assert faults.ENV_VAR not in os.environ  # restored afterwards
    scores = {s.metric: s for s in score_record(scn, record)}
    assert scores["failures_reported"].status == "pass", scores["failures_reported"]


@pytest.mark.usefixtures("tools")
def test_every_scenario_starts_with_empty_caches(monkeypatch):
    cleared = []
    monkeypatch.setattr(runner, "clear_all_caches", lambda: cleared.append(1))
    run_scenario(ITIN, agent(scripted(gather(), submit(plan()))), "run-1")
    assert cleared == [1]


# --- Pacing ---------------------------------------------------------------------


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def test_pacer_spaces_calls_and_counts_its_sleep():
    clock = FakeClock()
    pacer = CallPacer(min_interval=13, sleep=clock.sleep, clock=clock)
    pacer.on_chat_model_start({}, [])
    clock.now += 5  # the model answered after 5 s
    pacer.on_chat_model_start({}, [])
    assert pacer.calls == 2
    assert pacer.slept == 8
    clock.now += 20  # a slow call: no wait needed
    pacer.on_chat_model_start({}, [])
    assert pacer.slept == 8


@pytest.mark.usefixtures("tools")
def test_turn_time_leaves_out_the_pacing_sleeps():
    clock = FakeClock()
    pacer = CallPacer(min_interval=13, sleep=clock.sleep, clock=clock)
    record = run_scenario(
        ITIN,
        agent(scripted(gather(), submit(plan())), pacer=pacer),
        "run-1",
        clock=clock,
    )
    assert pacer.slept == 13  # the second call waited
    assert record.turns[0].seconds == 0  # the fake model itself takes no time


# --- A whole run: budget, resume, stop ------------------------------------------


ROUTE = scenario("route-a", "routing", ["Plan 2 days in Jaffna."], route="agent")


@pytest.mark.usefixtures("tools")
def test_run_all_respects_the_budget_and_resumes(tmp_path):
    store = RunStore("run-1", root=tmp_path)
    lines = []
    # Budget 1: the routing case fits (~1 call); the itinerary (~5) does not.
    summary = run_all(
        [ROUTE, ITIN], agent(scripted()), store, max_calls=1, report=lines.append
    )
    assert summary.saved == ["route-a"]
    assert summary.left == ["itin-ella"]
    assert "call budget" in summary.stopped
    assert len(lines) == 1 and lines[0].startswith("route-a: ok, 0 calls, ")

    # The next day: same run id, routing is skipped, the itinerary runs.
    again = run_all(
        [ROUTE, ITIN],
        agent(scripted(gather(), submit(plan()))),
        store,
        report=lines.append,
    )
    assert again.skipped == ["route-a"]
    assert again.saved == ["itin-ella"]
    assert again.calls == 2
    assert again.stopped is None
    assert store.scenario_ids() == ["itin-ella", "route-a"]


@pytest.mark.usefixtures("tools")
def test_run_all_stops_at_an_outage_and_saves_nothing_for_it(tmp_path):
    store = RunStore("run-1", root=tmp_path)
    summary = run_all(
        [ITIN, ROUTE],
        agent(scripted(fail_on={0: quota_error()})),
        store,
        report=lambda _: None,
    )
    assert summary.saved == []
    assert summary.left == ["itin-ella", "route-a"]
    assert "Gemini is unavailable" in summary.stopped
    assert store.scenario_ids() == []


# --- CLI (no Gemini) ------------------------------------------------------------


def test_dry_run_lists_scenarios_and_calls_nothing(capsys, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "RunStore", lambda run_id: RunStore(run_id, tmp_path))
    assert runner.main(["--dry-run", "--category", "routing"]) == 0
    out = capsys.readouterr().out
    assert "7 selected, 7 to run (~7 Gemini calls)" in out
    assert "route-clarify-vague (routing, ~1 calls)" in out
    assert list(tmp_path.iterdir()) == []  # nothing written


def test_a_fault_variable_in_the_shell_is_refused(monkeypatch, capsys):
    monkeypatch.setenv(faults.ENV_VAR, "weather")
    assert runner.main(["--dry-run"]) == 1
    assert "Remove it" in capsys.readouterr().err


def test_a_bad_selection_is_reported(capsys):
    assert runner.main(["--dry-run", "--only", "no-such-id"]) == 1
    assert "no scenario with ids" in capsys.readouterr().err
