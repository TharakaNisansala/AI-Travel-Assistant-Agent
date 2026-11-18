"""Stage 9, step 5: the scorecard (offline, no API key).

Records come from the real runner with fakes (the step 4 helpers), so the
whole chain runner -> records -> scorers -> scorecard is exercised.
"""

from __future__ import annotations

import pytest

from ai_travel_assistant.evals import scorecard as sc
from ai_travel_assistant.evals.records import RunRecord, RunStore, TurnRecord
from ai_travel_assistant.evals.scorecard import (
    SCORECARD_FILE,
    TARGETS,
    MetricRow,
    build_scorecard,
    latest_run_id,
    render_markdown,
)
from ai_travel_assistant.evals.scorers import SCORERS, Score
from tests.test_eval_run import ITIN, ROUTE, agent, run_scenario, scenario
from tests.test_itinerary_graph import (
    gather,
    plan,
    scripted,
    submit,
    tools,  # noqa: F401 - pytest fixture
)

CLARIFY = scenario("route-c", "routing", ["Help me plan a holiday."], route="clarify")


def run(scn, *replies):
    return run_scenario(scn, agent(scripted(*replies)), "run-1")


def row(card, metric):
    return next(r for r in card.rows if r.metric == metric)


def test_every_scorer_has_a_target_entry():
    assert set(SCORERS) <= set(TARGETS)


@pytest.mark.usefixtures("tools")
def test_a_clean_run_meets_every_target():
    records = [run(ITIN, gather(), submit(plan())), run(ROUTE), run(CLARIFY)]
    card = build_scorecard("run-1", [ITIN, ROUTE, CLARIFY], records)

    assert card.complete and card.unmet() == []
    assert card.scenario_pass == (3, 3)
    assert row(card, "grounding").counted == "2/2 places"
    # The plan and the agent-routed question count; the clarify case does not.
    assert row(card, "no_false_clarify").counted == "2/2 turns"
    assert row(card, "clarify_when_needed").counted == "1/1 turns"
    # Metrics no scenario applies to have no value and are not judged.
    assert row(card, "extraction").value is None
    assert row(card, "extraction").met is None
    assert card.calls_per_plan == 2
    assert card.total_calls == 2


@pytest.mark.usefixtures("tools")
def test_a_prose_plan_misses_the_100_percent_targets():
    records = [run(ITIN, gather(), "Day 1: walk."), run(ROUTE)]
    card = build_scorecard("run-1", [ITIN, ROUTE], records)
    unmet = " ".join(card.unmet())
    assert "schema_valid: 0% < 100%" in unmet
    assert "days: 0% < 100%" in unmet
    assert "scenario pass rate: 50% < 85%" in unmet
    # "structured" is tracked: it fails the scenario but is not a target.
    assert row(card, "structured").met is None
    assert ("itin-ella", "structured") in {(i, s.metric) for i, s in card.failures}


def test_ratio_metrics_weigh_by_items_not_scenarios():
    scores = [
        Score("grounding", "pass", hits=10, total=10),
        Score("grounding", "fail", hits=0, total=2),
        Score("grounding", "skip"),
    ]
    r = sc._row("grounding", scores)
    assert r.value == pytest.approx(10 / 12)
    assert r.counted == "10/12 places"
    assert r.met is False  # 83% < 95%

    rate = sc._row(
        "days", [Score("days", "pass"), Score("days", "fail"), Score("days", "skip")]
    )
    assert rate.value == 0.5 and rate.counted == "1/2 scenarios"


def test_a_met_check_tolerates_float_rounding():
    assert MetricRow("x", 0.9, "", 0.90).met is True
    assert MetricRow("x", 0.8999, "", 0.90).met is False


@pytest.mark.usefixtures("tools")
def test_missing_and_unknown_records_are_reported_not_hidden():
    stray = RunRecord(
        run_id="run-1",
        scenario_id="old-name",
        category="routing",
        started_at=run(ROUTE).started_at,
        model="fake",
        turns=[TurnRecord(text="x", message_count=0, llm_calls=0, seconds=0)],
    )
    card = build_scorecard("run-1", [ITIN, ROUTE], [run(ROUTE), stray])
    assert card.missing == ["itin-ella"]
    assert card.unknown == ["old-name"]
    assert not card.complete
    text = render_markdown(card)
    assert "1 still missing" in text
    assert "Incomplete: 1 scenarios have no record yet" in text
    assert "`old-name`" in text


def test_latency_counts_only_turns_that_reached_the_agent():
    def rec(sid, category, *turns):
        return RunRecord(
            run_id="r",
            scenario_id=sid,
            category=category,
            started_at="2026-11-18T09:00:00Z",
            model="fake",
            turns=[
                TurnRecord(
                    text="x", route=route, message_count=0, llm_calls=1, seconds=s
                )
                for route, s in turns
            ],
        )

    records = [
        rec("route-a", "routing", ("agent", 0.5)),  # stopped early: left out
        rec("itin-ella", "itinerary", ("agent", 20.0)),
        rec("multi-x", "multi_turn", ("clarify", 0.1), ("agent", 40.0)),
    ]
    card = build_scorecard("r", [], records)  # no scenarios: all unknown
    assert card.median_latency_s is None

    scns = [
        ROUTE,
        ITIN,
        scenario("multi-x", "multi_turn", ["a", "b"], days=2, destination="Ella"),
    ]
    card = build_scorecard("r", scns, records)
    assert card.median_latency_s == 30.0  # median of 20 and 40
    assert "median latency: 30.0s" not in " ".join(card.unmet())  # 30 <= 30

    slow = [rec("itin-ella", "itinerary", ("agent", 45.0))]
    assert "median latency: 45.0s > 30s" in build_scorecard("r", [ITIN], slow).unmet()


@pytest.mark.usefixtures("tools")
def test_extra_scorers_such_as_the_judge_join_the_card():
    def judge(scenario, record):
        return Score("judge", "fail", "feasibility 2")

    records = [run(ITIN, gather(), submit(plan()))]
    card = build_scorecard("run-1", [ITIN], records, extra_scorers={"judge": judge})
    assert row(card, "judge").counted == "0/1 scenarios"
    assert row(card, "judge").met is None  # tracked, never a hard target
    assert card.scenario_pass == (0, 1)


# --- Saved runs and the CLI -----------------------------------------------------


@pytest.mark.usefixtures("tools")
def test_cli_scores_the_latest_run_and_saves_the_markdown(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(sc, "load_scenarios", lambda: [ITIN, ROUTE])
    old = RunStore("2026-11-17T0900", root=tmp_path)
    old.save(run(ROUTE).model_copy(update={"run_id": old.run_id}))
    new = RunStore("2026-11-18T0900", root=tmp_path)
    for record in (run(ITIN, gather(), submit(plan())), run(ROUTE)):
        new.save(record.model_copy(update={"run_id": new.run_id}))

    assert latest_run_id(tmp_path) == "2026-11-18T0900"
    assert sc.main([], root=tmp_path) == 0
    out = capsys.readouterr().out
    assert "# Scorecard: run 2026-11-18T0900" in out
    assert "All targets met." in out
    saved = (new.dir / SCORECARD_FILE).read_text(encoding="utf-8")
    assert saved.startswith("# Scorecard: run 2026-11-18T0900")
    # The scorecard file is not mistaken for a record.
    assert new.scenario_ids() == ["itin-ella", "route-a"]

    assert sc.main(["--run-id", old.run_id, "--no-save"], root=tmp_path) == 0
    assert "1 still missing" in capsys.readouterr().out
    assert not (old.dir / SCORECARD_FILE).exists()


def test_cli_without_runs_says_so(tmp_path, capsys):
    assert sc.main([], root=tmp_path) == 1
    assert "No saved runs" in capsys.readouterr().err
    assert latest_run_id(tmp_path / "missing") is None


def test_cli_reports_a_broken_record(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sc, "load_scenarios", lambda: [ROUTE])
    store = RunStore("run-1", root=tmp_path)
    store.dir.mkdir(parents=True)
    store.path("route-a").write_text("{}", encoding="utf-8")
    assert sc.main(["--run-id", "run-1"], root=tmp_path) == 1
    assert "Cannot score run run-1" in capsys.readouterr().err
