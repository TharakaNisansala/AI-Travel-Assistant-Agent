"""Stage 9, step 5: the scorecard, scores of a saved run against the §7.3 targets.

What this file demonstrates:
  1. AGGREGATION. Ratio metrics (extraction, grounding, the clarify rates)
     divide summed hits by summed totals, so a scenario with 5 places weighs
     more than one with 2. Pass/fail metrics are a pass rate over the
     scenarios they apply to; skips are left out of both.
  2. Targets next to results. Each metric shows its §7.3 target and whether
     it was met; metrics without a target are tracked, not judged.
  3. Scoring is free and repeatable. The scorecard reads saved records, so it
     can be rebuilt any time, e.g. after fixing a scorer, at no Gemini cost.
     Only --judge spends calls (one per plan).
  4. An honest denominator. Scenarios with no record yet are listed as
     missing; the scorecard never pretends a partial run is complete.

Run from the project root:
    uv run python -m ai_travel_assistant.evals.scorecard              # latest run
    uv run python -m ai_travel_assistant.evals.scorecard --run-id 2026-11-18T1530
    uv run python -m ai_travel_assistant.evals.scorecard --judge      # + LLM judge
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ai_travel_assistant.evals.records import (
    DEFAULT_RUNS_DIR,
    RunRecord,
    RunRecordError,
    RunStore,
)
from ai_travel_assistant.evals.scenarios import (
    Scenario,
    ScenarioFileError,
    load_scenarios,
)
from ai_travel_assistant.evals.scorers import SCORERS, Score, Scorer, score_record

SCORECARD_FILE = "scorecard.md"


@dataclass(frozen=True)
class Target:
    kind: str  # "ratio": summed hits / summed totals; "rate": share of passes
    minimum: float | None  # None: tracked, not judged


# §7.3 (and §7.2 where §7.3 has no line). "no_false_clarify" is
# 1 - false-clarify rate, so "false-clarify <= 5%" becomes ">= 95%".
TARGETS: dict[str, Target] = {
    "extraction": Target("ratio", 0.90),
    "route": Target("rate", None),
    "clarify_when_needed": Target("ratio", 0.90),
    "no_false_clarify": Target("ratio", 0.95),
    "required_tools": Target("rate", 0.90),
    "crash_free": Target("rate", 1.00),
    "structured": Target("rate", None),
    "schema_valid": Target("rate", 1.00),
    "days": Target("rate", 1.00),
    "destination": Target("rate", None),
    "grounding": Target("ratio", 0.95),
    "failures_reported": Target("rate", 1.00),
    "plan_replaced": Target("rate", None),
    "judge": Target("rate", None),
}
SCENARIO_PASS_TARGET = 0.85
MEDIAN_LATENCY_TARGET_S = 30.0
PLAN_CATEGORIES_FOR_CALLS = frozenset({"itinerary", "fault"})  # one-message plans


@dataclass(frozen=True)
class MetricRow:
    metric: str
    value: float | None  # None: nothing to measure (all skipped)
    counted: str  # e.g. "27/30 scenarios" or "41/43 places"
    minimum: float | None

    @property
    def met(self) -> bool | None:
        if self.minimum is None or self.value is None:
            return None
        return self.value >= self.minimum - 1e-9


@dataclass
class Scorecard:
    run_id: str
    model: str
    rows: list[MetricRow]
    scenario_pass: tuple[int, int]  # (passed, scored)
    median_latency_s: float | None
    calls_per_plan: float | None
    total_calls: int
    failures: list[tuple[str, Score]] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)  # no record yet
    unknown: list[str] = field(default_factory=list)  # record, no scenario

    @property
    def complete(self) -> bool:
        return not self.missing

    @property
    def scenario_pass_rate(self) -> float | None:
        passed, scored = self.scenario_pass
        return passed / scored if scored else None

    def unmet(self) -> list[str]:
        """Every target that was missed, in words (empty: all met)."""
        problems = [
            f"{row.metric}: {_pct(row.value)} < {_pct(row.minimum)}"
            for row in self.rows
            if row.met is False
        ]
        rate = self.scenario_pass_rate
        if rate is not None and rate < SCENARIO_PASS_TARGET:
            problems.append(
                f"scenario pass rate: {_pct(rate)} < {_pct(SCENARIO_PASS_TARGET)}"
            )
        latency = self.median_latency_s
        if latency is not None and latency > MEDIAN_LATENCY_TARGET_S:
            problems.append(
                f"median latency: {latency:.1f}s > {MEDIAN_LATENCY_TARGET_S:.0f}s"
            )
        return problems


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def _row(metric: str, scores: Sequence[Score]) -> MetricRow:
    target = TARGETS.get(metric, Target("rate", None))
    applied = [s for s in scores if s.status != "skip"]
    if target.kind == "ratio":
        hits = sum(s.hits for s in applied)
        total = sum(s.total for s in applied)
        unit = {
            "extraction": "fields",
            "grounding": "places",
        }.get(metric, "turns")
        value = hits / total if total else None
        return MetricRow(metric, value, f"{hits}/{total} {unit}", target.minimum)
    passed = sum(s.status == "pass" for s in applied)
    value = passed / len(applied) if applied else None
    return MetricRow(
        metric, value, f"{passed}/{len(applied)} scenarios", target.minimum
    )


def build_scorecard(
    run_id: str,
    scenarios: Sequence[Scenario],
    records: Iterable[RunRecord],
    extra_scorers: dict[str, Scorer] | None = None,
) -> Scorecard:
    """Score every record against its scenario and aggregate the results."""
    by_id = {s.id: s for s in scenarios}
    records = list(records)
    unknown = sorted(r.scenario_id for r in records if r.scenario_id not in by_id)
    records = [r for r in records if r.scenario_id in by_id]
    have = {r.scenario_id for r in records}
    missing = [s.id for s in scenarios if s.id not in have]

    per_metric: dict[str, list[Score]] = {name: [] for name in SCORERS}
    for name in extra_scorers or {}:
        per_metric[name] = []
    failures: list[tuple[str, Score]] = []
    passed = 0
    for record in records:
        scenario = by_id[record.scenario_id]
        scores = score_record(scenario, record)
        for scorer in (extra_scorers or {}).values():
            scores.append(scorer(scenario, record))
        for score in scores:
            per_metric.setdefault(score.metric, []).append(score)
        bad = [s for s in scores if s.status == "fail"]
        failures.extend((record.scenario_id, s) for s in bad)
        passed += not bad

    # Latency: turns that reached the agent. Understanding and routing stop
    # before the agent on purpose, so their quick turns would flatter it.
    agent_turns = [
        t.seconds
        for r in records
        if r.category not in ("understanding", "routing")
        for t in r.turns
        if t.route == "agent"
    ]
    plan_calls = [
        r.llm_calls for r in records if r.category in PLAN_CATEGORIES_FOR_CALLS
    ]
    models = sorted({r.model for r in records})
    return Scorecard(
        run_id=run_id,
        model=", ".join(models) or "n/a",
        rows=[_row(metric, scores) for metric, scores in per_metric.items()],
        scenario_pass=(passed, len(records)),
        median_latency_s=statistics.median(agent_turns) if agent_turns else None,
        calls_per_plan=statistics.mean(plan_calls) if plan_calls else None,
        total_calls=sum(r.llm_calls for r in records),
        failures=failures,
        missing=missing,
        unknown=unknown,
    )


def render_markdown(card: Scorecard) -> str:
    passed, scored = card.scenario_pass
    lines = [
        f"# Scorecard: run {card.run_id}",
        "",
        f"**Model:** {card.model}  ",
        f"**Scenarios scored:** {scored}"
        + (f" ({len(card.missing)} still missing)" if card.missing else " (complete)")
        + "  ",
        f"**LLM calls in the run:** {card.total_calls}",
        "",
        "| Metric | Result | Counted | Target | Met |",
        "|---|---|---|---|---|",
    ]

    def mark(met: bool | None) -> str:
        return {True: "yes", False: "**NO**", None: "tracked"}[met]

    rate = card.scenario_pass_rate
    lines.append(
        f"| scenario pass rate | {_pct(rate)} | {passed}/{scored} scenarios | "
        f">= {_pct(SCENARIO_PASS_TARGET)} | "
        f"{mark(None if rate is None else rate >= SCENARIO_PASS_TARGET)} |"
    )
    for row in card.rows:
        target = "-" if row.minimum is None else f">= {_pct(row.minimum)}"
        lines.append(
            f"| {row.metric} | {_pct(row.value)} | {row.counted} | {target} | "
            f"{mark(row.met)} |"
        )
    latency = card.median_latency_s
    lines.append(
        "| median latency (agent turns) | "
        + ("n/a" if latency is None else f"{latency:.1f}s")
        + f" | | <= {MEDIAN_LATENCY_TARGET_S:.0f}s | "
        + mark(None if latency is None else latency <= MEDIAN_LATENCY_TARGET_S)
        + " |"
    )
    calls = card.calls_per_plan
    lines.append(
        "| LLM calls per plan | "
        + ("n/a" if calls is None else f"{calls:.1f}")
        + " | | - | tracked |"
    )

    unmet = card.unmet()
    lines += ["", "## Verdict", ""]
    if card.missing:
        lines.append(
            f"Incomplete: {len(card.missing)} scenarios have no record yet, so this "
            "is not a sign-off."
        )
    lines.append(
        "All targets met." if not unmet else "Targets missed: " + "; ".join(unmet) + "."
    )

    if card.failures:
        lines += ["", "## Failures", ""]
        for scenario_id, score in card.failures:
            lines.append(f"- `{scenario_id}` {score.metric}: {score.detail}")
    if card.missing:
        lines += ["", "## Missing", "", ", ".join(f"`{m}`" for m in card.missing)]
    if card.unknown:
        lines += [
            "",
            "## Ignored records",
            "",
            "No scenario with these ids in the dataset (renamed or removed): "
            + ", ".join(f"`{u}`" for u in card.unknown),
        ]
    return "\n".join(lines) + "\n"


def latest_run_id(root: Path = DEFAULT_RUNS_DIR) -> str | None:
    """The newest run (run ids sort by time), or None if there is none."""
    if not root.is_dir():
        return None
    runs = sorted(
        p.name for p in root.iterdir() if p.is_dir() and any(p.glob("*.json"))
    )
    return runs[-1] if runs else None


def scorecard_for_run(
    run_id: str,
    root: Path = DEFAULT_RUNS_DIR,
    scenarios: Sequence[Scenario] | None = None,
    extra_scorers: dict[str, Scorer] | None = None,
) -> Scorecard:
    store = RunStore(run_id, root=root)
    scenarios = load_scenarios() if scenarios is None else scenarios
    return build_scorecard(run_id, scenarios, store.load_all(), extra_scorers)


# --- CLI ------------------------------------------------------------------------


def main(argv: list[str] | None = None, root: Path = DEFAULT_RUNS_DIR) -> int:
    parser = argparse.ArgumentParser(
        description="Score a saved eval run against the §7.3 targets."
    )
    parser.add_argument("--run-id", help="the run to score (default: the latest)")
    parser.add_argument(
        "--judge",
        action="store_true",
        help="also ask the LLM judge about each plan (one LLM call per plan)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help=f"print only; do not write {SCORECARD_FILE}",
    )
    args = parser.parse_args(argv)

    run_id = args.run_id or latest_run_id(root)
    if run_id is None:
        print(f"No saved runs in {root}. Run the scenarios first.", file=sys.stderr)
        return 1

    extra: dict[str, Scorer] = {}
    if args.judge:
        # Imported here so scoring without the judge never loads Gemini code.
        from ai_travel_assistant.evals.judge import (
            METRIC,
            make_judge_scorer,
            make_llm_grader,
        )
        from ai_travel_assistant.evals.run import FREE_TIER_INTERVAL_S, CallPacer
        from ai_travel_assistant.llm_basics import ConfigError, build_llm, load_settings

        try:
            settings = load_settings()
        except ConfigError as exc:
            print(f"Configuration error: {exc}", file=sys.stderr)
            return 1
        grader = make_llm_grader(
            build_llm(settings, temperature=0),
            callbacks=[CallPacer(min_interval=FREE_TIER_INTERVAL_S)],
        )
        extra[METRIC] = make_judge_scorer(grader)

    try:
        card = scorecard_for_run(run_id, root=root, extra_scorers=extra)
    except (RunRecordError, ScenarioFileError) as exc:
        print(f"Cannot score run {run_id}: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # e.g. the judge's Gemini call failed
        print(f"Scoring failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    text = render_markdown(card)
    print(text)
    if not args.no_save:
        path = RunStore(run_id, root=root).dir / SCORECARD_FILE
        path.write_text(text, encoding="utf-8")
        print(f"Saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
