"""Stage 9 sign-off: `uv run pytest -m eval` (deselected by default).

Scores a SAVED run (no Gemini calls) against the §7.3 targets. Make the run
first with `python -m ai_travel_assistant.evals.run`, over several days if
needed. Pick a run with EVAL_RUN_ID (default: the latest), e.g. in PowerShell:
    $env:EVAL_RUN_ID = "2026-11-18T1530"; uv run pytest -m eval -s

An incomplete run FAILS rather than skips: a partial run is not a sign-off.
"""

from __future__ import annotations

import os

import pytest

from ai_travel_assistant.evals.scorecard import (
    latest_run_id,
    render_markdown,
    scorecard_for_run,
)

pytestmark = pytest.mark.eval


def test_saved_run_meets_the_v1_scorecard():
    run_id = os.environ.get("EVAL_RUN_ID") or latest_run_id()
    if run_id is None:
        pytest.skip("no saved eval run; run python -m ai_travel_assistant.evals.run")
    card = scorecard_for_run(run_id)
    print(render_markdown(card))
    assert card.complete, f"run {run_id} is incomplete; missing: {card.missing}"
    assert card.unmet() == [], f"run {run_id} missed targets: {card.unmet()}"
