"""Stage 9, step 2: run records and the run store (offline, no API key)."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ai_travel_assistant.evals.records import (
    MAX_ERROR_CHARS,
    ErrorRecord,
    RunRecord,
    RunRecordError,
    RunStore,
    TurnRecord,
    new_run_id,
)

STARTED = datetime(2026, 11, 18, 9, 30, tzinfo=timezone.utc)
ITINERARY = {"destination": "Ella", "days": 2, "day_plans": [{"day": 1}, {"day": 2}]}


def two_turn_messages():
    """A plan turn with a tool call, then a modify turn without tools."""
    return [
        HumanMessage("Plan 2 days in Ella."),
        AIMessage(
            "",
            tool_calls=[
                {
                    "name": "get_weather_forecast",
                    "args": {"destination": "Ella"},
                    "id": "c1",
                },
                {
                    "name": "find_attractions",
                    "args": {"destination": "Ella"},
                    "id": "c2",
                },
            ],
        ),
        ToolMessage(
            "Weather unavailable for 'Ella': down",
            tool_call_id="c1",
            name="get_weather_forecast",
        ),
        ToolMessage(
            "Attractions near Ella: Nine Arch Bridge",
            tool_call_id="c2",
            name="find_attractions",
        ),
        AIMessage("Here is your plan."),
        HumanMessage("Make day 2 easier."),
        AIMessage("Here is the easier plan."),
    ]


def make_record(run_id="test-run", scenario_id="multi-ella", **overrides):
    messages = two_turn_messages()
    fields = dict(
        run_id=run_id,
        scenario_id=scenario_id,
        category="multi_turn",
        started_at=STARTED,
        model="fake",
        turns=[
            TurnRecord(
                text="Plan 2 days in Ella.",
                route="agent",
                intent="plan",
                trip_request={"destination": "Ella", "days": 2},
                itinerary=ITINERARY,
                message_count=5,
                llm_calls=2,
                seconds=1.5,
            ),
            TurnRecord(
                text="Make day 2 easier.",
                route="agent",
                intent="modify",
                itinerary={**ITINERARY, "summary": "easier"},
                message_count=7,
                llm_calls=1,
                seconds=0.5,
            ),
        ],
        message_data=RunRecord.dump_messages(messages),
    )
    fields.update(overrides)
    return RunRecord(**fields)


# --- The record -----------------------------------------------------------------


def test_messages_come_back_as_real_langchain_messages():
    record = make_record()
    messages = record.messages()
    assert [type(m).__name__ for m in messages] == [
        "HumanMessage",
        "AIMessage",
        "ToolMessage",
        "ToolMessage",
        "AIMessage",
        "HumanMessage",
        "AIMessage",
    ]
    assert messages[1].tool_calls[0]["args"] == {"destination": "Ella"}
    assert messages[2].tool_call_id == "c1"


def test_helpers_read_tools_turns_and_totals():
    record = make_record()
    assert record.tool_names() == ["get_weather_forecast", "find_attractions"]
    assert [r.tool_call_id for r in record.tool_results()] == ["c1", "c2"]
    assert len(record.turn_messages(0)) == 5
    assert [m.content for m in record.turn_messages(1)] == [
        "Make day 2 easier.",
        "Here is the easier plan.",
    ]
    assert record.final_itinerary["summary"] == "easier"
    assert record.llm_calls == 3
    assert record.seconds == 2.0
    assert not record.crashed


def test_an_empty_record_has_safe_defaults():
    record = RunRecord(
        run_id="r",
        scenario_id="x",
        category="routing",
        started_at=STARTED,
        model="fake",
    )
    assert record.final_itinerary is None
    assert record.tool_names() == []
    assert record.llm_calls == 0


def test_error_record_keeps_type_and_a_short_one_line_message():
    error = ErrorRecord.from_exception(RuntimeError("boom\n" + "x" * 1000), turn=1)
    assert error.type == "RuntimeError"
    assert "\n" not in error.message
    assert len(error.message) == MAX_ERROR_CHARS
    assert error.message.endswith("...")

    record = make_record(error=error)
    assert record.crashed


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"run_id": "../escape"}, "run_id"),
        ({"scenario_id": "../../etc"}, "scenario_id"),
        ({"message_data": []}, "more messages than the record holds"),
        ({"error": ErrorRecord(type="E", message="m", turn=5)}, "never started"),
        ({"format": 2}, "format"),
    ],
)
def test_invalid_records_are_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        make_record(**overrides)


def test_turn_counts_must_not_go_down():
    record = make_record()
    turns = [record.turns[1], record.turns[0]]
    with pytest.raises(ValueError, match="must not go down"):
        make_record(turns=turns)


# --- The store ------------------------------------------------------------------


def test_save_and_load_round_trip(tmp_path):
    store = RunStore("test-run", root=tmp_path)
    record = make_record()
    path = store.save(record)

    assert path == tmp_path / "test-run" / "multi-ella.json"
    assert store.load("multi-ella") == record
    # Plain, readable JSON with no temp file left behind.
    assert json.loads(path.read_text(encoding="utf-8"))["scenario_id"] == "multi-ella"
    assert list(path.parent.glob("*.tmp")) == []


def test_store_knows_which_scenarios_are_done_so_a_run_can_resume(tmp_path):
    store = RunStore("test-run", root=tmp_path)
    assert store.scenario_ids() == []
    assert not store.has("multi-ella")

    store.save(make_record(scenario_id="multi-ella"))
    store.save(make_record(scenario_id="itin-ella"))

    # A new store for the same run id (e.g. the next day) sees both.
    again = RunStore("test-run", root=tmp_path)
    assert again.has("multi-ella")
    assert again.scenario_ids() == ["itin-ella", "multi-ella"]
    assert [r.scenario_id for r in again.load_all()] == ["itin-ella", "multi-ella"]
    # Another run id is a separate run.
    assert RunStore("other-run", root=tmp_path).scenario_ids() == []


def test_a_leftover_temp_file_is_not_a_finished_scenario(tmp_path):
    store = RunStore("test-run", root=tmp_path)
    store.dir.mkdir(parents=True)
    (store.dir / "itin-ella.json.tmp").write_text("{half", encoding="utf-8")
    assert store.scenario_ids() == []
    assert not store.has("itin-ella")


def test_store_rejects_a_record_from_another_run(tmp_path):
    store = RunStore("test-run", root=tmp_path)
    with pytest.raises(RunRecordError, match="store is 'test-run'"):
        store.save(make_record(run_id="other-run"))


@pytest.mark.parametrize("run_id", ["", "../x", "a/b", ".hidden"])
def test_store_rejects_unsafe_run_ids(tmp_path, run_id):
    with pytest.raises(RunRecordError, match="bad run id"):
        RunStore(run_id, root=tmp_path)


def test_store_rejects_unsafe_scenario_ids(tmp_path):
    with pytest.raises(RunRecordError, match="bad scenario id"):
        RunStore("test-run", root=tmp_path).path("../secrets")


def test_loading_a_missing_or_broken_file_is_a_clear_error(tmp_path):
    store = RunStore("test-run", root=tmp_path)
    with pytest.raises(RunRecordError, match="cannot read"):
        store.load("itin-ella")

    store.dir.mkdir(parents=True)
    store.path("itin-ella").write_text('{"format": 1}', encoding="utf-8")
    with pytest.raises(RunRecordError, match="itin-ella.json"):
        store.load("itin-ella")


def test_new_run_id_is_sortable_and_valid(tmp_path):
    run_id = new_run_id(datetime(2026, 11, 18, 15, 30))
    assert run_id == "2026-11-18T1530"
    RunStore(run_id, root=tmp_path)  # accepted
