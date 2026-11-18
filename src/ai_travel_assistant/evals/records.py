"""Stage 9, step 2: run records, the contract between the runner and scorers.

What this file demonstrates:
  1. RUN and SCORE are separate. The runner (live, spends Gemini calls) writes
     one RunRecord per scenario: the full message trace, plus what the graph
     state said after every turn. Scorers read only records, so a new or fixed
     scorer re-scores saved runs at no Gemini cost. This is "offline
     evaluation over recorded traces", the idea behind LangSmith experiments.
  2. Records are plain JSON. LangChain messages go through messages_to_dict /
     messages_from_dict, so a record loads back into real AIMessage and
     ToolMessage objects with their tool calls.
  3. Resumable runs. Each scenario is saved the moment it finishes, in
     evals/runs/<run-id>/<scenario-id>.json. Running again with the same run
     id skips saved scenarios, so a ~90-call run can span several days of the
     20-a-day free quota.

A record stores facts, never judgements: no pass/fail lives here. Errors keep
the exception type and a short message, never a traceback.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterator, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    ToolMessage,
    messages_from_dict,
    messages_to_dict,
)
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ai_travel_assistant.request_understanding import Route

DEFAULT_RUNS_DIR = Path(__file__).resolve().parents[3] / "evals" / "runs"
RUN_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]*$"
SCENARIO_ID_PATTERN = r"^[a-z0-9]+(-[a-z0-9]+)*$"
MAX_ERROR_CHARS = 500


class RunRecordError(ValueError):
    """A record file is missing, unreadable or does not match the format."""


class TurnRecord(BaseModel):
    """One traveller message and what the graph state said after it."""

    model_config = ConfigDict(extra="forbid")

    text: str
    # Stage 6 state after the turn (None when the graph has no understander).
    route: Route | None = None
    intent: str | None = None
    trip_request: dict | None = None
    missing_fields: list[str] = Field(default_factory=list)
    # The clarification question, when the turn paused to ask one.
    question: str | None = None
    # Stage 8: state["itinerary"] after the turn (None: no checked plan yet).
    itinerary: dict | None = None
    # How many of the record's messages exist after this turn, so a scorer can
    # take one turn's messages as messages[previous.message_count:message_count].
    message_count: int = Field(ge=0)
    llm_calls: int = Field(ge=0)
    # Time spent working, without the runner's free-tier pacing sleeps.
    seconds: float = Field(ge=0)


class ErrorRecord(BaseModel):
    """An exception that ended the run. Type and short message only."""

    model_config = ConfigDict(extra="forbid")

    type: str
    message: str = Field(max_length=MAX_ERROR_CHARS)
    turn: int = Field(ge=0)  # index of the turn that raised

    @classmethod
    def from_exception(cls, error: BaseException, turn: int) -> ErrorRecord:
        message = " ".join(str(error).split())
        if len(message) > MAX_ERROR_CHARS:
            message = message[: MAX_ERROR_CHARS - 3] + "..."
        return cls(type=type(error).__name__, message=message, turn=turn)


class RunRecord(BaseModel):
    """Everything one scenario run produced."""

    model_config = ConfigDict(extra="forbid")

    format: Literal[1] = 1
    run_id: str = Field(pattern=RUN_ID_PATTERN)
    # Same pattern as Scenario.id; it becomes a file name, so no / or ..
    scenario_id: str = Field(pattern=SCENARIO_ID_PATTERN)
    category: str
    started_at: datetime
    model: str  # e.g. the Gemini model name, or "fake" in tests
    faults: str | None = None
    turns: list[TurnRecord] = Field(default_factory=list)
    # The whole thread, as messages_to_dict data. Use .messages() to read it.
    message_data: list[dict] = Field(default_factory=list)
    error: ErrorRecord | None = None

    @model_validator(mode="after")
    def _check_counts(self) -> RunRecord:
        counts = [turn.message_count for turn in self.turns]
        if counts != sorted(counts):
            raise ValueError("turn message counts must not go down")
        if counts and counts[-1] > len(self.message_data):
            raise ValueError("a turn counts more messages than the record holds")
        if self.error and self.error.turn > len(self.turns):
            raise ValueError("the error names a turn that never started")
        return self

    # --- Building ---------------------------------------------------------------

    @staticmethod
    def dump_messages(messages: Sequence[AnyMessage]) -> list[dict]:
        return messages_to_dict(list(messages))

    # --- Reading (helpers the scorers share) ------------------------------------

    @property
    def crashed(self) -> bool:
        return self.error is not None

    def messages(self) -> list[AnyMessage]:
        return messages_from_dict(self.message_data)

    def turn_messages(self, index: int) -> list[AnyMessage]:
        """The messages added during turn `index`."""
        start = self.turns[index - 1].message_count if index > 0 else 0
        return self.messages()[start : self.turns[index].message_count]

    def tool_calls(self) -> list[dict]:
        """Every tool call the model made, in order ({name, args, id})."""
        return [
            call
            for message in self.messages()
            if isinstance(message, AIMessage)
            for call in message.tool_calls
        ]

    def tool_names(self) -> list[str]:
        return [call["name"] for call in self.tool_calls()]

    def tool_results(self) -> list[ToolMessage]:
        return [m for m in self.messages() if isinstance(m, ToolMessage)]

    @property
    def final_itinerary(self) -> dict | None:
        return self.turns[-1].itinerary if self.turns else None

    @property
    def llm_calls(self) -> int:
        return sum(turn.llm_calls for turn in self.turns)

    @property
    def seconds(self) -> float:
        return sum(turn.seconds for turn in self.turns)


def new_run_id(now: datetime | None = None) -> str:
    """A sortable id like 2026-11-18T1530 (local time, minute precision)."""
    return (now or datetime.now()).strftime("%Y-%m-%dT%H%M")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class RunStore:
    """The records of one run, one JSON file per scenario."""

    def __init__(self, run_id: str, root: Path = DEFAULT_RUNS_DIR) -> None:
        if not re.match(RUN_ID_PATTERN, run_id):
            raise RunRecordError(f"bad run id {run_id!r}; use letters, digits, . _ -")
        self.run_id = run_id
        self.dir = root / run_id

    def path(self, scenario_id: str) -> Path:
        if not re.match(SCENARIO_ID_PATTERN, scenario_id):
            raise RunRecordError(f"bad scenario id {scenario_id!r}")
        return self.dir / f"{scenario_id}.json"

    def has(self, scenario_id: str) -> bool:
        return self.path(scenario_id).is_file()

    def save(self, record: RunRecord) -> Path:
        """Write the record. The rename makes it all-or-nothing on disk, so an
        interrupted save never leaves a half file that looks finished."""
        if record.run_id != self.run_id:
            raise RunRecordError(
                f"record is for run {record.run_id!r}, store is {self.run_id!r}"
            )
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.path(record.scenario_id)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(record.model_dump(mode="json"), indent=2, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, path)
        return path

    def load(self, scenario_id: str) -> RunRecord:
        path = self.path(scenario_id)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as error:
            raise RunRecordError(f"cannot read {path}: {error}") from None
        try:
            return RunRecord.model_validate_json(text)
        except ValueError as error:
            raise RunRecordError(f"{path}: {error}") from None

    def scenario_ids(self) -> list[str]:
        if not self.dir.is_dir():
            return []
        return sorted(p.stem for p in self.dir.glob("*.json"))

    def load_all(self) -> Iterator[RunRecord]:
        for scenario_id in self.scenario_ids():
            yield self.load(scenario_id)
