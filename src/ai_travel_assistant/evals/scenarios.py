"""Stage 9, step 1: the golden dataset of evaluation scenarios.

What this file demonstrates:
  1. A GOLDEN DATASET for a non-deterministic agent. The answer text changes
     from run to run, so a scenario never stores an expected answer. It stores
     checkable PROPERTIES of the behaviour: the route taken, the trip fields
     extracted, the tools called, the number of days planned.
  2. Data separate from code. Scenarios live in evals/scenarios.yaml; adding
     one needs no Python.
  3. Validating the dataset itself. A typo in the YAML ("requird_tools", a
     tool that does not exist, the same id twice) fails when the file is
     LOADED. Otherwise it would show up later as a low score, and you would
     go looking for a bug in the agent that is really a bug in the test.

Each category has its own rules (CATEGORY_RULES): which `expect` fields it
must set, which it may set, and how many turns it has. A field set in the
wrong category is an error, so every expectation is one a scorer will read.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ai_travel_assistant.evals.records import SCENARIO_ID_PATTERN
from ai_travel_assistant.faults import FaultConfigError, parse_faults
from ai_travel_assistant.request_understanding import Route
from ai_travel_assistant.schemas import TripRequest
from ai_travel_assistant.tool_calling import TOOLS

# The dataset lives at the project root, next to the saved runs (not inside
# the package): it is test data, edited by hand. Run from a source checkout.
DEFAULT_PATH = Path(__file__).resolve().parents[3] / "evals" / "scenarios.yaml"

Category = Literal["understanding", "routing", "itinerary", "fault", "multi_turn"]
CATEGORIES: tuple[Category, ...] = (
    "understanding",
    "routing",
    "itinerary",
    "fault",
    "multi_turn",
)

TOOL_NAMES = frozenset(tool.name for tool in TOOLS)

# The tool whose results fail when FAULT_INJECT switches a service off.
FAULT_TOOLS: dict[str, str] = {
    "weather": "get_weather_forecast",
    "attractions": "find_attractions",
    "guide": "get_destination_guide",
    "travel": "estimate_travel_time",
    "currency": "convert_currency",
    "pricing": "get_attraction_prices",
}


class ScenarioFileError(ValueError):
    """The scenario file is missing, is not valid YAML, or breaks a rule."""


@dataclass(frozen=True)
class CategoryRule:
    required: frozenset[str]  # expect fields the category must set
    allowed: frozenset[str]  # expect fields it may set (includes required)
    min_turns: int = 1
    max_turns: int = 1
    faults: bool = False  # True: `faults` is required; False: not allowed


def _rule(required: set[str], optional: set[str] = frozenset(), **kw) -> CategoryRule:
    return CategoryRule(frozenset(required), frozenset(required | optional), **kw)


_PLAN = {"required_tools", "days", "destination"}

CATEGORY_RULES: dict[Category, CategoryRule] = {
    # One message to the understander; only the extracted fields are checked.
    "understanding": _rule({"trip"}),
    # One message; only the route (agent, clarify or off_topic) is checked.
    "routing": _rule({"route"}),
    # One message that should end in a checked itinerary.
    "itinerary": _rule(_PLAN),
    # As itinerary, with services switched off by FAULT_INJECT.
    "fault": _rule(_PLAN, faults=True),
    # A short conversation; the final plan is checked.
    "multi_turn": _rule(
        {"days", "destination"},
        {"routes", "required_tools", "plan_replaced"},
        min_turns=2,
        max_turns=3,
    ),
}


class Expectation(BaseModel):
    """What a scenario checks. Only the fields that are set are scored."""

    model_config = ConfigDict(extra="forbid")

    # Routing: the route the first (only) message takes.
    route: Route | None = None
    # Multi-turn: the route of every message, in order.
    routes: list[Route] | None = None
    # Understanding: TripRequest fields. A null value means the field must
    # stay empty (the traveller did not say it). Interests are matched loosely
    # by the scorer ("tea" matches "tea plantations").
    trip: dict[str, object] | None = None
    # Tools the run must call at least once (more calls are fine).
    required_tools: list[str] | None = None
    # The itinerary: its number of days and its destination.
    days: int | None = Field(default=None, ge=1, le=60)
    destination: str | None = None
    # Multi-turn: the last turn replaced the plan from an earlier turn.
    plan_replaced: bool | None = None

    @model_validator(mode="after")
    def _check_values(self) -> Expectation:
        if self.required_tools is not None:
            if not self.required_tools:
                raise ValueError("required_tools is empty; leave it out instead")
            unknown = sorted(set(self.required_tools) - TOOL_NAMES)
            if unknown:
                raise ValueError(f"unknown tools {unknown}; use {sorted(TOOL_NAMES)}")
            if len(set(self.required_tools)) != len(self.required_tools):
                raise ValueError("required_tools lists a tool twice")
        if self.trip is not None:
            if not self.trip:
                raise ValueError("trip is empty; name at least one field")
            unknown = sorted(set(self.trip) - set(TripRequest.model_fields))
            if unknown:
                raise ValueError(
                    f"unknown trip fields {unknown}; "
                    f"use {sorted(TripRequest.model_fields)}"
                )
            # The values must be ones the understander could produce.
            TripRequest.model_validate(
                {k: v for k, v in self.trip.items() if v is not None}
            )
        return self


class Scenario(BaseModel):
    """One evaluation case: the messages to send and what to check."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=SCENARIO_ID_PATTERN)
    category: Category
    turns: list[str] = Field(min_length=1)
    # A FAULT_INJECT value, e.g. "weather" or "guide:busy" (fault only).
    faults: str | None = None
    notes: str | None = None
    expect: Expectation

    @model_validator(mode="after")
    def _check_category_rules(self) -> Scenario:
        rule = CATEGORY_RULES[self.category]
        where = f"category {self.category!r}"

        if any(not turn.strip() for turn in self.turns):
            raise ValueError("a turn is empty")
        if not rule.min_turns <= len(self.turns) <= rule.max_turns:
            span = (
                str(rule.min_turns)
                if rule.min_turns == rule.max_turns
                else f"{rule.min_turns}-{rule.max_turns}"
            )
            raise ValueError(f"{where} needs {span} turns, got {len(self.turns)}")

        set_fields = {
            name
            for name in self.expect.model_fields_set
            if getattr(self.expect, name) is not None
        }
        missing = sorted(rule.required - set_fields)
        if missing:
            raise ValueError(f"{where} must set expect fields {missing}")
        extra = sorted(set_fields - rule.allowed)
        if extra:
            raise ValueError(
                f"{where} does not score expect fields {extra}; "
                f"allowed: {sorted(rule.allowed)}"
            )

        if rule.faults and not self.faults:
            raise ValueError(f"{where} needs `faults`")
        if not rule.faults and self.faults:
            raise ValueError(f"only category 'fault' may set `faults` ({where})")
        if self.faults:
            try:
                if not parse_faults(self.faults):
                    raise ValueError("`faults` names no service")
            except FaultConfigError as error:
                raise ValueError(str(error)) from None

        routes = self.expect.routes
        if routes is not None and len(routes) != len(self.turns):
            raise ValueError(
                f"expect.routes has {len(routes)} entries for {len(self.turns)} turns"
            )
        return self

    @property
    def failing_tools(self) -> frozenset[str]:
        """The tools whose results must fail because of `faults`."""
        return frozenset(FAULT_TOOLS[s] for s in parse_faults(self.faults))


class ScenarioFile(BaseModel):
    """The whole YAML file."""

    model_config = ConfigDict(extra="forbid")

    version: Literal[1]
    scenarios: list[Scenario] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_ids(self) -> ScenarioFile:
        twice = sorted(
            i for i, n in Counter(s.id for s in self.scenarios).items() if n > 1
        )
        if twice:
            raise ValueError(f"scenario ids used more than once: {twice}")
        return self


def parse_scenarios(text: str, source: str = "<text>") -> list[Scenario]:
    """Validate YAML text into scenarios. Errors name the source and the rule."""
    try:
        # safe_load builds only plain data (dicts, lists, strings, numbers);
        # yaml.load could construct arbitrary Python objects from tags.
        data = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise ScenarioFileError(f"{source}: not valid YAML: {error}") from None
    try:
        return ScenarioFile.model_validate(data).scenarios
    except ValidationError as error:
        raise ScenarioFileError(f"{source}: {error}") from None


def select_scenarios(
    scenarios: Iterable[Scenario],
    categories: Iterable[str] | None = None,
    only: Iterable[str] | None = None,
) -> list[Scenario]:
    """Keep the scenarios in `categories` and/or with an id in `only`.

    Unknown names are errors, so a typo never silently runs nothing.
    """
    scenarios = list(scenarios)
    if categories is not None:
        wanted = set(categories)
        unknown = sorted(wanted - set(CATEGORIES))
        if unknown:
            raise ScenarioFileError(
                f"unknown categories {unknown}; use {list(CATEGORIES)}"
            )
        scenarios = [s for s in scenarios if s.category in wanted]
    if only is not None:
        wanted = set(only)
        unknown = sorted(wanted - {s.id for s in scenarios})
        if unknown:
            raise ScenarioFileError(f"no scenario with ids {unknown} in this selection")
        scenarios = [s for s in scenarios if s.id in wanted]
    return scenarios


def load_scenarios(
    path: Path = DEFAULT_PATH,
    categories: Iterable[str] | None = None,
    only: Iterable[str] | None = None,
) -> list[Scenario]:
    """Load, validate and filter the scenario file."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ScenarioFileError(f"cannot read {path}: {error}") from None
    return select_scenarios(parse_scenarios(text, str(path)), categories, only)
