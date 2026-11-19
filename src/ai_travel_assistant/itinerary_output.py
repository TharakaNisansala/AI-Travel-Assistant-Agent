"""Stage 8: the structured itinerary, from the agent's submission to the reply.

What this file demonstrates:
  1. A "final-answer tool". The agent finishes an itinerary by calling
     submit_itinerary, whose arguments ARE the ItineraryDraft schema. Gemini's
     function calling fills the structure in the same call that would have
     written the prose plan, so structured output costs no extra request.
     The tool is never executed: the compose_answer node handles the call.
  2. Separating generation from presentation. compose_itinerary() validates
     the draft (Pydantic), grounds it in the tool results
     (itinerary_grounding.py) and renders Markdown with a fixed template. No
     LLM call: the same input always gives the same output.
  3. Bounded repair. An invalid submission gets ONE chance to be fixed: a
     short list of field errors goes back to the agent as the tool's result.
     A second failure is salvaged without any call: every day, activity and
     list item that validates on its own is kept.

Every submit call gets a ToolMessage result (accepted, or the errors), so the
saved thread never holds a tool call without its result (see Stage 7).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date

from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ValidationError

from ai_travel_assistant.context_window import current_turn_start
from ai_travel_assistant.itinerary_budget import budget_markdown, build_costs
from ai_travel_assistant.itinerary_grounding import SUBMIT_TOOL, ground_itinerary
from ai_travel_assistant.schemas import (
    Activity,
    CostItem,
    DayPlanDraft,
    Itinerary,
    ItineraryDraft,
    PracticalFact,
    TravelLeg,
)

# --- The final-answer tool ------------------------------------------------------------

SUBMIT_DESCRIPTION = (
    "Submit the finished itinerary. Call this once, as the last step of a "
    "request for an itinerary or a change to one, instead of writing the plan "
    "as text. Include every day. The plan is checked against the tool results "
    "and shown to the traveller automatically."
)


def _never_run(**_: object) -> str:
    raise RuntimeError("submit_itinerary is handled by the compose_answer node")


submit_itinerary = StructuredTool.from_function(
    func=_never_run,
    name=SUBMIT_TOOL,
    description=SUBMIT_DESCRIPTION,
    args_schema=ItineraryDraft,
)

ITINERARY_RULE = (
    " Itinerary output: when you have the data for an itinerary, or for a "
    "change to one, do not write the plan as text. Call submit_itinerary once "
    "with the whole plan, every day included. Fill it only from tool results: "
    "sights as named by find_attractions or the guide, with that source; "
    "travel legs only between places you estimated with estimate_travel_time; "
    "costs only for prices quoted in the guide (quoted_currency), such as "
    "rooms or boat trips, with the other currency from convert_currency, "
    "per (person or group) and the quantity the trip needs, explaining it and "
    "any assumption in quantity_note (e.g. '2 rooms x 2 nights, assuming 2 "
    "people per room'); never entrance fees, which are added automatically "
    "from get_attraction_prices for the sights in the plan; forecast_date "
    "only for days the forecast "
    "covers. Leave anything unknown null and list it in not_available. Put "
    "factual details (transport, opening hours, safety, prices) in facts, each "
    "with a short verbatim quote and its source; tips are general suggestions "
    "only. For a change to an earlier itinerary, submit the complete updated "
    "plan, calling tools only for new places or data. Answer other questions "
    "in text as usual."
)

# --- Messages -------------------------------------------------------------------------

ITINERARY_NAME = "itinerary"  # the name of the rendered reply (an AIMessage)
MAX_ITINERARY_REPAIRS = 1

ACCEPTED_RESULT = "Itinerary checked and shown to the traveller. Do not repeat it."
REPAIR_PREFIX = "The itinerary was not accepted. Fix these fields and call"
NOT_RUN_WITH_SUBMIT = "Not run: submit_itinerary ended this step."
SALVAGE_WARNING = (
    "Parts of this plan could not be validated and were left out; ask me to "
    "redo it if something is missing."
)
UNUSABLE_REPLY = (
    "Sorry, I gathered the information but could not put the itinerary "
    "together. Please ask again, perhaps for fewer days at a time."
)


def summarize_validation_error(exc: ValidationError, limit: int = 8) -> str:
    """Field paths and messages only (never the submitted values)."""
    lines = []
    for error in exc.errors()[:limit]:
        where = ".".join(str(part) for part in error["loc"]) or "itinerary"
        lines.append(f"- {where}: {error['msg']}")
    extra = exc.error_count() - limit
    if extra > 0:
        lines.append(f"- and {extra} more")
    return "\n".join(lines)


def repairs_this_turn(messages: Sequence[AnyMessage]) -> int:
    """How many submissions were sent back for repair since the traveller spoke."""
    return sum(
        1
        for message in messages[current_turn_start(messages) :]
        if isinstance(message, ToolMessage)
        and message.name == SUBMIT_TOOL
        and message.status == "error"
    )


# --- Salvage: keep whatever validates on its own --------------------------------------


def _valid_items(raw: object, model: type[BaseModel]) -> list:
    if not isinstance(raw, list):
        return []
    kept = []
    for item in raw:
        try:
            kept.append(model.model_validate(item))
        except ValidationError:
            continue
    return kept


def _text(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _strings(raw: object) -> list[str]:
    return (
        [s.strip() for s in raw if isinstance(s, str) and s.strip()]
        if isinstance(raw, list)
        else []
    )


def salvage_draft(
    args: object, trip_request: Mapping | None = None
) -> ItineraryDraft | None:
    """Build a valid draft from the parts of an invalid submission (no LLM).

    Days are kept in their submitted order and renumbered 1..n; a day survives
    if at least one of its activities is valid. Returns None when no day or no
    destination can be recovered.
    """
    if not isinstance(args, Mapping):
        return None
    destination = _text(args.get("destination")) or _text(
        (trip_request or {}).get("destination")
    )
    raw_days = args.get("day_plans")
    if destination is None or not isinstance(raw_days, list):
        return None
    days = []
    for raw in raw_days:
        if not isinstance(raw, Mapping):
            continue
        activities = _valid_items(raw.get("activities"), Activity)
        if not activities:
            continue
        number = len(days) + 1
        days.append(
            DayPlanDraft(
                day=number,
                theme=_text(raw.get("theme")) or f"Day {number}",
                forecast_date=raw.get("forecast_date")
                if isinstance(raw.get("forecast_date"), str)
                else None,
                travel=_valid_items(raw.get("travel"), TravelLeg),
                activities=activities,
            )
        )
    if not days:
        return None
    start = _text(args.get("start_from"))
    return ItineraryDraft(
        destination=destination,
        days=len(days),
        start_from=start or "Not stated",
        start_assumed=bool(args.get("start_assumed")) if start else False,
        summary=_text(args.get("summary")) or "",
        day_plans=days,
        costs=_valid_items(args.get("costs"), CostItem),
        facts=_valid_items(args.get("facts"), PracticalFact),
        tips=_strings(args.get("tips")),
        assumptions=_strings(args.get("assumptions")),
        not_available=_strings(args.get("not_available")),
    )


# --- Rendering ------------------------------------------------------------------------


def _duration(minutes: float) -> str:
    total = max(1, round(minutes))
    hours, mins = divmod(total, 60)
    if hours == 0:
        return f"{mins} min"
    return f"{hours} h {mins} min" if mins else f"{hours} h"


def _amount(value: float | None, currency: str) -> str:
    if value is None:
        return "-"
    if currency == "LKR" and value == int(value):
        return f"{value:,.0f}"
    return f"{value:,.2f}"


def _hours(value: float) -> str:
    return f"{value:g} h"


def _sources_used(itinerary: Itinerary) -> list[str]:
    used: list[str] = []

    def add(name: str) -> None:
        if name not in used:
            used.append(name)

    for plan in itinerary.day_plans:
        for activity in plan.activities:
            add(activity.source)
        if any(
            leg.road_distance_km is not None or leg.driving_minutes is not None
            for leg in plan.travel
        ):
            add("OSRM")
        if plan.weather:
            add("Open-Meteo")
    if itinerary.costs:
        add("Wikivoyage")
        if any(
            (c.amount_usd if c.quoted_currency == "LKR" else c.amount_lkr) is not None
            for c in itinerary.costs
        ):
            add("ExchangeRate-API")
    for fact in itinerary.facts:
        add(fact.source)
    for fee in itinerary.entrance_fees:
        if fee.name:
            for line in fee.sources:
                add(line.removeprefix("Source: ").split(" (")[0])
    return used


def render_itinerary_markdown(itinerary: Itinerary) -> str:
    """The traveller-facing plan. Empty sections are left out; nothing is invented."""
    lines = [f"# {itinerary.days}-day trip to {itinerary.destination}"]
    if itinerary.summary:
        lines += ["", itinerary.summary]
    start = itinerary.start_from + (" (assumed)" if itinerary.start_assumed else "")
    lines += ["", f"**Starting point:** {start}"]

    for plan in itinerary.day_plans:
        lines += ["", f"## Day {plan.day}: {plan.theme}"]
        lines.append(f"*Weather:* {plan.weather or 'No forecast for this day.'}")
        for leg in plan.travel:
            parts = []
            if leg.road_distance_km is not None:
                parts.append(f"{leg.road_distance_km:,.0f} km")
            if leg.driving_minutes is not None:
                parts.append(
                    f"about {_duration(leg.driving_minutes)} by car without traffic"
                )
            figures = (
                ", ".join(parts) + " (OSRM)"
                if parts
                else "distance and time not available"
            )
            lines.append(f"*Travel:* {leg.from_place} to {leg.to_place}: {figures}")
        for activity in plan.activities:
            details = []
            if activity.estimated_hours is not None:
                details.append(f"about {_hours(activity.estimated_hours)}")
            if activity.distance_km is not None:
                details.append(f"{activity.distance_km:g} km straight-line")
            detail = f" ({'; '.join(details)})" if details else ""
            text = f" - {activity.description}" if activity.description else ""
            lines.append(
                f"- **{activity.time_of_day.capitalize()}: {activity.name}**{detail}"
                f"{text} _({activity.source})_"
            )

    money = budget_markdown(itinerary.model_dump(mode="json"))
    if money:
        lines += money
    elif itinerary.costs:
        lines += ["", "## Costs (indicative)", "| Item | LKR | USD |", "|---|---|---|"]
        for cost in itinerary.costs:
            lkr = _amount(cost.amount_lkr, "LKR")
            usd = _amount(cost.amount_usd, "USD")
            lines.append(f"| {cost.item} | {lkr} | {usd} |")
        lines.append(
            "Prices are as quoted by Wikivoyage and may be out of date; conversions "
            "use mid-market rates."
        )
    sections = (
        (
            "Practical information",
            [f"{f.text} _({f.source})_" for f in itinerary.facts],
            None,
        ),
        (
            "Suggestions",
            itinerary.tips,
            "_General advice, not checked against live data._",
        ),
        ("Assumptions", itinerary.assumptions, None),
        ("Please note", itinerary.warnings, None),
        ("Not available", itinerary.not_available, None),
    )
    for title, items, note in sections:
        if items:
            lines += ["", f"## {title}"]
            if note:
                lines.append(note)
            lines += [f"- {item}" for item in items]
    sources = _sources_used(itinerary)
    if sources:
        lines += ["", f"_Sources: {', '.join(sources)}._"]
    return "\n".join(lines)


# --- Compose: from a submit call to the reply -----------------------------------------


@dataclass
class ComposeResult:
    """What compose_answer adds to the thread."""

    messages: list[AnyMessage]
    itinerary: dict | None  # the checked Itinerary as JSON-ready data
    repair: bool  # True: the errors went back to the agent for one more try


def has_submit_call(message: AnyMessage | None) -> bool:
    return isinstance(message, AIMessage) and any(
        call["name"] == SUBMIT_TOOL for call in message.tool_calls
    )


def compose_itinerary(
    last: AIMessage,
    messages: Sequence[AnyMessage],
    *,
    today: date,
    trip_request: Mapping | None = None,
    allow_repair: bool = True,
) -> ComposeResult:
    """Validate, ground and render the itinerary submitted in `last`.

    `messages` is the saved thread (including `last`), used for grounding.
    Every call in `last` gets a result, in the order of the calls.
    """
    submit = next(call for call in last.tool_calls if call["name"] == SUBMIT_TOOL)
    others = [
        ToolMessage(
            NOT_RUN_WITH_SUBMIT, tool_call_id=c["id"], name=c["name"], status="error"
        )
        for c in last.tool_calls
        if c is not submit
    ]

    def with_results(
        submit_result: ToolMessage, *after: AnyMessage
    ) -> list[AnyMessage]:
        ordered = []
        others_by_id = {m.tool_call_id: m for m in others}
        for call in last.tool_calls:
            ordered.append(
                submit_result if call is submit else others_by_id[call["id"]]
            )
        return [*ordered, *after]

    salvaged = False
    try:
        draft = ItineraryDraft.model_validate(submit["args"])
    except ValidationError as exc:
        if allow_repair and repairs_this_turn(messages) < MAX_ITINERARY_REPAIRS:
            problems = summarize_validation_error(exc)
            errors = ToolMessage(
                f"{REPAIR_PREFIX} submit_itinerary again:\n{problems}",
                tool_call_id=submit["id"],
                name=SUBMIT_TOOL,
                status="error",
            )
            return ComposeResult(with_results(errors), None, repair=True)
        draft = salvage_draft(submit["args"], trip_request)
        salvaged = True

    if draft is not None:
        try:
            checked = ground_itinerary(
                draft,
                messages,
                today=today,
                trip_request=dict(trip_request or {}) or None,
            )
            # Entrance fees, per-person and group costs and the budget are
            # worked out by code from the tool results (itinerary_budget.py).
            checked = checked.model_copy(
                update=build_costs(checked, messages, trip_request, today=today)
            )
            if salvaged:
                checked = checked.model_copy(
                    update={"warnings": [SALVAGE_WARNING, *checked.warnings]}
                )
            markdown = render_itinerary_markdown(checked)
        except Exception:  # a bug in checking or rendering must not crash the turn
            draft = None
    if draft is None:
        failed = ToolMessage(
            "The itinerary could not be used.",
            tool_call_id=submit["id"],
            name=SUBMIT_TOOL,
            status="error",
        )
        return ComposeResult(
            with_results(failed, AIMessage(UNUSABLE_REPLY, name=ITINERARY_NAME)),
            None,
            repair=False,
        )

    accepted = ToolMessage(ACCEPTED_RESULT, tool_call_id=submit["id"], name=SUBMIT_TOOL)
    return ComposeResult(
        with_results(accepted, AIMessage(markdown, name=ITINERARY_NAME)),
        checked.model_dump(mode="json"),
        repair=False,
    )


def make_compose_node(
    today: Callable[[], date] = date.today,
) -> Callable[[Mapping], dict]:
    """The compose_answer node: no LLM call; `today` is injectable for tests."""

    def compose_answer_node(state: Mapping) -> dict:
        messages = state["messages"]
        result = compose_itinerary(
            messages[-1],
            messages,
            today=today(),
            trip_request=state.get("trip_request"),
        )
        update: dict = {"messages": result.messages}
        if result.itinerary is not None:
            update["itinerary"] = result.itinerary
        return update

    return compose_answer_node


def route_after_compose(state: Mapping) -> str:
    """'repair' when the errors went back to the agent, otherwise 'done'."""
    last = state["messages"][-1]
    return "repair" if isinstance(last, ToolMessage) else "done"
