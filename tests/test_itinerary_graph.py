# ruff: noqa: E501 - tool-text fixtures keep their real line lengths.
"""Stage 8 checks: structured itineraries inside the agent graph, all offline.

A scripted fake model plays Gemini (including the submit_itinerary call); the
tool functions are replaced with fixed texts in the real tool formats. Covers
the happy path, repair, salvage, the cap, and Stages 5, 6 and 7 together.
"""

from __future__ import annotations

from datetime import date

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from ai_travel_assistant import itinerary_output, tool_calling
from ai_travel_assistant.itinerary_output import (
    ACCEPTED_RESULT,
    ITINERARY_NAME,
    ITINERARY_RULE,
    REPAIR_PREFIX,
    SALVAGE_WARNING,
)
from ai_travel_assistant.robustness import MAX_TOOL_ROUNDS, repair_history
from ai_travel_assistant.schemas import RequestUnderstanding, TripRequest
from ai_travel_assistant.tool_calling import (
    COMPOSE_ANSWER,
    TOOLS,
    TOOLS_NODE,
    ask_agent,
    build_tool_agent_graph,
    final_answer,
)

TODAY = date(2026, 11, 18)
REAL_WEATHER_TOOL = tool_calling.get_weather_forecast.func  # before any patching

OSM = {
    "ella": """Attractions near Ella, Badulla, Sri Lanka; category: all; within 5 km; nearest first:
- Little Adam's Peak (peak), 1.8 km
- Nine Arch Bridge (viewpoint), 2.1 km
Source: OpenStreetMap via the Overpass API (© OpenStreetMap contributors).""",
}
GUIDE = {
    "ella": """Wikivoyage guide for Ella (all topics):
[See] Ella Rock is a popular half-day hike. Entry to the Ravana Cave costs Rs 500 for foreigners.
[Get in] Trains from Kandy stop at Ella station.""",
}
WEATHER = {
    "ella": """Weather forecast for Ella, Badulla, Sri Lanka, at lat 6.87, lon 81.05.
Daily forecast (2 days):
- 2026-11-18 (Wed): Light rain; min 15.0 °C, max 22.0 °C; chance of precipitation 60%; rainfall 3.2 mm; max wind 12.0 km/h.
- 2026-11-19 (Thu): Partly cloudy; min 14.5 °C, max 24.0 °C; chance of precipitation 20%; rainfall 0.0 mm; max wind 9.0 km/h.""",
}
ROUTE = """Road trip from Colombo to Ella:
- Road distance: 212 km (straight line 150 km).
- Driving time: about 5 h 40 min by car without traffic or stops."""


@pytest.fixture
def tools(monkeypatch):
    """Fixed tool texts, chosen by the call's own arguments."""

    def by_place(table, label):
        return lambda destination, **_: table.get(
            destination.lower(), f"{label} unavailable for '{destination}': none."
        )

    monkeypatch.setattr(
        tool_calling.find_attractions, "func", by_place(OSM, "Attractions")
    )
    monkeypatch.setattr(
        tool_calling.get_destination_guide, "func", by_place(GUIDE, "Guide")
    )
    monkeypatch.setattr(
        tool_calling.get_weather_forecast, "func", by_place(WEATHER, "Weather")
    )
    monkeypatch.setattr(
        tool_calling.estimate_travel_time,
        "func",
        lambda origin, destination, **_: ROUTE,
    )
    monkeypatch.setattr(
        tool_calling.convert_currency,
        "func",
        lambda amount, from_currency, to_currency: (
            f"{amount:.2f} {from_currency} = 1.67 {to_currency}"
        ),
    )


class Scripted(GenericFakeChatModel):
    """Plays Gemini: scripted replies; records calls and the tools bound."""

    calls: list = []
    bound: list = []
    fail_on: dict = {}

    def bind_tools(self, tools, **kwargs):
        self.bound = [getattr(t, "name", getattr(t, "__name__", "")) for t in tools]
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        number = len(self.calls)
        self.calls.append(list(messages))
        if number in self.fail_on:
            raise self.fail_on[number]
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def scripted(*replies, fail_on=None) -> Scripted:
    messages = [r if isinstance(r, AIMessage) else AIMessage(r) for r in replies]
    return Scripted(messages=iter(messages), calls=[], fail_on=fail_on or {})


_IDS = iter(range(10**6))


def step(*calls: tuple[str, dict]) -> AIMessage:
    return AIMessage(
        "",
        tool_calls=[{"name": n, "args": a, "id": f"c{next(_IDS)}"} for n, a in calls],
    )


def gather(place: str = "Ella") -> AIMessage:
    return step(
        ("get_destination_guide", {"destination": place}),
        ("find_attractions", {"destination": place}),
        ("get_weather_forecast", {"destination": place}),
        ("estimate_travel_time", {"origin": "Colombo", "destination": place}),
    )


def plan(days: int = 2, easier: bool = False, **overrides) -> dict:
    day_plans = [
        {
            "day": n,
            "theme": "Easy views" if easier and n == 2 else f"Day {n}",
            "forecast_date": f"2026-11-{17 + n}",
            "travel": [{"from_place": "Colombo", "to_place": "Ella"}] if n == 1 else [],
            "activities": [
                {
                    "time_of_day": "morning",
                    "name": "Nine Arch Bridge"
                    if easier and n == 2
                    else "Little Adam's Peak",
                    "description": "A good walk.",
                    "source": "OpenStreetMap",
                }
            ],
        }
        for n in range(1, days + 1)
    ]
    args = {
        "destination": "Ella",
        "days": days,
        "start_from": "Colombo",
        "start_assumed": True,
        "summary": "Hill-country days.",
        "day_plans": day_plans,
        "costs": [
            {"item": "Ravana Cave entry", "quoted_currency": "LKR", "amount_lkr": 500}
        ],
    }
    args.update(overrides)
    return args


def bad_plan(claimed_days: int) -> dict:
    """A 2-day plan that claims another day count: fails days == len(day_plans)."""
    return {**plan(), "days": claimed_days}


def submit(args: dict) -> AIMessage:
    return step(("submit_itinerary", args))


def build(model, **kwargs):
    kwargs.setdefault("checkpointer", InMemorySaver())
    return build_tool_agent_graph(
        model, structured_itinerary=True, today=lambda: TODAY, **kwargs
    )


def state(graph, thread_id="t1") -> dict:
    return graph.get_state({"configurable": {"thread_id": thread_id}}).values


def every_call_answered(messages) -> bool:
    return repair_history(messages) == list(messages)


# --- Opt-in -----------------------------------------------------------------------------


def test_structured_output_is_off_by_default():
    model = scripted("Hi.")
    graph = build_tool_agent_graph(model, checkpointer=InMemorySaver())
    ask_agent(graph, "t1", "Hello")
    assert "submit_itinerary" not in model.bound
    assert len(model.bound) == len(TOOLS)
    assert COMPOSE_ANSWER not in graph.get_graph().nodes
    assert ITINERARY_RULE not in model.calls[0][0].content


def test_when_on_the_model_can_submit_but_the_tools_node_cannot_run_it():
    model = scripted("Hi.")
    graph = build(model)
    ask_agent(graph, "t1", "Hello")
    assert model.bound[-1] == "submit_itinerary"
    assert len(model.bound) == len(TOOLS) + 1
    assert (
        "submit_itinerary" not in graph.builder.nodes[TOOLS_NODE].runnable.tools_by_name
    )
    assert ITINERARY_RULE in model.calls[0][0].content


# --- The happy path ----------------------------------------------------------------------


def test_an_itinerary_turn_is_rendered_without_an_extra_model_call(tools):
    model = scripted(
        gather(),
        step(
            (
                "convert_currency",
                {"amount": 500, "from_currency": "LKR", "to_currency": "USD"},
            )
        ),
        submit(plan()),
    )
    graph = build(model)

    turn = ask_agent(graph, "t1", "Plan 2 days in Ella")

    assert len(model.calls) == 3  # the same as a prose plan
    reply = turn[-1]
    assert reply.name == ITINERARY_NAME
    assert final_answer(turn).startswith("# 2-day trip to Ella")
    assert turn[-2].content == ACCEPTED_RESULT
    saved = state(graph)
    assert saved["itinerary"]["costs"][0]["amount_usd"] == 1.67
    assert saved["itinerary"]["day_plans"][0]["weather"].startswith("2026-11-18")
    assert every_call_answered(saved["messages"])


def test_a_prose_plan_is_still_accepted(tools):
    model = scripted(gather(), "Day 1: hike. Day 2: views.")
    graph = build(model)
    turn = ask_agent(graph, "t1", "Plan 2 days in Ella")
    assert final_answer(turn) == "Day 1: hike. Day 2: views."
    assert "itinerary" not in state(graph)


def test_a_plain_question_is_unaffected():
    model = scripted("It is a hill town.")
    graph = build(model)
    assert (
        final_answer(ask_agent(graph, "t1", "What is Ella like?"))
        == "It is a hill town."
    )


def test_one_seven_day_plan_fits_the_history_budget(tools):
    model = scripted(
        gather(),
        submit(
            plan(
                days=7,
                day_plans=[
                    {
                        "day": n,
                        "theme": f"Day {n}",
                        "activities": [
                            {
                                "time_of_day": t,
                                "name": "Little Adam's Peak",
                                "description": "A good walk with wide views over the valley.",
                                "source": "OpenStreetMap",
                            }
                            for t in ("morning", "afternoon", "evening")
                        ],
                    }
                    for n in range(1, 8)
                ],
            )
        ),
    )
    graph = build(model)
    turn = ask_agent(graph, "t1", "Plan 7 days in Ella")
    assert final_answer(turn).startswith("# 7-day trip to Ella")
    assert count_tokens_approximately([turn[-1]]) < 2_000  # well under the 8k budget


# --- Repair and salvage -------------------------------------------------------------------


def test_an_invalid_submission_is_repaired_once(tools):
    model = scripted(gather(), submit(bad_plan(3)), submit(plan()))
    graph = build(model)

    turn = ask_agent(graph, "t1", "Plan 2 days in Ella")

    assert len(model.calls) == 3  # exactly one extra call for the repair
    errors = [
        m for m in model.calls[2] if isinstance(m, ToolMessage) and m.status == "error"
    ]
    assert errors[-1].content.startswith(REPAIR_PREFIX)
    assert final_answer(turn).startswith("# 2-day trip to Ella")
    assert state(graph)["itinerary"]["warnings"][:1] != [SALVAGE_WARNING]


def test_a_second_invalid_submission_is_salvaged_without_another_call(tools):
    model = scripted(gather(), submit(bad_plan(3)), submit(bad_plan(4)), "never used")
    graph = build(model)

    turn = ask_agent(graph, "t1", "Plan 2 days in Ella")

    assert len(model.calls) == 3
    assert state(graph)["itinerary"]["warnings"][0] == SALVAGE_WARNING
    assert final_answer(turn).startswith("# 2-day trip to Ella")
    assert every_call_answered(state(graph)["messages"])


# --- The cap and the fallback (Stage 7) -----------------------------------------------------


def weather_rounds(n: int) -> list[AIMessage]:
    return [step(("get_weather_forecast", {"destination": "Ella"})) for _ in range(n)]


def test_a_submission_after_the_last_allowed_round_is_accepted(tools):
    model = scripted(*weather_rounds(MAX_TOOL_ROUNDS), submit(plan()))
    graph = build(model)
    turn = ask_agent(graph, "t1", "Plan 2 days in Ella")
    assert turn[-1].name == ITINERARY_NAME


def test_a_submission_in_the_wrap_up_is_rendered(tools):
    model = scripted(*weather_rounds(MAX_TOOL_ROUNDS + 1), submit(plan()))
    graph = build(model)

    turn = ask_agent(graph, "t1", "Plan 2 days in Ella")

    assert len(model.calls) == MAX_TOOL_ROUNDS + 2
    assert turn[-1].name == ITINERARY_NAME
    assert state(graph)["itinerary"]["days"] == 2
    assert every_call_answered(state(graph)["messages"])


def test_an_invalid_wrap_up_submission_is_salvaged_not_repaired(tools):
    model = scripted(
        *weather_rounds(MAX_TOOL_ROUNDS + 1), submit(bad_plan(9)), "never used"
    )
    graph = build(model)
    ask_agent(graph, "t1", "Plan 2 days in Ella")
    assert len(model.calls) == MAX_TOOL_ROUNDS + 2
    assert state(graph)["itinerary"]["warnings"][0] == SALVAGE_WARNING


def test_a_transient_failure_before_the_submit_is_retried_once(tools):
    from google.genai import errors as genai_errors

    from ai_travel_assistant.robustness import llm_retry_policy

    busy = genai_errors.ServerError(503, {"error": {"code": 503, "message": "x"}})
    model = scripted(gather(), submit(plan()), fail_on={1: busy})
    graph = build(model, llm_retry=llm_retry_policy(initial_interval=0))

    turn = ask_agent(graph, "t1", "Plan 2 days in Ella")

    assert len(model.calls) == 3
    assert turn[-1].name == ITINERARY_NAME
    assert (
        sum(isinstance(m, ToolMessage) and m.content == ACCEPTED_RESULT for m in turn)
        == 1
    )


def test_a_quota_error_during_the_repair_leaves_a_valid_thread(tools):
    from google.genai import errors as genai_errors

    quota = genai_errors.ClientError(
        429, {"error": {"code": 429, "message": "PerDay quota"}}
    )
    model = scripted(gather(), submit(bad_plan(3)), "Hello again.", fail_on={2: quota})
    graph = build(model)

    with pytest.raises(genai_errors.ClientError):
        ask_agent(graph, "t1", "Plan 2 days in Ella")
    assert every_call_answered(
        state(graph)["messages"]
    )  # the bad submit has its result
    assert final_answer(ask_agent(graph, "t1", "Hello?")) == "Hello again."


def test_a_crash_in_compose_is_repaired_for_the_next_turn(tools, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("bug")

    monkeypatch.setattr(itinerary_output, "compose_itinerary", broken)
    model = scripted(gather(), submit(plan()), "Recovered.")
    graph = build(model)

    with pytest.raises(RuntimeError):
        ask_agent(graph, "t1", "Plan 2 days in Ella")
    turn = ask_agent(graph, "t1", "Hello?")

    assert final_answer(turn) == "Recovered."
    assert every_call_answered(
        [m for m in model.calls[-1] if not isinstance(m, SystemMessage)]
    )


def test_a_failed_tool_keeps_the_other_results_in_the_itinerary(tools, monkeypatch):
    # The REAL weather tool, with its service failing through Stage 7's switch.
    monkeypatch.setattr(tool_calling.get_weather_forecast, "func", REAL_WEATHER_TOOL)
    monkeypatch.setenv("FAULT_INJECT", "weather")
    model = scripted(gather(), submit(plan()))
    graph = build(model)

    ask_agent(graph, "t1", "Plan 2 days in Ella")

    itinerary = state(graph)["itinerary"]
    assert [d["weather"] for d in itinerary["day_plans"]] == [None, None]
    assert (
        "Weather forecast (the weather service was unavailable)"
        in itinerary["not_available"]
    )
    assert itinerary["day_plans"][0]["travel"][0]["road_distance_km"] == 212
    assert itinerary["day_plans"][0]["activities"][0]["distance_km"] == 1.8


# --- Stage 5: memory ----------------------------------------------------------------------


def test_a_follow_up_sees_the_rendered_plan_but_not_the_old_tool_data(tools):
    model = scripted(gather(), submit(plan()), "Day 2 is lighter now.")
    graph = build(model)
    ask_agent(graph, "t1", "Plan 2 days in Ella")

    ask_agent(graph, "t1", "Any tips?")

    seen = [m for m in model.calls[-1] if not isinstance(m, SystemMessage)]
    assert [type(m).__name__ for m in seen] == [
        "HumanMessage",
        "AIMessage",
        "HumanMessage",
    ]
    assert seen[1].content.startswith("# 2-day trip to Ella")


def test_preferences_are_saved_after_an_itinerary(tools):
    from ai_travel_assistant.long_term_memory import load_preferences

    store = InMemoryStore()
    graph = build(
        scripted(gather(), submit(plan())),
        store=store,
        preference_extractor=lambda text, known: ["vegetarian"],
    )
    ask_agent(graph, "t1", "I'm vegetarian. Plan 2 days in Ella")
    assert load_preferences(store, "local") == ["vegetarian"]
    assert state(graph)["itinerary"]["days"] == 2


def test_the_itinerary_survives_a_restart(tools, tmp_path):
    from ai_travel_assistant.tool_calling import sqlite_checkpointer

    db = tmp_path / "chat.sqlite"
    with sqlite_checkpointer(db) as saver:
        ask_agent(
            build(scripted(gather(), submit(plan())), checkpointer=saver),
            "t1",
            "Plan 2 days in Ella",
        )
    with sqlite_checkpointer(db) as saver:
        assert (
            state(build(scripted(), checkpointer=saver))["itinerary"]["destination"]
            == "Ella"
        )


# --- Stage 6: clarification and modify ----------------------------------------------------


class Understander:
    def __init__(self):
        self.calls = []

    def __call__(self, text, **context):
        self.calls.append((text, context))
        answers = {
            "Plan me a trip": RequestUnderstanding(intent="plan"),
            "Ella, 2 days": RequestUnderstanding(
                intent="plan",
                trip=TripRequest(
                    destination="Ella",
                    days=2,
                    origin="Colombo",
                    wants_hotels=False,
                    travellers=2,
                    nationality="British",
                    children=0,
                ),
            ),
            "Make day 2 easier": RequestUnderstanding(intent="modify"),
        }
        return answers[text]


def test_clarify_then_plan_then_modify(tools):
    understander = Understander()
    model = scripted(gather(), submit(plan()), submit(plan(easier=True)))
    graph = build(model, understander=understander)

    first = ask_agent(graph, "t1", "Plan me a trip")
    assert first[-1].name == "clarification"
    planned = ask_agent(graph, "t1", "Ella, 2 days")
    assert planned[-1].name == ITINERARY_NAME
    edited = ask_agent(graph, "t1", "Make day 2 easier")

    assert understander.calls[-1][1].get("has_plan") is True  # the rendered plan counts
    assert len(model.calls) == 3  # the edit reuses earlier tool data: one call
    itinerary = state(graph)["itinerary"]
    assert itinerary["day_plans"][1]["theme"] == "Easy views"
    # Grounded in the earlier Ella results: the OSM distance of THIS sight.
    assert itinerary["day_plans"][1]["activities"][0]["distance_km"] == 2.1
    assert itinerary["warnings"] == []
    assert final_answer(edited).startswith("# 2-day trip to Ella")


def test_the_requested_day_count_is_checked(tools):
    graph = build(scripted(gather(), submit(plan(days=1))), understander=Understander())
    ask_agent(graph, "t1", "Plan me a trip")
    ask_agent(graph, "t1", "Ella, 2 days")
    assert (
        "You asked for 2 days; this plan has 1."
        in state(graph)["itinerary"]["warnings"]
    )


# --- CLI: steps, /itinerary and --json ------------------------------------------------


def test_steps_show_the_submission_and_the_rendering(tools):
    from ai_travel_assistant.tool_calling import describe_steps

    turn = ask_agent(
        build(scripted(gather(), submit(plan()))), "t1", "Plan 2 days in Ella"
    )
    lines = describe_steps(turn)
    assert "[llm -> submit_itinerary] 2 days in Ella" in lines
    assert lines[-1] == "[compose -> itinerary checked and shown]"


def test_chat_itinerary_command_prints_the_saved_json(tools):
    import json

    from ai_travel_assistant.tool_calling import NO_ITINERARY, run_agent_chat

    graph = build(scripted(gather(), submit(plan())))
    replies = iter(["/itinerary", "Plan 2 days in Ella", "/itinerary", "/quit"])
    shown: list[str] = []

    run_agent_chat(graph, lambda prompt: next(replies), shown.append, thread_id="t1")

    assert NO_ITINERARY in shown
    saved = json.loads(shown[-2])
    assert saved == state(graph)["itinerary"]
    assert saved["destination"] == "Ella"


@pytest.fixture
def fake_settings(monkeypatch):
    from ai_travel_assistant import llm_basics

    settings = llm_basics.GeminiSettings(model="fake-model", api_key="fake")
    monkeypatch.setattr(llm_basics, "load_settings", lambda: settings)
    return llm_basics


def test_one_shot_json_prints_the_itinerary(tools, fake_settings, monkeypatch, capsys):
    import json

    model = scripted(gather(), submit(plan()))
    monkeypatch.setattr(fake_settings, "build_llm", lambda s, **kw: model)

    assert tool_calling.main(["--json", "Plan 2 days in Ella"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert (printed["destination"], printed["days"]) == ("Ella", 2)
    assert "submit_itinerary" in model.bound


def test_one_shot_json_without_an_itinerary_says_so(fake_settings, monkeypatch, capsys):
    from ai_travel_assistant.tool_calling import NO_ITINERARY

    monkeypatch.setattr(
        fake_settings, "build_llm", lambda s, **kw: scripted("Just text.")
    )
    assert tool_calling.main(["--json", "What is Ella like?"]) == 0
    out = capsys.readouterr()
    assert NO_ITINERARY in out.err
    assert out.out.strip() == "Just text."


def test_one_shot_without_json_prints_the_rendered_plan(
    tools, fake_settings, monkeypatch, capsys
):
    monkeypatch.setattr(
        fake_settings, "build_llm", lambda s, **kw: scripted(gather(), submit(plan()))
    )
    assert tool_calling.main(["Plan 2 days in Ella"]) == 0
    assert capsys.readouterr().out.startswith("# 2-day trip to Ella")


def test_json_and_chat_do_not_mix():
    with pytest.raises(SystemExit):
        tool_calling.main(["--chat", "--json"])


def test_chat_turns_structured_itineraries_on(fake_settings, monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(fake_settings, "build_llm", lambda s, **kw: scripted("Hi."))
    from ai_travel_assistant import chat_ui

    monkeypatch.setattr(
        chat_ui, "run_rich_chat", lambda graph, **kw: seen.setdefault("graph", graph)
    )
    assert tool_calling.main(["--chat", "--db", str(tmp_path / "chat.sqlite")]) == 0
    assert COMPOSE_ANSWER in seen["graph"].get_graph().nodes
