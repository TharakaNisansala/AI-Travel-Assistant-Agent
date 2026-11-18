"""Stage 10: the rich terminal chat (offline, no API key).

rich prints plain text into a StringIO console (no terminal, no colours), so
the tests read exactly what a user would see, minus the spinner animation.
"""

from __future__ import annotations

import io

import pytest
from rich.console import Console

from ai_travel_assistant import chat_ui, tool_calling
from ai_travel_assistant.chat_ui import TurnView, run_rich_chat
from ai_travel_assistant.streaming import Step, Token, TurnDone
from tests.test_itinerary_graph import (
    gather,
    plan,
    submit,
    tools,  # noqa: F401 - pytest fixture
)
from tests.test_streaming import build, streaming


def console() -> tuple[Console, io.StringIO]:
    out = io.StringIO()
    return Console(file=out, width=100, force_terminal=False, color_system=None), out


def chat(graph, *lines, **kwargs) -> str:
    con, out = console()
    inputs = iter(lines)
    run_rich_chat(
        graph, con, read=lambda prompt: next(inputs), thread_id="t1", **kwargs
    )
    return out.getvalue()


# --- Whole chats ----------------------------------------------------------------


@pytest.mark.usefixtures("tools")
def test_an_itinerary_shows_its_steps_then_the_plan_in_a_panel():
    graph = build(streaming(gather(), submit(plan())))
    text = chat(graph, "Plan 2 days in Ella.", "/itinerary", "/quit")
    assert "Sri Lanka travel assistant" in text
    assert "🔎 Checking the weather in Ella" in text
    assert "✅ Checked the itinerary against the tool results" in text
    assert "Your itinerary" in text
    assert "2-day trip to Ella" in text
    assert '"days": 2' in text  # /itinerary JSON
    assert "Bye! Reopen this conversation with --thread t1" in text
    # Steps come before the plan.
    assert text.index("Checking the weather") < text.index("Your itinerary")


def test_a_typed_reply_is_shown_once():
    text = chat(
        build(streaming("Ella is green and calm.")), "What is Ella like?", "/quit"
    )
    assert text.count("Ella is green and calm.") == 1


def test_a_question_gets_its_own_panel():
    text = chat(build(streaming()), "Plan me a trip.", "/quit")
    assert "A quick question" in text
    assert "how many days" in text


def test_off_topic_is_answered_without_the_agent():
    model = streaming()
    text = chat(build(model), "Tell me a joke.", "/quit")
    assert "I can only help with travel in Sri Lanka" in text
    assert model.calls == []


def test_a_failed_turn_is_reported_and_the_chat_goes_on():
    graph = build(
        streaming("Back again.", fail_on={0: RuntimeError("secret detail")}),
        understander=False,
    )
    text = chat(
        graph,
        "Hello",
        "Hello again",
        "/quit",
        describe_error=lambda exc: "Gemini is busy.",
    )
    assert "Request failed: Gemini is busy." in text
    assert "secret detail" not in text
    assert "Back again." in text


def test_commands_and_blank_lines_never_reach_the_agent():
    model = streaming()
    text = chat(build(model), "", "/prefs", "/new", "/quit")
    assert "Long-term memory is off for this chat." in text
    assert "New conversation:" in text
    assert model.calls == []


# --- The view on its own --------------------------------------------------------


def test_view_prints_a_reply_that_did_not_stream():
    con, out = console()
    view = TurnView(con)
    view.start()
    view.handle(Step("🧭 Understood your request"))
    view.handle(TurnDone(answer="**Hello** there", kind="answer"))
    text = out.getvalue()
    assert "🧭 Understood your request" in text
    assert "Hello there" in text  # rendered Markdown, no asterisks


def test_text_typed_before_a_tool_call_does_not_count_as_the_answer():
    con, out = console()
    view = TurnView(con)
    view.start()
    for event in (
        Token("Let me check."),
        Step("🔎 Checking the weather in Ella"),
        Token("It will rain."),
        TurnDone(answer="It will rain.", kind="answer"),
    ):
        view.handle(event)
    text = out.getvalue()
    assert text.count("It will rain.") == 1
    assert text.count("Let me check.") == 1


def test_abort_leaves_the_console_usable():
    con, out = console()
    view = TurnView(con)
    view.start()
    view.handle(Token("Half"))
    view.abort()
    con.print("next line")
    assert "next line" in out.getvalue()


# --- CLI wiring -----------------------------------------------------------------


@pytest.fixture
def fake_cli(monkeypatch, tmp_path):
    from ai_travel_assistant import llm_basics

    settings = llm_basics.GeminiSettings(model="fake-model", api_key="k")
    monkeypatch.setattr(llm_basics, "load_settings", lambda: settings)
    monkeypatch.setattr(llm_basics, "build_llm", lambda s, **kw: streaming("Hi."))
    used = []
    monkeypatch.setattr(
        tool_calling, "run_agent_chat", lambda g, **kw: used.append("plain")
    )
    monkeypatch.setattr(chat_ui, "run_rich_chat", lambda g, **kw: used.append("rich"))
    return used, str(tmp_path / "chat.sqlite")


def test_chat_is_rich_by_default_and_plain_on_request(fake_cli):
    used, db = fake_cli
    assert tool_calling.main(["--chat", "--db", db]) == 0
    assert tool_calling.main(["--chat", "--plain", "--db", db]) == 0
    assert used == ["rich", "plain"]


@pytest.mark.parametrize(
    "argv",
    [["--plain", "Where to hike?"], ["--chat", "--steps"]],
)
def test_flag_combinations_that_make_no_sense_are_refused(fake_cli, argv):
    with pytest.raises(SystemExit):
        tool_calling.main(argv)
