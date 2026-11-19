"""Stage 10: the Streamlit page, run headless with Streamlit's AppTest (offline).

AppTest executes the script in this process, so the usual fakes apply: the
streaming scripted model, the fake understander and fixed tool texts. Skipped
when the optional `ui` extra (streamlit) is not installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from ai_travel_assistant import llm_basics, tool_calling, web_app, web_ui  # noqa: E402
from ai_travel_assistant.request_understanding import QUICK_REPLIES  # noqa: E402
from tests.test_clarification import FakeUnderstander  # noqa: E402
from tests.test_itinerary_graph import (  # noqa: E402
    TODAY,
    gather,
    plan,
    submit,
    tools,  # noqa: F401 - pytest fixture
)
from tests.test_streaming import ANSWERS, streaming  # noqa: E402

APP = str(Path(web_app.__file__))


@pytest.fixture
def page(monkeypatch, tmp_path):
    """Returns a function: model -> a started AppTest on a fresh database."""
    import streamlit as st

    st.cache_resource.clear()
    settings = llm_basics.GeminiSettings(model="fake-model", api_key="k")
    monkeypatch.setattr(llm_basics, "load_settings", lambda: settings)
    monkeypatch.setenv(web_app.DB_ENV, str(tmp_path / "chat.sqlite"))

    def start(model):
        def fake_build(settings, checkpointer, store, **kwargs):
            return tool_calling.build_tool_agent_graph(
                model,
                checkpointer=checkpointer,
                store=store,
                understander=FakeUnderstander(ANSWERS),
                structured_itinerary=True,
                today=lambda: TODAY,
            )

        monkeypatch.setattr(tool_calling, "build_chat_graph", fake_build)
        app = AppTest.from_file(APP, default_timeout=30)
        app.run()
        assert not app.exception, app.exception
        return app

    yield start
    st.cache_resource.clear()


def texts(app) -> list[str]:
    return [md.value for message in app.chat_message for md in message.markdown]


def suggestion_buttons(app) -> list:
    return [b for b in app.button if (b.key or "").startswith("suggest-")]


def test_the_empty_page_shows_the_title_and_an_input(page):
    app = page(streaming())
    assert any(web_ui.APP_NAME in md.value for md in app.markdown)
    assert len(app.chat_input) == 1
    assert app.chat_message == []
    assert any("Thread" in c.value for c in app.sidebar.caption)
    # The empty conversation offers starter suggestions.
    assert len(suggestion_buttons(app)) == len(web_ui.SUGGESTIONS)


def test_the_header_names_the_provider_and_model(page):
    app = page(streaming())
    assert any("Gemini · fake-model" in md.value for md in app.markdown)


def test_plan_a_trip_for_me_is_the_first_card_and_starts_the_questions(page):
    assert web_ui.SUGGESTIONS[0][0] == "Plan a trip for me"
    app = page(streaming())
    suggestion_buttons(app)[0].click().run()
    assert not app.exception, app.exception
    shown = texts(app)
    assert shown[0] == "Plan me a trip."
    assert shown[-1] == "Where in Sri Lanka would you like to go?"
    assert quick_reply_buttons(app)  # the first question comes with buttons
    # Suggestions are only for an empty conversation.
    assert suggestion_buttons(app) == []


def test_a_suggestion_sends_its_message(page):
    app = page(streaming("Kandy is lovely."))
    [card] = [b for b in suggestion_buttons(app) if "Kandy" in b.label]
    card.click().run()
    assert not app.exception, app.exception
    shown = texts(app)
    assert shown[0] == "Plan 2 days in Kandy"
    assert shown[-1] == "Kandy is lovely."


@pytest.mark.usefixtures("tools")
def test_an_itinerary_turn_ends_with_the_plan_and_its_steps(page):
    app = page(streaming(gather(), submit(plan())))
    app.chat_input[0].set_value("Plan 2 days in Ella.").run()
    assert not app.exception, app.exception
    shown = texts(app)
    assert shown[0] == "Plan 2 days in Ella."
    assert shown[-1].startswith("# 2-day trip to Ella")  # the plain-text version
    # The plan is drawn as cards: one tab per day.
    assert [tab.label for tab in app.tabs] == ["Day 1 · Day 1", "Day 2 · Day 2"]
    assert any("2-day trip to Ella" in md.value for md in app.markdown)
    # The steps survive the rerun in an expander.
    assert any("How the last answer was made" in e.label for e in app.expander)
    # Travellers get the plan as a PDF, in the chat and in the sidebar; the
    # raw JSON is a developer view only.
    keys = [d.key for d in app.get("download_button")]
    assert len(keys) == 2 and "pdf-sidebar" in keys
    assert all(key.startswith("pdf-") for key in keys)
    assert app.sidebar.code == []


@pytest.mark.usefixtures("tools")
def test_dev_mode_shows_the_itinerary_json(page, monkeypatch):
    monkeypatch.setenv(web_app.DEV_ENV, "1")
    app = page(streaming(gather(), submit(plan())))
    app.chat_input[0].set_value("Plan 2 days in Ella.").run()
    assert not app.exception, app.exception
    assert any('"days": 2' in code.value for code in app.sidebar.code)


def quick_reply_buttons(app) -> list:
    return [b for b in app.button if (b.key or "").startswith("quick-")]


def click_quick_reply(app, label: str):
    [button] = [b for b in quick_reply_buttons(app) if b.label == label]
    button.click().run()
    assert not app.exception, app.exception


def test_a_question_waits_for_the_answer(page):
    app = page(streaming())
    app.chat_input[0].set_value("Plan me a trip.").run()
    assert texts(app)[-1] == "Where in Sri Lanka would you like to go?"
    assert any("waiting for your answer" in i.value for i in app.info)
    # The question comes with its answers as buttons, and no starter cards.
    labels = {b.label for b in quick_reply_buttons(app)}  # laid out in columns
    assert labels == {label for label, _ in QUICK_REPLIES["destination"]}
    assert suggestion_buttons(app) == []


@pytest.mark.usefixtures("tools")
def test_quick_replies_answer_every_question_then_the_plan_comes(page):
    app = page(streaming(gather(), submit(plan())))
    app.chat_input[0].set_value("Plan me a trip.").run()

    click_quick_reply(app, "Ella")
    assert texts(app)[-1] == "How many days would you like to spend in Ella?"
    click_quick_reply(app, "2 days")
    assert texts(app)[-1].startswith("How many people are travelling")
    click_quick_reply(app, "3 people")
    assert texts(app)[-1].startswith("What citizenship do the travellers have?")
    click_quick_reply(app, "All Sri Lankan citizens")
    assert texts(app)[-1].startswith("How many of the 3 travellers are children?")
    click_quick_reply(app, "No, all adults")
    assert texts(app)[-1] == "Where will you start your trip to Ella from?"
    click_quick_reply(app, "The airport (Katunayake)")

    shown = texts(app)
    assert "The airport (Katunayake)" in shown  # the click is the reply
    assert shown[-1].startswith("# 2-day trip to Ella")
    assert quick_reply_buttons(app) == []  # nothing is waiting any more


def test_a_failed_turn_shows_a_safe_error_and_the_page_survives(page):
    app = page(streaming(fail_on={0: RuntimeError("secret detail")}))
    app.chat_input[0].set_value("What is Ella like?").run()
    assert not app.exception
    [error] = app.error
    assert error.value.startswith("Request failed:")
    assert "secret detail" not in error.value


def test_new_conversation_starts_an_empty_thread(page):
    app = page(streaming("Ella is green."))
    app.chat_input[0].set_value("What is Ella like?").run()
    assert texts(app)
    first_thread = app.session_state["thread_id"]
    app.sidebar.button[0].click().run()  # "New conversation"
    assert app.session_state["thread_id"] != first_thread
    assert app.chat_message == []


def test_a_missing_key_is_reported_on_the_page(monkeypatch, tmp_path):
    import streamlit as st

    st.cache_resource.clear()

    def no_settings():
        raise llm_basics.ConfigError("Missing GEMINI_API_KEY.")

    monkeypatch.setattr(llm_basics, "load_settings", no_settings)
    monkeypatch.setenv(web_app.DB_ENV, str(tmp_path / "chat.sqlite"))
    app = AppTest.from_file(APP, default_timeout=30)
    app.run()
    assert not app.exception
    assert "Configuration error: Missing GEMINI_API_KEY." in app.error[0].value
    assert len(app.chat_input) == 0
    st.cache_resource.clear()
