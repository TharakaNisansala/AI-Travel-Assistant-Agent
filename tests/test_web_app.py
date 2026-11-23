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
from ai_travel_assistant.sri_lanka_places import PROVINCES  # noqa: E402
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
built_with: list = []  # the settings each build_chat_graph call received


@pytest.fixture
def page(monkeypatch, tmp_path):
    """Returns a function: model -> a started AppTest on a fresh database."""
    import streamlit as st

    built_with.clear()

    st.cache_resource.clear()
    settings = llm_basics.GeminiSettings(model="fake-model", api_key="k")
    monkeypatch.setattr(llm_basics, "load_settings", lambda: settings)
    monkeypatch.setenv(web_app.DB_ENV, str(tmp_path / "chat.sqlite"))
    monkeypatch.delenv(web_app.PUBLIC_ENV, raising=False)  # local mode unless set

    def start(model):
        def fake_build(settings, checkpointer, store, **kwargs):
            built_with.append(settings)
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
    assert texts(app)[-1].startswith("Would you like hotel suggestions")
    click_quick_reply(app, "No thanks")

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


def picker_buttons(app, kind: str) -> list:
    return [b for b in app.button if (b.key or "").startswith(f"{kind}-")]


def test_the_weather_card_opens_provinces_then_districts_without_the_llm(page):
    model = streaming("Light rain in Badulla.")
    app = page(model)
    [card] = [b for b in suggestion_buttons(app) if "Weather" in b.label]
    card.click().run()
    assert not app.exception, app.exception
    assert {b.label for b in picker_buttons(app, "province")} == set(PROVINCES)
    assert texts(app) == []  # nothing sent yet

    [uva] = [b for b in picker_buttons(app, "province") if b.label == "Uva"]
    uva.click().run()
    assert [b.label for b in picker_buttons(app, "district")] == [
        "Badulla",
        "Monaragala",
    ]
    assert model.calls == []  # no LLM call while picking

    [badulla] = [b for b in picker_buttons(app, "district") if b.label == "Badulla"]
    badulla.click().run()
    assert not app.exception, app.exception
    shown = texts(app)
    assert shown[0] == "What's the weather in Badulla?"
    assert shown[-1] == "Light rain in Badulla."
    assert picker_buttons(app, "province") == picker_buttons(app, "district") == []


def test_the_picker_goes_back_to_provinces_and_to_the_cards(page):
    app = page(streaming())
    [card] = [b for b in suggestion_buttons(app) if "Weather" in b.label]
    card.click().run()
    picker_buttons(app, "province")[0].click().run()  # Western
    app.button(key="picker-back").click().run()
    assert len(picker_buttons(app, "province")) == 9
    app.button(key="picker-back").click().run()
    assert len(suggestion_buttons(app)) == len(web_ui.SUGGESTIONS)


# --- Chat commands typed in the chat box (no LLM) -----------------------------------


def infos(app) -> list[str]:
    return [i.value for i in app.info]


def type_in_chat(app, text: str):
    app.chat_input[0].set_value(text).run()
    assert not app.exception, app.exception


@pytest.mark.parametrize(
    ("command", "reply"),
    [
        ("/prefs", "No saved preferences yet."),
        ("/forget", "Forgot 0 saved preferences."),
        ("  /PREFS ", "No saved preferences yet."),  # any case or spacing
    ],
)
def test_preference_commands_answer_without_the_llm(page, command, reply):
    model = streaming("This must not be used.")
    app = page(model)
    type_in_chat(app, command)
    assert reply in infos(app)
    assert texts(app) == []  # the command is not a chat message
    assert model.calls == []


def test_new_starts_an_empty_thread_and_keeps_the_old_one(page):
    model = streaming("Ella is green.")
    app = page(model)
    type_in_chat(app, "What is Ella like?")
    first_thread = app.session_state["thread_id"]

    type_in_chat(app, "/new")
    new_thread = app.session_state["thread_id"]
    assert new_thread != first_thread
    assert f"New conversation: {new_thread}" in infos(app)
    assert texts(app) == []
    assert len(model.calls) == 1  # only the Ella question used the model


def test_quit_keeps_the_browser_conversation(page):
    app = page(streaming("Ella is green."))
    type_in_chat(app, "What is Ella like?")
    thread = app.session_state["thread_id"]

    type_in_chat(app, "/quit")
    assert web_app.QUIT_NOTE in infos(app)
    assert app.session_state["thread_id"] == thread
    assert texts(app) == ["What is Ella like?", "Ella is green."]
    assert not any("Bye!" in text for text in infos(app))  # the terminal reply


def test_itinerary_without_a_plan_says_so(page):
    model = streaming()
    app = page(model)
    type_in_chat(app, "/itinerary")
    assert tool_calling.NO_ITINERARY in infos(app)
    assert model.calls == []


@pytest.mark.usefixtures("tools")
def test_itinerary_json_is_a_developer_view(page):
    app = page(streaming(gather(), submit(plan())))
    type_in_chat(app, "Plan 2 days in Ella.")

    type_in_chat(app, "/itinerary")
    assert web_app.JSON_DEV_ONLY in infos(app)
    assert not any('"days": 2' in code.value for code in app.code)


@pytest.mark.usefixtures("tools")
def test_itinerary_shows_the_json_in_dev_mode(page, monkeypatch):
    monkeypatch.setenv(web_app.DEV_ENV, "1")
    app = page(streaming(gather(), submit(plan())))
    type_in_chat(app, "Plan 2 days in Ella.")

    type_in_chat(app, "/itinerary")
    main_code = [c.value for c in app.main.code]
    assert any('"days": 2' in value for value in main_code)


def test_a_command_reply_is_shown_once(page):
    app = page(streaming())
    type_in_chat(app, "/prefs")
    assert "No saved preferences yet." in infos(app)
    app.run()  # any later rerun, e.g. a click elsewhere
    assert "No saved preferences yet." not in infos(app)


def test_a_command_while_a_question_waits_keeps_the_question(page):
    model = streaming()
    app = page(model)
    type_in_chat(app, "Plan me a trip.")
    calls = len(model.calls)

    type_in_chat(app, "/prefs")
    assert "No saved preferences yet." in infos(app)
    assert texts(app)[-1] == "Where in Sri Lanka would you like to go?"
    assert any("waiting for your answer" in text for text in infos(app))
    assert len(model.calls) == calls


# --- Public mode (TRAVEL_PUBLIC=1): provider fallback -----------------------------


def pool_settings():
    from ai_travel_assistant.provider_pool import PoolSettings

    return PoolSettings(
        members=(
            llm_basics.LLMSettings(model="flash", api_key="g-secret"),
            llm_basics.LLMSettings(
                model="@cf/zai-org/glm", api_key="c", provider="cloudflare"
            ),
            llm_basics.LLMSettings(model="oss", api_key="q", provider="groq"),
        )
    )


def test_local_mode_builds_the_agent_with_one_provider(page):
    page(streaming())
    assert [type(s).__name__ for s in built_with] == ["LLMSettings"]


def test_public_mode_builds_the_agent_with_the_provider_pool(page, monkeypatch):
    from ai_travel_assistant import provider_pool

    monkeypatch.setenv(web_app.PUBLIC_ENV, "1")
    monkeypatch.setattr(provider_pool, "load_pool_settings", pool_settings)
    app = page(streaming("Ella is green."))
    assert [type(s).__name__ for s in built_with] == ["PoolSettings"]
    header = " ".join(md.value for md in app.markdown)
    assert "Gemini · flash · fallback: Cloudflare, Groq" in header
    assert "g-secret" not in header
    app.chat_input[0].set_value("What is Ella like?").run()
    assert "Ella is green." in texts(app)


def test_public_mode_with_no_provider_configured_says_so(page, monkeypatch):
    from ai_travel_assistant import provider_pool

    def none_configured():
        raise llm_basics.ConfigError("No AI provider is configured.")

    monkeypatch.setenv(web_app.PUBLIC_ENV, "1")
    monkeypatch.setattr(provider_pool, "load_pool_settings", none_configured)
    import streamlit as st

    st.cache_resource.clear()
    app = AppTest.from_file(APP, default_timeout=30)
    app.run()
    assert not app.exception
    assert "No AI provider is configured." in app.error[0].value
    assert len(app.chat_input) == 0
