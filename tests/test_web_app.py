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

from ai_travel_assistant import (  # noqa: E402
    llm_basics,
    public_sessions,
    tool_calling,
    usage_limits,
    web_app,
    web_ui,
)
from ai_travel_assistant.provider_pool import AllProvidersUnavailable  # noqa: E402
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
built_persistence: list = []  # (checkpointer, store) of each build


@pytest.fixture
def page(monkeypatch, tmp_path):
    """Returns a function: model -> a started AppTest on a fresh database."""
    import streamlit as st

    built_with.clear()
    built_persistence.clear()

    st.cache_resource.clear()
    usage_limits.reset_shared_guard()  # fresh counters for every page
    public_sessions.reset_shared_janitor()
    settings = llm_basics.GeminiSettings(model="fake-model", api_key="k")
    monkeypatch.setattr(llm_basics, "load_settings", lambda: settings)
    monkeypatch.setenv(web_app.DB_ENV, str(tmp_path / "chat.sqlite"))
    monkeypatch.delenv(web_app.PUBLIC_ENV, raising=False)  # local mode unless set

    def start(model):
        def fake_build(settings, checkpointer, store, **kwargs):
            built_with.append(settings)
            built_persistence.append((checkpointer, store))
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
    usage_limits.reset_shared_guard()
    public_sessions.reset_shared_janitor()


def page_and_errors(app) -> str:
    """Every text on the page, errors and the sidebar included."""
    parts = [m.value for m in app.markdown] + [c.value for c in app.caption]
    for kind in ("error", "warning", "info", "success", "exception"):
        parts += [str(getattr(e, "value", e)) for e in app.get(kind)]
    parts += [b.label for b in app.button] + [t.label for t in app.text_input]
    return "\n".join(str(p) for p in parts)


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


def test_public_mode_with_no_provider_configured_says_so(page, monkeypatch, caplog):
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
    # Change 6: plain words for the visitor; the detail goes to the server log.
    from ai_travel_assistant.robustness import VISITOR_NOT_CONFIGURED

    assert app.error[0].value == VISITOR_NOT_CONFIGURED
    assert "No AI provider is configured." not in page_and_errors(app)
    assert "No AI provider is configured." in caplog.text
    assert len(app.chat_input) == 0


# --- Public mode: usage limits (usage_limits.py) ----------------------------------


@pytest.fixture
def public_page(page, monkeypatch):
    """A public-mode page with small limits; returns (start, model calls)."""
    from ai_travel_assistant import provider_pool
    from ai_travel_assistant.usage_limits import LIMIT_VARIABLES

    for variable, _ in LIMIT_VARIABLES.values():
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.delenv(public_sessions.THREAD_IDLE_ENV, raising=False)
    monkeypatch.setenv(web_app.PUBLIC_ENV, "1")
    monkeypatch.setenv("TRAVEL_MAX_MESSAGES_PER_SESSION", "2")
    monkeypatch.setenv("TRAVEL_MAX_MESSAGE_CHARS", "40")
    monkeypatch.setattr(provider_pool, "load_pool_settings", pool_settings)
    return page


def warnings(app) -> list[str]:
    return [w.value for w in app.warning]


def test_public_mode_passes_the_limits_to_the_agent(public_page):
    public_page(streaming())
    [settings] = built_with
    guard = web_app.get_usage_guard()
    assert guard.limits.max_message_chars == 40
    assert guard.limits.max_history_tokens == 8000


def test_a_too_long_message_is_refused_without_an_llm_call(public_page):
    model = streaming("Never sent.")
    app = public_page(model)
    app.chat_input[0].set_value("x" * 41).run()
    assert not app.exception
    assert any("limit is 40" in w for w in warnings(app))
    assert model.calls == [] and texts(app) == []
    assert app.session_state["messages_sent"] == 0


def test_the_session_message_limit_stops_the_third_message(public_page):
    model = streaming("One.", "Two.", "Three.")
    app = public_page(model)
    app.chat_input[0].set_value("What is Ella like?").run()
    app.chat_input[0].set_value("What is Ella like?").run()
    calls = len(model.calls)
    app.chat_input[0].set_value("What is Ella like?").run()
    assert any("limit of 2 messages" in w for w in warnings(app))
    assert len(model.calls) == calls  # the third never reached the model
    assert "Three." not in texts(app)


def test_chat_commands_do_not_count_against_the_session_limit(public_page):
    app = public_page(streaming("One."))
    for _ in range(3):
        app.chat_input[0].set_value("/prefs").run()
    assert app.session_state["messages_sent"] == 0
    app.chat_input[0].set_value("What is Ella like?").run()
    assert "One." in texts(app)


def test_a_used_up_daily_budget_is_explained_without_a_call(public_page):
    from ai_travel_assistant.usage_limits import DAILY_LIMIT_MESSAGE

    model = streaming("Never sent.")
    app = public_page(model)
    budget = web_app.get_usage_guard().budget
    while budget.try_consume():
        pass
    app.chat_input[0].set_value("What is Ella like?").run()
    assert DAILY_LIMIT_MESSAGE in warnings(app)
    assert model.calls == []


def test_a_busy_server_asks_the_visitor_to_try_again(public_page):
    from ai_travel_assistant.usage_limits import BUSY_MESSAGE

    model = streaming("Never sent.")
    app = public_page(model)
    guard = web_app.get_usage_guard()
    guard.slot_wait_s = 0.01
    held = [guard.request_slot() for _ in range(guard.limits.max_concurrent_requests)]
    for slot in held:
        slot.__enter__()
    try:
        app.chat_input[0].set_value("What is Ella like?").run()
    finally:
        for slot in held:
            slot.__exit__(None, None, None)
    assert not app.exception
    assert BUSY_MESSAGE in warnings(app)
    assert model.calls == []
    app.chat_input[0].set_value("What is Ella like?").run()  # slots are free again
    assert "Never sent." in texts(app)


def test_local_mode_has_no_usage_limits(page, monkeypatch):
    monkeypatch.setenv("TRAVEL_MAX_MESSAGE_CHARS", "5")
    monkeypatch.setenv("TRAVEL_MAX_MESSAGES_PER_SESSION", "1")
    app = page(streaming("One.", "Two."))
    app.chat_input[0].set_value("What is Ella like?").run()
    app.chat_input[0].set_value("What is Ella like?").run()
    assert "Two." in texts(app)
    assert warnings(app) == []


def test_a_bad_limit_value_is_reported_as_a_configuration_error(
    public_page, monkeypatch, caplog
):
    import streamlit as st

    monkeypatch.setenv("TRAVEL_MAX_DAILY_LLM_CALLS", "lots")
    st.cache_resource.clear()
    app = AppTest.from_file(APP, default_timeout=30)
    app.run()
    assert not app.exception
    # Change 6: the setting's name is for the site owner (server log) only.
    from ai_travel_assistant.robustness import VISITOR_NOT_CONFIGURED

    assert app.error[0].value == VISITOR_NOT_CONFIGURED
    assert "TRAVEL_MAX_DAILY_LLM_CALLS" not in page_and_errors(app)
    assert "TRAVEL_MAX_DAILY_LLM_CALLS" in caplog.text


# --- Public mode: visitor sessions and privacy (public_sessions.py) ---------------


def all_page_text(app) -> str:
    """Every text a visitor can read, sidebar included."""
    parts = [m.value for m in app.markdown]
    parts += [c.value for c in app.caption]
    parts += [i.value for i in app.info] + [w.value for w in app.warning]
    parts += [t.label for t in app.text_input] + [b.label for b in app.button]
    return "\n".join(str(p) for p in parts)


def test_public_mode_keeps_conversations_in_memory_without_a_store(public_page):
    from langgraph.checkpoint.memory import InMemorySaver

    public_page(streaming())
    [(checkpointer, store)] = built_persistence
    assert isinstance(checkpointer, InMemorySaver)
    assert store is None  # no saved preferences, no preference extractor


def test_local_mode_keeps_sqlite_and_saved_preferences(page):
    page(streaming())
    [(checkpointer, store)] = built_persistence
    assert type(checkpointer).__name__ == "SqliteSaver"
    assert store is not None


def test_public_page_hides_the_traveller_field_thread_id_and_terminal_hint(
    public_page,
):
    app = public_page(streaming("One."))
    app.chat_input[0].set_value("What is Ella like?").run()
    text = all_page_text(app)
    assert app.text_input == []  # no Traveller field
    assert app.session_state["thread_id"] not in text
    assert "--thread" not in text and "terminal" not in text.lower()
    assert "Saved preferences" not in text
    assert "Forget my preferences" not in text
    assert "private to this browser tab" in text


def test_each_public_visitor_gets_random_ids(public_page):
    first = public_page(streaming())
    second = public_page(streaming())
    ids = [
        (app.session_state["visitor_id"], app.session_state["thread_id"])
        for app in (first, second)
    ]
    assert ids[0][0] != ids[1][0] and ids[0][1] != ids[1][1]
    assert all(v.startswith("visitor-") and v != "local" for v, _ in ids)


def test_two_public_visitors_never_see_each_others_conversation(public_page):
    model = streaming("Ella is green.", "Galle has a fort.")
    alice = public_page(model)
    bob = public_page(model)  # a second browser session, same server
    alice.chat_input[0].set_value("What is Ella like?").run()
    bob.chat_input[0].set_value("What is Ella like?").run()
    assert "Ella is green." in texts(alice)
    assert "Galle has a fort." not in texts(alice)
    assert texts(bob) == ["What is Ella like?", "Galle has a fort."]
    alice.run()  # a later rerun still shows only her own thread
    assert "Galle has a fort." not in texts(alice)
    assert len(built_persistence) == 1  # one server graph serves both visitors


@pytest.mark.parametrize("command", ["/prefs", "/forget"])
def test_preference_commands_are_off_in_public(public_page, command):
    model = streaming("Never sent.")
    app = public_page(model)
    app.chat_input[0].set_value(command).run()
    assert public_sessions.PREFERENCES_OFF in [i.value for i in app.info]
    assert model.calls == []


def test_new_in_public_hides_the_thread_id_and_deletes_the_old_thread(public_page):
    app = public_page(streaming("Ella is green."))
    app.chat_input[0].set_value("What is Ella like?").run()
    old = app.session_state["thread_id"]
    graph = app.session_state["_graph"]
    config = {"configurable": {"thread_id": old}}
    assert graph.get_state(config).values.get("messages")
    app.chat_input[0].set_value("/new").run()
    new = app.session_state["thread_id"]
    assert new != old
    assert public_sessions.NEW_CONVERSATION in [i.value for i in app.info]
    assert new not in all_page_text(app) and old not in all_page_text(app)
    assert not graph.get_state(config).values.get("messages")  # deleted
    assert texts(app) == []


def test_the_new_conversation_button_deletes_the_old_thread_in_public(public_page):
    app = public_page(streaming("Ella is green."))
    app.chat_input[0].set_value("What is Ella like?").run()
    old = app.session_state["thread_id"]
    graph = app.session_state["_graph"]
    app.sidebar.button[0].click().run()  # "New conversation"
    assert app.session_state["thread_id"] != old
    config = {"configurable": {"thread_id": old}}
    assert not graph.get_state(config).values.get("messages")


def test_one_visitor_starting_over_does_not_touch_another(public_page):
    model = streaming("Ella is green.", "Galle has a fort.")
    alice, bob = public_page(model), public_page(model)
    alice.chat_input[0].set_value("What is Ella like?").run()
    bob.chat_input[0].set_value("What is Ella like?").run()
    alice.chat_input[0].set_value("/new").run()
    bob.run()
    assert texts(bob) == ["What is Ella like?", "Galle has a fort."]
    assert len(built_persistence) == 1


def test_public_quit_says_nothing_is_kept(public_page):
    app = public_page(streaming())
    app.chat_input[0].set_value("/quit").run()
    assert public_sessions.PUBLIC_QUIT_NOTE in [i.value for i in app.info]


def test_a_public_turn_is_tracked_for_cleanup(public_page):
    app = public_page(streaming("Ella is green."))
    app.chat_input[0].set_value("What is Ella like?").run()
    assert public_sessions.shared_janitor().tracked() == 1


def test_dev_views_are_never_on_in_public(monkeypatch):
    monkeypatch.setenv(web_app.DEV_ENV, "1")
    monkeypatch.setenv(web_app.PUBLIC_ENV, "1")
    assert not web_app.dev_mode()
    monkeypatch.delenv(web_app.PUBLIC_ENV)
    assert web_app.dev_mode()


def test_a_bad_thread_idle_time_is_a_configuration_error(
    public_page, monkeypatch, caplog
):
    import streamlit as st

    monkeypatch.setenv(public_sessions.THREAD_IDLE_ENV, "soon")
    st.cache_resource.clear()
    app = AppTest.from_file(APP, default_timeout=30)
    app.run()
    assert not app.exception
    # Change 6: the setting's name is for the site owner (server log) only.
    from ai_travel_assistant.robustness import VISITOR_NOT_CONFIGURED

    assert app.error[0].value == VISITOR_NOT_CONFIGURED
    assert public_sessions.THREAD_IDLE_ENV not in page_and_errors(app)
    assert public_sessions.THREAD_IDLE_ENV in caplog.text


# --- Public mode: Sample Demo Mode (sample_trip.py) ---------------------------------


def sample_open(app) -> bool:
    from ai_travel_assistant.sample_trip import SAMPLE_LABEL

    return any(SAMPLE_LABEL in w.value for w in app.warning)


def click(app, key):
    [button] = [b for b in app.button if b.key == key]
    return button.click().run()


def test_a_public_visitor_can_open_the_sample_without_any_ai_call(public_page):
    model = streaming("Never sent.")
    app = public_page(model)
    assert {"sidebar-sample", "empty-sample"} <= {b.key for b in app.button}
    click(app, "empty-sample")
    assert not app.exception
    assert sample_open(app)
    assert any("Sample Demo Mode" in md.value for md in app.markdown)
    assert any("2-day trip to Kandy" in md.value for md in app.markdown)
    labels = [tab.label for tab in app.tabs]
    assert len(labels) == 2 and labels[0].startswith("Day 1")
    assert "pdf-sample" in [d.key for d in app.get("download_button")]
    assert len(app.chat_input) == 0  # no chat box: nothing can reach the AI
    assert model.calls == []
    assert web_app.get_usage_guard().budget.used() == 0
    assert app.session_state["messages_sent"] == 0


def test_the_sample_is_never_shown_as_a_chat_answer(public_page):
    app = public_page(streaming())
    click(app, "sidebar-sample")
    assert texts(app) == []  # not in the conversation history
    graph = app.session_state["_graph"]
    config = {"configurable": {"thread_id": app.session_state["thread_id"]}}
    assert not graph.get_state(config).values.get("itinerary")


def test_back_to_live_chat_still_works_after_the_sample(public_page):
    model = streaming("Ella is green.")
    app = public_page(model)
    click(app, "sidebar-sample")
    click(app, "close-sample")
    assert not sample_open(app)
    assert len(app.chat_input) == 1
    app.chat_input[0].set_value("What is Ella like?").run()
    assert "Ella is green." in texts(app)


def test_the_sample_is_offered_when_the_daily_budget_is_used_up(public_page):
    model = streaming("Never sent.")
    app = public_page(model)
    budget = web_app.get_usage_guard().budget
    while budget.try_consume():
        pass
    app.chat_input[0].set_value("What is Ella like?").run()
    assert web_app.SAMPLE_OFFER in [i.value for i in app.info]
    click(app, "offer-sample")
    assert sample_open(app)
    assert model.calls == []


def test_the_sample_is_offered_after_the_session_limit(public_page):
    app = public_page(streaming("One.", "Two."))
    for _ in range(3):
        app.chat_input[0].set_value("What is Ella like?").run()
    assert web_app.SAMPLE_OFFER in [i.value for i in app.info]


def test_the_sample_is_offered_when_the_ai_fails(public_page):
    from ai_travel_assistant.provider_pool import AllProvidersUnavailable

    failure = AllProvidersUnavailable([("gemini", "quota"), ("groq", "quota")])
    app = public_page(streaming(fail_on={0: failure}))
    app.chat_input[0].set_value("What is Ella like?").run()
    assert app.error and web_app.SAMPLE_OFFER in [i.value for i in app.info]
    click(app, "offer-sample")
    assert sample_open(app)


def test_a_too_long_message_does_not_offer_the_sample(public_page):
    app = public_page(streaming())
    app.chat_input[0].set_value("x" * 41).run()
    assert web_app.SAMPLE_OFFER not in [i.value for i in app.info]


def test_a_successful_turn_clears_the_offer(public_page):
    from ai_travel_assistant.provider_pool import AllProvidersUnavailable

    failure = AllProvidersUnavailable([("gemini", "busy")])
    app = public_page(streaming("Ella is green.", fail_on={0: failure}))
    app.chat_input[0].set_value("What is Ella like?").run()
    assert app.session_state["offer_sample"]
    app.chat_input[0].set_value("What is Ella like?").run()
    assert not app.session_state["offer_sample"]
    assert web_app.SAMPLE_OFFER not in [i.value for i in app.info]


def test_with_no_provider_configured_the_sample_still_works(public_page, monkeypatch):
    import streamlit as st

    from ai_travel_assistant import provider_pool

    def none_configured():
        raise llm_basics.ConfigError("No AI provider is configured.")

    monkeypatch.setattr(provider_pool, "load_pool_settings", none_configured)
    st.cache_resource.clear()
    app = AppTest.from_file(APP, default_timeout=30)
    app.run()
    assert web_app.SAMPLE_OFFER in [i.value for i in app.info]
    click(app, "offer-sample")
    assert not app.exception
    assert sample_open(app)
    assert len(app.chat_input) == 0
    assert "pdf-sample" in [d.key for d in app.get("download_button")]


def test_local_mode_has_no_sample_buttons(page):
    app = page(streaming())
    keys = {b.key for b in app.button}
    assert not keys & {"sidebar-sample", "empty-sample", "offer-sample"}


# --- Public mode: visitor-friendly errors (robustness.describe_error_for_visitor) ---


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            "quota",
            "This demo has used up today's free AI allowance.",
        ),
        ("rate_limit", "The free AI service is getting too many requests"),
        ("busy", "The AI service is busy or slow"),
        ("rejected", "Live AI is temporarily unavailable on this demo."),
    ],
)
def test_a_failed_public_turn_shows_a_plain_message_and_the_sample(
    public_page, caplog, error, expected
):
    from tests.test_visitor_errors import FORBIDDEN

    failure = AllProvidersUnavailable([("gemini", error)], 30)
    failure.__cause__ = RuntimeError(
        "key=AIzaSyFAKE-KEY-123 GEMINI_API_KEY LLM_PROVIDER .env Traceback"
    )
    app = public_page(streaming(fail_on={0: failure}))
    app.chat_input[0].set_value("What is Ella like?").run()
    assert not app.exception
    [shown] = app.error
    assert shown.value.startswith(expected)
    assert web_app.SAMPLE_OFFER in [i.value for i in app.info]
    page = page_and_errors(app)
    for forbidden in FORBIDDEN:
        assert forbidden not in page
        assert forbidden not in caplog.text  # the server log stays clean too
    assert "Public turn failed" in caplog.text


def test_an_unexpected_turn_error_shows_no_type_name_or_text(public_page, caplog):
    from ai_travel_assistant.robustness import VISITOR_UNEXPECTED

    app = public_page(streaming(fail_on={0: RuntimeError("secret detail /srv/x.py")}))
    app.chat_input[0].set_value("What is Ella like?").run()
    assert app.error[0].value == VISITOR_UNEXPECTED
    page = page_and_errors(app)
    assert "RuntimeError" not in page and "secret detail" not in page
    assert "secret detail" not in caplog.text


def test_an_unexpected_page_error_never_shows_a_traceback_in_public(
    public_page, monkeypatch
):
    from ai_travel_assistant.robustness import VISITOR_UNEXPECTED

    app = public_page(streaming())

    def broken(*args):
        raise RuntimeError("boom at /srv/app/web_ui.py")

    monkeypatch.setattr(web_ui, "hero_html", broken)
    app.run()
    assert not app.exception  # no Streamlit traceback box
    assert [e.value for e in app.error] == [VISITOR_UNEXPECTED]
    assert "web_ui.py" not in page_and_errors(app)


def test_local_mode_still_shows_its_errors_to_the_developer(page, monkeypatch):
    app = page(streaming())

    def broken(*args):
        raise RuntimeError("boom")

    monkeypatch.setattr(web_ui, "hero_html", broken)
    app.run()
    assert app.exception  # unchanged: the developer sees the real error


def test_a_blank_public_message_is_refused_without_an_llm_call(public_page):
    model = streaming("Never sent.")
    app = public_page(model)
    app.chat_input[0].set_value("   ").run()
    assert web_app.BLANK_MESSAGE in [w.value for w in app.warning]
    assert model.calls == []
    assert app.session_state["messages_sent"] == 0


def test_a_public_turn_says_it_may_take_a_while(public_page):
    failure = AllProvidersUnavailable([("gemini", "busy")])
    app = public_page(streaming(fail_on={0: failure}))
    app.chat_input[0].set_value("What is Ella like?").run()
    assert web_app.SLOW_NOTE in [c.value for c in app.caption]


def test_local_turns_have_no_slow_note(page):
    app = page(streaming(fail_on={0: RuntimeError("x")}))
    app.chat_input[0].set_value("What is Ella like?").run()
    assert web_app.SLOW_NOTE not in [c.value for c in app.caption]


# --- Public mode: message length checked by the app, never cut (Finding B) ---------


def sent_to_model(model) -> list[str]:
    """The traveller texts the model received, in full."""
    from langchain_core.messages import HumanMessage

    return [
        m.content for call in model.calls for m in call if isinstance(m, HumanMessage)
    ]


def test_the_public_chat_box_has_no_max_chars(public_page):
    """Streamlit 1.67 cuts input to max_chars on the server; the app checks instead."""
    app = public_page(streaming())
    assert app.chat_input[0].proto.max_chars == 0  # 0: no limit on the widget


@pytest.mark.parametrize("length", [39, 40])
def test_a_message_up_to_the_limit_is_sent_in_full(public_page, length):
    model = streaming("Noted.")
    app = public_page(model)  # limit 40 in this fixture
    text = "a" * (length - 1) + "!"
    app.chat_input[0].set_value(text).run()
    assert not app.exception
    assert text in sent_to_model(model)  # whole, not shortened
    assert app.session_state["messages_sent"] == 1
    assert not any("limit is" in w.value for w in app.warning)


def test_a_message_over_the_limit_is_refused_not_cut(public_page):
    model = streaming("Never sent.")
    app = public_page(model)
    app.chat_input[0].set_value("a" * 40 + "!").run()  # 41 characters
    [warning] = [w.value for w in app.warning]
    assert "41 characters" in warning and "limit is 40" in warning
    assert model.calls == []  # no LLM call, not even with a shortened text
    assert texts(app) == []
    assert app.session_state["messages_sent"] == 0
    assert web_app.SAMPLE_OFFER not in [i.value for i in app.info]


@pytest.mark.parametrize(("length", "sent"), [(999, True), (1000, True), (1001, False)])
def test_the_default_limit_is_1000_characters(public_page, monkeypatch, length, sent):
    monkeypatch.delenv("TRAVEL_MAX_MESSAGE_CHARS")  # back to the default
    model = streaming("Noted.")
    app = public_page(model)
    text = "b" * (length - 1) + "?"
    app.chat_input[0].set_value(text).run()
    assert (text in sent_to_model(model)) is sent
    assert app.session_state["messages_sent"] == (1 if sent else 0)
    refused = any(f"{length:,} characters" in w.value for w in app.warning)
    assert refused is not sent
    if not sent:
        assert model.calls == []


def test_refused_long_messages_do_not_use_up_the_session_limit(public_page):
    model = streaming("One.", "Two.")
    app = public_page(model)  # 2 messages per session, 40 characters
    for _ in range(3):
        app.chat_input[0].set_value("c" * 60).run()
    assert app.session_state["messages_sent"] == 0
    app.chat_input[0].set_value("What is Ella like?").run()
    app.chat_input[0].set_value("What is Ella like?").run()
    assert app.session_state["messages_sent"] == 2
    assert "Two." in texts(app)


def test_a_long_blank_message_is_still_the_blank_message(public_page):
    model = streaming("Never sent.")
    app = public_page(model)
    app.chat_input[0].set_value(" " * 500).run()
    assert [w.value for w in app.warning] == [web_app.BLANK_MESSAGE]
    assert model.calls == []


def test_local_mode_has_no_length_limit(page):
    model = streaming("Noted.")
    app = page(model)
    text = "d" * 1500
    app.chat_input[0].set_value(text).run()
    assert text in sent_to_model(model)
    assert app.warning == []
