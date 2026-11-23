"""Stage 10: a web chat page for the agent, with Streamlit (optional extra).

Run from the project root:
    uv sync --extra ui
    uv run --extra ui streamlit run src/ai_travel_assistant/web_app.py

What this file demonstrates:
  1. The same events, a different screen. The page renders streaming.py's
     Step / Token / TurnDone events: steps go into a collapsible status box,
     tokens are typed into the reply, a finished itinerary replaces them.
     The agent and the event layer are exactly the ones the terminal uses.
  2. Streamlit's model: the whole script runs again on every interaction.
     Anything that must survive a rerun lives in st.session_state (the
     thread id, the last turn's steps) or in a cached resource (the graph,
     built once per server process with st.cache_resource).
  3. Shared persistence. Conversations and preferences use the same SQLite
     file as `--chat`, so a thread started here can be continued in the
     terminal with --thread, and the other way round.
  4. Structured output pays off in the UI: the latest itinerary is drawn as
     day cards from state["itinerary"] (see web_ui.py), not from its Markdown,
     and the same data becomes the downloadable PDF (itinerary_pdf.py).
     Set TRAVEL_DEV=1 to also see the raw itinerary JSON in the sidebar.
  5. Quick replies. While the agent waits for an answer, the page shows the
     asked field's QUICK_REPLIES as buttons. A click sends the label as the
     reply, which request_understanding reads without an LLM call.
  6. A picker that never calls the LLM. The weather card opens the 9
     provinces, a province opens its districts (sri_lanka_places.py), and only
     the chosen district is sent, as one plain question.
  7. Chat commands without the LLM. /new, /prefs, /forget, /itinerary and
     /quit typed in the chat box go to the terminal's chat_command, never to
     the agent. Only how a reply is shown is decided here: as a one-time
     notice, JSON only in dev mode, and /quit leaves the browser chat open.
"""

from __future__ import annotations

import atexit
import os
from contextlib import ExitStack
from pathlib import Path

import streamlit as st
from langgraph.graph.state import CompiledStateGraph

from ai_travel_assistant import web_ui
from ai_travel_assistant.long_term_memory import sqlite_store
from ai_travel_assistant.request_understanding import QUICK_REPLIES
from ai_travel_assistant.sri_lanka_places import PROVINCES, weather_question
from ai_travel_assistant.streaming import (
    Step,
    Token,
    TurnDone,
    stream_turn,
    visible_history,
)
from ai_travel_assistant.tool_calling import (
    DEFAULT_CHAT_DB,
    DEFAULT_USER,
    FORGET_COMMAND,
    ITINERARY_COMMAND,
    NO_ITINERARY,
    PREFS_COMMAND,
    build_chat_graph,
    chat_command,
    itinerary_json,
    new_thread_id,
    pending_question,
    preference_command,
    sqlite_checkpointer,
)

DB_ENV = "TRAVEL_CHAT_DB"  # another SQLite file, e.g. for a demo or a test
DEV_ENV = "TRAVEL_DEV"  # "1" shows developer-only views (the itinerary JSON)
PLACEHOLDER = "Plan 3 days in Ella; I like hiking and scenic places"
# Browser versions of the two command replies written for the terminal.
QUIT_NOTE = (
    "This conversation stays open in your browser and is saved. Close the tab "
    "when you are done, or type /new to start another one."
)
JSON_DEV_ONLY = (
    "The raw itinerary JSON is a developer view. Use “Download plan (PDF)” "
    "in the sidebar for your plan."
)


@st.cache_resource(show_spinner="Starting the agent…")
def get_graph(db_path: str) -> CompiledStateGraph:
    """Build the --chat agent once per server process.

    The SQLite connections stay open for the life of the process, like the
    terminal chat's `with` block. atexit holds a reference to the ExitStack:
    without one it is garbage-collected when this function returns, which
    closes the context-manager generators and with them the connections.
    """
    from ai_travel_assistant.llm_basics import load_settings

    settings = load_settings()  # ConfigError if .env is incomplete
    stack = ExitStack()
    atexit.register(stack.close)
    checkpointer = stack.enter_context(sqlite_checkpointer(Path(db_path)))
    store = stack.enter_context(sqlite_store(Path(db_path)))
    return build_chat_graph(settings, checkpointer, store)


def dev_mode() -> bool:
    """Developer views on? Off by default, so travellers see only the PDF."""
    return os.environ.get(DEV_ENV, "").strip().lower() in {"1", "true", "yes"}


def _thread_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def switch_thread(thread_id: str) -> None:
    """Make thread_id the active conversation; the old one stays saved."""
    state = st.session_state
    state.thread_id = thread_id
    state.last_steps = []
    state.picker = None


def run_command(graph: CompiledStateGraph, user_id: str, text: str) -> bool:
    """Handle a chat command without the LLM; True if the text was one.

    chat_command (shared with the terminal chats) decides what is a command
    and does the work. This only picks how its reply is shown: a notice that
    the next rerun draws once (see show_notice).
    """
    state = st.session_state
    result = chat_command(graph, state.thread_id, user_id, text)
    if result is None:
        return False
    is_itinerary = text.strip().lower() == ITINERARY_COMMAND
    if result.quit:  # the browser chat has nothing to close
        state.notice = ("info", QUIT_NOTE)
    elif is_itinerary and result.reply != NO_ITINERARY:
        state.notice = ("json", result.reply) if dev_mode() else ("info", JSON_DEV_ONLY)
    else:
        state.notice = ("info", result.reply)
    if result.thread_id:  # /new
        switch_thread(result.thread_id)
    return True


def show_notice() -> None:
    """Draw the last command's reply, once: pop() removes it for later reruns."""
    notice = st.session_state.pop("notice", None)
    if notice is None:
        return
    kind, text = notice
    if kind == "json":
        st.code(text, language="json")
    else:
        st.info(text)


def model_label() -> str | None:
    """'Cloudflare · glm-4.7-flash' for the header badge (no secrets)."""
    from ai_travel_assistant.llm_basics import ConfigError, load_settings

    try:
        settings = load_settings()
    except ConfigError:
        return None
    short = settings.model.rsplit("/", 1)[-1]
    return f"{settings.provider.capitalize()} · {short}"


def itinerary_markdown(itinerary: dict | None) -> str | None:
    """The Markdown the agent showed for this itinerary, to find its message."""
    if not itinerary:
        return None
    from pydantic import ValidationError

    from ai_travel_assistant.itinerary_output import render_itinerary_markdown
    from ai_travel_assistant.schemas import Itinerary

    try:
        return render_itinerary_markdown(Itinerary.model_validate(itinerary)).strip()
    except ValidationError:
        return None


def render_turn(graph: CompiledStateGraph, thread_id: str, text: str, user_id: str):
    """Stream one turn into an assistant message; return its steps."""
    steps: list[str] = []
    with st.chat_message("assistant", avatar=web_ui.ASSISTANT_AVATAR):
        status = st.status("Thinking…", expanded=False)
        reply = st.empty()
        typed = ""
        for event in stream_turn(graph, thread_id, text, user_id=user_id):
            if isinstance(event, Step):
                steps.append(event.text)
                status.write(event.text)
                status.update(label=event.text)
            elif isinstance(event, Token):
                typed += event.text
                reply.markdown(typed + " ▌")
            elif isinstance(event, TurnDone):
                status.update(
                    label=f"Done in {len(steps)} steps" if steps else "Done",
                    state="complete",
                )
                reply.markdown(event.answer)
    return steps


def sidebar(graph: CompiledStateGraph) -> str:
    """Conversation controls; returns the traveller's user id."""
    state = st.session_state
    with st.sidebar:
        st.markdown(f"## {web_ui.ASSISTANT_AVATAR} {web_ui.APP_NAME}")
        with st.container(border=True):
            st.markdown("### 💬 Conversation")
            user_id = st.text_input("Traveller", value=DEFAULT_USER) or DEFAULT_USER
            if st.button("✨ New conversation", use_container_width=True):
                state.thread_id = new_thread_id()
                state.last_steps = []
            st.caption(f"Thread `{state.thread_id}`")
            st.caption("Continue it in the terminal with `--chat --thread <id>`.")

        with st.container(border=True):
            st.markdown("### 🗺️ Itinerary")
            itinerary = graph.get_state(_thread_config(state.thread_id)).values.get(
                "itinerary"
            )
            if not itinerary:
                st.caption(NO_ITINERARY)
            else:
                web_ui.pdf_download_button(itinerary, key="pdf-sidebar")
                if dev_mode():  # raw data for debugging, hidden from travellers
                    with st.expander("Last itinerary as JSON (dev)"):
                        st.code(itinerary_json(graph, state.thread_id), language="json")

        with st.container(border=True):
            st.markdown("### ⭐ Saved preferences")
            st.caption(preference_command(graph.store, user_id, PREFS_COMMAND))
            if st.button("Forget my preferences", use_container_width=True):
                st.toast(preference_command(graph.store, user_id, FORGET_COMMAND))
    return user_id


def suggestions() -> str | None:
    """Starter cards for an empty conversation; returns a clicked prompt."""
    st.markdown("#### Where shall we go?")
    picked = None
    for row in range(0, len(web_ui.SUGGESTIONS), 2):
        for column, (title, subtitle, prompt) in zip(
            st.columns(2), web_ui.SUGGESTIONS[row : row + 2], strict=False
        ):
            label = f"**{title}**  \n{subtitle}"
            if column.button(label, key=f"suggest-{title}", use_container_width=True):
                if prompt == web_ui.WEATHER_PICKER:
                    st.session_state.picker = "province"
                    st.rerun()
                picked = prompt
    return picked


def weather_picker() -> str | None:
    """Province buttons, then that province's districts; returns the question.

    Kept in st.session_state between reruns: picker is "province" or the
    chosen province's name. No LLM call until a district is picked.
    """
    state = st.session_state
    columns_per_row = 3
    if state.picker == "province":
        st.markdown("#### 🌦️ Weather: pick a province")
        options = list(PROVINCES)
    else:
        st.markdown(f"#### 🌦️ Weather in {state.picker} Province: pick a district")
        options = list(PROVINCES[state.picker])
    columns = st.columns(columns_per_row)
    for i, name in enumerate(options):
        kind = "province" if state.picker == "province" else "district"
        if columns[i % columns_per_row].button(
            name, key=f"{kind}-{name}", use_container_width=True
        ):
            if kind == "province":
                state.picker = name
                st.rerun()
            state.picker = None
            return weather_question(name)
    back = "⬅️ All provinces" if state.picker != "province" else "⬅️ Back"
    if st.button(back, key="picker-back"):
        state.picker = "province" if state.picker != "province" else None
        st.rerun()
    return None


def quick_replies(field: str | None) -> str | None:
    """Answer buttons for the field the agent asked about; returns a clicked one."""
    options = QUICK_REPLIES.get(field or "", ())
    if not options:
        return None
    st.caption("Tap an answer, or type your own below.")
    columns = st.columns(min(len(options), 4))
    picked = None
    for i, (label, _) in enumerate(options):
        column = columns[i % len(columns)]
        if column.button(label, key=f"quick-{field}-{i}", use_container_width=True):
            picked = label
    return picked


def main() -> None:
    st.set_page_config(
        page_title=f"{web_ui.APP_NAME} · Sri Lanka travel planner", page_icon="🌴"
    )
    web_ui.apply_theme()
    st.markdown(web_ui.hero_html(model_label()), unsafe_allow_html=True)

    from ai_travel_assistant.llm_basics import ConfigError
    from ai_travel_assistant.robustness import describe_llm_error

    try:
        graph = get_graph(os.environ.get(DB_ENV, str(DEFAULT_CHAT_DB)))
    except ConfigError as exc:
        st.error(f"Configuration error: {exc}")
        st.stop()

    state = st.session_state
    state.setdefault("thread_id", new_thread_id())
    state.setdefault("last_steps", [])
    user_id = sidebar(graph)

    values = graph.get_state(_thread_config(state.thread_id)).values
    history = visible_history(values.get("messages", []))
    plan_text = itinerary_markdown(values.get("itinerary"))
    # Only the latest showing of the current plan becomes cards; earlier
    # versions (before an edit) stay as the text they were.
    card_at = max(
        (i for i, (role, text) in enumerate(history) if text.strip() == plan_text),
        default=None,
    )
    for i, (role, text) in enumerate(history):
        avatar = web_ui.USER_AVATAR if role == "user" else web_ui.ASSISTANT_AVATAR
        with st.chat_message(role, avatar=avatar):
            if i == card_at:
                web_ui.render_itinerary(values["itinerary"], text, key=str(i))
            else:
                st.markdown(text)
    if state.last_steps:
        label = f"How the last answer was made ({len(state.last_steps)} steps)"
        with st.expander(label):
            for line in state.last_steps:
                st.write(line)
    show_notice()  # the last chat command's reply, drawn once
    waiting = pending_question(graph, state.thread_id)
    if waiting:
        st.info("The assistant is waiting for your answer to its question.")

    state.setdefault("picker", None)
    if waiting:
        picked = quick_replies(values.get("clarification_field"))
    elif history:
        picked = None
    else:
        picked = weather_picker() if state.picker else suggestions()
    if prompt := st.chat_input(PLACEHOLDER) or picked:
        if run_command(graph, user_id, prompt):
            st.rerun()  # show the reply (and a /new thread) without the LLM
        with st.chat_message("user", avatar=web_ui.USER_AVATAR):
            st.markdown(prompt)
        try:
            state.last_steps = render_turn(graph, state.thread_id, prompt, user_id)
        except Exception as exc:  # keep the page alive after a failed turn
            st.error(f"Request failed: {describe_llm_error(exc)}")
        else:
            st.rerun()  # redraw from the saved thread (sidebar included)


if __name__ == "__main__":  # streamlit run executes the file as __main__
    main()
