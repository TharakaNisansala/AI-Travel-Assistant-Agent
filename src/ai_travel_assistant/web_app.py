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
"""

from __future__ import annotations

import os
from contextlib import ExitStack
from pathlib import Path

import streamlit as st
from langgraph.graph.state import CompiledStateGraph

from ai_travel_assistant.long_term_memory import sqlite_store
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
    NO_ITINERARY,
    PREFS_COMMAND,
    build_chat_graph,
    itinerary_json,
    new_thread_id,
    pending_question,
    preference_command,
    sqlite_checkpointer,
)

DB_ENV = "TRAVEL_CHAT_DB"  # another SQLite file, e.g. for a demo or a test
PLACEHOLDER = "Plan 3 days in Ella; I like hiking and scenic places"


@st.cache_resource(show_spinner="Starting the agent…")
def get_graph(db_path: str) -> CompiledStateGraph:
    """Build the --chat agent once per server process.

    The SQLite connections stay open for the life of the process (the
    ExitStack is never closed), like the terminal chat's `with` block.
    """
    from ai_travel_assistant.llm_basics import load_settings

    settings = load_settings()  # ConfigError if .env is incomplete
    stack = ExitStack()
    checkpointer = stack.enter_context(sqlite_checkpointer(Path(db_path)))
    store = stack.enter_context(sqlite_store(Path(db_path)))
    return build_chat_graph(settings, checkpointer, store)


def _thread_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def render_turn(graph: CompiledStateGraph, thread_id: str, text: str, user_id: str):
    """Stream one turn into an assistant message; return its steps."""
    steps: list[str] = []
    with st.chat_message("assistant"):
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
        st.header("Conversation")
        user_id = st.text_input("Traveller", value=DEFAULT_USER) or DEFAULT_USER
        if st.button("New conversation", use_container_width=True):
            state.thread_id = new_thread_id()
            state.last_steps = []
        st.caption(f"Thread `{state.thread_id}`")
        st.caption("Continue it in the terminal with `--chat --thread <id>`.")

        st.header("Itinerary")
        plan = itinerary_json(graph, state.thread_id)
        if plan == NO_ITINERARY:
            st.caption(plan)
        else:
            with st.expander("Last itinerary as JSON"):
                st.code(plan, language="json")

        st.header("Saved preferences")
        st.caption(preference_command(graph.store, user_id, PREFS_COMMAND))
        if st.button("Forget my preferences", use_container_width=True):
            st.toast(preference_command(graph.store, user_id, FORGET_COMMAND))
    return user_id


def main() -> None:
    st.set_page_config(page_title="Sri Lanka travel assistant", page_icon="🌴")
    st.title("🌴 Sri Lanka travel assistant")
    st.caption(
        "Plans are built from live weather, OpenStreetMap, Wikivoyage and road "
        "routing, and checked against them. Each message uses Gemini calls."
    )

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

    messages = graph.get_state(_thread_config(state.thread_id)).values.get(
        "messages", []
    )
    for role, text in visible_history(messages):
        with st.chat_message(role):
            st.markdown(text)
    if state.last_steps:
        with st.expander(f"How the last answer was made ({len(state.last_steps)} steps)"):
            for line in state.last_steps:
                st.write(line)
    if pending_question(graph, state.thread_id):
        st.info("The assistant is waiting for your answer to its question.")

    if prompt := st.chat_input(PLACEHOLDER):
        with st.chat_message("user"):
            st.markdown(prompt)
        try:
            state.last_steps = render_turn(graph, state.thread_id, prompt, user_id)
        except Exception as exc:  # keep the page alive after a failed turn
            st.error(f"Request failed: {describe_llm_error(exc)}")
        else:
            st.rerun()  # redraw from the saved thread (sidebar included)


if __name__ == "__main__":  # streamlit run executes the file as __main__
    main()
