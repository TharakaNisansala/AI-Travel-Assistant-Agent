"""Stage 3: a tool-calling agent (the ReAct loop) with LangGraph and Gemini.

What this file demonstrates:
  1. bind_tools(): telling the LLM which tools exist.
  2. MessagesState: a state that is just the growing list of chat messages.
  3. An LLM node that either answers or asks for a tool to be run (a tool call).
  4. ToolNode: runs the requested tools and adds their results as ToolMessages.
  5. tools_condition: routes to the tools if the LLM asked for one, else ends.
  6. The loop tools -> llm, so the LLM sees the tool result and can answer.

Flow:

    START -> llm --(AIMessage has tool_calls?)--> tools -> llm   (loop)
                 \\--(no tool_calls: final answer)--> END

Reason (llm decides) -> Act (tools runs) -> Observe (llm reads the result) ...
until the LLM replies without a tool call.

The model is passed in, so the same graph runs with Gemini in the app and with a
scripted fake model in the offline tests.

Run from the project root (usually 2 Gemini calls: one to choose the tool, one
to write the answer):
    uv run python -m ai_travel_assistant.tool_calling "Find places for hiking"
    uv run python -m ai_travel_assistant.tool_calling --steps "Where can I surf?"
    uv run python -m ai_travel_assistant.tool_calling --steps \\
        "What is the weather forecast for Ella?"
    uv run python -m ai_travel_assistant.tool_calling --steps \\
        "What nature spots can I see around Ella?"

Stage 4 adds get_weather_forecast, which calls the real Open-Meteo API (2 HTTP
requests, no API key). The graph does not change: a new tool only needs to be
added to TOOLS, and Gemini decides from its docstring when to call it.
find_attractions (OpenStreetMap via Overpass, 2 HTTP requests) is added the
same way. With three tools, SYSTEM_PROMPT says how they relate: search to choose
a place, attractions for what to see at a named place, weather for forecasts.

The rest of the Stage 4 suite is registered the same way:
get_destination_guide (Wikivoyage background), estimate_travel_time (OSRM road
distance and driving time) and convert_currency (exchange rates). Six tools now
overlap in places (guide "see_and_do" vs find_attractions), so SYSTEM_PROMPT
spells out which tool answers which kind of question.

For multi-day itineraries, SYSTEM_PROMPT adds a generic plan: gather guide,
attractions, weather and travel time in one step; optionally a few travel
times and currency conversions in a second; then write the plan day by day
from tool results only, with sources, a stated default origin (DEFAULT_ORIGIN)
and a "Not available" list. Usually 3-4 Gemini calls per itinerary.

Stage 5 adds short-term memory. build_tool_agent_graph(model, checkpointer=...)
compiles the same loop with a checkpointer, which saves the message history
after every step under a thread_id. ask_agent() sends one message on a thread:
LangGraph loads that thread's saved history first and the add_messages reducer
appends the new message, so a follow-up ("make day 2 easier") sees the earlier
plan. Without a checkpointer every run starts a fresh conversation, as before.
run_agent_chat() is the terminal loop around ask_agent, with /new for a fresh
thread (each message makes the usual 2-4 Gemini calls):
    uv run python -m ai_travel_assistant.tool_calling --chat
    uv run python -m ai_travel_assistant.tool_calling --chat --steps
The chat saves its checkpoints in a SQLite file (DEFAULT_CHAT_DB), so a
conversation survives a restart; reopen it with the id printed at the start:
    uv run python -m ai_travel_assistant.tool_calling --chat --thread <id>
The chat also has long-term memory (long_term_memory.py): load_memory reads the
traveller's saved preferences before the LLM, and save_memory stores new ones
after the answer, but only spends a Gemini call when a message looks like a
preference ("I'm vegetarian"). /prefs and /forget show or clear them; --user
picks the traveller:
    uv run python -m ai_travel_assistant.tool_calling --chat --user tharaka
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore

from ai_travel_assistant.context_window import MAX_HISTORY_TOKENS, history_for_llm
from ai_travel_assistant.long_term_memory import (
    PreferenceExtractor,
    clear_preferences,
    load_preferences,
    looks_like_preference,
    make_llm_preference_extractor,
    save_preferences,
    sqlite_store,
)
from ai_travel_assistant.tools import (
    convert_currency,
    estimate_travel_time,
    find_attractions,
    get_destination_guide,
    get_weather_forecast,
    search_destinations,
)

# The tools this agent may use. bind_tools() sends all of them to the LLM, and
# ToolNode can run any of them (and only them).
TOOLS: tuple[BaseTool, ...] = (
    search_destinations,
    get_weather_forecast,
    find_attractions,
    get_destination_guide,
    estimate_travel_time,
    convert_currency,
)

# Node names. "tools" is the name tools_condition returns, so the tool node must
# be called exactly that (or we would need a mapping in add_conditional_edges).
LLM = "llm"
TOOLS_NODE = "tools"

# Where a trip starts when the traveller does not say: the main entry point to
# Sri Lanka (international airport and rail hub). The agent must say it assumed
# this. It is the only place name in SYSTEM_PROMPT; destinations never appear.
DEFAULT_ORIGIN = "Colombo"

# Each tool's docstring says what that tool does; this prompt says how the
# tools relate, so the LLM can choose between overlapping ones.
SYSTEM_PROMPT = (
    "You are a friendly Sri Lanka travel assistant. Choose tools like this. "
    # Choosing WHERE to go.
    "When the traveller has not chosen a place yet and asks where to go or which "
    "places suit an activity, call search_destinations. Recommend only places it "
    "returned, and mention that they come from the built-in guide. If it finds "
    "nothing, say so and suggest a broader search. "
    "If they also want to know what to see, call find_attractions for the one "
    "place that fits best, not for every result. "
    # WHAT to see at a named place.
    "When the traveller names a place and asks what to see or do there, call "
    "find_attractions with just the place name and a category matching their "
    "interests: nature for waterfalls, peaks or viewpoints; culture for temples, "
    "museums or history; landmarks for well-known sights; all when unsure or "
    "mixed. Mention only attractions it returned, and note that distances are "
    "straight-line. If it finds nothing for a specific category, you may retry "
    "once with category all. It does not cover hotels, restaurants or hiking "
    "routes; say so instead of inventing them. "
    # Weather.
    "When the traveller asks about the weather, rain, temperature or what to "
    "pack, call get_weather_forecast with just the place name and base your "
    "answer only on the forecast it returns, using its units. "
    # Background from the travel guide.
    "When the traveller asks what a named place is like, how to get there or "
    "around, or for practical or safety tips, call get_destination_guide with "
    "the place name and the matching topic (overview, getting_there, "
    "getting_around, see_and_do, safety; all when planning a whole visit). "
    "Mention that the guide is from Wikivoyage and may be out of date. For a "
    "list of nearby sights with distances, prefer find_attractions. "
    # Distances between places.
    "When the traveller asks how far apart two places are or how long the "
    "journey takes, call estimate_travel_time with the two place names, and "
    "say the time is a car estimate without traffic. "
    # Money.
    "When the traveller asks about exchange rates or what an amount is worth "
    "in another currency, call convert_currency with ISO codes such as USD or "
    "LKR, and quote its result; never use a remembered rate or do the "
    "conversion yourself. "
    # Combining tools and handling failures.
    "When planning a visit to a named place, you may call several of these "
    "tools in the same step. If a tool says its data is "
    "unavailable, do not call it again; tell the traveller plainly and never "
    "guess the weather or invent places. "
    # Multi-day itineraries. Generic on purpose: no place names, so the same
    # rule works for every destination. Colombo is only the default ORIGIN.
    "When the traveller asks for an itinerary or a plan of one or more days "
    "at a named place, first gather real data in one step: "
    "get_destination_guide (topic all), find_attractions (category all) and "
    "get_weather_forecast for that place, and estimate_travel_time from the "
    "traveller's starting point to it. If they did not say where they start, "
    f"use {DEFAULT_ORIGIN} and state in the plan that you assumed it. "
    "In the next step you may call estimate_travel_time for at most two "
    "nearby places the guide suggests visiting from there, and "
    "convert_currency for at most three prices the guide quotes, so the plan "
    "shows each in both LKR and USD. "
    "Then write the plan day by day, using only sights, travel times, "
    "weather and prices from the tool results, and say where each comes "
    "from (OpenStreetMap, Wikivoyage, OSRM, Open-Meteo or the exchange rate "
    "service). If find_attractions returns a Wikivoyage fallback, use only "
    "the sights it names. Give weather only for the days the forecast "
    "covers and say no forecast is available for the others. End with a "
    "short 'Not available' list of what the tools could not provide, such "
    "as hotel or food prices, travel times between sights inside a town, or "
    "any tool that failed. "
    "Keep answers short; an itinerary may be longer but stays to the point."
)


# --- State and context ----------------------------------------------------------

# Who is chatting when nobody says: one local traveller on this machine.
DEFAULT_USER = "local"

# Node names for long-term memory (only added when the graph gets a Store).
LOAD_MEMORY = "load_memory"
SAVE_MEMORY = "save_memory"


class AgentState(MessagesState):
    """The messages, plus the traveller's saved preferences for this turn."""

    user_prefs: list[str]


@dataclass
class AgentContext:
    """Fixed for a whole run, so it is runtime CONTEXT, not state.

    Passed as graph.invoke(..., context=AgentContext(user_id=...)); nodes read
    it as runtime.context. In a real app this would come from the login.
    """

    user_id: str = DEFAULT_USER


def system_prompt_for(prefs: Sequence[str]) -> str:
    """SYSTEM_PROMPT, plus the traveller's saved preferences if there are any."""
    if not prefs:
        return SYSTEM_PROMPT
    return (
        f"{SYSTEM_PROMPT} Known traveller preferences from earlier "
        f"conversations: {'; '.join(prefs)}. Respect them in every plan, and "
        "mention briefly when one changes a suggestion."
    )


def _user_id(runtime: Runtime[AgentContext]) -> str:
    return runtime.context.user_id if runtime.context else DEFAULT_USER


# --- Nodes --------------------------------------------------------------------


def make_llm_node(
    llm_with_tools: BaseChatModel,
    max_history_tokens: int | None = MAX_HISTORY_TOKENS,
) -> Callable[[AgentState], dict]:
    """Build the llm node around a model that already has the tools bound."""

    def llm_node(state: AgentState) -> dict:
        # The system prompt is added on every call but never stored in the
        # state, so the message history holds only the real conversation.
        prompt = system_prompt_for(state.get("user_prefs", []))
        # Only what the LLM sees is trimmed; the state keeps every message.
        history = history_for_llm(state["messages"], max_history_tokens)
        messages = [SystemMessage(prompt), *history]
        response = llm_with_tools.invoke(messages)
        # MessagesState uses the add_messages reducer: returning a list APPENDS
        # it to the history instead of replacing the history.
        return {"messages": [response]}

    return llm_node


def load_memory_node(state: AgentState, runtime: Runtime[AgentContext]) -> dict:
    """Read the traveller's preferences from the Store (no LLM call)."""
    return {"user_prefs": load_preferences(runtime.store, _user_id(runtime))}


def make_save_memory_node(
    extractor: PreferenceExtractor,
) -> Callable[[AgentState, Runtime[AgentContext]], dict]:
    """Build the node that stores new durable preferences after a turn."""

    def save_memory_node(state: AgentState, runtime: Runtime[AgentContext]) -> dict:
        from ai_travel_assistant.llm_basics import content_to_text

        last_human = next(
            (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            None,
        )
        if last_human is None:
            return {}
        text = content_to_text(last_human.content)
        # The gate: most messages are trip requests, and those cost no call.
        if not looks_like_preference(text):
            return {}
        known = state.get("user_prefs", [])
        try:
            found = extractor(text, known)
        except Exception:  # e.g. a Gemini 429: keep the answer, skip saving
            return {}
        added = save_preferences(runtime.store, _user_id(runtime), found)
        return {"user_prefs": [*known, *added]} if added else {}

    return save_memory_node


# --- Graph --------------------------------------------------------------------


def build_tool_agent_graph(
    model: BaseChatModel,
    tools: Sequence[BaseTool] = TOOLS,
    checkpointer: BaseCheckpointSaver | None = None,
    store: BaseStore | None = None,
    preference_extractor: PreferenceExtractor | None = None,
    max_history_tokens: int | None = MAX_HISTORY_TOKENS,
) -> CompiledStateGraph:
    """Bind the tools to the model and wire the ReAct loop.

    bind_tools() does not call the API. It returns a wrapped model that sends
    the tools' names, descriptions and argument schemas with every request.

    With a checkpointer, every invoke must name a thread_id (see ask_agent),
    and each thread keeps its own message history between invokes.

    With a store, the graph gets long-term memory:
        START -> load_memory -> llm <-> tools -> save_memory -> END
    load_memory reads the traveller's preferences; save_memory (added only
    with a preference_extractor) stores new ones after the final answer.

    Every LLM call sees the current turn whole, plus earlier turns compacted
    and trimmed to max_history_tokens (None: send everything).
    """
    llm_with_tools = model.bind_tools(list(tools))

    builder = StateGraph(AgentState, context_schema=AgentContext)
    builder.add_node(LLM, make_llm_node(llm_with_tools, max_history_tokens))
    # ToolNode looks at the last AIMessage, runs every tool call in it (several
    # calls run in parallel), and returns one ToolMessage per call.
    builder.add_node(TOOLS_NODE, ToolNode(list(tools)))

    if store is None:
        builder.add_edge(START, LLM)
    else:
        builder.add_node(LOAD_MEMORY, load_memory_node)
        builder.add_edge(START, LOAD_MEMORY)
        builder.add_edge(LOAD_MEMORY, LLM)

    # tools_condition returns "tools" if the last message has tool_calls,
    # otherwise END ("__end__"). With save_memory, "done" goes there first.
    after_answer = END
    if store is not None and preference_extractor is not None:
        builder.add_node(SAVE_MEMORY, make_save_memory_node(preference_extractor))
        builder.add_edge(SAVE_MEMORY, END)
        after_answer = SAVE_MEMORY
    builder.add_conditional_edges(
        LLM, tools_condition, {TOOLS_NODE: TOOLS_NODE, END: after_answer}
    )
    # The loop: after a tool runs, the LLM reads its result and decides again.
    builder.add_edge(TOOLS_NODE, LLM)
    return builder.compile(checkpointer=checkpointer, store=store)


def run_tool_agent(graph: CompiledStateGraph, question: str) -> list[AnyMessage]:
    """Ask one question and return the full message history of the run."""
    state = graph.invoke({"messages": [HumanMessage(question)]})
    return state["messages"]


def ask_agent(
    graph: CompiledStateGraph,
    thread_id: str,
    text: str,
    user_id: str = DEFAULT_USER,
) -> list[AnyMessage]:
    """Send one message on a thread and return only this turn's messages.

    The graph must be built with a checkpointer. The saved history of the
    thread is loaded first, so the LLM sees the earlier turns; the returned
    list starts at the new HumanMessage (tool calls, results, final answer).
    user_id picks whose long-term preferences are loaded and saved.
    """
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    before = len(graph.get_state(config).values.get("messages", []))
    state = graph.invoke(
        {"messages": [HumanMessage(text)]},
        config,
        context=AgentContext(user_id=user_id),
    )
    return state["messages"][before:]


def final_answer(messages: Sequence[AnyMessage]) -> str:
    """The text of the last message: the LLM's final reply."""
    # Imported here to keep the module light; content_to_text handles both plain
    # strings and Gemini 3's list-of-blocks content.
    from ai_travel_assistant.llm_basics import content_to_text

    return content_to_text(messages[-1].content).strip()


def describe_steps(messages: Sequence[AnyMessage]) -> list[str]:
    """One readable line per message, to show the Reason/Act/Observe steps."""
    from ai_travel_assistant.llm_basics import content_to_text

    lines = []
    for message in messages:
        if isinstance(message, HumanMessage):
            lines.append(f"[human] {content_to_text(message.content)}")
        elif isinstance(message, AIMessage) and message.tool_calls:
            for call in message.tool_calls:
                lines.append(f"[llm -> tool call] {call['name']}({call['args']})")
        elif isinstance(message, AIMessage):
            lines.append("[llm -> final answer]")
        else:  # ToolMessage
            first_line = content_to_text(message.content).splitlines()[0]
            lines.append(f"[tool result] {first_line} ...")
    return lines


# --- Multi-turn chat ----------------------------------------------------------

NEW_COMMAND = "/new"
QUIT_COMMANDS = frozenset({"/quit", "/exit"})
PREFS_COMMAND = "/prefs"
FORGET_COMMAND = "/forget"


# Where --chat keeps its conversations. Git-ignored: it holds real chat text.
DEFAULT_CHAT_DB = Path(".data/chat.sqlite")


def new_thread_id() -> str:
    """A random, unique id for a new conversation."""
    return uuid.uuid4().hex


@contextmanager
def sqlite_checkpointer(path: Path) -> Iterator[SqliteSaver]:
    """Open a SQLite checkpointer on a file, and close the file afterwards.

    Same checkpoints as InMemorySaver, but written to disk: a new process that
    opens the same file and uses the same thread_id gets the old messages.
    check_same_thread=False because LangGraph may run nodes on worker threads.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    try:
        yield SqliteSaver(conn)
    finally:
        conn.close()


def saved_message_count(graph: CompiledStateGraph, thread_id: str) -> int:
    """How many messages the checkpointer holds for this thread (0 if new)."""
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    return len(graph.get_state(config).values.get("messages", []))


def preference_command(store: BaseStore | None, user_id: str, command: str) -> str:
    """Answer /prefs or /forget for this traveller."""
    if store is None:
        return "Long-term memory is off for this chat."
    if command == FORGET_COMMAND:
        return f"Forgot {clear_preferences(store, user_id)} saved preferences."
    prefs = load_preferences(store, user_id)
    if not prefs:
        return "No saved preferences yet."
    return "Saved preferences: " + "; ".join(prefs)


def run_agent_chat(
    graph: CompiledStateGraph,
    read: Callable[[str], str] = input,
    write: Callable[[str], None] = print,
    *,
    thread_id: str | None = None,
    user_id: str = DEFAULT_USER,
    show_steps: bool = False,
    describe_error: Callable[[Exception], str] = lambda exc: type(exc).__name__,
) -> None:
    """Chat with the agent in the terminal, one thread at a time.

    The graph must be built with a checkpointer. Every message goes to the
    current thread, so follow-ups see the earlier turns. Pass thread_id to
    continue a saved conversation; otherwise a new one starts. /new starts a
    fresh thread (the old one stays saved), /quit or /exit ends the chat. A
    failed turn (e.g. a Gemini 503) is reported and the chat goes on.

    If the graph has a Store, /prefs lists the traveller's saved preferences
    and /forget deletes them (neither calls the LLM).

    read and write default to input() and print(); tests pass fakes. Ctrl+C
    or end of input (EOFError) is left to the caller.
    """
    if thread_id is None:
        thread_id = new_thread_id()
        write(f"New conversation: {thread_id}")
    else:
        count = saved_message_count(graph, thread_id)
        if count:
            write(f"Continuing conversation {thread_id} ({count} saved messages).")
        else:
            write(f"No saved messages for {thread_id}; starting it now.")
    write(
        f"Type {NEW_COMMAND} for a new conversation, {PREFS_COMMAND} or "
        f"{FORGET_COMMAND} for saved preferences, /quit to stop."
    )
    while True:
        text = read("You: ").strip()
        if not text:
            continue
        if text.lower() in QUIT_COMMANDS:
            write(f"Bye! Reopen this conversation with --thread {thread_id}")
            return
        if text.lower() == NEW_COMMAND:
            thread_id = new_thread_id()
            write(f"New conversation: {thread_id}")
            continue
        if text.lower() in (PREFS_COMMAND, FORGET_COMMAND):
            write(preference_command(graph.store, user_id, text.lower()))
            continue
        try:
            turn = ask_agent(graph, thread_id, text, user_id=user_id)
        except Exception as exc:  # keep chatting after one failed turn
            write(f"Request failed: {describe_error(exc)}")
            continue
        if show_steps:
            write("\n".join(describe_steps(turn)))
        write(f"AI: {final_answer(turn)}")


# --- CLI ----------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ask the tool-calling travel agent (usually 2 Gemini calls)."
    )
    parser.add_argument(
        "question", nargs="?", help='e.g. "Find places in Sri Lanka for hiking"'
    )
    parser.add_argument(
        "--steps", action="store_true", help="also print each step of the loop"
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="multi-turn chat with memory (each message makes Gemini calls)",
    )
    parser.add_argument(
        "--thread", metavar="ID", help="with --chat: continue a saved conversation"
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_CHAT_DB,
        help=f"with --chat: SQLite file for conversations (default: {DEFAULT_CHAT_DB})",
    )
    parser.add_argument(
        "--user",
        default=DEFAULT_USER,
        help=f"with --chat: whose preferences to use (default: {DEFAULT_USER})",
    )
    args = parser.parse_args(argv)
    if args.question is None and not args.chat:
        parser.error("give a question, or use --chat")
    if args.question is not None and args.chat:
        parser.error("--chat reads its questions from the terminal")
    if args.thread is not None and not args.chat:
        parser.error("--thread is used together with --chat")
    if args.user != DEFAULT_USER and not args.chat:
        parser.error("--user is used together with --chat")

    # Imported here so the offline tests never load the Gemini setup code.
    from ai_travel_assistant.llm_basics import (
        ConfigError,
        build_llm,
        load_settings,
        safe_error_message,
    )

    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    llm = build_llm(settings, temperature=1.0)
    if args.chat:
        # Conversations (checkpointer) and preferences (Store) live in the same
        # SQLite file, so both survive a restart. The extractor gets its own
        # model at temperature 0 and only runs when the gate matches.
        extractor = make_llm_preference_extractor(build_llm(settings, temperature=0))
        with (
            sqlite_checkpointer(args.db) as checkpointer,
            sqlite_store(args.db) as store,
        ):
            graph = build_tool_agent_graph(
                llm,
                checkpointer=checkpointer,
                store=store,
                preference_extractor=extractor,
            )
            try:
                run_agent_chat(
                    graph,
                    thread_id=args.thread,
                    user_id=args.user,
                    show_steps=args.steps,
                    describe_error=lambda exc: safe_error_message(
                        exc, settings.api_key
                    ),
                )
            except (KeyboardInterrupt, EOFError):  # Ctrl+C / Ctrl+Z ends the chat
                print("\nBye!")
        return 0

    graph = build_tool_agent_graph(llm)
    try:
        messages = run_tool_agent(graph, args.question)
    except Exception as exc:  # show a clean message instead of a traceback
        print(
            f"Request failed: {safe_error_message(exc, settings.api_key)}",
            file=sys.stderr,
        )
        return 1

    if args.steps:
        print("\n".join(describe_steps(messages)))
        print()
    print(final_answer(messages))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
