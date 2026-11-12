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

No memory, MCP or multiple agents here; each run starts a fresh conversation.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from ai_travel_assistant.tools import search_destinations

# The tools this agent may use. Add more here in Stage 4.
TOOLS: tuple[BaseTool, ...] = (search_destinations,)

# Node names. "tools" is the name tools_condition returns, so the tool node must
# be called exactly that (or we would need a mapping in add_conditional_edges).
LLM = "llm"
TOOLS_NODE = "tools"

SYSTEM_PROMPT = (
    "You are a friendly Sri Lanka travel assistant. "
    "When the traveller asks where to go or which places suit an activity, call "
    "the search_destinations tool and base your answer on its results. "
    "Recommend only places the tool returned, and mention that they come from "
    "the built-in guide. If the tool finds nothing, say so and suggest a broader "
    "search. Keep answers short."
)


# --- Nodes --------------------------------------------------------------------


def make_llm_node(
    llm_with_tools: BaseChatModel,
) -> Callable[[MessagesState], dict]:
    """Build the llm node around a model that already has the tools bound."""

    def llm_node(state: MessagesState) -> dict:
        # The system prompt is added on every call but never stored in the
        # state, so the message history holds only the real conversation.
        messages = [SystemMessage(SYSTEM_PROMPT), *state["messages"]]
        response = llm_with_tools.invoke(messages)
        # MessagesState uses the add_messages reducer: returning a list APPENDS
        # it to the history instead of replacing the history.
        return {"messages": [response]}

    return llm_node


# --- Graph --------------------------------------------------------------------


def build_tool_agent_graph(
    model: BaseChatModel, tools: Sequence[BaseTool] = TOOLS
) -> CompiledStateGraph:
    """Bind the tools to the model and wire the ReAct loop.

    bind_tools() does not call the API. It returns a wrapped model that sends
    the tools' names, descriptions and argument schemas with every request.
    """
    llm_with_tools = model.bind_tools(list(tools))

    builder = StateGraph(MessagesState)
    builder.add_node(LLM, make_llm_node(llm_with_tools))
    # ToolNode looks at the last AIMessage, runs every tool call in it (several
    # calls run in parallel), and returns one ToolMessage per call.
    builder.add_node(TOOLS_NODE, ToolNode(list(tools)))

    builder.add_edge(START, LLM)
    # tools_condition returns "tools" if the last message has tool_calls,
    # otherwise END ("__end__").
    builder.add_conditional_edges(
        LLM, tools_condition, {TOOLS_NODE: TOOLS_NODE, END: END}
    )
    # The loop: after a tool runs, the LLM reads its result and decides again.
    builder.add_edge(TOOLS_NODE, LLM)
    return builder.compile()


def run_tool_agent(graph: CompiledStateGraph, question: str) -> list[AnyMessage]:
    """Ask one question and return the full message history of the run."""
    state = graph.invoke({"messages": [HumanMessage(question)]})
    return state["messages"]


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


# --- CLI ----------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ask the tool-calling travel agent (usually 2 Gemini calls)."
    )
    parser.add_argument("question", help='e.g. "Find places in Sri Lanka for hiking"')
    parser.add_argument(
        "--steps", action="store_true", help="also print each step of the loop"
    )
    args = parser.parse_args(argv)

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

    graph = build_tool_agent_graph(build_llm(settings, temperature=1.0))
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
