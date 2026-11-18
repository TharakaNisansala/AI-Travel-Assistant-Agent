"""Stage 10: the agent for LangGraph Studio, and its diagram for the README.

What this file demonstrates:
  1. Serving a graph to Studio. langgraph.json points Studio (`langgraph dev`)
     at `graph` in studio_graph.py, which calls make_graph() when the server
     loads it. Studio's server runs it with its OWN checkpointer and store,
     so it is built without them (persistence_provided=True). Studio then
     draws the nodes, runs a thread step by step, shows the state after each
     node and lets you resume the clarification interrupt.
     Why a built graph and not a factory function: Studio calls a factory on
     every request inside its event loop, where reading .env is a blocking
     call that `langgraph dev` refuses. A module is loaded once, in a worker
     thread, where that is fine.
  2. Docs generated from code. chat_graph_mermaid() draws the same --chat
     graph with get_graph().draw_mermaid(), using a model that is never
     called, so the README diagram needs no key, and a test fails if the
     README and the graph drift apart.

`langgraph dev` needs Python 3.11+, while this project runs on 3.10, so it
runs in its own environment (see the README):
    uvx --python 3.12 --from "langgraph-cli[inmem]" --with colorama --with-editable . langgraph dev
Print the diagram:
    uv run python -m ai_travel_assistant.studio --mermaid
"""

from __future__ import annotations

import argparse
from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.memory import InMemoryStore

from ai_travel_assistant.tool_calling import build_chat_graph, build_tool_agent_graph


def make_graph() -> CompiledStateGraph:
    """The --chat agent for Studio (reads GEMINI_API_KEY / GEMINI_MODEL).

    Long-term memory is left out here: its nodes exist only when a store is
    passed at build time, and Studio gives its store to the running graph.
    """
    from ai_travel_assistant.llm_basics import load_settings

    return build_chat_graph(load_settings(), None, None, persistence_provided=True)


class _NeverCalled(GenericFakeChatModel):
    """Stands in for Gemini when only the graph's shape is needed."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> _NeverCalled:
        return self


def chat_graph_mermaid() -> str:
    """The full --chat graph (memory, clarification, itineraries) as Mermaid."""
    graph = build_tool_agent_graph(
        _NeverCalled(messages=iter([])),
        checkpointer=InMemorySaver(),
        store=InMemoryStore(),
        preference_extractor=lambda *args, **kwargs: [],
        understander=lambda *args, **kwargs: None,
        structured_itinerary=True,
    )
    return graph.get_graph().draw_mermaid().strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diagram of the --chat agent graph.")
    parser.add_argument("--mermaid", action="store_true", help="print it as Mermaid")
    args = parser.parse_args(argv)
    if not args.mermaid:
        parser.error("use --mermaid")
    print(chat_graph_mermaid())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
