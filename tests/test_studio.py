"""Stage 10: the Studio graph factory and the README diagram (offline)."""

from __future__ import annotations

import importlib
import json
import re
import sys
from pathlib import Path

import pytest

from ai_travel_assistant import llm_basics, studio
from ai_travel_assistant.tool_calling import COMPOSE_ANSWER, LLM
from tests.test_itinerary_graph import scripted

ROOT = Path(__file__).resolve().parents[1]


def test_langgraph_json_points_at_the_factory():
    config = json.loads((ROOT / "langgraph.json").read_text(encoding="utf-8"))
    path, _, name = config["graphs"]["travel_agent"].partition(":")
    assert (ROOT / path).resolve() == ROOT / "src/ai_travel_assistant/studio_graph.py"
    assert name == "graph"  # a built graph, not a per-request factory
    assert config["env"] == ".env"  # the key stays in .env, never in the JSON


def test_factory_builds_the_chat_agent_without_its_own_persistence(monkeypatch):
    settings = llm_basics.GeminiSettings(model="fake-model", api_key="k")
    monkeypatch.setattr(llm_basics, "load_settings", lambda: settings)
    monkeypatch.setattr(llm_basics, "build_llm", lambda s, **kw: scripted())
    monkeypatch.delitem(sys.modules, "ai_travel_assistant.studio_graph", raising=False)
    graph = importlib.import_module("ai_travel_assistant.studio_graph").graph
    assert graph.checkpointer is None  # Studio attaches its own
    nodes = graph.get_graph().nodes
    assert {LLM, COMPOSE_ANSWER, "understand_request", "wait_for_reply"} <= set(nodes)


def test_the_builder_still_refuses_an_understander_without_persistence():
    from ai_travel_assistant.tool_calling import build_tool_agent_graph

    with pytest.raises(ValueError, match="needs a checkpointer"):
        build_tool_agent_graph(scripted(), understander=lambda *a, **k: None)


def test_mermaid_shows_the_whole_chat_graph():
    text = studio.chat_graph_mermaid()
    for node in (
        "load_memory",
        "understand_request",
        "wait_for_reply",
        "llm",
        "tools",
        "compose_answer",
        "fallback_response",
        "save_memory",
    ):
        assert f"\t{node}(" in text, node


def test_readme_diagram_matches_the_graph():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```mermaid\n(.*?)\n```", readme, flags=re.DOTALL)
    assert blocks, "README.md has no mermaid diagram"
    assert blocks[0].strip() == studio.chat_graph_mermaid(), (
        "README diagram is out of date; regenerate it with "
        "`uv run python -m ai_travel_assistant.studio --mermaid`"
    )


def test_cli_prints_the_diagram(capsys):
    assert studio.main(["--mermaid"]) == 0
    assert capsys.readouterr().out.strip() == studio.chat_graph_mermaid()
    with pytest.raises(SystemExit):
        studio.main([])
