"""Stage 10: the graph LangGraph Studio loads (see studio.py and langgraph.json).

Importing this module builds the --chat agent, so it needs GEMINI_API_KEY and
GEMINI_MODEL. Only Studio imports it; nothing else in the project does.
"""

from ai_travel_assistant.studio import make_graph

graph = make_graph()
