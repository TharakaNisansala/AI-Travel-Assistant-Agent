# Trip workflow graph

Generated from the compiled LangGraph in `src/ai_travel_assistant/trip_workflow.py`. Do not edit by hand; regenerate with:

```
uv run python -m ai_travel_assistant.trip_workflow --draw --png
```

Solid arrows are fixed edges. Dashed arrows are the conditional edge: `route_after_validation` picks `ask_clarification` when destination or days is missing, and `response` otherwise.

![Trip workflow graph](trip_workflow_graph.png)

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	trip_input(trip_input)
	validation(validation)
	ask_clarification(ask_clarification)
	response(response)
	__end__([<p>__end__</p>]):::last
	__start__ --> trip_input;
	trip_input --> validation;
	validation -.-> ask_clarification;
	validation -.-> response;
	ask_clarification --> __end__;
	response --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```
