# Trip workflow graph

Generated from the compiled LangGraph in `src/ai_travel_assistant/trip_workflow.py`. Do not edit by hand; regenerate with:

```
uv run python -m ai_travel_assistant.trip_workflow --draw --png
```

This is the multi-turn graph (built with a checkpointer). Without a checkpointer, `ask_clarification` goes straight to `__end__` instead and there is no `wait_for_reply`.

| Look | Meaning |
|---|---|
| Solid arrow | Fixed edge |
| Dashed arrow from `validation` | Conditional edge, chosen by `route_after_validation` |
| Lilac node | Unchanged from the single-turn graph |
| Yellow node | Changed: `trip_input` reads each reply together with the question it answers, merges new details into the saved ones and skips blank replies |
| Green node | New: `wait_for_reply` calls `interrupt()` |
| Dashed white box | The traveller, outside the graph |
| Blue cylinder | The checkpointer (conversation memory) |

How one conversation runs:

1. `send_message` starts a run for a `thread_id`. `trip_input` extracts details and `validation` lists what is missing.
2. Missing details: `ask_clarification` writes the question, then `wait_for_reply` calls `interrupt(question)`. The run stops and the checkpointer keeps the state under that `thread_id`.
3. The traveller answers. `send_message` sees the thread is paused and sends `Command(resume=reply)` with the same `thread_id`. The state is reloaded, `wait_for_reply` re-runs and `interrupt()` returns the reply. `wait_for_reply` returns the reply together with the question it answers, saved as `clarification_context`.
4. Back to `trip_input` with the reply. The extractor receives the reply, that question and the details already known, so a short answer like `3` is read as days after *How many days...?*. New details are merged into the saved ones (a missing value never erases a saved one), the question is cleared, and the result is validated again. Steps 2-4 repeat until nothing is missing; each round uses its own latest question.
5. All details present: `response` confirms the trip and the run ends.

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
	wait_for_reply(wait_for_reply)
	__end__([<p>__end__</p>]):::last
	__start__ --> trip_input;
	ask_clarification --> wait_for_reply;
	trip_input --> validation;
	validation -.->|"missing destination or days"| ask_clarification;
	validation -.->|"all details present"| response;
	wait_for_reply -->|"reply: extract, merge, validate again"| trip_input;
	response --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
	traveller{{"Traveller<br/>send_message(graph, thread_id, text)"}}:::outside
	memory[("InMemorySaver checkpointer<br/>state saved after every step<br/>one saved state per thread_id")]:::memory
	wait_for_reply -.->|"interrupt(question): run pauses"| traveller;
	traveller -.->|"Command(resume=reply), same thread_id"| wait_for_reply;
	wait_for_reply <-.->|"pause saved / reloaded on resume"| memory;
	classDef new fill:#c8f0d4,stroke:#1a7f4b,stroke-width:2px
	classDef modified fill:#fff0b3,stroke:#a67c00,stroke-width:2px
	classDef outside fill:#ffffff,stroke:#888888,stroke-dasharray:4 3
	classDef memory fill:#dbeafe,stroke:#1d4ed8,stroke-width:2px
	class wait_for_reply new;
	class trip_input modified;
```

<details>
<summary>Raw graph, exactly as LangGraph draws it</summary>

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
	wait_for_reply(wait_for_reply)
	__end__([<p>__end__</p>]):::last
	__start__ --> trip_input;
	ask_clarification --> wait_for_reply;
	trip_input --> validation;
	validation -.-> ask_clarification;
	validation -.-> response;
	wait_for_reply --> trip_input;
	response --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```

</details>
