# AI Travel Assistant — Project Blueprint (v1: Single Agent)

> Status: **DRAFT — awaiting review.** No implementation code until this plan is approved.
> Stack: Python 3.10 · LangGraph · Google Gemini · Windows / VS Code

---

## 1. Project Overview

### 1.1 What we are building

A **conversational, tool-using travel planning agent** for Sri Lanka (scope can widen later). The user writes a free-form request such as:

> "I want to plan a 3-day trip to Ella. I like nature, hiking, and scenic places. Can you prepare an itinerary?"

The agent:

1. **Extracts the trip requirements**: destination, duration, interests, budget, travel dates, group type, pace.
2. **Checks what's missing.** If a missing detail matters (no destination, say), it **asks the user**. If it doesn't matter much (no budget), it **assumes a sensible default and says so**.
3. **Decides which tools to call**: geocode Ella, get the weather forecast, find hiking trails and viewpoints nearby, pull background from Wikivoyage, estimate travel times.
4. **Runs the tools**, reads the results, and calls more tools if it needs to (the ReAct loop).
5. **Produces a structured itinerary**: days, time slots, places, a weather note, practical tips, and the assumptions it made.
6. **Remembers the conversation**, so "make day 2 more relaxed" works, and **remembers stable preferences** such as "I'm vegetarian" or "I hate early mornings" across sessions.

### 1.2 Why a single agent first

A single agent with good tools covers about 80% of what a travel assistant needs. Multi-agent systems mostly add *coordination* problems: hand-offs, shared state, and debugging across agents. You want to have mastered state, tools, memory, interrupts and evaluation in one graph before adding those. Everything in this plan is designed so the single agent can later become **one specialist inside a larger system** without a rewrite (see §9).

### 1.3 Why LangGraph

| Need | How LangGraph handles it | Alternative and its trade-off |
|---|---|---|
| Explicit control flow (extract → clarify → plan → respond) | You define the graph: nodes, edges, conditional routing | A plain `create_agent` loop is simpler but gives you less control over the steps |
| Loops (call tool → reason → call tool again) | Cycles are first-class; LangChain chains are acyclic | Hand-written `while` loop: fine at first, but you end up rebuilding persistence and interrupts yourself |
| Conversation memory | **Checkpointers** persist state per `thread_id` | Manual history storage |
| Human-in-the-loop | `interrupt()` pauses the graph mid-node; `Command(resume=...)` continues it | Very hard to do cleanly by hand |
| Long-term memory | **Store** API (namespaced key-value storage, optional semantic search) | Custom DB code |
| Debuggability | LangGraph Studio (visual graph and step-through) plus optional LangSmith traces | Print statements |
| Path to multi-agent | Subgraphs, supervisor patterns, `Command(goto=...)` hand-offs | Migrating frameworks later |

**Trade-off:** LangGraph is lower-level than "agent in one function call" frameworks, so there's more boilerplate. That cost is the point for a learning project, and it pays off once Izzy needs custom workflows.

### 1.4 Core Agentic AI concepts you will learn (in build order)

1. **LLM as a reasoning engine**: messages, roles, system prompts, temperature.
2. **Structured output**: making the LLM return validated Pydantic objects instead of free text.
3. **State machines and graphs**: state, nodes, edges, reducers.
4. **Tool calling**: the LLM *proposes* a call (name and arguments), and *your code executes it*.
5. **The ReAct loop**: Reason → Act → Observe → repeat until done.
6. **Conditional routing**: deciding the next node from state.
7. **Short-term memory**: per-conversation state persisted by checkpointers.
8. **Long-term memory**: facts about the user that persist across conversations.
9. **Human-in-the-loop**: pausing for clarification or approval.
10. **Guardrails and error recovery**: retries, fallbacks, iteration limits, graceful degradation.
11. **Evaluation**: testing non-deterministic systems, checking trajectories, LLM-as-judge.

---

## 2. Requirements and Scope

### 2.1 Functional requirements (v1)

| ID | Requirement |
|---|---|
| FR1 | Accept natural-language travel requests through a CLI chat interface |
| FR2 | Extract a `TripRequest` (destination, days, interests, budget, dates, travellers, pace) as structured data |
| FR3 | Ask a clarifying question when a **critical** field is missing or ambiguous; otherwise apply documented defaults |
| FR4 | Call real external tools: geocoding, weather, points of interest, destination knowledge, travel time, currency, and optionally web search |
| FR5 | Generate a structured day-by-day itinerary (Pydantic model) and render it to readable Markdown |
| FR6 | Support follow-up edits in the same conversation ("swap day 2 and 3", "add a tea factory visit") |
| FR7 | Persist conversations across process restarts (SQLite checkpointer) |
| FR8 | Remember user preferences across conversations (long-term store) |
| FR9 | Handle tool failures, rate limits, invalid LLM output and runaway loops without crashing |
| FR10 | Politely decline or redirect requests that aren't about travel |

### 2.2 Non-functional requirements

- **Secrets:** all keys live in `.env`, loaded with `pydantic-settings`, and `.env` is git-ignored. Add a `.env.example`.
- **Cost:** run entirely on free tiers (Gemini free tier, keyless public APIs).
- **Latency:** a full itinerary in under ~30 s on the free tier.
- **Observability:** every run can be traced, either locally in Studio or in LangSmith.
- **Testability:** deterministic unit tests that need no API key, plus a separate, opt-in suite that calls the real LLM.

### 2.3 Explicit v1 limitations (out of scope)

- No real bookings, payments or availability checks for hotels, trains or flights.
- No live prices. Budget figures are estimates and are labelled as such.
- No multiple agents, MCP servers, authentication, or web front end. A Streamlit UI is an optional stretch goal.
- No RAG over private RAAHI data (a small public-knowledge retrieval tool is in scope).
- Free public APIs have limits: OSM point-of-interest data can be sparse, and Nominatim and Overpass impose rate limits. The agent must say when data is thin instead of inventing it.
- English only.

---

## 3. Architecture Design

### 3.1 High-level graph

```
                         ┌──────────────────────────────┐
     user message ──────►│            START             │
                         └──────────────┬───────────────┘
                                        ▼
                         ┌──────────────────────────────┐
                         │  load_memory                 │  read long-term prefs (Store)
                         └──────────────┬───────────────┘
                                        ▼
                         ┌──────────────────────────────┐
                         │  understand_request          │  LLM → structured TripRequest
                         │  (classify intent + extract) │  + intent: plan|modify|question|off_topic
                         └──────────────┬───────────────┘
                                        ▼
                         route_after_understanding()  ◄── conditional edge
              ┌──────────────┬──────────┴─────────┬───────────────────┐
     off_topic│   missing    │ critical info      │ ready             │ simple question
              ▼              ▼                    ▼                   ▼
     ┌──────────────┐ ┌────────────────┐  ┌───────────────────┐  (goes to agent too,
     │ respond_     │ │ ask_clarifi-   │  │      agent        │◄─┐ with lighter prompt)
     │ off_topic    │ │ cation         │  │ (Gemini + bound   │  │
     └──────┬───────┘ │ interrupt() ⏸  │  │  tools; ReAct)    │  │
            │         └───────┬────────┘  └────────┬──────────┘  │
            │                 │ resume with         │             │
            │                 │ user answer         ▼             │
            │                 └──► understand   route_after_agent()
            │                      _request     ┌────┴──────┬──────────────┐
            │                                   │tool_calls │ no calls     │ iteration cap /
            │                                   ▼           ▼              │ fatal error
            │                          ┌─────────────┐ ┌─────────────────┐ ▼
            │                          │   tools     │ │ compose_answer  │ ┌──────────────┐
            │                          │ (ToolNode)  │ │ structured      │ │ fallback_    │
            │                          │ + error     │ │ Itinerary →     │ │ response     │
            │                          │ capture     │ │ Markdown        │ └──────┬───────┘
            │                          └──────┬──────┘ └───────┬─────────┘        │
            │                                 └───────────────►│(back to agent)   │
            │                                    loop ─────────┘                  │
            │                                                  ▼                  │
            │                                   ┌──────────────────────────────┐  │
            │                                   │  save_memory                 │◄─┘
            │                                   │  extract durable prefs →     │
            │                                   │  Store (only if new info)    │
            │                                   └──────────────┬───────────────┘
            └─────────────────────────────────────────────────►▼
                                                            ┌──────┐
                                                            │ END  │
                                                            └──────┘

   Persistence (cross-cutting):
     Checkpointer (SQLite) ── saves full state after every node, keyed by thread_id
     Store (in-memory → SQLite/Postgres) ── long-term user prefs, keyed by (user_id, "preferences")
```

The `tools → agent` edge is the ReAct loop. The agent keeps calling tools until it has enough to answer, at which point it goes to `compose_answer`.

### 3.2 Component responsibilities

| Component | Type | Responsibility | Why it's a separate node |
|---|---|---|---|
| `load_memory` | Node (no LLM) | Fetch the user's saved preferences from the Store and put them in state | Keeps memory I/O out of the reasoning prompts; easy to test |
| `understand_request` | Node (LLM, structured output) | Classify intent; extract or merge a `TripRequest`; list `missing_fields` | Separating *understanding* from *acting* makes clarification deterministic and testable |
| `route_after_understanding` | Conditional edge (pure Python) | Pick the next node from the intent and missing fields | Pure function, so it can be unit-tested without an LLM |
| `ask_clarification` | Node with `interrupt()` | Pause the graph, surface a question, and receive the user's answer on resume | The core human-in-the-loop pattern |
| `agent` | Node (LLM + `bind_tools`) | Reason about the plan; emit tool calls or signal it's done | The "brain" of the ReAct loop |
| `tools` | `ToolNode` (prebuilt) | Execute the requested tools, possibly in parallel; turn exceptions into `ToolMessage` errors | Standard, battle-tested executor |
| `route_after_agent` | Conditional edge | tools / compose / fallback, based on tool calls, iteration count and errors | Loop control and guardrails live here |
| `compose_answer` | Node (LLM, structured output) | Turn gathered facts into a validated `Itinerary`, then render Markdown | A dedicated output stage gives consistent structure and lets you validate it |
| `fallback_response` | Node | Give an honest partial answer when the loop hits its limit or tools are down | Graceful degradation |
| `save_memory` | Node (LLM, small) | Extract *durable* preferences ("vegetarian", "avoids long drives") and upsert them into the Store | Keeps long-term memory deliberate rather than dumping everything |

**Design note (trade-off).** The minimal alternative is just `agent ⇄ tools`, which is the classic ReAct graph. We'll **build that first in Stage 3**, then add `understand_request`, `clarification` and `compose_answer` around it. You'll see exactly what each addition buys you, and if a node doesn't earn its place we drop it.

### 3.3 State management

```python
# Conceptual: not final code
class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]   # conversation + tool messages (reducer appends)
    trip_request: TripRequest | None                       # structured requirements (merged across turns)
    intent: Literal["plan", "modify", "question", "off_topic"] | None
    missing_fields: list[str]                              # drives clarification routing
    user_prefs: dict                                       # loaded from long-term store
    itinerary: Itinerary | None                            # last generated structured plan
    tool_iterations: int                                   # loop guard
    errors: list[str]                                      # non-fatal problems to surface/log
```

Key concepts:

- **State is the single source of truth.** Nodes receive the state and return a *partial update*, never the whole thing.
- **Reducers** decide how updates merge. `add_messages` appends to the message list and de-duplicates by message ID. Every other field is overwritten by default.
- **Pydantic for domain objects** (`TripRequest`, `Itinerary`), and `TypedDict` for the graph state itself, which is lighter and the most common choice.
- **Runtime context, not state, for static per-run data** such as `user_id` and locale. This is passed via `context_schema` / `config`, which prepares us for authentication later.

### 3.4 Nodes, edges and conditional routing

- **Normal edge:** `add_edge("tools", "agent")` always goes the same way.
- **Conditional edge:** `add_conditional_edges("agent", route_after_agent, {...})`. A pure function reads state and returns the name of the next node.
- **`Command`:** a node can return `Command(update=..., goto=...)` to update state and route at once. We'll use it sparingly, mainly when resuming from interrupts. It becomes important for multi-agent hand-offs later.
- **Rule:** routing functions **never call the LLM**. Decisions the LLM makes are written into state (`intent`, `missing_fields`, tool calls), and routing just reads that state. This keeps routing cheap, deterministic and unit-testable.

### 3.5 LLM integration (Gemini)

- **Package:** `langchain-google-genai` → `ChatGoogleGenerativeAI`, which uses the `google-genai` SDK underneath and calls the Gemini Developer API with a free API key from Google AI Studio.
- **Model:** a current **Flash**-tier model for the agent loop (fast, generous free tier). Optionally use a Pro-tier model for `compose_answer` if quality needs it. The model ID lives in `.env` (`GEMINI_MODEL=...`) so upgrades need no code change. We'll confirm the current model IDs and free-tier limits on ai.google.dev during Stage 0.
- **Three ways we'll use it:**
  1. `llm.invoke(messages)` for plain chat.
  2. `llm.with_structured_output(TripRequest)` for extraction and the itinerary.
  3. `llm.bind_tools([...])` for the agent's tool use.
- **Single factory** (`llm/factory.py`): every node gets its model from one place. This centralises retries, timeouts and temperature, and lets tests inject a fake model.
- **Temperature:** around 0 for extraction and routing, around 0.4–0.7 for itinerary writing.

### 3.6 Tool calling and execution

The LLM never runs code. It returns an `AIMessage` with `tool_calls=[{name, args, id}]`. Our `ToolNode` executes them and appends `ToolMessage`s, and the LLM then sees those results on its next turn.

**Tool layering**, which is the important part for MCP later:

```
services/  → plain Python API clients (httpx), no LangChain imports. Typed inputs/outputs.
tools/     → thin @tool wrappers: docstring (the LLM reads this!), arg schema, call service, format result.
```

Tools wrap services and contain no logic of their own. When we move to MCP, the same `services/` get wrapped as MCP server tools instead, and the agent consumes them via `langchain-mcp-adapters`. The business logic doesn't change.

**v1 tool set (all real, free):**

| Tool | Backing API | Key needed? | Purpose |
|---|---|---|---|
| `geocode_place` | Open-Meteo Geocoding (fallback: Nominatim/OSM) | No | Place name → lat/lon, region, country |
| `get_weather_forecast` | Open-Meteo Forecast | No | Daily forecast (up to 16 days): rain, temperature, sunrise/sunset |
| `find_attractions` | OpenStreetMap Overpass API | No | Viewpoints, peaks, waterfalls, hiking trails, museums near a coordinate |
| `get_destination_guide` | Wikivoyage / Wikipedia REST API | No | Background, "see/do/get around" sections |
| `estimate_travel_time` | OSRM public routing server | No | Driving distance and time between points |
| `convert_currency` | Frankfurter (ECB rates) | No | LKR ↔ USD/EUR for budget estimates |
| `web_search` *(optional)* | Tavily free tier | Yes (free) | Current info: opening hours, closures, events |

All network calls get timeouts, retries with back-off (`tenacity`), a proper User-Agent header (which Nominatim and Overpass require), and simple response caching (`requests-cache`/`hishel` or a small disk cache). Caching speeds up development and keeps us polite to free services.

### 3.7 Memory and conversation history

| Layer | Scope | LangGraph mechanism | v1 backend | Later |
|---|---|---|---|---|
| **Short-term** | One conversation (thread) | **Checkpointer**: state snapshot after every node, keyed by `thread_id` | `InMemorySaver` → `SqliteSaver` | `PostgresSaver` |
| **Long-term** | One user, across threads | **Store**: `store.put((user_id, "preferences"), key, value)` | `InMemoryStore` → SQLite-backed | Postgres store + semantic search |

- **Context-window management:** long chats get trimmed with `trim_messages` (keep the system prompt plus the last N tokens). Old tool outputs, which are bulky JSON, get summarised or dropped once they've been used in an itinerary.
- **What goes into long-term memory:** only durable facts (diet, mobility, travel style, home currency), never one-off trip details. `save_memory` decides, and it is conservative.
- **Bonus from checkpoints:** "time travel". You can inspect or replay any past step, which is great for debugging.

### 3.8 Human-in-the-loop

- **Mechanism:** call `interrupt(payload)` inside `ask_clarification`. The graph stops and the checkpointer saves state. The CLI shows the question, then calls `graph.invoke(Command(resume=user_answer), config)`, and execution continues *inside the same node*.
- **v1 uses:**
  1. Clarifying critical missing info (for example, no destination).
  2. Optionally, confirming before an expensive regeneration ("This will replace your whole itinerary. Continue?").
- **Rule:** **a checkpointer is required** for interrupts. This is why persistence comes before HITL in the roadmap.
- **Gotcha to learn:** a node containing `interrupt()` **re-runs from its start** on resume, so code before the interrupt must be idempotent.

### 3.9 Error handling strategy (defence in depth)

| Failure | Layer | Handling |
|---|---|---|
| HTTP timeout, 5xx or 429 from an API | `services/` | `tenacity` retry with exponential back-off (max 3); then raise a typed `ServiceError` |
| A tool raises | `ToolNode` | `handle_tool_errors` returns a `ToolMessage` saying "Tool failed: …", so the LLM can retry differently or carry on without that data |
| Gemini 429 or 5xx | LLM factory / node | `max_retries` on the model plus a LangGraph `RetryPolicy` on LLM nodes |
| LLM returns invalid structured output | `understand_request` / `compose_answer` | Pydantic validation, then one repair retry with the error fed back, then a fallback (for example, plain Markdown itinerary) |
| Infinite tool loop | `route_after_agent` | `tool_iterations` cap (for example, 8) → `fallback_response`; plus graph `recursion_limit` as a hard stop |
| Hallucinated tool name or arguments | `ToolNode` and arg schema | Validation error goes back to the LLM as a `ToolMessage` |
| Unsafe or off-topic request | `understand_request` → routing | `respond_off_topic` node |
| Data gaps (OSM has no trails) | Prompt rules | The agent must say "limited data" rather than invent places |

All errors are logged with the `thread_id`. User-facing messages never show stack traces.

### 3.10 Graph execution lifecycle (one turn)

1. CLI receives input and builds `config = {"configurable": {"thread_id": ...}}` and `context = {"user_id": ...}`.
2. `graph.stream({"messages": [HumanMessage(...)]}, config, stream_mode="updates")`.
3. The checkpointer **loads the previous state** for that thread, and the new message is merged in via the reducer.
4. Nodes run along the edges. **After each step (super-step) a checkpoint is written.**
5. If a node calls `interrupt()`, the stream yields an `__interrupt__` event and the run stops. The CLI asks the user, then resumes with `Command(resume=...)`.
6. The run reaches `END`, the final state is checkpointed, and the CLI prints the final AI message or rendered itinerary.
7. The next user message on the same `thread_id` continues from that state.

---

## 4. Technology Stack

### 4.1 Required

| Tech | Purpose | Why |
|---|---|---|
| **Python 3.10** (consider 3.11+ later) | Runtime | Your current version. LangGraph 1.x supports 3.10. 3.11+ is faster and gives better async tracebacks; upgrading is optional |
| **uv** | Venv and dependency management | Very fast, works well on Windows, produces a lockfile. (Alternative: `venv` + `pip`) |
| **langgraph** (1.x) | Orchestration | The core framework |
| **langchain-core** / **langchain** (1.x) | Messages, `@tool`, chat model interface, `trim_messages` | LangGraph's building blocks |
| **langchain-google-genai** | Gemini chat model integration | Official LangChain ↔ Gemini package |
| **langgraph-checkpoint-sqlite** | Persistent checkpointer | Free, file-based, ideal on Windows |
| **pydantic** v2 / **pydantic-settings** | Schemas and validated config from `.env` | Structured output plus safe configuration |
| **python-dotenv** | `.env` loading | Secrets kept out of code |
| **httpx** | HTTP client for services | Sync and async, timeouts, modern |
| **tenacity** | Retries with back-off | Declarative and tested |
| **pytest** (+ `pytest-asyncio`, `pytest-mock`) | Testing | Standard |
| **respx** or **pytest-recording (VCR)** | Mocking or recording HTTP calls in tests | Deterministic tool tests without network |
| **ruff** | Lint and format | One fast tool |
| **rich** | Pretty CLI output | Better developer experience for the chat loop |

Exact versions get pinned in `pyproject.toml` / `uv.lock` in Stage 0, after checking the current releases on PyPI and the LangGraph docs.

### 4.2 External services (free)

- **Google AI Studio API key** for Gemini (free tier).
- **Open-Meteo, Nominatim, Overpass, Wikivoyage/Wikipedia, OSRM, Frankfurter.** No keys, but respect their usage policies.
- *(Optional)* **Tavily** for web search (free monthly credits).

### 4.3 Optional / recommended later

| Tech | When | Why |
|---|---|---|
| **LangSmith** (free developer tier) | From Stage 3 | Trace every LLM and tool call; run evaluations. Turned on purely by environment variables |
| **langgraph-cli + LangGraph Studio** | From Stage 2 | Visual graph debugging, step-through, editing state |
| **mypy** / **pyright** (via Pylance) | Anytime | Type safety on state and schemas |
| **Streamlit** or **Chainlit** | Stage 10 stretch | Quick chat UI demo |
| **FastAPI** | v2 | Serve the agent as an API (the Izzy integration path) |
| **agentevals** / **openevals** | Stage 9 | Ready-made trajectory checks and LLM-as-judge evaluators |
| **pre-commit** | Anytime | Run ruff and tests before commits |

---

## 5. Development Roadmap

Estimates assume about 3 focused hours a day. **Rule: a stage is done only when its tests pass and the deliverable runs.** Each stage ends with a git commit (and optionally a tag such as `stage-3`).

### Stage 0: Environment and Foundations (1 day)
- **Learn:** project hygiene for AI apps; Gemini free-tier limits; secret handling.
- **Concepts:** environment config, the `src/` layout, dependency pinning.
- **Tasks:** install uv; create the project; `.gitignore`, `.env.example`, `config.py` (pydantic-settings); get a Gemini key; check current model IDs and rate limits; set up ruff and pytest; a "hello Gemini" smoke script.
- **Deliverable:** `uv run python -m travel_assistant.smoke` prints a Gemini reply.
- **Tests:** config loads and fails clearly when the key is missing; smoke test, marked `@pytest.mark.live`.

### Stage 1: LLM Fundamentals with Gemini (1–2 days)
- **Learn:** messages (System/Human/AI), prompts, temperature, **structured output**.
- **Concepts:** why structured output matters for agents; Pydantic schemas as contracts with the LLM.
- **Tasks:** an LLM factory; define the `TripRequest` schema; an extraction function: text → `TripRequest`; try it on ~10 varied prompts.
- **Deliverable:** `extract_trip_request("3 days in Ella, love hiking")` → a valid object.
- **Tests:** schema validation unit tests; a live extraction test over 10 sample prompts with ≥ 9/10 correct on destination and days.

### Stage 2: First LangGraph: State, Nodes, Edges (2 days)
- **Learn:** `StateGraph`, `TypedDict` state, reducers (`add_messages`), `START`/`END`, `compile()`, `invoke` vs `stream`.
- **Concepts:** a graph as a state machine; partial state updates; super-steps.
- **Tasks:** a 2-node graph (`understand_request` → `respond`); draw it (`get_graph().draw_mermaid()`); run it in LangGraph Studio (`langgraph dev`).
- **Deliverable:** a CLI one-shot that extracts requirements and echoes a summary.
- **Tests:** node functions unit-tested with a **fake chat model** (no API calls); graph compiles; state transitions are as expected.

### Stage 3: Tools and the ReAct Loop (3 days) ⭐ core concept
- **Learn:** `@tool`, docstrings as the LLM's documentation, `bind_tools`, `tool_calls`, `ToolMessage`, `ToolNode`, `tools_condition`.
- **Concepts:** the LLM proposes and the code executes; the Reason → Act → Observe loop; parallel tool calls.
- **Tasks:** build the minimal `agent ⇄ tools` graph by hand (not prebuilt) with **2 real tools**: `geocode_place` and `get_weather_forecast` (Open-Meteo, no key). Then rebuild it with `create_agent` to compare.
- **Deliverable:** "What's the weather in Ella this weekend?" → the agent geocodes, gets the forecast and answers.
- **Tests:** tools unit-tested with recorded HTTP; routing function unit-tested; a trajectory test with a fake model that scripts tool calls; live test that the expected tools were called.

### Stage 4: Real Tool Suite (3–4 days)
- **Learn:** designing good tools (narrow purpose, clear arguments, compact outputs); the services/tools split; rate limiting and caching.
- **Concepts:** tool selection quality depends on names, descriptions and output size, which matters for context cost.
- **Tasks:** implement `find_attractions` (Overpass), `get_destination_guide` (Wikivoyage), `estimate_travel_time` (OSRM), `convert_currency`; optionally `web_search`. Add retries, timeouts, caching and the User-Agent header. Fall back to a second Overpass server, then to the Wikivoyage guide's sights, when OpenStreetMap is unavailable.
- **Deliverable:** the agent drafts grounded 3-day plans for Kandy, Galle, Nuwara Eliya, Sigiriya and Ella with one generic logic (no destination-specific code): attractions, OSRM travel times, weather, Wikivoyage background and converted prices, all from tool results, with the default origin (Colombo) stated and missing data listed as "Not available". Checked live with `itinerary_check.py`, one destination at a time (Kandy first).
- **Tests:** each service has success, empty-result and failure tests (mocked); a tool-selection test set of ~15 prompts → expected tools; offline itinerary tests for all five destinations (prompt and code free of destination names, full trajectory per place, Overpass-outage fallback).

### Stage 5: Memory: Short-term and Long-term (3 days)
- **Learn:** checkpointers, `thread_id`, state history, the Store API, `trim_messages`.
- **Concepts:** thread state vs user memory; the context-window budget; what is worth remembering.
- **Tasks:** `InMemorySaver` → `SqliteSaver`; multi-turn CLI with a `/new` thread command; `load_memory` and `save_memory` nodes; message trimming. As built: `tool_calling --chat` keeps conversations and preferences in one git-ignored SQLite file (`.data/chat.sqlite`), reopens a conversation with `--thread`, and picks the traveller with `--user` (passed as runtime context, not state). `save_memory` runs a keyword gate first and spends a Gemini call only when a message looks like a lasting preference; `/prefs` and `/forget` show or clear them without the LLM. Each LLM call sees the current turn whole plus earlier turns compacted (traveller messages and final answers only) and trimmed to about 8,000 tokens; the saved thread keeps everything.
- **Deliverable:** restart the app and the conversation continues. A preference stated in thread A shapes thread B.
- **Tests:** a follow-up turn uses earlier context; persistence survives a fresh graph instance; preference extraction precision (no one-off trip facts stored). As built, all offline: thread isolation, SQLite restart, cross-thread but per-user preferences, plain trip requests never calling the extractor, extractor failure keeping the answer, and trimming that never splits a tool call from its result. Still to check live: a preference stated in one chat shaping a plan in a new one.

### Stage 6: Requirement Understanding and Clarification (HITL) (3 days)
- **Learn:** `interrupt()`, `Command(resume=...)`, idempotency on resume, intent classification.
- **Concepts:** when to ask versus assume; critical versus optional fields; the agent being honest about its assumptions.
- **Tasks:** full `understand_request` (intent + merge with the previous `TripRequest`); `route_after_understanding`; an `ask_clarification` node; CLI handling of the interrupt and resume; an off-topic path. As built: the nodes live in `request_understanding.py` and are added to the agent only with `build_tool_agent_graph(..., understander=...)`, which needs a checkpointer; without it the graph is the Stage 5 one, so one-shot mode and `itinerary_check` are unchanged. One structured-output call (`RequestUnderstanding`: intent plus the trip details of the latest message) reads every chat message; `--chat --no-clarify` skips it to save quota. Only a `plan` missing destination or days asks; a question without a place goes to the agent. `ask_clarification` writes a fixed-template question (no LLM) and `wait_for_reply` only calls `interrupt()`, so re-running it on resume is safe; `ask_agent` resumes a paused thread with `Command(resume=text)`. Details merge across turns (a new destination starts a new trip) and reach the agent through the system prompt. At most 2 questions in a row, then the agent answers and states its assumptions; if understanding fails, the agent answers as before. Off-topic gets a fixed reply but still passes `save_memory`'s free keyword gate. A paused question survives a restart and is shown again with `--thread`.
- **Deliverable:** "Plan me a trip" → "Where would you like to go, and for how many days?" → continues correctly.
- **Tests:** ambiguous prompts trigger clarification (≥ 90%); complete prompts never trigger it (≥ 95%); resume produces the correct state. As built, offline (`test_clarification.py`, `test_clarification_graph.py`): every router branch, merging, the understanding node, pausing with no agent call, resume state, two-round clarification, one understander call per message on resume, the round limit, understander failure, off-topic, thread isolation, restart while paused, preferences around clarification, and the chat loop. Live (`test_clarification_live.py`, opt-in): the deliverable (~5 calls) and a 6-prompt routing set (6 calls). **Status: offline complete; live verification deferred** (not yet verified against Gemini). The first live run on 2026-11-17 hit Gemini 504 timeouts and then the daily quota; the live checks will be run when API access allows.

### Stage 7: Error Handling and Robustness (2–3 days)
- **Learn:** `RetryPolicy`, `handle_tool_errors`, `recursion_limit`, fallback nodes.
- **Concepts:** graceful degradation; never trusting LLM output blindly; typed errors.
- **Tasks:** implement everything in §3.9; add a fault-injection switch (`FAULT_INJECT=weather`) to simulate outages; structured logging. As built (`robustness.py`, `faults.py`):
  - **One retry layer for Gemini.** `--chat` builds its models with `max_retries=0`, and only the `llm` node gets a LangGraph `RetryPolicy` (`llm_retry_policy()`: 2 attempts, transient errors only, meaning 5xx, timeouts and dropped connections; never 429 or other 4xx). It is opt-in (`build_tool_agent_graph(..., llm_retry=...)`), so callers without it keep their Stage 5 behaviour; one-shot mode keeps the client's own 3 retries. Tools are not node-retried (their services retry at the HTTP level); the understander and preference extractor fall back instead of retrying.
  - **Tool errors.** `ToolNode(handle_tool_errors=tool_error_message)`: an unexpected exception becomes an error result for that call only, so the other parallel results are kept and the turn goes on.
  - **Repaired history.** `repair_history()` runs on what the LLM sees, after trimming. A tool call without a result gets a "not run" result, a result without its call is dropped, and two traveller messages in a row get a "no answer" note. It is pure: the saved thread is never changed.
  - **Loop cap.** `route_after_agent` replaces `tools_condition`. On the request after `MAX_TOOL_ROUNDS = 4` tool rounds in one turn, `fallback_response` closes the stopped calls, makes one wrap-up call, and falls back to a fixed reply if that fails or asks for tools again. At most 6 agent calls per turn.
  - **Daily quota.** A daily-quota 429 in the Stage 6 understander ends the turn at once; other understander errors still fall back to the agent.
  - **Safe messages.** `describe_llm_error()` gives busy, rate-limit, daily-quota, rejected or generic messages, and never the raw error text.
  - **Fault injection.** `FAULT_INJECT=weather,guide:busy` (modes `down`, `busy`, `timeout`) swaps in a failing transport in that service's `make_client`. It is off unless set, a typo is a configuration error, and the CLI prints a warning while it is on.
  - **Not done here.** Structured logging and a structured-output *repair* retry: the understander and extractor already fail safe, so a repair call would only spend quota. Repair belongs with `compose_answer` in Stage 8.
- **Deliverable:** with the weather API "down", the itinerary is still produced and carries a "weather unavailable" note.
- **Tests:** fault-injection tests for each tool; Gemini 429 simulation; loop-cap test; malformed structured-output repair test. As built, all offline (`test_robustness.py`, `test_robustness_graph.py`, `test_faults.py`):
  - error classification against real `google.genai` error objects
  - one retry for a 503, two attempts at most, a single attempt for 429/400/401, no tool re-run on a retry
  - a failing tool keeping the other parallel results
  - recovery after a Gemini failure and from an orphaned tool call, with the saved thread unchanged
  - the cap, wrap-up failure and a wrap-up that asks for tools
  - preferences still saved after the fallback
  - understander quota fail-fast, an agent failure after a Stage 6 resume, and paused-thread isolation
  - the chat's single retry layer and safe messages (no key, no traceback)
  - every service × every fault mode, and the weather-down itinerary
  - the full suite also passes with `FAULT_INJECT` set in the shell

  **Status: offline complete; live verification deferred** (Gemini quota and availability). Still to check live: a `FAULT_INJECT=weather` itinerary, and that Gemini accepts a repaired history.

### Stage 8: Structured Itinerary Output (2 days)
- **Learn:** complex nested schemas; separating content generation from rendering.
- **Concepts:** output contracts make later UIs and APIs trivial (Izzy's frontend can consume the JSON).
- **Tasks:** `Itinerary` / `DayPlan` / `Activity` schemas; `compose_answer` node; a Markdown renderer; a "modify itinerary" flow.
- **Deliverable:** a consistent, well-formatted 3-day Ella itinerary that you can edit with follow-ups.
- **Tests:** schema validity 100%; day count matches the request; every place named in the itinerary appears in the tool results (grounding check).

### Stage 9: Evaluation Suite (3 days)
- **Learn:** evaluating non-deterministic systems; golden datasets; trajectory evaluation; LLM-as-judge.
- **Concepts:** offline evaluation before shipping; regression tracking.
- **Tasks:** build a 25–40 scenario dataset (YAML); an evaluation runner (pytest marker `eval`); metrics report; optionally LangSmith datasets and experiments.
- **Deliverable:** `uv run pytest -m eval` → a scorecard (see §7.3).
- **Tests:** the evaluation suite *is* the test.

### Stage 10: Interface, Polish and Docs (2–3 days)
- **Learn:** streaming tokens and node updates to the UI; packaging.
- **Tasks:** a polished `rich` CLI with streaming and progress ("🔎 checking weather…"); README with an architecture diagram; optionally a Streamlit UI; a demo recording.
- **Deliverable:** a v1 release tag.

**Total: about 25–30 working days, roughly 6 weeks at 3 h/day.** Stages 3, 5 and 6 carry the most learning, so budget slack there.

---

## 6. Project Folder Structure

```
AI-Travel-Assistant-Agent/
├── pyproject.toml              # deps, ruff, pytest config (uv-managed)
├── uv.lock
├── langgraph.json              # LangGraph Studio / CLI entry point
├── .env.example                # GOOGLE_API_KEY=, GEMINI_MODEL=, TAVILY_API_KEY=, LANGSMITH_*=
├── .gitignore                  # .env, .venv, *.sqlite, .cache/
├── README.md
├── docs/
│   ├── PROJECT_PLAN.md         # this file
│   ├── architecture.md         # living architecture notes + diagrams
│   └── decisions/              # short ADRs: "why SQLite", "why separate compose node"
├── src/
│   └── travel_assistant/
│       ├── __init__.py
│       ├── config.py           # pydantic-settings: all env vars, one place
│       ├── logging_config.py
│       ├── llm/
│       │   └── factory.py      # get_chat_model(purpose="agent"|"extract"|"compose")
│       ├── schemas/            # domain models: no LangGraph imports
│       │   ├── trip.py         # TripRequest, Traveller, Budget
│       │   └── itinerary.py    # Itinerary, DayPlan, Activity
│       ├── services/           # pure API clients: reusable by MCP servers later
│       │   ├── http.py         # shared httpx client, retries, cache, UA
│       │   ├── geocoding.py
│       │   ├── weather.py
│       │   ├── places.py       # Overpass
│       │   ├── knowledge.py    # Wikivoyage / Wikipedia
│       │   ├── routing.py      # OSRM
│       │   └── currency.py
│       ├── tools/              # thin @tool wrappers over services
│       │   ├── __init__.py     # ALL_TOOLS registry
│       │   └── travel_tools.py
│       ├── prompts/            # prompt templates as versioned text/py
│       │   ├── system.py
│       │   ├── understand.py
│       │   └── compose.py
│       ├── memory/
│       │   ├── checkpointer.py # factory: memory | sqlite | (later) postgres
│       │   └── store.py        # long-term preference store helpers
│       ├── agents/
│       │   └── travel_planner/ # ← v1's single agent, self-contained
│       │       ├── state.py    # TravelState
│       │       ├── nodes.py    # understand, agent, compose, clarify, fallback, memory nodes
│       │       ├── routing.py  # pure routing functions
│       │       └── graph.py    # build_graph(checkpointer, store) -> CompiledStateGraph
│       └── interfaces/
│           └── cli.py          # chat loop, interrupt/resume handling, streaming
├── tests/
│   ├── conftest.py             # fake LLM fixtures, recorded-HTTP fixtures
│   ├── unit/                   # schemas, services (mocked HTTP), tools, routing, nodes
│   ├── integration/            # whole graph with fake LLM + mocked services
│   ├── live/                   # real Gemini + real APIs (opt-in: -m live)
│   └── evals/
│       ├── datasets/scenarios.yaml
│       └── test_eval_suite.py
└── scripts/
    └── smoke_gemini.py
```

**Why this shape:**
- `agents/travel_planner/` keeps the whole agent in one package. A future `agents/supervisor/`, `agents/booking/` and so on sit beside it, and each compiled graph can be used as a **subgraph**.
- `services/` has no LangChain dependency, so it can be wrapped into **MCP servers** later (or swapped for RAAHI's real backend) without touching the agent.
- `schemas/` are shared contracts that the CLI, a future API and future agents all consume.
- `interfaces/` separates "how users talk to it" from "what it does". The CLI now, then FastAPI or Izzy's channels later.

---

## 7. Testing and Evaluation Strategy

### 7.1 Test pyramid

| Level | What | LLM? | Network? | Runs |
|---|---|---|---|---|
| Unit | Schemas, services, tools, routing functions, renderers | No (fake) | No (mocked or recorded) | Every commit, under 10 s |
| Integration | Compiled graph end-to-end with a scripted fake LLM | Fake | Mocked | Every commit |
| Live | Real Gemini plus real APIs, small set | Yes | Yes | On demand (`-m live`) |
| Eval | Scenario dataset with scored metrics | Yes | Yes (cached) | Before each stage sign-off |

**Key technique:** LangChain's `GenericFakeChatModel`, or a small custom fake, lets you **script exact AIMessages with tool_calls**. That makes routing, the loop, error recovery and memory fully deterministic to test.

### 7.2 What to test, and how

| Area | How | Example scenario | Success criterion |
|---|---|---|---|
| **Reasoning / understanding** | Live: extraction against labelled prompts | "Weekend in Kandy with kids, cheap" → days=2, travellers incl. children, budget=low | ≥ 90% field-level accuracy |
| **Tool selection** | Live trajectory check: set of tools called | "Will it rain in Ella tomorrow?" → `geocode_place`, `get_weather_forecast` | ≥ 90% of scenarios call all *required* tools; 0 calls to clearly irrelevant tools |
| **Tool execution** | Unit with recorded HTTP | Overpass returns 0 results → tool returns "no results" text, not an exception | 100% pass |
| **Conditional routing** | Pure-function unit tests over state fixtures | `missing_fields=["destination"]` → `"ask_clarification"` | 100% branch coverage of routing functions |
| **Memory: short-term** | Integration: 2 turns, same thread | Turn 2: "make it 4 days" → `trip_request.days == 4`, destination kept | 100% |
| **Memory: persistence** | Integration: new graph instance, same SQLite file | Conversation resumes | 100% |
| **Memory: long-term** | Live: thread A "I'm vegetarian" → thread B itinerary | The food suggestions respect it | ≥ 90%; no one-off facts stored (precision ≥ 90%) |
| **HITL** | Integration with interrupt/resume | "Plan a trip" → interrupt → resume "Galle, 2 days" | Correct question asked; state correct after resume |
| **Error recovery** | Fault injection | Weather service raises → itinerary still produced with note | 100% no crashes; graceful message in all injected faults |
| **Loop guard** | Fake LLM that always calls a tool | Hits the cap → fallback node | Terminates; ≤ cap iterations |
| **Response quality** | Schema checks + grounding check + LLM-as-judge rubric | 3-day Ella plan | Schema-valid 100%; day count correct 100%; grounding ≥ 95% (named places come from tool output); judge score ≥ 4/5 on relevance, feasibility, preference match |
| **Safety / off-topic** | Live | "Write my Python homework" | Polite redirect ≥ 95% |

### 7.3 v1 scorecard (Stage 9 output)

```
Scenarios: 30     Pass rate: ≥ 85%
Extraction accuracy ........ ≥ 90%
Clarify when needed ........ ≥ 90%   (false-clarify rate ≤ 5%)
Required tools called ...... ≥ 90%
Itinerary schema-valid ..... 100%
Grounding (no invented places) ≥ 95%
Crash-free under faults .... 100%
Median latency ............. ≤ 30 s
Avg LLM calls per plan ..... tracked (cost proxy)
```

---

## 8. Final Deliverables (v1 Definition of Done)

v1 is complete when you can demo all of the following from the CLI:

1. ✅ "Plan a 3-day trip to Ella; I like nature, hiking and scenic places" → a **grounded, structured, day-by-day itinerary** with real attractions, a weather note, travel-time hints and explicit assumptions.
2. ✅ "Plan me a trip" → the agent **asks** for the destination and duration, then continues.
3. ✅ Follow-ups modify the plan in place ("make day 2 easier", "add Little Adam's Peak at sunrise").
4. ✅ Close the app, reopen it → **the conversation resumes** (SQLite).
5. ✅ Preferences stated once are **remembered in a new conversation**.
6. ✅ Simulated outages of each tool → **no crash**, honest degraded answer.
7. ✅ Off-topic requests are redirected politely.
8. ✅ `pytest` (unit + integration) green without an API key; `pytest -m eval` scorecard meets §7.3.
9. ✅ Graph is viewable and steppable in LangGraph Studio; optionally traced in LangSmith.
10. ✅ README, architecture doc, `.env.example`, and a short decision log (ADRs).

### Postponed to later versions

| Capability | Why postpone |
|---|---|
| Multi-agent (supervisor + specialists) | Need a mastered single agent and evaluation baseline first |
| MCP servers / clients | Services layer prepares for it; adds protocol and ops complexity |
| Real bookings, availability, payments | Needs RAAHI backend, authentication, approval flows, legal and compliance |
| Authentication and multi-tenant users | v1 uses a local `user_id`; context schema is ready for real identity |
| RAG over RAAHI's private content | Requires a data pipeline and vector store, which is a separate learning track |
| Semantic long-term memory (embeddings) | Key-value preferences are enough for v1 |
| Web or mobile UI, voice | Interface layer is pluggable; not core to learning agents |
| Human escalation to staff | Needs real operator tooling; the interrupt pattern is the foundation |
| Deployment (LangGraph Platform / Cloud Run), Postgres | After behaviour is stable |

---

## 9. Future Expansion Toward Izzy

How each v1 decision prepares for the RAAHI Izzy assistant:

| Future need | v1 foundation | Evolution path |
|---|---|---|
| **Multiple specialised agents** | Agent lives in `agents/travel_planner/` with its own state and a `build_graph()` factory | Add `agents/supervisor/` that routes to specialist **subgraphs** (planner, booking, support, professional-workflow) using `Command(goto=...)` hand-offs, or the `langgraph-supervisor` pattern. v1's eval suite becomes the regression baseline for the planner specialist |
| **Traveller vs professional workflows** | `intent` classification node + runtime `context` (`user_id`, later `role`) | `role` in context selects different graphs, tools and prompts; the supervisor routes on role + intent |
| **MCP-based tools** | `services/` free of LangChain; tools are thin wrappers | Wrap `services/` as **MCP servers** (FastMCP); agents load them via `langchain-mcp-adapters` `MultiServerMCPClient`. Other RAAHI systems can reuse the same servers |
| **Real backend/API integration** | Services are typed clients behind interfaces | Add `services/raahi_api.py` (inventory, pricing, user profiles); swap or augment public APIs without changing agent code |
| **Authentication and authorisation** | Identity passed via runtime context, never via the LLM | API layer (FastAPI) validates JWT → injects `user_id` and `roles` into context; tools check permissions *in code* (the LLM never decides authorisation) |
| **Persistent memory** | Checkpointer and store behind factory functions | Swap to `PostgresSaver` and Postgres store with semantic search; add memory TTLs and user "forget me" controls |
| **Booking-related actions** | HITL `interrupt()` pattern already mastered; structured `Itinerary` output | Booking tools are *write* actions → always gated by `interrupt()` for explicit user confirmation; idempotency keys; audit log |
| **Human escalation** | Fallback node + interrupt mechanism | An `escalate_to_human` tool/node creates a ticket (CRM/helpdesk), and the thread pauses until an operator responds via `Command(resume=...)` |
| **Observability and quality at scale** | LangSmith tracing + eval dataset from day one | Continuous evaluation on production traces; prompt versioning; A/B testing models |
| **Serving** | `interfaces/` layer separated | FastAPI or LangGraph Server deployment; Izzy's channels (web, WhatsApp, app) become new interfaces |

**Guiding principle:** the LLM decides *what* to do; deterministic code decides *whether it's allowed* and *how it's executed*. Keeping that boundary in v1 is what makes Izzy safe to extend later.

---

## 10. Open Decisions for Your Review

1. **Python version:** stay on 3.10 (works) or move to 3.11/3.12 now (faster, better async errors)? *Recommendation: 3.12 if nothing else constrains you; otherwise 3.10 is fine.*
2. **Package manager:** `uv` (recommended) or plain `venv` + `pip`?
3. **Geographic scope:** Sri Lanka-focused prompts and defaults (LKR, local transport tips), or global from the start? *Recommendation: Sri Lanka-first, global-capable tools.*
4. **Web search tool (Tavily):** include in Stage 4 (needs a free key), or skip for v1?
5. **LangSmith:** OK to send traces to LangSmith's free tier, or keep everything local (Studio only)?
6. **UI stretch goal:** CLI only, or add Streamlit in Stage 10?

Once you approve or adjust these, we start **Stage 0**, and I'll explain each concept before we implement it.
