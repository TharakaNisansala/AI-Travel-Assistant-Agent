# Architecture

> Status as of 2026-11-23. This describes the code as built. [PROJECT_PLAN.md](PROJECT_PLAN.md) is the original blueprint plus the stage-by-stage build notes; where the two differ, this file and the [decision records](decisions/) describe what exists.

## The graph

The agent is one LangGraph `StateGraph`, built by `build_tool_agent_graph()` in [tool_calling.py](../src/ai_travel_assistant/tool_calling.py). The terminal chat, the web page and LangGraph Studio get it from `build_chat_graph()`. The eval runner calls `build_tool_agent_graph()` directly with the same options but no store and no preference extractor, so scenarios stay isolated.

The node diagram is generated from the code and kept in the README, where a test fails if it drifts: see **[README → Architecture](../README.md#architecture)**. It is not copied here.

## One message, end to end

1. **Interface.** `--chat` (rich or `--plain`), the Streamlit page or Studio sends the message on a `thread_id`, with `user_id` as runtime context (`AgentContext`), never as state. Chat commands (`/new`, `/prefs`, `/forget`, `/itinerary`, `/quit`) are handled by `chat_command()` before the graph and make no LLM call.
2. **`load_memory`** reads the traveller's saved preferences from the Store.
3. **`understand_request`** makes one structured-output call (temperature 0) that returns the intent and the trip details of this message, merged with the details already known. A quick-reply button click is read without an LLM call. See [ADR-0002](decisions/0002-understand-and-compose-nodes.md).
4. **Routing (pure Python)** sends it to:
   - `respond_off_topic`: a fixed reply;
   - `ask_clarification` → `wait_for_reply`: a fixed-template question and an `interrupt()`; the reply resumes the run with `Command(resume=...)`;
   - `llm`: the agent.
5. **`llm` ⇄ `tools`**: the ReAct loop. The model sees the current turn whole and earlier turns compacted and trimmed to about 8,000 tokens (`context_window.py`). The tools node runs tool calls in parallel; a tool that raises becomes an error result for that call only.
6. **The turn ends** in one of three ways:
   - a prose answer;
   - a call to the final-answer tool `submit_itinerary`, which goes to **`compose_answer`** (no LLM): validate, repair once or salvage, ground against the tool results, render Markdown;
   - after `MAX_TOOL_ROUNDS = 4` tool rounds, **`fallback_response`** makes one wrap-up call.
7. **`save_memory`** runs a free keyword gate and calls the LLM only when the message looks like a lasting preference.
8. The checkpointer has saved the state after every step, so the thread can be continued later from any interface.

## Layers

| Layer | Modules | Notes |
|---|---|---|
| Interfaces | `tool_calling.py` (CLI, plain chat), `chat_ui.py`, `web_app.py`, `web_ui.py`, `studio.py`, `streaming.py` | All share `build_chat_graph()` and `chat_command()`. `streaming.py` turns graph events into `Step` / `Token` / `TurnDone` |
| Graph and nodes | `tool_calling.py`, `request_understanding.py`, `robustness.py`, `itinerary_output.py`, `itinerary_grounding.py`, `long_term_memory.py`, `context_window.py` | Routing functions never call the LLM |
| Schemas | `schemas.py` | `TripRequest`, `RequestUnderstanding`, `ItineraryDraft` / `Itinerary`. Shared by nodes, tools and interfaces |
| Tools | `tools.py` | `@tool` wrappers that format service results for the LLM. See [ADR-0003](decisions/0003-services-and-tools.md) |
| Services | `*_service.py`, `resilience.py`, `faults.py` | HTTP clients with no LangChain imports |
| LLM | `llm_basics.py` | `build_llm()` for Gemini, Groq or Cloudflare. See [ADR-0004](decisions/0004-llm-provider-switch.md) |
| Evaluation | `evals/` | Runner, records, scorers, scorecard |

## State and persistence

- **Thread state** (`AgentState`): messages, the merged trip, intent, missing fields, clarification context, the last itinerary, tool-round count. Saved by the checkpointer per `thread_id`.
- **User memory**: short preference strings in the Store, namespace `("users", <user_id>, "preferences")`.
- **Files** (both git-ignored under `.data/`): `chat.sqlite` holds checkpoints and the Store; `prices.sqlite` caches the ticket-price page. See [ADR-0001](decisions/0001-sqlite-persistence.md).
- **In-process caches**: `resilience.TTLCache` per service (1 hour for Overpass and exchange rates, 24 hours for Wikivoyage). Lost on restart.

## Error handling, by layer

| Failure | Where | What happens |
|---|---|---|
| API timeout, 429, 502–504 | `resilience.send_with_retry` | 2 attempts; honours `Retry-After` up to 5 s; then the service raises its own error |
| A tool raises | `ToolNode(handle_tool_errors=...)` | Error result for that call; other parallel results kept |
| LLM 5xx or timeout | `RetryPolicy` on the `llm` node | One retry. The only retry layer, except Groq's client retries for short 429s |
| LLM 429 (or Groq's 413 over the per-minute token limit) | `robustness.describe_llm_error` | Not retried; a safe rate-limit or quota message, never the raw error |
| Understander fails | `understand_request` | The message goes to the agent; a daily-quota 429 ends the turn |
| Invalid itinerary | `compose_answer` | Field errors back to the agent once, then salvage |
| Runaway loop | `route_after_agent` | `fallback_response` after 4 tool rounds |
| Simulated outage | `FAULT_INJECT=weather,guide:busy` | A failing HTTP transport in that service |

## Where the build differs from the blueprint

| Blueprint ([PROJECT_PLAN.md](PROJECT_PLAN.md) §3, §6) | As built | Why |
|---|---|---|
| `compose_answer` is an LLM call | The agent calls `submit_itinerary`; `compose_answer` has no LLM | Saves one call per plan on a 20-a-day quota ([ADR-0002](decisions/0002-understand-and-compose-nodes.md)) |
| Packages `agents/`, `services/`, `tools/`, `memory/`, `interfaces/` | One flat package; `*_service.py` and `tools.py`; one module per stage concept | Not recorded. The split is by naming convention, not by package |
| `llm/factory.py` with `purpose=` | `llm_basics.build_llm(settings, temperature, max_retries)`; callers pick 1.0 (agent) or 0 (understanding, preferences) | Not recorded |
| Gemini only | `LLM_PROVIDER`: Gemini, Groq or Cloudflare | Gemini's free quota limited testing ([ADR-0004](decisions/0004-llm-provider-switch.md)) |
| `tenacity` retries, 3 attempts | Hand-written `resilience.py`, 2 attempts; `tenacity` is not a dependency | 2 attempts so a chat turn does not feel stuck (`resilience.py`); why not `tenacity` is not recorded |
| Frankfurter for currency | ExchangeRate-API (open access) | It includes the Sri Lankan rupee (`currency_service.py`); Frankfurter's ECB rates do not |
| Store: in memory, then SQLite | SQLite from Stage 5 | One file for both stores ([ADR-0001](decisions/0001-sqlite-persistence.md)) |
| Tools contain no logic | `tools.py` also searches, falls back and works out group costs | See [ADR-0003](decisions/0003-services-and-tools.md) |
| No web front end (optional Streamlit) | Streamlit page with day cards, PDF export, quick replies | Stage 10 stretch goal, approved |

## Known limitations

- **One user on one machine.** SQLite and the in-process caches are not built for many writers or several server processes.
- **Free tiers shape behaviour.** Gemini allows about 20 requests a day. Groq `openai/gpt-oss-120b` (free tier, 8,000 tokens per minute) cannot run full itineraries: one request goes over the limit.
- **Providers differ.** Groq validates tool arguments strictly against the schema; the same prompt can be grounded differently by each model, so eval results name the model per scenario.
- **Grounding is checked for itineraries only.** Prose answers to questions are not checked (for example, opening times stated without a tool source).
- **Live verification is incomplete.** Stages 5 and 6 are live-verified (on Groq). Stages 7–9 are verified offline; the live eval run `2026-11-23T1156` is 21 of 30 scenarios done. See the checklist in [PROJECT_PLAN.md](PROJECT_PLAN.md).

## Future considerations (not implemented)

These are options, not commitments. Nothing below exists in the code.

- **Persistence:** `PostgresSaver` and a Postgres Store for several users or processes; semantic (embedding) search over preferences; a shared cache such as Redis.
- **Tools as MCP servers:** wrap the `*_service.py` clients as MCP servers and load them with `langchain-mcp-adapters`. The services already have no LangChain imports.
- **Package layout:** split the flat package into `services/`, `tools/` and `agents/` if a second agent is added.
- **Multi-agent:** use this graph as one specialist subgraph under a supervisor.
- **Serving:** an API layer (for example FastAPI) with real authentication passing `user_id` through runtime context.
- **Provider fallback:** switch provider automatically on quota errors. Today the switch is manual.
- **Observability:** LangSmith tracing is available (off unless `LANGSMITH_TRACING` is set) but not part of the workflow.
