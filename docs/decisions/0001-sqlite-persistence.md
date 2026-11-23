# ADR-0001: SQLite for conversations and preferences

- **Status:** Accepted, implemented (Stage 5). Live-verified 2026-11-23 on Groq: a preference stated in one chat shaped the answer in a new chat.
- **Date:** Stage 5; recorded 2026-11-23.
- **Related:** [architecture.md](../architecture.md), [PROJECT_PLAN.md](../PROJECT_PLAN.md) Stage 5.

## Context

The agent needs two kinds of memory:

- **Thread memory:** the whole conversation, so follow-ups ("make day 2 easier") work and a paused clarification question can be answered later. LangGraph's `interrupt()` only works with a checkpointer.
- **User memory:** a few lasting preferences ("vegetarian"), shared by every conversation of the same traveller.

Constraints: one developer on Windows, free tools only, no server to run, and conversations must survive a restart. The terminal chat and the web page should see the same conversations.

## Decision

Use one SQLite file, `.data/chat.sqlite` (git-ignored), for both:

| Memory | Class | Key | Code |
|---|---|---|---|
| Threads | `SqliteSaver` (`langgraph-checkpoint-sqlite`) | `thread_id` | `sqlite_checkpointer()` in `tool_calling.py` |
| Preferences | `SqliteStore` | namespace `("users", <user_id>, "preferences")` | `sqlite_store()` in `long_term_memory.py` |

Details as built:

- Each is opened as a context manager that creates `.data/` and closes the connection afterwards. The connections use `check_same_thread=False`, because LangGraph can run nodes on worker threads; the Store connection uses autocommit (`isolation_level=None`) and calls `setup()` to create its tables.
- `--db` (CLI) and `TRAVEL_CHAT_DB` (web page) point to another file, for example for a demo or a test.
- The Streamlit page opens the file once per server process (`st.cache_resource`), so a thread started on the web can be continued with `--chat --thread <id>`, and the other way round.
- `user_id` is runtime context (`AgentContext`), not state, so it is never written into a thread by the LLM.
- What goes into the Store is decided by `save_memory`: a free keyword gate first, then an LLM extractor that keeps only durable preferences. `/prefs` and `/forget` read and clear them without the LLM.
- Separately, `pricing_service.py` caches the ticket-price page in `.data/prices.sqlite` (24-hour TTL). It is a cache, not agent memory.

Where SQLite is **not** used, on purpose:

- **Eval runner:** `InMemorySaver`, no Store. Each scenario gets a fresh thread, and one scenario's preferences must not leak into the next.
- **LangGraph Studio:** the graph is built without persistence (`persistence_provided=True`); the Studio server attaches its own. `InMemorySaver` is used only to draw the Mermaid diagram.
- **Offline tests:** in-memory savers or temporary SQLite files.

## Alternatives considered

| Option | Why not (for v1) |
|---|---|
| `InMemorySaver` / `InMemoryStore` only | Lost on restart; a paused question would be lost too |
| Postgres (`PostgresSaver`, Postgres Store) | Needs a database server; more than one local user needs |
| Hand-written JSON history | No checkpoints per step, no `interrupt()`/resume, no state history |
| Two separate files for threads and preferences | Possible; one file is simpler to locate, back up and delete |

## Consequences

**Benefits**

- Conversations and paused questions survive a restart, with no server to run.
- One file shared by all interfaces.
- Checkpoints after every step allow inspecting past states while debugging.

**Costs and limitations**

- Built for one user on one machine: SQLite allows one writer at a time, and the file is not designed to be shared between processes on different machines.
- The file holds real chat text unencrypted. It is git-ignored; there is no retention limit or automatic cleanup.
- Checkpoints grow with every step. There is no pruning of old threads.
- The Store has key-value lookup only, no semantic search.

## Future considerations (not implemented)

- Postgres checkpointer and Store for several users or processes (the graph takes any `BaseCheckpointSaver` / `BaseStore`, so only the factory changes).
- Retention rules (delete old threads) and a "forget me" for whole conversations, not only preferences.
- Embedding-based search over preferences.
