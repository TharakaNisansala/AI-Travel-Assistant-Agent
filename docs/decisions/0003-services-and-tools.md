# ADR-0003: API clients (services) separate from LLM tools

- **Status:** Accepted, implemented (Stages 3–4, extended later for prices and hotels).
- **Date:** Stage 4; recorded 2026-11-23.
- **Related:** [architecture.md](../architecture.md).

## Context

The agent calls free public APIs. Each has its own URL, rate limits, failure modes and response format. The LLM, on the other hand, needs short, readable text with clear limits ("no data", "service busy"). These are two different jobs, and the API part should be testable and reusable without any LLM code, for example as an MCP server later.

## Decision

Two layers, split by naming convention in one flat package:

**Services** (`*_service.py`): plain Python HTTP clients built on `httpx`. None of them imports LangChain or LangGraph (checked 2026-11-23).

| Service | Source |
|---|---|
| `weather_service.py` | Open-Meteo geocoding and forecast |
| `attractions_service.py` | OpenStreetMap via Overpass, with a second Overpass server |
| `guide_service.py` | Wikivoyage (MediaWiki API) |
| `travel_service.py` | OSRM road routing |
| `currency_service.py` | ExchangeRate-API (open access) |
| `pricing_service.py` | Central Cultural Fund ticket page plus a curated YAML file; SQLite cache |
| `hotel_service.py` | Wikivoyage "Sleep" listings (wikitext) |

Shared by all services:

- `resilience.py`: `send_with_retry` (2 attempts; retries 429, 502–504, timeouts and dropped connections; honours `Retry-After` up to 5 s, otherwise reports "busy") and `TTLCache` (in-process; successes only).
- `faults.py`: `FAULT_INJECT=weather,guide:busy` replaces that service's HTTP transport with a failing one, set in each service's `make_client`.
- Each service raises its own typed errors (for example `WeatherServiceError`).

**Tools** (`tools.py`): `@tool` functions the LLM sees. The docstring is the description, the type hints are the argument schema (with `Literal` enums), and each turns a service result or error into compact text. `get_attraction_prices` and `suggest_hotels` also return the structured data as an artifact (`response_format="content_and_artifact"`), which `itinerary_budget.py` reads to build the trip budget.

### Difference from the blueprint

The blueprint said "tools wrap services and contain no logic of their own", in `services/` and `tools/` packages. As built, `tools.py` (about 860 lines) also holds logic:

- `search_destinations`: a keyword search over a small built-in dataset (no service behind it);
- `find_attractions`: falls back to the Wikivoyage guide's sights when Overpass fails;
- `get_attraction_prices` and `suggest_hotels`: group and per-person costs, currency conversion, and a room-budget filter, using `party.py` and `trip_costs.py`.

The reason for this placement is not recorded. The services themselves stayed free of LLM code, which is the part that matters for reuse.

## Alternatives considered

| Option | Why not (for v1) |
|---|---|
| API calls directly inside the `@tool` functions | Mixes HTTP, retries and LLM formatting; harder to test and reuse |
| MCP servers from the start | Protocol and process overhead before the tools themselves were stable |
| `tenacity` for retries | Not used; the reason is not recorded. `resilience.py` keeps retries to 2 so a chat turn does not stall |
| A shared or persistent HTTP cache (`hishel`, Redis) | A CLI process is short-lived; in-process caches were enough |

## Consequences

**Benefits**

- Services are tested with mocked HTTP and no LLM: success, empty result and failure for each.
- Fault injection works per service, so outages can be simulated in the eval (`fault` scenarios).
- The services can be wrapped as MCP servers without changes.
- Tool output size is controlled in one place (for example, a Wikivoyage article trimmed to about 1,000 tokens by `topic`).

**Costs and limitations**

- More files per feature, and the text the LLM sees is formatted far from the data it comes from.
- `tools.py` is large and holds some business logic, so "thin wrappers" is only partly true; an MCP version would have to move that logic too.
- Caches are per process: empty after a restart and not shared between the CLI and the web page.
- Free public APIs limit what the tools can return (thin OpenStreetMap data, Overpass outages, rate limits).

## Future considerations (not implemented)

- Move the logic in `tools.py` (search, fallback, cost calculation) into services or a domain module, so tools become thin wrappers.
- Split the package into `services/` and `tools/` when another agent needs the same services.
- Wrap the services as MCP servers and load them with `langchain-mcp-adapters`.
- A shared cache if the agent is served to several users.
