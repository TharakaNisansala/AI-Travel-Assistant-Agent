# ADR-0005: Provider fallback for the public web page only

- **Status:** Accepted, implemented on branch `feat/public-deployment` (commits `8288109`, `0091163`). Verified offline (fakes for every provider error); not yet verified against live providers in a deployment.
- **Date:** 2026-11-23.
- **Related:** [ADR-0004](0004-llm-provider-switch.md) (one provider chosen by `LLM_PROVIDER`), [DEPLOYMENT.md](../DEPLOYMENT.md), `provider_pool.py`, `usage_limits.py`.

## Context

ADR-0004 chose one provider at a time and rejected automatic fallback, because an eval record must name the one model that produced it. It listed "optional automatic fallback for the interactive chat, kept off for evaluation" as a future consideration.

The project is now also a free public portfolio demo on Streamlit Community Cloud. That changes the needs of the web page only:

- Visitors cannot switch `LLM_PROVIDER`; a used-up Gemini quota (about 20 requests a day, 3-5 per itinerary) would stop the demo for everyone until the next day.
- Every visitor shares the same free quotas, so failed calls must not be repeated, and a provider that is out of quota must not be hit again and again.
- The CLI, the local web page and the eval runner still need exactly one known model.

## Decision

With `TRAVEL_PUBLIC=1`, the web page builds its models from a **provider pool**; everything else keeps the single provider of ADR-0004.

| Mode | Models | Chosen by |
|---|---|---|
| CLI, local web page, Studio | One provider (unchanged) | `LLM_PROVIDER` |
| Eval runner | One provider (unchanged) | `LLM_PROVIDER` |
| Public web page | Provider pool | `LLM_FALLBACK_ORDER` (default `gemini,cloudflare,groq`) |

Details as built (`provider_pool.py`):

- **Order.** Gemini first (best itineraries in the evals), Cloudflare Workers AI second (passed 1 of 2 itinerary runs on 2026-11-18), Groq last (its free 8,000 tokens a minute stopped 4 of 5 itineraries on 2026-11-23). `LLM_FALLBACK_ORDER` can change or shorten the order; a provider without its key and model is left out; no configured provider is a `ConfigError`.
- **At most one call per provider per request.** `PoolRunnable.invoke()` walks the providers once, in order. There is no loop, so N providers mean at most N calls.
- **Shared health table** (a simple circuit breaker, thread-safe, one per server process). A failure pauses that provider for every visitor:

  | Failure | Pause |
  |---|---|
  | Daily quota (429 naming a per-day limit, or Cloudflare's daily neurons) | 1 hour, then one request checks it again |
  | Per-minute rate limit (429, or Groq's 413 `rate_limit_exceeded`) | The provider's `retry-after`, kept between 5 s and 10 min; 60 s without one |
  | 5xx, timeout, dropped connection | 30 s |
  | 401, 403, 404 (bad key or model) | 1 hour |
  | Other 400s | No pause; that request moves on |

- **Only provider errors fall over.** An error without an HTTP status or a connection problem (a bug, structured output that did not parse) is raised at once: another provider would only cost one more call.
- **One retry layer stays one layer.** When every provider fails, the pool raises `AllProvidersUnavailable`, which `is_transient_llm_error()` does not match, so the llm node's `RetryPolicy` never repeats the pool. Client retries inside the pool are 0 (Gemini, Cloudflare) and 1 (Groq, the last provider; its usual 429 clears in seconds).
- **The daily budget counts real calls.** `usage_limits.DailyCallBudget` is asked before every provider call, fallbacks included (default 150 a day for all visitors).
- **Same interface.** The pool offers `invoke()`, `bind_tools()` and `with_structured_output()`, so `build_tool_agent_graph()` is unchanged; each provider's own model is invoked, so LangGraph token streaming still works.
- **Safe reporting.** The log names the provider, the outcome and the exception type only, never the provider's error text. Visitors see one plain message per outcome (`robustness.describe_error_for_visitor`).

Other public-only choices that go with it: no long-term memory and in-memory conversations (`public_sessions.py`), usage limits (`usage_limits.py`), a Sample Trip that needs no AI (`sample_trip.py`).

## Alternatives considered

| Option | Why not |
|---|---|
| Fallback everywhere, including the CLI and evals | Eval records would mix models silently; ADR-0004's reason still holds |
| LangChain `with_fallbacks()` | No shared pause: every request would try the out-of-quota provider again first |
| A gateway (LiteLLM, OpenRouter) | One more dependency or account; free tiers and keys would move to a third party |
| Retrying the same provider with backoff | Retries spend the same small quota; a daily quota does not clear in seconds |
| A paid tier | The project has no budget |

## Consequences

**Benefits**

- The public demo keeps answering while at least one provider has quota.
- A used-up provider costs one failed call per hour, not one per request.
- The CLI, local page and eval behaviour (and their tests) are unchanged.

**Costs and limitations**

- **Answer quality varies by provider.** A visitor may get a Groq answer after Gemini's quota is gone; providers are not interchangeable (ADR-0004). The header badge lists the providers in the order they are tried ("Free AI: Gemini → Cloudflare → Groq", no model IDs), not the one that answered.
- **In memory only.** The health table and the daily budget reset on restart and are per process; several instances would each keep their own.
- **Pauses are estimates.** Providers reset quotas on their own clocks; the 1-hour quota pause is a re-check, not a reset time.
- **A mid-stream failure** can briefly show partial text before the next provider's answer replaces it.
- **Not live-verified yet.** The fallback paths are tested with fakes; the first deployment should be checked as in DEPLOYMENT.md.

## Future considerations (not implemented)

- Show which provider answered each turn (from the response metadata).
- Persist the daily budget (a free key-value store) if the demo ever runs more than one instance.
- Re-check the order after each provider's free tier or model changes, with the eval suite on each provider.
