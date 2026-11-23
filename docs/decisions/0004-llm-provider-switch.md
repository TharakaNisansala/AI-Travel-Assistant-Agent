# ADR-0004: Switching the LLM provider with `LLM_PROVIDER`

- **Status:** Accepted, implemented (after Stage 10). Gemini is the primary provider; Groq and Cloudflare are alternatives.
- **Date:** added in commits `c4988a1` (Groq) and `792aa66` (Cloudflare); recorded 2026-11-23.
- **Related:** [architecture.md](../architecture.md), [ADR-0002](0002-understand-and-compose-nodes.md).

## Context

The blueprint used Gemini only. The Gemini free tier allows about 5 requests a minute and 20 a day, and one itinerary takes 3–5 calls, so live testing, the eval run and the web page competed for the same small quota. Some live checks were also blocked by Gemini 504 timeouts. A second provider was needed for development and for comparing models.

## Decision

`LLM_PROVIDER` in `.env` (or the shell, which takes priority) picks the provider. Empty means `gemini`, so an old `.env` keeps working. All code gets its model from `llm_basics.build_llm(settings, temperature, max_retries)`:

| Provider | LangChain class | Settings |
|---|---|---|
| `gemini` (default) | `ChatGoogleGenerativeAI` | `GEMINI_API_KEY`, `GEMINI_MODEL` |
| `groq` | `ChatGroq` | `GROQ_API_KEY`, `GROQ_MODEL` |
| `cloudflare` | `ChatOpenAI` with the Workers AI OpenAI-compatible URL | `CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_ACCOUNT_ID`, optional `CLOUDFLARE_MODEL` (default `@cf/zai-org/glm-4.7-flash`) |

Details as built:

- `load_settings()` checks the chosen provider's key and model and raises `ConfigError` naming what is missing. The key is kept out of `repr`.
- The rest of the code depends only on LangChain's `BaseChatModel` interface (`invoke`, `stream`, `bind_tools`, `with_structured_output`).
- **Retries per provider** (`AGENT_CLIENT_RETRIES`): Gemini 0, Groq 6, Cloudflare 0. The graph's `RetryPolicy` on the `llm` node is the one retry layer for 5xx and timeouts. Groq's client retries exist because its usual 429 is a tokens-per-minute limit that clears in seconds.
- **Errors are classified once** for all providers in `robustness.py` (`error_status` reads `.code` or `.status_code`), so the traveller sees the same safe messages. Groq's 413 `rate_limit_exceeded` (one request over the per-minute token limit) counts as a rate limit.
- The web page header shows the provider and model; eval records store the model per scenario, and the scorecard lists every model in a run.
- `check_groq_models.py` lists the Groq models and their context windows.

## Alternatives considered

| Option | Why not (for v1) |
|---|---|
| Gemini only | Quota too small for development plus evaluation |
| A paid Gemini tier | The project runs on free tiers |
| A gateway (LiteLLM, OpenRouter) | One more dependency or account; LangChain's classes already share one interface |
| Automatic fallback between providers | Hides which model produced an answer, which matters for evaluation; adds routing logic |

## Consequences

**Benefits**

- Development and testing continue when the Gemini quota is used up.
- Models can be compared on the same prompts (comparison of 2026-11-18; live eval of 2026-11-23).
- Adding a provider touches one function and the settings table.

**Costs and limitations** (seen live)

- **Providers are not interchangeable.**
  - Groq validates tool-call arguments strictly against the JSON schema; `null` for a list field failed with 400 until the schema allowed it (2026-11-23).
  - Groq `openai/gpt-oss-120b` on the free tier allows 8,000 tokens per minute, and one itinerary request can exceed it (413; Kandy asked for 8,952). In the eval, only 1 of 5 itineraries finished on Groq; the other 4 stopped with an API status error (the one inspected was this 413). Plan scenarios therefore run on Gemini only.
  - In the 2026-11-18 comparison, Cloudflare `glm-4.7-flash` passed 1 of 2 itinerary runs and Groq `gpt-oss-20b` failed.
- **Quality depends on the model:** grounding and tool use differ by provider, so eval results are only comparable per model. Run `2026-11-23T1156` mixes Groq (understanding, routing, one itinerary) and Gemini (the other plans).
- **Leftover Gemini wording:** some messages and help texts (for example the eval runner's output and the CLI description) still say "Gemini" when another provider is active.
- The switch is manual; a quota error does not move to another provider.

## Future considerations (not implemented)

- Provider-neutral wording in the CLI and eval runner messages.
- A per-provider schema check in the tests (for example, strict JSON-schema validation like Groq's) before live runs.
- Optional automatic fallback for the interactive chat, kept off for evaluation.
- A larger-limit model per provider for itineraries, chosen with `check_groq_models.py` and re-evaluated before use.
