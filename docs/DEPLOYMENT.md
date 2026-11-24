# Deploying the public demo (Streamlit Community Cloud)

This guide puts the Sri Lanka AI Travel Assistant online as a **free public portfolio demo**: anyone with the link can use it without signing in. It runs on Streamlit Community Cloud (free hosting) and free AI API tiers (Gemini, Cloudflare Workers AI, Groq).

What to expect from a free demo:

- **Limited, not unlimited.** Free AI quotas are small. The app caps usage (15 messages per visitor, 150 AI calls a day in total) and moves to another provider when one runs out, but on a busy day the live AI can be used up. Visitors can still open the **Sample Trip**, which needs no AI.
- **Best effort, not production-grade.** Free hosting can be slow to start and may put the app to sleep when nobody uses it. Providers can change their free tiers at any time.

Local use, the CLI and the evals are not affected by anything here. The public behaviour is switched on only by `TRAVEL_PUBLIC = "1"` in the app's Secrets.

---

## 1. Before you start

You need:

- A GitHub account with **admin access** to the repository (Community Cloud requires it).
- A Streamlit Community Cloud account: sign in at <https://share.streamlit.io> with GitHub.
- At least one AI provider key (more is better: each one is a fallback):

| Provider | Where to get the key | Variables |
|---|---|---|
| Gemini | <https://aistudio.google.com/> | `GEMINI_API_KEY`, `GEMINI_MODEL` |
| Cloudflare Workers AI | Cloudflare dashboard: an API token with Workers AI permission, and your account ID | `CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_ACCOUNT_ID`, optional `CLOUDFLARE_MODEL` |
| Groq | <https://console.groq.com/keys> | `GROQ_API_KEY`, `GROQ_MODEL` |

Use the model IDs that already work in your local `.env`. Gemini and Groq model IDs change often, so the app has no default for them; Cloudflare defaults to `@cf/zai-org/glm-4.7-flash`.

## 2. Prepare the repository and branch

The deployment files are on the branch `feat/public-deployment`:

| File | Purpose |
|---|---|
| `deploy/streamlit_app.py` | The entry point Community Cloud runs |
| `deploy/requirements.txt` | Exact package versions for the server (from `uv.lock`) |
| `.streamlit/config.toml` | The page theme (stays at the repository root) |
| `.streamlit/secrets.toml.example` | A template for the Secrets box (placeholders only) |

1. Review the branch and open a pull request into `main` on GitHub. Merge it when you are happy (deploy from `main`, so the public app only changes when `main` does). You can also deploy the branch itself to try it first.
2. Check that no secret is in the repository:
   ```
   git ls-files | findstr /i "secrets.toml .env"
   ```
   The only results must be `.env.example` and `.streamlit/secrets.toml.example`. `.env` and `.streamlit/secrets.toml` are git-ignored.

**Why a `deploy/` folder?** Community Cloud installs from the **first** dependency file it finds, looking in the entry point's folder before the repository root, and it prefers `uv.lock` over `requirements.txt`. With the entry point in `deploy/`, `deploy/requirements.txt` is used and the root `uv.lock` (for local development) is left alone. The entry point adds `src/` to Python's import path, so the project itself does not need to be installed.

## 3. Create the app

1. Go to <https://share.streamlit.io> and click **Create app**, then choose to deploy from a GitHub repository.
2. Fill in:
   - **Repository:** `TharakaNisansala/AI-Travel-Assistant-Agent`
   - **Branch:** `main` (or `feat/public-deployment` to try it first)
   - **Main file path:** `deploy/streamlit_app.py`
   - **App URL:** optional, e.g. a short name like `srilanka-travel-assistant`
3. Click **Advanced settings**:
   - **Python version:** `3.12`
   - **Secrets:** paste your secrets (step 4)
4. Click **Deploy**. The first build installs about 100 packages and takes a few minutes. Watch the log in the lower-right panel ("Manage app").

Choose the Python version now: the docs only describe choosing it when you deploy. To change it later you may have to delete the app and deploy it again (you can reuse the same URL).

## 4. Secrets

Open `.streamlit/secrets.toml.example`, copy all of it into the **Secrets** box, and fill in your values. The full list:

| Name | Required | Value | Meaning |
|---|---|---|---|
| `TRAVEL_PUBLIC` | **Yes** | `"1"` | Turns on the public mode |
| `LLM_FALLBACK_ORDER` | No | `"gemini,cloudflare,groq"` | Which providers to try, in order |
| `GEMINI_API_KEY`, `GEMINI_MODEL` | One provider at least | your key, your model ID | Gemini |
| `CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_ACCOUNT_ID` | (as above) | your token, your account ID | Cloudflare Workers AI |
| `CLOUDFLARE_MODEL` | No | empty = `@cf/zai-org/glm-4.7-flash` | Cloudflare model |
| `GROQ_API_KEY`, `GROQ_MODEL` | (as above) | your key, your model ID | Groq |
| `TRAVEL_MAX_MESSAGES_PER_SESSION` | No | default `15` | Messages one visitor can send per visit |
| `TRAVEL_MAX_DAILY_LLM_CALLS` | No | default `150` | AI calls per day, all visitors together (UTC day) |
| `TRAVEL_MAX_CONCURRENT_REQUESTS` | No | default `2` | AI requests running at the same time |
| `TRAVEL_MAX_MESSAGE_CHARS` | No | default `1000` | Longest message a visitor can send |
| `TRAVEL_MAX_HISTORY_TOKENS` | No | default `8000` | Earlier conversation sent to the AI with each call |
| `TRAVEL_THREAD_IDLE_MINUTES` | No | default `120` | Idle conversations are deleted after this |

Three rules, or the app will not see the values:

1. **Top level only.** No `[section]` headers. Streamlit copies top-level secrets into environment variables, which is what the app reads.
2. **Quote every value:** `TRAVEL_PUBLIC = "1"`, `TRAVEL_MAX_DAILY_LLM_CALLS = "100"`. Streamlit does not copy `true`/`false` into environment variables.
3. **Empty means unset:** a provider with an empty key or model is skipped; an empty limit uses its default. Limits must be whole numbers of 1 or more.

Never add `TRAVEL_DEV` to the public app (the public page ignores it anyway).

### Choosing the providers

- **Order.** The default, Gemini → Cloudflare → Groq, follows this project's own tests: Gemini gave the best itineraries; Cloudflare passed 1 of 2 itinerary runs (2026-11-18); Groq's free tier (8,000 tokens a minute on `gpt-oss`) stopped 4 of 5 itineraries (2026-11-23), so it is the last resort.
- **One provider only:** set `LLM_FALLBACK_ORDER = "gemini"` (or leave the other keys empty).
- **How fallback works.** Each AI call tries each configured provider at most once, in order. A provider that fails is paused for everyone: 1 hour after a used-up daily quota, the provider's own wait (5 s to 10 min) after a rate limit, 30 s after a server error or timeout, and 1 hour after a rejected key or model. Details: [ADR-0005](decisions/0005-public-provider-fallback.md).
- **Quotas.** In November 2026 this project saw about 5 requests a minute and 20 a day on Gemini's free tier, and one itinerary uses 3-5 AI calls. Free tiers differ by account and change over time: check each provider's console.

### Changing the limits

Edit the Secrets (app menu **⋮** > **Settings** > **Secrets**), save, and reboot the app (**⋮** > **Reboot app**) so the new values are read. A reboot also resets the day's counters. Example: a stricter demo
```toml
TRAVEL_MAX_MESSAGES_PER_SESSION = "8"
TRAVEL_MAX_DAILY_LLM_CALLS = "60"
```

## 5. Check the deployed app

1. Open the app URL. The header badge should list your configured providers in the order they are tried, e.g. "Free AI: Gemini → Cloudflare → Groq" (no model IDs). No header badge means no provider is configured (see Troubleshooting).
2. Click **View a sample trip (no AI)** in the sidebar: a "Sample — not live AI" plan for Kandy appears, with day tabs and **Download sample plan (PDF)**. This uses no AI quota.
3. Go back and ask one short question, e.g. "What is the weather like in Ella?" (2-4 AI calls). The sidebar must not show a thread ID or a "Traveller" field.
4. Open the URL in a private window: it starts its own, empty conversation.
5. Open **Manage app** and read the log: provider pauses appear as `LLM provider gemini failed (quota, ...)`, never with a key or the provider's message.

Avoid long test sessions: they use the same daily quota as your visitors.

## 6. Troubleshooting

| What you see | Likely cause | What to do |
|---|---|---|
| Build log shows `uv.lock` being used, or `No module named 'streamlit'` | Main file path is not `deploy/streamlit_app.py` | Fix the main file path (delete and redeploy if it cannot be edited) |
| `ModuleNotFoundError: No module named 'ai_travel_assistant'` | The entry point is not `deploy/streamlit_app.py` | As above |
| Package installation fails | Wrong Python version, or `deploy/requirements.txt` out of date | Choose Python 3.12; regenerate the file (section 8) and push |
| "Live AI is not available on this demo right now." and a sample offer | No provider configured, or a limit value is invalid | Read the log line `Public page cannot start live AI: ...`; it names the missing or bad setting. Fix the Secrets, save, reboot |
| The badge shows fewer providers than you set (e.g. only "Free AI: Gemini") | The missing providers lack a key or a model | Add the missing provider's key and model |
| Header and page look like local mode (Traveller field, thread ID) | `TRAVEL_PUBLIC` missing or not `"1"` | Add `TRAVEL_PUBLIC = "1"` (quoted, top level), reboot |
| "This demo has used up today's free AI allowance." | Every provider's daily quota is used up | Wait (Gemini resets at about 13:30 Sri Lanka time), or add another provider |
| "This demo has used its free AI budget for today." | `TRAVEL_MAX_DAILY_LLM_CALLS` reached | Resets at 00:00 UTC (05:30 Sri Lanka time), or after a reboot |
| "The free AI service is getting too many requests…" | Per-minute rate limits | Wait as the message says |
| "Live AI is temporarily unavailable on this demo." | A key or model ID was rejected (401/403/404) | Check the key and model ID in the Secrets; the log names the provider |
| "Other trips are being planned right now…" | 2 requests already running | Normal under load; raise `TRAVEL_MAX_CONCURRENT_REQUESTS` carefully |
| The plain theme instead of the teal one | `.streamlit/config.toml` missing at the root | It must stay at the repository root |
| The app is asleep | No visitors for a while | Open it and click the wake-up button; it restarts in a minute or two |

## 7. Known limitations

- **Counters are in memory.** The daily AI budget, the provider pauses and the conversations reset when the app restarts (a reboot, a redeploy, or after sleeping).
- **One instance, one set of counters.** If the app ever ran on more than one server process, each would count separately.
- **Conversations are not kept.** A conversation lives only in its browser tab: a reload starts a new one, idle ones are deleted after 120 minutes, and there are no saved preferences in the public mode.
- **The per-visitor limit is per visit.** Reloading the page starts a new visit; the daily budget and the 2-request cap are the real protection.
- **The Sample Trip needs no AI.** It always works, even with no provider configured, and is clearly labelled "Sample — not live AI". Its prices and times are from saved Wikivoyage text (2017-2024) and may be out of date.
- **Free quotas are limited and can change.** Provider limits depend on the provider and your account, and answers differ by provider (a visitor may get a Groq answer after Gemini's quota is used up).
- **Live tool data** (weather, routes, guides, exchange rates) comes from free public APIs, cached for 15 minutes to 24 hours; they can be slow or down, and the agent then says what it could not get.

## 8. Updating and redeploying

- **Code changes:** push to the deployed branch (`main`). Community Cloud notices the push and redeploys within a few minutes.
- **Dependency changes:** after changing `pyproject.toml` or `uv.lock`, regenerate the pinned list, put its header back, and run the tests (`tests/test_deployment.py` fails if the two disagree):
  ```
  uv export --extra ui --no-dev --no-hashes --no-emit-project --format requirements-txt --output-file deploy/requirements.txt
  ```
- **Secrets or limits:** edit them in the app settings and reboot.
- **Rehearse locally first** (optional, uses no AI if you only open the sample):
  ```
  uv venv --python 3.12 ..\travel-deploy-venv
  uv pip install --python ..\travel-deploy-venv -r deploy\requirements.txt
  copy .streamlit\secrets.toml.example .streamlit\secrets.toml   (then fill it in; it is git-ignored)
  ..\travel-deploy-venv\Scripts\streamlit run deploy\streamlit_app.py
  ```
  Your local `.env` is read too; values in `secrets.toml` take priority.

## 9. Security checklist

- API keys live **only** in the Community Cloud Secrets box (and, if you rehearse, in the git-ignored `.env` / `.streamlit/secrets.toml`). Never commit them, paste them into an issue, or put them in a screenshot.
- If a key is ever exposed, revoke it in the provider's console at once and create a new one; deleting the commit is not enough.
- Visitors never see keys, provider error text, setting names or tracebacks. Community Cloud also hides Streamlit's error details (`client.showErrorDetails` is forced off there), and the public page replaces any unexpected error with a plain message.
- The server log records provider names, outcomes and exception types, never keys or provider error text. Configuration problems name the setting, never its value.
- Visitors cannot reach each other's conversations, and the public mode keeps no preferences.
- The app makes requests only to the AI providers and the free public data APIs; it stores no visitor data on disk in the public mode.
