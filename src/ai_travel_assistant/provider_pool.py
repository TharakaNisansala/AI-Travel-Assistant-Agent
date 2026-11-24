"""Deployment: one chat model backed by several free LLM providers, in order.

Used only by the public web page (TRAVEL_PUBLIC=1). The CLI and the eval runs
keep one provider (LLM_PROVIDER), so every eval record still names the one
model that produced it (see ADR-0004 and ADR-0005).

What this file demonstrates:
  1. Provider fallback WITHOUT multiplying calls. One request tries each
     configured provider at most once, in LLM_FALLBACK_ORDER (default
     gemini, cloudflare, groq). There is no loop: N providers mean at most N
     calls, and the graph's RetryPolicy never retries the pool's final error.
  2. A shared health table (a simple circuit breaker). A provider that failed
     is skipped for a while by EVERY visitor's requests, so a used-up daily
     quota is not hit again and again:
        daily quota used up -> 1 hour (then one request checks it again)
        per-minute limit    -> the provider's retry-after, else 60 s
        5xx / timeout       -> 30 s
        401/403/404         -> 1 hour (bad key or model: a config problem)
        other 400s          -> no pause (that request was the problem)
  3. Only provider errors fall over. An error with an HTTP status or a
     dropped connection moves to the next provider; anything else (e.g. a bug,
     or structured output that did not parse) is raised at once, because
     another provider would just cost one more call.
  4. Providers without credentials are left out when the pool is built.
  5. An optional daily call budget (usage_limits.DailyCallBudget) is asked
     before EVERY provider call, fallbacks included, so it counts real API
     requests. When it is used up, DailyLimitReached is raised and nothing
     is sent.

Duck-typed like a LangChain chat model for the three things the agent uses:
invoke(), bind_tools() and with_structured_output(). Each provider's own
model does the work, so token streaming (LangGraph's callbacks) still works.

The health table lives in process memory: a restart forgets it, and two
server processes would each keep their own.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from langchain_core.runnables import Runnable, RunnableConfig

from ai_travel_assistant import llm_basics
from ai_travel_assistant.llm_basics import (
    PROVIDERS,
    ConfigError,
    LLMSettings,
    build_llm,
    provider_settings,
)
from ai_travel_assistant.robustness import (
    error_status,
    is_quota_exhausted,
    is_rate_limited,
    is_transient_llm_error,
)

logger = logging.getLogger(__name__)


class CallBudget(Protocol):
    """What the pool needs from usage_limits.DailyCallBudget."""

    def try_consume(self) -> bool: ...


FALLBACK_ORDER_ENV = "LLM_FALLBACK_ORDER"
# Gemini gave the best itineraries in the evals; Cloudflare passed 1 of 2
# itinerary runs (2026-11-18); Groq's free 8,000 tokens a minute stopped 4 of
# 5 itineraries (2026-11-23), so it is the last resort.
DEFAULT_FALLBACK_ORDER: tuple[str, ...] = ("gemini", "cloudflare", "groq")

# Client retries inside the pool. Falling over to the next provider replaces
# most retries. Groq keeps one: its usual 429 (tokens per minute) clears in
# seconds, and it is the last provider, so there is nobody left to fall to.
POOL_CLIENT_RETRIES: dict[str, int] = {"gemini": 0, "cloudflare": 0, "groq": 1}

Outcome = Literal["quota", "rate_limit", "busy", "rejected", "bad_request"]

QUOTA_COOLDOWN_S = 3600.0
RATE_LIMIT_COOLDOWN_S = 60.0
MIN_RATE_LIMIT_COOLDOWN_S = 5.0
MAX_RATE_LIMIT_COOLDOWN_S = 600.0
BUSY_COOLDOWN_S = 30.0
REJECTED_COOLDOWN_S = 3600.0

# Cloudflare names its daily free allowance in "neurons".
_DAILY_HINT = re.compile(r"per\s*day|perday|daily|neurons", re.IGNORECASE)
# Gemini: "Please retry in 34.5s." or "'retryDelay': '34s'".
_RETRY_IN = re.compile(
    r"retry(?:[_ ]?delay)?['\"]?\s*(?:in|[:=])\s*['\"]?(\d+(?:\.\d+)?)\s*s",
    re.IGNORECASE,
)


# --- Classifying a failed call ------------------------------------------------


def classify(exc: BaseException) -> Outcome | None:
    """What a failed call means for its provider; None: not a provider error."""
    if is_rate_limited(exc):
        if is_quota_exhausted(exc) or _DAILY_HINT.search(str(exc)):
            return "quota"
        return "rate_limit"
    if is_transient_llm_error(exc):
        return "busy"
    status = error_status(exc)
    if status in (401, 403, 404):
        return "rejected"
    if status is not None and 400 <= status < 500:
        return "bad_request"
    if status is not None and status >= 500:
        return "busy"
    return None


def retry_after_hint(exc: BaseException) -> float | None:
    """The wait the provider asked for, in seconds, if the error says."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
        try:
            value = headers.get("retry-after")
        except Exception:  # noqa: BLE001 - an odd headers object: ignore it
            value = None
        if value:
            try:
                return max(0.0, float(value))
            except (TypeError, ValueError):
                pass
    match = _RETRY_IN.search(str(exc))
    return float(match.group(1)) if match else None


def cooldown_seconds(kind: Outcome, hint: float | None) -> float:
    """How long to skip a provider after this kind of failure."""
    if kind == "quota":
        return max(QUOTA_COOLDOWN_S, hint or 0.0)
    if kind == "rate_limit":
        wait = RATE_LIMIT_COOLDOWN_S if hint is None else hint
        return min(max(wait, MIN_RATE_LIMIT_COOLDOWN_S), MAX_RATE_LIMIT_COOLDOWN_S)
    if kind == "busy":
        return BUSY_COOLDOWN_S
    if kind == "rejected":
        return REJECTED_COOLDOWN_S
    return 0.0


# --- The shared health table --------------------------------------------------


class ProviderHealth:
    """Which providers are paused, and why. Thread-safe: shared by all visitors."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._paused: dict[str, tuple[float, Outcome]] = {}  # until, reason

    def paused_reason(self, provider: str) -> Outcome | None:
        """Why the provider is paused, or None when it may be called."""
        with self._lock:
            entry = self._paused.get(provider)
            if entry is None:
                return None
            until, reason = entry
            if self._clock() >= until:
                del self._paused[provider]
                return None
            return reason

    def record_failure(
        self, provider: str, kind: Outcome, hint: float | None = None
    ) -> float:
        """Pause the provider for this kind of failure; returns the pause."""
        seconds = cooldown_seconds(kind, hint)
        if seconds > 0:
            with self._lock:
                self._paused[provider] = (self._clock() + seconds, kind)
        return seconds

    def seconds_until_available(self, providers: Sequence[str]) -> float | None:
        """Time until the first of these providers may be called again."""
        now = self._clock()
        with self._lock:
            waits = []
            for provider in providers:
                entry = self._paused.get(provider)
                waits.append(0.0 if entry is None else max(0.0, entry[0] - now))
        return min(waits) if waits else None

    def snapshot(self) -> dict[str, tuple[Outcome, float]]:
        """{provider: (reason, seconds left)} for the paused providers."""
        now = self._clock()
        with self._lock:
            return {
                name: (reason, until - now)
                for name, (until, reason) in self._paused.items()
                if until > now
            }


class AllProvidersUnavailable(Exception):
    """No provider in the pool could answer this request.

    Carries only provider names and outcome kinds, never a provider's error
    text, so it is safe to log. The last provider error is its __cause__.
    """

    def __init__(
        self,
        outcomes: Sequence[tuple[str, Outcome]],
        retry_after_s: float | None = None,
    ) -> None:
        self.outcomes = tuple(outcomes)
        self.retry_after_s = retry_after_s
        summary = ", ".join(f"{name}={kind}" for name, kind in self.outcomes)
        super().__init__(f"No AI provider could answer ({summary or 'none'})")

    @property
    def kind(self) -> Outcome:
        """The one outcome to tell the traveller about."""
        kinds = {kind for _, kind in self.outcomes}
        if kinds and kinds == {"quota"}:
            return "quota"
        if "busy" in kinds:
            return "busy"
        if "rate_limit" in kinds:
            return "rate_limit"
        if "quota" in kinds:
            return "quota"
        return "rejected"


# --- The pool -----------------------------------------------------------------


class PoolRunnable(Runnable[Any, Any]):
    """Try each provider's runnable once, in order, skipping paused ones."""

    def __init__(
        self,
        members: Sequence[tuple[str, Runnable]],
        health: ProviderHealth,
        budget: CallBudget | None = None,
    ) -> None:
        if not members:
            raise ValueError("a provider pool needs at least one provider")
        self.members = list(members)
        self.health = health
        self.budget = budget

    @property
    def providers(self) -> list[str]:
        return [name for name, _ in self.members]

    def invoke(
        self, input: Any, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Any:
        outcomes: list[tuple[str, Outcome]] = []
        last_error: BaseException | None = None
        for name, runnable in self.members:
            reason = self.health.paused_reason(name)
            if reason is not None:
                outcomes.append((name, reason))  # skipped: no API call
                continue
            if self.budget is not None and not self.budget.try_consume():
                from ai_travel_assistant.usage_limits import DailyLimitReached

                raise DailyLimitReached()  # not a provider error: no fallback
            try:
                return runnable.invoke(input, config, **kwargs)
            except Exception as exc:
                kind = classify(exc)
                if kind is None:
                    raise  # not the provider's fault: another would not help
                paused = self.health.record_failure(name, kind, retry_after_hint(exc))
                # Names and kinds only: provider errors can echo request data.
                logger.warning(
                    "LLM provider %s failed (%s, %s); paused %.0f s",
                    name,
                    kind,
                    type(exc).__name__,
                    paused,
                )
                outcomes.append((name, kind))
                last_error = exc
        raise AllProvidersUnavailable(
            outcomes, self.health.seconds_until_available(self.providers)
        ) from last_error


class ProviderPool(PoolRunnable):
    """A chat model made of several providers' models (see the module doc)."""

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> PoolRunnable:
        return PoolRunnable(
            [(name, model.bind_tools(tools, **kwargs)) for name, model in self.members],
            self.health,
            self.budget,
        )

    def with_structured_output(self, schema: Any, **kwargs: Any) -> PoolRunnable:
        return PoolRunnable(
            [
                (name, model.with_structured_output(schema, **kwargs))
                for name, model in self.members
            ],
            self.health,
            self.budget,
        )


# --- Settings -----------------------------------------------------------------


@dataclass(frozen=True)
class PoolSettings:
    """The configured providers, in fallback order (keys hidden by LLMSettings)."""

    members: tuple[LLMSettings, ...]
    # Providers named in the order but missing a key or model: left out.
    skipped: tuple[str, ...] = ()

    @property
    def providers(self) -> list[str]:
        return [member.provider for member in self.members]

    @property
    def primary(self) -> LLMSettings:
        return self.members[0]


def fallback_order(raw: str | None) -> tuple[str, ...]:
    """Parse LLM_FALLBACK_ORDER ("gemini, cloudflare, groq"); empty: the default."""
    names: list[str] = []
    for part in (raw or "").split(","):
        name = part.strip().lower()
        if not name:
            continue
        if name not in PROVIDERS:
            raise ConfigError(
                f"Unknown provider '{name}' in {FALLBACK_ORDER_ENV}. "
                f"Use: {', '.join(PROVIDERS)}."
            )
        if name not in names:
            names.append(name)
    return tuple(names) or DEFAULT_FALLBACK_ORDER


def load_pool_settings() -> PoolSettings:
    """Every provider in LLM_FALLBACK_ORDER that has its key and model set."""
    llm_basics.load_dotenv()  # via the module, so tests can switch it off
    members: list[LLMSettings] = []
    skipped: list[str] = []
    for provider in fallback_order(os.getenv(FALLBACK_ORDER_ENV)):
        settings, _missing = provider_settings(provider)
        if settings is None:
            skipped.append(provider)
        else:
            members.append(settings)
    if not members:
        raise ConfigError(
            "No AI provider is configured. Set the API key and model of at least "
            f"one of: {', '.join(PROVIDERS)} (see .env.example)."
        )
    return PoolSettings(members=tuple(members), skipped=tuple(skipped))


def build_provider_pool(
    settings: PoolSettings,
    temperature: float,
    health: ProviderHealth,
    budget: CallBudget | None = None,
) -> ProviderPool:
    """One model per configured provider, sharing one health table (and budget).

    Building does not call any API.
    """
    return ProviderPool(
        [
            (
                member.provider,
                build_llm(
                    member,
                    temperature=temperature,
                    max_retries=POOL_CLIENT_RETRIES.get(member.provider, 0),
                ),
            )
            for member in settings.members
        ],
        health,
        budget,
    )
