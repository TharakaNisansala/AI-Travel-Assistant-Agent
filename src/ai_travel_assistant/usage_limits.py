"""Deployment: usage limits that protect the free LLM quotas (TRAVEL_PUBLIC=1).

What this file demonstrates:
  1. Checking BEFORE spending. A message that is too long, a visitor over
     their message limit, or a used-up daily budget is refused before any
     LLM call is made.
  2. Counting real calls, not chat turns. One turn makes about 3-6 LLM calls
     (understanding, the tool loop, the wrap-up), and a provider fallback adds
     more. The provider pool asks DailyCallBudget for permission before EVERY
     provider call, so the daily number is the true number of API requests.
  3. A cap on simultaneous requests (a semaphore). Free tiers have per-minute
     limits, and a small server has little CPU and memory: two turns at once
     is plenty. A third visitor waits a few seconds, then gets a "busy" note.
  4. Limits from the environment. Every limit has a TRAVEL_... variable. On
     Streamlit Community Cloud, top-level Secrets are environment variables
     too, so the same names work there.

Limitations (documented in docs/DEPLOYMENT.md):
  - Every counter lives in process memory. A restart or redeploy resets the
    daily budget, and two server processes would each count separately.
  - The per-visitor limit is per browser session (st.session_state). A page
    reload starts a new session. The daily budget and the concurrency cap
    are the real protection; the session limit only keeps one visitor from
    using the whole budget in one sitting.
  - The day is the UTC day (it resets at 05:30 Sri Lanka time). Providers
    reset their own quotas on their own clocks.

The CLI and the eval runs never use this module.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timezone

from ai_travel_assistant.context_window import MAX_HISTORY_TOKENS
from ai_travel_assistant.llm_basics import ConfigError

# Environment variables (or top-level Streamlit Secrets) and their defaults.
LIMIT_VARIABLES: dict[str, tuple[str, int]] = {
    # field: (variable, default)
    "max_session_messages": ("TRAVEL_MAX_MESSAGES_PER_SESSION", 15),
    "max_daily_llm_calls": ("TRAVEL_MAX_DAILY_LLM_CALLS", 150),
    "max_concurrent_requests": ("TRAVEL_MAX_CONCURRENT_REQUESTS", 2),
    "max_message_chars": ("TRAVEL_MAX_MESSAGE_CHARS", 1000),
    "max_history_tokens": ("TRAVEL_MAX_HISTORY_TOKENS", MAX_HISTORY_TOKENS),
}

# How long a turn waits for a free request slot before giving up.
SLOT_WAIT_S = 10.0


# --- Messages for the visitor (no internal details) -----------------------------


def too_long_message(length: int, limit: int) -> str:
    return (
        f"Your message is {length:,} characters long; the limit is {limit:,}. "
        "Please shorten it and send it again."
    )


def session_limit_message(limit: int) -> str:
    return (
        f"You have reached this demo's limit of {limit} messages per visit. "
        "It keeps the free AI quota fair for everyone. Please come back later."
    )


DAILY_LIMIT_MESSAGE = (
    "This demo has used its free AI budget for today. It resets at 00:00 UTC "
    "(05:30 Sri Lanka time). Please come back then."
)
BUSY_MESSAGE = (
    "Other trips are being planned right now. Please try again in a few seconds."
)


class UsageLimitReached(Exception):
    """A limit stopped the request. str(exc) is safe to show the visitor."""


class DailyLimitReached(UsageLimitReached):
    """The day's LLM call budget is used up."""

    def __init__(self) -> None:
        super().__init__(DAILY_LIMIT_MESSAGE)


class ServerBusy(UsageLimitReached):
    """Every request slot stayed taken for SLOT_WAIT_S seconds."""

    def __init__(self) -> None:
        super().__init__(BUSY_MESSAGE)


# --- Settings -----------------------------------------------------------------


@dataclass(frozen=True)
class UsageLimits:
    max_session_messages: int = 15
    max_daily_llm_calls: int = 150
    max_concurrent_requests: int = 2
    max_message_chars: int = 1000
    max_history_tokens: int = MAX_HISTORY_TOKENS

    @classmethod
    def from_env(cls) -> UsageLimits:
        """Read every limit; an empty variable means the default."""
        values = {
            name: _positive_int(variable, default)
            for name, (variable, default) in LIMIT_VARIABLES.items()
        }
        return cls(**values)


def _positive_int(variable: str, default: int) -> int:
    raw = os.getenv(variable, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value < 1:
        raise ConfigError(f"{variable} must be a whole number of 1 or more.")
    return value


# --- The daily call budget ------------------------------------------------------


def utc_today() -> date:
    return datetime.now(timezone.utc).date()


class DailyCallBudget:
    """Counts LLM API calls per UTC day, for all visitors. Thread-safe."""

    def __init__(self, limit: int, today: Callable[[], date] = utc_today) -> None:
        self.limit = limit
        self._today = today
        self._lock = threading.Lock()
        self._day = today()
        self._used = 0

    def _roll_over(self) -> None:  # call with the lock held
        day = self._today()
        if day != self._day:
            self._day, self._used = day, 0

    def try_consume(self) -> bool:
        """Count one call if the budget allows it; False when used up."""
        with self._lock:
            self._roll_over()
            if self._used >= self.limit:
                return False
            self._used += 1
            return True

    def remaining(self) -> int:
        with self._lock:
            self._roll_over()
            return max(0, self.limit - self._used)

    def used(self) -> int:
        with self._lock:
            self._roll_over()
            return self._used


# --- The guard the web page uses -------------------------------------------------


class UsageGuard:
    """All public limits in one object, shared by every visitor of a process."""

    def __init__(
        self,
        limits: UsageLimits,
        budget: DailyCallBudget | None = None,
        slot_wait_s: float = SLOT_WAIT_S,
    ) -> None:
        self.limits = limits
        self.budget = budget or DailyCallBudget(limits.max_daily_llm_calls)
        self.slot_wait_s = slot_wait_s
        self._slots = threading.BoundedSemaphore(limits.max_concurrent_requests)

    def refusal(self, text: str, messages_sent: int) -> str | None:
        """Why this message may not go to the agent, or None if it may.

        messages_sent: how many messages this visitor's session already sent.
        Makes no LLM call and changes no counter.
        """
        length = len(text)
        if length > self.limits.max_message_chars:
            return too_long_message(length, self.limits.max_message_chars)
        if messages_sent >= self.limits.max_session_messages:
            return session_limit_message(self.limits.max_session_messages)
        if self.budget.remaining() < 1:
            return DAILY_LIMIT_MESSAGE
        return None

    @contextmanager
    def request_slot(self) -> Iterator[None]:
        """Hold one of the simultaneous-request slots, or raise ServerBusy."""
        if not self._slots.acquire(timeout=self.slot_wait_s):
            raise ServerBusy()
        try:
            yield
        finally:
            self._slots.release()


# --- One guard per server process ----------------------------------------------

_shared_guard: UsageGuard | None = None
_shared_lock = threading.Lock()


def shared_guard() -> UsageGuard:
    """The process-wide guard, made from the environment on first use.

    A module-level object (not st.cache_resource) so every visitor's script
    run, and the tests, see the same counters. ConfigError if a limit
    variable is invalid; nothing is kept then, so a fixed value is read again.
    """
    global _shared_guard
    with _shared_lock:
        if _shared_guard is None:
            _shared_guard = UsageGuard(UsageLimits.from_env())
        return _shared_guard


def reset_shared_guard() -> None:
    """Forget the process-wide guard (tests; a new one is read on next use)."""
    global _shared_guard
    with _shared_lock:
        _shared_guard = None
