"""Deployment: keeping public visitors apart, and cleaning up after them.

Used only by the public web page (TRAVEL_PUBLIC=1). The CLI and the local
web page keep their SQLite conversations and saved preferences.

What this file demonstrates:
  1. Isolation by unguessable ids. Each browser session gets its own random
     thread id (the conversation) and visitor id. Nothing on the public page
     lets a visitor type or choose an id, so the only conversation a visitor
     can reach is the one their own session holds.
  2. No long-term memory in public. Saved preferences are keyed by a user id;
     on a shared page that is how one visitor would see or delete another's.
     The public graph is built without a store, so nothing is saved or read
     (and the preference extractor's LLM call is saved too).
  3. Conversations in memory, cleaned up. A visitor's thread id lives only
     in their browser session; after a reload it cannot be reached again. So
     public threads are kept in memory (InMemorySaver), and ThreadJanitor
     deletes those idle for TRAVEL_THREAD_IDLE_MINUTES (default 120), the
     oldest ones above MAX_PUBLIC_THREADS, and the previous one when a
     visitor starts a new conversation.

The janitor only deletes threads it was told about (created by public
visitors in this process), never any other thread in the checkpointer.
A restart forgets everything, which is fine for a demo.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from collections.abc import Callable

from ai_travel_assistant.llm_basics import ConfigError

logger = logging.getLogger(__name__)

THREAD_IDLE_ENV = "TRAVEL_THREAD_IDLE_MINUTES"
DEFAULT_THREAD_IDLE_MINUTES = 120
# A hard cap, so a burst of visitors cannot fill the server's memory.
MAX_PUBLIC_THREADS = 500
# Sweeping looks at every tracked thread; doing it once every few minutes is
# plenty and keeps it off the path of most chat turns.
SWEEP_EVERY_S = 300.0

# What /prefs, /forget and the old preference controls say in public mode.
PREFERENCES_OFF = (
    "Saved preferences are turned off in this public demo, so nothing you "
    "type is kept after you leave."
)
NEW_CONVERSATION = "Started a new conversation."
PUBLIC_QUIT_NOTE = (
    "You can simply close the tab when you are done. This demo does not keep "
    "conversations after you leave. Type /new to start another one."
)


def new_visitor_id() -> str:
    """A random id for one browser session (never shown, never shared)."""
    return f"visitor-{uuid.uuid4().hex}"


def thread_idle_seconds() -> float:
    """TRAVEL_THREAD_IDLE_MINUTES in seconds; empty means the default."""
    raw = os.getenv(THREAD_IDLE_ENV, "").strip()
    if not raw:
        return DEFAULT_THREAD_IDLE_MINUTES * 60.0
    try:
        minutes = int(raw)
    except ValueError:
        minutes = 0
    if minutes < 1:
        raise ConfigError(f"{THREAD_IDLE_ENV} must be a whole number of 1 or more.")
    return minutes * 60.0


DeleteThread = Callable[[str], None]


class ThreadJanitor:
    """Tracks public threads by last use and deletes the stale ones. Thread-safe."""

    def __init__(
        self,
        idle_s: float,
        max_threads: int = MAX_PUBLIC_THREADS,
        sweep_every_s: float = SWEEP_EVERY_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.idle_s = idle_s
        self.max_threads = max_threads
        self.sweep_every_s = sweep_every_s
        self._clock = clock
        self._lock = threading.Lock()
        self._last_used: dict[str, float] = {}
        self._last_sweep = clock()

    def touch(self, thread_id: str) -> None:
        """Record that a visitor used this thread now."""
        with self._lock:
            self._last_used[thread_id] = self._clock()

    def tracked(self) -> int:
        with self._lock:
            return len(self._last_used)

    def release(self, thread_id: str, delete: DeleteThread) -> bool:
        """Delete a thread its visitor left (e.g. "New conversation").

        Only a tracked thread is deleted; returns whether it was.
        """
        with self._lock:
            if self._last_used.pop(thread_id, None) is None:
                return False
        _delete(delete, thread_id)
        return True

    def sweep(self, delete: DeleteThread, force: bool = False) -> list[str]:
        """Delete threads idle too long, then the oldest above the cap.

        Runs at most once every sweep_every_s unless force=True. Returns the
        deleted thread ids.
        """
        now = self._clock()
        with self._lock:
            if not force and now - self._last_sweep < self.sweep_every_s:
                return []
            self._last_sweep = now
            stale = [
                t for t, used in self._last_used.items() if now - used > self.idle_s
            ]
            for thread_id in stale:
                del self._last_used[thread_id]
            extra = len(self._last_used) - self.max_threads
            if extra > 0:
                oldest = sorted(self._last_used, key=self._last_used.__getitem__)
                for thread_id in oldest[:extra]:
                    del self._last_used[thread_id]
                    stale.append(thread_id)
        for thread_id in stale:  # outside the lock: deleting can take a moment
            _delete(delete, thread_id)
        return stale


def _delete(delete: DeleteThread, thread_id: str) -> None:
    try:
        delete(thread_id)
    except Exception as exc:  # noqa: BLE001 - cleanup must never break a turn
        logger.warning("Could not delete a public thread (%s)", type(exc).__name__)


# --- One janitor per server process ----------------------------------------------

_shared_janitor: ThreadJanitor | None = None
_shared_lock = threading.Lock()


def shared_janitor() -> ThreadJanitor:
    """The process-wide janitor (ConfigError if the idle time is invalid)."""
    global _shared_janitor
    with _shared_lock:
        if _shared_janitor is None:
            _shared_janitor = ThreadJanitor(thread_idle_seconds())
        return _shared_janitor


def reset_shared_janitor() -> None:
    """Forget the process-wide janitor (tests)."""
    global _shared_janitor
    with _shared_lock:
        _shared_janitor = None
