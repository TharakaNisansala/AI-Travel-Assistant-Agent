"""Stage 5: long-term memory, traveller preferences kept across conversations.

What this file demonstrates:
  1. The LangGraph Store: a key-value database with namespaces, shared by
     every thread. Here one namespace per traveller:
         ("users", <user_id>, "preferences")  ->  {key: {"text": "vegetarian"}}
  2. Deciding what is worth remembering. Only DURABLE preferences (diet,
     mobility, pace, likes and dislikes) are stored; one-off trip details
     (destination, days, dates) stay in the thread's messages.
  3. A cheap gate before an LLM call. looks_like_preference() is a regex that
     spots preference-like phrases ("I'm vegetarian", "I hate early starts").
     Only when it matches does the extractor spend a Gemini call, so most
     turns cost nothing extra on the 20-calls-a-day free tier.

Thread memory (checkpointer, by thread_id) vs user memory (Store, by user_id):
the checkpointer holds the whole conversation; the Store holds a few short
facts that should shape EVERY future conversation of the same traveller.

No LangGraph graph here; tool_calling.py adds the load_memory and save_memory
nodes that use these helpers.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.store.base import BaseStore
from langgraph.store.sqlite import SqliteStore
from pydantic import BaseModel, Field

# --- The gate: does this message look like it states a preference? ------------

# Phrases people use to state lasting preferences. Plain "I'm" or "I want" is
# left out on purpose: "I'm planning a trip" or "I want 3 days" are trip details,
# and every false match costs a Gemini call.
PREFERENCE_HINT = re.compile(
    r"\b(?:"
    r"i\s+(?:prefer|like|love|enjoy|hate|dislike|avoid|always|never|usually|"
    r"can'?t|cannot|don'?t|do\s+not)\b"
    r"|i'?m\s+(?:a\s+|an\s+)?(?:vegetarian|vegan|allergic|afraid|scared|"
    r"not\s+(?:a\s+)?fan|travell?ing\s+with|on\s+a\s+(?:tight\s+)?budget)"
    r"|i\s+am\s+(?:a\s+|an\s+)?(?:vegetarian|vegan|allergic|afraid|scared|"
    r"not\s+(?:a\s+)?fan|travell?ing\s+with|on\s+a\s+(?:tight\s+)?budget)"
    r"|allerg(?:y|ic)|vegetarian|vegan|halal|wheelchair|mobility"
    r"|remember\s+that"
    r")",
    re.IGNORECASE,
)


def looks_like_preference(text: str) -> bool:
    """True if the message may state a lasting preference (cheap, no LLM)."""
    return bool(PREFERENCE_HINT.search(text))


# --- The extractor: which durable preferences does the message state? ---------


class TravellerPreferences(BaseModel):
    """What Gemini returns: the durable preferences found in one message."""

    preferences: list[str] = Field(
        default_factory=list,
        description=(
            "Short, lasting traveller preferences stated in the message, e.g. "
            "'vegetarian', 'avoids long drives', 'prefers late starts'. Empty "
            "if the message states none."
        ),
    )


EXTRACTION_INSTRUCTIONS = (
    "You keep a traveller's long-term profile. From the traveller's message, "
    "extract only preferences that will still matter on FUTURE trips: diet and "
    "allergies, mobility or health limits, travel pace, who they usually travel "
    "with, budget style, and lasting likes or dislikes. Never extract details of "
    "the current trip (destination, dates, number of days, this trip's budget). "
    "Write each as a short phrase without 'I', such as 'vegetarian' or 'avoids "
    "long drives'. Skip preferences already known. Return an empty list when "
    "the message states no new lasting preference."
)

# (message text, preferences already known) -> new preferences
PreferenceExtractor = Callable[[str, Sequence[str]], list[str]]


def make_llm_preference_extractor(llm: BaseChatModel) -> PreferenceExtractor:
    """An extractor that asks the LLM (one structured-output call)."""
    structured = llm.with_structured_output(TravellerPreferences)

    def extract(text: str, known: Sequence[str]) -> list[str]:
        known_line = "; ".join(known) if known else "none"
        result = structured.invoke(
            [
                SystemMessage(EXTRACTION_INSTRUCTIONS),
                HumanMessage(f"Already known: {known_line}\nMessage: {text}"),
            ]
        )
        if not isinstance(result, TravellerPreferences):
            raise ValueError(f"Expected preferences, got {type(result).__name__}")
        return result.preferences

    return extract


# --- The Store: where preferences are kept ------------------------------------


def preferences_namespace(user_id: str) -> tuple[str, str, str]:
    """The Store folder for one traveller's preferences."""
    return ("users", user_id, "preferences")


def preference_key(text: str) -> str:
    """A stable key, so 'Vegetarian ' and 'vegetarian' are the same entry."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def load_preferences(store: BaseStore, user_id: str) -> list[str]:
    """All saved preferences of a traveller, oldest first."""
    items = store.search(preferences_namespace(user_id), limit=100)
    items.sort(key=lambda item: item.created_at)
    return [item.value["text"] for item in items]


def save_preferences(
    store: BaseStore, user_id: str, preferences: Sequence[str]
) -> list[str]:
    """Save new preferences and return the ones that were really new."""
    namespace = preferences_namespace(user_id)
    added = []
    for text in preferences:
        text = text.strip()
        key = preference_key(text)
        if not key or store.get(namespace, key) is not None:
            continue
        store.put(namespace, key, {"text": text})
        added.append(text)
    return added


# Local or foreign is saved next to the preferences, under one fixed key, so
# it shows in the saved-preferences list (and in the agent's prompt) and
# "forget my preferences" clears it too. A new answer replaces the old one.
TRAVELLER_TYPE_KEY = "traveller-type"
TRAVELLER_TYPE_TEXT = {
    "local": "Sri Lankan or lives in Sri Lanka (local prices)",
    "foreign": "visits from abroad (foreigner prices)",
}


def save_traveller_type(store: BaseStore, user_id: str, kind: str) -> None:
    """Remember whether the traveller is local or foreign."""
    if kind not in TRAVELLER_TYPE_TEXT:
        raise ValueError(f"unknown traveller type: {kind!r}")
    store.put(
        preferences_namespace(user_id),
        TRAVELLER_TYPE_KEY,
        {"text": TRAVELLER_TYPE_TEXT[kind], "traveller_type": kind},
    )


def load_traveller_type(store: BaseStore, user_id: str) -> str | None:
    """'local', 'foreign', or None if the traveller never said."""
    item = store.get(preferences_namespace(user_id), TRAVELLER_TYPE_KEY)
    return item.value.get("traveller_type") if item is not None else None


def clear_preferences(store: BaseStore, user_id: str) -> int:
    """Delete every saved preference of a traveller; return how many."""
    namespace = preferences_namespace(user_id)
    items = store.search(namespace, limit=100)
    for item in items:
        store.delete(namespace, item.key)
    return len(items)


@contextmanager
def sqlite_store(path: Path) -> Iterator[SqliteStore]:
    """Open a SQLite-backed Store on a file, and close the file afterwards."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
    try:
        store = SqliteStore(conn)
        store.setup()  # creates the tables the first time
        yield store
    finally:
        conn.close()
