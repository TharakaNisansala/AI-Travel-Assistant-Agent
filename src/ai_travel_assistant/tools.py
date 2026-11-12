"""Stage 3: tools the LLM can call.

What this file demonstrates:
  1. A plain Python search function over a small, built-in dataset.
  2. Turning it into a LangChain tool with the @tool decorator. The function
     name becomes the tool name, the docstring becomes the description the LLM
     reads, and the type hints become the argument schema.

The dataset is a short hand-written guide to well-known Sri Lankan places. It is
NOT live data: no web search and no external API. The tool says so in its output,
so the LLM (and the traveller) never mistake it for live search results.

Try the tool on its own (offline, no Gemini call):
    uv run python -c "from ai_travel_assistant.tools import search_destinations; \
print(search_destinations.invoke({'query': 'hiking'}))"
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from langchain_core.tools import tool

# --- The offline dataset ------------------------------------------------------


@dataclass(frozen=True)
class Destination:
    name: str
    location: str  # province and district
    activities: tuple[str, ...]
    # Extra search words that are not activities, e.g. "beach", "mountains".
    tags: tuple[str, ...] = ()


DESTINATIONS: tuple[Destination, ...] = (
    Destination(
        name="Ella",
        location="Uva Province, Badulla District",
        activities=(
            "hiking Little Adam's Peak and Ella Rock",
            "Nine Arch Bridge",
            "tea plantation visits",
            "scenic train ride",
        ),
        tags=("hills", "mountains", "views", "nature"),
    ),
    Destination(
        name="Nuwara Eliya",
        location="Central Province, Nuwara Eliya District",
        activities=(
            "hiking to World's End in Horton Plains National Park",
            "tea factory tours",
            "Gregory Lake",
        ),
        tags=("hills", "mountains", "cool climate", "tea", "nature"),
    ),
    Destination(
        name="Adam's Peak (Sri Pada)",
        location="Central and Sabaragamuwa Provinces, near Dalhousie",
        activities=(
            "overnight pilgrimage hike to the summit",
            "sunrise views from the top",
        ),
        tags=("mountains", "pilgrimage", "climbing", "trekking"),
    ),
    Destination(
        name="Sigiriya",
        location="Central Province, Matale District",
        activities=(
            "climbing Sigiriya Lion Rock",
            "ancient frescoes and water gardens",
            "Pidurangala Rock hike",
        ),
        tags=("history", "ancient", "unesco", "culture", "climbing"),
    ),
    Destination(
        name="Kandy",
        location="Central Province, Kandy District",
        activities=(
            "Temple of the Sacred Tooth Relic",
            "Peradeniya Royal Botanical Gardens",
            "Kandyan cultural dance shows",
        ),
        tags=("culture", "temples", "history", "unesco", "gardens"),
    ),
    Destination(
        name="Anuradhapura",
        location="North Central Province, Anuradhapura District",
        activities=(
            "ancient stupas such as Ruwanwelisaya",
            "Jaya Sri Maha Bodhi sacred fig tree",
            "cycling around the ruins",
        ),
        tags=("history", "ancient", "unesco", "temples", "culture", "buddhist"),
    ),
    Destination(
        name="Galle",
        location="Southern Province, Galle District",
        activities=(
            "walking the ramparts of Galle Fort",
            "colonial-era streets, cafes and museums",
            "nearby beaches such as Unawatuna",
        ),
        tags=("history", "unesco", "beach", "coast", "culture"),
    ),
    Destination(
        name="Mirissa",
        location="Southern Province, Matara District",
        activities=("whale watching", "surfing", "beach relaxation"),
        tags=("beach", "coast", "ocean", "sea"),
    ),
    Destination(
        name="Arugam Bay",
        location="Eastern Province, Ampara District",
        activities=("surfing", "lagoon safaris", "beach relaxation"),
        tags=("beach", "coast", "surf", "ocean"),
    ),
    Destination(
        name="Yala National Park",
        location="Southern and Uva Provinces, Hambantota and Monaragala Districts",
        activities=("jeep safaris", "leopard and elephant spotting", "bird watching"),
        tags=("wildlife", "safari", "nature", "animals"),
    ),
)


# --- Searching ----------------------------------------------------------------

# Words that say nothing about WHAT the traveller wants, so they must not count
# as matches. "sri" and "lanka" are here because every place is in Sri Lanka.
_STOPWORDS = frozenset(
    {
        "a", "an", "and", "any", "are", "best", "can", "do", "find", "for",
        "go", "good", "i", "in", "is", "like", "looking", "me", "my", "of",
        "on", "or", "place", "places", "please", "recommend", "show", "some",
        "sri", "lanka", "suggest", "that", "the", "there", "to", "visit",
        "want", "what", "where", "which", "with", "you",
    }
)  # fmt: skip


def _stem(word: str) -> str:
    """Cut common endings so different forms of a word match each other.

    hiking/hike -> "hik", beaches/beach -> "beach", temples/temple -> "templ".
    Deliberately tiny: good enough for a short, fixed dataset.
    """
    if word.endswith("ing") and len(word) > 5:
        word = word[:-3]
    elif word.endswith("es") and len(word) > 4:
        word = word[:-2]
    elif word.endswith("s") and len(word) > 3:
        word = word[:-1]
    if word.endswith("e") and len(word) > 3:
        word = word[:-1]
    return word


def _keywords(text: str) -> set[str]:
    """Lowercase words of the text, without stopwords, stemmed."""
    words = re.findall(r"[a-z]+", text.lower())
    return {_stem(word) for word in words if word not in _STOPWORDS}


def _searchable_text(destination: Destination) -> str:
    return " ".join(
        (destination.name, destination.location, *destination.activities)
        + destination.tags
    )


def find_destinations(query: str, limit: int = 5) -> list[Destination]:
    """Return the places that best match the query, best match first.

    A place scores one point for each query keyword found in its name, location,
    activities or tags. Places with no points are left out.
    """
    wanted = _keywords(query)
    if not wanted:
        return []
    scored = []
    for destination in DESTINATIONS:
        score = len(wanted & _keywords(_searchable_text(destination)))
        if score:
            scored.append((score, destination))
    # sorted() is stable: equal scores keep the dataset's order.
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [destination for _, destination in scored[:limit]]


def format_destinations(query: str, matches: list[Destination]) -> str:
    """Turn search results into the text the LLM will read."""
    source = "Source: the assistant's built-in offline guide (not live data)."
    if not matches:
        return (
            f"No destinations in the offline guide matched '{query}'. "
            "Try a broader activity such as hiking, beaches, wildlife, history "
            f"or tea.\n{source}"
        )
    lines = [f"Found {len(matches)} destination(s) for '{query}':"]
    for destination in matches:
        lines.append(
            f"- {destination.name} ({destination.location}): "
            f"{'; '.join(destination.activities)}"
        )
    lines.append(source)
    return "\n".join(lines)


# --- The tool -----------------------------------------------------------------
# @tool reads three things from the function below and sends them to the LLM:
#   name        -> "search_destinations"
#   description -> the docstring (this is how the LLM knows WHEN to use the tool)
#   args schema -> {"query": string}, built from the type hints
# The LLM never runs this code. It only asks for it to be run (a tool call);
# LangGraph's ToolNode runs it and sends the returned text back to the LLM.


@tool
def search_destinations(query: str) -> str:
    """Search a built-in guide of Sri Lankan destinations.

    Use this when the traveller asks where to go in Sri Lanka or which places
    suit an activity or interest (for example hiking, beaches, surfing,
    wildlife, history, temples or tea plantations). Returns each matching
    place's name, location and activities, or a message saying nothing matched.

    Args:
        query: What the traveller is looking for, in plain words,
            e.g. "hiking" or "beaches for surfing".
    """
    return format_destinations(query, find_destinations(query))
