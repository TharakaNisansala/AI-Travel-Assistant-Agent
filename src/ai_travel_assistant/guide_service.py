"""Stage 4: a real external API client for Wikivoyage travel guides (no LangChain).

What this file demonstrates:
  1. Calling the MediaWiki Action API (https://en.wikivoyage.org/w/api.php) with
     the TextExtracts extension: one request returns a whole article as plain
     text, with "== Heading ==" markers and redirects already followed.
  2. Controlling CONTEXT COST. A Wikivoyage article is 10,000-50,000 characters
     (Kandy is ~32,000), far too much to hand to an LLM on every turn. So the
     service keeps only the sections a traveller needs, picked by a `topic`, and
     trims each one to a fixed size at a sentence boundary. The LLM reads about
     1,000 tokens instead of 8,000+.
  3. Telling apart the ways a lookup can fail: no such page, a disambiguation
     page ("Victoria" means many places), or a temporary outage (Wikimedia
     rate-limits anonymous clients quickly with HTTP 429 and Retry-After).

Weather and attractions return structured data. A guide returns prose, which is
the kind of tool output most likely to flood the context window, so the
trimming rules here are the main lesson of this file.

Wikivoyage content is CC BY-SA 4.0; the tool output names the source and links
the article.

Try it on its own (1 HTTP request, no Gemini call):
    uv run python -c "from ai_travel_assistant.guide_service import \
get_destination_guide; print(get_destination_guide('Ella'))"
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal, get_args

import httpx

from ai_travel_assistant import faults
from ai_travel_assistant.resilience import TTLCache, send_with_retry
from ai_travel_assistant.weather_service import USER_AGENT

WIKIVOYAGE_API = "https://en.wikivoyage.org/w/api.php"
REQUEST_TIMEOUT = 10.0

# Articles are edited occasionally, not hourly: keep a parsed article for a day.
# This also matters because Wikimedia rate-limits anonymous clients quickly.
ARTICLE_CACHE = TTLCache("wikivoyage articles", ttl_s=24 * 3600)

# --- Topics -------------------------------------------------------------------
# A fixed list, exposed to the LLM as an enum. Each topic maps to Wikivoyage's
# standard section names (https://en.wikivoyage.org/wiki/Wikivoyage:Where_you_can_stick_it).

Topic = Literal[
    "all",
    "overview",
    "getting_there",
    "getting_around",
    "see_and_do",
    "accommodation",
    "safety",
]
TOPICS: tuple[str, ...] = get_args(Topic)

INTRODUCTION = "Introduction"  # the lead text before the first heading
TOPIC_SECTIONS: dict[str, tuple[str, ...]] = {
    "overview": (INTRODUCTION, "Understand"),
    "getting_there": ("Get in",),
    "getting_around": ("Get around",),
    "see_and_do": ("See", "Do"),
    "accommodation": ("Sleep",),  # guesthouses and hotels, often with prices
    "safety": ("Stay safe", "Stay healthy", "Cope"),
}
# "all" is every single topic above, in this order.
SINGLE_TOPICS: tuple[str, ...] = tuple(TOPIC_SECTIONS)
TOPIC_SECTIONS["all"] = tuple(
    section for topic in SINGLE_TOPICS for section in TOPIC_SECTIONS[topic]
)

# Characters kept per section. One topic gets room for detail; "all" covers up
# to 11 sections, so each gets less. Roughly 4 characters make one token.
SECTION_CHARS_ONE_TOPIC = 1200
SECTION_CHARS_ALL = 450
# The most disambiguation choices we pass back to the LLM.
MAX_OPTIONS = 5


# --- Errors -------------------------------------------------------------------


class GuideServiceError(Exception):
    """Any failure to get a destination guide. The message is safe to show."""


class GuideNotFoundError(GuideServiceError):
    """Wikivoyage has no article with that name."""


class GuideAmbiguousError(GuideNotFoundError):
    """The name leads to a disambiguation page listing several places."""

    def __init__(self, message: str, options: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.options = options


class GuideTimeoutError(GuideServiceError):
    """Wikivoyage did not answer within REQUEST_TIMEOUT seconds."""


class GuideNetworkError(GuideServiceError):
    """Wikivoyage could not be reached (no internet, DNS failure, refused, ...)."""


class GuideHTTPError(GuideServiceError):
    """Wikivoyage answered with an HTTP error status or an API error."""


class GuideBusyError(GuideHTTPError):
    """Wikivoyage is rate-limiting us or overloaded (429/502/503/504).

    Temporary: a later request may succeed.
    """


class InvalidGuideResponseError(GuideServiceError):
    """Wikivoyage answered, but not with the JSON we expected."""


# --- Typed results ------------------------------------------------------------


@dataclass(frozen=True)
class Article:
    """A whole parsed article, before choosing a topic (this is what is cached)."""

    title: str
    url: str
    sections: dict[str, str]  # section title -> plain text (subsections merged)


@dataclass(frozen=True)
class GuideSection:
    title: str
    text: str
    truncated: bool  # True if text was cut to fit the size limit


@dataclass(frozen=True)
class DestinationGuide:
    title: str  # the article title, e.g. "Ella"
    url: str
    topic: str
    sections: tuple[GuideSection, ...]
    missing: tuple[str, ...]  # topics (e.g. "safety") the article has nothing on


# --- HTTP ---------------------------------------------------------------------


def make_client(transport: httpx.BaseTransport | None = None) -> httpx.Client:
    """An httpx client with our timeout and User-Agent.

    Wikimedia asks every client for a descriptive User-Agent. Ours names the
    project but carries no personal contact details.
    """
    return httpx.Client(
        timeout=REQUEST_TIMEOUT,
        headers={"User-Agent": USER_AGENT},
        # Stage 7: fails on purpose only if FAULT_INJECT names guide.
        transport=transport or faults.transport_for("guide"),
    )


def _get_json(client: httpx.Client, params: dict[str, Any]) -> dict:
    """GET the Wikivoyage API and return its JSON object, or raise."""
    try:
        response = send_with_retry(lambda: client.get(WIKIVOYAGE_API, params=params))
    except httpx.TimeoutException as exc:
        raise GuideTimeoutError(
            f"Wikivoyage did not respond within {REQUEST_TIMEOUT:g} seconds."
        ) from exc
    except httpx.TransportError as exc:
        raise GuideNetworkError(
            f"Could not connect to Wikivoyage ({type(exc).__name__}). "
            "Check the internet connection."
        ) from exc
    except httpx.HTTPError as exc:
        raise InvalidGuideResponseError(
            f"The Wikivoyage response could not be read ({type(exc).__name__})."
        ) from exc

    if response.status_code in (429, 502, 503, 504):
        raise GuideBusyError(
            f"Wikivoyage is busy (HTTP {response.status_code}). Try again in a minute."
        )
    if response.is_error:
        raise GuideHTTPError(f"Wikivoyage returned HTTP {response.status_code}.")

    try:
        data = response.json()
    except ValueError as exc:
        raise InvalidGuideResponseError(
            "Wikivoyage returned a response that is not JSON."
        ) from exc
    if not isinstance(data, dict):
        raise InvalidGuideResponseError(
            "Wikivoyage returned unexpected JSON (not an object)."
        )
    # The Action API reports bad requests as HTTP 200 with an "error" object.
    error = data.get("error")
    if isinstance(error, dict):
        raise GuideHTTPError(
            f"Wikivoyage reported an error: {error.get('info', 'no reason given')}"
        )
    return data


# --- Parsing ------------------------------------------------------------------

# A heading line in a TextExtracts plain-text article: "== See ==", "=== By bus ===".
_HEADING = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$", re.M)


def _clean(text: str) -> str:
    """Collapse every run of whitespace (including newlines) into one space."""
    return " ".join(text.split())


def split_sections(extract: str) -> dict[str, str]:
    """Split a plain-text article into its top-level (==) sections.

    The lead text becomes INTRODUCTION. Deeper headings (===, ====) stay inside
    their parent section as "Subheading: text", so nothing under "Get in" (By
    train, By bus) is lost. Empty sections are left out.
    """
    sections: dict[str, list[str]] = {INTRODUCTION: []}
    current = INTRODUCTION
    position = 0
    for match in _HEADING.finditer(extract):
        body = _clean(extract[position : match.start()])
        if body:
            sections[current].append(body)
        level, title = len(match.group(1)), _clean(match.group(2))
        if level == 2:
            current = title
            sections.setdefault(current, [])
        else:
            sections[current].append(f"{title}:")
        position = match.end()
    tail = _clean(extract[position:])
    if tail:
        sections[current].append(tail)

    merged: dict[str, str] = {}
    for title, parts in sections.items():
        # Drop subheadings that had no text of their own ("By bus:" followed by
        # nothing, or by the next subheading).
        kept = [
            part
            for i, part in enumerate(parts)
            if not (
                part.endswith(":")
                and (i + 1 == len(parts) or parts[i + 1].endswith(":"))
            )
        ]
        if kept:
            merged[title] = " ".join(kept)
    return merged


def trim(text: str, limit: int) -> tuple[str, bool]:
    """Cut text to at most `limit` characters, ending at a sentence if possible.

    Returns (text, truncated). A cut text ends with "..." so the LLM can tell
    there was more.
    """
    if len(text) <= limit:
        return text, False
    cut = text[: limit - 3]
    sentence_end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    if sentence_end >= limit // 2:
        cut = cut[: sentence_end + 1]
    else:
        space = cut.rfind(" ")
        if space > 0:
            cut = cut[:space]
    return cut.rstrip() + " ...", True


# A disambiguation line: "Victoria (Buenos Aires) - A city in ...".
_OPTION = re.compile(r"^(?P<title>[^\n=]+?) - (?P<description>.+)$", re.M)


def disambiguation_options(extract: str, prefer: str = "Sri Lanka") -> tuple[str, ...]:
    """Article titles listed on a disambiguation page, `prefer` matches first."""
    options = [
        (m["title"].strip(), m["description"]) for m in _OPTION.finditer(extract)
    ]
    options.sort(key=lambda option: prefer.casefold() not in option[1].casefold())
    return tuple(title for title, _ in options[:MAX_OPTIONS])


def parse_article(data: dict, requested: str) -> Article:
    """Turn the API response into an Article, or raise a typed error."""
    query = data.get("query")
    pages = query.get("pages") if isinstance(query, dict) else None
    if not isinstance(pages, list) or not pages or not isinstance(pages[0], dict):
        raise InvalidGuideResponseError("The Wikivoyage response has no page.")
    page = pages[0]

    if page.get("missing") or page.get("invalid"):
        raise GuideNotFoundError(f"Wikivoyage has no article called '{requested}'.")

    title = page.get("title") if isinstance(page.get("title"), str) else requested
    extract = page.get("extract")
    if not isinstance(extract, str):
        raise InvalidGuideResponseError(
            f"The Wikivoyage article '{title}' came back without its text."
        )

    pageprops = page.get("pageprops")
    if isinstance(pageprops, dict) and "disambiguation" in pageprops:
        options = disambiguation_options(extract)
        listed = f" Possible articles: {'; '.join(options)}." if options else ""
        raise GuideAmbiguousError(
            f"'{requested}' matches several places on Wikivoyage.{listed}", options
        )

    sections = split_sections(extract)
    if not sections:
        raise GuideNotFoundError(f"The Wikivoyage article '{title}' is empty.")
    url = page.get("fullurl")
    if not isinstance(url, str) or not url:
        url = f"https://en.wikivoyage.org/wiki/{title.replace(' ', '_')}"
    return Article(title=title, url=url, sections=sections)


def select_topic(article: Article, topic: str) -> DestinationGuide:
    """Keep only the sections for one topic, each trimmed to its size limit.

    A topic counts as missing only if the article has none of its sections:
    many articles have "Stay healthy" but no "Stay safe" or "Cope", and that
    still covers safety.
    """
    limit = SECTION_CHARS_ALL if topic == "all" else SECTION_CHARS_ONE_TOPIC
    # Section names vary in case between articles ("By Bus" vs "By bus").
    by_name = {name.casefold(): (name, text) for name, text in article.sections.items()}

    kept, missing = [], []
    for single in SINGLE_TOPICS if topic == "all" else (topic,):
        found_any = False
        for name in TOPIC_SECTIONS[single]:
            found = by_name.get(name.casefold())
            if found is None:
                continue
            found_any = True
            text, truncated = trim(found[1], limit)
            kept.append(GuideSection(title=found[0], text=text, truncated=truncated))
        if not found_any:
            missing.append(single)
    return DestinationGuide(
        title=article.title,
        url=article.url,
        topic=topic,
        sections=tuple(kept),
        missing=tuple(missing),
    )


# --- The whole flow -----------------------------------------------------------


def fetch_article(destination: str, *, client: httpx.Client) -> Article:
    """Fetch and parse one Wikivoyage article (1 HTTP request, cached a day)."""
    name = destination.strip()
    if not name:
        raise GuideNotFoundError("No destination name was given.")
    cache_key = name.casefold()
    cached = ARTICLE_CACHE.get(cache_key)
    if cached is not TTLCache.MISSING:
        return cached

    params = {
        "action": "query",
        "format": "json",
        "formatversion": 2,
        "titles": name,
        "redirects": 1,  # "Adam's Peak" style redirects land on the real article
        "prop": "extracts|pageprops|info",
        "explaintext": 1,  # plain text, not HTML
        "exsectionformat": "wiki",  # keep headings as "== Heading =="
        "ppprop": "disambiguation",
        "inprop": "url",
    }
    article = parse_article(_get_json(client, params), name)
    ARTICLE_CACHE.set(cache_key, article)
    return article


def get_destination_guide(
    destination: str, topic: str = "all", *, client: httpx.Client | None = None
) -> DestinationGuide:
    """Get the parts of a destination's Wikivoyage article that match a topic.

    Raises ValueError for an unknown topic (a programming error), and a
    GuideServiceError subclass for every expected failure.
    """
    if topic not in TOPIC_SECTIONS:
        raise ValueError(
            f"Unknown topic {topic!r}; expected one of {', '.join(TOPICS)}."
        )
    if client is not None:
        return select_topic(fetch_article(destination, client=client), topic)
    with make_client() as own_client:
        return select_topic(fetch_article(destination, client=own_client), topic)
