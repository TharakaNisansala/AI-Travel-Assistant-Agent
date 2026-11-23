"""Stage 4 checks: the Wikivoyage guide service and its get_destination_guide tool.

Offline tests use httpx.MockTransport with synthetic MediaWiki responses shaped
like the real API (action=query, formatversion=2, TextExtracts plain text).

Live tests call the real Wikivoyage API (never Gemini), 1 request each. Run them
ONE AT A TIME by node ID, e.g.:
    uv run pytest -m live tests/test_guide.py::test_live_guide_for_ella
Wikimedia rate-limits anonymous clients quickly (HTTP 429 with Retry-After), so
a busy answer is skipped as INCONCLUSIVE rather than failed.
"""

from __future__ import annotations

import httpx
import pytest
from langchain_core.messages import AIMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

from ai_travel_assistant import guide_service
from ai_travel_assistant.guide_service import (
    INTRODUCTION,
    SECTION_CHARS_ALL,
    SECTION_CHARS_ONE_TOPIC,
    TOPICS,
    WIKIVOYAGE_API,
    Article,
    GuideAmbiguousError,
    GuideBusyError,
    GuideHTTPError,
    GuideNetworkError,
    GuideNotFoundError,
    GuideServiceError,
    GuideTimeoutError,
    InvalidGuideResponseError,
    disambiguation_options,
    fetch_article,
    get_destination_guide,
    make_client,
    select_topic,
    split_sections,
    trim,
)
from ai_travel_assistant.tool_calling import build_tool_agent_graph, run_tool_agent
from ai_travel_assistant.tools import (
    GUIDE_SOURCE,
    format_destination_guide,
)
from ai_travel_assistant.tools import (
    get_destination_guide as guide_tool,
)
from tests.test_attractions import ToolCallingFake, tool_call

# --- Sample data --------------------------------------------------------------

ELLA_EXTRACT = """Ella is a small town in Uva province in the Hill Country.

It is a great base for walks.


== Get in ==


=== By train ===
The train from Kandy is scenic. It takes about 7 hours.

=== By bus ===
Buses run from Badulla.

== Get around ==
Tuk-tuks are everywhere.

== See ==
Little Adam's Peak. An easy hike with great views.
Nine Arch Bridge. A famous railway bridge.

== Do ==
Visit a tea factory.

== Sleep ==
Many guesthouses.

== Stay healthy ==
Watch out for leeches after rain.
"""


def page_response(extract=ELLA_EXTRACT, title="Ella", **page_extra) -> dict:
    page = {
        "pageid": 10815,
        "ns": 0,
        "title": title,
        "extract": extract,
        "fullurl": f"https://en.wikivoyage.org/wiki/{title}",
        **page_extra,
    }
    return {"batchcomplete": True, "query": {"pages": [page]}}


MISSING = {"query": {"pages": [{"ns": 0, "title": "Xyzzy", "missing": True}]}}

VICTORIA_EXTRACT = """Many places are called Victoria.

=== Argentina ===
Victoria (Buenos Aires) - A city near Buenos Aires.

=== Seychelles ===
Victoria (Seychelles) - The capital of the Seychelles.
Victoria Falls - Waterfalls on the Zambia-Zimbabwe border.
Lake Victoria (Sri Lanka) - A reservoir in the hill country of Sri Lanka.
"""


# --- Fake network -------------------------------------------------------------


class FakeWikivoyage:
    def __init__(self, body=None):
        self.body = page_response() if body is None else body
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if isinstance(self.body, httpx.Response):
            return self.body
        return httpx.Response(200, json=self.body)

    def client(self) -> httpx.Client:
        return make_client(transport=httpx.MockTransport(self))


def respond(status: int, **kwargs):
    return lambda request: httpx.Response(status, **kwargs)


def raising(exc: Exception):
    def handler(request):
        raise exc

    return handler


def client_for(handler) -> httpx.Client:
    return make_client(transport=httpx.MockTransport(handler))


@pytest.fixture
def fake_wiki(monkeypatch) -> FakeWikivoyage:
    """Route the tool's own HTTP client to a FakeWikivoyage."""
    api = FakeWikivoyage()
    monkeypatch.setattr(guide_service, "make_client", api.client)
    return api


# --- 1. Parsing ---------------------------------------------------------------


def test_sections_are_split_on_top_level_headings():
    sections = split_sections(ELLA_EXTRACT)
    assert list(sections) == [
        INTRODUCTION,
        "Get in",
        "Get around",
        "See",
        "Do",
        "Sleep",
        "Stay healthy",
    ]
    # Paragraph breaks are collapsed to single spaces.
    assert sections[INTRODUCTION] == (
        "Ella is a small town in Uva province in the Hill Country. "
        "It is a great base for walks."
    )


def test_subsections_stay_inside_their_parent_with_a_label():
    get_in = split_sections(ELLA_EXTRACT)["Get in"]
    assert get_in.startswith("By train: The train from Kandy is scenic.")
    assert "By bus: Buses run from Badulla." in get_in


def test_subheadings_without_text_are_dropped():
    sections = split_sections("== Sleep ==\n=== Budget ===\n=== Splurge ===\nA villa.")
    assert sections["Sleep"] == "Splurge: A villa."


def test_short_text_is_not_trimmed():
    assert trim("Short.", 100) == ("Short.", False)


def test_long_text_is_trimmed_at_a_sentence_end():
    text = "First sentence here. Second sentence here. Third one is long."
    cut, truncated = trim(text, 50)
    assert truncated
    assert cut == "First sentence here. Second sentence here. ..."
    assert len(cut) <= 50


def test_text_without_sentence_breaks_is_trimmed_at_a_word():
    cut, truncated = trim("word " * 40, 30)
    assert truncated and len(cut) <= 30
    assert cut.endswith("word ...")


def test_disambiguation_options_put_sri_lanka_first():
    options = disambiguation_options(VICTORIA_EXTRACT)
    assert options[0] == "Lake Victoria (Sri Lanka)"
    assert "Victoria (Buenos Aires)" in options


# --- 2. Choosing a topic ------------------------------------------------------

ELLA_ARTICLE = Article(
    title="Ella",
    url="https://en.wikivoyage.org/wiki/Ella",
    sections=split_sections(ELLA_EXTRACT),
)


def test_topics_are_exactly_the_agreed_ones():
    assert TOPICS == (
        "all",
        "overview",
        "getting_there",
        "getting_around",
        "see_and_do",
        "accommodation",
        "safety",
    )


def test_one_topic_keeps_only_its_sections():
    guide = select_topic(ELLA_ARTICLE, "see_and_do")
    assert [s.title for s in guide.sections] == ["See", "Do"]
    assert guide.missing == ()


def test_all_covers_every_topic_including_sleep_but_never_eat():
    guide = select_topic(ELLA_ARTICLE, "all")
    titles = [s.title for s in guide.sections]
    assert titles == [
        INTRODUCTION,
        "Get in",
        "Get around",
        "See",
        "Do",
        "Sleep",
        "Stay healthy",
    ]


def test_accommodation_is_the_sleep_section():
    guide = select_topic(ELLA_ARTICLE, "accommodation")
    assert [(s.title, s.text) for s in guide.sections] == [
        ("Sleep", "Many guesthouses.")
    ]
    no_sleep = Article("X", "u", {INTRODUCTION: "Intro."})
    assert select_topic(no_sleep, "accommodation").missing == ("accommodation",)


def test_a_topic_is_missing_only_when_none_of_its_sections_exist():
    # Ella has "Stay healthy" but no "Stay safe" or "Cope": safety is covered.
    assert select_topic(ELLA_ARTICLE, "safety").missing == ()
    no_safety = Article("X", "u", {INTRODUCTION: "Intro.", "See": "A fort."})
    assert select_topic(no_safety, "all").missing == (
        "getting_there",
        "getting_around",
        "accommodation",
        "safety",
    )


def test_section_names_match_regardless_of_case():
    article = Article("X", "u", {"GET IN": "By bus."})
    assert [s.title for s in select_topic(article, "getting_there").sections] == [
        "GET IN"
    ]


def test_all_uses_a_smaller_limit_per_section_than_one_topic():
    long_text = "A sentence about the sights. " * 100
    article = Article("X", "u", {"See": long_text})
    one = select_topic(article, "see_and_do").sections[0]
    every = select_topic(article, "all").sections[0]
    assert one.truncated and every.truncated
    assert (
        len(every.text) <= SECTION_CHARS_ALL < len(one.text) <= SECTION_CHARS_ONE_TOPIC
    )


# --- 3. The whole flow (fake network) -----------------------------------------


def test_request_asks_for_plain_text_with_redirects_and_disambiguation():
    api = FakeWikivoyage()
    fetch_article("Ella", client=api.client())
    request = api.requests[0]
    assert str(request.url.copy_with(query=None)) == WIKIVOYAGE_API
    params = request.url.params
    assert params["titles"] == "Ella"
    assert params["explaintext"] == "1"
    assert params["redirects"] == "1"
    assert params["exsectionformat"] == "wiki"
    assert "disambiguation" in params["ppprop"]
    assert request.headers["User-Agent"].startswith("ai-travel-assistant/")


def test_guide_for_a_known_place():
    guide = get_destination_guide(
        "Ella", "getting_there", client=FakeWikivoyage().client()
    )
    assert guide.title == "Ella"
    assert guide.url == "https://en.wikivoyage.org/wiki/Ella"
    assert guide.sections[0].text.startswith("By train:")


def test_articles_are_cached_so_a_second_topic_needs_no_request():
    api = FakeWikivoyage()
    get_destination_guide("Ella", "overview", client=api.client())
    get_destination_guide("ella", "safety", client=api.client())
    assert len(api.requests) == 1


def test_unknown_topic_fails_before_any_request():
    api = FakeWikivoyage()
    with pytest.raises(ValueError, match="Unknown topic"):
        get_destination_guide("Ella", "food", client=api.client())
    assert api.requests == []


def test_blank_name_fails_before_any_request():
    api = FakeWikivoyage()
    with pytest.raises(GuideNotFoundError):
        get_destination_guide("  ", client=api.client())
    assert api.requests == []


# --- 4. Failures --------------------------------------------------------------


def test_missing_page_is_not_found():
    with pytest.raises(GuideNotFoundError, match="no article called 'Xyzzy'"):
        get_destination_guide("Xyzzy", client=FakeWikivoyage(MISSING).client())


def test_disambiguation_page_is_ambiguous_and_lists_options():
    body = page_response(
        VICTORIA_EXTRACT, title="Victoria", pageprops={"disambiguation": ""}
    )
    with pytest.raises(GuideAmbiguousError) as error:
        get_destination_guide("Victoria", client=FakeWikivoyage(body).client())
    assert error.value.options[0] == "Lake Victoria (Sri Lanka)"
    assert "Lake Victoria (Sri Lanka)" in str(error.value)
    # Ambiguous is a kind of not-found, so callers can treat both alike.
    assert isinstance(error.value, GuideNotFoundError)


@pytest.mark.parametrize(
    ("handler", "error"),
    [
        (raising(httpx.ReadTimeout("slow")), GuideTimeoutError),
        (raising(httpx.ConnectError("down")), GuideNetworkError),
        (respond(429, headers={"Retry-After": "34"}), GuideBusyError),
        (respond(503), GuideBusyError),
        (respond(403), GuideHTTPError),
        (respond(200, text="<html>"), InvalidGuideResponseError),
        (respond(200, json=[1, 2]), InvalidGuideResponseError),
        (respond(200, json={"query": {}}), InvalidGuideResponseError),
        (
            respond(200, json={"error": {"code": "badvalue", "info": "Bad title"}}),
            GuideHTTPError,
        ),
    ],
    ids=[
        "timeout",
        "offline",
        "429",
        "503",
        "403",
        "not-json",
        "not-object",
        "no-pages",
        "api-error",
    ],
)
def test_failures_become_typed_errors(handler, error):
    with pytest.raises(error):
        get_destination_guide("Ella", client=client_for(handler))


def test_wikimedia_long_retry_after_is_not_waited_for(fresh_caches_and_no_sleep):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(429, headers={"Retry-After": "34"})

    with pytest.raises(GuideBusyError):
        get_destination_guide("Ella", client=client_for(handler))
    assert len(calls) == 1
    assert fresh_caches_and_no_sleep == []


def test_page_without_text_is_invalid():
    body = {"query": {"pages": [{"ns": 0, "title": "Ella", "pageid": 1}]}}
    with pytest.raises(InvalidGuideResponseError, match="without its text"):
        get_destination_guide("Ella", client=FakeWikivoyage(body).client())


# --- 5. The get_destination_guide tool (still offline) ------------------------


def test_tool_has_name_description_and_arguments():
    assert guide_tool.name == "get_destination_guide"
    assert "wikivoyage" in guide_tool.description.lower()
    assert set(guide_tool.args) == {"destination", "topic"}


def test_topic_is_an_enum_of_the_service_topics():
    schema = convert_to_openai_tool(guide_tool)["function"]["parameters"]
    assert schema["properties"]["topic"]["enum"] == list(TOPICS)
    assert schema["properties"]["topic"]["default"] == "all"
    assert schema["required"] == ["destination"]


def test_docstring_explains_every_topic_and_points_to_other_tools():
    for topic in TOPICS:
        assert f'"{topic}"' in guide_tool.description
    assert "get_weather_forecast" in guide_tool.description
    assert "find_attractions" in guide_tool.description


def test_tool_returns_labelled_sections_with_source(fake_wiki):
    text = guide_tool.invoke({"destination": "Ella", "topic": "see_and_do"})
    lines = text.splitlines()
    assert lines[0] == "Wikivoyage guide for Ella (see and do):"
    assert lines[1].startswith("[See] Little Adam's Peak.")
    assert lines[2] == "[Do] Visit a tea factory."
    assert lines[-2] == "Full article: https://en.wikivoyage.org/wiki/Ella"
    assert lines[-1] == GUIDE_SOURCE


def test_tool_output_stays_small_for_a_huge_article(monkeypatch):
    # Every section of a ~60,000-character article, as for a big city.
    huge = "Intro. " * 500 + "".join(
        f"\n== {name} ==\n" + "Plenty of detail here. " * 400
        for name in ("Understand", "Get in", "Get around", "See", "Do", "Stay safe")
    )
    api = FakeWikivoyage(page_response(huge, title="Colombo"))
    monkeypatch.setattr(guide_service, "make_client", api.client)
    text = guide_tool.invoke({"destination": "Colombo"})
    assert len(huge) > 50_000
    assert len(text) < 4_500  # about 1,100 tokens
    assert "were shortened; full article:" in text


def test_tool_reports_topics_the_article_lacks(monkeypatch):
    body = page_response("A small village.\n== See ==\nA temple.", title="Tiny")
    monkeypatch.setattr(guide_service, "make_client", FakeWikivoyage(body).client)
    text = guide_tool.invoke({"destination": "Tiny", "topic": "getting_there"})
    assert "The article has nothing on this topic." in text
    assert "Not covered by this article: getting there. Do not fill this in." in text


def test_tool_asks_to_retry_with_an_exact_name_when_ambiguous(monkeypatch):
    body = page_response(
        VICTORIA_EXTRACT, title="Victoria", pageprops={"disambiguation": ""}
    )
    monkeypatch.setattr(guide_service, "make_client", FakeWikivoyage(body).client)
    text = guide_tool.invoke({"destination": "Victoria"})
    assert text.startswith("Guide unavailable for 'Victoria'")
    assert "Lake Victoria (Sri Lanka)" in text
    assert "call this tool again with that exact name" in text


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (MISSING, "no article called"),
        (httpx.Response(429), "busy (HTTP 429)"),
    ],
    ids=["missing", "busy"],
)
def test_tool_reports_failures_instead_of_raising(monkeypatch, body, expected):
    monkeypatch.setattr(guide_service, "make_client", FakeWikivoyage(body).client)
    text = guide_tool.invoke({"destination": "Ella"})
    assert text.startswith("Guide unavailable for 'Ella'")
    assert expected in text
    assert "do not make up guide information" in text


def test_format_marks_shortened_sections():
    article = Article("X", "https://x", {"See": "Long sentence here. " * 200})
    text = format_destination_guide(select_topic(article, "see_and_do"))
    assert "Sections ending in ... were shortened; full article: https://x" in text


# --- 6. The agent using the guide (scripted model, offline) -------------------


def test_agent_binds_and_runs_the_guide_tool(fake_wiki):
    model = ToolCallingFake(
        messages=iter(
            [
                tool_call(
                    "get_destination_guide",
                    {"destination": "Ella", "topic": "getting_there"},
                    "call_g1",
                ),
                AIMessage("Take the scenic train from Kandy."),
            ]
        )
    )
    messages = run_tool_agent(build_tool_agent_graph(model), "How do I get to Ella?")

    assert guide_tool in model.bound_tools
    result = messages[2]
    assert (result.name, result.status) == ("get_destination_guide", "success")
    assert result.content.startswith("Wikivoyage guide for Ella (getting there):")
    assert model.calls[1][-1].content == result.content
    assert len(fake_wiki.requests) == 1


# --- 7. Live (real Wikivoyage, opt-in) ----------------------------------------

INCONCLUSIVE = "INCONCLUSIVE (infrastructure, not a code failure)"


def live_guide(destination: str, topic: str = "all"):
    try:
        return get_destination_guide(destination, topic)
    except (GuideBusyError, GuideTimeoutError, GuideNetworkError) as exc:
        pytest.skip(f"{INCONCLUSIVE}: {exc}")


@pytest.mark.live
def test_live_guide_for_ella():
    guide = live_guide("Ella", "see_and_do")
    assert guide.title == "Ella"
    text = " ".join(s.text for s in guide.sections)
    assert "Adam" in text  # Little Adam's Peak is Ella's best-known walk


@pytest.mark.live
def test_live_guide_for_kandy_getting_there():
    guide = live_guide("Kandy", "getting_there")
    assert guide.sections, "expected a Get in section for Kandy"
    assert "train" in guide.sections[0].text.lower()


@pytest.mark.live
def test_live_unknown_place_is_not_found():
    with pytest.raises(GuideNotFoundError):
        live_guide("Xyzzyplace Nowhere")


def test_live_helper_skips_busy_answers(monkeypatch):
    monkeypatch.setattr(
        guide_service, "make_client", FakeWikivoyage(httpx.Response(429)).client
    )
    with pytest.raises(pytest.skip.Exception) as skipped:
        live_guide("Ella")
    assert str(skipped.value.msg).startswith(INCONCLUSIVE)


def test_every_error_is_a_guide_service_error():
    for error in (
        GuideNotFoundError,
        GuideAmbiguousError,
        GuideTimeoutError,
        GuideNetworkError,
        GuideHTTPError,
        GuideBusyError,
        InvalidGuideResponseError,
    ):
        assert issubclass(error, GuideServiceError)
