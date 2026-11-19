"""pricing_service: verified prices per visitor category (offline).

The CCF page is a saved copy (tests/fixtures/ccf_ticket_prices.html, read on
2026-11-19); HTTP goes through httpx.MockTransport, and the cache is a
SQLite file in tmp_path.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

from ai_travel_assistant import pricing_service as ps
from ai_travel_assistant.pricing_service import (
    CuratedPriceError,
    PriceBook,
    PriceCache,
    PriceRecord,
    PriceSourceError,
    attraction_id,
    ccf_prices,
    load_curated,
    parse_ccf,
    price_for,
)

FIXTURE = Path(__file__).parent / "fixtures" / "ccf_ticket_prices.html"
CCF_PAGE = FIXTURE.read_text(encoding="utf-8")
TODAY = date(2026, 11, 19)
NOW = datetime(2026, 11, 19, 9, 0, tzinfo=timezone.utc)
MAX_AGE = timedelta(days=90)


@pytest.fixture(scope="module")
def ccf() -> list[PriceRecord]:
    return parse_ccf(CCF_PAGE, TODAY)


def find(records, aid, age="adult"):
    [record] = [r for r in records if r.attraction_id == aid and r.age_group == age]
    return record


# --- Parsing the CCF table ------------------------------------------------------


def test_ccf_prices_are_foreign_prices_with_their_source(ccf):
    sigiriya = find(ccf, "sigiriya")
    assert (sigiriya.amount, sigiriya.currency) == (35.0, "USD")
    assert sigiriya.lkr_equivalent == 11760.0
    assert sigiriya.exchange_rate == 336.0
    assert sigiriya.nationality == "foreign"  # CCF does not split SAARC
    assert sigiriya.source_url == "https://ccf.gov.lk/is/"
    assert sigiriya.status == "official"
    assert sigiriya.source_updated == date(2026, 10, 14)
    assert sigiriya.last_verified == TODAY
    assert "does not say whether SAARC" in sigiriya.conditions
    assert "lists no Sri Lankan price" in sigiriya.conditions


def test_a_half_ticket_is_a_child_record_without_an_age_rule(ccf):
    half = find(ccf, "sigiriya", "child")
    assert (half.amount, half.lkr_equivalent) == (20.0, 6720.0)
    assert half.min_age is None and half.max_age is None
    assert "does not publish who qualifies" in half.conditions


def test_rows_without_a_half_ticket_have_only_an_adult_price(ccf):
    museum = [r for r in ccf if r.attraction_id == "sigiriya-museum"]
    assert [(r.age_group, r.amount) for r in museum] == [("adult", 6.0)]


def test_rows_under_a_heading_are_activities_of_that_place(ccf):
    tours = [r for r in ccf if r.ticket_type == "activity"]
    assert tours, "the Galle diving tours should be read"
    assert all(r.attraction_name.startswith("Galle: ") for r in tours)
    assert "Galle: Shipwreck diving tour – Galle Harbor" in {
        r.attraction_name for r in tours
    }
    # The heading row itself ("Galle", no prices) is not an attraction.
    assert "galle" not in {r.attraction_id for r in ccf}


def test_a_page_without_the_table_is_an_error():
    with pytest.raises(PriceSourceError, match="layout"):
        parse_ccf("<html><body>Maintenance</body></html>", TODAY)


def test_attraction_ids():
    assert attraction_id("Galle (Museum)") == "galle-museum"
    assert attraction_id("Temple of the Tooth") == "temple-tooth"


# --- Choosing the price for a visitor category ------------------------------------


def test_a_non_saarc_visitor_gets_the_general_foreign_price_at_medium(ccf):
    sigiriya = [r for r in ccf if r.attraction_id == "sigiriya"]
    price = price_for(sigiriya, "non_saarc", "adult", today=TODAY, max_age=MAX_AGE)
    assert price.verified
    assert price.record.amount == 35.0
    assert price.confidence == "Medium"
    assert "no SAARC split" in price.note


def test_a_saarc_visitor_never_gets_the_general_foreign_price(ccf):
    sigiriya = [r for r in ccf if r.attraction_id == "sigiriya"]
    price = price_for(sigiriya, "saarc", "adult", today=TODAY, max_age=MAX_AGE)
    assert not price.verified
    assert price.confidence is None
    assert "does not publish a SAARC rate" in price.note
    assert "USD 35" in price.note  # shown as information, not applied


def test_a_local_visitor_never_gets_a_foreign_price(ccf):
    sigiriya = [r for r in ccf if r.attraction_id == "sigiriya"]
    price = price_for(sigiriya, "local", "adult", today=TODAY, max_age=MAX_AGE)
    assert not price.verified
    assert "No official Sri Lankan price" in price.note


def test_a_child_price_without_an_age_rule_is_not_applied(ccf):
    sigiriya = [r for r in ccf if r.attraction_id == "sigiriya"]
    price = price_for(sigiriya, "non_saarc", "child", today=TODAY, max_age=MAX_AGE)
    assert not price.verified
    assert "does not publish who qualifies" in price.note


def record(**changes) -> PriceRecord:
    base = {
        "attraction_id": "example-park",
        "attraction_name": "Example Park",
        "nationality": "local",
        "age_group": "adult",
        "amount": 100.0,
        "currency": "LKR",
        "ticket_type": "entrance",
        "unit": "person",
        "source_url": "https://example.gov.lk/tariff",
        "source_authority": "Example Authority",
        "last_verified": TODAY,
        "status": "official",
    }
    return PriceRecord(**{**base, **changes})


def test_an_exact_official_category_is_high_confidence():
    price = price_for([record()], "local", "adult", today=TODAY, max_age=MAX_AGE)
    assert price.verified and price.confidence == "High"
    assert price.note == ""


def test_a_child_price_with_its_age_rule_is_applied():
    child = record(age_group="child", min_age=3, max_age=12, amount=50)
    price = price_for([child], "local", "child", today=TODAY, max_age=MAX_AGE)
    assert price.verified and price.record.amount == 50


def test_an_exact_saarc_price_wins_over_the_general_foreign_price():
    rows = [
        record(nationality="foreign", currency="USD", amount=30),
        record(nationality="saarc", currency="USD", amount=15),
    ]
    price = price_for(rows, "saarc", "adult", today=TODAY, max_age=MAX_AGE)
    assert price.record.amount == 15 and price.confidence == "High"


@pytest.mark.parametrize(
    ("status", "confidence"),
    [("official", "High"), ("operator", "Medium"), ("secondary", "Low")],
)
def test_confidence_follows_the_kind_of_source(status, confidence):
    price = price_for(
        [record(status=status)], "local", "adult", today=TODAY, max_age=MAX_AGE
    )
    assert price.confidence == confidence


def test_an_old_price_loses_a_confidence_level_and_says_why():
    old = record(source_updated=TODAY - timedelta(days=200))
    price = price_for([old], "local", "adult", today=TODAY, max_age=MAX_AGE)
    assert price.verified
    assert price.confidence == "Medium"
    assert "May be out of date" in price.note and "2026-05-03" in price.note


def test_an_expired_or_future_price_is_not_used():
    expired = record(expiry_date=TODAY - timedelta(days=1))
    future = record(effective_date=TODAY + timedelta(days=10))
    for rows in ([expired], [future]):
        price = price_for(rows, "local", "adult", today=TODAY, max_age=MAX_AGE)
        assert not price.verified
        assert "not in force today" in price.note


def test_activity_prices_are_not_entrance_prices():
    tour = record(ticket_type="activity")
    assert not price_for([tour], "local", "adult", today=TODAY).verified
    assert price_for(
        [tour], "local", "adult", today=TODAY, ticket_type="activity"
    ).verified


# --- Finding an attraction by name ------------------------------------------------


def test_names_match_on_every_word_of_the_attraction(ccf):
    book = PriceBook(ccf)
    assert book.match("Sigiriya Rock Fortress") == "sigiriya"
    assert book.match("sigiriya") == "sigiriya"
    assert book.match("Sigiriya Museum") == "sigiriya-museum"
    assert book.match("Anuradhapura Ancient City") == "anuradhapura"
    # Run by the temple, not CCF: never priced from "Dambulla (Museum)".
    assert book.match("Dambulla Cave Temple") is None
    assert book.match("Galle Fort") is None  # only CCF's Galle museum is listed
    assert book.match("Ella Rock") is None


def test_aliases_find_an_attraction():
    book = PriceBook([record(aliases=("Lion Rock Park",))])
    assert book.match("lion rock park") == "example-park"
    lookup = book.lookup("Lion Rock Park")
    assert lookup.attraction_name == "Example Park" and len(lookup.records) == 1
    assert book.lookup("Nowhere").attraction_id is None


# --- The curated file -----------------------------------------------------------


def write(tmp_path, text: str) -> Path:
    path = tmp_path / "prices.yaml"
    path.write_text(text, encoding="utf-8")
    return path


GOOD = """
prices:
  - attraction_id: example-park
    attraction_name: Example Park
    aliases: [Example Gardens]
    nationality: local
    age_group: child
    min_age: 6
    max_age: 12
    amount: 40
    currency: LKR
    ticket_type: entrance
    unit: person
    source_url: https://example.gov.lk/tariff.pdf
    source_authority: Example Authority
    status: official
    last_verified: 2026-11-19
    effective_date: 2026-01-01
"""


def test_a_complete_curated_entry_loads(tmp_path):
    [loaded] = load_curated(write(tmp_path, GOOD))
    assert loaded.amount == 40.0 and loaded.min_age == 6
    assert loaded.last_verified == TODAY
    assert loaded.effective_date == date(2026, 1, 1)
    assert loaded.aliases == ("Example Gardens",)


@pytest.mark.parametrize(
    ("change", "problem"),
    [
        ("    source_url: https://example.gov.lk/tariff.pdf\n", "missing source_url"),
        ("    last_verified: 2026-11-19\n", "missing last_verified"),
        ("    source_authority: Example Authority\n", "missing source_authority"),
    ],
)
def test_an_entry_without_its_source_is_rejected(tmp_path, change, problem):
    with pytest.raises(CuratedPriceError, match=problem):
        load_curated(write(tmp_path, GOOD.replace(change, "")))


@pytest.mark.parametrize(
    ("old", "new", "problem"),
    [
        ("https://example.gov.lk", "http://example.gov.lk", "https://"),
        ("nationality: local", "nationality: tourist", "nationality must be"),
        ("status: official", "status: blog", "status must be"),
        ("amount: 40", "amount: lots", "amount must be a number"),
        ("last_verified: 2026-11-19", "last_verified: last week", "date"),
    ],
)
def test_an_invalid_entry_is_rejected(tmp_path, old, new, problem):
    with pytest.raises(CuratedPriceError, match=problem):
        load_curated(write(tmp_path, GOOD.replace(old, new)))


def test_a_missing_file_has_no_prices(tmp_path):
    assert load_curated(tmp_path / "none.yaml") == []


def test_the_shipped_curated_file_is_valid():
    load_curated(ps.CURATED_FILE)  # raises if an entry is incomplete


# --- Cache and revalidation -------------------------------------------------------


class Server:
    """A fake CCF server that counts requests and can fail."""

    def __init__(self, page: str = CCF_PAGE, status: int = 200):
        self.page, self.status, self.requests = page, status, 0

    def client(self) -> httpx.Client:
        def handle(request):
            self.requests += 1
            return httpx.Response(self.status, text=self.page)

        return ps.make_client(httpx.MockTransport(handle))


def fetch(server: Server, cache: PriceCache, now=NOW, ttl=timedelta(hours=24)):
    with server.client() as client:
        return ccf_prices(client=client, cache=cache, now=now, ttl=ttl)


def test_fresh_cached_prices_need_no_request(tmp_path):
    cache, server = PriceCache(tmp_path / "p.sqlite"), Server()
    first = fetch(server, cache)
    second = fetch(server, cache, now=NOW + timedelta(hours=23))
    assert server.requests == 1
    assert first.records == second.records and second.problem is None


def test_expired_prices_are_read_again(tmp_path):
    cache, server = PriceCache(tmp_path / "p.sqlite"), Server()
    fetch(server, cache)
    later = fetch(server, cache, now=NOW + timedelta(hours=25))
    assert server.requests == 2
    assert later.problem is None
    assert find(later.records, "sigiriya").last_verified == date(2026, 11, 20)


def test_a_failed_recheck_keeps_the_old_prices_and_says_so(tmp_path):
    cache = PriceCache(tmp_path / "p.sqlite")
    fetch(Server(), cache)
    result = fetch(Server(status=503), cache, now=NOW + timedelta(days=2))
    assert find(result.records, "sigiriya").amount == 35.0
    assert "could not be re-checked" in result.problem
    assert "2026-11-19" in result.problem


def test_no_cache_and_no_page_means_no_prices(tmp_path):
    result = fetch(Server(status=500), PriceCache(tmp_path / "p.sqlite"))
    assert result.records == []
    assert "CCF prices unavailable" in result.problem


def test_a_changed_page_layout_is_reported_not_guessed(tmp_path):
    result = fetch(Server(page="<p>new site</p>"), PriceCache(tmp_path / "p.sqlite"))
    assert result.records == [] and "layout" in result.problem


def test_the_cache_survives_a_new_cache_object(tmp_path):
    fetch(Server(), PriceCache(tmp_path / "p.sqlite"))
    reopened = PriceCache(tmp_path / "p.sqlite").get("ccf")
    assert reopened is not None and find(reopened.records, "sigiriya").amount == 35.0


def test_ttl_and_max_age_come_from_the_environment(monkeypatch):
    monkeypatch.setenv(ps.TTL_ENV, "6")
    monkeypatch.setenv(ps.MAX_AGE_ENV, "30")
    assert ps.cache_ttl() == timedelta(hours=6)
    assert ps.max_price_age() == timedelta(days=30)
    monkeypatch.setenv(ps.TTL_ENV, "soon")
    assert ps.cache_ttl() == timedelta(hours=ps.DEFAULT_TTL_HOURS)


def test_load_price_book_puts_curated_prices_first(tmp_path):
    book = ps.load_price_book(
        client=Server().client(),
        cache=PriceCache(tmp_path / "p.sqlite"),
        now=lambda: NOW,
        curated_path=write(tmp_path, GOOD),
    )
    assert book.records[0].attraction_id == "example-park"
    assert book.match("Sigiriya") == "sigiriya"
    assert book.problems == []
