"""Verified ticket prices for Sri Lankan attractions, one record per visitor category.

Prices change, differ by citizenship and age, and are easy for an LLM to
invent. This service keeps them out of the model's memory: every price comes
from a named source with a date, and a price that cannot be matched to the
traveller's exact category is reported as "not verified" instead of guessed.

What this file demonstrates:
  1. One record per (attraction, nationality, age group). Never one generic
     fee: a Sri Lankan adult, a SAARC adult and a German child are priced
     separately, and one category's price is never used for another.
  2. Two kinds of source, both with provenance (URL, authority, dates):
       - CCF live: the Central Cultural Fund publishes its foreign ticket
         prices as an HTML table with a "Last Updated" date. parse_ccf()
         reads it. It lists one foreign price (no SAARC split, no local
         price) and a half ticket without an age rule.
       - Curated: attraction_prices.yaml, entered by a person from official
         notices, gazettes or tariff PDFs (DWC, Botanic Gardens, Museums,
         local authorities). Every entry must carry its source URL,
         authority and verification date, or loading fails.
  3. Caching with revalidation. CCF results are kept in SQLite for
     PRICE_CACHE_TTL_HOURS; after that the page is fetched again. If the
     fetch fails, the old prices are still shown, flagged as not re-checked.
  4. Freshness is not the same as being on a website: a price whose source
     was last updated (or verified) more than PRICE_MAX_AGE_DAYS ago loses
     a confidence level and says so.
  5. price_for() holds the eligibility rules in one pure function:
       - exact category match: verified;
       - non-SAARC traveller, source gives one "foreign" price: used, at most
         Medium confidence, with a note that SAARC rates are not stated;
       - SAARC traveller, only a general foreign price: NOT verified;
       - child: verified only when the source publishes the age limit;
       - expired or not-yet-effective prices: not verified.

No LLM here. The get_attraction_prices tool (tools.py) formats a PriceBook
lookup for the agent; trip_costs.py does the arithmetic.
"""

from __future__ import annotations

import html
import json
import os
import re
import sqlite3
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

import httpx
import yaml

from ai_travel_assistant import faults
from ai_travel_assistant.resilience import send_with_retry

Nationality = Literal["local", "saarc", "non_saarc", "foreign"]
AgeGroup = Literal["adult", "child", "student", "senior"]
Status = Literal["official", "operator", "secondary"]
Unit = Literal["person", "vehicle", "group", "activity"]
Confidence = Literal["High", "Medium", "Low"]

# "foreign" is a source's single price for every non-Sri Lankan visitor.
NATIONALITIES = ("local", "saarc", "non_saarc", "foreign")
AGE_GROUPS = ("adult", "child", "student", "senior")
STATUSES = ("official", "operator", "secondary")
UNITS = ("person", "vehicle", "group", "activity")
TICKET_TYPES = ("entrance", "activity", "vehicle", "parking", "other")
CURRENCIES = ("LKR", "USD")

CCF_URL = "https://ccf.gov.lk/is/"
CCF_AUTHORITY = "Central Cultural Fund (CCF), Sri Lanka"
CCF_SOURCE = "ccf"

CURATED_FILE = Path(__file__).with_name("data") / "attraction_prices.yaml"

CACHE_ENV = "PRICE_CACHE_DB"
TTL_ENV = "PRICE_CACHE_TTL_HOURS"
MAX_AGE_ENV = "PRICE_MAX_AGE_DAYS"
DEFAULT_CACHE_DB = Path(".data/prices.sqlite")
DEFAULT_TTL_HOURS = 24.0
DEFAULT_MAX_AGE_DAYS = 90

REQUEST_TIMEOUT = 20.0
USER_AGENT = "ai-travel-assistant/0.1 (learning project; ticket price check)"


class PriceSourceError(Exception):
    """A price source could not be read (network, HTTP error or a new layout)."""


class CuratedPriceError(ValueError):
    """An entry in the curated price file is incomplete or invalid."""


# --- The record -----


@dataclass(frozen=True)
class PriceRecord:
    """One price, for one visitor category, with where it came from."""

    attraction_id: str
    attraction_name: str
    nationality: Nationality
    age_group: AgeGroup
    amount: float
    currency: str
    ticket_type: str
    unit: Unit
    source_url: str
    source_authority: str
    last_verified: date  # when the source was last read (CCF) or checked (curated)
    status: Status
    conditions: str = ""
    min_age: int | None = None
    max_age: int | None = None
    source_updated: date | None = None  # the source's own "last updated" date
    effective_date: date | None = None
    expiry_date: date | None = None
    lkr_equivalent: float | None = None  # the source's own LKR figure, if given
    exchange_rate: float | None = None  # LKR per USD the source used
    aliases: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        for key, value in data.items():
            if isinstance(value, date):
                data[key] = value.isoformat()
        data["aliases"] = list(self.aliases)
        return data

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> PriceRecord:
        data = dict(data)
        for key in (
            "last_verified",
            "source_updated",
            "effective_date",
            "expiry_date",
        ):
            if data.get(key):
                data[key] = date.fromisoformat(data[key])
        data["aliases"] = tuple(data.get("aliases") or ())
        return cls(**data)


# --- Names -----

_STOP_WORDS = {"the", "of", "a", "an", "and", "at", "in", "to"}


def _words(text: str) -> list[str]:
    text = text.lower().replace("–", "-").replace("—", "-")
    return [w for w in re.findall(r"[a-z0-9]+", text) if w not in _STOP_WORDS]


def attraction_id(name: str) -> str:
    """'Galle (Museum)' -> 'galle-museum'."""
    return "-".join(_words(name))


# --- CCF: parse the published table -----

_CELL = re.compile(r"<t([dh])[^>]*>(.*?)</t\1>", re.S | re.I)
_ROW = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S | re.I)
_RATE = re.compile(r"Exchange Rate:\s*([\d.,]+)\s*LKR\s+per\s+USD", re.I)
_UPDATED = re.compile(r"Last Updated:\s*([A-Za-z]+\s+\d{1,2},\s*\d{4})", re.I)


def _cell_text(raw: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", raw)).strip()


def _number(text: str) -> float | None:
    text = text.replace(",", "").strip()
    if not text or text == "-":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_ccf(page: str, checked_on: date) -> list[PriceRecord]:
    """The CCF ticket table as PriceRecords (foreign adult and half tickets).

    Columns: Site, Full Ticket (USD), Full Ticket (LKR), Half Ticket (USD),
    Half Ticket (LKR). A row without prices ("Galle") heads the rows below it
    that start with "-" (activities such as diving tours).
    """
    rate_match = _RATE.search(page)
    rate = _number(rate_match.group(1)) if rate_match else None
    updated_match = _UPDATED.search(page)
    updated = None
    if updated_match:
        text = re.sub(r"\s+", " ", updated_match.group(1))
        try:
            updated = datetime.strptime(text, "%B %d, %Y").date()
        except ValueError:
            updated = None

    rate_note = (
        f"Prices are set in USD; the LKR figure uses CCF's rate of {rate:.2f} LKR "
        "per USD."
        if rate
        else "Prices are set in USD."
    )
    records: list[PriceRecord] = []
    heading: str | None = None
    for row in _ROW.findall(page):
        cells = [(kind, _cell_text(raw)) for kind, raw in _CELL.findall(row)]
        if len(cells) != 5 or any(kind.lower() == "h" for kind, _ in cells):
            continue  # the header row, or a row of another shape
        name = cells[0][1]
        full_usd, full_lkr, half_usd, half_lkr = (_number(t) for _, t in cells[1:])
        if full_usd is None and half_usd is None:
            heading = name  # e.g. "Galle", over its diving tours
            continue
        is_activity = name.startswith("-")
        if is_activity:
            label = name.lstrip("- ").strip()
            name = f"{heading}: {label}" if heading else label
        else:
            heading = None
        common = {
            "attraction_id": attraction_id(name),
            "attraction_name": name,
            "nationality": "foreign",
            "currency": "USD",
            "ticket_type": "activity" if is_activity else "entrance",
            "unit": "person",
            "source_url": CCF_URL,
            "source_authority": CCF_AUTHORITY,
            "last_verified": checked_on,
            "status": "official",
            "source_updated": updated,
            "exchange_rate": rate,
        }
        foreign_note = (
            "CCF publishes one foreign price; it does not say whether SAARC "
            "citizens pay a different rate, and lists no Sri Lankan price. "
        )
        if full_usd is not None:
            records.append(
                PriceRecord(
                    **common,
                    age_group="adult",
                    amount=full_usd,
                    lkr_equivalent=full_lkr,
                    conditions=foreign_note + rate_note,
                )
            )
        if half_usd is not None:
            records.append(
                PriceRecord(
                    **common,
                    age_group="child",
                    amount=half_usd,
                    lkr_equivalent=half_lkr,
                    conditions=(
                        "CCF lists this as a half ticket but does not publish "
                        "who qualifies (no age limit given). " + rate_note
                    ),
                )
            )
    if not records:
        raise PriceSourceError(
            "The CCF price page has no readable price table (its layout may "
            "have changed)."
        )
    return records


# --- Curated prices: entered by a person, with their source -----

_REQUIRED = (
    "attraction_id",
    "attraction_name",
    "nationality",
    "age_group",
    "amount",
    "currency",
    "ticket_type",
    "unit",
    "source_url",
    "source_authority",
    "last_verified",
    "status",
)


def _as_date(value: object, what: str) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise CuratedPriceError(f"{what} must be a date (YYYY-MM-DD)") from exc


def _curated_record(entry: dict, where: str) -> PriceRecord:
    def fail(problem: str) -> CuratedPriceError:
        return CuratedPriceError(f"{where}: {problem}")

    if not isinstance(entry, dict):
        raise fail("each price must be a mapping of fields")
    missing = [name for name in _REQUIRED if entry.get(name) in (None, "")]
    if missing:
        raise fail(f"missing {', '.join(missing)}")
    checks = (
        ("nationality", NATIONALITIES),
        ("age_group", AGE_GROUPS),
        ("status", STATUSES),
        ("unit", UNITS),
        ("ticket_type", TICKET_TYPES),
        ("currency", CURRENCIES),
    )
    for name, allowed in checks:
        if entry[name] not in allowed:
            raise fail(f"{name} must be one of {', '.join(allowed)}")
    if not str(entry["source_url"]).startswith("https://"):
        raise fail("source_url must be an https:// link to the source")
    try:
        amount = float(entry["amount"])
    except (TypeError, ValueError) as exc:
        raise fail("amount must be a number") from exc
    if amount < 0:
        raise fail("amount must not be negative")
    try:
        record = PriceRecord(
            attraction_id=str(entry["attraction_id"]),
            attraction_name=str(entry["attraction_name"]),
            nationality=entry["nationality"],
            age_group=entry["age_group"],
            amount=amount,
            currency=entry["currency"],
            ticket_type=entry["ticket_type"],
            unit=entry["unit"],
            source_url=str(entry["source_url"]),
            source_authority=str(entry["source_authority"]),
            last_verified=_as_date(entry["last_verified"], "last_verified"),  # type: ignore[arg-type]
            status=entry["status"],
            conditions=str(entry.get("conditions") or ""),
            min_age=entry.get("min_age"),
            max_age=entry.get("max_age"),
            source_updated=_as_date(entry.get("source_updated"), "source_updated"),
            effective_date=_as_date(entry.get("effective_date"), "effective_date"),
            expiry_date=_as_date(entry.get("expiry_date"), "expiry_date"),
            aliases=tuple(entry.get("aliases") or ()),
        )
    except CuratedPriceError as exc:
        raise fail(str(exc)) from exc
    return record


def load_curated(path: Path = CURATED_FILE) -> list[PriceRecord]:
    """Every price in the curated file; an incomplete entry stops the load."""
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = data.get("prices") or [] if isinstance(data, dict) else None
    if not isinstance(entries, list):
        raise CuratedPriceError(f"{path.name}: expected a 'prices:' list")
    return [
        _curated_record(entry, f"{path.name}, price {i + 1}")
        for i, entry in enumerate(entries)
    ]


# --- Cache -----


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, default))
    except ValueError:
        return default
    return value if value > 0 else default


def cache_ttl() -> timedelta:
    return timedelta(hours=_env_float(TTL_ENV, DEFAULT_TTL_HOURS))


def max_price_age() -> timedelta:
    return timedelta(days=_env_float(MAX_AGE_ENV, DEFAULT_MAX_AGE_DAYS))


@dataclass
class CachedPrices:
    fetched_at: datetime
    records: list[PriceRecord]


class PriceCache:
    """Prices per source in a small SQLite file (survives restarts)."""

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path or os.environ.get(CACHE_ENV) or DEFAULT_CACHE_DB)

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS prices "
            "(source TEXT PRIMARY KEY, fetched_at TEXT NOT NULL, records TEXT NOT NULL)"
        )
        return conn

    def get(self, source: str) -> CachedPrices | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT fetched_at, records FROM prices WHERE source = ?", (source,)
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        try:
            records = [PriceRecord.from_json(r) for r in json.loads(row[1])]
            return CachedPrices(datetime.fromisoformat(row[0]), records)
        except (ValueError, TypeError, KeyError):
            return None  # an unreadable entry is treated as no entry

    def put(self, source: str, fetched_at: datetime, records: Iterable[PriceRecord]):
        payload = json.dumps([r.to_json() for r in records])
        conn = self._connect()
        try:
            with conn:
                conn.execute(
                    "INSERT OR REPLACE INTO prices VALUES (?, ?, ?)",
                    (source, fetched_at.isoformat(), payload),
                )
        finally:
            conn.close()


# --- Fetching CCF with revalidation -----


def make_client(transport: httpx.BaseTransport | None = None) -> httpx.Client:
    return httpx.Client(
        timeout=REQUEST_TIMEOUT,
        headers={"User-Agent": USER_AGENT},
        follow_redirects=True,
        transport=transport or faults.transport_for("pricing"),
    )


def _get_page(client: httpx.Client, url: str) -> str:
    try:
        response = send_with_retry(lambda: client.get(url))
    except httpx.HTTPError as exc:
        raise PriceSourceError(
            f"Could not reach {url} ({type(exc).__name__})."
        ) from exc
    if response.is_error:
        raise PriceSourceError(f"{url} returned HTTP {response.status_code}.")
    return response.text


@dataclass
class SourceResult:
    records: list[PriceRecord]
    problem: str | None = None  # why the source could not be (re)read


def _now() -> datetime:
    return datetime.now(timezone.utc)


def ccf_prices(
    *,
    client: httpx.Client,
    cache: PriceCache,
    now: datetime,
    ttl: timedelta,
) -> SourceResult:
    """CCF prices from the cache while fresh; otherwise read the page again.

    If reading fails, older cached prices are still returned with a problem
    note, so a CCF outage never turns a known price into a guess.
    """
    cached = cache.get(CCF_SOURCE)
    if cached is not None and now - cached.fetched_at < ttl:
        return SourceResult(cached.records)
    try:
        records = parse_ccf(_get_page(client, CCF_URL), now.date())
    except PriceSourceError as exc:
        if cached is None:
            return SourceResult([], f"CCF prices unavailable: {exc}")
        return SourceResult(
            cached.records,
            f"CCF prices could not be re-checked today ({exc}); showing the "
            f"prices read on {cached.fetched_at.date().isoformat()}.",
        )
    cache.put(CCF_SOURCE, now, records)
    return SourceResult(records)


# --- Choosing the price for one visitor category -----


@dataclass(frozen=True)
class CategoryPrice:
    """The price for one (nationality, age group), or why there is none."""

    nationality: str
    age_group: str
    record: PriceRecord | None
    confidence: Confidence | None
    note: str = ""

    @property
    def verified(self) -> bool:
        return self.record is not None


_LOWER = {"High": "Medium", "Medium": "Low", "Low": "Low"}
_BY_STATUS: dict[str, Confidence] = {
    "official": "High",
    "operator": "Medium",
    "secondary": "Low",
}


def _age_note(record: PriceRecord, today: date, max_age: timedelta) -> str | None:
    """Why a record may be out of date, or None if it is fresh enough."""
    dated = record.source_updated or record.last_verified
    if today - dated > max_age:
        return f"the source was last updated on {dated.isoformat()}"
    return None


def price_for(
    records: Iterable[PriceRecord],
    nationality: str,
    age_group: str,
    *,
    today: date,
    max_age: timedelta | None = None,
    ticket_type: str = "entrance",
) -> CategoryPrice:
    """The price one visitor category pays, applying the eligibility rules."""
    max_age = max_age or max_price_age()
    usable: list[PriceRecord] = []
    expired: list[PriceRecord] = []
    for record in records:
        if record.ticket_type != ticket_type:
            continue
        if (record.expiry_date and record.expiry_date < today) or (
            record.effective_date and record.effective_date > today
        ):
            expired.append(record)
            continue
        usable.append(record)

    def none(note: str) -> CategoryPrice:
        return CategoryPrice(nationality, age_group, None, None, note)

    exact = [
        r for r in usable if (r.nationality, r.age_group) == (nationality, age_group)
    ]
    general = [
        r
        for r in usable
        if r.nationality == "foreign"
        and r.age_group == age_group
        and nationality in ("saarc", "non_saarc")
    ]
    if not exact and not general and expired:
        dates = ", ".join(
            (r.expiry_date or r.effective_date).isoformat()  # type: ignore[union-attr]
            for r in expired
        )
        return none(f"The published price is not in force today ({dates}).")
    record = (exact or general or [None])[0]
    if record is None:
        if nationality == "local":
            return none("No official Sri Lankan price was found.")
        return none("No official price was found for this category.")
    if nationality == "saarc" and record.nationality == "foreign":
        return none(
            f"{record.source_authority} does not publish a SAARC rate; its general "
            f"foreign price is {record.currency} {record.amount:g}."
        )
    if age_group == "child" and record.min_age is None and record.max_age is None:
        return none(
            f"{record.source_authority} lists a child or half ticket of "
            f"{record.currency} {record.amount:g} but does not publish who "
            "qualifies, so it is not applied."
        )

    confidence: Confidence = _BY_STATUS[record.status]
    notes = []
    if record.nationality == "foreign":
        confidence = "Medium" if confidence == "High" else confidence
        notes.append("the source gives one foreign price and no SAARC split")
    stale = _age_note(record, today, max_age)
    if stale:
        confidence = _LOWER[confidence]  # type: ignore[assignment]
        notes.append(f"may be out of date: {stale}")
    note = "; ".join(notes)
    return CategoryPrice(
        nationality, age_group, record, confidence, note[:1].upper() + note[1:]
    )


# --- The price book: every source, looked up by attraction name -----


@dataclass
class AttractionPrices:
    query: str
    attraction_id: str | None
    attraction_name: str | None
    records: list[PriceRecord] = field(default_factory=list)


@dataclass
class PriceBook:
    """All prices currently known, plus any source problems to report."""

    records: list[PriceRecord]
    problems: list[str] = field(default_factory=list)

    def _names(self) -> dict[str, set[str]]:
        """attraction_id -> every name and alias it can be found by (as ids)."""
        names: dict[str, set[str]] = {}
        for record in self.records:
            keys = names.setdefault(record.attraction_id, {record.attraction_id})
            keys.add(attraction_id(record.attraction_name))
            keys.update(attraction_id(alias) for alias in record.aliases)
        return names

    def match(self, query: str) -> str | None:
        """The attraction a name refers to, or None.

        A name matches when every word of the attraction's name (or one of its
        aliases) is in the query: "Sigiriya Rock Fortress" finds "Sigiriya",
        but "Dambulla Cave Temple" does not find "Dambulla (Museum)". The
        longest such name wins ("Sigiriya Museum" -> the museum).
        """
        query_words = set(_words(query))
        best: tuple[int, str] | None = None
        for aid, keys in self._names().items():
            for key in keys:
                words = set(key.split("-"))
                if words and words <= query_words:
                    score = (len(words), aid)
                    if best is None or score[0] > best[0]:
                        best = score
        return best[1] if best else None

    def lookup(self, query: str) -> AttractionPrices:
        aid = self.match(query)
        if aid is None:
            return AttractionPrices(query, None, None)
        records = [r for r in self.records if r.attraction_id == aid]
        return AttractionPrices(query, aid, records[0].attraction_name, records)


def load_price_book(
    *,
    client: httpx.Client | None = None,
    cache: PriceCache | None = None,
    now: Callable[[], datetime] = _now,
    curated_path: Path = CURATED_FILE,
) -> PriceBook:
    """Read every source. Curated prices come first, so they win ties."""
    curated = load_curated(curated_path)
    own_client = client is None
    client = client or make_client()
    try:
        ccf = ccf_prices(
            client=client, cache=cache or PriceCache(), now=now(), ttl=cache_ttl()
        )
    finally:
        if own_client:
            client.close()
    problems = [ccf.problem] if ccf.problem else []
    return PriceBook([*curated, *ccf.records], problems)
