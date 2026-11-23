"""Pydantic schemas: the contracts between our code and the LLM.

The LLM fills these in, and Pydantic checks the result. If the model returns
something that breaks a rule (for example days=0, or pace="super fast"), we get a
ValidationError instead of bad data quietly moving through the program.

Each Field's description is sent to Gemini as part of the JSON schema, so the
descriptions are instructions for the model as well as documentation for us.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# These fields are essential for planning. If one is missing, the assistant should
# ask the traveller. The LangGraph agent will use this list to decide when to ask.
CRITICAL_FIELDS = ("destination", "days")

# What the chat agent (request_understanding.py) asks for before planning, one
# question at a time in this order. The Stage 4 workflow keeps CRITICAL_FIELDS.
# travellers sets the group size; citizenship, party and children decide the
# ticket category of each traveller (see party.py); origin is where day 1
# starts. citizenship, party and children are asked only when party.py says
# they are still open. wants_hotels asks for place-to-stay suggestions; budget
# (a level) is asked only when they want them and gave no nightly amount.
PLAN_FIELDS = (
    "destination",
    "days",
    "travellers",
    "citizenship",
    "party",
    "children",
    "origin",
    "wants_hotels",
    "budget",
)


class PartyMember(BaseModel):
    """Some travellers who share a citizenship and an age group."""

    nationality: str | None = Field(
        default=None,
        description=(
            "Their citizenship as the traveller said it, e.g. 'Sri Lankan', "
            "'Indian', 'German'. Null if not said. Write 'dual' for a dual "
            "citizen. Citizenship, not the country they live in."
        ),
    )
    age_group: Literal["adult", "child"] = Field(
        default="adult", description="child only if they said so."
    )
    count: int = Field(default=1, ge=1, le=50)


class TripRequest(BaseModel):
    """A traveller's trip requirements, extracted from their message."""

    destination: str | None = Field(
        default=None,
        description=(
            "The main place in Sri Lanka the traveller wants to visit, e.g. 'Ella'. "
            "Null if the traveller did not name one."
        ),
    )
    days: int | None = Field(
        default=None,
        ge=1,
        le=60,
        description="Trip length in days. Null if not stated.",
    )
    # The list fields are `| None` only in the schema the LLM sees: models write
    # null for "none stated", and Groq rejects a tool call whose arguments
    # break the schema (400 "expected array, but got null"). _null_to_empty
    # turns null into [], so the value in Python is always a list.
    interests: list[str] | None = Field(
        default_factory=list,
        description="Short lowercase interests, e.g. ['hiking', 'tea plantations'].",
    )
    budget: Literal["budget", "mid-range", "luxury"] | None = Field(
        default=None,
        description="Spending level, only if the traveller indicated one.",
    )
    travel_dates: str | None = Field(
        default=None,
        description=(
            "When they travel, in the traveller's words, e.g. 'March' or "
            "'next weekend'. Null if not stated."
        ),
    )
    travellers: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Number of people travelling, children included. 'Just me' is 1. "
            "Null if not stated."
        ),
    )
    children: int | None = Field(
        default=None,
        ge=0,
        description=(
            "How many of the travellers are children, only if the traveller "
            "said so. 0 when they say everyone is an adult, e.g. 'two Indian "
            "adults'. Null otherwise."
        ),
    )
    pace: Literal["relaxed", "moderate", "packed"] | None = Field(
        default=None,
        description="How busy each day should be, only if indicated.",
    )
    nationality: str | None = Field(
        default=None,
        description=(
            "The citizenship the WHOLE group shares, as the traveller said it, "
            "e.g. 'Sri Lankan', 'Maldivian', 'German'. Null if not said or if "
            "the group has several citizenships. Citizenship, not the country "
            "they live in."
        ),
    )
    citizenship: Literal["local", "saarc", "non_saarc", "mixed"] | None = Field(
        default=None,
        description=(
            "mixed if the group has several citizenships; otherwise null. Do "
            "not classify countries yourself: put the nationality in "
            "'nationality' and the code works out the ticket category."
        ),
    )
    party: list[PartyMember] | None = Field(
        default_factory=list,
        description=(
            "Who is travelling, only when the traveller listed citizenships, "
            "e.g. '2 Sri Lankans and an Indian child'. Empty otherwise."
        ),
    )
    origin: str | None = Field(
        default=None,
        description=(
            "Where in Sri Lanka the trip starts, e.g. 'Colombo' or 'Katunayake' "
            "(the international airport). Null if not stated."
        ),
    )
    wants_hotels: bool | None = Field(
        default=None,
        description=(
            "true if they ask for hotels or where to stay; false if they say "
            "they already have a place or do not want suggestions. Null "
            "otherwise."
        ),
    )
    room_budget: float | None = Field(
        default=None,
        gt=0,
        description=(
            "The most they want to pay for a room per night, only if they gave "
            "an amount, e.g. 10000 for 'up to 10,000 rupees a night'."
        ),
    )
    room_budget_currency: Literal["LKR", "USD"] | None = Field(
        default=None, description="The currency of room_budget, as they said it."
    )

    @field_validator("interests", "party", mode="before")
    @classmethod
    def _null_to_empty(cls, value: object) -> object:
        return [] if value is None else value

    def missing_critical_fields(self) -> list[str]:
        """Names of the critical fields the traveller has not given yet."""
        return [name for name in CRITICAL_FIELDS if getattr(self, name) is None]


# What the traveller's latest message is for. The agent graph routes on this
# (see request_understanding.route_after_understanding).
Intent = Literal["plan", "modify", "question", "off_topic"]


class RequestUnderstanding(BaseModel):
    """The intent of one traveller message, plus the trip details it states."""

    intent: Intent = Field(
        description=(
            "plan: asks for a trip plan or itinerary. "
            "modify: asks to change a plan already given. "
            "question: any other travel question, greeting, or statement about "
            "the traveller (preferences such as diet or pace). "
            "off_topic: clearly not about travel, e.g. homework or coding."
        ),
    )
    trip: TripRequest = Field(
        default_factory=TripRequest,
        description="Trip details stated in the latest message only.",
    )


# --- Stage 8: the structured itinerary ------------------------------------------
#
# Two layers. The agent SUBMITS an ItineraryDraft (the arguments of the
# submit_itinerary tool). compose_answer then checks every factual value against
# the tool results and returns an Itinerary: the same fields, plus the weather
# text taken from the forecast and the warnings the checks produced. The model
# is never asked to fill a field that the code would overwrite.
#
# Nesting is at most three levels (ItineraryDraft -> DayPlanDraft -> Activity),
# and optional values are plain "X | None", which converts cleanly to a Gemini
# function declaration (see tests/test_itinerary_schema.py).

ActivitySource = Literal["OpenStreetMap", "Wikivoyage", "Built-in guide"]
FactSource = Literal["Wikivoyage", "OpenStreetMap", "OSRM", "Open-Meteo"]
Currency = Literal["LKR", "USD"]

_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class Activity(BaseModel):
    """One thing to do on a day, at a place named in a tool result."""

    time_of_day: Literal["morning", "afternoon", "evening"] = Field(
        description="When in the day. No clock times."
    )
    name: str = Field(
        min_length=1,
        description="The place or activity, spelled exactly as in the tool result.",
    )
    description: str = Field(
        description=(
            "One or two sentences on what to do there. No prices, opening hours "
            "or schedules (put those in facts, with a quote)."
        )
    )
    estimated_hours: float | None = Field(
        default=None,
        gt=0,
        le=12,
        description="Your rough planning estimate of the time needed, or null.",
    )
    distance_km: float | None = Field(
        default=None,
        ge=0,
        description=(
            "Straight-line distance from find_attractions (OpenStreetMap) for this "
            "exact place; null if that tool did not give one."
        ),
    )
    source: ActivitySource = Field(description="The tool result that names the place.")


class TravelLeg(BaseModel):
    """A road journey between two places, as estimated by OSRM."""

    from_place: str = Field(min_length=1)
    to_place: str = Field(min_length=1)
    road_distance_km: float | None = Field(
        default=None,
        ge=0,
        description="From estimate_travel_time for these two places; else null.",
    )
    driving_minutes: float | None = Field(
        default=None,
        ge=0,
        description="From estimate_travel_time for these two places; else null.",
    )


class DayPlanDraft(BaseModel):
    """One day of the plan, as the agent submits it."""

    day: int = Field(ge=1, description="Day number, starting at 1.")
    theme: str = Field(min_length=1, description="A short title for the day.")
    forecast_date: str | None = Field(
        default=None,
        description=(
            "The forecast date (YYYY-MM-DD) from get_weather_forecast that this "
            "day falls on; null if the forecast does not cover it. The weather "
            "text is filled in from the forecast automatically."
        ),
    )
    travel: list[TravelLeg] = Field(default_factory=list)
    activities: list[Activity] = Field(min_length=1)

    @field_validator("forecast_date", mode="before")
    @classmethod
    def _date_or_none(cls, value: object) -> object:
        # A malformed date only loses that day's weather; it is not worth a
        # repair call, so it becomes null instead of a validation error.
        if isinstance(value, str) and not _ISO_DATE.match(value.strip()):
            return None
        return value.strip() if isinstance(value, str) else value


class CostItem(BaseModel):
    """A price quoted in the guide, with its conversion (never an entrance fee)."""

    item: str = Field(min_length=1, description="What the price is for.")
    per: Literal["person", "group"] = Field(
        default="group",
        description=(
            "person if the quoted price is for one person; group if it is "
            "shared, e.g. a room, a vehicle or a boat."
        ),
    )
    quantity: int = Field(
        default=1,
        ge=1,
        le=1000,
        description=(
            "How many of these the trip needs, e.g. 2 rooms x 2 nights = 4. "
            "For a per-person price, per person (people are counted for you)."
        ),
    )
    quantity_note: str | None = Field(
        default=None,
        description=(
            "How quantity was worked out, with any assumption, e.g. '2 rooms x "
            "2 nights, assuming 2 people per room'."
        ),
    )
    quoted_currency: Currency = Field(
        description="The currency the guide quotes the price in."
    )
    amount_lkr: float | None = Field(default=None, ge=0)
    amount_usd: float | None = Field(
        default=None,
        ge=0,
        description="The other currency only from convert_currency; else null.",
    )

    @model_validator(mode="after")
    def _quoted_amount_present(self) -> CostItem:
        quoted = self.amount_lkr if self.quoted_currency == "LKR" else self.amount_usd
        if quoted is None:
            raise ValueError(
                f"the quoted amount (amount_{self.quoted_currency.lower()}) is missing"
            )
        return self


class PracticalFact(BaseModel):
    """A factual detail (transport, hours, safety, prices), backed by a quote."""

    text: str = Field(min_length=1, description="The fact, in your own words.")
    quote: str = Field(
        min_length=1,
        description="A short verbatim quote from the tool result that states it.",
    )
    source: FactSource


class ItineraryDraft(BaseModel):
    """The whole plan, as the agent submits it with submit_itinerary."""

    destination: str = Field(min_length=1)
    days: int = Field(ge=1, le=60)
    start_from: str = Field(min_length=1, description="Where the trip starts.")
    start_assumed: bool = Field(
        description="True if the traveller did not say and the default was used."
    )
    summary: str = Field(description="Two or three sentences about the trip.")
    day_plans: list[DayPlanDraft] = Field(min_length=1)
    costs: list[CostItem] = Field(default_factory=list)
    facts: list[PracticalFact] = Field(default_factory=list)
    tips: list[str] = Field(
        default_factory=list,
        description="General suggestions only; no prices, times or schedules.",
    )
    assumptions: list[str] = Field(default_factory=list)
    not_available: list[str] = Field(
        default_factory=list, description="What the tools could not provide."
    )

    @model_validator(mode="after")
    def _days_match_the_plans(self) -> ItineraryDraft:
        if self.days != len(self.day_plans):
            raise ValueError(
                f"days is {self.days} but day_plans has {len(self.day_plans)} entries"
            )
        numbers = [plan.day for plan in self.day_plans]
        if numbers != list(range(1, len(numbers) + 1)):
            raise ValueError(
                f"day numbers must be 1..{len(numbers)} in order, got {numbers}"
            )
        return self


class DayPlan(DayPlanDraft):
    """A checked day: the draft plus the weather taken from the forecast."""

    weather: str | None = None


# --- Filled in by code after the checks (itinerary_budget.py), never by the LLM ---


class FeeCategory(BaseModel):
    """One visitor category's entrance fee at one attraction, as text."""

    label: str  # "Sri Lankan adult"
    text: str  # "LKR 300 (USD 1) · Confidence: High" or "Not verified. ..."
    verified: bool


class FeeLine(BaseModel):
    """What the group's travellers of one category pay at one attraction."""

    people: str  # "2 Sri Lankan adults"
    status: Literal["verified", "estimate", "not_verified"]
    text: str  # "2 Sri Lankan adults x LKR 300 = LKR 600"
    total_lkr: float | None = None
    total_usd: float | None = None


class AttractionFees(BaseModel):
    """Official entrance fees at one attraction, per category and for the group."""

    query: str
    name: str | None  # None: no official price was found
    categories: list[FeeCategory] = Field(default_factory=list)
    group: list[FeeLine] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    conditions: str = ""
    extras: list[str] = Field(default_factory=list)
    verified_lkr: float | None = 0.0
    verified_usd: float | None = 0.0


class CostLine(BaseModel):
    """A checked guide cost, per person and for the whole group."""

    item: str
    per: Literal["person", "group"]
    quantity: int
    quantity_note: str | None = None
    each_lkr: float | None = None
    each_usd: float | None = None
    group_lkr: float | None = None
    group_usd: float | None = None
    per_person_lkr: float | None = None
    per_person_usd: float | None = None


class StayItem(BaseModel):
    """One suggested place to stay, as the guide lists it."""

    name: str
    level: str  # "Budget", "Mid-range", "Luxury", "Other"
    price_text: str = ""
    listed_year: int | None = None
    recent: bool = False  # listed within hotel_service.RECENT_YEARS
    url: str = ""
    description: str = ""


class StaySuggestions(BaseModel):
    """Places to stay from the guide; never booked, availability not checked."""

    source_url: str
    wanted: str = ""  # "mid-range places", "up to LKR 10,000 a room a night"
    items: list[StayItem] = Field(default_factory=list)
    unpriced: list[str] = Field(default_factory=list)
    note: str = ""


class Budget(BaseModel):
    """The trip's money in one place: verified, estimated and missing."""

    group: str = ""  # "2 Sri Lankan adults, 1 SAARC adult"
    travellers: int | None = None
    entrance_verified_lkr: float | None = 0.0
    entrance_verified_usd: float | None = 0.0
    entrance_estimated_lkr: float | None = 0.0
    entrance_estimated_usd: float | None = 0.0
    other_estimated_lkr: float | None = 0.0
    other_estimated_usd: float | None = 0.0
    accommodation_lkr: float | None = None  # upper estimate; None: not included
    accommodation_usd: float | None = None
    accommodation_note: str = ""
    total_lkr: float | None = 0.0
    total_usd: float | None = 0.0
    per_person_lkr: float | None = None  # only when everyone pays the same
    per_person_usd: float | None = None
    not_verified: list[str] = Field(default_factory=list)
    requires_confirmation: list[str] = Field(default_factory=list)
    not_included: list[str] = Field(default_factory=list)
    rate_note: str = ""


class Itinerary(ItineraryDraft):
    """The checked plan that is rendered and saved (see itinerary_output.py)."""

    day_plans: list[DayPlan] = Field(min_length=1)
    warnings: list[str] = Field(default_factory=list)
    entrance_fees: list[AttractionFees] = Field(default_factory=list)
    cost_lines: list[CostLine] = Field(default_factory=list)
    stays: StaySuggestions | None = None
    budget: Budget | None = None
