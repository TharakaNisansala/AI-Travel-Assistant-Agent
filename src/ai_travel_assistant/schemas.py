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
# they are still open.
PLAN_FIELDS = (
    "destination",
    "days",
    "travellers",
    "citizenship",
    "party",
    "children",
    "origin",
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
    interests: list[str] = Field(
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
            "said so. Null otherwise."
        ),
    )
    pace: Literal["relaxed", "moderate", "packed"] | None = Field(
        default=None,
        description="How busy each day should be, only if indicated.",
    )
    citizenship: Literal["local", "saarc", "non_saarc", "mixed"] | None = Field(
        default=None,
        description=(
            "Only when the WHOLE group shares one: local = Sri Lankan citizens, "
            "saarc = citizens of another SAARC country, non_saarc = citizens of "
            "any other country, mixed = several. Null if not said. Living in "
            "Sri Lanka does not make a foreigner local."
        ),
    )
    party: list[PartyMember] = Field(
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
    """A price quoted in the guide, with its conversion."""

    item: str = Field(min_length=1, description="What the price is for.")
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


class Itinerary(ItineraryDraft):
    """The checked plan that is rendered and saved (see itinerary_output.py)."""

    day_plans: list[DayPlan] = Field(min_length=1)
    warnings: list[str] = Field(default_factory=list)
