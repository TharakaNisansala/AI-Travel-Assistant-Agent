"""Pydantic schemas: the contracts between our code and the LLM.

The LLM fills these in, and Pydantic checks the result. If the model returns
something that breaks a rule (for example days=0, or pace="super fast"), we get a
ValidationError instead of bad data quietly moving through the program.

Each Field's description is sent to Gemini as part of the JSON schema, so the
descriptions are instructions for the model as well as documentation for us.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# These fields are essential for planning. If one is missing, the assistant should
# ask the traveller. The LangGraph agent will use this list to decide when to ask.
CRITICAL_FIELDS = ("destination", "days")


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
        description="Number of people travelling. Null if not stated.",
    )
    pace: Literal["relaxed", "moderate", "packed"] | None = Field(
        default=None,
        description="How busy each day should be, only if indicated.",
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
