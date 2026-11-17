"""Stage 6 checks: understanding requests and asking for missing details.

Offline only. A dictionary-lookup understander plays the extraction LLM and a
scripted fake model plays the agent, so no Gemini calls are made.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ai_travel_assistant.schemas import RequestUnderstanding, TripRequest

# --- Schema -------------------------------------------------------------------


def test_understanding_defaults_to_an_empty_trip():
    understanding = RequestUnderstanding(intent="question")
    assert understanding.trip == TripRequest()


def test_understanding_rejects_an_unknown_intent():
    with pytest.raises(ValidationError):
        RequestUnderstanding(intent="booking")


def test_understanding_requires_an_intent():
    assert RequestUnderstanding.model_json_schema()["required"] == ["intent"]
