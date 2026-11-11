"""Stage 1 - Lesson 3 checks.

The offline tests use a fake model. The live test calls Gemini and runs only
with:  uv run pytest -m live
"""

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from ai_travel_assistant import structured_output
from ai_travel_assistant.schemas import TripRequest

# --- Schema validation --------------------------------------------------------


def test_empty_trip_request_is_valid_and_all_unknown():
    trip = TripRequest()
    assert trip.destination is None
    assert trip.interests == []
    assert trip.missing_critical_fields() == ["destination", "days"]


def test_valid_trip_request_parses_from_json():
    trip = TripRequest.model_validate_json(
        '{"destination": "Ella", "days": 3, "interests": ["hiking"], '
        '"budget": "mid-range", "travellers": 2, "pace": "relaxed"}'
    )
    assert trip.destination == "Ella"
    assert trip.days == 3
    assert trip.missing_critical_fields() == []


@pytest.mark.parametrize(
    "bad",
    [
        {"days": 0},  # below ge=1
        {"days": 365},  # above le=60
        {"travellers": 0},
        {"budget": "cheap"},  # not one of the allowed values
        {"pace": "super fast"},
        {"days": "three"},  # not an integer
    ],
)
def test_invalid_values_are_rejected(bad):
    with pytest.raises(ValidationError):
        TripRequest.model_validate(bad)


def test_schema_descriptions_reach_the_json_schema():
    # These descriptions are what Gemini reads, so they must be in the schema.
    props = TripRequest.model_json_schema()["properties"]
    assert "Null if" in props["destination"]["description"]
    assert set(props) == {
        "destination",
        "days",
        "interests",
        "budget",
        "travel_dates",
        "travellers",
        "pace",
    }


# --- Extraction with a fake model ---------------------------------------------


class FakeStructuredModel(FakeListChatModel):
    """Fakes with_structured_output: parses a scripted JSON reply into the schema."""

    received: list = []
    schemas: list = []

    def with_structured_output(self, schema, **kwargs):
        self.schemas.append(schema)

        def respond(messages):
            self.received.append(messages)
            return schema.model_validate_json(self.responses[0])

        return RunnableLambda(respond)


def test_extract_returns_trip_request_and_sends_instructions():
    fake = FakeStructuredModel(responses=['{"destination": "Ella", "days": 3}'])
    trip = structured_output.extract_trip_request(fake, "3 days in Ella")

    assert trip == TripRequest(destination="Ella", days=3)
    assert fake.schemas == [TripRequest]
    [messages] = fake.received
    assert isinstance(messages[0], SystemMessage)
    assert messages[0].content == structured_output.EXTRACTION_INSTRUCTIONS
    assert messages[1] == HumanMessage("3 days in Ella")


def test_extract_sends_the_question_and_known_details_with_a_short_reply():
    fake = FakeStructuredModel(responses=['{"days": 3}'])
    trip = structured_output.extract_trip_request(
        fake,
        "3",
        clarification_context="How many days would you like to spend in Kandy?",
        existing_data={"destination": "Kandy"},
    )

    assert trip == TripRequest(days=3)
    messages = fake.received[-1]
    assert messages[0].content == structured_output.EXTRACTION_INSTRUCTIONS
    assert messages[1] == HumanMessage(
        "Trip details already known: destination=Kandy\n"
        "Question the assistant just asked: "
        "How many days would you like to spend in Kandy?\n"
        "Traveller's latest reply: 3"
    )


def test_extraction_message_is_plain_text_without_context():
    assert structured_output.build_extraction_message("Kandy") == "Kandy"


def test_instructions_explain_how_to_use_the_question():
    rules = structured_output.EXTRACTION_INSTRUCTIONS
    assert "question the assistant just asked" in rules
    assert '"We are 3" is travellers, not days' in rules
    assert "Do not copy known details" in rules


def test_extract_surfaces_invalid_model_output():
    fake = FakeStructuredModel(responses=['{"days": -2}'])
    with pytest.raises(ValidationError):
        structured_output.extract_trip_request(fake, "minus two days?")


def test_extract_rejects_none_result():
    class NoneModel(FakeListChatModel):
        def with_structured_output(self, schema, **kwargs):
            return RunnableLambda(lambda _: None)

    with pytest.raises(ValueError, match="Expected a TripRequest"):
        structured_output.extract_trip_request(NoneModel(responses=[""]), "hi")


def test_schema_flag_prints_without_api(capsys):
    assert structured_output.main(["--schema"]) == 0
    assert '"destination"' in capsys.readouterr().out


def test_sample_prompts_has_ten_cases():
    assert len(structured_output.SAMPLE_PROMPTS) == 10


# --- Live accuracy check (calls Gemini) ---------------------------------------


@pytest.mark.live
def test_live_extraction_accuracy():
    import time

    from ai_travel_assistant.llm_basics import build_llm, load_settings

    llm = build_llm(load_settings())
    correct = 0
    failures = []
    for i, (text, want_dest, want_days) in enumerate(structured_output.SAMPLE_PROMPTS):
        if i > 0:
            time.sleep(structured_output.FREE_TIER_DELAY_SECONDS)
        trip = structured_output.extract_trip_request(llm, text)
        dest_ok = (trip.destination is None) == (want_dest is None) and (
            want_dest is None or want_dest.lower() in trip.destination.lower()
        )
        if dest_ok and trip.days == want_days:
            correct += 1
        else:
            failures.append((text, trip.destination, trip.days))
    assert correct >= 9, f"{correct}/10 correct; failures: {failures}"
