"""Stage 1 - Lesson 2 checks. No network calls: fake chat models stand in."""

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ai_travel_assistant import messages_demo


class RecordingFakeModel(FakeListChatModel):
    """A fake model that also remembers the messages it was sent."""

    received: list = []

    def _call(self, messages, stop=None, run_manager=None, **kwargs):
        self.received.append(messages)
        return super()._call(messages, stop, run_manager, **kwargs)


def test_string_prompt_becomes_a_human_message():
    fake = RecordingFakeModel(responses=["Visit Galle Fort."])
    reply = messages_demo.ask(fake, messages_demo.build_example_1())

    assert isinstance(reply, AIMessage)
    assert reply.content == "Visit Galle Fort."
    [sent] = fake.received
    assert len(sent) == 1
    assert isinstance(sent[0], HumanMessage)


def test_example_2_has_system_then_human():
    messages = messages_demo.build_example_2()
    assert [m.type for m in messages] == ["system", "human"]


def test_travel_prompt_puts_instructions_first():
    messages = messages_demo.build_travel_messages("Where should I go?")

    assert [m.type for m in messages] == ["system", "human"]
    assert messages[0].content == messages_demo.TRAVEL_ASSISTANT_INSTRUCTIONS
    assert messages[1].content == "Where should I go?"


def test_instructions_cover_the_required_behaviour():
    text = messages_demo.TRAVEL_ASSISTANT_INSTRUCTIONS.lower()
    assert "sri lanka" in text
    assert "interests" in text
    assert "clarification" in text
    assert "do not invent" in text


def test_example_4_keeps_history_in_order():
    messages = messages_demo.build_example_4()

    assert [type(m) for m in messages] == [
        SystemMessage,
        HumanMessage,  # earlier turn
        AIMessage,  # earlier reply
        HumanMessage,  # follow-up
    ]
    assert messages[1:3] == messages_demo.EXAMPLE_4_HISTORY
    assert "there" in messages[-1].content


def test_example_4_sends_full_history_to_model():
    fake = RecordingFakeModel(responses=["Take the train to Ella."])
    messages_demo.ask(fake, messages_demo.build_example_4())

    [sent] = fake.received
    assert len(sent) == 4
    assert sent[2].content == messages_demo.EXAMPLE_4_HISTORY[1].content


def test_example_5_differs_only_by_system_message():
    variants = messages_demo.build_example_5()
    without = variants["without system instruction"]
    with_system = variants["with system instruction"]

    assert [m.type for m in without] == ["human"]
    assert [m.type for m in with_system] == ["system", "human"]
    assert without[0].content == with_system[1].content


def test_dry_run_prints_messages_without_a_model(capsys):
    assert messages_demo.main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "[system]" in out
    assert "[human]" in out
    assert "[ai]" in out  # the AIMessage from Example 4's history
    assert "[plain string]" in out


def test_run_example_prints_fake_reply(capsys):
    fake = FakeListChatModel(responses=["Try Mirissa.", "Which month?"])
    messages_demo.run_example(fake, "5")
    out = capsys.readouterr().out
    assert "Try Mirissa." in out
    assert "Which month?" in out
