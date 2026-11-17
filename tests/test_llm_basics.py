"""Stage 1 - Lesson 1 checks. No network calls: a fake chat model stands in."""

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from ai_travel_assistant import llm_basics


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch):
    # Stop load_settings() from reading the real .env during tests.
    monkeypatch.setattr(llm_basics, "load_dotenv", lambda: None)


def test_missing_settings_raise(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    with pytest.raises(llm_basics.ConfigError, match="GEMINI_API_KEY"):
        llm_basics.load_settings()


def test_settings_hide_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-123")
    monkeypatch.setenv("GEMINI_MODEL", "some-flash-model")
    settings = llm_basics.load_settings()
    assert settings.model == "some-flash-model"
    assert "fake-key-123" not in repr(settings)


def test_build_llm_sets_temperature_without_calling_api():
    settings = llm_basics.GeminiSettings(model="some-flash-model", api_key="fake")
    llm = llm_basics.build_llm(settings, temperature=0.3)
    assert llm.temperature == 0.3
    assert llm.model.endswith("some-flash-model")


def test_build_llm_keeps_three_client_retries_unless_told_otherwise():
    settings = llm_basics.GeminiSettings(model="some-flash-model", api_key="fake")
    assert llm_basics.build_llm(settings).max_retries == 3
    assert llm_basics.build_llm(settings, max_retries=0).max_retries == 0


def test_content_to_text_handles_str_and_blocks():
    assert llm_basics.content_to_text("Hello") == "Hello"
    blocks = [
        {"type": "text", "text": "Ayubowan ", "extras": {"signature": "xyz"}},
        {"type": "reasoning", "reasoning": "hidden"},
        {"type": "text", "text": "Sri Lanka!"},
    ]
    assert llm_basics.content_to_text(blocks) == "Ayubowan Sri Lanka!"


def test_ask_returns_ai_message():
    fake = FakeListChatModel(responses=["Visit Sigiriya."])
    response = llm_basics.ask(fake, "Where should I go?")
    assert response.content == "Visit Sigiriya."


def test_safe_error_message_redacts_key():
    msg = llm_basics.safe_error_message(ValueError("bad key secret123"), "secret123")
    assert "secret123" not in msg
    assert "ValueError" in msg
