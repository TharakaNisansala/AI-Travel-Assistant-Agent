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


# --- Choosing the provider (LLM_PROVIDER) ----------------------------------------


def _clear_provider_env(monkeypatch):
    for name in (
        "LLM_PROVIDER",
        "GEMINI_API_KEY",
        "GEMINI_MODEL",
        "GROQ_API_KEY",
        "GROQ_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)


def test_provider_defaults_to_gemini(monkeypatch):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "g-key")
    monkeypatch.setenv("GEMINI_MODEL", "some-flash-model")
    assert llm_basics.load_settings().provider == "gemini"


def test_groq_provider_reads_groq_variables(monkeypatch):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", " Groq ")
    monkeypatch.setenv("GROQ_API_KEY", "groq-key-123")
    monkeypatch.setenv("GROQ_MODEL", "some-groq-model")
    settings = llm_basics.load_settings()
    assert (settings.provider, settings.model) == ("groq", "some-groq-model")
    assert "groq-key-123" not in repr(settings)


def test_groq_provider_needs_groq_key_not_gemini_key(monkeypatch):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GEMINI_API_KEY", "g-key")
    monkeypatch.setenv("GEMINI_MODEL", "some-flash-model")
    with pytest.raises(llm_basics.ConfigError, match="GROQ_API_KEY, GROQ_MODEL"):
        llm_basics.load_settings()


def test_unknown_provider_raises(monkeypatch):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "nope")
    with pytest.raises(llm_basics.ConfigError, match="Unknown LLM_PROVIDER"):
        llm_basics.load_settings()


def test_build_llm_makes_a_groq_model_without_calling_api():
    from langchain_groq import ChatGroq

    settings = llm_basics.LLMSettings(
        model="some-groq-model", api_key="fake", provider="groq"
    )
    llm = llm_basics.build_llm(settings, temperature=0.2, max_retries=0)
    assert isinstance(llm, ChatGroq)
    assert llm.model_name == "some-groq-model"
    assert (llm.temperature, llm.max_retries) == (0.2, 0)


def test_agent_client_retries_per_provider():
    gemini = llm_basics.LLMSettings(model="m", api_key="k")
    groq = llm_basics.LLMSettings(model="m", api_key="k", provider="groq")
    assert llm_basics.agent_client_retries(gemini) == 0
    assert llm_basics.agent_client_retries(groq) == 3
