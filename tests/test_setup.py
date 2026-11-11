"""Stage 0 checks: the environment is wired up correctly. No network calls."""

import pytest

import smoke_gemini


def test_package_imports():
    import ai_travel_assistant

    assert ai_travel_assistant.__version__


def test_core_dependencies_import():
    import langchain  # noqa: F401
    import langchain_google_genai  # noqa: F401
    import langgraph  # noqa: F401
    import pydantic_settings  # noqa: F401


def test_missing_settings_fail_clearly(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)

    with pytest.raises(smoke_gemini.ConfigError) as exc_info:
        smoke_gemini.load_settings()

    message = str(exc_info.value)
    assert "GEMINI_API_KEY" in message
    assert "GEMINI_MODEL" in message


def test_settings_load_without_exposing_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-test")
    monkeypatch.setenv("GEMINI_MODEL", "some-flash-model")

    settings = smoke_gemini.load_settings()

    assert settings.model == "some-flash-model"
    assert "fake-key-for-test" not in repr(settings)


def test_redact_hides_key():
    assert "secret123" not in smoke_gemini._redact("bad key secret123", "secret123")
