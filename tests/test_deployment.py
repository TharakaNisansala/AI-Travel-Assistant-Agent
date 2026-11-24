"""Deployment: the files Streamlit Community Cloud uses, checked offline.

- deploy/requirements.txt matches uv.lock and pins everything the app needs.
- The secrets template, the deployment guide and the code name the same
  settings, with the same defaults.
- Secrets reach the app the way Streamlit delivers them (os.environ).
- deploy/streamlit_app.py runs the real page, in public and local mode.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

from ai_travel_assistant import (  # noqa: E402
    llm_basics,
    provider_pool,
    public_sessions,
    usage_limits,
    web_app,
)

ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / "deploy" / "requirements.txt"
ENTRY = ROOT / "deploy" / "streamlit_app.py"
SECRETS_EXAMPLE = ROOT / ".streamlit" / "secrets.toml.example"
GUIDE = ROOT / "docs" / "DEPLOYMENT.md"
EXPORT = [
    "uv",
    "export",
    "--extra",
    "ui",
    "--no-dev",
    "--no-hashes",
    "--no-emit-project",
    "--format",
    "requirements-txt",
    "--quiet",
]


def load_toml(path: Path) -> dict:
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10: the toml package (via streamlit)
        import toml

        return toml.loads(path.read_text(encoding="utf-8"))
    return tomllib.loads(path.read_text(encoding="utf-8"))


def requirement_lines(text: str) -> list[str]:
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def pinned_names(text: str) -> set[str]:
    return {line.split("==")[0].lower() for line in requirement_lines(text)}


def normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


# --- deploy/requirements.txt ----------------------------------------------------


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv is not installed")
def test_the_requirements_match_uv_lock():
    exported = subprocess.run(
        EXPORT, cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    assert requirement_lines(REQUIREMENTS.read_text(encoding="utf-8")) == (
        requirement_lines(exported)
    ), "deploy/requirements.txt is out of date: regenerate it (docs/DEPLOYMENT.md)"


def test_every_runtime_and_ui_dependency_is_pinned():
    project = load_toml(ROOT / "pyproject.toml")["project"]
    wanted = [*project["dependencies"], *project["optional-dependencies"]["ui"]]
    names = {normalise(re.split(r"[<>=!~ \[;]", spec)[0]) for spec in wanted}
    pinned = {normalise(n) for n in pinned_names(REQUIREMENTS.read_text("utf-8"))}
    assert {"streamlit", "fpdf2", "langgraph", "langchain-google-genai"} <= names
    assert names <= pinned, names - pinned


def test_every_line_is_an_exact_pin_and_nothing_else_is_installed():
    lines = requirement_lines(REQUIREMENTS.read_text(encoding="utf-8"))
    assert len(lines) > 50
    for line in lines:
        assert "==" in line, line
        assert not line.startswith(("-e", ".", "/", "git+", "http://", "https://"))
    names = pinned_names(REQUIREMENTS.read_text(encoding="utf-8"))
    for dev_only in ("pytest", "ruff", "pypdf", "langgraph-cli"):
        assert dev_only not in names


def test_the_entry_point_sits_next_to_its_requirements():
    assert ENTRY.parent == REQUIREMENTS.parent
    # Community Cloud would prefer these over requirements.txt in this folder.
    for other in ("uv.lock", "Pipfile", "environment.yml", "pyproject.toml"):
        assert not (ENTRY.parent / other).exists()
    # The theme must stay at the repository root.
    assert (ROOT / ".streamlit" / "config.toml").exists()


# --- Secrets -------------------------------------------------------------------

CODE_SETTINGS = {
    web_app.PUBLIC_ENV,
    provider_pool.FALLBACK_ORDER_ENV,
    llm_basics.ACCOUNT_ID_VAR,
    public_sessions.THREAD_IDLE_ENV,
    *(name for pair in llm_basics.PROVIDERS.values() for name in pair),
    *(variable for variable, _ in usage_limits.LIMIT_VARIABLES.values()),
}


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
@pytest.mark.parametrize(
    ("path", "ignored"),
    [
        (".streamlit/secrets.toml", True),
        (".env", True),
        (".streamlit/secrets.toml.example", False),
        (".env.example", False),
    ],
)
def test_real_secrets_are_git_ignored_and_examples_are_not(path, ignored):
    result = subprocess.run(
        ["git", "check-ignore", "-q", path], cwd=ROOT, capture_output=True
    )
    assert (result.returncode == 0) is ignored


def test_the_secrets_example_lists_every_setting_with_safe_placeholders():
    secrets = load_toml(SECRETS_EXAMPLE)
    assert set(secrets) == CODE_SETTINGS
    assert all(isinstance(value, str) for value in secrets.values())  # top level
    assert secrets["TRAVEL_PUBLIC"] == "1"
    assert secrets["LLM_FALLBACK_ORDER"] == ",".join(
        provider_pool.DEFAULT_FALLBACK_ORDER
    )
    for name, value in secrets.items():
        if name not in ("TRAVEL_PUBLIC", "LLM_FALLBACK_ORDER"):
            assert value == "", name  # placeholders only, never a value
    assert "TRAVEL_DEV" not in secrets
    assert SECRETS_EXAMPLE.read_text(encoding="utf-8").startswith("# EXAMPLE ONLY")


def test_the_guide_names_every_setting_and_the_code_defaults():
    guide = GUIDE.read_text(encoding="utf-8")
    for name in CODE_SETTINGS:
        assert f"`{name}`" in guide, name
    for variable, default in usage_limits.LIMIT_VARIABLES.values():
        row = next(line for line in guide.splitlines() if f"`{variable}`" in line)
        assert f"default `{default}`" in row, variable
    idle = next(
        line
        for line in guide.splitlines()
        if f"`{public_sessions.THREAD_IDLE_ENV}`" in line
    )
    assert f"default `{public_sessions.DEFAULT_THREAD_IDLE_MINUTES}`" in idle
    assert "deploy/streamlit_app.py" in guide and "3.12" in guide
    assert "unlimited" not in guide.lower().replace("limited, not unlimited", "")


def test_streamlit_turns_top_level_string_secrets_into_environment_variables(
    tmp_path, monkeypatch
):
    """The mechanism the deployment relies on, with this Streamlit version."""
    from streamlit import config
    from streamlit.runtime.secrets import Secrets

    path = tmp_path / "secrets.toml"
    path.write_text(
        'TRAVEL_PUBLIC = "1"\n'
        'TRAVEL_MAX_DAILY_LLM_CALLS = "40"\n'
        'TRAVEL_MAX_MESSAGE_CHARS = ""\n'
        "TRAVEL_MAX_CONCURRENT_REQUESTS = true\n"  # a bool: NOT copied
        "[section]\n"
        'TRAVEL_MAX_MESSAGES_PER_SESSION = "3"\n',  # nested: NOT copied
        encoding="utf-8",
    )
    for variable, _ in usage_limits.LIMIT_VARIABLES.values():
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.delenv(web_app.PUBLIC_ENV, raising=False)
    old = config.get_option("secrets.files")
    config.set_option("secrets.files", [str(path)])
    secrets = Secrets()
    try:
        assert secrets.load_if_toml_exists()
        assert web_app.public_mode()
        limits = usage_limits.UsageLimits.from_env()
        assert limits.max_daily_llm_calls == 40
        assert limits.max_message_chars == 1000  # "" means the default
        assert limits.max_concurrent_requests == 2  # the bool was ignored
        assert limits.max_session_messages == 15  # the nested value was ignored
    finally:
        secrets._reset()  # removes what it put into os.environ
        config.set_option("secrets.files", old)
    assert "TRAVEL_MAX_DAILY_LLM_CALLS" not in os.environ


# --- deploy/streamlit_app.py runs the real page -------------------------------------


@pytest.fixture
def clean_state(monkeypatch):
    import streamlit as st

    for name in (*CODE_SETTINGS, web_app.DEV_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(llm_basics, "load_dotenv", lambda: None)  # not the real .env
    st.cache_resource.clear()
    usage_limits.reset_shared_guard()
    public_sessions.reset_shared_janitor()
    yield monkeypatch
    st.cache_resource.clear()
    usage_limits.reset_shared_guard()
    public_sessions.reset_shared_janitor()


def test_the_entry_point_runs_the_public_page_with_no_provider(clean_state):
    clean_state.setenv(web_app.PUBLIC_ENV, "1")
    app = AppTest.from_file(str(ENTRY), default_timeout=30)
    app.run()
    assert not app.exception
    from ai_travel_assistant.robustness import VISITOR_NOT_CONFIGURED

    assert [e.value for e in app.error] == [VISITOR_NOT_CONFIGURED]
    [offer] = [b for b in app.button if b.key == "offer-sample"]
    offer.click().run()
    assert not app.exception
    assert "pdf-sample" in [d.key for d in app.get("download_button")]
    assert len(app.chat_input) == 0


def test_the_entry_point_reads_the_fallback_order_from_settings(clean_state):
    clean_state.setenv(web_app.PUBLIC_ENV, "1")
    clean_state.setenv("LLM_FALLBACK_ORDER", "cloudflare,gemini")
    clean_state.setenv("GEMINI_API_KEY", "fake-gemini-key")
    clean_state.setenv("GEMINI_MODEL", "flash-model")
    clean_state.setenv("CLOUDFLARE_API_TOKEN", "fake-cf-token")
    clean_state.setenv("CLOUDFLARE_ACCOUNT_ID", "fake-account")
    app = AppTest.from_file(str(ENTRY), default_timeout=30)
    app.run()  # builds the real pool and graph; no API call until a message
    assert not app.exception
    header = " ".join(md.value for md in app.markdown)
    assert "Cloudflare · glm-4.7-flash · fallback: Gemini" in header
    for secret in ("fake-gemini-key", "fake-cf-token", "fake-account"):
        assert secret not in header
    assert len(app.chat_input) == 1 and app.text_input == []


def test_the_entry_point_keeps_local_mode_unchanged(clean_state, tmp_path):
    clean_state.setenv("GEMINI_API_KEY", "fake-gemini-key")
    clean_state.setenv("GEMINI_MODEL", "flash-model")
    clean_state.setenv(web_app.DB_ENV, str(tmp_path / "chat.sqlite"))
    app = AppTest.from_file(str(ENTRY), default_timeout=30)
    app.run()
    assert not app.exception
    assert [t.label for t in app.text_input] == ["Traveller"]
    assert any("Thread" in c.value for c in app.sidebar.caption)
    assert not {b.key for b in app.button} & {"sidebar-sample", "empty-sample"}
