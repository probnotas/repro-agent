"""Configuration: the key comes from the environment, the slug is never hardcoded."""

from __future__ import annotations

import pytest

from repro.config import (
    DEFAULT_MODEL,
    ConfigError,
    Settings,
    get_api_key,
    get_model,
    get_settings,
)


def test_missing_key_raises_with_copy_instructions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr("repro.config.load_env", lambda: None)
    with pytest.raises(ConfigError) as excinfo:
        get_settings()
    message = str(excinfo.value)
    assert "cp .env.example .env" in message
    assert "openrouter.ai/keys" in message
    assert "repro demo" in message  # the no-key escape hatch is advertised


def test_placeholder_key_counts_as_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "<<<PUT_YOUR_OPENROUTER_KEY_IN_.env>>>")
    with pytest.raises(ConfigError):
        get_api_key()


def test_blank_key_counts_as_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "   ")
    with pytest.raises(ConfigError):
        get_api_key()


def test_model_defaults_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    assert get_model() == DEFAULT_MODEL


def test_model_comes_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_MODEL", "openai/gpt-5.4")
    assert get_model() == "openai/gpt-5.4"


def test_cli_override_beats_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_MODEL", "openai/gpt-5.4")
    assert get_model("google/gemini-3.5-flash") == "google/gemini-3.5-flash"


def test_blank_model_env_var_falls_back_to_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_MODEL", "  ")
    assert get_model() == DEFAULT_MODEL


def test_settings_redaction_hides_the_key() -> None:
    settings = Settings(api_key="sk-or-v1-supersecret", model="anthropic/claude-sonnet-5")
    rendered = settings.redacted()
    assert "supersecret" not in rendered
    assert "sk-or-v1" not in rendered
    assert settings.base_url == "https://openrouter.ai/api/v1"


def test_default_model_is_a_plausible_openrouter_slug() -> None:
    """Slugs are `vendor/model`; verified against https://openrouter.ai/models."""
    assert "/" in DEFAULT_MODEL
    vendor, _, name = DEFAULT_MODEL.partition("/")
    assert vendor and name
    assert " " not in DEFAULT_MODEL
