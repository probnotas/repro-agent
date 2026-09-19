"""OpenRouter error branches: 401 / 402 / 429, plus retry and JSON-mode fallback.

The HTTP layer is mocked throughout. Nothing here touches the network.
"""

from __future__ import annotations

import json
import urllib.error
from types import SimpleNamespace

import pytest

from repro import llm
from repro.config import Settings
from repro.llm import (
    AuthError,
    CreditError,
    ModelCallError,
    RateLimitError,
    complete,
    list_models,
)

SETTINGS = Settings(api_key="sk-or-v1-test", model="anthropic/claude-sonnet-5")


class FakeStatusError(Exception):
    """Stands in for openai.APIStatusError, which is what `complete` catches."""

    def __init__(self, status_code: int, message: str = "boom") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response = SimpleNamespace(status_code=status_code)


class FakeConnectionError(Exception):
    """Stands in for openai.APIConnectionError."""


@pytest.fixture(autouse=True)
def patch_openai_exceptions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the module's except-clauses at our fakes so no real SDK is needed."""
    monkeypatch.setattr(llm, "APIStatusError", FakeStatusError)
    monkeypatch.setattr(llm, "APIConnectionError", FakeConnectionError)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Backoff is real code; waiting for it in tests is not."""
    monkeypatch.setattr(llm.time, "sleep", lambda _seconds: None)


def install_client(monkeypatch: pytest.MonkeyPatch, side_effect) -> list[dict]:
    """Replace the OpenAI client with one whose create() runs ``side_effect``.

    Returns the list of kwargs each call was made with, so tests can assert on
    the request shape (e.g. that response_format was dropped).
    """
    calls: list[dict] = []

    def create(**kwargs):
        calls.append(kwargs)
        return side_effect(len(calls), kwargs)

    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(llm, "_build_client", lambda settings: llm._Client(fake, settings.model))
    return calls


def message(content: str, finish_reason: str = "stop"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content), finish_reason=finish_reason
            )
        ]
    )


# --- the three error branches the CLI must distinguish ---------------------

def test_401_raises_auth_error_naming_the_env_var(monkeypatch) -> None:
    install_client(monkeypatch, lambda n, kw: (_ for _ in ()).throw(FakeStatusError(401)))
    with pytest.raises(AuthError) as excinfo:
        complete("sys", "user", settings=SETTINGS)
    message_text = str(excinfo.value)
    assert "401" in message_text
    assert "OPENROUTER_API_KEY" in message_text
    assert "openrouter.ai/keys" in message_text


def test_402_raises_credit_error_pointing_at_credits(monkeypatch) -> None:
    install_client(monkeypatch, lambda n, kw: (_ for _ in ()).throw(FakeStatusError(402)))
    with pytest.raises(CreditError) as excinfo:
        complete("sys", "user", settings=SETTINGS)
    assert "402" in str(excinfo.value)
    assert "credits" in str(excinfo.value).lower()


def test_429_retries_then_raises_rate_limit_error(monkeypatch) -> None:
    calls = install_client(
        monkeypatch, lambda n, kw: (_ for _ in ()).throw(FakeStatusError(429))
    )
    with pytest.raises(RateLimitError) as excinfo:
        complete("sys", "user", settings=SETTINGS)
    assert len(calls) == llm.MAX_RETRIES  # it really did retry
    assert "429" in str(excinfo.value)


def test_429_that_recovers_returns_the_content(monkeypatch) -> None:
    def side_effect(n: int, kwargs: dict):
        if n < 3:
            raise FakeStatusError(429)
        return message('{"ok": true}')

    calls = install_client(monkeypatch, side_effect)
    assert complete("sys", "user", settings=SETTINGS) == '{"ok": true}'
    assert len(calls) == 3


def test_429_backoff_is_exponential(monkeypatch) -> None:
    slept: list[float] = []
    monkeypatch.setattr(llm.time, "sleep", slept.append)
    install_client(monkeypatch, lambda n, kw: (_ for _ in ()).throw(FakeStatusError(429)))
    with pytest.raises(RateLimitError):
        complete("sys", "user", settings=SETTINGS)
    assert slept == [2.0, 4.0, 8.0]


def test_other_status_raises_model_call_error(monkeypatch) -> None:
    install_client(monkeypatch, lambda n, kw: (_ for _ in ()).throw(FakeStatusError(500)))
    with pytest.raises(ModelCallError, match="HTTP 500"):
        complete("sys", "user", settings=SETTINGS)


def test_connection_error_retries_then_raises(monkeypatch) -> None:
    calls = install_client(
        monkeypatch, lambda n, kw: (_ for _ in ()).throw(FakeConnectionError("no route"))
    )
    with pytest.raises(ModelCallError, match="Could not reach OpenRouter"):
        complete("sys", "user", settings=SETTINGS)
    assert len(calls) == llm.MAX_RETRIES


# --- JSON mode is a hint, not a guarantee ---------------------------------

def test_json_mode_sets_response_format(monkeypatch) -> None:
    calls = install_client(monkeypatch, lambda n, kw: message("{}"))
    complete("sys", "user", json_mode=True, settings=SETTINGS)
    assert calls[0]["response_format"] == {"type": "json_object"}


def test_json_mode_off_omits_response_format(monkeypatch) -> None:
    calls = install_client(monkeypatch, lambda n, kw: message("print('hi')"))
    complete("sys", "user", json_mode=False, settings=SETTINGS)
    assert "response_format" not in calls[0]


def test_model_without_json_mode_falls_back(monkeypatch) -> None:
    """A 400 naming response_format retries once without it, rather than dying."""

    def side_effect(n: int, kwargs: dict):
        if n == 1:
            raise FakeStatusError(400, "response_format is not supported by this model")
        return message('{"ok": true}')

    calls = install_client(monkeypatch, side_effect)
    assert complete("sys", "user", json_mode=True, settings=SETTINGS) == '{"ok": true}'
    assert "response_format" in calls[0]
    assert "response_format" not in calls[1]


def test_unrelated_400_is_not_retried_as_a_json_mode_problem(monkeypatch) -> None:
    calls = install_client(
        monkeypatch,
        lambda n, kw: (_ for _ in ()).throw(FakeStatusError(400, "context length exceeded")),
    )
    with pytest.raises(ModelCallError, match="HTTP 400"):
        complete("sys", "user", settings=SETTINGS)
    assert len(calls) == 1


def test_the_model_slug_comes_from_settings(monkeypatch) -> None:
    calls = install_client(monkeypatch, lambda n, kw: message("{}"))
    complete("sys", "user", settings=Settings(api_key="k", model="openai/gpt-5.4"))
    assert calls[0]["model"] == "openai/gpt-5.4"


# --- empty / malformed responses ------------------------------------------

def test_empty_content_raises_with_the_finish_reason(monkeypatch) -> None:
    """An empty reply names why it was empty. (finish_reason="length" is its own
    recoverable case -- see the budget-escalation tests below.)"""
    install_client(monkeypatch, lambda n, kw: message("", finish_reason="stop"))
    with pytest.raises(ModelCallError, match="finish_reason=stop"):
        complete("sys", "user", settings=SETTINGS)


def test_no_choices_raises(monkeypatch) -> None:
    install_client(monkeypatch, lambda n, kw: SimpleNamespace(choices=[]))
    with pytest.raises(ModelCallError, match="no choices"):
        complete("sys", "user", settings=SETTINGS)


# --- /models -------------------------------------------------------------

class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._body = json.dumps(payload).encode()

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None


def http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://openrouter.ai/api/v1/models", code, "nope", {}, None)


def test_list_models_returns_the_catalogue(monkeypatch) -> None:
    payload = {"data": [{"id": "anthropic/claude-sonnet-5", "pricing": {"prompt": "0.000002"}}]}
    monkeypatch.setattr(llm.urllib.request, "urlopen", lambda *a, **k: FakeResponse(payload))
    catalogue = list_models(SETTINGS)
    assert catalogue[0]["id"] == "anthropic/claude-sonnet-5"


def test_list_models_sends_auth_and_attribution_headers(monkeypatch) -> None:
    captured: dict = {}

    def urlopen(request, *args, **kwargs):
        captured.update(request.headers)
        return FakeResponse({"data": []})

    monkeypatch.setattr(llm.urllib.request, "urlopen", urlopen)
    list_models(SETTINGS)
    # urllib title-cases header names.
    assert captured["Authorization"] == "Bearer sk-or-v1-test"
    assert captured["Http-referer"] == "https://github.com/probnotas/repro-agent"
    assert captured["X-title"] == "repro-agent"


@pytest.mark.parametrize(
    "code, expected",
    [(401, AuthError), (402, CreditError), (429, RateLimitError), (500, ModelCallError)],
)
def test_list_models_maps_status_codes(monkeypatch, code: int, expected: type) -> None:
    def urlopen(*args, **kwargs):
        raise http_error(code)

    monkeypatch.setattr(llm.urllib.request, "urlopen", urlopen)
    with pytest.raises(expected):
        list_models(SETTINGS)


def test_list_models_unreachable_raises_model_call_error(monkeypatch) -> None:
    def urlopen(*args, **kwargs):
        raise urllib.error.URLError("name resolution failed")

    monkeypatch.setattr(llm.urllib.request, "urlopen", urlopen)
    with pytest.raises(ModelCallError, match="Could not reach"):
        list_models(SETTINGS)


def test_list_models_rejects_an_unexpected_shape(monkeypatch) -> None:
    monkeypatch.setattr(
        llm.urllib.request, "urlopen", lambda *a, **k: FakeResponse({"error": "nope"})
    )
    with pytest.raises(ModelCallError, match="unexpected payload"):
        list_models(SETTINGS)


def test_settings_never_expose_the_key() -> None:
    assert "sk-or-v1-test" not in SETTINGS.redacted()
    assert "api_key=<set>" in SETTINGS.redacted()


# --- truncated replies escalate the output budget -------------------------

def test_truncated_reply_retries_with_a_doubled_budget(monkeypatch) -> None:
    """A reply cut off before any text is retried with more room, not failed."""

    def side_effect(n: int, kwargs: dict):
        if n == 1:
            return message("", finish_reason="length")
        return message("print('RESULT: score=1')")

    calls = install_client(monkeypatch, side_effect)
    assert complete("sys", "user", settings=SETTINGS) == "print('RESULT: score=1')"
    assert calls[0]["max_tokens"] == llm.DEFAULT_MAX_TOKENS
    assert calls[1]["max_tokens"] == llm.DEFAULT_MAX_TOKENS * 2


def test_budget_escalation_stops_at_the_ceiling(monkeypatch) -> None:
    calls = install_client(monkeypatch, lambda n, kw: message("", finish_reason="length"))
    with pytest.raises(llm.TruncatedError) as excinfo:
        complete("sys", "user", settings=SETTINGS, max_tokens=llm.MAX_TOKEN_CEILING)
    assert len(calls) == 1  # already at the ceiling, so no pointless retry
    text = str(excinfo.value)
    assert "output-token limit" in text
    assert "OPENROUTER_MODEL" in text  # tells the user which knob to turn


def test_budget_never_exceeds_the_ceiling(monkeypatch) -> None:
    calls = install_client(monkeypatch, lambda n, kw: message("", finish_reason="length"))
    with pytest.raises(llm.TruncatedError):
        complete("sys", "user", settings=SETTINGS, max_tokens=llm.MAX_TOKEN_CEILING // 2)
    assert all(call["max_tokens"] <= llm.MAX_TOKEN_CEILING for call in calls)


def test_truncated_error_is_an_llm_error_so_the_cli_catches_it() -> None:
    from repro.llm import LLMError, TruncatedError

    assert issubclass(TruncatedError, LLMError)


def test_empty_reply_for_another_reason_is_still_a_hard_error(monkeypatch) -> None:
    calls = install_client(monkeypatch, lambda n, kw: message("", finish_reason="content_filter"))
    with pytest.raises(ModelCallError, match="content_filter"):
        complete("sys", "user", settings=SETTINGS)
    assert len(calls) == 1  # not retried -- more room would not help


def test_generate_asks_for_a_bigger_budget_than_the_default(monkeypatch) -> None:
    """Script generation needs more room than a claim JSON."""
    captured: dict = {}

    def fake_complete(system, user, json_mode=True, **kwargs):
        captured.update(kwargs)
        return "print('RESULT: score=1')"

    monkeypatch.setattr("repro.generate.complete", fake_complete)
    from repro.extract import Claim
    from repro.generate import generate_script

    generate_script("T", Claim(
        claim_text="c", method="m", environment="Pendulum-v1", metric="x",
        reported_value=1.0, conditions=[], specified={}, missing=[],
    ))
    assert captured["max_tokens"] > llm.DEFAULT_MAX_TOKENS
