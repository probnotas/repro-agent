"""The only place in repro-agent that talks to a language model.

All calls go through OpenRouter's OpenAI-compatible endpoint using the ``openai``
package pointed at ``https://openrouter.ai/api/v1``. No vendor SDK is used
directly, and the model slug always comes from configuration, never a call site.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from openai import APIConnectionError, APIStatusError, OpenAI

from .config import APP_TITLE, REFERER, Settings, get_settings

MAX_RETRIES = 4
BASE_BACKOFF_SECONDS = 2.0

#: Default output budget. Generous enough for a claim JSON; script generation
#: asks for more (see repro.generate).
DEFAULT_MAX_TOKENS = 8192

#: A reply truncated before any usable text is retried with the budget doubled,
#: up to this ceiling. Reasoning models can spend a lot before the first token
#: of the answer, and that cost is invisible from here.
MAX_TOKEN_CEILING = 32768


class LLMError(RuntimeError):
    """Base class for every failure raised out of this module."""


class AuthError(LLMError):
    """HTTP 401 -- the key is missing, malformed, or revoked."""


class CreditError(LLMError):
    """HTTP 402 -- the OpenRouter account is out of credits."""


class RateLimitError(LLMError):
    """HTTP 429 that survived every retry."""


class ModelCallError(LLMError):
    """Any other non-recoverable failure from the provider."""


class TruncatedError(LLMError):
    """The model hit its output-token budget before emitting usable text.

    Raised only after the automatic budget escalation below has been exhausted,
    so by the time a caller sees this the request has genuinely outgrown what
    this model will produce in one turn.
    """


_AUTH_MESSAGE = (
    "OpenRouter rejected the key (HTTP 401).\n"
    "Check OPENROUTER_API_KEY in your .env -- keys look like `sk-or-v1-...` and "
    "can be re-issued at https://openrouter.ai/keys"
)
_CREDIT_MESSAGE = (
    "OpenRouter says this account is out of credits (HTTP 402).\n"
    "Top up at https://openrouter.ai/credits, or switch OPENROUTER_MODEL to a "
    "cheaper slug (`repro models` prints prices)."
)
_RATE_MESSAGE = (
    "OpenRouter rate-limited the request (HTTP 429) and it did not recover after "
    "{n} retries.\nWait a moment, or switch OPENROUTER_MODEL to a less contended "
    "slug (`repro models`)."
)


_TRUNCATED_MESSAGE = (
    "The model hit its output-token limit ({budget:,}) before producing any usable "
    "text (finish_reason=length).\n"
    "repro already retried with a larger budget. Some models -- particularly "
    "reasoning models -- spend the whole budget before emitting an answer.\n"
    "Try a different OPENROUTER_MODEL (`repro models`), or use --model on this run."
)


@dataclass(frozen=True)
class _Client:
    client: OpenAI
    model: str


def _build_client(settings: Settings) -> _Client:
    client = OpenAI(
        base_url=settings.base_url,
        api_key=settings.api_key,
        default_headers={"HTTP-Referer": REFERER, "X-Title": APP_TITLE},
        max_retries=0,  # retries live here so 429s get explicit, visible backoff
    )
    return _Client(client=client, model=settings.model)


def _status_of(exc: Exception) -> int:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else 0


def _raise_for_status(exc: Exception, status: int, retries: int) -> None:
    if status == 401:
        raise AuthError(_AUTH_MESSAGE) from exc
    if status == 402:
        raise CreditError(_CREDIT_MESSAGE) from exc
    if status == 429:
        raise RateLimitError(_RATE_MESSAGE.format(n=retries)) from exc
    raise ModelCallError(f"OpenRouter call failed (HTTP {status}): {exc}") from exc


def complete(
    system: str,
    user: str,
    json_mode: bool = True,
    *,
    settings: Settings | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.0,
) -> str:
    """Run one chat completion and return the assistant's text.

    ``json_mode`` is a *hint*: OpenRouter's support for ``response_format`` varies
    by model, so it is requested and dropped if the provider refuses it. Callers
    must still parse defensively (see :mod:`repro.extract`).

    ``max_tokens`` is a floor, not a cap: a reply truncated before any usable text
    is retried with the budget doubled, up to :data:`MAX_TOKEN_CEILING`.
    """
    resolved = settings or get_settings()
    handle = _build_client(resolved)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    kwargs: dict[str, Any] = {
        "model": handle.model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            response = handle.client.chat.completions.create(**kwargs)
            return _extract_text(response)
        except APIStatusError as exc:
            status = _status_of(exc)
            last_exc = exc
            if status == 429 and attempt < MAX_RETRIES - 1:
                time.sleep(BASE_BACKOFF_SECONDS * (2**attempt))
                continue
            if status == 400 and json_mode and _looks_like_json_mode_refusal(exc):
                # This model exposes no JSON mode on OpenRouter. The prompt already
                # demands raw JSON, so drop the hint and try again without it.
                kwargs.pop("response_format", None)
                json_mode = False
                continue
            _raise_for_status(exc, status, MAX_RETRIES)
        except TruncatedError as exc:
            last_exc = exc
            budget = int(kwargs["max_tokens"])
            if budget < MAX_TOKEN_CEILING and attempt < MAX_RETRIES - 1:
                kwargs["max_tokens"] = min(budget * 2, MAX_TOKEN_CEILING)
                continue
            raise TruncatedError(_TRUNCATED_MESSAGE.format(budget=budget)) from exc
        except APIConnectionError as exc:
            last_exc = exc
            if attempt < MAX_RETRIES - 1:
                time.sleep(BASE_BACKOFF_SECONDS * (2**attempt))
                continue
            raise ModelCallError(
                f"Could not reach OpenRouter after {MAX_RETRIES} attempts: {exc}"
            ) from exc

    raise ModelCallError(f"OpenRouter call failed: {last_exc}")


def _looks_like_json_mode_refusal(exc: Exception) -> bool:
    text = str(exc).lower()
    return "response_format" in text or "json" in text


def _extract_text(response: Any) -> str:
    """Pull the message text out of a completion, failing loudly if it is empty.

    An empty reply with ``finish_reason="length"`` is reported as
    :class:`TruncatedError` so the caller can retry with a bigger budget rather
    than treating a truncation as an unrecoverable provider error.
    """
    choices = getattr(response, "choices", None) or []
    if not choices:
        raise ModelCallError("OpenRouter returned no choices -- nothing to parse.")
    content = getattr(choices[0].message, "content", None)
    if not content or not content.strip():
        finish = getattr(choices[0], "finish_reason", "unknown")
        if finish == "length":
            raise TruncatedError(finish)
        raise ModelCallError(
            f"OpenRouter returned an empty message (finish_reason={finish})."
        )
    return content


def list_models(settings: Settings | None = None) -> list[dict[str, Any]]:
    """Fetch the ``/models`` catalogue so users can pick a slug and see pricing."""
    resolved = settings or get_settings()
    url = f"{resolved.base_url}/models"
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {resolved.api_key}",
            "HTTP-Referer": REFERER,
            "X-Title": APP_TITLE,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            raise AuthError(_AUTH_MESSAGE) from exc
        if exc.code == 402:
            raise CreditError(_CREDIT_MESSAGE) from exc
        if exc.code == 429:
            raise RateLimitError(_RATE_MESSAGE.format(n=0)) from exc
        raise ModelCallError(f"/models returned HTTP {exc.code}: {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise ModelCallError(f"Could not reach {url}: {exc.reason}") from exc

    data = payload.get("data")
    if not isinstance(data, list):
        raise ModelCallError("/models returned an unexpected payload shape.")
    return data
