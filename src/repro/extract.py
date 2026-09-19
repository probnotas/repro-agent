"""Turn paper text into one structured, testable claim.

The model is told to emit raw JSON. Because JSON-mode support varies by model on
OpenRouter, the parser here never assumes it worked: it strips markdown fences,
recovers the outermost JSON object, and retries once before giving up.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from .config import Settings
from .llm import complete

SPECIFIED_KEYS: tuple[str, ...] = (
    "hyperparameters",
    "seeds",
    "env_version",
    "success_threshold",
)

_FENCE = re.compile(r"^\s*```(?:json|JSON)?\s*|\s*```\s*$")

SYSTEM_PROMPT = """\
You extract ONE testable empirical claim from a model-based robot control paper.

You are not a summarizer. You are selecting the single headline empirical result
that someone could re-run and either confirm or refute on a laptop.

Return a JSON object with EXACTLY these keys:
{
  "claim_text": string,        // one sentence, the testable claim
  "method": string,            // the method name used to make the claim
  "environment": string,       // gymnasium env id if identifiable, else the env name
  "metric": string,            // e.g. "transitions to reach -200 return"
  "reported_value": number or null,
  "conditions": [string],      // conditions the claim is stated under
  "specified": {
    "hyperparameters": boolean,     // were the hyperparameters actually stated?
    "seeds": boolean,               // was the seed count/values actually stated?
    "env_version": boolean,         // was the exact env version actually stated?
    "success_threshold": boolean    // was the success threshold actually stated?
  },
  "missing": [string]          // what a re-implementer would have to GUESS
}

Rules:
- "specified" is about what the paper ACTUALLY STATES, not what is conventional.
  If a value is implied or standard but never written down, that is false.
- "missing" must list every value someone would have to invent to run the test.
  An empty list means the paper is fully self-contained. Be honest: an over-full
  "missing" list is much better than a fabricated reproduction.
- "reported_value" is the headline number as a plain number, or null if the
  paper states no single number.
- Prefer a claim testable on classic control (Pendulum-v1, CartPole-v1,
  MountainCarContinuous-v0, Acrobot-v1). If the paper is MuJoCo-only, still
  extract the real claim and say so in "conditions".

Return JSON only. No prose, no markdown fences, no trailing commentary.
"""

USER_TEMPLATE = """\
Paper title: {title}

Paper text (may be truncated):
---
{text}
---

Extract the single headline empirical claim as JSON.
"""

RETRY_SUFFIX = """\

Your previous response was not valid JSON. Return ONLY the JSON object this time:
no fences, no explanation, no text before or after the opening and closing braces.
"""


class ExtractionError(RuntimeError):
    """Raised when no usable claim JSON could be obtained."""


@dataclass
class Claim:
    """The structured claim, exactly as the pipeline downstream expects it."""

    claim_text: str
    method: str
    environment: str
    metric: str
    reported_value: float | None
    conditions: list[str] = field(default_factory=list)
    specified: dict[str, bool] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def unspecified(self) -> list[str]:
        """Schema fields the paper did not pin down."""
        return [key for key in SPECIFIED_KEYS if not self.specified.get(key, False)]


def strip_fences(text: str) -> str:
    """Remove ```json fences and any prose around the outermost JSON object."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        while lines and not lines[-1].strip():
            lines.pop()
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    else:
        cleaned = _FENCE.sub("", cleaned).strip()

    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        cleaned = cleaned[start : end + 1]
    return cleaned.strip()


def _coerce_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", value.replace(",", ""))
        if match:
            return float(match.group(0))
    return None


def _coerce_str_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def parse_claim(raw: str) -> Claim:
    """Parse a model response into a :class:`Claim`, tolerating fences and prose.

    Raises :class:`ExtractionError` if the text is not JSON or is missing the
    fields the rest of the pipeline depends on.
    """
    cleaned = strip_fences(raw)
    if not cleaned:
        raise ExtractionError("Model returned an empty response; no claim to parse.")
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        preview = cleaned[:200].replace("\n", " ")
        raise ExtractionError(
            f"Model response was not valid JSON ({exc.msg} at char {exc.pos}). "
            f"First 200 chars: {preview!r}"
        ) from exc

    if not isinstance(payload, dict):
        raise ExtractionError(
            f"Expected a JSON object, got {type(payload).__name__}."
        )

    claim_text = str(payload.get("claim_text", "")).strip()
    if not claim_text:
        raise ExtractionError(
            "Claim JSON has no 'claim_text'. Without a claim there is nothing to test."
        )

    raw_specified = payload.get("specified")
    specified = {
        key: bool(raw_specified.get(key, False)) if isinstance(raw_specified, dict) else False
        for key in SPECIFIED_KEYS
    }

    return Claim(
        claim_text=claim_text,
        method=str(payload.get("method", "") or "unspecified").strip(),
        environment=str(payload.get("environment", "") or "unspecified").strip(),
        metric=str(payload.get("metric", "") or "unspecified").strip(),
        reported_value=_coerce_float(payload.get("reported_value")),
        conditions=_coerce_str_list(payload.get("conditions")),
        specified=specified,
        missing=_coerce_str_list(payload.get("missing")),
    )


def extract_claim(
    title: str,
    text: str,
    *,
    settings: Settings | None = None,
) -> tuple[Claim, str]:
    """Ask the model for the claim; retry once on a parse failure.

    Returns ``(claim, raw_response)`` so the raw text can be written to
    ``runs/`` and every reported number stays traceable to a file.
    """
    user = USER_TEMPLATE.format(title=title, text=text)
    raw = complete(SYSTEM_PROMPT, user, json_mode=True, settings=settings)
    try:
        return parse_claim(raw), raw
    except ExtractionError as first_error:
        retry_raw = complete(
            SYSTEM_PROMPT,
            user + RETRY_SUFFIX,
            json_mode=True,
            settings=settings,
        )
        try:
            return parse_claim(retry_raw), retry_raw
        except ExtractionError as second_error:
            raise ExtractionError(
                f"Claim extraction failed twice.\n  first attempt: {first_error}\n"
                f"  retry: {second_error}"
            ) from second_error
