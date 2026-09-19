"""Environment-driven configuration.

Every knob comes from the environment (loaded from ``.env`` via python-dotenv).
Nothing here is ever hardcoded at a call site, and no secret is ever printed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

OPENROUTER_BASE_URL: Final[str] = "https://openrouter.ai/api/v1"

#: Sent to OpenRouter so the request is attributed to this project.
REFERER: Final[str] = "https://github.com/probnotas/repro-agent"
APP_TITLE: Final[str] = "repro-agent"

#: Used when OPENROUTER_MODEL is unset. Verified against openrouter.ai/models.
DEFAULT_MODEL: Final[str] = "anthropic/claude-sonnet-5"

DEFAULT_TOLERANCE: Final[float] = 0.25
DEFAULT_SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
DEFAULT_TIMEOUT: Final[int] = 600

PROJECT_ROOT: Final[Path] = Path.cwd()
CACHE_DIR: Final[Path] = PROJECT_ROOT / ".cache"
RUNS_DIR: Final[Path] = PROJECT_ROOT / "runs"

_MISSING_KEY_MESSAGE: Final[str] = (
    "OPENROUTER_API_KEY is not set.\n"
    "\n"
    "  1. cp .env.example .env\n"
    "  2. put your key in .env (get one at https://openrouter.ai/keys)\n"
    "  3. optionally set OPENROUTER_MODEL to a slug from https://openrouter.ai/models\n"
    "\n"
    "No key handy? `repro demo` runs end-to-end from the shipped fixtures "
    "with no key and no network."
)


class ConfigError(RuntimeError):
    """Raised when the environment is not usable for live model calls."""


@dataclass(frozen=True)
class Settings:
    """Resolved runtime settings. ``api_key`` is never logged or displayed."""

    api_key: str
    model: str
    base_url: str = OPENROUTER_BASE_URL

    def redacted(self) -> str:
        """A safe one-line description with no key material in it."""
        return f"model={self.model} base_url={self.base_url} api_key=<set>"


def load_env() -> None:
    """Load ``.env`` from the current directory without clobbering real env vars."""
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)
    load_dotenv(override=False)  # fall back to a .env found by walking upward


def get_model(override: str | None = None) -> str:
    """Resolve the model slug: CLI override > OPENROUTER_MODEL > DEFAULT_MODEL."""
    if override:
        return override
    return os.environ.get("OPENROUTER_MODEL", "").strip() or DEFAULT_MODEL


def get_api_key() -> str:
    """Return the OpenRouter key, or raise :class:`ConfigError` with instructions."""
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key or key.startswith("<<<"):
        raise ConfigError(_MISSING_KEY_MESSAGE)
    return key


def get_settings(model_override: str | None = None) -> Settings:
    """Load ``.env`` and build :class:`Settings`, raising if the key is absent."""
    load_env()
    return Settings(api_key=get_api_key(), model=get_model(model_override))
