"""Shared fixtures. No test in this suite touches the network."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from repro.extract import Claim, parse_claim

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def never_read_a_real_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop any test from picking up a developer's real .env.

    get_settings() loads .env from the working directory. Without this, a test
    that means to exercise the "no API key" path would instead find a real key
    on a configured machine and start a live run against arXiv and OpenRouter --
    silently breaking this suite's promise that nothing here touches the network.
    Tests that want a key set it explicitly.
    """
    monkeypatch.setattr("repro.config.load_env", lambda: None)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture
def testable_claim() -> Claim:
    return parse_claim((FIXTURES / "demo_testable" / "claim.json").read_text())


@pytest.fixture
def untestable_claim() -> Claim:
    return parse_claim((FIXTURES / "demo_untestable" / "claim.json").read_text())


@pytest.fixture
def claim_payload() -> dict:
    """A minimal well-formed claim payload tests can mutate."""
    return {
        "claim_text": "MPC over a learned model reaches -150 return on Pendulum-v1.",
        "method": "random-shooting MPC",
        "environment": "Pendulum-v1",
        "metric": "mean_episode_return",
        "reported_value": -150.0,
        "conditions": ["2000 transitions"],
        "specified": {
            "hyperparameters": True,
            "seeds": True,
            "env_version": True,
            "success_threshold": True,
        },
        "missing": [],
    }


@pytest.fixture
def claim_json(claim_payload: dict) -> str:
    return json.dumps(claim_payload)
