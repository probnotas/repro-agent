"""Shared fixtures. No test in this suite touches the network."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from repro.extract import Claim, parse_claim

FIXTURES = Path(__file__).parent / "fixtures"


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
