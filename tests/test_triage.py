"""The UNTESTABLE triage path -- a first-class outcome, not a failure path."""

from __future__ import annotations

from repro.extract import Claim
from repro.triage import MAX_MISSING_ITEMS, triage_claim

ALL_SPECIFIED = {
    "hyperparameters": True,
    "seeds": True,
    "env_version": True,
    "success_threshold": True,
}


def make_claim(**overrides) -> Claim:
    base = dict(
        claim_text="MPC reaches -150 return on Pendulum-v1.",
        method="random-shooting MPC",
        environment="Pendulum-v1",
        metric="mean_episode_return",
        reported_value=-150.0,
        conditions=[],
        specified=dict(ALL_SPECIFIED),
        missing=[],
    )
    base.update(overrides)
    return Claim(**base)  # type: ignore[arg-type]


def test_fully_specified_claim_is_testable() -> None:
    verdict = triage_claim(make_claim())
    assert verdict.testable
    assert verdict.reasons == []
    assert verdict.underspecified == []
    assert verdict.summary() == "no blocking gaps found"


def test_no_reported_value_is_untestable() -> None:
    verdict = triage_claim(make_claim(reported_value=None))
    assert not verdict.testable
    assert any("no single headline number" in reason for reason in verdict.reasons)


def test_no_environment_is_untestable() -> None:
    for value in ("", "unspecified", "unknown", "N/A", "none"):
        verdict = triage_claim(make_claim(environment=value))
        assert not verdict.testable, value
        assert any("no identifiable environment" in r for r in verdict.reasons)


def test_all_four_fields_unstated_is_untestable() -> None:
    verdict = triage_claim(make_claim(specified=dict.fromkeys(ALL_SPECIFIED, False)))
    assert not verdict.testable
    assert len(verdict.underspecified) == 4
    assert any("4 of 4 required details" in reason for reason in verdict.reasons)


def test_three_unstated_fields_is_still_testable_at_the_default_threshold() -> None:
    """Three gaps is the boundary: still testable, and every gap is listed."""
    specified = dict(ALL_SPECIFIED)
    specified.update(hyperparameters=False, seeds=False, env_version=False)
    verdict = triage_claim(make_claim(specified=specified))
    assert verdict.testable
    assert len(verdict.underspecified) == 3


def test_threshold_is_configurable() -> None:
    specified = dict(ALL_SPECIFIED)
    specified.update(hyperparameters=False, seeds=False)
    assert triage_claim(make_claim(specified=specified)).testable
    assert not triage_claim(make_claim(specified=specified), max_unspecified=1).testable


def test_long_missing_list_is_untestable() -> None:
    missing = [f"value {i}" for i in range(MAX_MISSING_ITEMS + 1)]
    verdict = triage_claim(make_claim(missing=missing))
    assert not verdict.testable
    assert any("would have to be invented" in reason for reason in verdict.reasons)


def test_missing_list_at_the_limit_is_still_testable() -> None:
    missing = [f"value {i}" for i in range(MAX_MISSING_ITEMS)]
    assert triage_claim(make_claim(missing=missing)).testable


def test_underspecified_items_are_human_readable() -> None:
    verdict = triage_claim(make_claim(specified=dict.fromkeys(ALL_SPECIFIED, False)))
    joined = " ".join(verdict.underspecified)
    assert "hyperparameters" in joined and "learning rate" in joined
    assert "random seeds" in joined
    assert "environment version" in joined
    assert "success threshold" in joined


def test_reasons_accumulate_rather_than_short_circuit() -> None:
    verdict = triage_claim(
        make_claim(
            reported_value=None,
            environment="unspecified",
            specified=dict.fromkeys(ALL_SPECIFIED, False),
        )
    )
    assert not verdict.testable
    assert len(verdict.reasons) >= 3
    assert "; " in verdict.summary()


def test_shipped_untestable_fixture_trips_triage(untestable_claim) -> None:
    verdict = triage_claim(untestable_claim)
    assert not verdict.testable
    assert len(verdict.reasons) >= 3


def test_shipped_testable_fixture_passes_triage(testable_claim) -> None:
    assert triage_claim(testable_claim).testable
