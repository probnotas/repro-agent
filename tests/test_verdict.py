"""Tolerance and verdict logic, especially at the boundaries."""

from __future__ import annotations

import pytest

from repro.compare import (
    Comparison,
    SeedResult,
    Verdict,
    compare,
    parse_result_line,
    relative_delta,
    untestable,
)


def ok_seed(seed: int, value: float) -> SeedResult:
    return SeedResult(seed=seed, value=value, metric_name="mean_episode_return", exit_code=0)


def bad_seed(seed: int, error: str = "boom") -> SeedResult:
    return SeedResult(seed=seed, value=None, metric_name=None, exit_code=1, error=error)


# --- RESULT line parsing ---------------------------------------------------

def test_parses_result_line() -> None:
    assert parse_result_line("RESULT: mean_episode_return=-150.5") == (
        "mean_episode_return",
        -150.5,
    )


def test_parses_result_line_among_noise() -> None:
    stdout = "loading...\nepoch 1\nRESULT: transitions=2000\ndone\n"
    assert parse_result_line(stdout) == ("transitions", 2000.0)


def test_last_result_line_wins() -> None:
    stdout = "RESULT: x=1\nRESULT: x=2\n"
    assert parse_result_line(stdout) == ("x", 2.0)


def test_parses_scientific_notation() -> None:
    assert parse_result_line("RESULT: one_step_error=1.5e-4") == ("one_step_error", 1.5e-4)


def test_no_result_line_returns_none() -> None:
    assert parse_result_line("nothing to see here\n") is None
    assert parse_result_line("RESULT mean=5") is None
    assert parse_result_line("RESULT: mean=not_a_number") is None


# --- boundary behaviour ----------------------------------------------------

def test_exactly_at_tolerance_is_reproduced() -> None:
    # reported 100, tolerance 25% -> 125 is exactly on the boundary.
    result = compare(100.0, [ok_seed(0, 125.0)], tolerance=0.25)
    assert result.verdict is Verdict.REPRODUCED
    assert result.delta_pct == pytest.approx(25.0)


def test_exactly_at_negative_tolerance_is_reproduced() -> None:
    result = compare(100.0, [ok_seed(0, 75.0)], tolerance=0.25)
    assert result.verdict is Verdict.REPRODUCED
    assert result.delta_pct == pytest.approx(-25.0)


def test_just_outside_tolerance_is_not_reproduced() -> None:
    result = compare(100.0, [ok_seed(0, 125.01)], tolerance=0.25)
    assert result.verdict is Verdict.NOT_REPRODUCED


def test_just_inside_tolerance_is_reproduced() -> None:
    assert compare(100.0, [ok_seed(0, 124.99)], tolerance=0.25).verdict is Verdict.REPRODUCED


def test_zero_tolerance_requires_an_exact_match() -> None:
    assert compare(100.0, [ok_seed(0, 100.0)], tolerance=0.0).verdict is Verdict.REPRODUCED
    assert compare(100.0, [ok_seed(0, 100.1)], tolerance=0.0).verdict is Verdict.NOT_REPRODUCED


def test_negative_reported_value_uses_magnitude() -> None:
    # -200 with 25% tolerance spans [-250, -150]; -250 is on the boundary.
    assert compare(-200.0, [ok_seed(0, -250.0)], tolerance=0.25).verdict is Verdict.REPRODUCED
    assert compare(-200.0, [ok_seed(0, -251.0)], tolerance=0.25).verdict is Verdict.NOT_REPRODUCED
    assert compare(-200.0, [ok_seed(0, -149.0)], tolerance=0.25).verdict is Verdict.NOT_REPRODUCED


def test_zero_reported_value_falls_back_to_absolute_delta() -> None:
    assert relative_delta(0.0, 0.1) == 0.1
    assert compare(0.0, [ok_seed(0, 0.1)], tolerance=0.25).verdict is Verdict.REPRODUCED
    assert compare(0.0, [ok_seed(0, 0.9)], tolerance=0.25).verdict is Verdict.NOT_REPRODUCED


def test_negative_tolerance_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        compare(100.0, [ok_seed(0, 100.0)], tolerance=-0.1)


# --- aggregation across seeds ----------------------------------------------

def test_aggregates_mean_and_sample_std() -> None:
    result = compare(100.0, [ok_seed(0, 90.0), ok_seed(1, 100.0), ok_seed(2, 110.0)], 0.25)
    assert result.our_mean == pytest.approx(100.0)
    assert result.our_std == pytest.approx(10.0)  # sample std, ddof=1
    assert result.n_seeds == 3
    assert result.delta_pct == pytest.approx(0.0)


def test_single_seed_std_is_zero_and_says_so() -> None:
    result = compare(100.0, [ok_seed(0, 100.0)], 0.25)
    assert result.our_std == 0.0
    assert any("one seed" in note for note in result.notes)


def test_partial_seed_failure_still_aggregates_and_warns() -> None:
    result = compare(100.0, [ok_seed(0, 100.0), bad_seed(1), ok_seed(2, 120.0)], 0.25)
    assert result.verdict is Verdict.REPRODUCED
    assert result.n_seeds == 2
    assert result.our_mean == pytest.approx(110.0)
    assert any("2/3 seeds" in note for note in result.notes)
    assert [r.seed for r in result.failed_seeds] == [1]


# --- failure paths ---------------------------------------------------------

def test_all_seeds_failing_is_run_failed() -> None:
    result = compare(100.0, [bad_seed(0, "timed out after 600s"), bad_seed(1)], 0.25)
    assert result.verdict is Verdict.RUN_FAILED
    assert result.our_mean is None
    assert result.n_seeds == 0
    assert "timed out after 600s" in result.notes[0]


def test_no_seeds_at_all_is_run_failed() -> None:
    result = compare(100.0, [], 0.25)
    assert result.verdict is Verdict.RUN_FAILED
    assert "no seeds were run" in result.notes[0]


def test_exit_zero_without_result_line_is_not_ok() -> None:
    silent = SeedResult(seed=0, value=None, metric_name=None, exit_code=0, error="no RESULT")
    assert not silent.ok
    assert compare(100.0, [silent], 0.25).verdict is Verdict.RUN_FAILED


def test_number_but_no_reported_value_is_run_failed_not_reproduced() -> None:
    """A run that produced a number the paper never stated cannot be a pass."""
    result = compare(None, [ok_seed(0, 42.0)], 0.25)
    assert result.verdict is Verdict.RUN_FAILED
    assert result.our_mean == 42.0
    assert result.delta_pct is None
    assert any("states none" in note for note in result.notes)


def test_untestable_builder_carries_reasons() -> None:
    result: Comparison = untestable(None, 0.25, ["no environment given"], metric_name="score")
    assert result.verdict is Verdict.UNTESTABLE
    assert result.our_mean is None
    assert result.notes == ["no environment given"]
    assert result.metric_name == "score"


# --- non-finite results are not results ------------------------------------

def test_nan_result_line_is_not_parsed_as_a_number() -> None:
    from repro.compare import find_non_finite_result

    assert parse_result_line("RESULT: speedup=nan") is None
    assert find_non_finite_result("RESULT: speedup=nan") == "speedup"
    assert find_non_finite_result("RESULT: speedup=-inf") == "speedup"
    assert find_non_finite_result("RESULT: speedup=Infinity") == "speedup"
    assert find_non_finite_result("RESULT: speedup=1.5") is None
    assert find_non_finite_result("no result here") is None
