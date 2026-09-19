"""Aggregate run results and turn them into a verdict.

Every number here traces back to a file written under ``runs/``: nothing is
estimated, interpolated, or filled in when a run did not produce it.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum

RESULT_LINE = re.compile(
    r"^RESULT:\s*([A-Za-z0-9_.\-]+)\s*=\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$"
)

#: A RESULT line whose value is nan/inf. Matched separately so the failure says
#: "produced a non-finite value" rather than "never printed a RESULT line".
NON_FINITE_RESULT_LINE = re.compile(
    r"^RESULT:\s*([A-Za-z0-9_.\-]+)\s*=\s*([-+]?(?:nan|inf(?:inity)?))\s*$",
    re.IGNORECASE,
)


class Verdict(str, Enum):
    """The four possible outcomes of a replication attempt."""

    REPRODUCED = "REPRODUCED"
    NOT_REPRODUCED = "NOT_REPRODUCED"
    UNTESTABLE = "UNTESTABLE"
    RUN_FAILED = "RUN_FAILED"


@dataclass
class SeedResult:
    """One seed's outcome, good or bad."""

    seed: int
    value: float | None
    metric_name: str | None
    exit_code: int
    error: str | None = None
    artifact_dir: str | None = None

    @property
    def ok(self) -> bool:
        return self.value is not None and self.exit_code == 0


@dataclass
class Comparison:
    """The aggregate across seeds, compared against the paper's number."""

    verdict: Verdict
    reported_value: float | None
    our_mean: float | None
    our_std: float | None
    n_seeds: int
    delta_pct: float | None
    tolerance: float
    metric_name: str | None = None
    seed_results: list[SeedResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def failed_seeds(self) -> list[SeedResult]:
        return [result for result in self.seed_results if not result.ok]


def parse_result_line(stdout: str) -> tuple[str, float] | None:
    """Find the last ``RESULT: <name>=<float>`` line in ``stdout``.

    The last one wins so a script that prints progress lines in the same shape
    still reports its final answer.
    """
    found: tuple[str, float] | None = None
    for line in stdout.splitlines():
        match = RESULT_LINE.match(line.strip())
        if match:
            found = (match.group(1), float(match.group(2)))
    return found


def find_non_finite_result(stdout: str) -> str | None:
    """Return the metric name of a ``RESULT:`` line whose value is nan or inf.

    Such a line is not a result -- reporting it as a number would fabricate one --
    but it deserves a more useful error than "no RESULT line was printed".
    """
    for line in reversed(stdout.splitlines()):
        match = NON_FINITE_RESULT_LINE.match(line.strip())
        if match:
            return match.group(1)
    return None


def relative_delta(reported: float, observed: float) -> float:
    """Signed relative difference of ``observed`` from ``reported``.

    When the reported value is zero, fall back to an absolute difference so the
    comparison stays defined rather than dividing by zero.
    """
    if reported == 0.0:
        return observed
    return (observed - reported) / abs(reported)


def compare(
    reported_value: float | None,
    seed_results: list[SeedResult],
    tolerance: float,
    metric_name: str | None = None,
) -> Comparison:
    """Aggregate seeds and decide REPRODUCED / NOT_REPRODUCED / RUN_FAILED.

    UNTESTABLE is never produced here -- that verdict is the triage stage's, and
    reaching this function at all means the claim was judged testable.
    """
    if tolerance < 0:
        raise ValueError(f"tolerance must be non-negative, got {tolerance}")

    successful = [result for result in seed_results if result.ok]
    notes: list[str] = []

    if not successful:
        reasons = "; ".join(
            f"seed {result.seed}: {result.error or f'exit {result.exit_code}'}"
            for result in seed_results
        ) or "no seeds were run"
        return Comparison(
            verdict=Verdict.RUN_FAILED,
            reported_value=reported_value,
            our_mean=None,
            our_std=None,
            n_seeds=0,
            delta_pct=None,
            tolerance=tolerance,
            metric_name=metric_name,
            seed_results=seed_results,
            notes=[f"No seed produced a RESULT line ({reasons})."],
        )

    if len(successful) < len(seed_results):
        failed = ", ".join(str(result.seed) for result in seed_results if not result.ok)
        notes.append(
            f"{len(successful)}/{len(seed_results)} seeds produced a result; "
            f"seeds {failed} failed and are excluded from the mean."
        )

    values = [result.value for result in successful if result.value is not None]
    mean = sum(values) / len(values)
    if len(values) > 1:
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        std = math.sqrt(variance)
    else:
        std = 0.0
        notes.append("Only one seed produced a result, so the std is 0 by construction.")

    resolved_metric = metric_name or successful[0].metric_name

    if reported_value is None:
        return Comparison(
            verdict=Verdict.RUN_FAILED,
            reported_value=None,
            our_mean=mean,
            our_std=std,
            n_seeds=len(values),
            delta_pct=None,
            tolerance=tolerance,
            metric_name=resolved_metric,
            seed_results=seed_results,
            notes=notes
            + ["The run produced a number but the paper states none to compare it to."],
        )

    delta = relative_delta(reported_value, mean)
    within = abs(delta) <= tolerance + 1e-12  # inclusive at the boundary
    return Comparison(
        verdict=Verdict.REPRODUCED if within else Verdict.NOT_REPRODUCED,
        reported_value=reported_value,
        our_mean=mean,
        our_std=std,
        n_seeds=len(values),
        delta_pct=delta * 100.0,
        tolerance=tolerance,
        metric_name=resolved_metric,
        seed_results=seed_results,
        notes=notes,
    )


def untestable(
    reported_value: float | None,
    tolerance: float,
    reasons: list[str],
    metric_name: str | None = None,
) -> Comparison:
    """Build the UNTESTABLE comparison produced by the triage stage."""
    return Comparison(
        verdict=Verdict.UNTESTABLE,
        reported_value=reported_value,
        our_mean=None,
        our_std=None,
        n_seeds=0,
        delta_pct=None,
        tolerance=tolerance,
        metric_name=metric_name,
        seed_results=[],
        notes=reasons,
    )
