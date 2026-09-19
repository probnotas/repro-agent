"""Decide whether a claim is specified well enough to test fairly.

UNTESTABLE is a first-class outcome. When a paper does not say enough to run a
fair test, saying so precisely is more valuable than guessing and reporting a
number that measures the agent's guesses rather than the paper.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .extract import SPECIFIED_KEYS, Claim

#: A claim is UNTESTABLE once this many of the four schema fields are unstated.
MAX_UNSPECIFIED = 3

#: ...or once this many free-form items appear in the claim's "missing" list.
MAX_MISSING_ITEMS = 6

_LABELS: dict[str, str] = {
    "hyperparameters": "hyperparameters (network size, learning rate, training epochs)",
    "seeds": "random seeds (count and/or values)",
    "env_version": "exact environment version",
    "success_threshold": "success threshold defining the metric",
}


@dataclass
class Triage:
    """The verdict of the triage stage plus the reasons behind it."""

    testable: bool
    reasons: list[str] = field(default_factory=list)
    underspecified: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return "; ".join(self.reasons) if self.reasons else "no blocking gaps found"


def triage_claim(claim: Claim, *, max_unspecified: int = MAX_UNSPECIFIED) -> Triage:
    """Judge whether ``claim`` can be tested without inventing the experiment.

    Blocking conditions, each reported explicitly:
      * no ``reported_value`` -- nothing to compare a run against;
      * no identifiable environment -- nothing to run;
      * too many of the four schema fields unstated;
      * an unusually long ``missing`` list.
    """
    reasons: list[str] = []
    underspecified: list[str] = [
        _LABELS.get(key, key) for key in SPECIFIED_KEYS if not claim.specified.get(key, False)
    ]

    if claim.reported_value is None:
        reasons.append(
            "the paper states no single headline number, so there is nothing to "
            "compare a run against"
        )

    environment = (claim.environment or "").strip().lower()
    if not environment or environment in {"unspecified", "unknown", "n/a", "none"}:
        reasons.append("no identifiable environment, so there is nothing to run")

    if len(underspecified) > max_unspecified:
        reasons.append(
            f"{len(underspecified)} of {len(SPECIFIED_KEYS)} required details are "
            "never stated in the paper"
        )

    if len(claim.missing) > MAX_MISSING_ITEMS:
        reasons.append(
            f"{len(claim.missing)} distinct values would have to be invented to run "
            "the experiment"
        )

    return Triage(testable=not reasons, reasons=reasons, underspecified=underspecified)
