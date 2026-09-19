"""Report rendering, with the mandatory ASSUMPTIONS section under scrutiny."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from rich.console import Console

from repro.compare import SeedResult, Verdict, compare, untestable
from repro.extract import Claim
from repro.report import (
    ReportCard,
    assumptions_section,
    render_markdown,
    render_terminal,
    write_report,
)


def make_claim(**overrides) -> Claim:
    base = dict(
        claim_text="MPC reaches -150 return on Pendulum-v1.",
        method="random-shooting MPC",
        environment="Pendulum-v1",
        metric="mean_episode_return",
        reported_value=-150.0,
        conditions=["2000 transitions"],
        specified={
            "hyperparameters": True,
            "seeds": True,
            "env_version": True,
            "success_threshold": True,
        },
        missing=[],
    )
    base.update(overrides)
    return Claim(**base)  # type: ignore[arg-type]


def make_card(**overrides) -> ReportCard:
    comparison = compare(
        -150.0,
        [SeedResult(seed=s, value=v, metric_name="mean_episode_return", exit_code=0)
         for s, v in [(0, -145.0), (1, -155.0)]],
        0.25,
    )
    base = dict(
        arxiv_id="demo-0001",
        title="A Synthetic Paper",
        claim=make_claim(),
        comparison=comparison,
        model_slug="anthropic/claude-sonnet-5",
        assumptions=["horizon 15 (paper does not say)"],
        run_path="/tmp/runs/demo-0001/x",
        tolerance=0.25,
        seeds=[0, 1],
    )
    base.update(overrides)
    return ReportCard(**base)  # type: ignore[arg-type]


# --- the mandatory assumptions section ------------------------------------

def test_assumptions_from_the_script_are_listed() -> None:
    assert "horizon 15 (paper does not say)" in assumptions_section(make_card())


def test_unstated_paper_values_are_folded_in() -> None:
    card = make_card(claim=make_claim(missing=["the learning rate"]))
    items = assumptions_section(card)
    assert any("the learning rate" in item for item in items)


def test_assumptions_are_never_silently_empty() -> None:
    card = make_card(assumptions=[])
    items = assumptions_section(card)
    assert len(items) == 1
    assert "None recorded" in items[0]


def test_assumptions_do_not_duplicate() -> None:
    card = make_card(
        assumptions=["paper does not state: seeds"],
        claim=make_claim(missing=["seeds"]),
    )
    assert len(assumptions_section(card)) == 1


def test_markdown_always_has_an_assumptions_section() -> None:
    for assumptions in ([], ["horizon 15"]):
        body = render_markdown(make_card(assumptions=assumptions))
        assert "## ASSUMPTIONS" in body
        section = body.split("## ASSUMPTIONS", 1)[1]
        assert section.strip(), "the assumptions section must not be empty"


# --- report content --------------------------------------------------------

def test_markdown_carries_every_required_field() -> None:
    body = render_markdown(make_card())
    for needle in [
        "A Synthetic Paper",
        "demo-0001",
        "MPC reaches -150 return on Pendulum-v1.",
        "-150",
        "REPRODUCED",
        "anthropic/claude-sonnet-5",
        "±25%",
        "N=2",
    ]:
        assert needle in body, needle


def test_markdown_lists_each_seed_with_its_artifact_dir() -> None:
    comparison = compare(
        100.0,
        [
            SeedResult(0, 100.0, "score", 0, artifact_dir="/runs/x/seed_0"),
            SeedResult(1, None, None, 1, error="crashed", artifact_dir="/runs/x/seed_1"),
        ],
        0.25,
    )
    body = render_markdown(make_card(comparison=comparison))
    assert "/runs/x/seed_0" in body
    assert "/runs/x/seed_1" in body
    assert "crashed" in body


def test_untestable_report_explains_itself() -> None:
    comparison = untestable(None, 0.25, ["no identifiable environment"])
    body = render_markdown(make_card(comparison=comparison, assumptions=[]))
    assert "UNTESTABLE" in body
    assert "no identifiable environment" in body
    assert "does not specify enough" in body


def test_unspecified_fields_are_reported() -> None:
    card = make_card(claim=make_claim(specified={"hyperparameters": False}))
    body = render_markdown(card)
    assert "## What the paper left unspecified" in body
    assert "hyperparameters" in body


def test_fully_specified_paper_says_so() -> None:
    assert "nothing: all four schema fields were stated." in render_markdown(make_card())


# --- writing and terminal rendering ---------------------------------------

def test_write_report_emits_markdown_and_json(tmp_path: Path) -> None:
    card = make_card()
    path = write_report(card, tmp_path)
    assert path == tmp_path / "report.md"
    assert path.exists()

    payload = json.loads((tmp_path / "report.json").read_text())
    assert payload["verdict"] == "REPRODUCED"
    assert payload["model_slug"] == "anthropic/claude-sonnet-5"
    assert payload["our_mean"] == pytest.approx(-150.0)
    assert payload["assumptions"] == ["horizon 15 (paper does not say)"]
    assert len(payload["seed_results"]) == 2


@pytest.mark.parametrize("verdict", list(Verdict))
def test_terminal_render_works_for_every_verdict(verdict: Verdict) -> None:
    comparison = make_card().comparison
    comparison.verdict = verdict
    console = Console(record=True, width=120)
    render_terminal(make_card(comparison=comparison), console)
    output = console.export_text()
    assert verdict.value in output
    assert "ASSUMPTIONS" in output


def test_terminal_render_shows_failed_seeds() -> None:
    comparison = compare(
        100.0,
        [
            SeedResult(0, 100.0, "score", 0, artifact_dir="/runs/x/seed_0"),
            SeedResult(1, None, None, 1, error="it diverged", artifact_dir="/runs/x/seed_1"),
        ],
        0.25,
    )
    console = Console(record=True, width=120)
    render_terminal(make_card(comparison=comparison), console)
    output = console.export_text()
    assert "FAILED SEEDS" in output
    assert "it diverged" in output
