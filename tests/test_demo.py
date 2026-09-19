"""The fixture demo: end-to-end with no network and no API key."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from rich.console import Console

from repro.compare import Verdict
from repro.demo import DemoError, fixture_dir, load_fixture, run_demo
from repro.triage import triage_claim


@pytest.fixture(autouse=True)
def run_in_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Keep demo artifacts out of the developer's real runs/ directory."""
    monkeypatch.setattr("repro.config.RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr("repro.execute.RUNS_DIR", tmp_path / "runs")
    return tmp_path


@pytest.fixture(autouse=True)
def no_api_key(monkeypatch: pytest.MonkeyPatch):
    """The demo must work with the key explicitly absent."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)


def test_fixtures_are_present() -> None:
    for case in ("testable", "untestable"):
        directory = fixture_dir(case)
        assert (directory / "meta.json").exists()
        assert (directory / "text.txt").exists()
        assert (directory / "claim.json").exists()


def test_unknown_case_raises() -> None:
    with pytest.raises(DemoError, match="Unknown demo case"):
        fixture_dir("nonsense")


def test_load_fixture_testable_has_a_script() -> None:
    paper, claim, script = load_fixture("testable")
    assert paper.arxiv_id == "demo-0001"
    assert paper.text_chars > 1000
    assert claim.reported_value == -210.0
    assert script is not None and "RESULT:" in script
    assert triage_claim(claim).testable


def test_load_fixture_untestable_has_no_script() -> None:
    paper, claim, script = load_fixture("untestable")
    assert paper.arxiv_id == "demo-0002"
    assert claim.reported_value is None
    assert script is None
    assert not triage_claim(claim).testable


def test_untestable_demo_stops_before_executing(run_in_tmp: Path) -> None:
    card = run_demo(Console(), "untestable")
    assert card.comparison.verdict is Verdict.UNTESTABLE
    assert card.comparison.our_mean is None
    assert card.comparison.seed_results == []
    assert card.comparison.notes, "UNTESTABLE must say why"
    # The assumptions section is mandatory and must not be empty here.
    assert card.assumptions


def test_untestable_demo_writes_a_report(run_in_tmp: Path) -> None:
    card = run_demo(Console(), "untestable")
    report = Path(card.run_path) / "report.md"
    assert report.exists()
    body = report.read_text()
    assert "UNTESTABLE" in body
    assert "## ASSUMPTIONS" in body
    payload = json.loads((Path(card.run_path) / "report.json").read_text())
    assert payload["verdict"] == "UNTESTABLE"


@pytest.mark.slow
def test_testable_demo_runs_end_to_end(run_in_tmp: Path) -> None:
    """The real thing: collect, fit, plan, compare. Takes ~15s."""
    card = run_demo(Console(), "testable", seeds=(0,), timeout=900)
    assert card.comparison.verdict in (Verdict.REPRODUCED, Verdict.NOT_REPRODUCED)
    assert card.comparison.our_mean is not None
    assert card.comparison.n_seeds == 1
    assert card.comparison.metric_name

    run_path = Path(card.run_path)
    # Every number in the report traces back to a file under runs/.
    assert (run_path / "repro_script.py").exists()
    assert (run_path / "seed_0" / "stdout.txt").exists()
    assert (run_path / "seed_0" / "exit_code.txt").read_text() == "0"
    assert f"{card.comparison.our_mean}" in (run_path / "seed_0" / "stdout.txt").read_text()
    assert card.assumptions, "the script declared assumptions; they must be captured"


@pytest.mark.slow
def test_demo_never_reads_the_api_key(run_in_tmp: Path, monkeypatch) -> None:
    """A poisoned key must not break the demo, because nothing calls out."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "definitely-not-a-key")

    def explode(*args: object, **kwargs: object) -> None:
        raise AssertionError("the demo made a model call")

    monkeypatch.setattr("repro.pipeline.extract_claim", explode)
    monkeypatch.setattr("repro.pipeline.generate_script", explode)
    card = run_demo(Console(), "testable", seeds=(0,), timeout=900)
    assert card.comparison.our_mean is not None
    assert "not called" in card.model_slug
