"""The offline demo: the whole pipeline with no network and no API key.

Stages 1, 2 and 4 (fetch, extract, generate) are replaced by shipped fixtures.
Stages 5, 6 and 7 (execute, compare, report) run for real -- the number in the
demo's report card is measured by an actual subprocess, not canned.
"""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.markup import escape

from .config import DEFAULT_MODEL, OPENROUTER_BASE_URL, Settings
from .extract import Claim, parse_claim
from .fetch import Paper, load_paper_from_fixture
from .pipeline import RunOptions, run_pipeline

DEMO_CASES: tuple[str, ...] = ("testable", "untestable")

#: The demo runs a single seed by default so a live demo finishes in ~15 seconds.
DEMO_SEEDS: tuple[int, ...] = (0,)
DEMO_TIMEOUT: int = 900


class DemoError(RuntimeError):
    """Raised when the shipped fixtures cannot be located or read."""


def fixtures_root() -> Path:
    """Find ``tests/fixtures`` whether running from a checkout or an editable install."""
    candidates = [
        Path(__file__).resolve().parents[2] / "tests" / "fixtures",
        Path.cwd() / "tests" / "fixtures",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise DemoError(
        "Could not find tests/fixtures. Run `repro demo` from a repro-agent "
        f"checkout (looked in: {', '.join(str(path) for path in candidates)})."
    )


def fixture_dir(case: str = "testable") -> Path:
    """Directory for one demo case."""
    if case not in DEMO_CASES:
        raise DemoError(f"Unknown demo case {case!r}; choose from {list(DEMO_CASES)}")
    path = fixtures_root() / f"demo_{case}"
    if not path.is_dir():
        raise DemoError(f"Demo fixture directory is missing: {path}")
    return path


def load_fixture(case: str = "testable") -> tuple[Paper, Claim, str | None]:
    """Load ``(paper, claim, script)`` for a demo case. ``script`` may be None."""
    directory = fixture_dir(case)
    paper = load_paper_from_fixture(directory)

    claim_path = directory / "claim.json"
    if not claim_path.exists():
        raise DemoError(f"Demo fixture is missing claim.json: {claim_path}")
    claim = parse_claim(json.dumps(json.loads(claim_path.read_text())))

    script_path = directory / "script.py"
    script = script_path.read_text() if script_path.exists() else None
    return paper, claim, script


def demo_settings() -> Settings:
    """Settings with no key -- the demo never makes a model call."""
    return Settings(
        api_key="",
        model=f"{DEFAULT_MODEL} (not called: offline fixture demo)",
        base_url=OPENROUTER_BASE_URL,
    )


def run_demo(
    console: Console,
    case: str = "testable",
    *,
    tolerance: float = 0.25,
    seeds: tuple[int, ...] = DEMO_SEEDS,
    timeout: int = DEMO_TIMEOUT,
):
    """Run one demo case end-to-end and return its report card."""
    paper, claim, script = load_fixture(case)
    console.print(
        f"[bold]repro demo[/bold] [dim]— offline fixture, no API key, no network[/dim]"
    )
    console.print(f"[cyan]fetch[/cyan] loading fixture: {fixture_dir(case)}")
    console.print(f"  {escape(paper.title)}  ({paper.text_chars:,} chars of text)")
    console.print(
        "  [yellow]note:[/yellow] this is a synthetic paper, not a real one — see "
        "tests/fixtures/README.md"
    )
    options = RunOptions(tolerance=tolerance, seeds=seeds, timeout=timeout)
    return run_pipeline(
        paper, demo_settings(), options, console, claim=claim, script=script
    )
