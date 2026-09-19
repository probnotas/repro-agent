"""The seven-stage pipeline: fetch → extract → triage → generate → execute →
compare → report.

Each stage writes its artifacts under ``runs/<arxiv_id>/<timestamp>/`` before the
next one begins, so a run that dies halfway still leaves an auditable trail.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

from .compare import Comparison, compare, untestable
from .config import DEFAULT_SEEDS, DEFAULT_TIMEOUT, DEFAULT_TOLERANCE, Settings
from .execute import execute_script, paper_dir, run_dir
from .extract import Claim, extract_claim
from .fetch import Paper, fetch_paper
from .generate import generate_script
from .report import ReportCard, render_terminal, write_report
from .triage import triage_claim


@dataclass
class RunOptions:
    """Everything the user can dial on a single run."""

    tolerance: float = DEFAULT_TOLERANCE
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    timeout: int = DEFAULT_TIMEOUT


def _save(directory: Path, name: str, payload: object) -> None:
    path = directory / name
    if isinstance(payload, str):
        path.write_text(payload)
    else:
        path.write_text(json.dumps(payload, indent=2))


def run_pipeline(
    paper: Paper,
    settings: Settings,
    options: RunOptions,
    console: Console,
    *,
    claim: Claim | None = None,
    script: str | None = None,
) -> ReportCard:
    """Run the pipeline for one already-fetched paper and return its report card.

    ``claim`` and ``script`` may be supplied to skip the model calls -- that is
    how ``repro demo`` runs end-to-end with no key and no network.
    """
    directory = run_dir(paper.arxiv_id)
    _save(directory, "paper.json", {
        "arxiv_id": paper.arxiv_id,
        "title": paper.title,
        "authors": paper.authors,
        "published": paper.published,
        "text_chars": paper.text_chars,
    })

    # 2. EXTRACT
    if claim is None:
        console.print("[cyan]extract[/cyan] asking the model for the testable claim…")
        claim, raw = extract_claim(paper.title, paper.text, settings=settings)
        _save(directory, "claim_raw.txt", raw)
    else:
        console.print("[cyan]extract[/cyan] using the supplied claim (no model call)")
    _save(directory, "claim.json", claim.to_dict())
    console.print(f"  claim: [italic]{claim.claim_text}[/italic]")

    # 3. TRIAGE
    console.print("[cyan]triage[/cyan] is this specified well enough to test fairly?")
    verdict = triage_claim(claim)
    _save(directory, "triage.json", {
        "testable": verdict.testable,
        "reasons": verdict.reasons,
        "underspecified": verdict.underspecified,
    })

    if not verdict.testable:
        console.print("  [yellow]UNTESTABLE[/yellow] — stopping before we invent an experiment")
        comparison = untestable(
            claim.reported_value,
            options.tolerance,
            verdict.reasons,
            metric_name=claim.metric,
        )
        card = ReportCard(
            arxiv_id=paper.arxiv_id,
            title=paper.title,
            claim=claim,
            comparison=comparison,
            model_slug=settings.model,
            assumptions=[f"underspecified in the paper: {item}" for item in verdict.underspecified],
            run_path=str(directory),
            tolerance=options.tolerance,
            seeds=list(options.seeds),
        )
        return _finish(card, paper, directory, console)

    console.print("  testable — proceeding")

    # 4. GENERATE
    if script is None:
        console.print("[cyan]generate[/cyan] writing the repro script…")
        script = generate_script(paper.title, claim, settings=settings)
    else:
        console.print("[cyan]generate[/cyan] using the supplied script (no model call)")

    # 5. EXECUTE
    console.print(
        f"[cyan]execute[/cyan] running seeds {list(options.seeds)} "
        f"(timeout {options.timeout}s each)…"
    )
    results, assumptions = execute_script(
        script, directory, list(options.seeds), options.timeout, progress=console
    )

    # 6. COMPARE
    comparison: Comparison = compare(
        claim.reported_value, results, options.tolerance, metric_name=claim.metric
    )

    card = ReportCard(
        arxiv_id=paper.arxiv_id,
        title=paper.title,
        claim=claim,
        comparison=comparison,
        model_slug=settings.model,
        assumptions=assumptions,
        run_path=str(directory),
        tolerance=options.tolerance,
        seeds=list(options.seeds),
    )
    return _finish(card, paper, directory, console)


def _finish(card: ReportCard, paper: Paper, directory: Path, console: Console) -> ReportCard:
    """Stage 7: write the report into the run dir and the paper dir, then print."""
    write_report(card, directory)
    write_report(card, paper_dir(paper.arxiv_id))
    render_terminal(card, console)
    return card


def run_from_arxiv(
    arxiv_id_or_url: str,
    settings: Settings,
    options: RunOptions,
    console: Console,
    *,
    refresh: bool = False,
) -> ReportCard:
    """Stage 1 (fetch) plus the rest of the pipeline."""
    console.print(f"[cyan]fetch[/cyan] {arxiv_id_or_url}")
    paper = fetch_paper(arxiv_id_or_url, refresh=refresh)
    console.print(f"  {paper.title}  ({paper.text_chars:,} chars of text)")
    return run_pipeline(paper, settings, options, console)
