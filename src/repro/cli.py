"""``repro`` -- the command-line entry point."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from . import __version__
from .compare import Verdict
from .config import (
    DEFAULT_SEEDS,
    DEFAULT_TIMEOUT,
    DEFAULT_TOLERANCE,
    ConfigError,
    get_settings,
)
from .demo import DEMO_CASES, DemoError, run_demo
from .execute import paper_dir
from .extract import ExtractionError, extract_claim
from .fetch import FetchError, fetch_paper, parse_arxiv_id
from .generate import GenerationError
from .llm import LLMError, list_models
from .pipeline import RunOptions, run_from_arxiv
from .report import VERDICT_BLURB
from .triage import triage_claim

console = Console()
error_console = Console(stderr=True)

VERDICT_COLOR: dict[str, str] = {
    Verdict.REPRODUCED.value: "green",
    Verdict.NOT_REPRODUCED.value: "red",
    Verdict.UNTESTABLE.value: "yellow",
    Verdict.RUN_FAILED.value: "magenta",
}


def _die(message: str, code: int = 1) -> None:
    """Print a loud, human-readable error and exit."""
    error_console.print(f"[bold red]error[/bold red] {message}")
    sys.exit(code)


def _parse_seeds(raw: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(part) for part in raw.split(",") if part.strip())
    except ValueError:
        raise click.BadParameter(f"{raw!r} is not a comma-separated list of integers")
    if not seeds:
        raise click.BadParameter("at least one seed is required")
    return seeds


def _settings(model: str | None):
    try:
        return get_settings(model)
    except ConfigError as exc:
        _die(str(exc), code=2)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="repro")
def main() -> None:
    """repro — a paper replication agent for model-based robot control research.

    Point it at an arXiv paper; it extracts the headline empirical claim, decides
    whether the paper says enough to test it fairly, and if so runs the test and
    reports a verdict. It is not a summarizer: the unit of value is a claim that
    survived execution.
    """


@main.command()
@click.argument("arxiv_id_or_url")
@click.option("--tolerance", default=DEFAULT_TOLERANCE, show_default=True,
              help="Relative band around the paper's number that counts as reproduced.")
@click.option("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS), show_default=True,
              help="Comma-separated seeds to run.")
@click.option("--timeout", default=DEFAULT_TIMEOUT, show_default=True,
              help="Hard per-seed timeout in seconds.")
@click.option("--model", default=None, help="OpenRouter slug; overrides OPENROUTER_MODEL.")
@click.option("--refresh", is_flag=True, help="Ignore the cache and re-fetch the paper.")
def run(arxiv_id_or_url: str, tolerance: float, seeds: str, timeout: int,
        model: str | None, refresh: bool) -> None:
    """Run the full pipeline on a paper and print a report card."""
    if tolerance < 0:
        _die("--tolerance must be non-negative.")
    settings = _settings(model)
    options = RunOptions(tolerance=tolerance, seeds=_parse_seeds(seeds), timeout=timeout)
    console.print(f"[dim]model: {settings.model}[/dim]")
    try:
        card = run_from_arxiv(
            arxiv_id_or_url, settings, options, console, refresh=refresh
        )
    except (FetchError, ExtractionError, GenerationError, LLMError) as exc:
        _die(str(exc))
    if card.comparison.verdict in (Verdict.RUN_FAILED,):
        sys.exit(1)


@main.command()
@click.argument("arxiv_id_or_url")
@click.option("--model", default=None, help="OpenRouter slug; overrides OPENROUTER_MODEL.")
@click.option("--refresh", is_flag=True, help="Ignore the cache and re-fetch the paper.")
def extract(arxiv_id_or_url: str, model: str | None, refresh: bool) -> None:
    """Print the extracted claim JSON. No script is generated and nothing is run."""
    settings = _settings(model)
    try:
        paper = fetch_paper(arxiv_id_or_url, refresh=refresh)
        claim, _ = extract_claim(paper.title, paper.text, settings=settings)
    except (FetchError, ExtractionError, LLMError) as exc:
        _die(str(exc))

    console.print_json(json.dumps(claim.to_dict(), indent=2))

    verdict = triage_claim(claim)
    if verdict.testable:
        console.print("\n[green]triage:[/green] testable — `repro run` would proceed.")
    else:
        console.print("\n[yellow]triage: UNTESTABLE[/yellow] — `repro run` would stop here:")
        for reason in verdict.reasons:
            console.print(f"  • {reason}")


@main.command()
@click.argument("arxiv_id_or_url")
def show(arxiv_id_or_url: str) -> None:
    """Print the cached report for a paper."""
    try:
        arxiv_id = parse_arxiv_id(arxiv_id_or_url)
    except FetchError:
        arxiv_id = arxiv_id_or_url.strip()  # demo ids are not arXiv-shaped

    report = paper_dir(arxiv_id) / "report.md"
    if not report.exists():
        _die(
            f"No cached report for {arxiv_id!r} (looked for {report}).\n"
            f"Run `repro run {arxiv_id}` first, or `repro list` to see what is there."
        )
    console.print(report.read_text())


@main.command(name="list")
def list_runs() -> None:
    """List every past run and its verdict."""
    from .config import RUNS_DIR

    if not RUNS_DIR.is_dir():
        console.print("[dim]No runs yet. Try `repro demo`.[/dim]")
        return

    rows: list[tuple[str, str, str, str, str]] = []
    for report_path in sorted(RUNS_DIR.glob("*/report.json")):
        try:
            data = json.loads(report_path.read_text())
        except json.JSONDecodeError:
            continue
        verdict = str(data.get("verdict", "?"))
        ours = data.get("our_mean")
        rows.append((
            str(data.get("arxiv_id", report_path.parent.name)),
            str(data.get("title", ""))[:48],
            verdict,
            "n/a" if ours is None else f"{float(ours):.4g}",
            str(data.get("generated_at", "")),
        ))

    if not rows:
        console.print("[dim]No completed runs yet. Try `repro demo`.[/dim]")
        return

    table = Table(title="repro runs", header_style="bold cyan")
    table.add_column("arxiv id")
    table.add_column("title")
    table.add_column("verdict")
    table.add_column("our value", justify="right")
    table.add_column("when", style="dim")
    for arxiv_id, title, verdict, ours, when in sorted(rows, key=lambda r: r[4], reverse=True):
        color = VERDICT_COLOR.get(verdict, "white")
        table.add_row(arxiv_id, title, f"[{color}]{verdict}[/{color}]", ours, when)
    console.print(table)


@main.command()
@click.option("--filter", "needle", default="", help="Only show slugs containing this text.")
@click.option("--limit", default=40, show_default=True, help="Maximum rows to print.")
def models(needle: str, limit: int) -> None:
    """List OpenRouter model slugs and prices, so you can switch OPENROUTER_MODEL."""
    settings = _settings(None)
    try:
        catalogue = list_models(settings)
    except LLMError as exc:
        _die(str(exc))

    def price(entry: dict, key: str) -> str:
        raw = (entry.get("pricing") or {}).get(key)
        try:
            return f"${float(raw) * 1_000_000:.2f}"
        except (TypeError, ValueError):
            return "—"

    matches = [
        entry for entry in catalogue
        if not needle or needle.lower() in str(entry.get("id", "")).lower()
    ]
    matches.sort(key=lambda entry: str(entry.get("id", "")))

    table = Table(
        title=f"OpenRouter models ({len(matches)} match, showing {min(limit, len(matches))})",
        header_style="bold cyan",
        caption="prices are USD per 1M tokens · full list: https://openrouter.ai/models",
    )
    table.add_column("slug")
    table.add_column("context", justify="right")
    table.add_column("$/1M in", justify="right")
    table.add_column("$/1M out", justify="right")
    for entry in matches[:limit]:
        slug = str(entry.get("id", "?"))
        table.add_row(
            f"[bold]{slug}[/bold]" if slug == settings.model else slug,
            f"{entry.get('context_length') or '—':,}" if entry.get("context_length") else "—",
            price(entry, "prompt"),
            price(entry, "completion"),
        )
    console.print(table)
    console.print(f"[dim]current OPENROUTER_MODEL: {settings.model}[/dim]")


@main.command()
@click.option("--case", type=click.Choice(DEMO_CASES), default="testable", show_default=True,
              help="Which shipped fixture to run.")
@click.option("--tolerance", default=DEFAULT_TOLERANCE, show_default=True)
@click.option("--seeds", default="0", show_default=True, help="Comma-separated seeds to run.")
@click.option("--timeout", default=900, show_default=True, help="Hard per-seed timeout in seconds.")
def demo(case: str, tolerance: float, seeds: str, timeout: int) -> None:
    """Run a pre-cached example end-to-end. No API key, no network.

    Stages 5-7 (execute, compare, report) run for real: the number in the report
    card is measured by an actual subprocess.
    """
    try:
        card = run_demo(
            console, case, tolerance=tolerance, seeds=_parse_seeds(seeds), timeout=timeout
        )
    except DemoError as exc:
        _die(str(exc))

    console.print(
        f"[dim]verdict meaning: {VERDICT_BLURB[card.comparison.verdict]}[/dim]"
    )
    if card.comparison.verdict is Verdict.RUN_FAILED:
        sys.exit(1)


@main.command()
def verdicts() -> None:
    """Explain the four verdicts."""
    table = Table(title="the four verdicts", header_style="bold cyan")
    table.add_column("verdict")
    table.add_column("means")
    for verdict, blurb in VERDICT_BLURB.items():
        table.add_row(f"[{VERDICT_COLOR[verdict.value]}]{verdict.value}[/]", blurb)
    console.print(table)


if __name__ == "__main__":  # pragma: no cover
    main()
