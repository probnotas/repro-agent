"""Render the report card: a colored terminal table and ``runs/<id>/report.md``.

The ASSUMPTIONS section is mandatory. If anything was guessed it must appear
there; if nothing was guessed that is stated explicitly rather than left blank.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .compare import Comparison, Verdict
from .extract import Claim

VERDICT_STYLE: dict[Verdict, str] = {
    Verdict.REPRODUCED: "bold green",
    Verdict.NOT_REPRODUCED: "bold red",
    Verdict.UNTESTABLE: "bold yellow",
    Verdict.RUN_FAILED: "bold magenta",
}

VERDICT_BLURB: dict[Verdict, str] = {
    Verdict.REPRODUCED: "our number landed inside the tolerance band around the paper's",
    Verdict.NOT_REPRODUCED: "the script ran, but our number fell outside the tolerance band",
    Verdict.UNTESTABLE: "the paper does not specify enough to run a fair test",
    Verdict.RUN_FAILED: "the repro script did not produce a usable number",
}


@dataclass
class ReportCard:
    """Everything needed to render one report, in one place."""

    arxiv_id: str
    title: str
    claim: Claim
    comparison: Comparison
    model_slug: str
    assumptions: list[str] = field(default_factory=list)
    run_path: str | None = None
    tolerance: float = 0.25
    seeds: list[int] = field(default_factory=list)
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    )

    def to_dict(self) -> dict[str, object]:
        comparison = self.comparison
        return {
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "generated_at": self.generated_at,
            "model_slug": self.model_slug,
            "tolerance": self.tolerance,
            "seeds": self.seeds,
            "verdict": comparison.verdict.value,
            "claim": self.claim.to_dict(),
            "reported_value": comparison.reported_value,
            "our_mean": comparison.our_mean,
            "our_std": comparison.our_std,
            "n_seeds": comparison.n_seeds,
            "delta_pct": comparison.delta_pct,
            "metric_name": comparison.metric_name,
            "assumptions": self.assumptions,
            "notes": comparison.notes,
            "run_path": self.run_path,
            "seed_results": [
                {
                    "seed": result.seed,
                    "value": result.value,
                    "exit_code": result.exit_code,
                    "error": result.error,
                    "artifact_dir": result.artifact_dir,
                }
                for result in comparison.seed_results
            ],
        }


def _fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}g}"


def _our_value(comparison: Comparison) -> str:
    if comparison.our_mean is None:
        return "n/a"
    return (
        f"{_fmt(comparison.our_mean)} ± {_fmt(comparison.our_std)} "
        f"(N={comparison.n_seeds})"
    )


def _delta(comparison: Comparison) -> str:
    if comparison.delta_pct is None:
        return "n/a"
    return f"{comparison.delta_pct:+.1f}%"


def assumptions_section(card: ReportCard) -> list[str]:
    """The mandatory assumptions list, never silently empty."""
    items = list(card.assumptions)
    for entry in card.claim.missing:
        text = f"paper does not state: {entry}"
        if text not in items:
            items.append(text)
    if not items:
        items.append(
            "None recorded. The paper specified every value the run needed, and the "
            "script declared no invented value."
        )
    return items


def render_terminal(card: ReportCard, console: Console | None = None) -> None:
    """Print the colored report card."""
    out = console or Console()
    comparison = card.comparison
    style = VERDICT_STYLE[comparison.verdict]

    table = Table(show_header=False, box=None, pad_edge=False, padding=(0, 2))
    table.add_column("field", style="bold cyan", no_wrap=True)
    table.add_column("value", overflow="fold")

    table.add_row("paper", f"{card.title}  [dim](arXiv:{card.arxiv_id})[/dim]")
    table.add_row("claim", card.claim.claim_text)
    table.add_row("metric", comparison.metric_name or card.claim.metric)
    table.add_row("reported", _fmt(comparison.reported_value))
    table.add_row("ours", _our_value(comparison))
    table.add_row("delta", _delta(comparison))
    table.add_row("tolerance", f"±{card.tolerance:.0%}")
    table.add_row("verdict", Text(comparison.verdict.value, style=style))
    table.add_row("model", card.model_slug)
    if card.run_path:
        table.add_row("artifacts", card.run_path)

    out.print()
    out.print(Panel(table, title="repro report card", border_style=style, expand=False))

    if comparison.notes:
        out.print("[bold]NOTES[/bold]")
        for note in comparison.notes:
            out.print(f"  • {note}")
        out.print()

    out.print("[bold]ASSUMPTIONS[/bold] [dim](every value the agent had to invent)[/dim]")
    for item in assumptions_section(card):
        out.print(f"  • {item}")
    out.print()

    if comparison.failed_seeds:
        out.print("[bold]FAILED SEEDS[/bold]")
        for result in comparison.failed_seeds:
            out.print(f"  • seed {result.seed}: {result.error}  [dim]{result.artifact_dir}[/dim]")
        out.print()


def render_markdown(card: ReportCard) -> str:
    """Render the same report card as markdown."""
    comparison = card.comparison
    lines: list[str] = [
        f"# repro report card — arXiv:{card.arxiv_id}",
        "",
        f"**{card.title}**",
        "",
        f"- generated: {card.generated_at}",
        f"- model: `{card.model_slug}`",
        f"- tolerance: ±{card.tolerance:.0%}",
        f"- seeds: {card.seeds or 'n/a'}",
        f"- artifacts: `{card.run_path or 'n/a'}`",
        "",
        "## Verdict",
        "",
        f"## `{comparison.verdict.value}`",
        "",
        f"_{VERDICT_BLURB[comparison.verdict]}_",
        "",
        "## Claim",
        "",
        f"> {card.claim.claim_text}",
        "",
        f"- method: {card.claim.method}",
        f"- environment: {card.claim.environment}",
        f"- metric: {card.claim.metric}",
        "",
        "## Numbers",
        "",
        "| | value |",
        "| --- | --- |",
        f"| reported (paper) | {_fmt(comparison.reported_value)} |",
        f"| ours (mean ± std over N={comparison.n_seeds} seeds) | {_our_value(comparison)} |",
        f"| delta | {_delta(comparison)} |",
        f"| tolerance band | ±{card.tolerance:.0%} |",
        "",
    ]

    if comparison.seed_results:
        lines += ["### Per seed", "", "| seed | value | exit | artifacts |", "| --- | --- | --- | --- |"]
        for result in comparison.seed_results:
            value = _fmt(result.value) if result.value is not None else f"— ({result.error})"
            lines.append(
                f"| {result.seed} | {value} | {result.exit_code} | `{result.artifact_dir or '—'}` |"
            )
        lines.append("")

    if comparison.notes:
        lines += ["## Notes", ""] + [f"- {note}" for note in comparison.notes] + [""]

    lines += [
        "## ASSUMPTIONS",
        "",
        "Every value the agent had to invent because the paper does not state it.",
        "A non-reproduction may reflect these choices rather than the paper.",
        "",
    ]
    lines += [f"- {item}" for item in assumptions_section(card)]
    lines += [
        "",
        "## What the paper left unspecified",
        "",
    ]
    unspecified = card.claim.unspecified
    if unspecified:
        lines += [f"- {item}" for item in unspecified]
    else:
        lines.append("- nothing: all four schema fields were stated.")
    lines += ["", "---", "", "Generated by [repro-agent](https://github.com/probnotas/repro-agent).", ""]
    return "\n".join(lines)


def write_report(card: ReportCard, directory: Path) -> Path:
    """Write ``report.md`` and ``report.json`` into ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    markdown_path = directory / "report.md"
    markdown_path.write_text(render_markdown(card))
    (directory / "report.json").write_text(json.dumps(card.to_dict(), indent=2))
    return markdown_path
