"""Run a generated script in a subprocess, once per seed, with a hard timeout.

Generated code is never ``exec()``-ed in this process. Every script, its stdout,
its stderr and its exit code land in ``runs/<arxiv_id>/<timestamp>/`` so every
number in a report can be traced back to the file that produced it.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .compare import SeedResult, find_non_finite_result, parse_result_line
from .config import RUNS_DIR

ASSUMPTION_PREFIX = "ASSUMPTION:"


def run_dir(arxiv_id: str, timestamp: str | None = None) -> Path:
    """Create and return ``runs/<arxiv_id>/<timestamp>/``."""
    stamp = timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = RUNS_DIR / arxiv_id.replace("/", "_") / stamp
    path.mkdir(parents=True, exist_ok=True)
    return path


def paper_dir(arxiv_id: str) -> Path:
    """Create and return ``runs/<arxiv_id>/``."""
    path = RUNS_DIR / arxiv_id.replace("/", "_")
    path.mkdir(parents=True, exist_ok=True)
    return path


def collect_assumptions(stdout: str) -> list[str]:
    """Pull every ``ASSUMPTION:`` line out of a script's stdout, in order."""
    seen: list[str] = []
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith(ASSUMPTION_PREFIX):
            text = stripped[len(ASSUMPTION_PREFIX) :].strip()
            if text and text not in seen:
                seen.append(text)
    return seen


def execute_script(
    script: str,
    directory: Path,
    seeds: list[int],
    timeout: int,
    *,
    progress: object | None = None,
) -> tuple[list[SeedResult], list[str]]:
    """Run ``script`` once per seed and return ``(results, assumptions)``.

    ``progress`` may be any object with a ``print(str)`` method (e.g. a rich
    Console); it receives one line per seed as the runs complete.
    """
    script_path = directory / "repro_script.py"
    script_path.write_text(script)

    environment = os.environ.copy()
    # Make the installed package importable even from a bare checkout, and keep
    # torch from oversubscribing cores when several seeds share a laptop.
    src = str(Path(__file__).resolve().parents[1])
    environment["PYTHONPATH"] = (
        src + os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else src
    )
    environment.setdefault("OMP_NUM_THREADS", "2")
    environment.setdefault("MKL_NUM_THREADS", "2")
    environment.setdefault("PYTHONUNBUFFERED", "1")

    results: list[SeedResult] = []
    assumptions: list[str] = []

    for seed in seeds:
        seed_dir = directory / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        error: str | None = None
        try:
            completed = subprocess.run(
                [sys.executable, str(script_path), "--seed", str(seed)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(directory),
                env=environment,
            )
            stdout, stderr, exit_code = completed.stdout, completed.stderr, completed.returncode
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            exit_code = -1
            error = f"timed out after {timeout}s"

        (seed_dir / "stdout.txt").write_text(stdout)
        (seed_dir / "stderr.txt").write_text(stderr)
        (seed_dir / "exit_code.txt").write_text(str(exit_code))

        parsed = parse_result_line(stdout)
        if error is None and exit_code != 0:
            tail = (stderr.strip().splitlines() or ["no stderr"])[-1]
            error = f"exit {exit_code}: {tail[:200]}"
        elif error is None and parsed is None:
            non_finite = find_non_finite_result(stdout)
            error = (
                f"script reported RESULT: {non_finite}=nan/inf, which is not a number"
                if non_finite
                else "script exited 0 but never printed a RESULT: line"
            )

        for assumption in collect_assumptions(stdout):
            if assumption not in assumptions:
                assumptions.append(assumption)

        result = SeedResult(
            seed=seed,
            value=parsed[1] if parsed else None,
            metric_name=parsed[0] if parsed else None,
            exit_code=exit_code,
            error=error,
            artifact_dir=str(seed_dir),
        )
        results.append(result)

        if progress is not None and hasattr(progress, "print"):
            if result.ok:
                progress.print(f"  seed {seed}: {result.metric_name}={result.value:g}")
            else:
                progress.print(f"  seed {seed}: [red]FAILED[/red] ({result.error})")

    return results, assumptions
