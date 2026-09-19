"""Subprocess execution: artifacts, timeouts, exit codes, assumption capture."""

from __future__ import annotations

from pathlib import Path

import pytest
from rich.console import Console

from repro.compare import Verdict, compare
from repro.execute import collect_assumptions, execute_script

GOOD = """
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
print("ASSUMPTION: horizon 15 (paper does not say)")
print("ASSUMPTION: horizon 15 (paper does not say)")
print(f"working on seed {args.seed}")
print(f"RESULT: score={100 + args.seed}")
"""

CRASHES = """
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.parse_args()
raise RuntimeError("the dynamics model diverged")
"""

SILENT = """
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.parse_args()
print("did some work, forgot to report")
"""

HANGS = """
import argparse, time
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.parse_args()
time.sleep(60)
print("RESULT: score=1")
"""


def test_successful_run_writes_every_artifact(tmp_path: Path) -> None:
    results, assumptions = execute_script(GOOD, tmp_path, [0, 1], timeout=60)
    assert [r.value for r in results] == [100.0, 101.0]
    assert all(r.metric_name == "score" for r in results)
    assert all(r.ok for r in results)

    assert (tmp_path / "repro_script.py").read_text() == GOOD
    for seed in (0, 1):
        seed_dir = tmp_path / f"seed_{seed}"
        assert (seed_dir / "stdout.txt").exists()
        assert (seed_dir / "stderr.txt").exists()
        assert (seed_dir / "exit_code.txt").read_text() == "0"
        assert results[seed].artifact_dir == str(seed_dir)

    # Assumptions are captured once, in order, without duplicates.
    assert assumptions == ["horizon 15 (paper does not say)"]


def test_seed_is_actually_passed_through(tmp_path: Path) -> None:
    results, _ = execute_script(GOOD, tmp_path, [0, 5], timeout=60)
    assert results[1].value == 105.0
    assert "working on seed 5" in (tmp_path / "seed_5" / "stdout.txt").read_text()


def test_crash_is_recorded_not_swallowed(tmp_path: Path) -> None:
    results, _ = execute_script(CRASHES, tmp_path, [0], timeout=60)
    assert not results[0].ok
    assert results[0].exit_code != 0
    assert results[0].error is not None
    assert "diverged" in (tmp_path / "seed_0" / "stderr.txt").read_text()
    assert compare(100.0, results, 0.25).verdict is Verdict.RUN_FAILED


def test_clean_exit_without_a_result_line_is_a_failure(tmp_path: Path) -> None:
    results, _ = execute_script(SILENT, tmp_path, [0], timeout=60)
    assert results[0].exit_code == 0
    assert not results[0].ok
    assert "never printed a RESULT" in (results[0].error or "")


def test_timeout_is_recorded_with_its_budget(tmp_path: Path) -> None:
    results, _ = execute_script(HANGS, tmp_path, [0], timeout=2)
    assert not results[0].ok
    assert results[0].exit_code == -1
    assert "timed out after 2s" in (results[0].error or "")
    assert (tmp_path / "seed_0" / "exit_code.txt").read_text() == "-1"


def test_mixed_outcomes_across_seeds(tmp_path: Path) -> None:
    script = GOOD.replace(
        'print(f"RESULT: score={100 + args.seed}")',
        'import sys\n'
        'if args.seed == 1:\n'
        '    sys.exit("seed 1 is cursed")\n'
        'print(f"RESULT: score={100 + args.seed}")',
    )
    results, _ = execute_script(script, tmp_path, [0, 1, 2], timeout=60)
    assert [r.ok for r in results] == [True, False, True]
    comparison = compare(100.0, results, 0.25)
    assert comparison.n_seeds == 2
    assert any("2/3 seeds" in note for note in comparison.notes)


def test_progress_lines_reach_the_console(tmp_path: Path) -> None:
    console = Console(record=True, width=120)
    execute_script(GOOD, tmp_path, [0], timeout=60, progress=console)
    assert "seed 0: score=100" in console.export_text()


def test_collect_assumptions_dedupes_and_preserves_order() -> None:
    stdout = (
        "ASSUMPTION: b second\n"
        "noise\n"
        "  ASSUMPTION: a first  \n"
        "ASSUMPTION: b second\n"
        "ASSUMPTION:\n"  # empty payload is ignored
    )
    assert collect_assumptions(stdout) == ["b second", "a first"]


def test_generated_script_can_import_the_harness(tmp_path: Path) -> None:
    """PYTHONPATH must let the subprocess reach repro.harness."""
    script = (
        "import argparse\n"
        "parser = argparse.ArgumentParser(); parser.add_argument('--seed', type=int, default=0)\n"
        "parser.parse_args()\n"
        "from repro.harness import collect\n"
        "data = collect('Pendulum-v1', 16, seed=0)\n"
        "print(f'RESULT: n_transitions={len(data)}')\n"
    )
    results, _ = execute_script(script, tmp_path, [0], timeout=120)
    assert results[0].ok, (tmp_path / "seed_0" / "stderr.txt").read_text()
    assert results[0].value == 16.0


NAN_RESULT = """
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.parse_args()
print("RESULT: speedup=nan")
"""


def test_nan_result_is_reported_as_such_not_as_a_missing_line(tmp_path: Path) -> None:
    results, _ = execute_script(NAN_RESULT, tmp_path, [0], timeout=60)
    assert not results[0].ok
    assert "nan/inf" in (results[0].error or "")
    assert "speedup" in (results[0].error or "")
