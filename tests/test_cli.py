"""CLI surface: commands exist, the missing-key path exits cleanly, demo needs no key."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from repro.cli import main


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture(autouse=True)
def isolated_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("repro.config.RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr("repro.execute.RUNS_DIR", tmp_path / "runs")
    return tmp_path


@pytest.fixture
def no_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr("repro.cli.get_settings", _raise_config_error)


def _raise_config_error(*args, **kwargs):
    from repro.config import ConfigError, _MISSING_KEY_MESSAGE

    raise ConfigError(_MISSING_KEY_MESSAGE)


def test_every_documented_command_exists(runner: CliRunner) -> None:
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    for command in ("run", "extract", "show", "list", "models", "demo"):
        assert command in result.output


@pytest.mark.parametrize("command", ["run", "extract", "show", "list", "models", "demo"])
def test_each_command_has_help(runner: CliRunner, command: str) -> None:
    assert runner.invoke(main, [command, "--help"]).exit_code == 0


def test_run_has_the_documented_options(runner: CliRunner) -> None:
    output = runner.invoke(main, ["run", "--help"]).output
    for option in ("--tolerance", "--seeds", "--timeout", "--model"):
        assert option in output


@pytest.mark.parametrize("command", [["run", "2103.01955"], ["extract", "2103.01955"], ["models"]])
def test_missing_key_exits_2_with_instructions(runner: CliRunner, no_key, command) -> None:
    result = runner.invoke(main, command)
    assert result.exit_code == 2
    assert "cp .env.example .env" in result.output


def test_bad_seed_list_is_rejected(runner: CliRunner) -> None:
    result = runner.invoke(main, ["demo", "--seeds", "a,b"])
    assert result.exit_code != 0
    assert "comma-separated list of integers" in result.output


def test_empty_seed_list_is_rejected(runner: CliRunner) -> None:
    assert runner.invoke(main, ["demo", "--seeds", ","]).exit_code != 0


def test_list_with_no_runs_says_so(runner: CliRunner) -> None:
    result = runner.invoke(main, ["list"])
    assert result.exit_code == 0
    assert "No runs yet" in result.output


def test_show_without_a_cached_report_exits_1(runner: CliRunner) -> None:
    result = runner.invoke(main, ["show", "2103.01955"])
    assert result.exit_code == 1
    assert "No cached report" in result.output


def test_verdicts_command_explains_all_four(runner: CliRunner) -> None:
    result = runner.invoke(main, ["verdicts"])
    assert result.exit_code == 0
    for verdict in ("REPRODUCED", "NOT_REPRODUCED", "UNTESTABLE", "RUN_FAILED"):
        assert verdict in result.output


def test_demo_runs_with_the_key_explicitly_absent(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, isolated_runs: Path
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    result = runner.invoke(main, ["demo", "--case", "untestable"])
    assert result.exit_code == 0, result.output
    assert "UNTESTABLE" in result.output


def test_list_shows_a_completed_run(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, isolated_runs: Path
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert runner.invoke(main, ["demo", "--case", "untestable"]).exit_code == 0

    result = runner.invoke(main, ["list"])
    assert result.exit_code == 0
    assert "demo-0002" in result.output
    assert "UNTESTABLE" in result.output

    shown = runner.invoke(main, ["show", "demo-0002"])
    assert shown.exit_code == 0
    assert "ASSUMPTIONS" in shown.output
