"""arXiv id parsing, fixture loading, and generated-script sanity checks.

The network is mocked or avoided entirely.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from repro import fetch
from repro.extract import Claim
from repro.fetch import FetchError, load_paper_from_fixture, parse_arxiv_id
from repro.generate import GenerationError, generate_script, strip_code_fences


# --- id parsing ------------------------------------------------------------

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("2103.01955", "2103.01955"),
        ("1708.02596v2", "1708.02596v2"),
        ("arXiv:1708.02596", "1708.02596"),
        ("https://arxiv.org/abs/1708.02596", "1708.02596"),
        ("https://arxiv.org/abs/1708.02596v2", "1708.02596v2"),
        ("http://arxiv.org/pdf/2103.01955v1", "2103.01955v1"),
        ("https://arxiv.org/pdf/2103.01955.pdf", "2103.01955"),
        ("  2103.01955  ", "2103.01955"),
        ("cs/0701001", "cs/0701001"),
        ("https://arxiv.org/abs/cs/0701001", "cs/0701001"),
    ],
)
def test_parse_arxiv_id(raw: str, expected: str) -> None:
    assert parse_arxiv_id(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", "not a paper", "https://example.com/paper"])
def test_parse_arxiv_id_rejects_junk(raw: str) -> None:
    with pytest.raises(FetchError):
        parse_arxiv_id(raw)


# --- fixture loading -------------------------------------------------------

def test_load_paper_from_fixture(fixtures_dir: Path) -> None:
    paper = load_paper_from_fixture(fixtures_dir / "demo_testable")
    assert paper.arxiv_id == "demo-0001"
    assert "Pendulum" in paper.title
    assert paper.authors
    assert paper.text_chars == len(paper.text)


def test_load_paper_from_an_incomplete_fixture(tmp_path: Path) -> None:
    (tmp_path / "meta.json").write_text(json.dumps({"arxiv_id": "x", "title": "y"}))
    with pytest.raises(FetchError, match="missing meta.json or text.txt"):
        load_paper_from_fixture(tmp_path)


def test_extract_text_rejects_an_empty_pdf(tmp_path: Path, monkeypatch) -> None:
    class FakeDoc:
        def __iter__(self):
            return iter([])

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    monkeypatch.setattr(fetch.pymupdf, "open", lambda path: FakeDoc())
    with pytest.raises(FetchError, match="does not OCR"):
        fetch.extract_text(tmp_path / "paper.pdf")


def test_extract_text_truncates_long_papers(tmp_path: Path, monkeypatch) -> None:
    class FakePage:
        def get_text(self) -> str:
            return "x" * 5000

    class FakeDoc:
        def __iter__(self):
            return iter([FakePage()] * 10)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    monkeypatch.setattr(fetch.pymupdf, "open", lambda path: FakeDoc())
    text = fetch.extract_text(tmp_path / "paper.pdf", max_chars=1000)
    assert text.startswith("x" * 1000)
    assert "truncated by repro-agent" in text


# --- script generation -----------------------------------------------------

def test_strip_code_fences() -> None:
    assert strip_code_fences("```python\nprint(1)\n```") == "print(1)"
    assert strip_code_fences("```\nprint(1)\n```") == "print(1)"
    assert strip_code_fences("print(1)") == "print(1)"


def claim() -> Claim:
    return Claim(
        claim_text="MPC reaches -150 return on Pendulum-v1.",
        method="random-shooting MPC",
        environment="Pendulum-v1",
        metric="mean_episode_return",
        reported_value=-150.0,
        conditions=["2000 transitions"],
        specified={"hyperparameters": True, "seeds": True,
                   "env_version": True, "success_threshold": True},
        missing=[],
    )


def test_generate_strips_fences_from_the_model_response(monkeypatch) -> None:
    script = "import argparse\nprint('RESULT: score=1')\n"
    monkeypatch.setattr(
        "repro.generate.complete", lambda *a, **k: f"```python\n{script}```"
    )
    assert generate_script("T", claim()) == script.strip()


def test_generate_rejects_a_script_with_no_result_line(monkeypatch) -> None:
    monkeypatch.setattr("repro.generate.complete", lambda *a, **k: "print('hello')")
    with pytest.raises(GenerationError, match="never prints a `RESULT:` line"):
        generate_script("T", claim())


def test_generate_rejects_an_empty_response(monkeypatch) -> None:
    monkeypatch.setattr("repro.generate.complete", lambda *a, **k: "   ")
    with pytest.raises(GenerationError, match="empty script"):
        generate_script("T", claim())


def test_generate_asks_for_plain_text_not_json(monkeypatch) -> None:
    captured: dict = {}

    def fake_complete(system, user, json_mode=True, **kwargs):
        captured.update(system=system, user=user, json_mode=json_mode)
        return "print('RESULT: score=1')"

    monkeypatch.setattr("repro.generate.complete", fake_complete)
    generate_script("A Paper Title", claim())
    assert captured["json_mode"] is False
    assert "A Paper Title" in captured["user"]
    assert "Pendulum-v1" in captured["user"]
    assert "RESULT:" in captured["system"]
    assert "ASSUMPTION:" in captured["system"]


def test_generate_passes_the_missing_list_to_the_model(monkeypatch) -> None:
    captured: dict = {}

    def fake_complete(system, user, json_mode=True, **kwargs):
        captured["user"] = user
        return "print('RESULT: score=1')"

    monkeypatch.setattr("repro.generate.complete", fake_complete)
    unspecified = claim()
    unspecified.missing = ["the learning rate"]
    generate_script("T", unspecified)
    assert "the learning rate" in captured["user"]
