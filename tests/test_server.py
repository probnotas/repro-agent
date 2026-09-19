"""The local web front end: routing, job lifecycle, and error surfacing.

The server is driven through a real socket on an ephemeral port, but every run
it starts is the offline fixture demo -- no network, no API key.
"""

from __future__ import annotations

import json
import re
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from repro import server as srv
from repro.server import Handler, web_root


@pytest.fixture(autouse=True)
def isolated_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Keep run artifacts out of the real runs/ directory.

    The no-key path is kept hermetic by the suite-wide fixture in conftest.
    """
    monkeypatch.setattr("repro.config.RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr("repro.execute.RUNS_DIR", tmp_path / "runs")
    return tmp_path


@pytest.fixture
def base_url():
    """A live server on an ephemeral port, torn down after the test."""
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()


def get(url: str) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()


def post(url: str, payload: dict | str) -> tuple[int, str]:
    body = payload.encode() if isinstance(payload, str) else json.dumps(payload).encode()
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()


def wait_for(base: str, job_id: str, timeout: float = 240.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        _, body = get(f"{base}/api/job/{job_id}")
        job = json.loads(body)
        if job["status"] != "running":
            return job
        time.sleep(0.4)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s")


# --- the page itself -------------------------------------------------------

def test_index_html_is_shipped_and_self_contained() -> None:
    """The page must render with no network: nothing is fetched from off-box.

    An http(s) string is not itself a violation -- the footer links to the repo
    and the inline favicon carries the SVG namespace URI. What matters is that
    nothing is *loaded* from elsewhere.
    """
    html = (web_root() / "index.html").read_text()
    assert "<!DOCTYPE html>" in html
    assert "<script src=" not in html
    assert '<link rel="stylesheet"' not in html
    assert "@import" not in html
    assert "url(http" not in html
    assert "<img" not in html
    # Only the footer's repo link may point off-box, and only as a link.
    external = re.findall(r'(?:src|href)\s*=\s*"(https?://[^"]+)"', html)
    assert external == ["https://github.com/probnotas/repro-agent"], external
    # It must know all four verdicts.
    for verdict in ("REPRODUCED", "NOT_REPRODUCED", "UNTESTABLE", "RUN_FAILED"):
        assert verdict in html


def test_get_root_serves_the_page(base_url: str) -> None:
    status, body = get(base_url + "/")
    assert status == 200
    assert "<!DOCTYPE html>" in body


def test_unknown_route_is_404(base_url: str) -> None:
    status, body = get(base_url + "/nope")
    assert status == 404
    assert "no route" in json.loads(body)["error"]


def test_unknown_job_is_404(base_url: str) -> None:
    status, body = get(base_url + "/api/job/deadbeef")
    assert status == 404
    assert "no such job" in json.loads(body)["error"]


def test_verdicts_endpoint(base_url: str) -> None:
    status, body = get(base_url + "/api/verdicts")
    assert status == 200
    assert json.loads(body)["verdicts"] == [
        "REPRODUCED", "NOT_REPRODUCED", "UNTESTABLE", "RUN_FAILED"
    ]


# --- request validation ----------------------------------------------------

def test_malformed_body_is_400(base_url: str) -> None:
    status, body = post(base_url + "/api/run", "not json")
    assert status == 400
    assert "not valid JSON" in json.loads(body)["error"]


def test_non_object_body_is_400(base_url: str) -> None:
    status, body = post(base_url + "/api/run", "[1, 2]")
    assert status == 400
    assert "must be a JSON object" in json.loads(body)["error"]


def test_post_to_the_wrong_path_is_404(base_url: str) -> None:
    assert post(base_url + "/api/nope", {})[0] == 404


# --- job lifecycle ---------------------------------------------------------

def test_untestable_demo_through_the_api(base_url: str) -> None:
    status, body = post(base_url + "/api/run", {"mode": "demo", "case": "untestable"})
    assert status == 202
    job = wait_for(base_url, json.loads(body)["job_id"])
    assert job["status"] == "done"
    assert job["report"]["verdict"] == "UNTESTABLE"
    assert job["report"]["assumptions"], "the assumptions list is mandatory"
    assert "UNTESTABLE" in job["log"]


def test_empty_input_falls_back_to_the_demo(base_url: str) -> None:
    """The page's placeholder promises this: empty box -> offline demo."""
    _, body = post(base_url + "/api/run", {"input": "  ", "case": "untestable"})
    job = wait_for(base_url, json.loads(body)["job_id"])
    assert job["status"] == "done"
    assert job["report"]["arxiv_id"] == "demo-0002"


def test_bad_arxiv_id_surfaces_the_real_message(base_url: str) -> None:
    _, body = post(base_url + "/api/run", {"input": "not-a-paper"})
    job = wait_for(base_url, json.loads(body)["job_id"])
    assert job["status"] == "error"
    assert "arXiv id" in job["error"]
    assert job["report"] is None


def test_unknown_demo_case_is_an_error_not_a_crash(base_url: str) -> None:
    _, body = post(base_url + "/api/run", {"mode": "demo", "case": "nonsense"})
    job = wait_for(base_url, json.loads(body)["job_id"])
    assert job["status"] == "error"
    assert "nonsense" in job["error"]


def test_bad_seeds_are_rejected_with_a_message(base_url: str) -> None:
    _, body = post(base_url + "/api/run", {"mode": "demo", "seeds": "a,b"})
    job = wait_for(base_url, json.loads(body)["job_id"])
    assert job["status"] == "error"


def test_negative_tolerance_is_rejected(base_url: str) -> None:
    _, body = post(base_url + "/api/run", {"mode": "demo", "tolerance": -1})
    job = wait_for(base_url, json.loads(body)["job_id"])
    assert job["status"] == "error"
    assert "non-negative" in job["error"]


def test_live_run_without_a_key_reports_the_config_error(base_url: str) -> None:
    """The no-key path must say how to fix it, not just fail."""
    _, body = post(base_url + "/api/run", {"input": "1708.02596"})
    job = wait_for(base_url, json.loads(body)["job_id"])
    assert job["status"] == "error"
    assert "cp .env.example .env" in job["error"]


@pytest.mark.slow
def test_testable_demo_through_the_api_really_executes(base_url: str) -> None:
    _, body = post(base_url + "/api/run", {"mode": "demo", "case": "testable", "seeds": "0"})
    job = wait_for(base_url, json.loads(body)["job_id"])
    assert job["status"] == "done"
    report = job["report"]
    assert report["verdict"] in ("REPRODUCED", "NOT_REPRODUCED")
    assert report["our_mean"] is not None
    assert report["n_seeds"] == 1
    assert report["seed_results"][0]["exit_code"] == 0


# --- helpers ---------------------------------------------------------------

def test_parse_seeds_accepts_both_shapes() -> None:
    assert srv._parse_seeds("0,1,2") == (0, 1, 2)
    assert srv._parse_seeds([0, 1]) == (0, 1)
    assert srv._parse_seeds("") == srv.DEFAULT_SEEDS
    assert srv._parse_seeds(None) == srv.DEFAULT_SEEDS
    with pytest.raises(ValueError):
        srv._parse_seeds("a,b")
