"""A tiny local web front end for repro.

Standard library only -- no web framework. It binds to localhost by default and
is meant for one person driving one machine, because a run executes
model-generated Python on that machine.

Runs are long (minutes), so the browser does not wait on one request: POST
/api/run starts a job and returns its id, and the page polls /api/job/<id> for
the live stage log and, eventually, the report card JSON.
"""

from __future__ import annotations

import json
import threading
import uuid
import webbrowser
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO
from pathlib import Path
from typing import Any

from rich.console import Console

from .compare import Verdict
from .config import ConfigError, DEFAULT_SEEDS, DEFAULT_TOLERANCE, get_settings
from .demo import DEMO_CASES, run_demo
from .fetch import parse_arxiv_id
from .pipeline import RunOptions, run_from_arxiv

MAX_BODY_BYTES = 64 * 1024

#: Where index.html lives, whether running from a checkout or an editable install.
_WEB_CANDIDATES = (
    Path(__file__).resolve().parents[2] / "web",
    Path.cwd() / "web",
)


def web_root() -> Path:
    for candidate in _WEB_CANDIDATES:
        if (candidate / "index.html").is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find web/index.html. Run `repro serve` from a repro-agent "
        f"checkout (looked in: {', '.join(str(p) for p in _WEB_CANDIDATES)})."
    )


@dataclass
class Job:
    """One pipeline run, its live log, and its eventual result."""

    job_id: str
    status: str = "running"  # running | done | error
    log: StringIO = field(default_factory=StringIO)
    report: dict[str, Any] | None = None
    error: str | None = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "log": self.log.getvalue(),
            "report": self.report,
            "error": self.error,
        }


_jobs: dict[str, Job] = {}
_jobs_lock = threading.Lock()


def _console_for(job: Job) -> Console:
    """A Console that writes into the job's log buffer instead of a terminal."""
    return Console(file=job.log, force_terminal=False, width=100, no_color=True)


def _parse_seeds(raw: Any) -> tuple[int, ...]:
    if isinstance(raw, list):
        return tuple(int(value) for value in raw)
    text = str(raw or "").strip()
    if not text:
        return DEFAULT_SEEDS
    seeds = tuple(int(part) for part in text.split(",") if part.strip())
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


def _run_job(job: Job, payload: dict[str, Any]) -> None:
    """Body of the worker thread: run either the fixture demo or a live paper."""
    console = _console_for(job)
    try:
        tolerance = float(payload.get("tolerance") or DEFAULT_TOLERANCE)
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative")
        seeds = _parse_seeds(payload.get("seeds"))
        timeout = int(payload.get("timeout") or 900)
        target = str(payload.get("input") or "").strip()

        if payload.get("mode") == "demo" or target.lower() in {"", "demo"}:
            case = str(payload.get("case") or "testable")
            if case not in DEMO_CASES:
                raise ValueError(f"unknown demo case {case!r}")
            card = run_demo(
                console, case, tolerance=tolerance, seeds=seeds, timeout=timeout,
                render=False,  # the page renders the card; the log stays progress
            )
        else:
            # Check the input the user actually typed before asking them for a
            # key: "that is not an arXiv id" is more useful than "no key set".
            parse_arxiv_id(target)
            settings = get_settings(payload.get("model") or None)
            console.print(f"model: {settings.model}")
            card = run_from_arxiv(
                target,
                settings,
                RunOptions(tolerance=tolerance, seeds=seeds, timeout=timeout),
                console,
                render=False,  # the page renders the card; the log stays progress
            )

        job.report = card.to_dict()
        job.status = "done"
    except ConfigError as exc:
        job.error = str(exc)
        job.status = "error"
    except Exception as exc:  # surface the real message rather than a blank 500
        job.error = f"{type(exc).__name__}: {exc}"
        job.status = "error"


class Handler(BaseHTTPRequestHandler):
    server_version = "repro"

    def log_message(self, fmt: str, *args: Any) -> None:  # quieter console
        return

    # --- helpers ---------------------------------------------------------

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, code: int, payload: dict[str, Any]) -> None:
        self._send(code, json.dumps(payload).encode(), "application/json")

    # --- routes ----------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802  (stdlib naming)
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            try:
                body = (web_root() / "index.html").read_bytes()
            except FileNotFoundError as exc:
                self._send(500, str(exc).encode(), "text/plain; charset=utf-8")
                return
            self._send(200, body, "text/html; charset=utf-8")
            return

        if path.startswith("/api/job/"):
            job_id = path.rsplit("/", 1)[-1]
            with _jobs_lock:
                job = _jobs.get(job_id)
            if job is None:
                self._send_json(404, {"error": f"no such job {job_id}"})
                return
            self._send_json(200, job.snapshot())
            return

        if path == "/api/verdicts":
            self._send_json(200, {"verdicts": [v.value for v in Verdict]})
            return

        self._send_json(404, {"error": f"no route for {path}"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path.split("?", 1)[0] != "/api/run":
            self._send_json(404, {"error": f"no route for {self.path}"})
            return

        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY_BYTES:
            self._send_json(413, {"error": "request body too large"})
            return
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as exc:
            self._send_json(400, {"error": f"body is not valid JSON: {exc}"})
            return
        if not isinstance(payload, dict):
            self._send_json(400, {"error": "body must be a JSON object"})
            return

        job = Job(job_id=uuid.uuid4().hex[:12])
        with _jobs_lock:
            _jobs[job.job_id] = job
        threading.Thread(target=_run_job, args=(job, payload), daemon=True).start()
        self._send_json(202, {"job_id": job.job_id})


def serve(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    """Start the local server and block until interrupted."""
    web_root()  # fail fast with a clear message if index.html is missing
    httpd = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}/"
    print(f"repro is serving at {url}")
    print("A run executes model-generated Python on this machine, so this")
    print("server binds to localhost. Do not expose it to a network.")
    print("Press Ctrl-C to stop.")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        httpd.server_close()
