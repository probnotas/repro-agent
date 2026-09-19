"""arXiv fetching: metadata from the export API, PDF text via pymupdf.

Everything is cached under ``.cache/<arxiv_id>/`` so a second run of the same
paper costs nothing and works offline.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from xml.etree import ElementTree

import pymupdf

from .config import CACHE_DIR

ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_PDF = "https://arxiv.org/pdf/{arxiv_id}"
USER_AGENT = "repro-agent/0.1 (https://github.com/probnotas/repro-agent)"

_ATOM = {"atom": "http://www.w3.org/2005/Atom"}

#: 2007-and-later ids (2103.01955v2) and the legacy scheme (cs/0701001).
_ID_PATTERNS = (
    re.compile(r"(\d{4}\.\d{4,5}(?:v\d+)?)"),
    re.compile(r"([a-z\-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)"),
)


class FetchError(RuntimeError):
    """Raised when a paper cannot be fetched or parsed."""


@dataclass
class Paper:
    """Everything downstream stages need to know about a paper."""

    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    published: str
    text: str
    pdf_path: str | None = None

    @property
    def text_chars(self) -> int:
        return len(self.text)


def parse_arxiv_id(raw: str) -> str:
    """Accept an id or any arXiv URL and return the bare id.

    >>> parse_arxiv_id("https://arxiv.org/abs/1708.02596v2")
    '1708.02596v2'
    """
    candidate = raw.strip()
    if not candidate:
        raise FetchError("No arXiv id or URL given.")
    candidate = candidate.removeprefix("arXiv:").removeprefix("arxiv:")
    for pattern in _ID_PATTERNS:
        match = pattern.search(candidate)
        if match:
            return match.group(1)
    raise FetchError(
        f"Could not read an arXiv id out of {raw!r}. "
        "Try a bare id (2103.01955) or a full URL (https://arxiv.org/abs/2103.01955)."
    )


def cache_dir(arxiv_id: str) -> Path:
    """Per-paper cache directory, created on demand."""
    safe = arxiv_id.replace("/", "_")
    path = CACHE_DIR / safe
    path.mkdir(parents=True, exist_ok=True)
    return path


def _get(url: str, timeout: int = 60) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        raise FetchError(f"{url} returned HTTP {exc.code}: {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise FetchError(f"Could not reach {url}: {exc.reason}") from exc


def fetch_metadata(arxiv_id: str, *, refresh: bool = False) -> dict[str, object]:
    """Query the arXiv export API for title/authors/abstract, with caching."""
    target = cache_dir(arxiv_id) / "meta.json"
    if target.exists() and not refresh:
        return json.loads(target.read_text())

    query = urllib.parse.urlencode({"id_list": arxiv_id, "max_results": 1})
    raw = _get(f"{ARXIV_API}?{query}")
    try:
        root = ElementTree.fromstring(raw)
    except ElementTree.ParseError as exc:
        raise FetchError(f"arXiv returned unparseable XML for {arxiv_id}: {exc}") from exc

    entry = root.find("atom:entry", _ATOM)
    if entry is None:
        raise FetchError(f"arXiv has no entry for {arxiv_id!r}.")

    def _text(tag: str) -> str:
        node = entry.find(f"atom:{tag}", _ATOM)
        return (node.text or "").strip() if node is not None else ""

    title = " ".join(_text("title").split())
    if not title:
        raise FetchError(f"arXiv entry for {arxiv_id!r} has no title -- bad id?")

    meta: dict[str, object] = {
        "arxiv_id": arxiv_id,
        "title": title,
        "authors": [
            " ".join((node.findtext("atom:name", "", _ATOM) or "").split())
            for node in entry.findall("atom:author", _ATOM)
        ],
        "abstract": " ".join(_text("summary").split()),
        "published": _text("published"),
    }
    target.write_text(json.dumps(meta, indent=2))
    return meta


def fetch_pdf(arxiv_id: str, *, refresh: bool = False) -> Path:
    """Download the PDF into the cache and return its path."""
    target = cache_dir(arxiv_id) / "paper.pdf"
    if target.exists() and target.stat().st_size > 0 and not refresh:
        return target
    data = _get(ARXIV_PDF.format(arxiv_id=arxiv_id), timeout=120)
    if not data.startswith(b"%PDF"):
        raise FetchError(
            f"arXiv did not return a PDF for {arxiv_id!r} (got {len(data)} bytes "
            "that are not a PDF). The id may be wrong or withdrawn."
        )
    target.write_bytes(data)
    return target


def extract_text(pdf_path: Path, *, max_chars: int = 200_000) -> str:
    """Extract plain text from a PDF, truncated to keep prompts affordable."""
    try:
        with pymupdf.open(pdf_path) as document:
            pages = [page.get_text() for page in document]
    except Exception as exc:
        raise FetchError(f"Could not extract text from {pdf_path}: {exc}") from exc

    text = "\n".join(pages).strip()
    if not text:
        raise FetchError(
            f"{pdf_path} yielded no text. It is probably a scanned image; "
            "repro-agent does not OCR."
        )
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[...truncated by repro-agent...]"
    return text


def fetch_paper(arxiv_id_or_url: str, *, refresh: bool = False) -> Paper:
    """Fetch (or load from cache) everything about one paper."""
    arxiv_id = parse_arxiv_id(arxiv_id_or_url)
    directory = cache_dir(arxiv_id)
    text_path = directory / "text.txt"

    meta = fetch_metadata(arxiv_id, refresh=refresh)
    if text_path.exists() and not refresh:
        text = text_path.read_text()
        pdf_path = directory / "paper.pdf"
    else:
        pdf_path = fetch_pdf(arxiv_id, refresh=refresh)
        text = extract_text(pdf_path)
        text_path.write_text(text)

    return Paper(
        arxiv_id=arxiv_id,
        title=str(meta["title"]),
        authors=list(meta.get("authors", [])),  # type: ignore[arg-type]
        abstract=str(meta.get("abstract", "")),
        published=str(meta.get("published", "")),
        text=text,
        pdf_path=str(pdf_path) if pdf_path.exists() else None,
    )


def load_paper_from_fixture(fixture_dir: Path) -> Paper:
    """Build a :class:`Paper` from a fixture directory (``meta.json`` + ``text.txt``).

    This is what makes ``repro demo`` work with no network and no API key.
    """
    meta_path = fixture_dir / "meta.json"
    text_path = fixture_dir / "text.txt"
    if not meta_path.exists() or not text_path.exists():
        raise FetchError(f"Fixture at {fixture_dir} is missing meta.json or text.txt")
    meta = json.loads(meta_path.read_text())
    return Paper(
        arxiv_id=str(meta["arxiv_id"]),
        title=str(meta["title"]),
        authors=list(meta.get("authors", [])),
        abstract=str(meta.get("abstract", "")),
        published=str(meta.get("published", "")),
        text=text_path.read_text(),
        pdf_path=None,
    )


def paper_to_dict(paper: Paper) -> dict[str, object]:
    """Serialisable view of a paper, minus the full text."""
    data = asdict(paper)
    data.pop("text", None)
    return data
