"""Source adapters that normalize PDF / Markdown / web content into Documents.

Each adapter returns a list of Documents with a common metadata contract:
    source_type: "pdf" | "md" | "web"
    source_id:   stable, human-readable id used in citations
    title:       best-effort human title
    uri:         present for web sources
"""
from __future__ import annotations

import logging
from pathlib import Path

from skai.models import Document

logger = logging.getLogger(__name__)


def load_markdown(path: str | Path) -> list[Document]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    title = _first_heading(text) or path.stem
    return [
        Document(
            text=text,
            metadata={
                "source_type": "md",
                "source_id": str(path),
                "title": title,
            },
        )
    ]


def load_pdf(path: str | Path) -> list[Document]:
    from pypdf import PdfReader

    path = Path(path)
    reader = PdfReader(str(path))
    docs: list[Document] = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        docs.append(
            Document(
                text=text,
                metadata={
                    "source_type": "pdf",
                    "source_id": f"{path.name}#p{i + 1}",
                    "title": path.stem,
                    "page": i + 1,
                },
            )
        )
    return docs


def load_web(url: str, *, _html: str | None = None) -> list[Document]:
    """Fetch and extract the main article text from a URL.

    Pass `_html` to supply page HTML directly (tests avoid the network).
    """
    import trafilatura

    html = _html if _html is not None else _fetch(url)
    text = trafilatura.extract(html, include_comments=False, include_tables=False)
    if not text:
        raise ValueError(f"no extractable content at {url}")
    title = _html_title(html) or url
    return [
        Document(
            text=text,
            metadata={
                "source_type": "web",
                "source_id": url,
                "title": title,
                "uri": url,
            },
        )
    ]


def load_sources(doc_dir: str | Path, urls: list[str] | None = None) -> list[Document]:
    """Load every .md/.pdf under doc_dir plus each URL. Failures are logged and skipped."""
    doc_dir = Path(doc_dir)
    urls = urls or []
    docs: list[Document] = []

    for path in sorted(doc_dir.rglob("*")):
        try:
            if path.suffix.lower() in {".md", ".markdown", ".txt"}:
                docs += load_markdown(path)
            elif path.suffix.lower() == ".pdf":
                docs += load_pdf(path)
        except Exception as e:  # noqa: BLE001 - one bad file must not abort ingest
            logger.warning("skipping %s: %s", path, e)

    for url in urls:
        try:
            docs += load_web(url)
        except Exception as e:  # noqa: BLE001
            logger.warning("skipping %s: %s", url, e)

    return docs


def _fetch(url: str) -> str:
    import httpx

    resp = httpx.get(url, timeout=20.0, follow_redirects=True, headers={"User-Agent": "skai/0.1"})
    resp.raise_for_status()
    return resp.text


def _first_heading(md: str) -> str | None:
    for line in md.splitlines():
        line = line.strip()
        if line.startswith("#"):
            return line.lstrip("#").strip()
    return None


def _html_title(html: str) -> str | None:
    import re

    m = re.search(r"<title>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else None
