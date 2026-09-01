"""User feedback capture: local SQLite always, plus a best-effort Langfuse score.

This is the closed loop that turns real usage into an eval signal: every rating
is stored locally (and can be exported to a DeepEval dataset) and, when Langfuse
is on, attached as a score to the exact trace of that turn.
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from pathlib import Path

from skai.config import Settings

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL, question TEXT, answer TEXT, route TEXT, model TEXT,
    citations TEXT, rating TEXT, comment TEXT, trace_id TEXT
)
"""

# Retrieval gaps: turns the agent could not ground (no relevant docs after
# retries, or self_check hedged). Collected automatically, not user-triggered.
_GAPS_SCHEMA = """
CREATE TABLE IF NOT EXISTS gaps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL, question TEXT, route TEXT, answer TEXT,
    docs_ok INTEGER, grounded INTEGER
)
"""


def _conn(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(_SCHEMA)
    conn.execute(_GAPS_SCHEMA)
    return conn


def record(
    db_path: str,
    *,
    question: str,
    answer: str,
    route: str | None,
    model: str,
    citations: list[str],
    rating: str,
    comment: str = "",
    trace_id: str | None = None,
    ts: float | None = None,
) -> None:
    with _conn(db_path) as conn:
        conn.execute(
            "INSERT INTO feedback (ts, question, answer, route, model, citations, rating, comment, trace_id)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (
                ts if ts is not None else time.time(),
                question, answer, route, model,
                json.dumps(citations), rating, comment, trace_id,
            ),
        )


def stats(db_path: str) -> dict:
    if not Path(db_path).exists():
        return {"up": 0, "down": 0, "total": 0}
    with _conn(db_path) as conn:
        rows = dict(conn.execute("SELECT rating, COUNT(*) FROM feedback GROUP BY rating").fetchall())
    up, down = rows.get("up", 0), rows.get("down", 0)
    return {"up": up, "down": down, "total": up + down}


def export_jsonl(db_path: str, out_path: str) -> int:
    """Export feedback rows as a JSONL eval-seed dataset. Returns the row count."""
    with _conn(db_path) as conn:
        rows = conn.execute(
            "SELECT question, answer, citations, rating, comment FROM feedback"
        ).fetchall()
    with open(out_path, "w") as f:
        for q, a, cites, rating, comment in rows:
            f.write(json.dumps({
                "question": q, "answer": a,
                "citations": json.loads(cites), "rating": rating, "comment": comment,
            }) + "\n")
    return len(rows)


# --- closing the loop: gaps -> content backlog, 👎 -> golden eval set ---------

def log_gap(
    db_path: str,
    *,
    question: str,
    route: str | None,
    answer: str,
    docs_ok: bool | None,
    grounded: bool | None,
    ts: float | None = None,
) -> None:
    """Record a turn the agent couldn't ground. Best-effort; never raise into a turn."""
    def _flag(v: bool | None) -> int | None:
        return None if v is None else int(bool(v))

    try:
        with _conn(db_path) as conn:
            conn.execute(
                "INSERT INTO gaps (ts, question, route, answer, docs_ok, grounded)"
                " VALUES (?,?,?,?,?,?)",
                (ts if ts is not None else time.time(), question, route, answer,
                 _flag(docs_ok), _flag(grounded)),
            )
    except Exception as e:  # noqa: BLE001 - logging a gap must not break the answer
        logger.warning("gap log failed: %s", e)


def gap_report(db_path: str) -> list[dict]:
    """The content backlog: unanswered/hedged questions, most frequent first.

    ponytail: clusters by normalized-exact text (lower/collapse-ws); swap for
    embedding clustering when the backlog outgrows eyeballing.
    """
    if not Path(db_path).exists():
        return []
    with _conn(db_path) as conn:
        rows = conn.execute("SELECT question, ts, docs_ok, grounded FROM gaps").fetchall()
    clusters: dict[str, dict] = {}
    for q, ts, docs_ok, grounded in rows:
        key = " ".join((q or "").lower().split())
        c = clusters.setdefault(key, {"question": q, "count": 0, "last_ts": 0.0, "reasons": set()})
        c["count"] += 1
        c["last_ts"] = max(c["last_ts"], ts or 0.0)
        if docs_ok == 0:
            c["reasons"].add("no relevant docs")
        if grounded == 0:
            c["reasons"].add("ungrounded/hedged")
    out = [{**c, "reasons": sorted(c["reasons"])} for c in clusters.values()]
    out.sort(key=lambda c: (-c["count"], -c["last_ts"]))
    return out


_STOPWORDS = {
    "this", "that", "with", "from", "have", "what", "when", "where", "which",
    "should", "would", "about", "there", "wrong", "answer", "because", "does",
}


def _keywords(text: str, n: int = 4) -> list[str]:
    seen: list[str] = []
    for w in re.findall(r"[a-z0-9]{4,}", (text or "").lower()):
        if w not in _STOPWORDS and w not in seen:
            seen.append(w)
    return seen[:n]


def promote_downvotes(db_path: str, golden_path: str) -> int:
    """Turn reviewed 👎 (a thumbs-down carrying a comment) into golden eval cases.

    Appends de-duplicated {question, must_include} rows to `golden_path` (JSONL);
    must_include is seeded from the comment's keywords for a human to refine.
    Returns the number of new cases added. This is the body of the weekly
    "grow the eval set" job — the scheduler lives at deploy time, not here.
    """
    if not Path(db_path).exists():
        return 0
    with _conn(db_path) as conn:
        rows = conn.execute(
            "SELECT question, comment FROM feedback WHERE rating='down' AND comment != ''"
        ).fetchall()

    existing: set[str] = set()
    p = Path(golden_path)
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                existing.add(json.loads(line)["question"].strip().lower())

    added = 0
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(golden_path, "a") as f:
        for q, comment in rows:
            key = (q or "").strip().lower()
            if not key or key in existing:
                continue
            existing.add(key)
            f.write(json.dumps({"question": q, "must_include": _keywords(comment)}) + "\n")
            added += 1
    return added


def push_langfuse_score(settings: Settings, trace_id: str | None, rating: str, comment: str = "") -> None:
    """Attach the rating to the Langfuse trace as a 0/1 score. Best-effort."""
    if not settings.langfuse_enabled or not trace_id:
        return
    try:
        from langfuse import get_client

        get_client().create_score(
            trace_id=trace_id,
            name="user-feedback",
            value=1.0 if rating == "up" else 0.0,
            comment=comment or None,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("Langfuse score failed: %s", e)
