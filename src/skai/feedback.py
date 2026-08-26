"""User feedback capture: local SQLite always, plus a best-effort Langfuse score.

This is the closed loop that turns real usage into an eval signal: every rating
is stored locally (and can be exported to a DeepEval dataset) and, when Langfuse
is on, attached as a score to the exact trace of that turn.
"""
from __future__ import annotations

import json
import logging
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


def _conn(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(_SCHEMA)
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
