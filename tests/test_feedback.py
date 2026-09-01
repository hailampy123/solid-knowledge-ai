from skai import feedback


def _db(tmp_path):
    return str(tmp_path / "feedback.sqlite")


def test_record_and_stats(tmp_path):
    db = _db(tmp_path)
    feedback.record(db, question="q1", answer="a1", route="kb", model="haiku", citations=["a"], rating="up")
    feedback.record(db, question="q2", answer="a2", route="kb", model="haiku", citations=[], rating="down", comment="wrong")
    feedback.record(db, question="q3", answer="a3", route="kb", model="sonnet", citations=["c"], rating="up")

    s = feedback.stats(db)
    assert s == {"up": 2, "down": 1, "total": 3}


def test_stats_missing_db_is_zero(tmp_path):
    assert feedback.stats(_db(tmp_path)) == {"up": 0, "down": 0, "total": 0}


def test_export_jsonl(tmp_path):
    db = _db(tmp_path)
    feedback.record(db, question="q", answer="a", route="kb", model="haiku", citations=["a", "b"], rating="up", comment="nice")
    out = str(tmp_path / "seed.jsonl")
    n = feedback.export_jsonl(db, out)
    assert n == 1
    line = open(out).read().strip()
    import json
    row = json.loads(line)
    assert row["question"] == "q"
    assert row["citations"] == ["a", "b"]
    assert row["rating"] == "up"


def test_gap_report_clusters_by_question(tmp_path):
    db = _db(tmp_path)
    # same question hedged twice + a different one with no relevant docs
    feedback.log_gap(db, question="Do orcas sleep?", route="kb", answer="hedge", docs_ok=True, grounded=False)
    feedback.log_gap(db, question="do orcas SLEEP?", route="kb", answer="hedge", docs_ok=True, grounded=False)
    feedback.log_gap(db, question="What is a narwhal?", route="kb", answer="hedge", docs_ok=False, grounded=False)

    report = feedback.gap_report(db)
    assert [r["count"] for r in report] == [2, 1]  # most frequent first
    top = report[0]
    assert top["count"] == 2 and top["reasons"] == ["ungrounded/hedged"]
    assert set(report[1]["reasons"]) == {"no relevant docs", "ungrounded/hedged"}


def test_gap_report_missing_db_is_empty(tmp_path):
    assert feedback.gap_report(_db(tmp_path)) == []


def test_promote_downvotes_seeds_golden_and_dedups(tmp_path):
    db = _db(tmp_path)
    feedback.record(db, question="up q", answer="a", route="kb", model="haiku", citations=[], rating="up", comment="great")
    feedback.record(db, question="no comment", answer="a", route="kb", model="haiku", citations=[], rating="down")
    feedback.record(db, question="Where do orcas migrate?", answer="a", route="kb", model="haiku",
                    citations=[], rating="down", comment="missed the Antarctic migration route")

    out = str(tmp_path / "golden.jsonl")
    assert feedback.promote_downvotes(db, out) == 1  # only the 👎 with a comment
    import json
    rows = [json.loads(ln) for ln in open(out) if ln.strip()]
    assert rows[0]["question"] == "Where do orcas migrate?"
    assert "antarctic" in rows[0]["must_include"] and "migration" in rows[0]["must_include"]

    assert feedback.promote_downvotes(db, out) == 0  # idempotent: no duplicate case


def test_push_langfuse_score_noop_without_keys(tmp_path):
    from skai.config import Settings

    settings = Settings(_env_file=None, LANGFUSE_PUBLIC_KEY=None, LANGFUSE_SECRET_KEY=None)
    # disabled + no trace id => silent no-op, must not raise
    feedback.push_langfuse_score(settings, None, "up", "x")
