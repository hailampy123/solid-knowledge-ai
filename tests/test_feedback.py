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


def test_push_langfuse_score_noop_without_keys(tmp_path):
    from skai.config import Settings

    settings = Settings(_env_file=None, LANGFUSE_PUBLIC_KEY=None, LANGFUSE_SECRET_KEY=None)
    # disabled + no trace id => silent no-op, must not raise
    feedback.push_langfuse_score(settings, None, "up", "x")
