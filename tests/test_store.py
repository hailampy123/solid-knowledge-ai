from skai.ingest.store import Store
from skai.models import Chunk


def _chunks():
    return [
        Chunk(text="orcas hunt seals in coordinated pods", metadata={"source_type": "md", "source_id": "a", "chunk_index": 0}),
        Chunk(text="python is a programming language for data", metadata={"source_type": "web", "source_id": "b", "chunk_index": 0}),
        Chunk(text="orca pods have distinct vocal dialects", metadata={"source_type": "md", "source_id": "c", "chunk_index": 0}),
    ]


def test_add_and_count(tmp_path, ef):
    store = Store(str(tmp_path), "test_kb", embedding_function=ef)
    assert store.add(_chunks()) == 3
    assert store.count() == 3


def test_query_ranks_relevant_first(tmp_path, ef):
    store = Store(str(tmp_path), "test_kb", embedding_function=ef)
    store.add(_chunks())
    hits = store.query("orca pods dialects", k=3)
    assert hits
    assert "orca" in hits[0].text
    assert hits[0].score >= hits[-1].score  # sorted best-first


def test_source_type_filter(tmp_path, ef):
    store = Store(str(tmp_path), "test_kb", embedding_function=ef)
    store.add(_chunks())
    hits = store.query("orca", k=5, source_type="web")
    assert all(h.metadata["source_type"] == "web" for h in hits)


def test_upsert_is_idempotent(tmp_path, ef):
    store = Store(str(tmp_path), "test_kb", embedding_function=ef)
    store.add(_chunks())
    store.add(_chunks())  # same ids => overwrite, not duplicate
    assert store.count() == 3
