from skai.models import AgentState, Chunk, Document, RetrievedChunk


def test_document_and_chunk_metadata_roundtrip():
    d = Document(text="hello", metadata={"source_type": "md", "source_id": "a.md"})
    assert d.text == "hello"
    assert d.metadata["source_type"] == "md"

    c = Chunk(text="hi", metadata={"chunk_index": 0, "source_id": "a.md"})
    assert c.metadata["chunk_index"] == 0


def test_document_default_metadata_is_isolated():
    a = Document(text="x")
    b = Document(text="y")
    a.metadata["k"] = 1
    assert b.metadata == {}  # no shared mutable default


def test_retrieved_chunk_has_score():
    r = RetrievedChunk(text="t", metadata={"source_id": "a"}, score=0.9)
    assert r.score == 0.9


def test_agent_state_keys():
    keys = AgentState.__annotations__.keys()
    for k in ("question", "messages", "route", "docs", "answer", "citations", "retries"):
        assert k in keys
