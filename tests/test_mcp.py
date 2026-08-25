import asyncio

from skai import mcp_server
from skai.ingest.store import Store
from skai.models import Chunk


def test_tools_registered():
    tools = asyncio.run(mcp_server.server.list_tools())
    names = {t.name for t in tools}
    assert {"search_kb", "ask"} <= names


def test_search_core_returns_hits(tmp_path, ef):
    store = Store(str(tmp_path), "mcp_kb", embedding_function=ef)
    store.add(
        [
            Chunk(text="orcas hunt seals in coordinated pods", metadata={"source_type": "md", "source_id": "a", "chunk_index": 0}),
            Chunk(text="python data language", metadata={"source_type": "web", "source_id": "b", "chunk_index": 0}),
        ]
    )
    hits = mcp_server.search_core(store, "orca pods", k=2)
    assert hits
    assert hits[0]["source_id"] == "a"
    assert "score" in hits[0]
