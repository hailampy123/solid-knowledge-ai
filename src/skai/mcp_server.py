"""MCP server exposing the knowledge base as tools.

Any MCP client (Claude Desktop, an IDE) can call `search_kb` for raw retrieval
or `ask` to run the full self-reflective agent. Both reuse the exact same core
as the CLI — no logic duplication.

Run with:  skai mcp
"""
from __future__ import annotations

from functools import lru_cache

from mcp.server.mcpserver import MCPServer

from skai.agent.graph import answer_question, build_graph, make_checkpointer
from skai.agent.llm import make_llm
from skai.config import get_settings
from skai.ingest.store import Store

server = MCPServer(name="solid-knowledge-ai")


@lru_cache
def _store() -> Store:
    s = get_settings()
    return Store(s.chroma_dir, s.collection)


@lru_cache
def _graph():
    s = get_settings()
    return build_graph(
        _store(),
        make_llm(s, callbacks=[]),
        top_k=s.top_k,
        max_retries=s.max_retries,
        checkpointer=make_checkpointer(s.memory_db),
    )


def search_core(store: Store, query: str, source_type: str | None = None, k: int = 5) -> list[dict]:
    return [
        {"text": h.text, "source_id": h.metadata.get("source_id"), "score": round(h.score, 4)}
        for h in store.query(query, k=k, source_type=source_type)
    ]


@server.tool()
def search_kb(query: str, source_type: str | None = None) -> list[dict]:
    """Semantic search over the ingested documents. Optional source_type: pdf|md|web."""
    return search_core(_store(), query, source_type, k=get_settings().top_k)


@server.tool()
def ask(question: str) -> str:
    """Ask the self-reflective knowledge agent. Returns a grounded, cited answer."""
    out = answer_question(_graph(), question, thread_id="mcp")
    sources = f"\n\nSources: {', '.join(out['citations'])}" if out.get("citations") else ""
    return out["answer"] + sources


def serve() -> None:
    server.run(transport="stdio")
