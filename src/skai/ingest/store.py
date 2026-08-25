"""Persistent Chroma vector store: add chunks, query with optional source filter.

Embeddings default to Chroma's local MiniLM (all-MiniLM-L6-v2) so the demo runs
offline and free (Anthropic has no embeddings API). A custom embedding_function
can be injected — tests use a deterministic one to stay fully offline.
"""
from __future__ import annotations

from pathlib import Path

import chromadb

from skai.models import Chunk, RetrievedChunk


class Store:
    def __init__(
        self,
        persist_dir: str = "./.chroma",
        collection: str = "knowledge",
        embedding_function=None,
    ):
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._name = collection
        self._ef = embedding_function
        self._col = self._get_or_create()

    def _get_or_create(self):
        kwargs = {"name": self._name}
        if self._ef is not None:
            kwargs["embedding_function"] = self._ef
        return self._client.get_or_create_collection(**kwargs)

    def add(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        ids, docs, metas = [], [], []
        for c in chunks:
            sid = c.metadata.get("source_id", "doc")
            idx = c.metadata.get("chunk_index", 0)
            ids.append(f"{sid}::{idx}")
            docs.append(c.text)
            metas.append(c.metadata)
        # upsert => re-ingesting the same source overwrites rather than duplicates
        self._col.upsert(ids=ids, documents=docs, metadatas=metas)
        return len(chunks)

    def query(
        self, text: str, k: int = 5, source_type: str | None = None
    ) -> list[RetrievedChunk]:
        where = {"source_type": source_type} if source_type else None
        res = self._col.query(query_texts=[text], n_results=k, where=where)
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        out: list[RetrievedChunk] = []
        for doc, meta, dist in zip(docs, metas, dists):
            out.append(
                RetrievedChunk(text=doc, metadata=meta or {}, score=1.0 / (1.0 + dist))
            )
        return out

    def count(self) -> int:
        return self._col.count()

    def reset(self) -> None:
        self._client.delete_collection(self._name)
        self._col = self._get_or_create()
