"""Split Documents into overlapping, retrieval-sized Chunks."""
from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from skai.models import Chunk, Document


def chunk_documents(
    docs: list[Document], *, size: int = 800, overlap: int = 120
) -> list[Chunk]:
    """Recursively split each Document; carry metadata + a per-document chunk_index."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[Chunk] = []
    for doc in docs:
        pieces = splitter.split_text(doc.text)
        for i, piece in enumerate(pieces):
            meta = dict(doc.metadata)
            meta["chunk_index"] = i
            chunks.append(Chunk(text=piece, metadata=meta))
    return chunks
