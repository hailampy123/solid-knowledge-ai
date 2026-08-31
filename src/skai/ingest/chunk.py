"""Split Documents into overlapping, retrieval-sized Chunks."""
from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from skai import guard
from skai.models import Chunk, Document


def chunk_documents(
    docs: list[Document],
    *,
    size: int = 800,
    overlap: int = 120,
    redact_pii: bool = True,
    scan_injection: bool = True,
) -> list[Chunk]:
    """Recursively split each Document; carry metadata + a per-document chunk_index.

    Guardrails applied here (the one point every ingest path routes through):
    - redact_pii: strip PII/secrets so the vector store never holds them.
    - scan_injection: flag chunks whose text looks like a prompt-injection payload
      with metadata["quarantined"]=True; they are stored (audit) but excluded from
      retrieval in `nodes.retrieve`.
    """
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
            if scan_injection:
                flags = guard.scan_injection(piece)
                if flags:
                    meta["quarantined"] = True
                    meta["injection_flags"] = ",".join(flags)
            if redact_pii:
                piece, kinds = guard.redact_pii(piece)
                if kinds:
                    meta["pii_redacted"] = ",".join(kinds)
            chunks.append(Chunk(text=piece, metadata=meta))
    return chunks
