from skai.ingest.chunk import chunk_documents
from skai.models import Document


def test_splits_long_document_and_indexes():
    text = "\n\n".join(f"Paragraph {i}. " + ("orca " * 40) for i in range(20))
    doc = Document(text=text, metadata={"source_type": "md", "source_id": "big.md"})
    chunks = chunk_documents([doc], size=300, overlap=50)

    assert len(chunks) > 1
    # per-document chunk_index is contiguous from 0
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))
    # original metadata preserved on every chunk
    assert all(c.metadata["source_id"] == "big.md" for c in chunks)


def test_short_document_single_chunk():
    doc = Document(text="tiny", metadata={"source_id": "s"})
    chunks = chunk_documents([doc], size=300, overlap=50)
    assert len(chunks) == 1
    assert chunks[0].metadata["chunk_index"] == 0


def test_overlap_shares_content():
    text = "abcdefghij " * 100
    doc = Document(text=text, metadata={"source_id": "o"})
    chunks = chunk_documents([doc], size=200, overlap=80)
    assert len(chunks) >= 2
    # consecutive chunks should share some overlapping tail/head content
    tail = chunks[0].text[-40:]
    assert any(word in chunks[1].text for word in tail.split())
