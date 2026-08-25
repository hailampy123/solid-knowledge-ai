from conftest import StubLLM

from skai.agent import nodes, prompts
from skai.ingest.store import Store
from skai.models import Chunk, RetrievedChunk


def _seeded_store(tmp_path, ef):
    store = Store(str(tmp_path), "nodes_kb", embedding_function=ef)
    store.add(
        [
            Chunk(text="orcas hunt seals in coordinated pods", metadata={"source_type": "md", "source_id": "a", "chunk_index": 0}),
            Chunk(text="orca pods have distinct vocal dialects", metadata={"source_type": "md", "source_id": "c", "chunk_index": 0}),
        ]
    )
    return store


def test_route_classifies_kb_and_chitchat():
    llm = StubLLM(lambda s, u: "kb" if "orca" in u.lower() else "chitchat")
    assert nodes.route({"question": "what do orcas eat?"}, llm=llm)["route"] == "kb"
    assert nodes.route({"question": "hi there"}, llm=llm)["route"] == "chitchat"


def test_retrieve_fills_docs(tmp_path, ef):
    store = _seeded_store(tmp_path, ef)
    out = nodes.retrieve({"question": "orca diet"}, store=store, top_k=2)
    assert out["docs"]
    assert all(isinstance(d, RetrievedChunk) for d in out["docs"])


def test_grade_relevant():
    llm = StubLLM(lambda s, u: "RELEVANT")
    state = {"question": "q", "docs": [RetrievedChunk("t", {"source_id": "a"}, 0.9)], "retries": 0}
    out = nodes.grade_docs(state, llm=llm, max_retries=2)
    assert out["docs_ok"] is True
    assert out["action"] == "generate"


def test_grade_irrelevant_rewrites_and_retries():
    llm = StubLLM(lambda s, u: "IRRELEVANT\nbetter orca query")
    state = {"question": "vague", "docs": [RetrievedChunk("t", {"source_id": "a"}, 0.1)], "retries": 0}
    out = nodes.grade_docs(state, llm=llm, max_retries=2)
    assert out["docs_ok"] is False
    assert out["action"] == "retry"
    assert out["question"] == "better orca query"
    assert out["retries"] == 1


def test_grade_exhausted_gives_up_to_generate():
    llm = StubLLM(lambda s, u: "IRRELEVANT\nx")
    state = {"question": "q", "docs": [RetrievedChunk("t", {"source_id": "a"}, 0.1)], "retries": 2}
    out = nodes.grade_docs(state, llm=llm, max_retries=2)
    assert out["action"] == "generate"  # no more retries


def test_generate_cites_sources():
    llm = StubLLM(lambda s, u: "Orcas hunt seals in pods [a].")
    docs = [RetrievedChunk("orcas hunt seals", {"source_id": "a"}, 0.9),
            RetrievedChunk("dialects", {"source_id": "c"}, 0.5)]
    out = nodes.generate({"question": "q", "route": "kb", "docs": docs}, llm=llm)
    assert "Orcas hunt" in out["answer"]
    assert out["citations"] == ["a"]  # only the cited source


def test_generate_out_of_scope_declines_without_llm():
    called = []
    llm = StubLLM(lambda s, u: called.append(1) or "should not be called")
    out = nodes.generate({"question": "capital of mars?", "route": "out_of_scope"}, llm=llm)
    assert out["answer"] == prompts.OUT_OF_SCOPE_MSG
    assert called == []


def test_self_check_grounded_ends():
    llm = StubLLM(lambda s, u: "GROUNDED")
    out = nodes.self_check({"docs": [], "answer": "x", "retries": 0}, llm=llm, max_retries=2)
    assert out["grounded"] is True
    assert out["action"] == "end"


def test_self_check_ungrounded_exhausted_hedges():
    llm = StubLLM(lambda s, u: "UNGROUNDED")
    out = nodes.self_check({"docs": [], "answer": "hallucinated", "retries": 2}, llm=llm, max_retries=2)
    assert out["action"] == "end"
    assert out["answer"] == prompts.HEDGE
    assert out["citations"] == []
