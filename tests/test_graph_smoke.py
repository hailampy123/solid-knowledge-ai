"""Full-graph tests with a stubbed LLM: no network, no keys."""
from conftest import StubLLM

from skai.agent import prompts
from skai.agent.graph import answer_question, build_graph, make_checkpointer
from skai.ingest.store import Store
from skai.models import Chunk


def _store(tmp_path, ef):
    store = Store(str(tmp_path / "chroma"), "graph_kb", embedding_function=ef)
    store.add(
        [
            Chunk(text="orcas hunt seals in coordinated pods", metadata={"source_type": "md", "source_id": "a", "chunk_index": 0}),
            Chunk(text="orca pods have distinct vocal dialects", metadata={"source_type": "md", "source_id": "c", "chunk_index": 0}),
        ]
    )
    return store


def _responder(route="kb", grade="RELEVANT", answer="Orcas hunt seals [a].", check="GROUNDED"):
    def r(system, user):
        if system == prompts.ROUTE_SYSTEM:
            return route
        if system == prompts.GRADE_SYSTEM:
            return grade
        if system == prompts.SELFCHECK_SYSTEM:
            return check
        return answer  # GENERATE / CHITCHAT
    return r


def test_happy_path_returns_cited_answer(tmp_path, ef):
    store = _store(tmp_path, ef)
    graph = build_graph(store, StubLLM(_responder()), max_retries=2)
    out = answer_question(graph, "what do orcas eat?")
    assert "Orcas hunt" in out["answer"]
    assert out["citations"] == ["a"]
    assert out["route"] == "kb"


def test_out_of_scope_declines(tmp_path, ef):
    store = _store(tmp_path, ef)
    graph = build_graph(store, StubLLM(_responder(route="out_of_scope")), max_retries=2)
    out = answer_question(graph, "what's the capital of Mars?")
    assert out["answer"] == prompts.OUT_OF_SCOPE_MSG
    assert out["citations"] == []


def test_retry_loop_is_bounded_and_hedges(tmp_path, ef):
    store = _store(tmp_path, ef)
    # grader always rejects, self-check always fails -> must terminate, not loop forever
    responder = _responder(grade="IRRELEVANT\ntry again", check="UNGROUNDED")
    graph = build_graph(store, StubLLM(responder), max_retries=2)
    out = answer_question(graph, "ambiguous question")
    assert out["answer"] == prompts.HEDGE  # refused to assert ungrounded content


def test_memory_persists_across_turns(tmp_path, ef):
    store = _store(tmp_path, ef)
    saver = make_checkpointer(str(tmp_path / "mem.sqlite"))
    graph = build_graph(store, StubLLM(_responder()), checkpointer=saver)
    answer_question(graph, "first question", thread_id="t1")
    answer_question(graph, "second question", thread_id="t1")
    # both turns' messages accumulate in the checkpointed thread state
    state = graph.get_state({"configurable": {"thread_id": "t1"}})
    contents = [m.content for m in state.values["messages"]]
    assert "first question" in contents
    assert "second question" in contents
