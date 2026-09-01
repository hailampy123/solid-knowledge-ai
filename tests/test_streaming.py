"""astream_answer event stream, driven by a stubbed LLM (no network, no keys).

Token-by-token streaming needs a real streaming model, so with StubLLM no
`token` events fire; the wrapper's status/final/correction logic is what we
assert here. The final answer falls back to the generate node's state patch.
"""
import asyncio

from conftest import StubLLM

from skai import feedback
from skai.agent import prompts
from skai.agent.graph import CORRECTION_BANNER, astream_answer, build_graph
from skai.ingest.store import Store
from skai.models import Chunk


def _store(tmp_path, ef):
    store = Store(str(tmp_path / "chroma"), "stream_kb", embedding_function=ef)
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
        return answer
    return r


def _collect(agen):
    async def run():
        return [ev async for ev in agen]
    return asyncio.run(run())


def test_stream_emits_status_and_final(tmp_path, ef):
    graph = build_graph(_store(tmp_path, ef), StubLLM(_responder()), max_retries=2)
    events = _collect(astream_answer(graph, "what do orcas eat?"))

    statuses = [e["node"] for e in events if e["type"] == "status"]
    assert statuses[:2] == ["route", "retrieve"]         # emitted in execution order
    assert {"grade_docs", "generate", "self_check"} <= set(statuses)

    final = events[-1]
    assert final["type"] == "final"
    assert "Orcas hunt" in final["answer"]
    assert final["citations"] == ["a"]
    assert final["route"] == "kb"
    assert final["grounded"] is True
    assert not any(e["type"] == "correction" for e in events)


def test_stream_appends_correction_banner_when_ungrounded(tmp_path, ef):
    # grader rejects and verifier says ungrounded -> both agree -> banner, but the
    # streamed answer is still shown (post-hoc correction, not a pre-emptive hedge).
    responder = _responder(grade="IRRELEVANT\nretry", check="UNGROUNDED")
    graph = build_graph(_store(tmp_path, ef), StubLLM(responder), max_retries=1)
    events = _collect(astream_answer(graph, "ambiguous"))

    banners = [e for e in events if e["type"] == "correction"]
    assert banners and banners[0]["text"] == CORRECTION_BANNER
    final = events[-1]
    assert final["grounded"] is False
    assert final["answer"]  # the generated answer is preserved, not replaced


def test_stream_logs_gap_only_when_ungrounded(tmp_path, ef):
    db = str(tmp_path / "feedback.sqlite")
    # grounded turn: no gap logged
    ok_graph = build_graph(_store(tmp_path, ef), StubLLM(_responder()), max_retries=2)
    _collect(astream_answer(ok_graph, "what do orcas eat?", gap_log=db))
    assert feedback.gap_report(db) == []

    # ungrounded turn: one gap logged, keyed to the original question
    bad = _responder(grade="IRRELEVANT\nretry", check="UNGROUNDED")
    bad_graph = build_graph(_store(tmp_path, ef), StubLLM(bad), max_retries=1)
    _collect(astream_answer(bad_graph, "ambiguous", gap_log=db))
    report = feedback.gap_report(db)
    assert len(report) == 1 and report[0]["question"] == "ambiguous"
    assert "ungrounded/hedged" in report[0]["reasons"]


def test_stream_out_of_scope_has_no_correction(tmp_path, ef):
    graph = build_graph(_store(tmp_path, ef), StubLLM(_responder(route="out_of_scope")))
    events = _collect(astream_answer(graph, "capital of Mars?"))
    final = events[-1]
    assert final["answer"] == prompts.OUT_OF_SCOPE_MSG
    assert final["grounded"] is True
    assert not any(e["type"] == "correction" for e in events)
