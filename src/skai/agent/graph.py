"""Wire the nodes into a self-reflective corrective-RAG StateGraph.

    route ─kb─▶ retrieve ─▶ grade_docs ─irrelevant(rewrite,retry)─▶ retrieve
      │                         │ relevant
   other                    generate ─kb─▶ self_check ─▶ END
      └────────▶ generate ─▶ END

Correction (query rewrite) happens in grade_docs, bounded by `max_retries`.
self_check is a terminal groundedness guard (see nodes.self_check).
"""
from __future__ import annotations

from functools import partial
from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from skai.agent import nodes
from skai.models import AgentState

# Human-readable status shown while each node runs (streaming surfaces).
_NODE_STATUS = {
    "route": "routing…",
    "retrieve": "retrieving…",
    "grade_docs": "checking sources…",
    "generate": "answering…",
    "self_check": "verifying…",
}
CORRECTION_BANNER = "⚠️ I couldn't fully verify this answer against the sources."


def build_graph(
    store,
    llm,
    *,
    top_k: int = 5,
    max_retries: int = 2,
    checkpointer=None,
    pii_redaction: bool = True,
    refusal_topics: list[str] | None = None,
):
    g = StateGraph(AgentState)

    g.add_node("route", partial(nodes.route, llm=llm, refusal_topics=refusal_topics))
    g.add_node("retrieve", partial(nodes.retrieve, kb=store, top_k=top_k))
    g.add_node("grade_docs", partial(nodes.grade_docs, llm=llm, max_retries=max_retries))
    g.add_node("generate", partial(nodes.generate, llm=llm, pii_redaction=pii_redaction))
    g.add_node("self_check", partial(nodes.self_check, llm=llm, max_retries=max_retries))

    g.add_edge(START, "route")
    g.add_conditional_edges(
        "route",
        lambda s: "retrieve" if s.get("route") == "kb" else "generate",
        {"retrieve": "retrieve", "generate": "generate"},
    )
    g.add_edge("retrieve", "grade_docs")
    g.add_conditional_edges(
        "grade_docs",
        lambda s: "retrieve" if s.get("action") == "retry" else "generate",
        {"retrieve": "retrieve", "generate": "generate"},
    )
    g.add_conditional_edges(
        "generate",
        lambda s: "self_check" if s.get("route") == "kb" else END,
        {"self_check": "self_check", END: END},
    )
    # self_check is terminal: re-querying the same question would return the same
    # docs, so a retry loop here can't help. Correction happens in grade_docs.
    g.add_edge("self_check", END)

    return g.compile(checkpointer=checkpointer)


def make_checkpointer(path: str):
    """A persistent SQLite checkpointer for multi-turn memory (thread_id keyed)."""
    import sqlite3

    from langgraph.checkpoint.sqlite import SqliteSaver

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return saver


def answer_question(
    graph,
    question: str,
    thread_id: str = "default",
    callbacks: list | None = None,
    source_type: str | None = None,
) -> dict:
    config: dict = {"configurable": {"thread_id": thread_id}}
    if callbacks:
        config["callbacks"] = callbacks
    result = graph.invoke(
        {
            "question": question,
            "messages": [HumanMessage(content=question)],
            "source_type": source_type,
            "retries": 0,
        },
        config=config,
    )
    return {
        "answer": result.get("answer", ""),
        "citations": result.get("citations", []),
        "route": result.get("route"),
    }


async def astream_answer(
    graph,
    question: str,
    thread_id: str = "default",
    callbacks: list | None = None,
    source_type: str | None = None,
):
    """Async streaming counterpart of `answer_question`. Yields event dicts:

        {"type": "status", "node", "label"}   once per node as it runs
        {"type": "token",  "text"}            generate-node tokens, live
        {"type": "correction", "text"}        only if self_check found it ungrounded
        {"type": "final", "answer", "citations", "route", "grounded"}

    Post-hoc self_check (roadmap §3.1): the generated answer streams first; the
    verifier's verdict becomes a correction banner instead of a pre-emptive block.
    So unlike `answer_question`, the streamed answer is shown even when ungrounded,
    with the banner appended — the honest-refusal signal is preserved, not hidden.
    """
    config: dict = {"configurable": {"thread_id": thread_id}}
    if callbacks:
        config["callbacks"] = callbacks
    inputs = {
        "question": question,
        "messages": [HumanMessage(content=question)],
        "source_type": source_type,
        "retries": 0,
    }

    gen: dict = {}       # the generate node's state patch (authoritative answer)
    verdict: dict = {}   # the self_check node's state patch (groundedness)
    route: str | None = None
    streamed = ""

    async for mode, chunk in graph.astream(
        inputs, config=config, stream_mode=["updates", "messages"]
    ):
        if mode == "updates":
            for node_name, patch in chunk.items():
                patch = patch or {}
                if node_name == "route":
                    route = patch.get("route", route)
                elif node_name == "generate":
                    gen = patch
                elif node_name == "self_check":
                    verdict = patch
                label = _NODE_STATUS.get(node_name)
                if label:
                    yield {"type": "status", "node": node_name, "label": label}
        elif mode == "messages":
            msg, meta = chunk
            if meta.get("langgraph_node") == "generate":
                text = getattr(msg, "content", "") or ""
                if text:
                    streamed += text
                    yield {"type": "token", "text": text}

    answer = streamed or gen.get("answer", "")
    citations = gen.get("citations", [])
    # self_check only ran on the kb path; elsewhere there is no verdict => grounded.
    grounded = verdict.get("grounded", True) if verdict else True
    if grounded is False:
        yield {"type": "correction", "text": CORRECTION_BANNER}
    yield {
        "type": "final",
        "answer": answer,
        "citations": citations,
        "route": route,
        "grounded": grounded,
    }
