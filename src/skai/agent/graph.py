"""Wire the nodes into a self-reflective corrective-RAG StateGraph.

    route ─kb─▶ retrieve ─▶ grade_docs ─retry─▶ retrieve
      │                         │
   other                    generate ─kb─▶ self_check ─retry─▶ retrieve
      └────────▶ generate ─▶ END          └── end ──▶ END

A shared `retries` budget (max_retries) bounds both correction loops.
"""
from __future__ import annotations

from functools import partial
from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from skai.agent import nodes
from skai.models import AgentState


def build_graph(store, llm, *, top_k: int = 5, max_retries: int = 2, checkpointer=None):
    g = StateGraph(AgentState)

    g.add_node("route", partial(nodes.route, llm=llm))
    g.add_node("retrieve", partial(nodes.retrieve, kb=store, top_k=top_k))
    g.add_node("grade_docs", partial(nodes.grade_docs, llm=llm, max_retries=max_retries))
    g.add_node("generate", partial(nodes.generate, llm=llm))
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
    g.add_conditional_edges(
        "self_check",
        lambda s: "retrieve" if s.get("action") == "retry" else END,
        {"retrieve": "retrieve", END: END},
    )

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
