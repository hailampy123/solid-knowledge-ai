"""Agent nodes. Each takes the state plus injected deps and returns a state patch.

The corrective loop uses a shared `retries` budget across grade_docs and
self_check; `action` tells the conditional edges where to go next.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from skai.agent import prompts
from skai.models import AgentState, RetrievedChunk


def _ask(llm, system: str, user: str) -> str:
    resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    return (resp.content or "").strip()


def _format_context(docs: list[RetrievedChunk]) -> str:
    return "\n\n".join(f"[{d.metadata.get('source_id', '?')}] {d.text}" for d in docs)


def route(state: AgentState, *, llm) -> dict:
    label = _ask(llm, prompts.ROUTE_SYSTEM, state["question"]).lower()
    if "out_of_scope" in label or "out of scope" in label:
        r = "out_of_scope"
    elif "chitchat" in label or "chit" in label:
        r = "chitchat"
    else:
        r = "kb"
    return {"route": r, "retries": state.get("retries", 0)}


def retrieve(state: AgentState, *, kb, top_k: int = 5) -> dict:
    # param is named `kb`, not `store`: LangGraph reserves `store` for its own
    # BaseStore injection and would overwrite a bound value.
    hits = kb.query(state["question"], k=top_k)
    return {"docs": hits}


def grade_docs(state: AgentState, *, llm, max_retries: int = 2) -> dict:
    docs = state.get("docs", [])
    retries = state.get("retries", 0)

    if not docs:
        relevant, rewritten = False, state["question"]
    else:
        text = _ask(
            llm,
            prompts.GRADE_SYSTEM,
            prompts.GRADE_USER.format(
                question=state["question"], context=_format_context(docs)
            ),
        )
        relevant, rewritten = _parse_grade(text, state["question"])

    if relevant:
        return {"docs_ok": True, "action": "generate"}
    if retries < max_retries:
        # rewrite the query and retry retrieval (corrective-RAG)
        return {
            "docs_ok": False,
            "action": "retry",
            "question": rewritten,
            "retries": retries + 1,
        }
    return {"docs_ok": False, "action": "generate"}  # exhausted -> generate will hedge


def generate(state: AgentState, *, llm) -> dict:
    r = state.get("route", "kb")
    if r == "out_of_scope":
        msg = prompts.OUT_OF_SCOPE_MSG
        return {"answer": msg, "citations": [], "messages": [AIMessage(content=msg)]}
    if r == "chitchat":
        msg = _ask(llm, prompts.CHITCHAT_SYSTEM, state["question"])
        return {"answer": msg, "citations": [], "messages": [AIMessage(content=msg)]}

    docs = state.get("docs", [])
    answer = _ask(
        llm,
        prompts.GENERATE_SYSTEM,
        prompts.GENERATE_USER.format(
            question=state["question"], context=_format_context(docs)
        ),
    )
    citations = _extract_citations(answer, docs)
    return {"answer": answer, "citations": citations, "messages": [AIMessage(content=answer)]}


def self_check(state: AgentState, *, llm, max_retries: int = 2) -> dict:
    docs = state.get("docs", [])
    retries = state.get("retries", 0)
    text = _ask(
        llm,
        prompts.SELFCHECK_SYSTEM,
        prompts.SELFCHECK_USER.format(
            context=_format_context(docs), answer=state.get("answer", "")
        ),
    )
    grounded = "UNGROUNDED" not in text.upper() and "GROUNDED" in text.upper()

    if grounded:
        return {"grounded": True, "action": "end"}
    if retries < max_retries:
        return {"grounded": False, "action": "retry", "retries": retries + 1}
    # exhausted: refuse to assert ungrounded content
    return {
        "grounded": False,
        "action": "end",
        "answer": prompts.HEDGE,
        "citations": [],
        "messages": [AIMessage(content=prompts.HEDGE)],
    }


# --- parsing helpers ---------------------------------------------------------

def _parse_grade(text: str, original_q: str) -> tuple[bool, str]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return True, original_q  # be permissive if the grader is silent
    if lines[0].upper().startswith("RELEVANT"):
        return True, original_q
    rewritten = lines[1] if len(lines) > 1 else original_q
    return False, rewritten


def _extract_citations(answer: str, docs: list[RetrievedChunk]) -> list[str]:
    sids = []
    for d in docs:
        sid = d.metadata.get("source_id")
        if sid and sid not in sids:
            sids.append(sid)
    used = [sid for sid in sids if f"[{sid}]" in answer]
    return used or sids  # fall back to all retrieved sources
