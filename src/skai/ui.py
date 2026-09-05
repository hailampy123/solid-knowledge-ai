"""Gradio UI for the knowledge agent.

Features: chat with per-session memory, source/route/model transparency, 👍/👎
feedback (local + Langfuse score), example prompts, and live data enrichment
(upload a file or add a URL to grow the knowledge base mid-demo).

Run:  skai ui
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from uuid import uuid4

import gradio as gr

from skai import feedback
from skai.agent.graph import CORRECTION_BANNER, astream_answer, build_graph, make_checkpointer
from skai.agent.llm import make_llm
from skai.config import Settings, get_settings, resolve_model
from skai.ingest.chunk import chunk_documents
from skai.ingest.loaders import load_markdown, load_pdf, load_web
from skai.ingest.store import Store
from skai.observability import get_callbacks

EXAMPLES = [
    "What do orcas eat?",
    "How do orcas communicate?",
    "What threats do orcas face?",
]

# Offline-safe font list (no GoogleFont fetch) so the UI themes even without network.
THEME = gr.themes.Soft(
    primary_hue="blue", secondary_hue="cyan", neutral_hue="slate",
    font=["Inter", "system-ui", "sans-serif"],
)

CSS = """
.gradio-container {max-width: 1080px !important; margin: 0 auto !important;}
#skai-title h1 {margin-bottom: .25rem;}
#skai-title p {color: var(--body-text-color-subdued); margin-top: 0;}
footer {visibility: hidden;}
"""

WELCOME = (
    "### 🐋 Ask me about orcas\n"
    "I answer from ingested PDF, Markdown, and web sources — with citations — "
    "and I say so honestly when I can't ground an answer.\n\n"
    "Pick an example below, or grow my knowledge base from the panel on the right."
)


@lru_cache
def _store() -> Store:
    s = get_settings()
    return Store(s.chroma_dir, s.collection)


@lru_cache
def _graph(model: str):
    """One compiled graph per model; all share the single persistent store."""
    s = get_settings().model_copy(update={"model": resolve_model(model)})
    return build_graph(
        _store(),
        make_llm(s, callbacks=[]),
        top_k=s.top_k,
        max_retries=s.max_retries,
        checkpointer=make_checkpointer(s.memory_db),
        pii_redaction=s.pii_redaction,
        refusal_topics=s.refusal_topics,
    )


def _stats_md() -> str:
    fb = feedback.stats(get_settings().feedback_db)
    return (
        f"**Knowledge base:** {_store().count()} chunks\n\n"
        f"**Feedback:** 👍 {fb['up']} · 👎 {fb['down']}"
    )


def _open_trace_span(settings: Settings):
    """Open a Langfuse span for this turn (best-effort). Returns (client, cm, span)
    or (None, None, None). The graph's callbacks already trace LLM calls; this span
    groups them and yields a trace_id so feedback can be scored against the turn."""
    if not settings.langfuse_enabled:
        return None, None, None
    try:
        from langfuse import get_client

        client = get_client()
        cm = client.start_as_current_span(name="skai-ui-turn")
        span = cm.__enter__()
        return client, cm, span
    except Exception:  # noqa: BLE001 - tracing must never break the chat
        return None, None, None


def _render(out: dict) -> str:
    parts = [out["answer"]]
    if out.get("citations"):
        parts.append("\n\n---\n**Sources:** " + ", ".join(f"`{c}`" for c in out["citations"]))
    parts.append(f"\n<sub>route: `{out.get('route')}` · model: `{out.get('model')}`</sub>")
    return "".join(parts)


# --- event handlers ----------------------------------------------------------

async def on_send(message, history, model, source_filter, thread_id):
    """Stream a turn into the chat: grow the assistant bubble as tokens arrive,
    show node status while waiting, append a correction banner if self_check
    (post-hoc) found the answer ungrounded."""
    message = (message or "").strip()
    if not message:
        yield history, "", None, gr.update(visible=False), ""
        return

    settings = get_settings().model_copy(update={"model": resolve_model(model)})
    resolved = resolve_model(model)
    graph = _graph(model)
    source = None if source_filter == "all" else source_filter
    callbacks = get_callbacks(settings)
    history = (history or []) + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": "_…_"},
    ]

    client, cm, span = _open_trace_span(settings)
    trace_id = getattr(span, "trace_id", None) if span is not None else None
    acc, citations, route, corrected = "", [], None, False
    try:
        async for ev in astream_answer(
            graph, message, thread_id=thread_id, callbacks=callbacks,
            source_type=source, gap_log=settings.feedback_db,
        ):
            kind = ev["type"]
            if kind == "status":
                if not acc:  # only show status before the first token
                    history[-1]["content"] = f"_{ev['label']}_"
                    yield history, "", None, gr.update(visible=False), ""
            elif kind == "token":
                acc += ev["text"]
                history[-1]["content"] = acc
                yield history, "", None, gr.update(visible=False), ""
            elif kind == "correction":
                corrected = True
            elif kind == "final":
                acc = acc or ev["answer"]
                citations = ev.get("citations", [])
                route = ev.get("route")
        if client is not None:
            try:
                client.update_current_trace(input=message, output=acc)
            except Exception:  # noqa: BLE001
                pass
    finally:
        if cm is not None:
            cm.__exit__(None, None, None)
        if client is not None:
            try:
                client.flush()
            except Exception:  # noqa: BLE001
                pass

    rendered = _render({"answer": acc, "citations": citations, "route": route, "model": resolved})
    if corrected:
        rendered = f"> {CORRECTION_BANNER}\n\n" + rendered
    history[-1]["content"] = rendered
    last = {
        "question": message, "answer": acc, "route": route,
        "model": resolved, "citations": citations, "trace_id": trace_id,
    }
    yield history, "", last, gr.update(visible=True), ""


def on_feedback(rating, comment, last):
    if not last:
        return "No response to rate yet."
    settings = get_settings()
    feedback.record(settings.feedback_db, rating=rating, comment=comment or "", **last)
    feedback.push_langfuse_score(settings, last.get("trace_id"), rating, comment or "")
    return f"Thanks — recorded 👍/👎 as **{rating}**."


def on_clear():
    # new thread id => fresh agent memory for the next conversation
    return [], str(uuid4()), None, gr.update(visible=False), ""


def on_ingest_file(file):
    if not file:
        return _stats_md(), "No file selected."
    path = Path(file.name if hasattr(file, "name") else file)
    ext = path.suffix.lower()
    try:
        if ext in {".md", ".markdown", ".txt"}:
            docs = load_markdown(path)
        elif ext == ".pdf":
            docs = load_pdf(path)
        else:
            return _stats_md(), f"Unsupported file type: {ext}"
        n = _store().add(chunk_documents(docs))
        return _stats_md(), f"Ingested **{path.name}** → {n} chunks."
    except Exception as e:  # noqa: BLE001
        return _stats_md(), f"Failed to ingest {path.name}: {e}"


def on_ingest_url(url):
    url = (url or "").strip()
    if not url:
        return _stats_md(), "", "Enter a URL first."
    try:
        n = _store().add(chunk_documents(load_web(url)))
        return _stats_md(), "", f"Ingested **{url}** → {n} chunks."
    except Exception as e:  # noqa: BLE001
        return _stats_md(), url, f"Failed to ingest {url}: {e}"


# --- layout ------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Solid Knowledge AI", fill_height=True) as demo:  # theme/css -> launch() in Gradio 6
        gr.Markdown(
            "# 🐋 Solid Knowledge AI\n"
            "A self-reflective knowledge agent over your PDF · Markdown · web sources.",
            elem_id="skai-title",
        )
        thread = gr.State(str(uuid4()))
        last = gr.State(None)

        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    height=520,
                    show_label=False,
                    render_markdown=True,
                    resizable=True,
                    placeholder=WELCOME,
                )
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Ask about orcas…", show_label=False,
                        autofocus=True, scale=8, container=False,
                    )
                    send = gr.Button("Send", variant="primary", scale=1, min_width=90)
                gr.Examples(EXAMPLES, inputs=msg, label="Try an example")
                clear = gr.Button("🗑 Clear conversation", variant="secondary", size="sm")

                with gr.Row(visible=False) as fb_row:
                    up = gr.Button("👍 Helpful", size="sm", variant="secondary")
                    down = gr.Button("👎 Not helpful", size="sm", variant="secondary")
                    comment = gr.Textbox(
                        placeholder="optional comment", show_label=False, scale=3, container=False,
                    )
                fb_status = gr.Markdown("")

            with gr.Column(scale=1):
                stats = gr.Markdown(_stats_md)
                with gr.Accordion("⚙️ Model & filters", open=True):
                    model = gr.Dropdown(
                        ["haiku", "sonnet", "gemini-flash", "gemini-pro", "gpt-4o", "gpt-4o-mini"],
                        value="haiku", label="Model", allow_custom_value=True,
                    )
                    source_filter = gr.Dropdown(
                        ["all", "pdf", "md", "web"], value="all", label="Source filter",
                    )
                with gr.Accordion("📚 Grow the knowledge base", open=False):
                    file = gr.File(label="Upload .md / .txt / .pdf", file_types=[".md", ".txt", ".pdf"])
                    add_file = gr.Button("Ingest file", size="sm")
                    url = gr.Textbox(label="…or add a URL")
                    add_url = gr.Button("Ingest URL", size="sm")
                    ingest_status = gr.Markdown("")

        send_args = dict(
            fn=on_send,
            inputs=[msg, chatbot, model, source_filter, thread],
            outputs=[chatbot, msg, last, fb_row, fb_status],
        )
        send.click(**send_args)
        msg.submit(**send_args)

        # record → clear the comment box → refresh the sidebar stats
        up.click(lambda c, l: on_feedback("up", c, l), [comment, last], fb_status).then(
            lambda: "", None, comment
        ).then(_stats_md, None, stats)
        down.click(lambda c, l: on_feedback("down", c, l), [comment, last], fb_status).then(
            lambda: "", None, comment
        ).then(_stats_md, None, stats)

        clear.click(on_clear, None, [chatbot, thread, last, fb_row, fb_status])
        add_file.click(on_ingest_file, file, [stats, ingest_status])
        add_url.click(on_ingest_url, url, [stats, url, ingest_status])

    return demo


def launch(**kwargs) -> None:
    # Gradio 6: theme/css are launch() args, not Blocks() args.
    kwargs.setdefault("theme", THEME)
    kwargs.setdefault("css", CSS)
    build_ui().launch(**kwargs)


if __name__ == "__main__":
    launch()
