"""Typer CLI: ingest | ask | chat | mcp | eval."""
from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import typer
from rich.console import Console

from skai.agent.graph import astream_answer, build_graph, make_checkpointer
from skai.agent.llm import make_llm
from skai.config import Settings, get_settings, resolve_model
from skai.ingest.chunk import chunk_documents
from skai.ingest.loaders import load_sources
from skai.ingest.store import Store
from skai.observability import flush, get_callbacks

app = typer.Typer(add_completion=False, help="Solid Knowledge AI — multi-source document assistant.")
console = Console()
err_console = Console(stderr=True)  # node status goes here so stdout is just the answer


def _open_store(settings: Settings) -> Store:
    """Seam for tests to inject a deterministic embedding function."""
    return Store(settings.chroma_dir, settings.collection)


def _settings_with_model(model: str | None) -> Settings:
    """Settings with an optional per-invocation model override (alias resolved, Opus blocked)."""
    settings = get_settings()
    chosen = model or settings.model
    try:
        resolved = resolve_model(chosen)  # validates (blocks Opus) for both flag and .env
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=2)
    return settings.model_copy(update={"model": resolved})


def _read_urls(urls_file: str) -> list[str]:
    p = Path(urls_file)
    if not p.exists():
        return []
    return [
        ln.strip()
        for ln in p.read_text().splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]


def _build_agent(settings: Settings):
    store = _open_store(settings)
    llm = make_llm(settings, callbacks=[])  # tracing attached at graph invoke instead
    checkpointer = make_checkpointer(settings.memory_db)
    graph = build_graph(
        store, llm, top_k=settings.top_k, max_retries=settings.max_retries,
        checkpointer=checkpointer,
        pii_redaction=settings.pii_redaction, refusal_topics=settings.refusal_topics,
    )
    return graph


def _stream_answer(graph, question, thread_id, callbacks, source, gap_log=None) -> dict:
    """Stream a turn to the console: status on stderr, tokens on stdout as they
    arrive, then citations/route. Returns the final event dict."""
    async def go() -> dict:
        streamed = False
        final: dict = {}
        async for ev in astream_answer(
            graph, question, thread_id=thread_id, callbacks=callbacks,
            source_type=source, gap_log=gap_log,
        ):
            kind = ev["type"]
            if kind == "status":
                err_console.print(f"[dim]{ev['label']}[/dim]")
            elif kind == "token":
                streamed = True
                console.print(ev["text"], end="", highlight=False)
            elif kind == "correction":
                console.print(f"\n[yellow]{ev['text']}[/yellow]")
            elif kind == "final":
                final = ev
        if streamed:
            console.print()  # newline after the streamed answer
        else:
            console.print(f"[bold green]answer[/bold green]: {final.get('answer', '')}")
        if final.get("citations"):
            console.print(f"[dim]sources: {', '.join(final['citations'])}[/dim]")
        console.print(f"[dim]route: {final.get('route')}[/dim]")
        return final

    return asyncio.run(go())


@app.command()
def ingest(
    path: str = typer.Option("data/docs", help="Folder of .md/.pdf/.txt files"),
    urls: str = typer.Option("data/urls.txt", help="File of URLs, one per line"),
    reset: bool = typer.Option(False, help="Wipe the collection first"),
):
    """Load sources, chunk, embed, and persist to Chroma."""
    settings = get_settings()
    store = _open_store(settings)
    if reset:
        store.reset()
        console.print("[yellow]collection reset[/yellow]")

    docs = load_sources(path, _read_urls(urls))
    chunks = chunk_documents(docs)
    n = store.add(chunks)
    by_type: dict[str, int] = {}
    for d in docs:
        by_type[d.metadata["source_type"]] = by_type.get(d.metadata["source_type"], 0) + 1
    console.print(
        f"[green]ingested[/green] {len(docs)} documents "
        f"({by_type}) -> {n} chunks; store now holds {store.count()} chunks"
    )


_MODEL_HELP = (
    "Model alias (haiku, sonnet, gemini-flash, gemini-pro, gpt-4o, gpt-4o-mini) "
    "or any non-Opus LiteLLM id (e.g. openai/gpt-4o, gemini/gemini-2.5-pro)"
)


@app.command()
def ask(
    question: str,
    source: str = typer.Option(None, help="Filter: pdf | md | web"),
    thread_id: str = typer.Option("cli", help="Conversation thread for memory"),
    model: str = typer.Option(None, "--model", "-m", help=_MODEL_HELP),
):
    """Ask a single question."""
    settings = _settings_with_model(model)
    graph = _build_agent(settings)
    _stream_answer(graph, question, thread_id, get_callbacks(settings), source, settings.feedback_db)
    flush(settings)  # ensure traces are sent before the process exits


@app.command()
def chat(model: str = typer.Option(None, "--model", "-m", help=_MODEL_HELP)):
    """Interactive multi-turn chat with memory."""
    settings = _settings_with_model(model)
    graph = _build_agent(settings)
    callbacks = get_callbacks(settings)
    console.print(f"[bold]skai chat[/bold] ({resolve_model(settings.model)}) — type 'exit' to quit.")
    while True:
        try:
            q = console.input("[cyan]you> [/cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"exit", "quit"}:
            break
        if not q:
            continue
        _stream_answer(graph, q, "chat", callbacks, None, settings.feedback_db)
        flush(settings)


@app.command()
def mcp():
    """Run the MCP server (stdio) exposing search_kb and ask."""
    from skai.mcp_server import serve

    serve()


@app.command()
def ui(
    port: int = typer.Option(7860, help="Port to serve the UI on"),
    share: bool = typer.Option(False, help="Create a public Gradio share link"),
):
    """Launch the Gradio chat UI (feedback, examples, live data ingestion)."""
    from skai.ui import launch

    launch(server_port=port, share=share)


feedback_app = typer.Typer(help="Close the feedback loop: gap backlog + grow the eval set.")
app.add_typer(feedback_app, name="feedback")


@feedback_app.command("report")
def feedback_report():
    """Content backlog: questions the agent couldn't ground, most frequent first."""
    from skai import feedback

    db = get_settings().feedback_db
    fb = feedback.stats(db)
    console.print(f"[bold]feedback[/bold]: 👍 {fb['up']} · 👎 {fb['down']} (total {fb['total']})")
    gaps = feedback.gap_report(db)
    if not gaps:
        console.print("[dim]no retrieval gaps logged yet[/dim]")
        return
    console.print("\n[bold]retrieval gaps (content backlog):[/bold]")
    for g in gaps:
        console.print(f"  [cyan]{g['count']:>3}×[/cyan]  {g['question']}  [dim]({', '.join(g['reasons'])})[/dim]")


@feedback_app.command("promote")
def feedback_promote(
    out: str = typer.Option("evals/golden.jsonl", help="Golden JSONL to append cases to"),
):
    """Promote reviewed 👎 (thumbs-down + comment) into the DeepEval golden set."""
    from skai import feedback

    n = feedback.promote_downvotes(get_settings().feedback_db, out)
    console.print(f"[green]promoted[/green] {n} thumbs-down case(s) into {out}")


@app.command()
def eval():
    """Run the DeepEval quality suite (needs `uv sync --group eval` + a key)."""
    try:
        raise SystemExit(subprocess.call(["deepeval", "test", "run", "evals"]))
    except FileNotFoundError:
        console.print("[red]deepeval not installed.[/red] Run: uv sync --group eval")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
