"""Typer CLI: ingest | ask | chat | mcp | eval."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console

from skai.agent.graph import answer_question, build_graph, make_checkpointer
from skai.agent.llm import make_llm
from skai.config import Settings, get_settings, resolve_model
from skai.ingest.chunk import chunk_documents
from skai.ingest.loaders import load_sources
from skai.ingest.store import Store
from skai.observability import flush, get_callbacks

app = typer.Typer(add_completion=False, help="Solid Knowledge AI — multi-source document assistant.")
console = Console()


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
    )
    return graph


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


_MODEL_HELP = "Model: haiku (default) | sonnet | any non-Opus LiteLLM id"


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
    out = answer_question(
        graph, question, thread_id=thread_id,
        callbacks=get_callbacks(settings), source_type=source,
    )
    _print_answer(out)
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
        out = answer_question(graph, q, thread_id="chat", callbacks=callbacks)
        _print_answer(out)
        flush(settings)


@app.command()
def mcp():
    """Run the MCP server (stdio) exposing search_kb and ask."""
    from skai.mcp_server import serve

    serve()


@app.command()
def eval():
    """Run the DeepEval quality suite (needs `uv sync --group eval` + a key)."""
    try:
        raise SystemExit(subprocess.call(["deepeval", "test", "run", "evals"]))
    except FileNotFoundError:
        console.print("[red]deepeval not installed.[/red] Run: uv sync --group eval")
        raise typer.Exit(code=1)


def _print_answer(out: dict) -> None:
    console.print(f"\n[bold green]answer[/bold green]: {out['answer']}")
    if out.get("citations"):
        console.print(f"[dim]sources: {', '.join(out['citations'])}[/dim]")
    console.print(f"[dim]route: {out.get('route')}[/dim]")


if __name__ == "__main__":
    app()
