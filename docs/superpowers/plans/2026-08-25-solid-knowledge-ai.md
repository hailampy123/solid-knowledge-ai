# Solid Knowledge AI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A runnable multi-source (PDF/Markdown/web) knowledge assistant driven by a self-reflective LangGraph agent, with LiteLLM→Claude generation, Chroma retrieval, Langfuse tracing, DeepEval quality tests, and an MCP tool surface.

**Architecture:** Ingestion adapters normalize three source types into a common `Document`, chunk them, and persist to a local Chroma store with local MiniLM embeddings. A LangGraph `StateGraph` (route → retrieve → grade_docs → generate → self_check, with bounded corrective retries) answers questions with citations and SQLite-checkpointed memory. LLM access is injectable so the graph runs offline with a stub in tests.

**Tech Stack:** Python 3.11, uv, LangGraph, langchain-litellm/LiteLLM, ChromaDB, trafilatura, pypdf, Typer, Langfuse, DeepEval, MCP (FastMCP), pytest.

**Spec:** `docs/superpowers/specs/2026-08-25-solid-knowledge-ai-design.md`

## Global Constraints
- Python 3.11+ (LangGraph requires ≥3.10). Package manager: `uv`.
- Default LLM: `anthropic/claude-3-5-sonnet-latest` via `ChatLiteLLM`, model configurable in `config.py`.
- Embeddings: Chroma default local `all-MiniLM-L6-v2` — no provider embeddings.
- `tests/` MUST pass offline: no network, no API keys. LLM is stubbed/injected.
- Missing `LANGFUSE_*` → tracing no-ops. Missing `ANTHROPIC_API_KEY` → clear error only when a real LLM call is attempted.
- LLM must be dependency-injected into agent nodes so tests never call an API.

---

### Task 1: Project scaffold & config
**Files:** Create `pyproject.toml`, `.env.example`, `.gitignore`, `src/skai/__init__.py`, `src/skai/config.py`, `tests/test_config.py`

**Interfaces:**
- Produces: `Settings` (pydantic-settings) with `anthropic_api_key: str|None`, `model: str`, `chroma_dir: str`, `collection: str`, `memory_db: str`, `langfuse_public_key/secret_key/host: str|None`; `get_settings() -> Settings`.

- [ ] Write `pyproject.toml` (deps: langgraph, langchain-litellm, litellm, chromadb, trafilatura, pypdf, typer, rich, pydantic-settings, langfuse, mcp; dev: pytest, deepeval). Console script `skai = "skai.cli:app"`.
- [ ] Write `test_config.py`: `get_settings()` reads env, defaults `model` to the Claude id, `chroma_dir` to `./.chroma`.
- [ ] Run test → fails (no config module).
- [ ] Implement `config.py`.
- [ ] `uv sync`; run test → passes.
- [ ] Commit.

### Task 2: Domain models
**Files:** Create `src/skai/models.py`, `tests/test_models.py`

**Interfaces:**
- Produces: `Document(text:str, metadata:dict)`, `Chunk(text:str, metadata:dict)`, `RetrievedChunk(text:str, metadata:dict, score:float)`, `AgentState` (TypedDict: question, messages, route, docs, answer, citations, critique, retries).

- [ ] Write `test_models.py`: construct each, metadata round-trips, `AgentState` keys present.
- [ ] Run → fails.
- [ ] Implement dataclasses + TypedDict.
- [ ] Run → passes. Commit.

### Task 3: Loaders (PDF / Markdown / Web)
**Files:** Create `src/skai/ingest/__init__.py`, `src/skai/ingest/loaders.py`, `tests/test_loaders.py`, `tests/fixtures/sample.md`, `tests/fixtures/article.html`

**Interfaces:**
- Consumes: `Document`.
- Produces: `load_markdown(path)->list[Document]`, `load_pdf(path)->list[Document]`, `load_web(url, *, _html=None)->list[Document]` (`_html` injects HTML so tests skip the network), `load_sources(doc_dir, urls)->list[Document]` (skips failures, logs).

- [ ] Write `test_loaders.py`: markdown fixture → 1 Document with `source_type="md"`, `source_id`=path; web via `_html` fixture → extracted text, `source_type="web"`, `uri`=url. (PDF tested in a generated-file test if pypdf can write; else assert graceful skip on missing file.)
- [ ] Run → fails.
- [ ] Implement adapters; `load_web` uses trafilatura on `_html` when provided, else httpx GET then trafilatura.
- [ ] Run → passes. Commit.

### Task 4: Chunking
**Files:** Create `src/skai/ingest/chunk.py`, `tests/test_chunk.py`

**Interfaces:**
- Consumes: `Document`, `Chunk`.
- Produces: `chunk_documents(docs, *, size=800, overlap=120)->list[Chunk]`, each with `chunk_index` in metadata, original metadata preserved.

- [ ] Write `test_chunk.py`: a 3000-char Document → >1 chunks, indices 0..n, metadata carried, overlap present.
- [ ] Run → fails.
- [ ] Implement with `RecursiveCharacterTextSplitter`.
- [ ] Run → passes. Commit.

### Task 5: Vector store (Chroma)
**Files:** Create `src/skai/ingest/store.py`, `tests/test_store.py`

**Interfaces:**
- Consumes: `Chunk`, `RetrievedChunk`, `Settings`.
- Produces: `Store(persist_dir, collection)` with `.add(chunks)`, `.query(text, k=5, source_type=None)->list[RetrievedChunk]`, `.reset()`, `.count()`.

- [ ] Write `test_store.py` (tmp dir): add 3 chunks, query returns most-relevant first; `source_type` filter restricts results; `count()` correct.
- [ ] Run → fails.
- [ ] Implement with `chromadb.PersistentClient`, default embedding fn, metadata `where` filter.
- [ ] Run → passes. Commit.

### Task 6: LLM factory + observability
**Files:** Create `src/skai/agent/__init__.py`, `src/skai/agent/llm.py`, `src/skai/observability.py`, `tests/test_observability.py`

**Interfaces:**
- Consumes: `Settings`.
- Produces: `make_llm(settings)->ChatLiteLLM`; `get_callbacks(settings)->list` (Langfuse handler or `[]`).

- [ ] Write `test_observability.py`: no Langfuse env → `get_callbacks()` returns `[]` (no crash).
- [ ] Run → fails.
- [ ] Implement; `make_llm` wires model + callbacks; Langfuse handler only built when keys present.
- [ ] Run → passes. Commit.

### Task 7: Agent nodes
**Files:** Create `src/skai/agent/prompts.py`, `src/skai/agent/nodes.py`, `tests/test_nodes.py`

**Interfaces:**
- Consumes: `AgentState`, `Store`, an injected `llm` (any object with `.invoke(messages)->msg.content`).
- Produces: node fns `route(state, llm)`, `retrieve(state, store)`, `grade_docs(state, llm)`, `generate(state, llm)`, `self_check(state, llm)`, each returning a partial state dict. Grading/self-check bounded by `retries<2`.

- [ ] Write `test_nodes.py` with a `StubLLM` returning canned outputs: `route` maps a KB question → `"kb"` and greeting → `"chitchat"`; `retrieve` fills `docs` from a seeded store; `generate` produces answer + citations from docs; `self_check` sets `critique`/keeps answer.
- [ ] Run → fails.
- [ ] Implement nodes + prompts.
- [ ] Run → passes. Commit.

### Task 8: Graph wiring + memory
**Files:** Create `src/skai/agent/graph.py`, `tests/test_graph_smoke.py`

**Interfaces:**
- Consumes: nodes, `Store`, injected `llm`, `SqliteSaver`.
- Produces: `build_graph(store, llm, checkpointer=None)->CompiledGraph`; `answer_question(graph, question, thread_id)->{answer,citations}`.

- [ ] Write `test_graph_smoke.py`: build graph with `StubLLM` + seeded tmp store; a KB question routes→retrieve→generate→self_check and returns a cited answer; an out-of-scope question declines; retry loop cannot exceed 2. No network.
- [ ] Run → fails.
- [ ] Implement `StateGraph`, conditional edges, `SqliteSaver` checkpointer.
- [ ] Run → passes. Commit.

### Task 9: CLI (Typer)
**Files:** Create `src/skai/cli.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: loaders, chunk, store, graph, make_llm.
- Produces: `app` with `ingest`, `ask`, `chat`, `mcp`, `eval` commands.

- [ ] Write `test_cli.py` (Typer `CliRunner`): `ingest` on a tmp docs dir populates the store (LLM not needed); `ask --help` lists options. Real `ask` (LLM) not run in tests.
- [ ] Run → fails.
- [ ] Implement CLI; `ask`/`chat` build real llm via `make_llm`; `eval` shells `deepeval test run evals`.
- [ ] Run → passes. Commit.

### Task 10: MCP server
**Files:** Create `src/skai/mcp_server.py`, `tests/test_mcp.py`

**Interfaces:**
- Consumes: `Store`, `build_graph`, `make_llm`.
- Produces: FastMCP `server` exposing `search_kb(query, source_type=None)` and `ask(question)`; `serve()` runs stdio.

- [ ] Write `test_mcp.py`: import server, assert both tools registered (introspect FastMCP), and `search_kb` core fn returns store hits (call the underlying function directly with a seeded store — no LLM).
- [ ] Run → fails.
- [ ] Implement `mcp_server.py` reusing core fns.
- [ ] Run → passes. Commit.

### Task 11: Sample data + DeepEval suite
**Files:** Create `data/docs/*.md`, `data/docs/*.pdf`, `data/urls.txt`, `evals/__init__.py`, `evals/dataset.py`, `evals/test_rag.py`

**Interfaces:**
- Consumes: full pipeline + real LLM.
- Produces: golden Q/context dataset; DeepEval faithfulness / answer-relevancy / contextual-relevancy / hallucination tests (skipped when `ANTHROPIC_API_KEY` unset).

- [ ] Add themed sample data (one topic across md/pdf/urls); generate the PDF with a tiny script.
- [ ] Write `evals/dataset.py` (golden set) and `evals/test_rag.py` with `@pytest.mark.skipif(no key)`.
- [ ] Run `pytest tests/` (offline) → all pass. Commit.

### Task 12: Docs
**Files:** Create `README.md`, `docs/DECISIONS.md`

- [ ] `README.md`: quickstart (`uv sync`, `.env`, `skai ingest`, `skai ask`, `skai chat`, `skai mcp`, `skai eval`), architecture diagram (ASCII), command table.
- [ ] `docs/DECISIONS.md`: expand spec §4 table into prose (interview cheat-sheet) + the scale-up notes.
- [ ] Commit.

## Self-Review
- **Spec coverage:** ingest (T3–5), agent+memory (T7–8), llm/observability (T6), MCP (T10), eval (T11), CLI (T9), docs (T12), config/models (T1–2). All spec sections covered.
- **Placeholder scan:** none — every task has concrete files/interfaces/tests.
- **Type consistency:** `Document/Chunk/RetrievedChunk/AgentState` defined T2, used consistently; `Store`, `make_llm`, `build_graph`, `answer_question` names stable across T5–10.
