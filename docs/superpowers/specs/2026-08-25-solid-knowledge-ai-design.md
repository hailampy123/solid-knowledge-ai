# Solid Knowledge AI — Design Spec

**Date:** 2026-08-25
**Status:** Approved for planning
**Purpose:** A thin-slice, genuinely-working multi-source document knowledge assistant that showcases agentic-development experience (LangGraph, LiteLLM, ChromaDB, MCP, Langfuse, DeepEval) for an interview.

---

## 1. Goal & Non-Goals

### Goal
Ingest three heterogeneous document sources (PDF, Markdown, web URLs) into one vector store, then answer questions through a **self-reflective LangGraph agent** that grades its own retrieval and verifies its own answer for grounding. Every run is traced (Langfuse) and the pipeline is tested for quality (DeepEval). The same core is exposed over MCP.

The slice must **actually run** end-to-end with only an `ANTHROPIC_API_KEY`.

### Non-Goals (deliberately cut — YAGNI)
- Multi-agent supervisor topology (documented as a scale-up, not built).
- Semantic long-term memory (conversational thread memory only).
- Auth, multi-tenancy, a web UI, or a hosted deployment.
- Incremental/real-time re-ingestion (batch ingest is fine).
- Provider embeddings (local embeddings only — see decision D3).

### Success Criteria
1. `skai ingest` builds a persistent Chroma store from `data/` (PDF + MD) and `data/urls.txt` (web).
2. `skai ask "..."` returns a grounded, cited answer; `skai chat` keeps multi-turn memory.
3. On an out-of-scope question the agent declines instead of hallucinating.
4. Langfuse (when keys present) shows one trace with a span per graph node + LLM call.
5. `deepeval test run evals/` reports faithfulness / relevancy / hallucination metrics.
6. `skai mcp` serves `search_kb` and `ask` as MCP tools.
7. `pytest tests/` passes offline (LLM stubbed) — no network, no keys.

---

## 2. Architecture

### 2.1 Data flow (ingestion)
```
source adapters ──► Document{text, metadata} ──► source-aware chunk ──► Chroma (persistent, local MiniLM embeddings)
   pdf|md|web
```
- **loaders.py** — three adapters, each yielding a common `Document` shape. `metadata = {source_type, source_id, title, uri?}`.
  - PDF: `pypdf`, page-aware; `source_id = filename#page`.
  - Markdown: read file, keep heading context; `source_id = filepath`.
  - Web: `httpx` GET → `trafilatura` extract main article text; `source_id = url`.
- **chunk.py** — `RecursiveCharacterTextSplitter` (token-ish, ~800 chars / 120 overlap). Markdown may split on headers first, then recursive. Emits `Chunk` with `chunk_index` added to metadata.
- **store.py** — `chromadb.PersistentClient` at `./.chroma`. Collection `knowledge`. Uses Chroma's default embedding function (local MiniLM `all-MiniLM-L6-v2`). `add(chunks)` and `query(text, k, source_filter?)`.

### 2.2 Agent (LangGraph `StateGraph`)
```
AgentState = {
  question: str
  messages: list            # conversation history (checkpointed)
  route: Literal["kb","chitchat","out_of_scope"]
  docs: list[RetrievedChunk]
  answer: str
  citations: list[str]
  critique: str | None
  retries: int
}
```

Nodes (in `agent/nodes.py`):
1. **route** — LLM classifies the question → `kb | chitchat | out_of_scope`. Conditional edge: `kb`→retrieve, else→generate (direct short reply / polite decline).
2. **retrieve** — `store.query(question, k=5, source_filter?)` → `docs`.
3. **grade_docs** — LLM scores retrieved-doc relevance. If below threshold and `retries < 2`: rewrite query, increment `retries`, loop back to retrieve. Else proceed. *(Corrective-RAG loop.)*
4. **generate** — Claude synthesizes an answer **with inline citations** `[source_id]` drawn only from `docs`.
5. **self_check** — LLM verifies the answer is entailed by `docs`. If ungrounded and `retries < 2`: loop back (retry generate/retrieve). Else: hedge honestly ("I couldn't find this in the knowledge base") rather than assert.

Edges are conditional functions in `graph.py`. **Checkpointer:** `SqliteSaver` at `./.skai/memory.sqlite`, keyed by `thread_id` → multi-turn memory. `ask` uses an ephemeral thread; `chat` uses a stable thread per session.

### 2.3 LLM & observability
- **agent/llm.py** — `ChatLiteLLM` factory. Default model `anthropic/claude-...` (configurable via `config.py`). Centralizes temperature, retries, and callback wiring.
- **observability.py** — returns a Langfuse `CallbackHandler` when `LANGFUSE_*` env is set, else `None` (graph runs with no callbacks → clean no-op). Handler passed via `config={"callbacks": [...]}` on graph invoke so every node + LLM call is a span.

### 2.4 MCP
- **mcp_server.py** — `FastMCP` server exposing:
  - `search_kb(query: str, source_type?: str) -> list[chunk]` — thin wrapper over `store.query`.
  - `ask(question: str) -> str` — runs the full agent, returns cited answer.
  Reuses the exact core functions (no logic duplication).

### 2.5 CLI (`cli.py`, Typer)
- `skai ingest [--path data/ --urls data/urls.txt --reset]`
- `skai ask "question" [--source pdf|md|web]`
- `skai chat` — REPL with memory
- `skai mcp` — run MCP server (stdio)
- `skai eval` — thin passthrough to `deepeval test run evals/`

---

## 3. Module boundaries (isolation & clarity)

| Module | Does | Depends on |
|---|---|---|
| `config.py` | Load settings from `.env` | pydantic-settings |
| `models.py` | `Document`, `Chunk`, `RetrievedChunk`, `AgentState` | — |
| `ingest/loaders.py` | source → `Document` | pypdf, httpx, trafilatura |
| `ingest/chunk.py` | `Document` → `Chunk[]` | langchain-text-splitters |
| `ingest/store.py` | persist/query vectors | chromadb |
| `agent/llm.py` | LLM client + callbacks | litellm, langchain-litellm |
| `agent/nodes.py` | one function per node | store, llm, prompts |
| `agent/graph.py` | wire nodes + edges + checkpointer | langgraph, nodes |
| `observability.py` | Langfuse handler (or None) | langfuse |
| `mcp_server.py` | MCP tool surface | mcp, store, graph |
| `cli.py` | user entrypoints | typer, everything |

Each unit is testable alone: loaders/chunk/store are deterministic and offline; nodes take an injectable LLM so the graph runs with a stub.

---

## 4. Tech Decisions & Validation

| ID | Layer | Choice | Alternatives rejected | Rationale |
|---|---|---|---|---|
| D1 | Orchestration | **LangGraph** | LangChain AgentExecutor, hand-rolled loop | Explicit graph = inspectable state, conditional retry edges, checkpointed memory. AgentExecutor hides control flow, making the corrective loop hard to reason about. |
| D2 | LLM gateway | **LiteLLM** via `ChatLiteLLM` | Anthropic SDK direct | Single interface, provider swap by config, built-in retry/fallback. LangChain wrapper gives clean Langfuse callback integration. |
| D3 | Embeddings | **Chroma built-in local MiniLM** | OpenAI/Voyage embeddings | Anthropic exposes no embeddings API. Decoupling embeddings (local, free, offline) from generation (Claude) keeps the demo runnable with one key and is a deliberate, defensible separation. |
| D4 | Vector store | **ChromaDB (persistent)** | FAISS, pgvector, Pinecone | Zero infra, metadata filtering (needed for `source_type`), disk persistence. FAISS lacks metadata; pgvector/Pinecone are infra overkill for a slice. |
| D5 | Web extraction | **trafilatura** | BeautifulSoup, readability-lxml | Purpose-built article extraction; removes nav/boilerplate far better than hand-rolled parsing. |
| D6 | Observability | **Langfuse** callback handler | LangSmith | OSS/self-hostable; one handler traces every node + LLM call; no-ops cleanly without keys. |
| D7 | Eval | **DeepEval** (pytest) | Ragas, manual eval | RAG-triad metrics (faithfulness, answer/context relevancy, hallucination) run as pytest → CI-friendly; directly answers "how do you know it works." |
| D8 | Tool exposure | **MCP** (FastMCP) | REST API | Surfaces the same core as MCP tools usable from Claude Desktop/IDE; high-signal interview topic, ~1 file. |
| D9 | Runtime/pkg | **Python 3.11 + uv**, Typer CLI | 3.9 system Python, pip | 3.9 too old for current LangGraph; `uv` is fast + modern; Typer gives a clean multi-command CLI. |
| D10 | Agent pattern | **Self-reflective / corrective RAG** | Linear RAG, multi-agent supervisor | Linear shows no agency; multi-agent adds failure surface without helping a slice. Corrective RAG is one loop that demonstrates real agency and stays runnable. |

---

## 5. Error handling
- **Ingestion:** a failing source (bad PDF, dead URL) logs a warning and is skipped; ingest continues. Report counts at the end.
- **Empty retrieval / low grade after retries:** agent declines honestly, no fabrication.
- **LLM/API errors:** LiteLLM retries; on exhaustion, surface a clear error, don't crash the REPL.
- **Missing keys:** Langfuse absent → no-op tracing. Anthropic key absent → clear startup error naming the env var.

---

## 6. Testing strategy
- **Offline unit tests** (`tests/`, `pytest`, no network/keys):
  - `test_loaders.py` — each adapter produces correct `Document` + metadata (web adapter fed a local HTML fixture, not a live URL).
  - `test_chunk.py` — chunk counts, overlap, metadata carried through.
  - `test_store.py` — add then query returns the seeded chunk (uses an ephemeral Chroma dir).
  - `test_graph_smoke.py` — full graph with a **stubbed LLM** (deterministic canned routes/answers) → verifies routing, retry loop bound, and citation assembly without any API call.
- **Quality eval** (`evals/`, needs `ANTHROPIC_API_KEY`): small golden set over the sample corpus; DeepEval faithfulness / answer-relevancy / contextual-relevancy / hallucination. Documented as a separate, keyed run.
- TDD during implementation.

---

## 7. Sample data (`data/`)
- `docs/*.md` — 1–2 short Markdown notes.
- `docs/*.pdf` — 1 small PDF (generated or checked in).
- `urls.txt` — 1–2 stable article URLs.
Content themed around one topic so cross-source questions are meaningful.

---

## 8. Scale-up notes (talking points, not built)
- Multi-agent supervisor routing one agent per source/domain.
- Semantic long-term memory (vector-backed user memory) alongside the checkpointer.
- Provider embeddings + reranking (e.g. Cohere/Voyage rerank) for precision.
- Incremental ingestion with content hashing to avoid re-embedding.
- Serve the MCP/agent behind a small API + streaming.

---

## 9. Deliverables
1. Runnable package under `src/skai/` with the CLI above.
2. `README.md` — quickstart, architecture diagram, run commands.
3. `docs/DECISIONS.md` — the Section 4 table, expanded into prose (interview cheat-sheet).
4. Sample data + a working DeepEval suite.
5. Offline test suite that passes without keys.
