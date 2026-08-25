# Solid Knowledge AI

A multi-source document knowledge assistant driven by a **self-reflective LangGraph
agent**. It ingests **PDF + Markdown + web pages** into one vector store, then answers
questions through an agent that *grades its own retrieval and verifies its own answer
for grounding* — retrying with a rewritten query when either check fails, and refusing
to fabricate when it can't ground an answer. Traced with Langfuse, quality-tested with
DeepEval, and exposed over MCP.

Built to showcase agentic development: **LangGraph · LiteLLM · ChromaDB · MCP · Langfuse · DeepEval**.

## Why this is not "just RAG"

The agent is a **corrective / self-reflective RAG** loop, not a linear
`retrieve → generate` chain:

```
question
   │
   ▼
 route ──chitchat/out_of_scope──▶ generate ──▶ END
   │ kb
   ▼
retrieve ──▶ grade_docs ──irrelevant (rewrite query, retry)──▶ retrieve
                 │ relevant
                 ▼
             generate ──▶ self_check ──ungrounded (retry)──▶ retrieve
                              │ grounded / budget spent
                              ▼
                     answer + citations  (or an honest "I don't know")
```

- **route** — skips retrieval on small talk / out-of-scope questions.
- **grade_docs** — an LLM relevance gate; on failure it rewrites the query and retries.
- **self_check** — verifies the drafted answer is entailed by the retrieved context;
  if not, it retries or hedges instead of hallucinating.
- A shared **retry budget** (`max_retries`, default 2) bounds both loops.
- **Memory** — a SQLite checkpointer keeps multi-turn conversation state per `thread_id`.

## Quickstart

```bash
# 1. Install (Python 3.11+, uv)
uv sync

# 2. Configure — only ANTHROPIC_API_KEY is required
cp .env.example .env      # then edit .env

# 3. Ingest the sample corpus (2 Markdown + 1 PDF + 1 Wikipedia article)
uv run skai ingest        # -> builds ./.chroma  (local MiniLM embeddings, no API)

# 4. Ask (defaults to Haiku 4.5; switch per-call with --model)
uv run skai ask "What do orcas eat?"
uv run skai ask "How do orcas communicate?" --source md
uv run skai ask "Summarize orca threats" --model sonnet   # haiku | sonnet (Opus blocked)

# 5. Multi-turn chat (remembers the conversation)
uv run skai chat

# 6. Serve over MCP (stdio) for Claude Desktop / an IDE
uv run skai mcp
```

## Commands

| Command | What it does |
|---|---|
| `skai ingest [--path data/docs --urls data/urls.txt --reset]` | Load → chunk → embed → persist to Chroma |
| `skai ask "..." [--source pdf\|md\|web] [--thread-id X]` | One-shot question with citations |
| `skai chat` | Interactive multi-turn chat with memory |
| `skai mcp` | Run the MCP server exposing `search_kb` and `ask` |
| `skai eval` | Run the DeepEval quality suite (needs `--group eval` + key) |

## MCP client config

The server exposes two tools — `search_kb(query, source_type?)` (raw retrieval) and
`ask(question)` (full agent). Point an MCP client at it:

```json
{
  "mcpServers": {
    "solid-knowledge-ai": {
      "command": "uv",
      "args": ["run", "skai", "mcp"],
      "cwd": "/absolute/path/to/solid-knowledge-ai"
    }
  }
}
```

## Model selection

Default is **Haiku 4.5** (fast, cheap — good for a Q&A router+grader+generator loop).
Switch per call with `--model`, or globally via `SKAI_MODEL` in `.env`:

| Value | Resolves to |
|---|---|
| `haiku` (default) | `anthropic/claude-haiku-4-5` |
| `sonnet` | `anthropic/claude-sonnet-4-5` |
| any LiteLLM id | passed through (e.g. `openai/gpt-4o-mini`) |

Opus is intentionally blocked (`resolve_model` raises), so the assistant can't be
pointed at the most expensive tier by accident.

## Observability

Set `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` (and optionally `LANGFUSE_HOST`) in
`.env`. Every graph run then produces one trace with a span per node and per LLM call.
Without keys, tracing is a clean no-op — nothing else changes.

## Quality evaluation (DeepEval)

```bash
uv sync --group eval
export ANTHROPIC_API_KEY=...
uv run skai ingest
uv run --group eval pytest evals -v      # or: skai eval
```

The judge is **Claude via LiteLLM**, so no OpenAI key is needed. Metrics: faithfulness,
answer relevancy, contextual relevancy — plus a cheap keyword gate.

## Tests

```bash
uv run pytest        # 39 tests, fully offline: no network, no API keys
```

The LLM is dependency-injected, so the whole graph runs in tests against a deterministic
stub, and Chroma uses a deterministic in-process embedding function.

## How it's put together

```
src/skai/
  config.py            settings (.env)              agent/llm.py     ChatLiteLLM -> Claude
  models.py            Document/Chunk/AgentState    agent/nodes.py   route/retrieve/grade/generate/self_check
  ingest/loaders.py    pdf | md | web  -> Document  agent/prompts.py node prompts
  ingest/chunk.py      source-aware splitting       agent/graph.py   StateGraph + SQLite memory
  ingest/store.py      Chroma add/query             observability.py Langfuse handler (or no-op)
  cli.py               ingest|ask|chat|mcp|eval     mcp_server.py    search_kb / ask as MCP tools
evals/                 DeepEval suite               tests/           offline unit + graph tests
```

**Design rationale and tech trade-offs:** see [`docs/DECISIONS.md`](docs/DECISIONS.md).

## Status

Verified: `uv run skai ingest` loads all three source types (2 md + 1 pdf + 1 web →
170 chunks) and real semantic retrieval returns relevant passages. 39 offline tests
pass. `ask`/`chat`/`eval` require an `ANTHROPIC_API_KEY`.
