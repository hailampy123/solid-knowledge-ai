# Tech Decisions & Trade-offs

Interview cheat-sheet: each decision, what it was weighed against, and the one-line
reason. The theme throughout: **make the agentic control flow explicit and inspectable,
keep the demo runnable with a single API key, and be able to prove it works.**

## D1 — Orchestration: LangGraph (vs. AgentExecutor / a hand-rolled loop)

The value of this project is the *agentic control flow* — routing, a corrective
retrieval loop, a self-check loop, bounded retries, and checkpointed memory. LangGraph
makes that flow a first-class, inspectable state machine: nodes return state patches,
conditional edges make routing explicit, and the checkpointer gives memory for free. A
LangChain `AgentExecutor` hides control flow inside a ReAct loop you can't easily bound
or branch; a hand-rolled loop reinvents the checkpointer and edge logic. LangGraph is the
reason the retry budget and the "decline instead of hallucinate" behavior are easy to
reason about and test.

## D2 — LLM gateway: LiteLLM via `ChatLiteLLM` (vs. the Anthropic SDK directly)

LiteLLM gives one interface across providers, built-in retries/fallbacks, and config-only
model swaps (`SKAI_MODEL=...`). Wrapping it as LangChain's `ChatLiteLLM` means it plugs
straight into LangGraph and the Langfuse callback handler. Calling the Anthropic SDK
directly would couple every node to one provider and lose the callback integration. Cost:
one more abstraction layer — worth it for provider-agnosticism in an agent that may need
a cheaper model for routing/grading later.

## D3 — Embeddings: Chroma's local MiniLM (vs. provider embeddings)

**Anthropic has no embeddings API.** Rather than pull in a second provider (OpenAI /
Voyage) just for embeddings, the store uses Chroma's built-in local `all-MiniLM-L6-v2`.
Embeddings become free, offline, and fast; generation stays on Claude. This *decoupling
of embeddings from generation* is deliberate — it keeps the demo runnable with a single
key and removes a network dependency from ingestion. Upgrade path: swap in provider
embeddings + a reranker when retrieval precision matters more than portability.

## D4 — Vector store: ChromaDB, persistent (vs. FAISS / pgvector / Pinecone)

Chroma gives zero-infra local persistence *and* metadata filtering — the latter is
required for the `--source pdf|md|web` filter. FAISS is a raw index with no metadata
layer; pgvector and Pinecone add a server or a SaaS account for no benefit at this scale.
The embedding function is injectable, which is what lets tests stay fully offline.

## D5 — Web extraction: trafilatura (vs. BeautifulSoup / readability)

Getting clean article text out of arbitrary HTML is its own problem. trafilatura is
purpose-built for it and, in the test fixture, correctly strips nav and footer
boilerplate that a hand-rolled BeautifulSoup pass would leave in. Less code, better input
to the chunker.

## D6 — Observability: Langfuse callback handler (vs. LangSmith)

Langfuse is OSS/self-hostable and integrates as a single LangChain callback, so one
handler traces every node and LLM call with no code changes at the call sites. It's wired
to **no-op without keys**, so tracing never becomes a setup blocker or a test dependency.

## D7 — Evaluation: DeepEval as pytest (vs. Ragas / manual)

"How do you know it works?" is answered with the RAG triad — faithfulness, answer
relevancy, contextual relevancy — expressed as `pytest` cases, so eval runs in CI like
any other test. The judge is **Claude via a custom LiteLLM-backed `DeepEvalBaseLLM`**, so
the eval needs only the same `ANTHROPIC_API_KEY` as the app — no OpenAI dependency, which
DeepEval otherwise defaults to.

## D8 — Tool exposure: MCP (`MCPServer`) (vs. a REST API)

The same core (`search_kb`, `ask`) is surfaced as MCP tools, so any MCP client — Claude
Desktop, an IDE — can drive the knowledge base with no bespoke client. The tools reuse the
exact CLI/agent functions, so there's one implementation, not two. A REST API would be
more code and less relevant to an agent-native workflow.

## D9 — Runtime & packaging: Python 3.11 + uv, Typer CLI

3.11 clears LangGraph's floor with the widest wheel coverage for Chroma's native deps.
`uv` gives fast, reproducible installs (committed `uv.lock`) and dependency groups — which
is how the heavy DeepEval dependency stays *out* of the default install (`--group eval`),
keeping the offline test path lean. Typer gives a clean multi-command entrypoint.

## D10 — Agent pattern: self-reflective / corrective RAG (vs. linear RAG / multi-agent)

Linear RAG demonstrates no agency. A multi-agent supervisor (one agent per source) is
more impressive on a slide but adds failure surface and coordination cost without helping
answer quality at this scale — so it's a documented scale-up, not built. Corrective RAG is
the sweet spot: a single loop that shows real agency (self-grading, query rewriting,
groundedness verification, honest refusal) while staying runnable and testable.

---

## Testing strategy (why the tests are trustworthy)

The LLM is **dependency-injected** into every node, so the full graph runs in tests
against a deterministic `StubLLM` — routing, the retry loop, citation assembly, and the
"hedge when ungrounded" path are all asserted with **no network and no keys**. Chroma uses
a deterministic in-process embedding function for the same reason. The corrective loop's
termination (bounded retries) is an explicit test, not an assumption.

## Scale-up notes (talking points, not built — YAGNI)

- **Multi-agent supervisor** routing a specialist agent per source/domain.
- **Semantic long-term memory** (vector-backed user memory) alongside the checkpointer.
- **Provider embeddings + reranking** (Voyage/Cohere) when precision beats portability.
- **Incremental ingestion** with content hashing to skip re-embedding unchanged sources.
- **Streaming + a thin API/UI** in front of the agent for a product surface.
