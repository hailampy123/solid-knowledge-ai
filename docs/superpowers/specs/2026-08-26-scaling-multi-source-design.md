# Scaling Solid Knowledge AI to Many Sources & Thousands of Documents — Design Spec

**Date:** 2026-08-26
**Status:** Draft for review (design only — no implementation yet)
**Supersedes ingestion/storage/retrieval sections of:** `2026-08-25-solid-knowledge-ai-design.md`
**Context:** The current build is a thin slice: batch ingest of a small corpus into a local Chroma file, dense-only top-k retrieval, an LLM relevance grade. This spec describes what changes when the target becomes **many internal hubs (Confluence, Notion, Google Drive, SharePoint, Slack, Jira, git) and thousands→millions of chunks**, optimizing for performance, latency, storage, freshness, and conflict management — plus access control, which internal hubs make mandatory.

---

## 1. Goals & Non-Goals

### Goals
1. Ingest from **multiple heterogeneous hubs** via pluggable connectors, incrementally.
2. Serve **low-latency, high-precision** answers over 10⁴–10⁶ chunks.
3. Keep the index **fresh** (near-real-time for critical hubs; deletes propagate).
4. Handle **duplication and contradictions** across sources with explicit precedence.
5. Enforce **per-user access control** at retrieval time.
6. Preserve everything that already works: the LangGraph agent, LiteLLM model switching, Langfuse tracing, DeepEval + feedback loop, citations/transparency.

### Non-Goals
- Rewriting the agent control flow (route → retrieve → generate → self_check stays).
- Building our own vector database or crawler framework (use managed/OSS).
- Full document-management/versioning UI (we store what we need for retrieval + audit, not a DMS).
- Real-time collaborative editing or write-back to source hubs (read-only ingestion).

### Success Criteria
1. A new hub is added by writing one connector class implementing a fixed interface — no core changes.
2. Incremental sync re-embeds **only changed docs** (measured: unchanged docs skipped via content hash); deletes remove chunks within one sync cycle.
3. p95 end-to-end answer latency ≤ **3 s** at 10⁵ chunks (hot path ≤ 2 LLM calls) with streaming first-token ≤ 1 s.
4. Retrieval precision (DeepEval contextual-relevancy diagnostic) improves materially vs the dense-only baseline (current worst case 0.27) after hybrid + rerank.
5. A user never receives a chunk their ACL forbids (enforced by a pre-filter, verified by test).
6. When sources contradict, the answer surfaces both with source + date rather than silently choosing.

---

## 2. Current-State Gaps (mapped to code)

| Component | Today | Gap at scale |
|---|---|---|
| `ingest/store.py` | local Chroma `PersistentClient` | single-process/file; no concurrency, sharding, or server ops |
| `cli.py ingest` + `loaders.load_sources` | batch, re-embeds all | O(corpus) every run; no freshness; no deletes |
| `agent/nodes.retrieve` | dense-only `top_k` | recall/precision fall at scale; misses exact terms |
| `agent/nodes.grade_docs` | LLM grade per query | extra hot-path LLM call; slow + costly |
| `models.Document` | `{text, metadata{source_type, source_id, title, uri}}` | no `content_hash`, `updated_at`, `acl`, `version`, `authority` → can't dedup/freshness/ACL/conflict |
| `agent/graph.py` checkpointer | `SqliteSaver` | single-writer; not for concurrent users |

---

## 3. Target Architecture

```
 hubs ── connectors ──► normalize(Document + rich metadata)
                              │
                    incremental sync (delta API / webhook)
                    content_hash gate → chunk → embed(cache)
                              │
                    UPSERT by stable doc id / DELETE on tombstone
                              ▼
        server vector DB (dense)  +  keyword index (BM25/sparse)
                              │
   query ─► ACL + metadata pre-filter ─► hybrid search (RRF) ─► top-N (~50)
                              │
                     cross-encoder rerank ─► top-k (~5)
                              ▼
     LangGraph agent: route → retrieve → [rerank-gate] → generate → self_check
                              │
                 conflict-aware synthesis + citations (+ freshness)
```

**Separation of planes.** Ingestion (write) and serving (read) are decoupled services sharing only the vector DB. Ingestion runs on a scheduler; serving is a stateless API that scales horizontally.

---

## 4. Data Model Changes (`models.Document` / `Chunk`)

Extend the common metadata contract. Every chunk carries:

| Field | Purpose |
|---|---|
| `source_system` | `confluence` \| `notion` \| `gdrive` \| `slack` \| `jira` \| `git` \| `web` \| `file` |
| `source_id` | stable id within the hub (used for UPSERT/DELETE) |
| `url`, `title`, `author` | citation + authority |
| `updated_at` (ISO) | freshness, recency ranking |
| `content_hash` | skip unchanged; near-dup detection |
| `version` | latest-wins; optional history |
| `acl` | list of principals/groups allowed to read (retrieval pre-filter) |
| `authority` (int) | precedence for conflict resolution (higher wins) |
| `space` / `collection` | namespace/partition + metadata filter |

Chunk id = `f"{source_system}:{source_id}:{chunk_index}"` (stable → idempotent upsert/delete).

---

## 5. Component Specs

### 5.1 Connectors (`ingest/connectors/`)
- Interface: `class Connector(Protocol): def changes(self, since: datetime|None) -> Iterable[SourceDoc]; def fetch(self, id) -> SourceDoc; def deletions(self, since) -> Iterable[str]`.
- Each hub adapter maps its native objects → `Document` with full metadata (incl. `acl` from the hub's sharing model).
- Delta strategy per hub: change/audit API where available (Confluence CQL `lastmodified`, Notion `last_edited_time`, Drive changes feed, Jira `updated`), else timestamp watermark.
- Webhooks for near-real-time hubs; scheduled polling for the rest.
- Failures isolate per doc (already the pattern in `loaders.load_sources`).

### 5.2 Sync service (`ingest/sync.py` + scheduler)
- Watermark per source (last successful sync time) persisted.
- For each changed doc: fetch → `content_hash`; if unchanged in the index, skip; else chunk → embed → UPSERT.
- For each deletion: DELETE all chunks with that `source_id` (tombstone).
- Orchestrate with Dagster/Prefect/Airflow (or cron + a queue for a lean start). Idempotent and resumable.
- Embedding cache keyed by `content_hash` (avoid re-embedding identical text).

### 5.3 Storage (`ingest/store.py` — same interface, new backend)
- Default recommendation: **pgvector** (one system for vectors + metadata + ACL + transactions) if Postgres is already in the stack; else **Qdrant** for payload filtering + quantization + sharding at higher scale; managed (Pinecone/Vertex/Databricks VS) when ops must be minimal.
- Keep the `Store` API (`add`, `query`, `count`, `reset`) so the agent is untouched; add `delete(source_id)` and `upsert` semantics.
- Index: HNSW (tune `m`, `ef`); partition/namespace by `space`/tenant. Quantization (int8/PQ) only at millions of vectors.
- Source-of-truth documents live in object storage / DB; the vector store holds chunks + embeddings + metadata.

### 5.4 Retrieval (`agent/nodes.retrieve` + new `retrieval/` module)
1. **ACL + metadata pre-filter** (hard filter by requesting principal, optional `source`/recency).
2. **Hybrid search**: dense (vector) + sparse (BM25) candidates fused with **Reciprocal Rank Fusion**.
3. **Cross-encoder rerank** the ~50 candidates → top-k (~5). (Voyage/Cohere rerank API, or a local `bge-reranker`.)
4. Return chunks + rerank scores.
- The **rerank score replaces the LLM `grade_docs`** as the relevance gate → one fewer hot-path LLM call. Corrective query-rewrite still triggers when top rerank score < threshold.

### 5.5 Agent latency (`agent/graph.py`, `nodes`)
- Per-node model tier: small/fast (Haiku) for `route`; strong only for `generate`. (Reuse `resolve_model`; add `SKAI_ROUTE_MODEL`.)
- `self_check` runs only when confidence is low (rerank/coverage heuristic), not every turn.
- Concurrent retrieval across namespaces (async); stream `generate` tokens.
- Caches: embedding cache, semantic query-result cache, rerank cache.
- Checkpointer `SqliteSaver` → **Postgres/Redis** for concurrent users; serving behind an API with pooling.

### 5.6 Freshness
- Incremental sync + webhooks; per-source cadence policy.
- **Tombstones**: deletions remove chunks within one cycle (test-enforced).
- Staleness surfaced in citations (`updated_at` → "updated N days ago").
- Optional re-index/TTL policy per source for embedding-model upgrades.

### 5.7 Conflict / contradiction management
1. **Dedup**: drop near-duplicate chunks (`content_hash` exact + embedding-similarity threshold).
2. **Authority precedence**: `authority` + `updated_at` as rerank tie-breakers (official > wiki > chat; newer > older).
3. **Recency weighting** for time-sensitive spaces.
4. **Conflict-aware generation**: a prompt variant that, when top chunks disagree, presents each position with source + date and flags the conflict instead of silently choosing. Optional `detect_conflict` node.
5. **Governance**: source-of-truth designation per topic via `authority`/ownership metadata.

### 5.8 Access control (mandatory)
- `acl` stored per chunk; retrieval applies it as a **hard pre-filter** using the requesting user's principals/groups.
- The API authenticates the user and passes principals into `retrieve`; no principal → only public docs.
- Test: a restricted chunk is never returned to a user lacking its ACL.

### 5.9 Observability & eval at scale
- Langfuse spans already per node; add retrieval metrics (candidate count, rerank score distribution, cache hit rate, per-node latency).
- DeepEval golden set + the existing user-feedback loop run on a schedule to catch regressions when the corpus shifts; contextual-relevancy tracked as the retriever-precision diagnostic.
- Freshness SLA + sync-failure alerts.

---

## 6. Config additions (`config.py`)
`vector_backend` (`chroma|pgvector|qdrant|...`), backend DSN/keys, `retrieval_candidates` (N, default 50), `rerank_model`, `rerank_top_k`, `hybrid_alpha` (dense/sparse weight), `route_model`, `embedding_cache_dir`, `checkpointer_backend`, per-source sync cadence. Aliases resolved as today; Opus still blocked.

---

## 7. Migration & Phasing (thin-slice discipline preserved)
Each phase ships working, testable software.

1. **Retrieval quality (no infra change):** hybrid + cross-encoder rerank behind the current `Store`/`retrieve` seam; rerank-gate replaces LLM grade. Re-run DeepEval — contextual-relevancy diagnostic is the proof metric.
2. **Storage backend:** implement pgvector/Qdrant behind `Store`; add `upsert`/`delete`; migration script from Chroma.
3. **Metadata + freshness:** extend `Document`; add `content_hash` skip + tombstones; incremental sync for one hub (e.g. Confluence).
4. **Conflict + ACL:** authority precedence + dedup; ACL pre-filter + auth on the serving API.
5. **Scale-out & latency:** caches, Postgres/Redis checkpointer, async retrieval, per-node model tiers, streaming API.

## 8. Risks & Mitigations
| Risk | Mitigation |
|---|---|
| Rerank adds latency | rerank only ~50 candidates; cache; local reranker option |
| Connector ACL models differ per hub | normalize to principals/groups; default-deny on unknown |
| Embedding-model upgrade invalidates index | versioned index + background re-embed via `content_hash` cache |
| Contradictions hard to auto-detect | start with authority/recency precedence; add LLM conflict node later |
| pgvector limits at millions | path to Qdrant/managed already isolated by `Store` interface |

## 9. Open Questions (to resolve before Phase 1)
1. Which hubs are in scope first, and do their APIs expose reliable delta + ACL?
2. Is Postgres already in the stack (→ pgvector) or do we add a dedicated vector DB?
3. Rerank: managed API (Cohere/Voyage) vs self-hosted `bge-reranker` (latency/cost/privacy)?
4. Identity provider for ACL principals (SSO/groups source of truth)?
5. Freshness SLA per hub (seconds vs hours) — drives webhook vs poll.
