"""Core data shapes shared across ingestion and the agent."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph.message import add_messages

Route = Literal["kb", "chitchat", "out_of_scope"]


@dataclass
class Document:
    """A normalized unit of source content before chunking.

    metadata carries at least: source_type ("pdf"|"md"|"web"), source_id, title,
    and (for web) uri.
    """

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """A retrievable slice of a Document. metadata includes chunk_index."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    """A chunk returned from vector search, with its distance-derived score."""

    text: str
    metadata: dict[str, Any]
    score: float


class AgentState(TypedDict, total=False):
    """LangGraph state. `messages` accumulates across turns via the checkpointer."""

    question: str
    messages: Annotated[list, add_messages]
    source_type: str | None
    route: Route
    docs: list[RetrievedChunk]
    docs_ok: bool
    answer: str
    citations: list[str]
    critique: str | None
    grounded: bool
    retries: int
