"""Shared test helpers: a deterministic, offline embedding function for Chroma."""
from __future__ import annotations

import hashlib

import pytest
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

DIM = 64


class DeterministicEmbeddingFunction(EmbeddingFunction[Documents]):
    """Hashing bag-of-words embedding. Offline, deterministic, good enough that
    a query overlapping a document ranks that document first."""

    def __init__(self) -> None:
        pass

    def __call__(self, input: Documents) -> Embeddings:
        import numpy as np

        return [np.array(self._embed(text), dtype=np.float32) for text in input]

    @staticmethod
    def name() -> str:
        return "deterministic-test-ef"

    def get_config(self) -> dict:
        return {}

    @classmethod
    def build_from_config(cls, config):
        return cls()

    @staticmethod
    def _embed(text: str) -> list[float]:
        vec = [0.0] * DIM
        for token in text.lower().split():
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            vec[h % DIM] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]


@pytest.fixture
def ef():
    return DeterministicEmbeddingFunction()


class _Msg:
    def __init__(self, content: str):
        self.content = content


class StubLLM:
    """Offline chat model. `responder(system, user) -> str` drives each node."""

    def __init__(self, responder):
        self._responder = responder
        self.calls: list[tuple[str, str]] = []

    def invoke(self, messages):
        system = messages[0].content
        user = messages[-1].content
        self.calls.append((system, user))
        return _Msg(self._responder(system, user))

