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
