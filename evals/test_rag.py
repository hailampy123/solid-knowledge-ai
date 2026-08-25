"""DeepEval quality suite for the RAG agent.

Not collected by `pytest tests/` (offline). Run it explicitly:

    uv sync --group eval
    export ANTHROPIC_API_KEY=...
    skai ingest                      # build the store first
    uv run --group eval pytest evals -v      (or: skai eval)

Judged by Claude via LiteLLM, so only an ANTHROPIC_API_KEY is required.
"""
from __future__ import annotations

import json
import os
import re

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY required for the DeepEval judge",
)

deepeval = pytest.importorskip("deepeval")
from deepeval import assert_test  # noqa: E402
from deepeval.metrics import (  # noqa: E402
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.models import DeepEvalBaseLLM  # noqa: E402
from deepeval.test_case import LLMTestCase  # noqa: E402

from evals.dataset import GOLDEN
from skai.agent.graph import answer_question, build_graph
from skai.agent.llm import make_llm
from skai.config import get_settings
from skai.ingest.store import Store


def _extract_json(text: str) -> str:
    text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start != -1 else text


class LiteLLMJudge(DeepEvalBaseLLM):
    """DeepEval judge backed by LiteLLM -> Claude (no OpenAI key needed)."""

    def __init__(self, model: str):
        self._model = model

    def load_model(self):
        return self._model

    def _call(self, prompt: str) -> str:
        import litellm

        resp = litellm.completion(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return resp.choices[0].message.content or ""

    def generate(self, prompt: str, schema=None):
        if schema is None:
            return self._call(prompt)
        guided = (
            prompt
            + "\n\nRespond ONLY with a JSON object matching this schema:\n"
            + json.dumps(schema.model_json_schema())
        )
        data = json.loads(_extract_json(self._call(guided)))
        return schema(**data)

    async def a_generate(self, prompt: str, schema=None):
        return self.generate(prompt, schema)

    def get_model_name(self) -> str:
        return self._model


@pytest.fixture(scope="module")
def agent():
    settings = get_settings()
    store = Store(settings.chroma_dir, settings.collection)
    if store.count() == 0:
        pytest.skip("empty store — run `skai ingest` before the eval suite")
    graph = build_graph(store, make_llm(settings, callbacks=[]), top_k=settings.top_k)
    return graph, store, LiteLLMJudge(settings.model)


@pytest.mark.parametrize("case", GOLDEN, ids=[c["question"] for c in GOLDEN])
def test_rag_quality(agent, case):
    graph, store, judge = agent
    q = case["question"]
    out = answer_question(graph, q, thread_id=f"eval-{q}")
    context = [h.text for h in store.query(q, k=get_settings().top_k)]

    # cheap deterministic gate before the (LLM-judged) metrics
    haystack = (out["answer"] + " " + " ".join(context)).lower()
    assert any(k in haystack for k in case["must_include"]), f"no expected keyword for: {q}"

    tc = LLMTestCase(
        input=q,
        actual_output=out["answer"],
        retrieval_context=context,
    )
    assert_test(
        tc,
        [
            FaithfulnessMetric(threshold=0.7, model=judge),
            AnswerRelevancyMetric(threshold=0.7, model=judge),
            ContextualRelevancyMetric(threshold=0.5, model=judge),
        ],
    )
