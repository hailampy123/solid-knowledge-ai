"""Golden question set over the sample orca corpus.

`must_include` is used only for a cheap keyword sanity check; the DeepEval
metrics (faithfulness, relevancy, hallucination) do the real judging.
"""

import json
import os

GOLDEN = [
    {
        "question": "What do orcas eat?",
        "must_include": ["fish", "seal"],
    },
    {
        "question": "How do orcas communicate?",
        "must_include": ["dialect", "call"],
    },
    {
        "question": "Are orcas whales or dolphins?",
        "must_include": ["dolphin"],
    },
    {
        "question": "What threats do orcas face?",
        "must_include": ["salmon", "noise", "pollut"],
    },
]

GROWN_PATH = "evals/golden.jsonl"


def load_golden(extra_path: str = GROWN_PATH) -> list[dict]:
    """Base golden set + cases grown from real 👎 (feedback.promote_downvotes).

    The grown file is git-ignored and human-reviewed before it counts; a
    promoted case with an empty `must_include` skips only the keyword gate.
    """
    cases = list(GOLDEN)
    if os.path.exists(extra_path):
        with open(extra_path) as f:
            cases += [json.loads(ln) for ln in f if ln.strip()]
    return cases
