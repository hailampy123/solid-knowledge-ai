"""Golden question set over the sample orca corpus.

`must_include` is used only for a cheap keyword sanity check; the DeepEval
metrics (faithfulness, relevancy, hallucination) do the real judging.
"""

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
