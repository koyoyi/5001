import numpy as np
import pytest
from ragstudy.retrieval import BM25Index, top_k
from ragstudy.features import build_features


@pytest.mark.parametrize("question", ["IBM IBM timeout unknown", "C++ 7.1 retry", "", "missing"])
def test_postings_features_equal_text_features(question):
    documents = ["IBM retry timeout", "C++ 7.1 retry retry", "unrelated text", ""]
    index = BM25Index(documents)
    lexical = index.score(question)
    semantic = np.asarray([.1, .3, .2, -.1])
    lexical_order = top_k(lexical, 3, min_score=0)
    semantic_order = top_k(semantic, 3)
    candidates = np.asarray([0, 1, 2, 3])
    slow = build_features(question, candidates, documents, lexical, semantic, lexical_order, semantic_order)
    fast = build_features(question, candidates, documents, lexical, semantic, lexical_order, semantic_order, index)
    np.testing.assert_allclose(slow, fast, atol=1e-12)


def test_partial_top_k_matches_full_sort_with_ties():
    rng = np.random.default_rng(45)
    for count in [0, 1, 5, 1000]:
        values = rng.integers(-3, 4, size=count).astype(float)
        ids = [f"doc{count-index:05d}" for index in range(count)]
        for k in [0, 1, 3, 100, count + 2]:
            expected = sorted(range(count), key=lambda pos: (-values[pos], ids[pos], pos))[:k]
            assert top_k(values, k, ids).tolist() == expected
