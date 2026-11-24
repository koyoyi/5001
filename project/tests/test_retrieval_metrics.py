"""Small formula and leakage-boundary checks, runnable without model downloads."""

import math
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ragstudy.metrics import paired_group_bootstrap, ranking_metrics
from ragstudy.retrieval import (
    BM25Index, SentenceEncoderIndex, SparseLSAIndex, rrf_scores, tokenize, top_k,
)


class RetrievalTests(unittest.TestCase):
    def test_tokenizer_retains_technical_terms(self):
        self.assertEqual(tokenize("C++ C# CUDA11.7 gpu_0 error-404"),
                         ["c++", "c#", "cuda11.7", "gpu_0", "error-404"])

    def test_bm25_matches_hand_computed_score(self):
        index = BM25Index(["cat cat dog", "dog fox", ""])
        expected = math.log1p((3 - 1 + .5) / (1 + .5)) * (
            2 * 2.2 / (2 + 1.2 * (.25 + .75 * 3 / (5 / 3)))
        )
        self.assertAlmostEqual(float(index.score("cat")[0]), expected, places=6)
        np.testing.assert_allclose(index.score("cat cat"), 2 * index.score("cat"))
        np.testing.assert_array_equal(index.score("unknown"), [0, 0, 0])
        np.testing.assert_array_equal(top_k(index.score("cat"), 3, min_score=0), [0])

    def test_empty_bm25_corpora_are_safe(self):
        np.testing.assert_array_equal(BM25Index([]).score("cat"), [])
        np.testing.assert_array_equal(BM25Index(["", "!!!"]).score("cat"), [0, 0])

    def test_top_k_is_stable_and_threshold_is_explicit(self):
        np.testing.assert_array_equal(top_k([1, 2, 2, 0, -1], 4), [1, 2, 0, 3])
        np.testing.assert_array_equal(top_k([2, 2, 1], 3, ["z", "a", "m"]), [1, 0, 2])
        np.testing.assert_array_equal(top_k([0, -1], 2), [0, 1])
        np.testing.assert_array_equal(top_k([0, -1], 2, min_score=0), [])
        np.testing.assert_array_equal(top_k([1], 0), [])
        with self.assertRaises(ValueError):
            top_k([float("nan")], 1)

    def test_lsa_is_finite_and_labels_its_actual_method(self):
        index = SparseLSAIndex([
            "gpu cuda driver memory", "stocks bonds finance market",
            "gpu compute cuda hardware",
        ], dimensions=2)
        scores = index.score("gpu cuda")
        self.assertEqual(scores.shape, (3,))
        self.assertTrue(np.isfinite(scores).all())
        self.assertGreater(float(scores[0]), float(scores[1]))
        self.assertFalse(index.metadata["neural_encoder"])
        np.testing.assert_allclose(index.query_scores_batch(["gpu cuda"])[0], scores)

    def test_degenerate_lsa_corpora(self):
        for documents in ([], [""], ["word"], ["word", "word"]):
            scores = SparseLSAIndex(documents).score("word")
            self.assertEqual(scores.shape, (len(documents),))
            self.assertTrue(np.isfinite(scores).all())

    def test_sentence_encoder_rejects_nonlocal_model_before_import(self):
        with self.assertRaises(FileNotFoundError):
            SentenceEncoderIndex(["gpu"], Path(__file__).parent / "missing-model")

    def test_sentence_encoder_is_offline_normalized_and_encodes_query_once(self):
        calls = []

        class FakeEncoder:
            def __init__(self, path, **kwargs):
                calls.append(("load", path, kwargs))

            def encode(self, texts, **kwargs):
                calls.append(("encode", texts, kwargs))
                return np.tile(np.asarray([[1., 0.]], dtype=np.float32), (len(texts), 1))

        with TemporaryDirectory() as directory:
            (Path(directory) / "config.json").write_text(
                json.dumps({"_commit_hash": "pinned"}), encoding="utf-8",
            )
            with patch.dict(sys.modules, {
                "sentence_transformers": SimpleNamespace(SentenceTransformer=FakeEncoder),
            }):
                index = SentenceEncoderIndex(["gpu", "finance"], directory)
                scores = index.score("cuda")
        self.assertEqual(calls[0][2], {"device": "cpu", "local_files_only": True})
        self.assertEqual(calls[1][1], ["gpu", "finance"])
        self.assertTrue(calls[1][2]["normalize_embeddings"])
        self.assertEqual(calls[2][1], [
            "Represent this sentence for searching relevant passages: cuda",
        ])
        self.assertEqual(len(calls), 3)
        self.assertEqual(index.model.max_seq_length, 512)
        self.assertEqual(index.metadata["model_config_revision"], "pinned")
        self.assertEqual(len(index.metadata["model_config_sha256"]), 64)
        np.testing.assert_array_equal(scores, [1., 1.])

    def test_rrf_ranks_start_at_one_and_keep_complete_union(self):
        candidates = ["a", "b", "c"]
        scores = rrf_scores(candidates, ["a", "b"], ["c", "a"], constant=60)
        np.testing.assert_allclose(scores, [1 / 61 + 1 / 62, 1 / 62, 1 / 61])
        order = [candidates[index] for index in top_k(scores, len(candidates))]
        self.assertEqual(set(order), set(candidates))
        self.assertEqual(order[0], "a")
        with self.assertRaises(ValueError):
            rrf_scores(["a", "a"], ["a"], [])


class MetricTests(unittest.TestCase):
    def test_missing_gold_still_contributes_to_ideal_dcg(self):
        metrics = ranking_metrics(["a", "x"], ["a", "b"])
        self.assertAlmostEqual(metrics["ndcg@10"], 1 / (1 + 1 / math.log2(3)))
        self.assertEqual(metrics["recall@20"], .5)

    def test_duplicate_ranked_documents_do_not_create_relevance(self):
        metrics = ranking_metrics(["a", "a", "b"], ["a", "b"], ndcg_k=2, recall_k=2)
        self.assertEqual(metrics, {"ndcg@2": 1., "recall@2": 1.})
        self.assertEqual(ranking_metrics([], ["a"]), {"ndcg@10": 0., "recall@20": 0.})
        self.assertEqual(ranking_metrics(["a"], []), {"ndcg@10": 0., "recall@20": 0.})

    def test_group_bootstrap_preserves_group_sizes_and_query_macro_mean(self):
        rows = [
            {"query_id": "a", "group_id": "shared", "base": {"m": 0}, "new": {"m": 1}},
            {"query_id": "b", "group_id": "shared", "base": {"m": 0}, "new": {"m": 1}},
            {"query_id": "c", "group_id": "other", "base": {"m": 1}, "new": {"m": 0}},
        ]
        result = paired_group_bootstrap(rows, "base", "new", "m", iterations=500, seed=7)
        draws = np.random.default_rng(7).integers(0, 2, size=(500, 2))
        expected = np.asarray([2, -1])[draws].sum(axis=1) / np.asarray([2, 1])[draws].sum(axis=1)
        lower, upper = np.percentile(expected, [2.5, 97.5])
        self.assertAlmostEqual(result["difference"], 1 / 3)
        self.assertEqual(result["ci_lower"], lower)
        self.assertEqual(result["ci_upper"], upper)
        self.assertEqual(result["n_queries"], 3)
        self.assertEqual(result["n_groups"], 2)
        self.assertEqual(result, paired_group_bootstrap(
            rows, "base", "new", "m", iterations=500, seed=7,
        ))

    def test_bootstrap_rejects_duplicate_queries_and_missing_rows(self):
        row = {"query_id": "a", "group_id": "g", "base": {"m": 0}, "new": {"m": 1}}
        with self.assertRaises(ValueError):
            paired_group_bootstrap([row, row], "base", "new", "m")
        with self.assertRaises(ValueError):
            paired_group_bootstrap([], "base", "new", "m")


if __name__ == "__main__":
    unittest.main()
