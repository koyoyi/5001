"""Offline regression tests for locked cohorts and train-only iteration choices."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import zipfile

import numpy as np
import pytest

from ragstudy.data import load_techqa
from ragstudy.features import FEATURE_NAMES, sample_training_rows
from ragstudy import experiment


QUESTIONS = [
    "Amethyst orchard irrigation pressure",
    "Bronze satellite navigation calibration",
    "Crimson filesystem journal recovery",
    "Delta compiler vectorization warning",
    "Emerald database transaction deadlock",
    "Fuchsia network certificate renewal",
    "Golden display brightness oscillation",
    "Hazel container image synchronization",
    "Indigo printer authentication issue",
    "Jade hypervisor packet routing",
    "Khaki scheduler reservation lifecycle",
    "Lilac firmware bootloader recovery",
]


@pytest.fixture
def locked_corpus(tmp_path):
    source = tmp_path / "nvidia__TechQA-RAG-Eval"
    source.mkdir()
    documents = {f"d{i:02}.txt": f"Evidence for {question}.\n  Full source answer and tail {i}."
                 for i, question in enumerate(QUESTIONS)}
    records = [{"id": f"q{i:02}", "question": question, "is_impossible": False,
                "answer": "Full source answer", "contexts": [
                    {"filename": f"d{i:02}.txt", "text": documents[f"d{i:02}.txt"]}]}
               for i, question in enumerate(QUESTIONS)]
    (source / "train.json").write_text(json.dumps(records), encoding="utf-8")
    with zipfile.ZipFile(source / "corpus.zip", "w") as archive:
        for name, text in documents.items():
            archive.writestr("corpus/" + name, text)
    entries = [{"local_file": source.name + "/" + name,
                "repo": "nvidia/TechQA-RAG-Eval", "revision": "locked-fixture",
                "sha256": hashlib.sha256((source / name).read_bytes()).hexdigest()}
               for name in ("train.json", "corpus.zip")]
    (tmp_path / "download_manifest.json").write_text(json.dumps({"files": entries}), encoding="utf-8")
    lock = {f"q{i:02}": "train" if i < 6 else "dev" if i < 9 else "test" for i in range(12)}
    return source, lock


def split_map(dataset):
    return {query["query_id"]: query["split"] for query in dataset["queries"]}


def test_historical_holdout_membership_survives_new_random_seed(locked_corpus):
    source, lock = locked_corpus
    original_lock = dict(lock)
    for seed in (1, 20261114, 999):
        dataset = load_techqa(source, seed=seed, split_lock=lock)
        assert split_map(dataset) == lock
        assert dataset["audit"]["split_policy"] == "historical_split_lock_with_component_quarantine"
        assert dataset["audit"]["quarantined_queries"] == []
    assert lock == original_lock


def test_reviewed_transitive_component_crossing_holdout_is_wholly_quarantined(locked_corpus):
    source, lock = locked_corpus
    reviewed = [["q00", "q01"], ["q01", "q09"]]
    dataset = load_techqa(source, split_lock=lock, additional_query_groups=reviewed)
    expected_removed = {"q00", "q01", "q09"}
    assert split_map(dataset) == {qid: split for qid, split in lock.items() if qid not in expected_removed}
    quarantine = dataset["audit"]["quarantined_queries"]
    assert {row["query_id"] for row in quarantine} == expected_removed
    assert len({row["group_id"] for row in quarantine}) == 1
    assert all(row["previous_split"] == lock[row["query_id"]] for row in quarantine)
    assert all(row["reason"] == "reviewed_component_crosses_historical_splits" for row in quarantine)
    assert {qid for qid, split in split_map(dataset).items() if split == "test"} == {"q10", "q11"}
    assert dataset["audit"]["canonical_queries_before_quarantine"] == 12
    assert dataset["audit"]["canonical_answerable_queries"] == 9


def test_train_dev_conflict_does_not_become_a_new_holdout(locked_corpus):
    source, lock = locked_corpus
    dataset = load_techqa(source, split_lock=lock, additional_query_groups=[["q02", "q06"]])
    actual = split_map(dataset)
    assert "q02" not in actual and "q06" not in actual
    assert {qid for qid, split in actual.items() if split == "test"} == {"q09", "q10", "q11"}
    assert all(actual[qid] == lock[qid] for qid in actual)


def test_same_split_reviewed_merges_change_groups_without_moving_queries(locked_corpus):
    source, lock = locked_corpus
    reviewed = [["q00", "q01"], ["q06", "q07"], ["q09", "q10"]]
    dataset = load_techqa(source, seed=999, split_lock=lock, additional_query_groups=reviewed)
    by_id = {query["query_id"]: query for query in dataset["queries"]}
    assert split_map(dataset) == lock
    assert dataset["audit"]["quarantined_queries"] == []
    assert Counter(query["split"] for query in dataset["queries"]) == {"train": 6, "dev": 3, "test": 3}
    for left, right in reviewed:
        assert by_id[left]["group_id"] == by_id[right]["group_id"]
    assert dataset["audit"]["independent_components"] == 9


@pytest.mark.parametrize("change", ["missing", "extra", "replacement"])
def test_split_lock_requires_exact_original_canonical_ids(locked_corpus, change):
    source, lock = locked_corpus
    broken = dict(lock)
    if change in ("missing", "replacement"):
        broken.pop("q00")
    if change in ("extra", "replacement"):
        broken["unknown-query"] = "train"
    with pytest.raises(ValueError, match="exactly"):
        load_techqa(source, split_lock=broken)


def test_unknown_historical_split_is_rejected(locked_corpus):
    source, lock = locked_corpus
    lock["q00"] = "validation"
    with pytest.raises(ValueError, match="Unknown split"):
        load_techqa(source, split_lock=lock)


@pytest.mark.parametrize("reviewed", [[["q00"]], [["q00", "unknown-query"]]])
def test_reviewed_groups_require_known_canonical_members(locked_corpus, reviewed):
    source, lock = locked_corpus
    with pytest.raises(ValueError, match="known canonical"):
        load_techqa(source, split_lock=lock, additional_query_groups=reviewed)


def test_full_document_policy_preserves_full_text_without_changing_locked_queries(locked_corpus):
    source, lock = locked_corpus
    head = load_techqa(source, split_lock=lock, head_words=3, document_policy="head")
    full = load_techqa(source, split_lock=lock, head_words=3, document_policy="full")
    assert head["document_ids"] == full["document_ids"]
    assert head["queries"] == full["queries"]
    assert head["audit"]["source_hashes"] == full["audit"]["source_hashes"]
    assert full["audit"]["document_policy"] == "full"
    for doc_id, prefix, complete in zip(full["document_ids"], head["documents"], full["documents"]):
        assert complete == full["originals"][doc_id]
        assert prefix == " ".join(complete.split()[:3])
        assert len(complete.split()) > 3


def test_unknown_document_policy_is_rejected(locked_corpus):
    source, lock = locked_corpus
    with pytest.raises(ValueError, match="document_policy"):
        load_techqa(source, split_lock=lock, document_policy="undefined")


def ranking_record(number, *, positives=1, negatives=8, group_id=None):
    count = positives + negatives
    features = np.random.RandomState(number + 100).normal(size=(count, len(FEATURE_NAMES)))
    features[:, 0] = np.arange(count, 0, -1) + number * 10
    features[:, 1] = np.arange(count) + number * 100
    start = number * count
    return {"query": {"query_id": f"synthetic-{number}", "group_id": group_id or f"component-{number}",
                      "positive_document_ids": [f"doc-{start + i}" for i in range(positives)]},
            "features": features, "labels": np.r_[np.ones(positives, dtype=int), np.zeros(negatives, dtype=int)],
            "candidates": np.arange(start, start + count)}


def model_config(**overrides):
    config = {"seed": 20261114, "training_seed": 7, "max_proxy_negatives": 2,
              "features": ["semantic_score", "bm25_score"], "rf_trees": 5,
              "cv_folds": 2, "logistic_C": [1.0], "rf_depth": [2], "rf_min_leaf": [1]}
    config.update(overrides)
    return config


def test_training_seed_changes_negative_sample_without_changing_historical_splits(locked_corpus, monkeypatch):
    source, lock = locked_corpus
    initial = load_techqa(source, seed=20261114, split_lock=lock)
    records = [ranking_record(number) for number in range(2)]
    captured = []
    original_sampler = sample_training_rows

    def capture(records, max_negatives, seed, **kwargs):
        result = original_sampler(records, max_negatives, seed, **kwargs)
        captured.append((seed, result[0].copy()))
        return result

    monkeypatch.setattr(experiment, "sample_training_rows", capture)
    models = [experiment.fit_model("logistic", {"C": 1.0}, records, model_config(training_seed=seed))
              for seed in (7, 8)]
    assert [seed for seed, _ in captured] == [7, 8]
    assert not np.array_equal(captured[0][1], captured[1][1])
    assert [bundle["model"].random_state for bundle in models] == [7, 8]
    assert split_map(load_techqa(source, seed=20261114, split_lock=lock)) == split_map(initial) == lock


def test_hard_bm25_sampling_selects_highest_unjudged_scores_and_retains_all_positives():
    record = ranking_record(1, positives=2, negatives=4)
    record["features"][:, 0] = [1000, 999, 5, 9, 9, 1]
    record["features"][:, 1] = [100, 101, 102, 103, 104, 105]
    original = deepcopy(record)
    for seed in (7, 99):
        features, labels, _, _, missed = sample_training_rows(
            [record], max_negatives=2, seed=seed, negative_sampling="hard_bm25")
        assert features[labels == 1, 1].tolist() == [100, 101]
        assert features[labels == 0, 1].tolist() == [103, 104]
        assert missed == []
    tied, labels, *_ = sample_training_rows([record], max_negatives=1, negative_sampling="hard_bm25")
    assert tied[labels == 0, 1].tolist() == [103]
    np.testing.assert_array_equal(record["features"], original["features"])
    np.testing.assert_array_equal(record["labels"], original["labels"])


def test_class_balance_allocates_half_of_each_query_weight_to_each_class():
    small = ranking_record(1, positives=1, negatives=1, group_id="shared-component")
    large = ranking_record(2, positives=2, negatives=4, group_id="shared-component")
    features, labels, weights, groups, missed = sample_training_rows(
        [small, large], max_negatives=3, negative_sampling="hard_bm25", balance_classes=True)
    budgets = []
    for membership in (features[:, 1] < 200, features[:, 1] >= 200):
        query_total = weights[membership].sum()
        budgets.append(query_total)
        for label in (0, 1):
            assert np.isclose(weights[membership & (labels == label)].sum() / query_total, .5)
    assert np.isclose(budgets[0], budgets[1])
    assert np.isclose(weights.mean(), 1)
    assert set(groups) == {"shared-component"} and missed == []


@pytest.mark.parametrize("features", [[], ["bm25_score", "bm25_score"], ["unknown_feature"]])
def test_invalid_feature_subsets_are_rejected(features):
    with pytest.raises(ValueError):
        experiment.fit_model("logistic", {"C": 1.0}, [ranking_record(0)], model_config(features=features))


def test_feature_subset_and_scaler_remain_fold_local_during_group_cv(monkeypatch):
    records = [ranking_record(number, negatives=2) for number in range(4)]
    document_ids = [f"doc-{index}" for index in range(12)]
    original_fit = experiment.fit_model
    observed = []

    def checked_fit(kind, parameters, local_records, config):
        bundle = original_fit(kind, parameters, local_records, config)
        expected_indices = [FEATURE_NAMES.index(name) for name in config["features"]]
        assert bundle["feature_indices"] == expected_indices == [1, 0]
        assert bundle["model"].n_features_in_ == 2
        if kind == "logistic":
            matrix, _, weights, _, _ = sample_training_rows(
                local_records, config["max_proxy_negatives"], config["training_seed"])
            expected_mean = np.average(matrix[:, expected_indices], axis=0, weights=weights)
            np.testing.assert_allclose(bundle["scaler"].mean_, expected_mean)
            before = bundle["scaler"].mean_.copy()
            # A held-out outlier must be transformed by the training-fold scaler.
            validation = np.full((2, len(FEATURE_NAMES)), 10000.)
            predictions = experiment.predict(bundle, validation)
            assert predictions.shape == (2,) and np.isfinite(predictions).all()
            np.testing.assert_array_equal(bundle["scaler"].mean_, before)
        probe = local_records[0]["features"].copy()
        predictions = experiment.predict(bundle, probe)
        unused = [index for index in range(len(FEATURE_NAMES)) if index not in expected_indices]
        probe[:, unused] = 1e9
        np.testing.assert_allclose(experiment.predict(bundle, probe), predictions)
        observed.append((kind, len(local_records)))
        return bundle

    monkeypatch.setattr(experiment, "fit_model", checked_fit)
    models, cv = experiment.select_models(records, model_config(), document_ids)
    assert set(models) == {"logistic", "random_forest"}
    assert Counter(observed) == {("logistic", 2): 2, ("logistic", 4): 1,
                                 ("random_forest", 2): 2, ("random_forest", 4): 1}
    assert len(cv) == 2 and all(row["queries"] == 4 and row["folds"] == 2 for row in cv)
