"""Synthetic-only checks for the one-shot scientific holdout workflow.

No test opens the downloaded TechQA input or a real experiment artifact.
"""

from copy import deepcopy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import zipfile

import joblib
import numpy as np
import pytest

from ragstudy import experiment
from ragstudy.features import FEATURE_NAMES


PROJECT = Path(__file__).resolve().parents[1]


def load_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, PROJECT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes(source):
    return {path.name: digest(path) for path in source.iterdir() if path.is_file()}


class SavedIndexOnly:
    metadata = {"method": "synthetic-saved-index-no-encoding"}

    def score(self, *_args, **_kwargs):
        raise AssertionError("Frozen holdout must not rerun retrieval")

    def fit(self, *_args, **_kwargs):
        raise AssertionError("Frozen holdout must not rebuild an index")


class FittedModelOnly:
    def predict_proba(self, features):
        positive = np.clip(np.asarray(features)[:, 0], .01, .99)
        return np.column_stack((1. - positive, positive))

    def fit(self, *_args, **_kwargs):
        raise AssertionError("Frozen holdout must not refit the saved model")


@pytest.fixture
def frozen_module():
    return load_script("score_frozen_holdout_synthetic_test", "score_frozen_holdout.py")


@pytest.fixture
def synthetic_main(tmp_path):
    """Complete six-query saved evidence with two synthetic heldout queries."""
    source = tmp_path / "project/artifacts/synthetic_head_onnx"
    tracked = tmp_path / "project/results" / source.name
    input_path = tmp_path / "source/input"
    for path in (source, tracked, input_path):
        path.mkdir(parents=True)
    ids = [f"d{i}" for i in range(6)] + ["noise0", "noise1"]
    texts = [f"Synthetic source passage {document}." for document in ids]
    queries = [{"query_id": f"q{i}", "question": f"Synthetic diagnostic issue {i}",
                "group_id": f"g{i}", "split": ("train", "dev", "test")[i // 2],
                "positive_document_ids": [f"d{i}"], "source_record_ids": [f"q{i}"]}
               for i in range(6)]
    dump(input_path / "train.json", [{"id": query["query_id"]} for query in queries])
    with zipfile.ZipFile(input_path / "corpus.zip", "w") as archive:
        for document, text in zip(ids, texts):
            archive.writestr(document, text)
    hashes = {name: digest(input_path / name) for name in ("train.json", "corpus.zip")}
    lock_path = tmp_path / "project/configs/synthetic_split_lock.json"
    dump(lock_path, {"query_splits": {query["query_id"]: query["split"] for query in queries}})
    config = {
        "source": "source/input", "seed": 20261114, "head_words": 256,
        "near_duplicate_threshold": .9, "semantic_backend": "onnx", "lsa_dimensions": 3,
        "encoder_path": "project/models/synthetic", "document_policy": "head",
        "encoder_max_seq_length": 512, "encoder_threads": 1, "encoder_batch_size": 2,
        "top_k": 2, "max_proxy_negatives": 50, "negative_sampling": "random",
        "balance_classes": False, "training_seed": 20261114, "features": list(FEATURE_NAMES),
        "cv_folds": 2, "logistic_C": [1.], "rf_depth": [3], "rf_min_leaf": [1],
        "rf_trees": 3, "rf_prediction_jobs": 1, "rrf_constants": [60],
        "rrf_semantic_weights": [.5], "comparison_baselines": ["bm25", "rrf", "weighted_rrf"],
        "bootstrap_iterations": 20, "final_test_enabled": False,
        "split_lock_manifest": lock_path.relative_to(tmp_path).as_posix(),
        "additional_query_groups": [], "analysis_name": "head_onnx",
    }
    audit = {"dataset": "synthetic-only", "source_records": 6,
             "canonical_answerable_queries": 6, "independent_components": 6,
             "raw_documents": len(ids), "unique_documents": len(ids), "source_hashes": hashes,
             "split_policy": "historical_split_lock_with_component_quarantine",
             "quarantined_queries": [],
             "splits": {split: {"queries": 2, "groups": 2} for split in ("train", "dev", "test")}}
    dataset = {"document_ids": ids, "documents": texts, "queries": queries, "audit": audit}
    records, retrieval = [], []
    for number, query in enumerate(queries):
        candidates = np.asarray([number, 6, 7])
        features = np.zeros((3, len(FEATURE_NAMES)))
        features[:, 0] = [.9, .2, .1]
        latency = {"bm25_ms": 1., "semantic_ms": 2., "feature_ms": .5}
        records.append({"query": query, "candidates": candidates, "features": features,
                        "labels": np.asarray([1, 0, 0]), "bm25_order": np.asarray([number, 6]),
                        "semantic_order": np.asarray([7, number]), "latency": latency})
        retrieval.append({"query_id": query["query_id"], "bm25_order": [ids[number], "noise0"],
                          "semantic_order": ["noise1", ids[number]],
                          "candidate_ids": [ids[number], "noise0", "noise1"], "latency": latency})
    models = {name: {"model": FittedModelOnly(), "scaler": None,
                     "feature_indices": list(range(len(FEATURE_NAMES))),
                     "parameters": {"C": 1.} if name == "logistic" else {"max_depth": 3, "min_samples_leaf": 1},
                     "training_pairs": 6, "training_queries_without_positive": [], "cv_ndcg@10": .5}
              for name in ("logistic", "random_forest")}
    weighted = {"constant": 60, "semantic_weight": .5, "dev_ndcg@10": 1.}
    # Source evidence deliberately contains no synthetic heldout metric/ranking.
    rows = experiment.evaluate([record for record in records if record["query"]["split"] != "test"],
                               models, 60, dataset, weighted=weighted)
    dump(source / "configuration.json", config)
    dump(source / "data_audit.json", audit)
    dump(source / "split_manifest.json", queries)
    dump(source / "candidate_runs.json", retrieval)
    dump(source / "per_query_metrics.json", rows)
    experiment.store_sqlite(source / "experiments.sqlite", dataset, records, rows, source.name, config)
    bundle = {"bm25": SavedIndexOnly(), "semantic": SavedIndexOnly(), "models": models,
              "rrf_constant": 60, "weighted_rrf_selection": weighted,
              "document_ids": ids, "documents": texts, "configuration": config}
    joblib.dump(bundle, source / "search_bundle.joblib")
    result = {
        "run_id": source.name, "scope": "synthetic-exploratory",
        "configuration_sha256": digest(source / "configuration.json"),
        "data_audit": audit, "source_hashes": hashes, "source_sha256_at_run_start": {},
        "protocol_input_hashes": {config["split_lock_manifest"]: digest(lock_path)},
        "final_test_evaluated": False, "feature_names": list(FEATURE_NAMES),
        "rrf_constant": 60, "rrf_development_selection": [{"constant": 60, "dev_ndcg@10": 1.}],
        "weighted_rrf_selection": weighted, "weighted_rrf_development_grid": [weighted],
        "selected_models": {name: {key: value for key, value in model.items() if key not in ("model", "scaler")}
                            for name, model in models.items()},
        "cross_validation": [{"model": name, "parameters": model["parameters"],
                              "cv_ndcg@10": .5, "folds": 2, "queries": 2}
                             for name, model in models.items()],
        "summary": experiment.summarize(rows, models, config), "offline_index_seconds": 0.,
        "retrieval_metadata": {"bm25": bundle["bm25"].metadata, "semantic": bundle["semantic"].metadata},
        "runtime": {"synthetic": True}, "limitations": ["Synthetic test fixture, not TechQA results."],
    }
    dump(source / "results.json", result)
    dump(tracked / "results.json", result)
    with (tracked / "per_query_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["query_id", "group_id", "split", "method", "ndcg@10", "recall@20", "candidate_recall"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            for method in row["rankings"]:
                writer.writerow({name: row[name] for name in ("query_id", "group_id", "split", "candidate_recall")}
                                | {"method": method} | row[method])
    protocol = read(PROJECT / "configs/iteration2/heldout_protocol.json")
    protocol.update({"protocol": "Synthetic frozen holdout test only", "additional_query_groups": [],
                     "split_lock_manifest": config["split_lock_manifest"], "split_lock_sha256": digest(lock_path)})
    protocol_path = tmp_path / "project/configs/synthetic_heldout_protocol.json"
    dump(protocol_path, protocol)
    verifier = load_script("synthetic_freeze_fixture_verifier", "verify_experiment.py")
    check = verifier.verify_experiment(source, write=False)
    assert check["status"] == "passed", check
    return {"root": tmp_path, "source": source, "protocol": protocol_path,
            "lock": lock_path, "rows": deepcopy(rows), "verifier": verifier}


def freeze(frozen_module, fixture, **kwargs):
    return frozen_module.freeze_holdout(fixture["source"], fixture["protocol"],
                                       root=fixture["root"], **kwargs)


def test_default_freeze_never_loads_model_or_scores(frozen_module, synthetic_main, monkeypatch):
    before = source_hashes(synthetic_main["source"])
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Default freeze must not load models or score")
    monkeypatch.setattr(frozen_module.joblib, "load", forbidden)
    monkeypatch.setattr(frozen_module, "evaluate", forbidden)
    artifact, _ = freeze(frozen_module, synthetic_main)
    assert artifact.name.startswith("frozen_")
    assert read(artifact / "configuration.json")["final_test_enabled"] is True
    assert (artifact / "freeze_manifest.json").is_file()
    assert not (artifact / "per_query_metrics.json").exists()
    assert not (artifact / "results.json").exists()
    assert source_hashes(synthetic_main["source"]) == before


@pytest.mark.parametrize("target", ["results.json", "search_bundle.joblib", "candidate_runs.json"])
def test_changed_source_is_rejected_before_scoring(frozen_module, synthetic_main, monkeypatch, target):
    freeze(frozen_module, synthetic_main)
    path = synthetic_main["source"] / target
    path.write_bytes(path.read_bytes() + b"\n")
    monkeypatch.setattr(frozen_module, "evaluate", lambda *_args, **_kwargs: pytest.fail("Tampered source reached scoring"))
    with pytest.raises(frozen_module.FreezeError):
        freeze(frozen_module, synthetic_main, evaluate=True)


@pytest.mark.parametrize("target", ["protocol", "lock"])
def test_changed_protocol_or_split_lock_is_rejected(frozen_module, synthetic_main, monkeypatch, target):
    freeze(frozen_module, synthetic_main)
    path = synthetic_main[target]
    value = read(path)
    value["tampered_after_freeze"] = True
    dump(path, value)
    monkeypatch.setattr(frozen_module, "evaluate", lambda *_args, **_kwargs: pytest.fail("Changed identity reached scoring"))
    with pytest.raises(frozen_module.FreezeError):
        freeze(frozen_module, synthetic_main, evaluate=True)


def test_evaluation_reuses_models_preserves_source_and_is_once_only(frozen_module, synthetic_main, monkeypatch):
    before = source_hashes(synthetic_main["source"])
    artifact, _ = freeze(frozen_module, synthetic_main)
    original_evaluate = frozen_module.evaluate
    calls = []
    def test_records_only(records, *args, **kwargs):
        assert {record["query"]["split"] for record in records} == {"test"}
        calls.append([record["query"]["query_id"] for record in records])
        return original_evaluate(records, *args, **kwargs)
    monkeypatch.setattr(frozen_module, "evaluate", test_records_only)
    monkeypatch.setattr(experiment, "fit_model", lambda *_args, **_kwargs: pytest.fail("Holdout refit a model"))
    monkeypatch.setattr(experiment, "select_models", lambda *_args, **_kwargs: pytest.fail("Holdout reran CV"))
    artifact2, result = freeze(frozen_module, synthetic_main, evaluate=True)
    assert artifact2 == artifact
    assert result["final_test_evaluated"] is True
    assert set(result["summary"]) == {"train", "dev", "test"}
    saved = read(artifact / "per_query_metrics.json")
    assert [row for row in saved if row["split"] != "test"] == synthetic_main["rows"]
    assert len([row for row in saved if row["split"] == "test"]) == 2
    frozen_config = read(artifact / "configuration.json")
    assert joblib.load(artifact / "search_bundle.joblib")["configuration"] == frozen_config
    check = synthetic_main["verifier"].verify_experiment(artifact, write=False)
    assert check["status"] == "passed", check
    assert source_hashes(synthetic_main["source"]) == before
    repeat_artifact, repeat_result = freeze(frozen_module, synthetic_main, evaluate=True)
    assert repeat_artifact == artifact and repeat_result == result
    assert len(calls) == 1


def test_resume_reuses_predictions_after_later_write_failure(frozen_module, synthetic_main, monkeypatch):
    before = source_hashes(synthetic_main["source"])
    artifact, _ = freeze(frozen_module, synthetic_main)
    original_evaluate = frozen_module.evaluate
    original_store = frozen_module.store_sqlite
    calls = []
    def count_predictions(*args, **kwargs):
        calls.append(1)
        return original_evaluate(*args, **kwargs)
    def broken_write(*_args, **_kwargs):
        raise OSError("synthetic write failure after predictions")
    monkeypatch.setattr(frozen_module, "evaluate", count_predictions)
    monkeypatch.setattr(frozen_module, "store_sqlite", broken_write)
    with pytest.raises((OSError, frozen_module.FreezeError), match="synthetic write failure"):
        freeze(frozen_module, synthetic_main, evaluate=True)
    assert (artifact / "test_predictions.json").is_file()
    assert len(calls) == 1
    with pytest.raises(frozen_module.FreezeError):
        freeze(frozen_module, synthetic_main, evaluate=True)
    monkeypatch.setattr(frozen_module, "store_sqlite", original_store)
    _, result = freeze(frozen_module, synthetic_main, evaluate=True, resume=True)
    assert result["final_test_evaluated"] is True
    assert len(calls) == 1
    assert source_hashes(synthetic_main["source"]) == before


def test_resume_rejects_changed_source_identity(frozen_module, synthetic_main, monkeypatch):
    freeze(frozen_module, synthetic_main)
    monkeypatch.setattr(frozen_module, "store_sqlite", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("synthetic write failure")))
    with pytest.raises((OSError, frozen_module.FreezeError)):
        freeze(frozen_module, synthetic_main, evaluate=True)
    path = synthetic_main["source"] / "candidate_runs.json"
    path.write_bytes(path.read_bytes() + b"\n")
    monkeypatch.setattr(frozen_module, "evaluate", lambda *_args, **_kwargs: pytest.fail("Changed resume identity reached scoring"))
    with pytest.raises(frozen_module.FreezeError):
        freeze(frozen_module, synthetic_main, evaluate=True, resume=True)
