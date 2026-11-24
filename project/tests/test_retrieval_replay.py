"""Small real replay runs guard candidate reuse and fixed-cohort ablations."""

from contextlib import closing
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
import zipfile

import numpy as np
import pytest
from threadpoolctl import threadpool_limits


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from ragstudy import experiment
from ragstudy.data import load_techqa
from ragstudy.features import FEATURE_NAMES, sample_training_rows
from ragstudy.replay import replay_retrieval


def import_script(name, filename):
    specification = importlib.util.spec_from_file_location(name, PROJECT / "scripts" / filename)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


ablations = import_script("training_ablations_test", "run_training_ablations.py")
verifier = import_script("replay_verifier_test", "verify_experiment.py")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def snapshot(directory):
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in directory.iterdir() if path.is_file()}


def load_dataset(root, config):
    lock = json.loads((root / config["split_lock_manifest"]).read_text())["query_splits"]
    return load_techqa(root / config["source"], seed=config["seed"], head_words=config["head_words"],
                       near_duplicate_threshold=config["near_duplicate_threshold"],
                       document_policy=config["document_policy"],
                       additional_query_groups=config["additional_query_groups"], split_lock=lock)


@pytest.fixture(scope="module")
def completed_source(tmp_path_factory):
    root = tmp_path_factory.mktemp("retrieval_replay")
    source = root / "source/input"
    source.mkdir(parents=True)
    terms = ["cuda", "bond", "packet", "kernel", "router", "cache",
             "driver", "schema", "module", "thread", "tensor", "volume"]
    rows = []
    with zipfile.ZipFile(source / "corpus.zip", "w") as archive:
        for index, term in enumerate(terms):
            text = f"{term} uniqueproduct{index} service configuration diagnostic manual"
            document = f"doc{index:02}.txt"
            archive.writestr(document, text)
            rows.append({"id": f"q{index:02}", "question": f"{term} uniqueproduct{index}",
                         "is_impossible": False, "contexts": [{"filename": document, "text": text}]})
        for index in range(6):
            archive.writestr(f"noise{index:02}.txt", f"Unrelated noise{index} technical support notes")
    write_json(source / "train.json", rows)
    write_json(source.parent / "download_manifest.json", {"files": [
        {"local_file": "input/" + name, "revision": "fixture-v2", "repo": "offline-fixture",
         "sha256": hashlib.sha256((source / name).read_bytes()).hexdigest()}
        for name in ("train.json", "corpus.zip")]})
    initial = load_techqa(source, seed=12, head_words=256, near_duplicate_threshold=.99999)
    lock = {query["query_id"]: query["split"] for query in initial["queries"]}
    reviewed = [next(query["query_id"] for query in initial["queries"] if query["split"] == split)
                for split in ("train", "dev")]
    write_json(root / "project/configs/fixture_lock.json", {"query_splits": lock})
    script = root / "project/scripts/run_experiment.py"
    script.parent.mkdir(parents=True)
    script.write_text("# Offline replay provenance fixture.\n", encoding="utf-8")
    config = {"source": "source/input", "seed": 12, "training_seed": 20261114,
              "head_words": 256, "document_policy": "head", "near_duplicate_threshold": .99999,
              "semantic_backend": "lsa", "lsa_dimensions": 3, "top_k": 4,
              "max_proxy_negatives": 50, "cv_folds": 2, "logistic_C": [1.],
              "rf_depth": [3], "rf_min_leaf": [1], "rf_trees": 5,
              "rrf_constants": [60], "bootstrap_iterations": 20, "final_test_enabled": False,
              "split_lock_manifest": "project/configs/fixture_lock.json",
              "additional_query_groups": [reviewed]}
    with threadpool_limits(limits=2):
        artifact, result = experiment.run(root, config)
    assert len(result["data_audit"]["quarantined_queries"]) == 2
    assert result["data_audit"]["split_policy"] == "historical_split_lock_with_component_quarantine"
    return root, artifact, config, result


def test_fixed_parameter_replay_keeps_candidates_rankings_and_metrics(completed_source, monkeypatch):
    root, source, base, result = completed_source
    before = snapshot(source)
    relative = source.relative_to(root).as_posix()
    config = deepcopy(base)
    config.update({"fixed_model_parameters": {kind: result["selected_models"][kind]["parameters"]
                                               for kind in ("logistic", "random_forest")},
                   "reuse_retrieval_run": relative, "model_source_run": relative})

    def forbid_new_retrieval(*args, **kwargs):
        raise AssertionError("A replay must not regenerate retrieval candidates")

    monkeypatch.setattr(experiment, "make_records", forbid_new_retrieval)
    with threadpool_limits(limits=2):
        replayed, replay_result = experiment.run(root, config)
    assert replay_result["cross_validation"] == []
    assert replay_result["data_audit"] == result["data_audit"]
    assert replay_result["final_test_evaluated"] is False
    assert (replayed / "candidate_runs.json").read_bytes() == (source / "candidate_runs.json").read_bytes()
    assert (replayed / "split_manifest.json").read_bytes() == (source / "split_manifest.json").read_bytes()
    original_rows = json.loads((source / "per_query_metrics.json").read_text())
    replay_rows = json.loads((replayed / "per_query_metrics.json").read_text())
    assert len(original_rows) == len(replay_rows)
    for original, observed in zip(original_rows, replay_rows):
        for name in ("query_id", "group_id", "split", "candidate_count", "candidate_recall", "rankings",
                     "bm25", "semantic", "rrf", "logistic", "random_forest"):
            assert observed[name] == original[name]
    with closing(sqlite3.connect(source / "experiments.sqlite")) as old, closing(sqlite3.connect(replayed / "experiments.sqlite")) as new:
        query = "SELECT query_id,document_id,observed_positive,features FROM candidates ORDER BY query_id,document_id"
        assert old.execute(query).fetchall() == new.execute(query).fetchall()
        assert new.execute("SELECT COUNT(*) FROM rankings JOIN queries USING(query_id) WHERE split='test'").fetchone()[0] == 0
    check = verifier.verify_experiment(replayed)
    assert check["status"] == "passed", check
    assert snapshot(source) == before


def test_six_declared_training_changes_leave_raw_retrieval_and_split_unchanged(completed_source):
    root, source, base, result = completed_source
    before = snapshot(source)
    original_dataset = load_dataset(root, base)
    _, _, original_records, _ = replay_retrieval(source, original_dataset, base)
    for name in ablations.VARIANTS:
        config = ablations.build_variant_config(base, result, name, source.relative_to(root).as_posix())
        assert config["fixed_model_parameters"] == {
            kind: result["selected_models"][kind]["parameters"] for kind in ("logistic", "random_forest")}
        assert config["final_test_enabled"] is False
        dataset = load_dataset(root, config)
        assert dataset["queries"] == original_dataset["queries"]
        _, _, observed, provenance = replay_retrieval(source, dataset, config)
        assert provenance["source_results_sha256"] == before["results.json"]
        for expected, actual in zip(original_records, observed):
            assert actual["query"] == expected["query"]
            for key in ("candidates", "features", "labels", "bm25_order", "semantic_order"):
                np.testing.assert_array_equal(actual[key], expected[key])
    rank_only = ablations.build_variant_config(base, result, "rank_features_only", "source")
    assert rank_only["features"] == [FEATURE_NAMES[index] for index in (2, 3, 4, 5)]
    assert snapshot(source) == before


@pytest.mark.parametrize("change", [{"top_k": 5}, {"document_policy": "full"}, {"head_words": 128}])
def test_replay_rejects_changed_retrieval_policy(completed_source, change):
    root, source, base, _ = completed_source
    before = snapshot(source)
    changed = {**base, **change}
    dataset = load_dataset(root, changed)
    with pytest.raises(ValueError, match="identical audited cohort|Retrieval configuration differs"):
        replay_retrieval(source, dataset, changed)
    assert snapshot(source) == before


def test_configuration_hash_and_builder_refuse_silent_baseline_drift(completed_source):
    _, _, base, result = completed_source
    config = ablations.build_variant_config(base, result, "negative_cap10", "source")
    assert ablations.configuration_hash(config) == ablations.configuration_hash(dict(reversed(list(config.items()))))
    assert ablations.configuration_hash(config) != ablations.configuration_hash({**config, "max_proxy_negatives": 11})
    with pytest.raises(ValueError, match="unmodified training baseline"):
        ablations.build_variant_config({**base, "balance_classes": True}, result, "negative_cap10", "source")
    with pytest.raises(ValueError, match="final test locked"):
        ablations.build_variant_config({**base, "final_test_enabled": True}, result, "negative_cap10", "source")


def test_suite_dry_run_checks_resume_configuration_before_skipping(completed_source, monkeypatch):
    root, source, base, result = completed_source
    # The production ONNX/cohort gate is separate from this offline replay
    # fixture. Exercise actual persistent resume logic without six extra fits.
    monkeypatch.setattr(ablations, "validate_main_source", lambda *args: (deepcopy(base), deepcopy(result)))
    write_json(root / "project/results/iteration2_plan.json",
               {"robustness_one_factor_at_a_time": list(ablations.VARIANTS)})
    summary_path = root / "project/results/fixture_ablations.json"
    summary = ablations.run_ablations(source, summary_path, dry_run=True, root=root)
    assert summary["status"] == "planned"
    assert list(summary["variants"]) == list(ablations.VARIANTS)
    assert all(row["state"] == "pending" for row in summary["variants"].values())
    summary["variants"]["negative_cap10"]["configuration"]["max_proxy_negatives"] = 11
    write_json(summary_path, summary)
    with pytest.raises(ValueError, match="Resume configuration hash mismatch"):
        ablations.run_ablations(source, summary_path, dry_run=True, root=root)


@pytest.mark.parametrize("balanced", [False, True])
def test_negative_caps_preserve_fixed_total_query_weight(balanced):
    records = []
    for index, positives in enumerate((1, 2), 1):
        features = np.zeros((positives + 70, len(FEATURE_NAMES)))
        features[:, 0] = index
        records.append({"query": {"query_id": f"q{index}", "group_id": f"g{index}"},
                        "features": features, "labels": np.asarray([1] * positives + [0] * 70)})
    totals = []
    for cap in (10, 50):
        features, labels, weights, _, _ = sample_training_rows(
            records, max_negatives=cap, balance_classes=balanced, query_weight_total=51)
        totals.append([weights[features[:, 0] == index].sum() for index in (1, 2)])
        for index in (1, 2):
            query = features[:, 0] == index
            assert np.isclose(weights[query].sum(), 51.)
            if balanced:
                assert np.isclose(weights[query & (labels == 1)].sum(), 25.5)
                assert np.isclose(weights[query & (labels == 0)].sum(), 25.5)
    np.testing.assert_allclose(totals[0], totals[1])


@pytest.mark.parametrize("target", [0, -1, float("nan"), float("inf")])
def test_invalid_fixed_query_weight_target_is_rejected(target):
    record = {"query": {"query_id": "q", "group_id": "g"},
              "features": np.zeros((2, len(FEATURE_NAMES))), "labels": np.asarray([1, 0])}
    with pytest.raises(ValueError, match="finite and positive"):
        sample_training_rows([record], query_weight_total=target)
