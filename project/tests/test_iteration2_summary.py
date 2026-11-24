"""Report-only gates: verified source hashes, fixed partitions and closed test."""

import csv
import importlib.util
import json
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("iteration2_summary_test", PROJECT / "scripts/summarize_iteration2.py")
summary_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary_module)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


@pytest.fixture
def source_runs(tmp_path):
    manifest = [{"query_id": f"q{i}", "split": ("train", "dev", "test")[i // 2]}
                for i in range(6)]
    for pipeline, backend, policy in (("head_lsa", "lsa", "head"), ("full_lsa", "lsa", "full"),
                                       ("main_onnx", "onnx", "head")):
        artifact = tmp_path / "project/artifacts" / pipeline
        tracked = tmp_path / "project/results" / pipeline
        artifact.mkdir(parents=True)
        tracked.mkdir(parents=True)
        config = {"semantic_backend": backend, "document_policy": policy, "seed": 1,
                  "bootstrap_iterations": 20, "rf_trees": 5, "max_proxy_negatives": 50,
                  "final_test_enabled": False, "training_seed": 20261114,
                  "analysis_name": "head_onnx" if backend == "onnx" else pipeline}
        dump(artifact / "configuration.json", config)
        dump(artifact / "split_manifest.json", manifest)
        audit = {"split_policy": "historical_split_lock_with_component_quarantine", "raw_documents": 8,
                 "unique_documents": 8, "canonical_queries_before_quarantine": 6,
                 "canonical_answerable_queries": 6, "quarantined_queries": [],
                 "splits": {split: {"queries": 2, "groups": 2} for split in ("train", "dev", "test")}}
        dump(artifact / "data_audit.json", audit)
        rows = []
        for index in range(2):
            metrics = {method: {"ndcg@10": .2 + .05 * number + .1 * index,
                                "recall@20": .3 + .05 * number + .1 * index}
                       for number, method in enumerate(("bm25", "semantic", "rrf", "weighted_rrf", "logistic", "random_forest"))}
            rows.append({"query_id": f"q{index+2}", "group_id": f"g{index}", "split": "dev",
                         "candidate_recall": .8, "rankings": {method: [] for method in metrics}, **metrics})
        dump(artifact / "per_query_metrics.json", rows)
        result = {"run_id": pipeline, "final_test_evaluated": False,
                  "configuration_sha256": summary_module.sha256(artifact / "configuration.json"),
                  "source_hashes": {"train.json": "synthetic-source", "corpus.zip": "synthetic-corpus"},
                  "data_audit": audit, "selected_models": {
                      "logistic": {"parameters": {"C": 1.}},
                      "random_forest": {"parameters": {"max_depth": 3, "min_samples_leaf": 1}}},
                  "weighted_rrf_selection": {"constant": 60, "semantic_weight": .5},
                  "retrieval_metadata": {"semantic": {"precision": "float32", "model_revision": "fixture-only"}},
                  "summary": {"dev": {"queries": 2, "groups": 2, "candidate_recall": .8,
                      "methods": {method: {metric: sum(row[method][metric] for row in rows)/2
                                           for metric in ("ndcg@10", "recall@20")} for method in rows[0]["rankings"]}}}}
        dump(artifact / "results.json", result)
        dump(tracked / "results.json", result)
        dump(artifact / "candidate_runs.json", [])
        (artifact / "experiments.sqlite").write_bytes(b"verified-evidence-fixture-only")
        proof = {"status": "passed", "run_id": pipeline, "checks": {"exact_candidate_union": "saved_retrieval_runs"},
                 "evidence_sha256": {path.name: summary_module.sha256(path) for path in artifact.iterdir() if path.is_file()}}
        dump(artifact / "verification.json", proof)
        dump(tracked / "verification.json", proof)
    return tmp_path


def render(root, **kwargs):
    return summary_module.summarize("project/artifacts/main_onnx", root=root,
                                    head_lsa="head_lsa", full_lsa="full_lsa", **kwargs)


def test_verified_summary_exports_exact_values_without_touching_source(source_runs):
    root = source_runs
    paths = list((root / "project/artifacts").glob("*/*"))
    before = {str(path): summary_module.sha256(path) for path in paths if path.is_file()}
    result = render(root)
    assert result["main_pipeline"] == "head_onnx"
    assert result["holdout"]["test_metrics_read"] is False
    assert len(result["development"]["methods"]) == 18
    assert len(result["development"]["contrasts"]) == 24
    assert len(result["development"]["per_query_metrics"]) == 36
    assert before == {str(path): summary_module.sha256(path) for path in paths if path.is_file()}
    output = root / "project/results"
    assert (output / "iteration2_development_contrasts.png").stat().st_size > 1000
    markdown = (output / "iteration2_summary.md").read_text(encoding="utf-8")
    assert "BGE FP32" in markdown and "not_requested" in markdown
    with (output / "iteration2_development_methods.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert float(rows[0]["ndcg@10"]) == result["development"]["methods"][0]["ndcg@10"]


def test_different_verified_partition_rejects_direct_comparison(source_runs):
    root = source_runs
    artifact = root / "project/artifacts/full_lsa"
    manifest = summary_module.read_json(artifact / "split_manifest.json")
    manifest[0]["split"] = "dev"
    dump(artifact / "split_manifest.json", manifest)
    proof = summary_module.read_json(artifact / "verification.json")
    proof["evidence_sha256"]["split_manifest.json"] = summary_module.sha256(artifact / "split_manifest.json")
    dump(artifact / "verification.json", proof)
    with pytest.raises(ValueError, match="byte-identical"):
        render(root, write=False)


def test_prepared_holdout_does_not_read_unfinished_test_results(source_runs):
    root = source_runs
    artifact = root / "project/artifacts/frozen_fixture"
    artifact.mkdir()
    identity = {"source_artifact": "project/artifacts/main_onnx"}
    registration = {"run_id": artifact.name, "identity_sha256": "fixture-identity", "identity": identity}
    dump(root / "project/artifacts/frozen_holdout_registry.json", registration)
    dump(artifact / "freeze_manifest.json", {"identity_sha256": "fixture-identity", "identity": identity,
                                             "status": "prepared", "heldout_scored": False})
    (artifact / "results.json").write_text("MUST NOT PARSE UNFINISHED TEST RESULTS", encoding="utf-8")
    result = render(root, include_holdout=True, write=False)
    assert result["holdout"]["status"] == "prepared"
    assert result["holdout"]["test_metrics_read"] is False
