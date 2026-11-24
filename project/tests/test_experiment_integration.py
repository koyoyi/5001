"""One small offline run guards experiment, evidence and test-isolation wiring."""

from collections import defaultdict
from contextlib import closing
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sqlite3
import sys
import zipfile

import pytest
from threadpoolctl import threadpool_limits


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from ragstudy.experiment import run
from ragstudy.search import load_bundle, search


@pytest.mark.parametrize("weighted", [False, True])
def test_small_experiment_keeps_test_locked_and_preserves_citations(tmp_path, weighted):
    """Exercise actual ZIP/JSON audit, CV, SQLite, search and independent QA."""
    source = tmp_path / "source" / "input"
    source.mkdir(parents=True)
    terms = ["cuda", "bond", "packet", "kernel", "router", "cache",
             "driver", "schema", "module", "thread", "tensor", "volume"]
    originals = {}
    records = []
    for number, term in enumerate(terms):
        document_id = f"doc{number:02d}.txt"
        text = (f"{term} uniqueproduct{number} service repair configuration error "
                "solution diagnostic support manual")
        originals[document_id] = text
        records.append({"id": f"q{number:02d}",
                        "question": f"{term} uniqueproduct{number}",
                        "answer": "UNUSED_REFERENCE_ANSWER_MUST_NOT_ENTER_THE_CORPUS",
                        "is_impossible": False,
                        "contexts": [{"filename": document_id, "text": text}]})
    for number in range(6):
        originals[f"noise{number:02d}.txt"] = (
            f"Unrelated noise{number} archive entry notes technical support manual")
    with zipfile.ZipFile(source / "corpus.zip", "w") as archive:
        for document_id, text in originals.items():
            archive.writestr(document_id, text)
    (source / "train.json").write_text(json.dumps(records), encoding="utf-8")
    manifest = {"files": [{
        "local_file": "input/" + name,
        "repo": "synthetic-offline-fixture",
        "revision": "fixture-v1",
        "sha256": hashlib.sha256((source / name).read_bytes()).hexdigest(),
    } for name in ("train.json", "corpus.zip")]}
    (source.parent / "download_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    # The fixture workspace includes provenance inputs because run() hashes its
    # workspace's entry point. Actual executable modules remain the real code.
    fixture_script = tmp_path / "project" / "scripts" / "run_experiment.py"
    fixture_script.parent.mkdir(parents=True)
    fixture_script.write_text("# Offline integration provenance fixture.\n", encoding="utf-8")
    fixture_module = tmp_path / "project" / "src" / "ragstudy" / "__init__.py"
    fixture_module.parent.mkdir(parents=True)
    fixture_module.write_text('"""Offline integration provenance fixture."""\n', encoding="utf-8")
    config = {
        "source": "source/input", "seed": 12, "head_words": 256,
        "near_duplicate_threshold": .99999, "semantic_backend": "lsa",
        "lsa_dimensions": 3, "top_k": 4, "max_proxy_negatives": 3,
        "cv_folds": 2, "logistic_C": [1.], "rf_depth": [3],
        "rf_min_leaf": [1], "rf_trees": 5, "rrf_constants": [60],
        "bootstrap_iterations": 20, "final_test_enabled": False,
    }
    if weighted:
        config.update({"document_policy": "full", "rrf_semantic_weights": [0., .5, 1.],
                       "comparison_baselines": ["bm25", "rrf", "weighted_rrf"]})
    with pytest.raises(ValueError, match="Final test"):
        run(tmp_path, config, include_test=True)
    assert not (tmp_path / "project" / "artifacts").exists()

    # Exactly one tiny experiment; no downloaded model and no network access.
    with threadpool_limits(limits=2):
        artifact, result = run(tmp_path, config)
    assert result["data_audit"]["unique_documents"] == 18
    assert result["data_audit"]["independent_components"] == 12
    assert result["data_audit"]["splits"] == {
        "train": {"queries": 8, "groups": 8},
        "dev": {"queries": 2, "groups": 2},
        "test": {"queries": 2, "groups": 2},
    }
    assert result["final_test_evaluated"] is False
    assert ("weighted_rrf" in result["summary"]["dev"]["methods"]) == weighted
    if weighted:
        assert result["summary"]["dev"]["paired_contrasts"]
        assert result["data_audit"]["document_policy"] == "full"
    assert set(result["summary"]) == {"train", "dev"}
    queries = json.loads((artifact / "split_manifest.json").read_text(encoding="utf-8"))
    group_splits, gold_splits = defaultdict(set), defaultdict(set)
    for query in queries:
        group_splits[query["group_id"]].add(query["split"])
        for document_id in query["positive_document_ids"]:
            gold_splits[document_id].add(query["split"])
    assert all(len(splits) == 1 for splits in group_splits.values())
    assert all(len(splits) == 1 for splits in gold_splits.values())
    with closing(sqlite3.connect(artifact / "experiments.sqlite")) as database:
        assert database.execute("SELECT COUNT(*) FROM queries").fetchone()[0] == 12
        assert database.execute("""
            SELECT COUNT(*) FROM rankings JOIN queries USING(query_id)
            WHERE split='test'
        """).fetchone()[0] == 0
        evaluated = {row[0] for row in database.execute("SELECT DISTINCT query_id FROM rankings")}
    assert evaluated == {query["query_id"] for query in queries if query["split"] != "test"}

    specification = importlib.util.spec_from_file_location(
        "verify_experiment_integration", PROJECT / "scripts" / "verify_experiment.py")
    verifier = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(verifier)
    verification = verifier.verify_experiment(artifact)
    assert verification["status"] == "passed", verification
    assert verification["checks"]["exact_candidate_union"] == "saved_retrieval_runs"
    assert verification["counts"]["evaluated_queries"] == 10

    bundle = load_bundle(artifact)
    assert not any("UNUSED_REFERENCE_ANSWER" in text for text in bundle["documents"])
    for method in ("bm25", "semantic", "rrf", "logistic", "random_forest"):
        with threadpool_limits(limits=2):
            response = search(bundle, "cuda uniqueproduct0", method=method, limit=3)
        assert 1 <= len(response["evidence"]) <= 3
        assert response["candidate_count"] <= 2 * config["top_k"]
        assert "not a generated answer" in response["mode"]
        assert all(math.isfinite(value) and value >= 0 for value in response["timing_ms"].values())
        for evidence in response["evidence"]:
            document_id = evidence["document_id"]
            assert document_id in originals
            assert evidence["citation"] == f"[{document_id}]"
            assert evidence["excerpt"] in originals[document_id]
            assert math.isfinite(evidence["ranking_score"])
