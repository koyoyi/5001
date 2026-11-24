"""Artifact-tampering checks against the independent experiment verifier."""

from contextlib import closing
import csv
import importlib.util
import json
import math
from pathlib import Path
import sqlite3
import zipfile

import pytest


VERIFIER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "verify_experiment.py"
SPEC = importlib.util.spec_from_file_location("verify_experiment", VERIFIER_PATH)
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


def dump(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


@pytest.fixture
def evidence(tmp_path):
    artifact = tmp_path / "project" / "artifacts" / "fixture_lsa"
    tracked = tmp_path / "project" / "results" / artifact.name
    source = tmp_path / "source" / "input"
    for directory in (artifact, tracked, source):
        directory.mkdir(parents=True)
    questions = [{"query_id": f"q{i}", "question": f"diagnostic case {i}",
                  "group_id": f"g{i}", "split": ("train", "dev", "test")[i // 2],
                  "positive_document_ids": [f"d{i}"]} for i in range(6)]
    dump(source / "train.json", [{"id": q["query_id"]} for q in questions])
    with zipfile.ZipFile(source / "corpus.zip", "w") as archive:
        for document in [f"d{i}" for i in range(6)] + ["noise"]:
            archive.writestr(document, document + " reference text")
    config = {"source": "source/input", "top_k": 2, "final_test_enabled": False}
    dump(artifact / "configuration.json", config)
    hashes = {name: verifier.digest(source / name) for name in ("train.json", "corpus.zip")}
    audit = {"source_records": 6, "canonical_answerable_queries": 6,
             "independent_components": 6, "unique_documents": 7,
             "splits": {split: {"queries": 2, "groups": 2} for split in ("train", "dev", "test")},
             "source_hashes": hashes}
    dump(artifact / "data_audit.json", audit)
    dump(artifact / "split_manifest.json", questions)
    rows = []
    retrieval = []
    for query in questions:
        qid = query["query_id"]
        gold = query["positive_document_ids"][0]
        retrieval.append({"query_id": qid, "bm25_order": [gold, "noise"],
                          "semantic_order": ["noise", gold], "candidate_ids": [gold, "noise"]})
        if query["split"] == "test":
            continue
        rankings = {method: (["noise", gold] if method == "semantic" else [gold, "noise"])
                    for method in verifier.METHODS}
        rows.append({"query_id": qid, "group_id": query["group_id"], "split": query["split"],
                     "candidate_count": 2, "candidate_recall": 1., "rankings": rankings, "latency": {},
                     **{method: {"ndcg@10": 1 / math.log2(3) if method == "semantic" else 1.,
                                 "recall@20": 1.} for method in verifier.METHODS}})
    dump(artifact / "per_query_metrics.json", rows)
    dump(artifact / "candidate_runs.json", {"queries": retrieval})
    summary = {split: {"queries": 2, "groups": 2, "candidate_recall": 1.,
                       "methods": {method: rows[0][method] for method in verifier.METHODS},
                       "paired_comparisons": {}, "latency_single_observation": {}}
               for split in ("train", "dev")}
    result = {"run_id": artifact.name, "data_audit": audit,
              "configuration_sha256": verifier.digest(artifact / "configuration.json"),
              "source_hashes": hashes, "final_test_evaluated": False, "summary": summary}
    dump(artifact / "results.json", result)
    dump(tracked / "results.json", result)
    with closing(sqlite3.connect(artifact / "experiments.sqlite")) as database, database:
        database.executescript("""
            CREATE TABLE experiment_runs(run_id TEXT, configuration TEXT);
            CREATE TABLE passages(document_id TEXT PRIMARY KEY);
            CREATE TABLE queries(query_id TEXT PRIMARY KEY,question TEXT,split TEXT,group_id TEXT);
            CREATE TABLE qrels(query_id TEXT,document_id TEXT);
            CREATE TABLE candidates(query_id TEXT,document_id TEXT,observed_positive INTEGER);
            CREATE TABLE rankings(query_id TEXT,method TEXT,rank INTEGER,document_id TEXT);
        """)
        database.execute("INSERT INTO experiment_runs VALUES (?,?)", (artifact.name, json.dumps(config)))
        database.executemany("INSERT INTO passages VALUES (?)", [(f"d{i}",) for i in range(6)] + [("noise",)])
        for query in questions:
            qid, gold = query["query_id"], query["positive_document_ids"][0]
            database.execute("INSERT INTO queries VALUES (?,?,?,?)", (qid, query["question"], query["split"], query["group_id"]))
            database.execute("INSERT INTO qrels VALUES (?,?)", (qid, gold))
            database.executemany("INSERT INTO candidates VALUES (?,?,?)", [(qid, gold, 1), (qid, "noise", 0)])
        for row in rows:
            for method, ranking in row["rankings"].items():
                database.executemany("INSERT INTO rankings VALUES (?,?,?,?)", [(row["query_id"], method, rank, doc) for rank, doc in enumerate(ranking, 1)])
    with (tracked / "per_query_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["query_id", "group_id", "split", "method", "ndcg@10", "recall@20", "candidate_recall"])
        writer.writeheader()
        for row in rows:
            for method in verifier.METHODS:
                writer.writerow({key: row[key] for key in ("query_id", "group_id", "split", "candidate_recall")} | {"method": method} | row[method])
    return artifact


def test_independent_verifier_accepts_consistent_saved_evidence(evidence):
    report = verifier.verify_experiment(evidence)
    assert report["status"] == "passed", report
    assert report["checks"]["exact_candidate_union"] == "saved_retrieval_runs"
    assert report["checks"]["tracked_csv"] == "passed"
    assert report["counts"]["evaluated_queries"] == 4
    assert json.loads((evidence / "verification.json").read_text()) == report
    mirror = evidence.parents[1] / "results" / evidence.name / "verification.json"
    assert json.loads(mirror.read_text()) == report


def test_verifier_rejects_metric_tampering(evidence):
    rows = verifier.read_json(evidence / "per_query_metrics.json")
    rows[0]["semantic"]["ndcg@10"] = 1.
    dump(evidence / "per_query_metrics.json", rows)
    report = verifier.verify_experiment(evidence, write=False)
    assert report["status"] == "failed"
    assert "ndcg@10" in report["error"]


def test_verifier_rejects_candidate_union_tampering_within_budget(evidence):
    with closing(sqlite3.connect(evidence / "experiments.sqlite")) as database, database:
        database.execute("INSERT INTO candidates VALUES ('q0','d5',0)")
    report = verifier.verify_experiment(evidence, write=False)
    assert report["status"] == "failed"
    assert "exact retriever union" in report["error"]


def test_verifier_rejects_candidate_label_tampering(evidence):
    with closing(sqlite3.connect(evidence / "experiments.sqlite")) as database, database:
        database.execute("UPDATE candidates SET observed_positive=1 WHERE query_id='q0' AND document_id='noise'")
    report = verifier.verify_experiment(evidence, write=False)
    assert report["status"] == "failed"
    assert "relevance label" in report["error"]


def test_verifier_rejects_held_out_rankings(evidence):
    with closing(sqlite3.connect(evidence / "experiments.sqlite")) as database, database:
        database.execute("INSERT INTO rankings VALUES ('q4','bm25',1,'d4')")
    report = verifier.verify_experiment(evidence, write=False)
    assert report["status"] == "failed"
    assert "Held-out" in report["error"]


def test_missing_retrieval_evidence_is_explicitly_unverified(evidence):
    (evidence / "candidate_runs.json").unlink()
    report = verifier.verify_experiment(evidence, write=False)
    assert report["status"] == "passed", report
    assert report["checks"]["exact_candidate_union"] == "unverified_no_complete_retrieval_runs"
    assert any("not proven" in warning for warning in report["warnings"])


def test_verifier_rejects_cross_split_component(evidence):
    manifest = verifier.read_json(evidence / "split_manifest.json")
    manifest[2]["group_id"] = manifest[0]["group_id"]
    dump(evidence / "split_manifest.json", manifest)
    report = verifier.verify_experiment(evidence, write=False)
    assert report["status"] == "failed"
    assert "component crosses splits" in report["error"]
