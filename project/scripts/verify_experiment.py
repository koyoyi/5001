"""Independently verify local experiment evidence without rerunning model training.

Usage: python project/scripts/verify_experiment.py ARTIFACT_DIR
Add --recompute-candidates only for a trusted bundle produced by this project.
That option loads a local joblib pickle and reruns retrieval, never training or
held-out metric selection. Without retrieval runs or this option, candidate
union provenance is explicitly reported as unverified.
"""

from collections import defaultdict
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import sqlite3
import sys


SOURCE_ROOT = Path(__file__).resolve().parents[2]
METHODS = ("bm25", "semantic", "rrf", "logistic", "random_forest")


class VerificationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise VerificationError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def close_number(actual, expected, message):
    require(math.isfinite(float(actual)) and math.isfinite(float(expected))
            and math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-12),
            f"{message}: {actual!r} != {expected!r}")


def independent_metrics(ranking, gold):
    """Binary DCG and recall, using every gold ID for the ideal ranking."""
    require(len(ranking) == len(set(ranking)), "Duplicate document in a saved ranking")
    gold = set(gold)
    require(bool(gold), "Evaluated answerable query has no known gold document")
    dcg = math.fsum(1 / math.log2(rank + 2)
                    for rank, doc in enumerate(ranking[:10]) if doc in gold)
    ideal = math.fsum(1 / math.log2(rank + 2) for rank in range(min(10, len(gold))))
    return {"ndcg@10": dcg / ideal,
            "recall@20": len(set(ranking[:20]) & gold) / len(gold)}


def percentile(values, quantile):
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def locate(artifact):
    artifact = Path(artifact).expanduser().resolve()
    require(artifact.is_dir(), "Artifact directory does not exist")
    require(artifact.parent.name == "artifacts" and artifact.parent.parent.name == "project",
            "Expected WORKSPACE/project/artifacts/RUN_ID")
    root = artifact.parents[2]
    tracked = root / "project" / "results" / artifact.name
    require(tracked.resolve().is_relative_to(root), "Tracked results path escapes the workspace")
    return artifact, root, tracked


def current_validation_hashes():
    # These are current validation sources, never a claim about the code loaded
    # by a process that may have started before the latest edit.
    paths = list((SOURCE_ROOT / "project" / "src" / "ragstudy").glob("*.py"))
    paths += [Path(__file__).resolve(), SOURCE_ROOT / "project" / "scripts" / "run_experiment.py"]
    return {path.relative_to(SOURCE_ROOT).as_posix(): digest(path)
            for path in sorted(set(paths)) if path.is_file()}


def _verify(artifact, root, tracked, report, recompute_candidates):
    result = read_json(artifact / "results.json")
    rows = read_json(artifact / "per_query_metrics.json")
    manifest = read_json(artifact / "split_manifest.json")
    audit = read_json(artifact / "data_audit.json")
    config = read_json(artifact / "configuration.json")
    methods = (*METHODS, *(["weighted_rrf"] if result.get("weighted_rrf_selection") is not None else []))
    require(result["run_id"] == artifact.name, "Run ID differs from the artifact directory")
    require(result["data_audit"] == audit, "Embedded and standalone data audits differ")
    require(result["configuration_sha256"] == digest(artifact / "configuration.json"),
            "Configuration SHA-256 mismatch")
    require(result["source_hashes"] == audit["source_hashes"], "Input hash manifests differ")
    for relative, expected in result.get("protocol_input_hashes", {}).items():
        require(digest(root / relative) == expected, "Protocol input SHA-256 mismatch")
    source = (root / config["source"]).resolve()
    require(source.is_relative_to(root), "Configured input path escapes the workspace")
    for name in ("train.json", "corpus.zip"):
        require(digest(source / name) == audit["source_hashes"][name], f"Input SHA-256 mismatch: {name}")
    require(len(read_json(source / "train.json")) == audit["source_records"], "Raw record count differs from audit")
    evaluated_test = bool(result["final_test_evaluated"])
    require(not evaluated_test or config.get("final_test_enabled", False), "Held-out metrics bypassed the configuration lock")

    queries = {query["query_id"]: query for query in manifest}
    require(len(queries) == len(manifest), "Duplicate query IDs in split manifest")
    require(len(queries) == audit["canonical_answerable_queries"], "Canonical query count differs from audit")
    if config.get("split_lock_manifest"):
        locked = read_json(root / config["split_lock_manifest"])["query_splits"]
        require(all(qid in locked and query["split"] == locked[qid] for qid, query in queries.items()),
                "Historical split membership was changed")
        quarantined = {entry["query_id"] for entry in audit["quarantined_queries"]}
        require(set(queries).isdisjoint(quarantined) and set(queries) | quarantined == set(locked),
                "Historical cohort differs beyond documented quarantine")
    group_split, gold_split = {}, {}
    for query in queries.values():
        split, group = query["split"], query["group_id"]
        require(split in ("train", "dev", "test"), "Unknown split in manifest")
        require(group not in group_split or group_split[group] == split, "One component crosses splits")
        group_split[group] = split
        require(bool(query["positive_document_ids"]), "Answerable query lacks a gold ID")
        for doc in query["positive_document_ids"]:
            require(doc not in gold_split or gold_split[doc] == split, "Shared gold document crosses splits")
            gold_split[doc] = split
    require(len(group_split) == audit["independent_components"], "Independent component count differs from audit")
    for split in ("train", "dev", "test"):
        cohort = [query for query in queries.values() if query["split"] == split]
        require(audit["splits"][split] == {"queries": len(cohort), "groups": len({query["group_id"] for query in cohort})},
                f"Split audit count mismatch: {split}")
    expected_evaluated = {qid for qid, query in queries.items() if evaluated_test or query["split"] != "test"}
    observed_rows = {row["query_id"]: row for row in rows}
    require(len(observed_rows) == len(rows), "Duplicate evaluated query rows")
    require(set(observed_rows) == expected_evaluated, "Evaluation omitted or added queries")

    candidates, qrels = defaultdict(dict), defaultdict(set)
    saved_rankings = defaultdict(list)
    with closing(sqlite3.connect((artifact / "experiments.sqlite").as_uri() + "?mode=ro", uri=True)) as database:
        require(database.execute("PRAGMA quick_check").fetchone()[0] == "ok", "SQLite integrity check failed")
        sql_runs = database.execute("SELECT run_id,configuration FROM experiment_runs").fetchall()
        require(len(sql_runs) == 1 and sql_runs[0][0] == result["run_id"] and json.loads(sql_runs[0][1]) == config,
                "SQLite run configuration differs")
        passages = {row[0] for row in database.execute("SELECT document_id FROM passages")}
        require(len(passages) == audit["unique_documents"], "SQLite document count differs from audit")
        sql_queries = {qid: (question, split, group) for qid, question, split, group in
                       database.execute("SELECT query_id,question,split,group_id FROM queries")}
        require(sql_queries == {qid: (query["question"], query["split"], query["group_id"]) for qid, query in queries.items()},
                "SQLite queries differ from the split manifest")
        for qid, doc in database.execute("SELECT query_id,document_id FROM qrels"):
            require(qid in queries and doc in passages, "Orphan relevance judgment")
            qrels[qid].add(doc)
        for qid, query in queries.items():
            require(qrels[qid] == set(query["positive_document_ids"]), "SQLite gold IDs differ from split manifest")
        for qid, doc, label in database.execute("SELECT query_id,document_id,observed_positive FROM candidates"):
            require(qid in queries and doc in passages, "Orphan candidate")
            require(label in (0, 1) and label == int(doc in qrels[qid]), "Candidate relevance label differs from qrels")
            require(doc not in candidates[qid], "Duplicate candidate")
            candidates[qid][doc] = label
        for qid, method, rank, doc in database.execute("SELECT query_id,method,rank,document_id FROM rankings ORDER BY query_id,method,rank"):
            require(qid in expected_evaluated, "Held-out or unknown query has a saved ranking")
            require(method in methods and doc in candidates[qid], "Ranking is outside the shared candidate set")
            order = saved_rankings[qid, method]
            require(rank == len(order) + 1 and rank <= 20 and doc not in order, "Saved ranking positions are invalid")
            order.append(doc)
    for qid in queries:
        require(len(candidates[qid]) <= 2 * config["top_k"], "Candidate union exceeds the fixed retrieval budget")

    # Validate original retrieval runs if saved, otherwise reconstruct only when
    # explicitly authorized to load this project's trusted local pickle.
    retrieval_path = artifact / "candidate_runs.json"
    provenance = None
    retrieval_rows = None
    if retrieval_path.is_file():
        saved = read_json(retrieval_path)
        retrieval_rows = saved["queries"] if isinstance(saved, dict) else saved
        provenance = "saved_retrieval_runs"
    elif recompute_candidates:
        import joblib
        import numpy as np
        sys.path.insert(0, str(SOURCE_ROOT / "project" / "src"))
        from ragstudy.search import load_bundle
        bundle = load_bundle(artifact)
        require(bundle["configuration"] == config, "Saved search bundle configuration differs")
        ids = bundle["document_ids"]
        require(set(ids) == passages and len(ids) == len(passages), "Saved index document IDs differ")
        retrieval_rows = []
        for number, query in enumerate(queries.values(), 1):
            orders = {}
            for name, index_name in (("bm25_order", "bm25"), ("semantic_order", "semantic")):
                values = np.asarray(bundle[index_name].score(query["question"]))
                require(values.shape == (len(ids),) and np.isfinite(values).all(), "Reconstructed scores are invalid")
                positions = np.arange(len(ids))
                if name == "bm25_order":
                    positions = positions[values > 0]
                order = np.lexsort((positions, np.asarray(ids)[positions], -values[positions]))
                orders[name] = [ids[index] for index in positions[order[:config["top_k"]]]]
            retrieval_rows.append({"query_id": query["query_id"], **orders})
            if number % 50 == 0 or number == len(queries):
                print(f"Recomputed retrieval {number}/{len(queries)}", flush=True)
        provenance = "recomputed_from_saved_local_index"
        report["warnings"].append("Candidate provenance was independently recomputed from the saved index; it is not an original-run retrieval log.")
    if retrieval_rows is not None:
        retrievers = {row["query_id"]: row for row in retrieval_rows}
        require(len(retrievers) == len(retrieval_rows) and set(retrievers) == set(queries), "Retrieval evidence query set differs")
        for qid, retrieval in retrievers.items():
            union = set()
            for key in ("bm25_order", "semantic_order"):
                order = retrieval[key]
                require(len(order) == len(set(order)) and len(order) <= config["top_k"], "Retriever exceeds unique top-k budget")
                require(set(order) <= passages, "Retriever returned an unknown document")
                union.update(order)
                if qid in expected_evaluated:
                    method = "bm25" if key == "bm25_order" else "semantic"
                    require(order[:20] == saved_rankings[qid, method], "Saved retrieval ranking differs from candidate evidence")
            require(union == set(candidates[qid]), "Candidates differ from the exact retriever union (possible injection or omission)")
            if "candidate_ids" in retrieval:
                require(set(retrieval["candidate_ids"]) == union, "Candidate evidence differs from its retrieval union")
            if qid in expected_evaluated and "rrf_constant" in result:
                configurations = [("rrf", result["rrf_constant"], 1.)]
                if result.get("weighted_rrf_selection") is not None:
                    chosen = result["weighted_rrf_selection"]
                    configurations.append(("weighted_rrf", chosen["constant"], chosen["semantic_weight"]))
                lexical_ranks = {doc: rank for rank, doc in enumerate(retrieval["bm25_order"], 1)}
                semantic_ranks = {doc: rank for rank, doc in enumerate(retrieval["semantic_order"], 1)}
                for name, constant, weight in configurations:
                    scores = {doc: (1 / (constant + lexical_ranks[doc]) if doc in lexical_ranks else 0.)
                                   + (weight / (constant + semantic_ranks[doc]) if doc in semantic_ranks else 0.)
                              for doc in union}
                    ranking = sorted(union, key=lambda doc: (-scores[doc], doc))[:20]
                    require(ranking == saved_rankings[qid, name], "RRF formula/ranking differs from retrieval evidence")
        report["checks"]["exact_candidate_union"] = provenance
    else:
        report["checks"]["exact_candidate_union"] = "unverified_no_complete_retrieval_runs"
        report["warnings"].append("No full top-k retrieval log was saved: candidate labels and ranking membership pass, but absence of gold injection is not proven. Use --recompute-candidates with a trusted local bundle.")

    recomputed = {}
    for qid, row in observed_rows.items():
        query = queries[qid]
        require(row["split"] == query["split"] and row["group_id"] == query["group_id"], "Metric row split/group differs")
        require(row["candidate_count"] == len(candidates[qid]), "Metric candidate count differs")
        candidate_recall = len(set(candidates[qid]) & qrels[qid]) / len(qrels[qid])
        close_number(row["candidate_recall"], candidate_recall, "Candidate recall differs")
        require(set(row["rankings"]) == set(methods), "Metric ranking method set differs")
        recomputed[qid] = {"candidate_recall": candidate_recall}
        for method in methods:
            ranking = saved_rankings[qid, method]
            require(row["rankings"][method] == ranking, "JSON and SQLite rankings differ")
            if method not in ("bm25", "semantic"):
                require(len(ranking) == min(20, len(candidates[qid])), "Shared-union ranking is prematurely truncated")
            metrics = independent_metrics(ranking, qrels[qid])
            for metric, value in metrics.items():
                close_number(row[method][metric], value, f"{qid}/{method}/{metric}")
            recomputed[qid][method] = metrics

    expected_splits = {queries[qid]["split"] for qid in expected_evaluated}
    require(set(result["summary"]) == expected_splits, "Aggregate split set differs")
    for split, summary in result["summary"].items():
        ids = [qid for qid in observed_rows if queries[qid]["split"] == split]
        require(summary["queries"] == len(ids) and summary["groups"] == len({queries[qid]["group_id"] for qid in ids}), "Aggregate cohort count differs")
        close_number(summary["candidate_recall"], math.fsum(recomputed[qid]["candidate_recall"] for qid in ids) / len(ids), "Aggregate candidate recall differs")
        require(set(summary["methods"]) == set(methods), "Aggregate method set differs")
        for method in methods:
            for metric in ("ndcg@10", "recall@20"):
                close_number(summary["methods"][method][metric], math.fsum(recomputed[qid][method][metric] for qid in ids) / len(ids), "Aggregate metric differs")
        for method, comparison in summary["paired_comparisons"].items():
            require(method in ("logistic", "random_forest") and comparison["metric"] == "ndcg@10", "Unexpected paired comparison")
            base = summary["methods"]["rrf"]["ndcg@10"]
            challenger = summary["methods"][method]["ndcg@10"]
            close_number(comparison["baseline_mean"], base, "Bootstrap baseline mean differs")
            close_number(comparison["challenger_mean"], challenger, "Bootstrap challenger mean differs")
            close_number(comparison["difference"], challenger - base, "Bootstrap mean difference differs")
            require(comparison["n_queries"] == len(ids) and comparison["n_groups"] == summary["groups"], "Bootstrap cohort count differs")
            require(-1 <= comparison["ci_lower"] <= comparison["ci_upper"] <= 1, "Bootstrap interval bounds invalid")
        for comparison in summary.get("paired_contrasts", []):
            baseline, challenger, metric = comparison["baseline"], comparison["challenger"], comparison["metric"]
            require(baseline in methods and challenger in methods and baseline != challenger
                    and metric in ("ndcg@10", "recall@20"), "Unexpected extended paired contrast")
            base = summary["methods"][baseline][metric]
            other = summary["methods"][challenger][metric]
            close_number(comparison["baseline_mean"], base, "Contrast baseline mean differs")
            close_number(comparison["challenger_mean"], other, "Contrast challenger mean differs")
            close_number(comparison["difference"], other - base, "Contrast difference differs")
            require(comparison["n_queries"] == len(ids) and comparison["n_groups"] == summary["groups"], "Contrast group count differs")
            require(-1 <= comparison["ci_lower"] <= comparison["ci_upper"] <= 1, "Contrast interval invalid")
        for name, estimates in summary["latency_single_observation"].items():
            values = [float(observed_rows[qid]["latency"][name]) for qid in ids]
            require(all(math.isfinite(value) and value >= 0 for value in values), "Invalid observed latency")
            close_number(estimates["median_ms"], percentile(values, .5), "Latency median differs")
            close_number(estimates["p95_ms"], percentile(values, .95), "Latency p95 differs")

    csv_path = tracked / "per_query_metrics.csv"
    if tracked.is_dir():
        require(read_json(tracked / "results.json") == result, "Tracked and local result JSON differ")
    if csv_path.is_file():
        with csv_path.open(encoding="utf-8", newline="") as handle:
            compact = list(csv.DictReader(handle))
        compact_rows = {(row["query_id"], row["method"]): row for row in compact}
        require(len(compact_rows) == len(compact) and set(compact_rows) == {(qid, method) for qid in expected_evaluated for method in methods}, "Tracked CSV row set differs")
        for (qid, method), row in compact_rows.items():
            require(row["split"] == queries[qid]["split"] and row["group_id"] == str(queries[qid]["group_id"]), "CSV split/group differs")
            for metric in ("ndcg@10", "recall@20"):
                close_number(row[metric], recomputed[qid][method][metric], "CSV ranking metric differs")
            close_number(row["candidate_recall"], recomputed[qid]["candidate_recall"], "CSV candidate recall differs")
        report["checks"]["tracked_csv"] = "passed"
    else:
        report["checks"]["tracked_csv"] = "not_present"
    report["checks"].update({
        "configuration_and_input_hashes": "passed", "sqlite_integrity": "passed",
        "component_and_shared_gold_split_isolation": "passed",
        "complete_evaluation_query_cohort": "passed", "candidate_label_consistency": "passed",
        "independent_ranking_metrics": "passed", "independent_aggregate_metrics": "passed",
        "same_complete_candidate_set_membership": "passed", "default_test_ranking_lock": "passed",
        "bootstrap_scope": "means, group counts and interval bounds verified; interval draws not independently recomputed",
    })
    report["counts"] = {"queries": len(queries), "evaluated_queries": len(rows),
                        "groups": len(group_split), "documents": len(passages),
                        "candidates": sum(len(value) for value in candidates.values()),
                        "saved_top20_rankings": sum(len(value) for value in saved_rankings.values()),
                        "test_evaluated": evaluated_test}
    evidence = [artifact / name for name in ("results.json", "per_query_metrics.json", "split_manifest.json", "data_audit.json", "configuration.json", "experiments.sqlite")]
    if retrieval_path.is_file():
        evidence.append(retrieval_path)
    report["evidence_sha256"] = {path.name: digest(path) for path in evidence}
    if csv_path.is_file():
        report["tracked_csv_sha256"] = digest(csv_path)


def verify_experiment(artifact, *, recompute_candidates=False, write=True):
    artifact, root, tracked = locate(artifact)
    report = {"status": "failed", "run_id": artifact.name,
              "verified_at_utc": datetime.now(timezone.utc).isoformat(),
              "artifact_directory": str(artifact), "checks": {}, "warnings": [],
              "validation_source_sha256": current_validation_hashes(),
              "source_hash_scope": "Current validation source files; not evidence of code loaded by the original run",
              "recomputed_candidates": bool(recompute_candidates)}
    try:
        _verify(artifact, root, tracked, report, recompute_candidates)
        report["status"] = "passed"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    if write:
        text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
        (artifact / "verification.json").write_text(text, encoding="utf-8")
        if tracked.is_dir():
            (tracked / "verification.json").write_text(text, encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_directory", nargs="?", type=Path)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--recompute-candidates", action="store_true", help="Load only a trusted, locally generated joblib bundle and recompute exact retrieval unions")
    args = parser.parse_args()
    if (args.artifact is None) == (args.artifact_directory is None):
        parser.error("Provide exactly one artifact directory, positional or --artifact")
    try:
        report = verify_experiment(args.artifact or args.artifact_directory,
                                   recompute_candidates=args.recompute_candidates)
    except (VerificationError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
