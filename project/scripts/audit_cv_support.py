"""Reconstruct grouped-CV support from a completed run without fitting models."""
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "project/src"))
for directory in (".vendor", ".vendor-hf"):
    vendor = ROOT / "project" / directory
    if vendor.is_dir():
        sys.path.insert(0, str(vendor))

import numpy as np
import sklearn
from sklearn.model_selection import GroupKFold
from ragstudy.data import sha256
from ragstudy.features import FEATURE_NAMES, sample_training_rows


def list_hash(values):
    return hashlib.sha256(json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def raw_support(records):
    qids = [record["query"]["query_id"] for record in records]
    groups = sorted({record["query"]["group_id"] for record in records})
    missed = [record["query"]["query_id"] for record in records if not np.any(record["labels"] == 1)]
    no_negative = [record["query"]["query_id"] for record in records if not np.any(record["labels"] == 0)]
    positive_rows = sum(int(np.sum(record["labels"] == 1)) for record in records)
    negative_rows = sum(int(np.sum(record["labels"] == 0)) for record in records)
    return {"queries": len(records), "groups": len(groups), "candidate_positive_rows": positive_rows,
            "candidate_proxy_negative_rows": negative_rows, "candidate_rows": positive_rows + negative_rows,
            "queries_without_retrieved_positive": len(missed), "missed_positive_query_ids": missed,
            "queries_without_proxy_negative": len(no_negative), "no_negative_query_ids": no_negative,
            "query_ids": qids, "query_ids_sha256": list_hash(qids), "group_ids_sha256": list_hash(groups)}


def sampled_support(records, config):
    seed = config.get("training_seed", config["seed"])
    maximum = config["max_proxy_negatives"]
    features, labels, weights, groups, missed = sample_training_rows(
        records, maximum, seed, negative_sampling=config.get("negative_sampling", "random"),
        balance_classes=config.get("balance_classes", False), query_weight_total=config.get("query_weight_total"))
    offset, selected_pairs, budgets, fractions = 0, [], [], []
    used_qids = []
    for record in records:
        positive_count = int(np.sum(record["labels"] == 1))
        negative_count = int(np.sum(record["labels"] == 0))
        if not positive_count or not negative_count:
            continue
        count = positive_count + min(maximum, negative_count)
        section = slice(offset, offset + count)
        qid = record["query"]["query_id"]
        used_qids.append(qid)
        assert np.all(groups[section] == record["query"]["group_id"])
        available = defaultdict(list)
        for doc, feature, label in zip(record["candidate_ids"], record["features"], record["labels"]):
            available[(feature.tobytes(), int(label))].append(doc)
        for feature, label in zip(features[section], labels[section]):
            matches = available[(feature.tobytes(), int(label))]
            assert len(matches) == 1, f"Selected candidate identity is ambiguous for {qid}"
            selected_pairs.append([qid, matches.pop(), int(label)])
        budget = float(weights[section].sum())
        budgets.append(budget)
        fractions.append(float(weights[section][labels[section] == 1].sum() / budget))
        offset += count
    assert offset == len(labels) == len(selected_pairs)
    assert np.isfinite(weights).all() and np.all(weights > 0)
    if config.get("query_weight_total") is None:
        assert np.isclose(weights.mean(), 1)
    else:
        assert np.allclose(budgets, config["query_weight_total"]), "Fixed per-query weight total differs"
    assert np.allclose(budgets, budgets[0]), "Per-query weight budgets differ"
    if config.get("balance_classes", False):
        assert np.allclose(fractions, .5), "Within-query class balance differs"
    positive_weight = float(weights[labels == 1].sum())
    negative_weight = float(weights[labels == 0].sum())
    return {"rows": len(labels), "positive_rows": int(np.sum(labels == 1)),
            "proxy_negative_rows": int(np.sum(labels == 0)), "used_queries": len(used_qids),
            "missed_positive_queries": len(missed), "missed_positive_query_ids": missed,
            "positive_weight_sum": positive_weight, "proxy_negative_weight_sum": negative_weight,
            "positive_weight_fraction": positive_weight / float(weights.sum()),
            "query_weight_budget_min": min(budgets), "query_weight_budget_max": max(budgets),
            "within_query_positive_weight_fraction_min": min(fractions),
            "within_query_positive_weight_fraction_max": max(fractions),
            "selected_pair_ids_sha256": list_hash(selected_pairs), "used_query_ids_sha256": list_hash(used_qids),
            "feature_rows_sha256": hashlib.sha256(np.ascontiguousarray(features).tobytes()).hexdigest(),
            "labels_sha256": hashlib.sha256(np.ascontiguousarray(labels).tobytes()).hexdigest(),
            "weights_sha256": hashlib.sha256(np.ascontiguousarray(weights).tobytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    assert artifact.parent.name == "artifacts" and artifact.parent.parent.name == "project"
    config = json.loads((artifact / "configuration.json").read_text(encoding="utf-8"))
    result = json.loads((artifact / "results.json").read_text(encoding="utf-8"))
    assert not result["final_test_evaluated"], "Audit is restricted to train/dev runs"
    assert config.get("fixed_model_parameters") is None and result["cross_validation"], \
        "This run did not perform grouped CV; do not present hypothetical folds as actual CV support"
    assert np.__version__ == result["runtime"]["numpy"], "Use the run's NumPy environment"
    assert sklearn.__version__ == result["runtime"]["sklearn"], "Use the run's sklearn environment"
    feature_path = ROOT / "project/src/ragstudy/features.py"
    source_hash = result.get("source_sha256_at_run_start", {}).get("project/src/ragstudy/features.py")
    if source_hash is not None:
        assert sha256(feature_path) == source_hash, "Sampling implementation changed since the run started"
    manifest = json.loads((artifact / "split_manifest.json").read_text(encoding="utf-8"))
    # Keep the original canonical-query order: random-negative draws consume one RNG per fit.
    training_queries = [query for query in manifest if query["split"] == "train"]
    candidates = defaultdict(list)
    database_path = artifact / "experiments.sqlite"
    with sqlite3.connect(database_path.as_uri() + "?mode=ro", uri=True) as database:
        for qid, doc, label, values in database.execute(
                "SELECT c.query_id,c.document_id,c.observed_positive,c.features FROM candidates c "
                "JOIN queries q USING(query_id) WHERE q.split='train' ORDER BY c.query_id,c.document_id"):
            candidates[qid].append((doc, int(label), json.loads(values)))
    assert set(candidates).issubset({query["query_id"] for query in training_queries})
    records = []
    for query in training_queries:
        entries = candidates[query["query_id"]]
        features = np.asarray([entry[2] for entry in entries], dtype=np.float64).reshape(-1, len(FEATURE_NAMES))
        labels = np.asarray([entry[1] for entry in entries], dtype=int)
        assert np.isfinite(features).all() and set(labels).issubset({0, 1})
        records.append({"query": query, "candidate_ids": [entry[0] for entry in entries],
                        "features": features, "labels": labels})
    group_ids = np.asarray([record["query"]["group_id"] for record in records])
    folds = min(config["cv_folds"], len(set(group_ids)))
    assert folds >= 2
    assert all(row["folds"] == folds and row["queries"] == len(records)
               for row in result["cross_validation"]), "Saved CV scope differs from reconstructed support"
    fold_reports, seen_validation = [], []
    for fold, (fit_indices, valid_indices) in enumerate(GroupKFold(folds).split(np.arange(len(records)), groups=group_ids), 1):
        fit = [records[index] for index in fit_indices]
        valid = [records[index] for index in valid_indices]
        assert not set(group_ids[fit_indices]) & set(group_ids[valid_indices])
        seen_validation.extend(record["query"]["query_id"] for record in valid)
        fold_reports.append({"fold": fold, "fit": raw_support(fit), "validation": raw_support(valid),
                             "fit_sampled_rows": sampled_support(fit, config),
                             "validation_sampling": "none; every validation query and complete candidate union is evaluated"})
    assert len(seen_validation) == len(set(seen_validation)) == len(records)
    assert set(seen_validation) == {record["query"]["query_id"] for record in records}
    full_fit = sampled_support(records, config)
    for name, model in result["selected_models"].items():
        assert model["training_pairs"] == full_fit["rows"], f"Full-fit row count mismatch: {name}"
        assert model["training_queries_without_positive"] == full_fit["missed_positive_query_ids"], name
    report = {"status": "passed", "verified_at_utc": datetime.now(timezone.utc).isoformat(),
              "run_id": artifact.name, "analysis_name": config.get("analysis_name"),
              "scope": "support of CV folds inside historical training cohort; no models refitted or held-out scores read",
              "folds": folds, "seed": config["seed"], "training_seed": config.get("training_seed", config["seed"]),
              "negative_sampling": config.get("negative_sampling", "random"),
              "balance_classes": config.get("balance_classes", False),
              "query_weight_total": config.get("query_weight_total"),
              "feature_names": config.get("features", FEATURE_NAMES), "feature_mask_applied_after_sampling": True,
              "training_cohort": raw_support(records), "fold_support": fold_reports, "full_fit_sampled_rows": full_fit,
              "runtime": {"numpy": np.__version__, "sklearn": sklearn.__version__},
              "provenance": {"sqlite_sha256": sha256(database_path),
                             "split_manifest_sha256": sha256(artifact / "split_manifest.json"),
                             "configuration_sha256": sha256(artifact / "configuration.json"),
                             "sampling_source_sha256": sha256(feature_path), "audit_script_sha256": sha256(Path(__file__))}}
    content = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    (artifact / "CV_support.json").write_text(content, encoding="utf-8")
    tracked = artifact.parents[1] / "results" / artifact.name
    assert tracked.is_dir(), "Run's tracked results directory does not exist"
    (tracked / "CV_support.json").write_text(content, encoding="utf-8")
    print(json.dumps({"status": "passed", "artifact": str(artifact), "folds": folds,
                      "training_queries": len(records), "training_groups": len(set(group_ids)),
                      "full_fit_rows": full_fit["rows"], "missed_positive_queries": full_fit["missed_positive_queries"],
                      "positive_weight_fraction": full_fit["positive_weight_fraction"]}, indent=2))


if __name__ == "__main__":
    main()
