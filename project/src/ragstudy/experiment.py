"""Audited retrieval, group-CV model selection, development evaluation and evidence storage."""
from datetime import datetime, timezone
from contextlib import closing
from pathlib import Path
import csv
import hashlib
import itertools
import json
import platform
import sqlite3
import time

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .data import load_techqa, sha256
from .features import FEATURE_NAMES, build_features, sample_training_rows
from .metrics import ranking_metrics, paired_group_bootstrap
from .retrieval import BM25Index, SparseLSAIndex, SentenceEncoderIndex, top_k, rrf_scores


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def rank_record(record, scores, document_ids):
    order = top_k(np.asarray(scores), len(record["candidates"]),
                  document_ids=[document_ids[index] for index in record["candidates"]])
    return [document_ids[record["candidates"][index]] for index in order]


def fit_model(kind, parameters, records, config):
    training_seed = config.get("training_seed", config["seed"])
    X, y, weights, _, missed = sample_training_rows(records, config["max_proxy_negatives"], training_seed,
        negative_sampling=config.get("negative_sampling", "random"), balance_classes=config.get("balance_classes", False),
        query_weight_total=config.get("query_weight_total"))
    feature_indices = [FEATURE_NAMES.index(name) for name in config.get("features", FEATURE_NAMES)]
    if not feature_indices or len(set(feature_indices)) != len(feature_indices):
        raise ValueError("Use a nonempty, unique set of known feature names.")
    X = X[:, feature_indices]
    if len(np.unique(y)) != 2:
        raise ValueError("Training requires both observed positives and proxy negatives.")
    scaler = StandardScaler() if kind == "logistic" else None
    if scaler is not None:
        X = scaler.fit_transform(X, sample_weight=weights)
        model = LogisticRegression(C=parameters["C"], max_iter=2000, solver="lbfgs", random_state=training_seed)
    else:
        model = RandomForestClassifier(n_estimators=config["rf_trees"], max_depth=parameters["max_depth"],
                                       min_samples_leaf=parameters["min_samples_leaf"],
                                       random_state=training_seed, n_jobs=2)
    model.fit(X, y, sample_weight=weights)
    if kind != "logistic":
        model.set_params(n_jobs=config.get("rf_prediction_jobs", 1))
    return {"model": model, "scaler": scaler, "parameters": parameters,
            "feature_indices": feature_indices,
            "training_pairs": len(y), "training_queries_without_positive": missed}


def predict(bundle, features):
    if not len(features):
        return np.empty(0)
    X = features[:, bundle["feature_indices"]] if "feature_indices" in bundle else features
    X = bundle["scaler"].transform(X) if bundle["scaler"] is not None else X
    return bundle["model"].predict_proba(X)[:, 1]


def score_model(bundle, records, document_ids):
    return [ranking_metrics(rank_record(record, predict(bundle, record["features"]), document_ids),
                            record["query"]["positive_document_ids"]) for record in records]


def select_models(train, config, document_ids):
    if config.get("fixed_model_parameters") is not None:
        fitted = {kind: fit_model(kind, parameters, train, config)
                  for kind, parameters in config["fixed_model_parameters"].items()}
        for bundle in fitted.values():
            bundle["cv_ndcg@10"] = None
        return fitted, []
    group_names = sorted({record["query"]["group_id"] for record in train})
    folds = min(config["cv_folds"], len(group_names))
    if folds < 2:
        raise ValueError("At least two independent training components are required for CV.")
    group_ids = np.asarray([record["query"]["group_id"] for record in train])
    splits = list(GroupKFold(folds).split(np.arange(len(train)), groups=group_ids))
    choices = {
        "logistic": [{"C": value} for value in config["logistic_C"]],
        "random_forest": [{"max_depth": depth, "min_samples_leaf": leaf}
                          for depth, leaf in itertools.product(config["rf_depth"], config["rf_min_leaf"])]}
    results, fitted = [], {}
    for kind, parameters in choices.items():
        best, best_score = None, -1.
        for values in parameters:
            validation_scores = []
            for train_indices, validation_indices in splits:
                fold_train = [train[index] for index in train_indices]
                fold_valid = [train[index] for index in validation_indices]
                bundle = fit_model(kind, values, fold_train, config)
                validation_scores.extend(metric["ndcg@10"] for metric in score_model(bundle, fold_valid, document_ids))
            score = float(np.mean(validation_scores))
            results.append({"model": kind, "parameters": values, "cv_ndcg@10": score,
                            "queries": len(validation_scores), "folds": folds})
            print(f"CV {kind} {values}: nDCG@10={score:.4f}", flush=True)
            if score > best_score:
                best, best_score = values, score
        fitted[kind] = fit_model(kind, best, train, config)
        fitted[kind]["cv_ndcg@10"] = best_score
    return fitted, results


def make_records(dataset, bm25, semantic, config):
    records = []
    ids = dataset["document_ids"]
    for number, query in enumerate(dataset["queries"], 1):
        start = time.perf_counter()
        lexical_scores = bm25.score(query["question"])
        lexical_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        semantic_scores = semantic.score(query["question"])
        semantic_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        lexical_order = top_k(lexical_scores, config["top_k"], document_ids=ids, min_score=0.)
        semantic_order = top_k(semantic_scores, config["top_k"], document_ids=ids)
        candidates = np.asarray(sorted(set(lexical_order.tolist()) | set(semantic_order.tolist())), dtype=int)
        features = build_features(query["question"], candidates, dataset["documents"], lexical_scores,
                                  semantic_scores, lexical_order, semantic_order, bm25_index=bm25)
        feature_ms = (time.perf_counter() - start) * 1000
        records.append({"query": query, "candidates": candidates, "features": features,
                        "labels": np.asarray([int(ids[doc] in query["positive_document_ids"]) for doc in candidates]),
                        "bm25_order": lexical_order, "semantic_order": semantic_order,
                        "latency": {"bm25_ms": lexical_ms, "semantic_ms": semantic_ms, "feature_ms": feature_ms}})
        if number % 50 == 0 or number == len(dataset["queries"]):
            print(f"Retrieved {number}/{len(dataset['queries'])} queries", flush=True)
    return records


def evaluate(records, bundles, constant, dataset, weighted=None):
    rows = []
    ids = dataset["document_ids"]
    for record in records:
        query = record["query"]
        rankings = {"bm25": [ids[index] for index in record["bm25_order"]],
                    "semantic": [ids[index] for index in record["semantic_order"]]}
        rrf = rrf_scores(record["candidates"], record["bm25_order"], record["semantic_order"], constant=constant)
        rankings["rrf"] = rank_record(record, rrf, ids)
        if weighted is not None:
            scores = rrf_scores(record["candidates"], record["bm25_order"], record["semantic_order"],
                                constant=weighted["constant"], semantic_weight=weighted["semantic_weight"])
            rankings["weighted_rrf"] = rank_record(record, scores, ids)
        latency = dict(record["latency"])
        for name, bundle in bundles.items():
            start = time.perf_counter()
            scores = predict(bundle, record["features"])
            latency[name + "_rerank_ms"] = (time.perf_counter() - start) * 1000
            rankings[name] = rank_record(record, scores, ids)
        gold = set(query["positive_document_ids"])
        candidate_recall = len(gold.intersection(ids[index] for index in record["candidates"])) / len(gold)
        rows.append({"query_id": query["query_id"], "group_id": query["group_id"], "split": query["split"],
                     "candidate_count": len(record["candidates"]), "candidate_recall": candidate_recall,
                     **{name: ranking_metrics(ranking, gold) for name, ranking in rankings.items()},
                     "latency": latency, "rankings": {name: ranking[:20] for name, ranking in rankings.items()}})
    return rows


def summarize(rows, bundles, config):
    summary = {}
    for split in ("train", "dev", "test"):
        cohort = [row for row in rows if row["split"] == split]
        if not cohort:
            continue
        summary[split] = {"queries": len(cohort), "groups": len({row["group_id"] for row in cohort}),
                          "candidate_recall": float(np.mean([row["candidate_recall"] for row in cohort])),
                          "methods": {}, "paired_comparisons": {}}
        for name in cohort[0]["rankings"]:
            summary[split]["methods"][name] = {metric: float(np.mean([row[name][metric] for row in cohort]))
                                               for metric in ("ndcg@10", "recall@20")}
        if split != "train":
            for name in bundles:
                summary[split]["paired_comparisons"][name] = paired_group_bootstrap(
                    cohort, "rrf", name, "ndcg@10", iterations=config["bootstrap_iterations"], seed=config["seed"])
            if config.get("comparison_baselines"):
                summary[split]["paired_contrasts"] = []
                challengers = [*bundles, *(["weighted_rrf"] if "weighted_rrf" in cohort[0] else [])]
                for baseline, challenger, metric in itertools.product(
                        config["comparison_baselines"], challengers, ("ndcg@10", "recall@20")):
                    if baseline == challenger or baseline not in cohort[0]:
                        continue
                    summary[split]["paired_contrasts"].append(paired_group_bootstrap(
                        cohort, baseline, challenger, metric,
                        iterations=config["bootstrap_iterations"], seed=config["seed"]))
        summary[split]["latency_single_observation"] = {}
        for metric in cohort[0]["latency"]:
            values = [row["latency"][metric] for row in cohort]
            summary[split]["latency_single_observation"][metric] = {
                "median_ms": float(np.median(values)), "p95_ms": float(np.percentile(values, 95))}
    return summary


def store_sqlite(path, dataset, records, rows, run_id, config):
    with closing(sqlite3.connect(path)) as database, database:
        database.executescript("""
            CREATE TABLE passages(document_id TEXT PRIMARY KEY, retrieval_text TEXT NOT NULL);
            CREATE TABLE queries(query_id TEXT PRIMARY KEY, question TEXT, split TEXT, group_id TEXT);
            CREATE TABLE qrels(query_id TEXT, document_id TEXT, PRIMARY KEY(query_id,document_id));
            CREATE TABLE candidates(query_id TEXT, document_id TEXT, observed_positive INTEGER, features TEXT,
                                    PRIMARY KEY(query_id,document_id));
            CREATE TABLE rankings(query_id TEXT, method TEXT, rank INTEGER, document_id TEXT,
                                  PRIMARY KEY(query_id,method,rank));
            CREATE TABLE experiment_runs(run_id TEXT PRIMARY KEY, configuration TEXT);
        """)
        database.executemany("INSERT INTO passages VALUES (?,?)", zip(dataset["document_ids"], dataset["documents"]))
        database.executemany("INSERT INTO queries VALUES (?,?,?,?)",
                             [(q["query_id"], q["question"], q["split"], q["group_id"]) for q in dataset["queries"]])
        database.executemany("INSERT INTO qrels VALUES (?,?)",
                             [(q["query_id"], doc) for q in dataset["queries"] for doc in q["positive_document_ids"]])
        database.execute("INSERT INTO experiment_runs VALUES (?,?)", (run_id, json.dumps(config)))
        for record in records:
            database.executemany("INSERT INTO candidates VALUES (?,?,?,?)",
                                 [(record["query"]["query_id"], dataset["document_ids"][doc], int(label), json.dumps(feature.tolist()))
                                  for doc, label, feature in zip(record["candidates"], record["labels"], record["features"])])
        database.executemany("INSERT INTO rankings VALUES (?,?,?,?)",
                             [(row["query_id"], method, rank, doc) for row in rows
                              for method, ranking in row["rankings"].items() for rank, doc in enumerate(ranking, 1)])


def make_chart(summary, path, document_policy="head"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dev = summary["dev"]
    methods = list(dev["methods"])
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    for axis, metric in zip(axes, ("ndcg@10", "recall@20")):
        axis.bar(methods, [dev["methods"][name][metric] for name in methods], color="#326c96")
        axis.set_ylim(0, 1)
        axis.set_title(metric + " on development queries")
        axis.tick_params(axis="x", labelrotation=25)
    figure.suptitle(f"TechQA exploratory {document_policy} retrieval (development)")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def run(root, config, include_test=False):
    if include_test and not config.get("final_test_enabled", False):
        raise ValueError("Final test requires an explicitly reviewed frozen configuration.")
    root = Path(root)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + config["semantic_backend"]
    output = root / "project" / "artifacts" / run_id
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    source_paths = sorted((root / "project/src/ragstudy").glob("*.py")) + [root / "project/scripts/run_experiment.py"]
    source_hashes = {str(path.relative_to(root)).replace("\\", "/"): sha256(path) for path in source_paths}
    print("Checking inputs, duplicate aliases and independent splits", flush=True)
    split_lock = None
    if config.get("split_lock_manifest"):
        split_lock = json.loads((root / config["split_lock_manifest"]).read_text(encoding="utf-8"))["query_splits"]
    dataset = load_techqa(root / config["source"], seed=config["seed"], head_words=config["head_words"],
                          near_duplicate_threshold=config["near_duplicate_threshold"],
                          document_policy=config.get("document_policy", "head"),
                          additional_query_groups=config.get("additional_query_groups"), split_lock=split_lock)
    write_json(output / "data_audit.json", dataset["audit"])
    write_json(output / "split_manifest.json", dataset["queries"])
    write_json(output / "configuration.json", config)
    print(f"Corpus: {len(dataset['documents'])}; split counts: {dataset['audit']['splits']}", flush=True)
    index_started = time.perf_counter()
    reuse = None
    if config.get("reuse_retrieval_run"):
        from .replay import replay_retrieval
        bm25, semantic, records, reuse = replay_retrieval(root / config["reuse_retrieval_run"], dataset, config)
    else:
        bm25 = BM25Index(dataset["documents"])
    print("BM25 index ready; building semantic index", flush=True)
    if reuse is not None:
        pass
    elif config["semantic_backend"] == "lsa":
        semantic = SparseLSAIndex(dataset["documents"], dimensions=config["lsa_dimensions"], seed=config["seed"])
    elif config["semantic_backend"] == "neural":
        semantic = SentenceEncoderIndex(dataset["documents"], root / config["encoder_path"])
        model_manifest = root / config["encoder_path"] / "download_manifest.json"
        if model_manifest.is_file():
            semantic.metadata["pinned_download_manifest"] = json.loads(model_manifest.read_text(encoding="utf-8"))
    elif config["semantic_backend"] == "onnx":
        from .neural import OnnxEncoderIndex
        semantic = OnnxEncoderIndex(dataset["documents"], root / config["encoder_path"],
            cache_dir=root / "project/data/cache/embeddings", batch_size=config.get("encoder_batch_size", 32),
            max_seq_length=config.get("encoder_max_seq_length", 512), threads=config.get("encoder_threads", 8),
            provider=config.get("encoder_provider", "CPUExecutionProvider"), device_id=config.get("encoder_device_id", 0))
    else:
        raise ValueError("Unknown semantic backend; use lsa, neural or onnx.")
    offline_seconds = time.perf_counter() - index_started
    if reuse is None:
        records = make_records(dataset, bm25, semantic, config)
    write_json(output / "candidate_runs.json", [{"query_id": record["query"]["query_id"],
        "bm25_order": [dataset["document_ids"][index] for index in record["bm25_order"]],
        "semantic_order": [dataset["document_ids"][index] for index in record["semantic_order"]],
        "candidate_ids": [dataset["document_ids"][index] for index in record["candidates"]],
        "latency": record["latency"]} for record in records])
    train = [record for record in records if record["query"]["split"] == "train"]
    dev = [record for record in records if record["query"]["split"] == "dev"]
    bundles, cv = select_models(train, config, dataset["document_ids"])
    constant_scores = []
    for constant in config["rrf_constants"]:
        evaluation = evaluate(dev, {}, constant, dataset)
        constant_scores.append({"constant": constant, "dev_ndcg@10": float(np.mean([row["rrf"]["ndcg@10"] for row in evaluation]))})
    constant = max(constant_scores, key=lambda item: item["dev_ndcg@10"])["constant"]
    weighted_choices = []
    for weight, value in itertools.product(config.get("rrf_semantic_weights", []), config["rrf_constants"]):
        choice = {"constant": value, "semantic_weight": weight}
        evaluation = evaluate(dev, {}, constant, dataset, weighted=choice)
        weighted_choices.append({**choice, "dev_ndcg@10": float(np.mean([row["weighted_rrf"]["ndcg@10"] for row in evaluation]))})
    weighted = max(weighted_choices, key=lambda item: item["dev_ndcg@10"]) if weighted_choices else None
    # The default workflow never evaluates held-out queries; test labels are used only to construct group-disjoint cohorts.
    evaluated = records if include_test else [record for record in records if record["query"]["split"] != "test"]
    rows = evaluate(evaluated, bundles, constant, dataset, weighted=weighted)
    summary = summarize(rows, bundles, config)
    selected = {name: {key: value for key, value in bundle.items() if key not in ("model", "scaler")}
                for name, bundle in bundles.items()}
    result = {"run_id": run_id, "scope": "exploratory", "final_test_evaluated": bool(include_test),
              "test_review_status": "semantic duplicate review and final protocol freeze pending" if not include_test else "frozen configuration held-out evaluation",
              "configuration_sha256": sha256(output / "configuration.json"),
              "source_sha256_at_run_start": source_hashes,
              "source_hashes": dataset["audit"]["source_hashes"], "data_audit": dataset["audit"],
              "retrieval_metadata": {"bm25": bm25.metadata, "semantic": semantic.metadata},
              "rrf_constant": constant, "rrf_development_selection": constant_scores,
              "weighted_rrf_selection": weighted, "weighted_rrf_development_grid": weighted_choices,
              "selected_models": selected, "cross_validation": cv, "summary": summary,
              "feature_names": FEATURE_NAMES, "offline_index_seconds": offline_seconds,
              "retrieval_reuse": reuse,
              "protocol_input_hashes": {config["split_lock_manifest"]: sha256(root / config["split_lock_manifest"])} if config.get("split_lock_manifest") else {},
              "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                          "numpy": np.__version__, "sklearn": sklearn.__version__, "blas_threads": 2},
              "latency_scope": "One observation/query during candidate generation and prediction; not repeated steady-state latency",
              "limitations": ["Custom supervised archive study, not official TechQA benchmark",
                              "Only document heads are indexed; long-document answer evidence may be lost",
                              "Unjudged candidates are proxy negatives, not confirmed irrelevant documents",
                              "Automated near-duplicate grouping does not replace human semantic review",
                              "LSA, if selected, is a corpus-fitted baseline and not a pretrained neural encoder",
                              "Development confidence intervals are exploratory after pipeline selection"]}
    write_json(output / "per_query_metrics.json", rows)
    write_json(output / "results.json", result)
    store_sqlite(output / "experiments.sqlite", dataset, records, rows, run_id, config)
    joblib.dump({"bm25": bm25, "semantic": semantic, "models": bundles, "rrf_constant": constant,
                 "weighted_rrf_selection": weighted,
                 "document_ids": dataset["document_ids"], "documents": dataset["documents"],
                 "configuration": config}, output / "search_bundle.joblib", compress=0)
    make_chart(summary, output / "development_metrics.png", config.get("document_policy", "head"))
    result["wall_seconds"] = time.perf_counter() - started
    tracked = root / "project" / "results" / run_id
    tracked.mkdir(parents=True, exist_ok=False)
    write_json(tracked / "results.json", result)
    # Compact, auditable query-level evidence; candidate text and indexes stay out of Git.
    with (tracked / "per_query_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["query_id", "group_id", "split", "method", "ndcg@10", "recall@20", "candidate_recall"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            for name in row["rankings"]:
                writer.writerow({"query_id": row["query_id"], "group_id": row["group_id"], "split": row["split"],
                                 "method": name, "candidate_recall": row["candidate_recall"], **row[name]})
    (tracked / "development_metrics.png").write_bytes((output / "development_metrics.png").read_bytes())
    write_json(output / "results.json", result)
    print(json.dumps({"artifacts": str(output), "results": str(tracked), "development": summary["dev"]}, indent=2), flush=True)
    return output, result
