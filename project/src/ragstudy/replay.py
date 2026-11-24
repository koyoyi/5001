"""Reuse audited retrieval evidence for one-factor training robustness runs."""
from contextlib import closing
from pathlib import Path
import json
import sqlite3
import joblib
import numpy as np
from .data import sha256
from .features import FEATURE_NAMES


def replay_retrieval(source, dataset, config):
    source = Path(source)
    result = json.loads((source / "results.json").read_text(encoding="utf-8"))
    previous = json.loads((source / "configuration.json").read_text(encoding="utf-8"))
    if result["data_audit"] != dataset["audit"]:
        raise ValueError("Retrieval replay requires the identical audited cohort and document policy.")
    keys = ("source", "semantic_backend", "head_words", "lsa_dimensions", "top_k", "encoder_path",
            "document_policy", "encoder_batch_size", "encoder_max_seq_length", "encoder_threads", "seed",
            "encoder_provider", "encoder_device_id")
    if any(previous.get(key) != config.get(key) for key in keys):
        raise ValueError("Retrieval configuration differs from the replay source.")
    if result["feature_names"] != FEATURE_NAMES:
        raise ValueError("Replay source uses different raw features.")
    bundle = joblib.load(source / "search_bundle.joblib")
    if bundle["document_ids"] != dataset["document_ids"] or bundle["documents"] != dataset["documents"]:
        raise ValueError("Replay corpus IDs or text differ.")
    runs = json.loads((source / "candidate_runs.json").read_text(encoding="utf-8"))
    by_query = {row["query_id"]: row for row in runs}
    positions = {doc: index for index, doc in enumerate(dataset["document_ids"])}
    features = {}
    with closing(sqlite3.connect((source / "experiments.sqlite").resolve().as_uri() + "?mode=ro", uri=True)) as database:
        for qid, doc, label, values in database.execute("SELECT query_id,document_id,observed_positive,features FROM candidates"):
            features[qid, doc] = (int(label), json.loads(values))
    rows = []
    for query in dataset["queries"]:
        retrieved = by_query[query["query_id"]]
        candidates = sorted(retrieved["candidate_ids"], key=positions.get)
        values = [features[query["query_id"], doc] for doc in candidates]
        if "latency" not in retrieved:
            raise ValueError("Replay requires original timing evidence for every query.")
        rows.append({"query": query, "candidates": np.asarray([positions[doc] for doc in candidates]),
                     "features": np.asarray([value[1] for value in values], dtype=float),
                     "labels": np.asarray([value[0] for value in values]),
                     "bm25_order": np.asarray([positions[doc] for doc in retrieved["bm25_order"]]),
                     "semantic_order": np.asarray([positions[doc] for doc in retrieved["semantic_order"]]),
                     "latency": retrieved["latency"]})
    provenance = {"source_run": source.name, "source_results_sha256": sha256(source / "results.json"),
                  "source_candidate_runs_sha256": sha256(source / "candidate_runs.json"),
                  "source_sqlite_sha256": sha256(source / "experiments.sqlite"),
                  "scope": "Identical raw features/candidates; refit supervised models only; retrieval timing inherited"}
    return bundle["bm25"], bundle["semantic"], rows, provenance
