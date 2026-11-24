"""Repeated warm, serial query timing with all evidence-search stages included."""
from pathlib import Path
import argparse
import json
import platform
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for folder in (".vendor", ".vendor-hf", ".vendor-onnx-compat"):
    path = ROOT / "project" / folder
    if path.is_dir():
        sys.path.insert(0, str(path))
sys.path.insert(0, str(ROOT / "project/src"))
from ragstudy.search import load_bundle, search
from ragstudy.experiment import write_json
from threadpoolctl import threadpool_limits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.queries < 1 or args.repeats < 2:
        parser.error("Use at least one query and two repeats.")
    queries = json.loads((args.artifact / "split_manifest.json").read_text(encoding="utf-8"))
    dev = sorted((q for q in queries if q["split"] == "dev"), key=lambda q: q["query_id"])[:args.queries]
    started = time.perf_counter()
    rows = []
    with threadpool_limits(limits=2):
        bundle = load_bundle(args.artifact)
        methods = ["bm25", "semantic", "rrf", *(["weighted_rrf"] if bundle.get("weighted_rrf_selection") is not None else []), "logistic", "random_forest"]
        for method in methods:
            for query in dev:
                search(bundle, query["question"], method)
            for repeat in range(args.repeats):
                for query in dev:
                    result = search(bundle, query["question"], method)
                    rows.append({"method": method, "query_id": query["query_id"], "repeat": repeat,
                                 **result["timing_ms"]})
    summary = {method: {stage: {"median_ms": float(np.median([row[stage] for row in rows if row["method"] == method])),
                               "p95_ms": float(np.percentile([row[stage] for row in rows if row["method"] == method], 95))}
                        for stage in ("retrieval", "candidate_features", "rerank_and_output", "total")}
               for method in methods}
    report = {"protocol": "Warm serial CPU search; one excluded warm pass per method; deterministic development subset only",
              "query_count": len(dev), "repeats": args.repeats, "samples": len(rows), "concurrency": 1,
              "threads": "BLAS/OpenMP limited to two", "platform": platform.platform(),
              "encoder_threads": bundle["configuration"].get("encoder_threads"),
              "timing_boundary": "Query encoding/BM25, exact search, candidate union, feature building, prediction and top-k; excludes bundle loading, model download and text generation",
              "wall_seconds": time.perf_counter() - started, "summary": summary, "samples_ms": rows}
    write_json(args.artifact / "latency_benchmark.json", report)
    tracked = ROOT / "project/results" / args.artifact.name
    if tracked.is_dir():
        write_json(tracked / "latency_benchmark.json", report)
    print(json.dumps({"summary": summary, "samples": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
