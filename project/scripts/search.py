"""Query a locally generated experiment bundle and display original evidence IDs."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
for folder in (".vendor", ".vendor-hf", ".vendor-onnx-compat"):
    path = ROOT / "project" / folder
    if path.is_dir():
        sys.path.insert(0, str(path))
sys.path.insert(0, str(ROOT / "project" / "src"))
from ragstudy.search import load_bundle, search
from threadpoolctl import threadpool_limits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--query", required=True)
    parser.add_argument("--method", choices=["bm25", "semantic", "rrf", "weighted_rrf", "logistic", "random_forest"], default="logistic")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        result = search(load_bundle(args.artifact), args.query, args.method, args.limit)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
