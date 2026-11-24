"""Run reproducible, local TechQA development experiments; no test scoring by default."""
from pathlib import Path
import argparse
import json
import sys
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
for folder in (".vendor", ".vendor-hf", ".vendor-onnx-compat"):
    vendor = ROOT / "project" / folder
    if vendor.is_dir():
        sys.path.insert(0, str(vendor))
sys.path.insert(0, str(ROOT / "project" / "src"))
from ragstudy.experiment import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "project/configs/techqa_exploratory.json")
    parser.add_argument("--backend", choices=["lsa", "neural", "onnx"])
    parser.add_argument("--evaluate-test", action="store_true", help="Requires final_test_enabled in a reviewed configuration")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("encoder_provider") == "DmlExecutionProvider":
        vendor = ROOT / "project/.vendor-onnx-dml"
        if vendor.is_dir():
            sys.path.insert(0, str(vendor))
    if args.backend:
        config["semantic_backend"] = args.backend
    if args.evaluate_test and not config.get("final_test_enabled", False):
        parser.error("Final test is locked. Complete cohort/protocol review and explicitly enable it in a separate frozen configuration.")
    with threadpool_limits(limits=2):
        run(ROOT, config, include_test=args.evaluate_test)


if __name__ == "__main__":
    main()
