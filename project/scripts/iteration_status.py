"""Inspect completed experiment evidence and partial embedding checkpoints."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json

ROOT = Path(__file__).resolve().parents[2]


def load(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Also save the current local status in project/results")
    args = parser.parse_args()
    primary = {name: {"status": "not_completed"} for name in ("head_lsa", "full_lsa", "head_onnx")}
    for folder in sorted((ROOT / "project/artifacts").iterdir()):
        if not folder.is_dir():
            continue
        config, result = load(folder / "configuration.json"), load(folder / "results.json")
        if not config or config.get("analysis_name") not in primary or not result:
            continue
        verification = load(folder / "verification.json")
        primary[config["analysis_name"]] = {"status": "verified" if verification and verification["status"] == "passed" else "completed_unverified",
                                           "run_id": folder.name, "test_evaluated": result["final_test_evaluated"]}
    caches = []
    parent = ROOT / "project/data/cache/embeddings"
    if parent.is_dir():
        for folder in parent.iterdir():
            identity = load(folder / "identity.json")
            if not identity or identity["identity"]["document_count"] < 100:
                continue
            batches = [value for path in folder.glob("batch_*.json") if (value := load(path))]
            complete = sum(batch["end"] - batch["start"] for batch in batches)
            total = identity["identity"]["document_count"]
            seconds = sum(batch["encoding_wall_seconds"] for batch in batches)
            caches.append({"cache_key": folder.name, "precision": identity["identity"]["precision"],
                           "execution_provider": identity["identity"]["execution_provider"],
                           "max_seq_length": identity["identity"]["max_seq_length"],
                           "intra_op_threads": identity["identity"]["intra_op_threads"],
                           "completed_documents": complete, "total_documents": total,
                           "completion_percent": round(complete / total * 100, 2),
                           "recorded_encoding_seconds": seconds,
                           "scope": "Partial corpus encoding is not a completed retrieval/training experiment"})
    ablations = load(ROOT / "project/results/iteration2_ablations.json")
    registry = load(ROOT / "project/artifacts/frozen_holdout_registry.json")
    frozen = load(ROOT / "project/artifacts" / registry["run_id"] / "freeze_manifest.json") if registry else None
    report = {"updated_at_utc": datetime.now(timezone.utc).isoformat(), "primary_experiments": primary,
              "embedding_caches": caches,
              "training_ablations": {"status": ablations["status"],
                  "completed_variants": sum(value.get("status") in ("complete", "completed") for value in ablations["variants"].values()),
                  "total_variants": len(ablations["variants"])} if ablations else None,
              "frozen_holdout": {"run_id": frozen["run_id"], "status": frozen["status"],
                                 "heldout_scored": frozen["heldout_scored"]} if frozen else None}
    if args.write:
        (ROOT / "project/results/iteration2_status.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
