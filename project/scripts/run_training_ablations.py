"""Run the six predeclared training sensitivities using one trusted retrieval run.

Usage: python project/scripts/run_training_ablations.py MAIN_ARTIFACT
The main artifact must be the completed iteration2 head/ONNX run. Final-test
evaluation is always disabled. Completed, independently verified variants can
be resumed only when configuration, source evidence and protocol hashes match.
"""

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import uuid


ROOT = Path(__file__).resolve().parents[2]
for vendor in (".vendor-hf", ".vendor-onnx-compat"):
    path = ROOT / "project" / vendor
    if path.is_dir():
        sys.path.insert(0, str(path))
sys.path.insert(0, str(ROOT / "project" / "src"))

from threadpoolctl import threadpool_limits
from ragstudy.experiment import run
from ragstudy.features import FEATURE_NAMES
from ragstudy.metrics import paired_group_bootstrap


VARIANTS = ("negative_cap10", "hard_bm25_cap50", "balanced_query_classes",
            "training_seed20261115", "training_seed20261116", "rank_features_only")
RANK_FEATURES = ["bm25_reciprocal_rank", "semantic_reciprocal_rank",
                 "missing_bm25_rank", "missing_semantic_rank"]
SOURCE_EVIDENCE = ("results.json", "configuration.json", "data_audit.json",
                   "split_manifest.json", "candidate_runs.json", "per_query_metrics.json",
                   "experiments.sqlite", "search_bundle.joblib")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def configuration_hash(config):
    raw = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".part")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def load_verifier():
    specification = importlib.util.spec_from_file_location(
        "ablation_independent_verifier", ROOT / "project/scripts/verify_experiment.py")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def build_variant_config(base, result, name, source_relative):
    """Change one training factor while keeping raw features and cohort fixed."""
    if name not in VARIANTS:
        raise ValueError("Unknown predeclared training ablation: " + name)
    if (base.get("max_proxy_negatives") != 50 or base.get("negative_sampling", "random") != "random"
            or base.get("balance_classes", False)
            or base.get("training_seed", base["seed"]) != 20261114
            or base.get("features", FEATURE_NAMES) != FEATURE_NAMES):
        raise ValueError("The source is not the declared unmodified training baseline")
    if base.get("final_test_enabled") is not False or result["final_test_evaluated"]:
        raise ValueError("Ablations require a source run with the final test locked")
    parameters = {kind: deepcopy(result["selected_models"][kind]["parameters"])
                  for kind in ("logistic", "random_forest")}
    config = deepcopy(base)
    config.update({"final_test_enabled": False, "fixed_model_parameters": parameters,
                   "model_source_run": source_relative, "reuse_retrieval_run": source_relative,
                   "analysis_name": "head_onnx__" + name, "ablation_name": name})
    changes = {
        "negative_cap10": {"max_proxy_negatives": 10},
        "hard_bm25_cap50": {"negative_sampling": "hard_bm25", "max_proxy_negatives": 50},
        "balanced_query_classes": {"balance_classes": True},
        "training_seed20261115": {"training_seed": 20261115},
        "training_seed20261116": {"training_seed": 20261116},
        "rank_features_only": {"features": RANK_FEATURES},
    }
    config.update(deepcopy(changes[name]))
    return config


def validate_main_source(root, source, verifier):
    source = source.resolve()
    if source.parent != (root / "project/artifacts").resolve() or not source.is_dir():
        raise ValueError("Use a completed artifact directly inside this workspace's project/artifacts")
    config, result = read_json(source / "configuration.json"), read_json(source / "results.json")
    planned = read_json(root / "project/configs/iteration2/head_onnx.json")
    policy_keys = ("source", "seed", "head_words", "near_duplicate_threshold", "semantic_backend",
                   "lsa_dimensions", "top_k", "encoder_path", "document_policy", "encoder_batch_size",
                   "encoder_max_seq_length", "encoder_threads", "split_lock_manifest", "additional_query_groups")
    if any(config.get(key) != planned.get(key) for key in policy_keys):
        raise ValueError("Source retrieval/cohort policy differs from the iteration2 head_onnx plan")
    if (config.get("semantic_backend") != "onnx" or config.get("document_policy") != "head"
            or result["data_audit"].get("split_policy") != "historical_split_lock_with_component_quarantine"
            or not config.get("split_lock_manifest") or not config.get("additional_query_groups")):
        raise ValueError("Use the iteration2 ONNX historical-lock/quarantine source")
    if config.get("fixed_model_parameters") is not None or not result.get("cross_validation"):
        raise ValueError("The main source must select model parameters by training-component CV")
    for kind in ("logistic", "random_forest"):
        if result["selected_models"][kind].get("cv_ndcg@10") is None:
            raise ValueError("Missing training CV provenance for " + kind)
    if config.get("final_test_enabled") is not False or result["final_test_evaluated"]:
        raise ValueError("The main source must leave final-test evaluation disabled")
    lock = (root / config["split_lock_manifest"]).resolve()
    if not lock.is_relative_to(root):
        raise ValueError("Historical split lock escapes the workspace")
    if result.get("protocol_input_hashes", {}).get(config["split_lock_manifest"]) != sha256(lock):
        raise ValueError("Historical split-lock hash differs from the main run")
    check = verifier.verify_experiment(source, write=False)
    if check["status"] != "passed" or check["checks"].get("exact_candidate_union") != "saved_retrieval_runs":
        raise ValueError("Main retrieval evidence failed independent verification: " + check.get("error", str(check["checks"])))
    return config, result


def comparison_to_main(source_rows, variant_rows, config):
    main = {row["query_id"]: row for row in source_rows if row["split"] == "dev"}
    variant = {row["query_id"]: row for row in variant_rows if row["split"] == "dev"}
    if set(main) != set(variant) or any(main[qid]["group_id"] != variant[qid]["group_id"] for qid in main):
        raise ValueError("A training ablation changed the development cohort")
    comparisons = []
    for model in ("logistic", "random_forest"):
        rows = [{"query_id": qid, "group_id": main[qid]["group_id"],
                 "main": main[qid][model], "variant": variant[qid][model]} for qid in main]
        for metric in ("ndcg@10", "recall@20"):
            comparisons.append({"model": model, **paired_group_bootstrap(
                rows, "main", "variant", metric,
                iterations=config["bootstrap_iterations"], seed=config["seed"])})
    return comparisons


def completed_matching_artifacts(root, config):
    """Inspect only small configuration files one level below artifact storage."""
    target = configuration_hash(config)
    matches = []
    for directory in (root / "project/artifacts").iterdir():
        path = directory / "configuration.json"
        if (directory.is_dir() and directory.resolve().parent == (root / "project/artifacts").resolve()
                and path.is_file() and (directory / "results.json").is_file()):
            if configuration_hash(read_json(path)) == target:
                matches.append(directory)
    return sorted(matches)


def run_ablations(source, summary_path=None, *, dry_run=False, root=ROOT):
    root = Path(root).resolve()
    source = (root / source).resolve() if not Path(source).is_absolute() else Path(source).resolve()
    summary_path = Path(summary_path) if summary_path is not None else root / "project/results/iteration2_ablations.json"
    if not summary_path.is_absolute():
        summary_path = root / summary_path
    summary_path = summary_path.resolve()
    if not summary_path.is_relative_to(root / "project/results"):
        raise ValueError("The summary must stay inside this workspace's project/results")
    verifier = load_verifier()
    with threadpool_limits(limits=2):
        base, result = validate_main_source(root, source, verifier)
    plan_path = root / "project/results/iteration2_plan.json"
    plan = read_json(plan_path)
    if tuple(plan["robustness_one_factor_at_a_time"]) != VARIANTS:
        raise ValueError("The predeclared six-variant plan changed; review the runner before execution")
    source_relative = source.relative_to(root).as_posix()
    source_hashes = {name: sha256(source / name) for name in SOURCE_EVIDENCE}
    code_paths = sorted((root / "project/src/ragstudy").glob("*.py"))
    code_paths += [root / "project/scripts/run_experiment.py", Path(__file__).resolve(),
                   ROOT / "project/scripts/verify_experiment.py"]
    provenance = {"matrix_protocol": "iteration2_training_ablation_v1", "source_artifact": source_relative,
                  "source_evidence_sha256": source_hashes, "plan_sha256": sha256(plan_path),
                  "protocol_input_sha256": {base["split_lock_manifest"]: sha256(root / base["split_lock_manifest"])},
                  "current_training_and_validation_source_sha256": {str(path.resolve()): sha256(path) for path in code_paths}}
    variants = {name: build_variant_config(base, result, name, source_relative) for name in VARIANTS}
    if summary_path.is_file():
        summary = read_json(summary_path)
        if summary.get("provenance") != provenance:
            raise ValueError("Existing summary belongs to different source/protocol/code hashes; preserve it and choose a new --summary path")
        if set(summary.get("variants", {})) != set(VARIANTS):
            raise ValueError("Existing summary has a different variant set")
    else:
        summary = {"status": "planned", "provenance": provenance,
                   "scope": "Six predeclared development sensitivities; fixed main training-CV parameters, no final test",
                   "inference_scope": "Paired component bootstrap on the same development cohort; exploratory multiple sensitivity comparisons",
                   "latency_scope": "Retrieval timings are inherited from the source; these runs are not fresh retrieval benchmarks",
                   "baseline_development": result["summary"]["dev"],
                   "variants": {name: {"state": "pending", "configuration": config,
                                        "configuration_sha256_canonical": configuration_hash(config)}
                                for name, config in variants.items()}}
    for name, config in variants.items():
        entry = summary["variants"][name]
        if entry.get("configuration") != config or entry.get("configuration_sha256_canonical") != configuration_hash(config):
            raise ValueError("Resume configuration hash mismatch for " + name)
    atomic_json(summary_path, summary)
    if dry_run:
        return summary
    source_rows = read_json(source / "per_query_metrics.json")
    summary["status"] = "running"
    atomic_json(summary_path, summary)
    for name, config in variants.items():
        entry = summary["variants"][name]
        try:
            existing = None
            if entry.get("artifact"):
                existing = (root / entry["artifact"]).resolve()
                if existing.parent != (root / "project/artifacts").resolve():
                    raise ValueError("Resume artifact escapes local artifact storage")
                if not (existing / "results.json").is_file():
                    existing = None
            if existing is None:
                recovered = completed_matching_artifacts(root, config)
                existing = recovered[-1] if recovered else None
            if existing is not None:
                if configuration_hash(read_json(existing / "configuration.json")) != configuration_hash(config):
                    raise ValueError("Saved artifact configuration hash changed for " + name)
                artifact, variant_result = existing, read_json(existing / "results.json")
            else:
                entry.update({"state": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()})
                atomic_json(summary_path, summary)
                print("Running fixed-parameter training ablation: " + name, flush=True)
                with threadpool_limits(limits=2):
                    artifact, variant_result = run(root, config, include_test=False)
            entry["artifact"] = artifact.relative_to(root).as_posix()
            if variant_result["final_test_evaluated"] or variant_result["cross_validation"]:
                raise ValueError("An ablation evaluated test or retuned model parameters")
            if variant_result["data_audit"] != result["data_audit"]:
                raise ValueError("An ablation changed the audited cohort")
            if (variant_result["rrf_constant"] != result["rrf_constant"]
                    or variant_result.get("weighted_rrf_selection") != result.get("weighted_rrf_selection")):
                raise ValueError("Fixed retrieval fusion baseline changed")
            reuse = variant_result.get("retrieval_reuse") or {}
            if (reuse.get("source_results_sha256") != source_hashes["results.json"]
                    or reuse.get("source_candidate_runs_sha256") != source_hashes["candidate_runs.json"]
                    or reuse.get("source_sqlite_sha256") != source_hashes["experiments.sqlite"]):
                raise ValueError("Retrieval replay source hashes differ")
            with threadpool_limits(limits=2):
                check = verifier.verify_experiment(artifact)
            if check["status"] != "passed" or check["checks"].get("exact_candidate_union") != "saved_retrieval_runs":
                raise ValueError("Independent variant verification failed: " + check.get("error", str(check["checks"])))
            variant_rows = read_json(artifact / "per_query_metrics.json")
            entry.update({"state": "complete", "run_id": variant_result["run_id"],
                          "artifact_configuration_sha256": variant_result["configuration_sha256"],
                          "development": variant_result["summary"]["dev"],
                          "paired_difference_vs_main": comparison_to_main(source_rows, variant_rows, config),
                          "verification": {"status": check["status"],
                                           "path": (artifact / "verification.json").relative_to(root).as_posix(),
                                           "sha256": sha256(artifact / "verification.json")},
                          "completed_at_utc": datetime.now(timezone.utc).isoformat()})
            entry.pop("error", None)
            if {filename: sha256(source / filename) for filename in SOURCE_EVIDENCE} != source_hashes:
                raise ValueError("The original main artifact changed during ablation execution")
            atomic_json(summary_path, summary)
        except Exception as error:
            entry.update({"state": "failed", "error": f"{type(error).__name__}: {error}"})
            summary["status"] = "incomplete"
            atomic_json(summary_path, summary)
            raise
    summary.update({"status": "complete", "completed_at_utc": datetime.now(timezone.utc).isoformat()})
    atomic_json(summary_path, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("main_artifact", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Write/review all six frozen configurations without fitting")
    args = parser.parse_args()
    try:
        summary = run_ablations(args.main_artifact, args.summary, dry_run=args.dry_run)
    except (ValueError, KeyError, OSError) as error:
        print(f"Ablation suite stopped: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"status": summary["status"],
                      "variants": {name: row["state"] for name, row in summary["variants"].items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
