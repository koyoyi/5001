"""Freeze one trusted primary artifact; score its retained holdout only explicitly.

Usage: python project/scripts/score_frozen_holdout.py MAIN_ARTIFACT --protocol JSON
The default prepares hashes/configuration only and never loads a model. Add
--evaluate to use the saved models and cached features once. Technical failures
can continue with --evaluate --resume under the identical frozen inputs; saved
test predictions are reused. Completed evaluations are independently verified
and returned without scoring again. Only trusted project joblib files are read.
"""
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
for folder in (".vendor-hf", ".vendor-onnx-compat"):
    vendor = ROOT / "project" / folder
    if vendor.is_dir():
        sys.path.insert(0, str(vendor))
sys.path.insert(0, str(ROOT / "project/src"))

import joblib
from threadpoolctl import threadpool_limits
from ragstudy.experiment import evaluate, summarize, store_sqlite
from ragstudy.features import FEATURE_NAMES
from ragstudy.replay import replay_retrieval

SOURCE_FILES = ("results.json", "configuration.json", "data_audit.json", "split_manifest.json",
                "candidate_runs.json", "per_query_metrics.json", "experiments.sqlite", "search_bundle.joblib")
OUTPUT_FILES = (*SOURCE_FILES, "protocol.json", "test_predictions.json")
EVALUATION_MODULES = ("__init__.py", "data.py", "features.py", "retrieval.py", "neural.py",
                      "experiment.py", "replay.py", "metrics.py")


class FreezeError(ValueError):
    pass


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_sha256(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".part")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def execution_hashes():
    # Keep training/model/metric/replay code fixed across preparation and resume.
    # Wrapper-only I/O repairs are separately recorded per attempt.
    # Generation and passage-display modules do not participate in this saved-
    # feature ranking evaluation and are intentionally outside its code identity.
    paths = [ROOT / "project/src/ragstudy" / name for name in EVALUATION_MODULES]
    return {path.relative_to(ROOT).as_posix(): sha256(path) for path in paths}


def load_verifier():
    spec = importlib.util.spec_from_file_location(
        "frozen_holdout_independent_verifier", ROOT / "project/scripts/verify_experiment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require(condition, message):
    if not condition:
        raise FreezeError(message)


def validate_source(root, source, protocol_path):
    require(source.parent == (root / "project/artifacts").resolve() and source.is_dir(),
            "MAIN_ARTIFACT must be directly inside this workspace's project/artifacts")
    require(protocol_path.is_file() and protocol_path.is_relative_to(root),
            "Protocol must be a file inside this workspace")
    config, result, protocol = (read_json(source / "configuration.json"),
                                 read_json(source / "results.json"), read_json(protocol_path))
    require(config.get("final_test_enabled") is False and not result.get("final_test_evaluated", True),
            "Primary source must have its final test locked and unscored")
    require(result.get("run_id") == source.name and
            result.get("configuration_sha256") == sha256(source / "configuration.json"),
            "Primary source identity/configuration hash mismatch")
    require(protocol.get("primary_challenger") == "logistic" and
            protocol.get("primary_baseline") == "weighted_rrf" and protocol.get("primary_metric") == "ndcg@10",
            "Protocol must preregister Logistic versus weighted RRF on nDCG@10")
    require(protocol.get("final_test_opened") is False and protocol.get("limitations") and protocol.get("stopping_rule"),
            "Protocol must retain unopened-test status, limitations and stopping rule")
    expected = {"analysis_name": protocol.get("primary_analysis_name"),
                "semantic_backend": protocol.get("semantic_backend"),
                "document_policy": protocol.get("document_policy"),
                "head_words": protocol.get("head_words"),
                "encoder_max_seq_length": protocol.get("encoder_max_seq_length")}
    require(expected["analysis_name"] == "head_onnx" and expected["semantic_backend"] == "onnx"
            and expected["document_policy"] == "head" and all(config.get(k) == v for k, v in expected.items()),
            "Primary source differs from the preregistered head_onnx pipeline")
    require(config.get("max_proxy_negatives") == 50 and config.get("negative_sampling", "random") == "random"
            and not config.get("balance_classes", False) and config.get("features", FEATURE_NAMES) == FEATURE_NAMES
            and config.get("training_seed", config.get("seed")) == 20261114
            and not config.get("ablation_name") and config.get("fixed_model_parameters") is None,
            "Primary source must retain random50, query weighting and the original nine features")
    require(result.get("feature_names") == FEATURE_NAMES and result.get("cross_validation"),
            "Primary source lacks original features or training-CV provenance")
    require(result.get("weighted_rrf_selection") is not None and
            all(result.get("selected_models", {}).get(kind, {}).get("cv_ndcg@10") is not None
                for kind in ("logistic", "random_forest")),
            "Completed source model/weighted-RRF selection is required")
    require(config.get("split_lock_manifest") == protocol.get("split_lock_manifest") and
            config.get("additional_query_groups", []) == protocol.get("additional_query_groups", []),
            "Protocol and primary cohort/group policy differ")
    lock = (root / config["split_lock_manifest"]).resolve()
    require(lock.is_relative_to(root) and lock.is_file(), "Historical split lock escapes the workspace or is missing")
    require(sha256(lock) == protocol.get("split_lock_sha256") ==
            result.get("protocol_input_hashes", {}).get(config["split_lock_manifest"]),
            "Historical split-lock SHA-256 mismatch")
    require(read_json(source / "data_audit.json") == result.get("data_audit") and
            result["data_audit"].get("split_policy") == "historical_split_lock_with_component_quarantine",
            "Primary source must retain the audited historical-lock/quarantine cohort")
    return config, result, protocol


def prepare_freeze(source, protocol_path, root):
    config, result, protocol = validate_source(root, source, protocol_path)
    identity = {"source_artifact": source.relative_to(root).as_posix(),
                "source_files_sha256": {name: sha256(source / name) for name in SOURCE_FILES},
                "protocol_path": protocol_path.relative_to(root).as_posix(),
                "protocol_sha256": sha256(protocol_path), "execution_code_sha256": execution_hashes()}
    identity_hash = object_sha256(identity)
    registry = root / "project/artifacts/frozen_holdout_registry.json"
    registration = {"identity_sha256": identity_hash, "identity": identity,
                    "run_id": "frozen_" + identity_hash[:24], "registered_at_utc": utc_now()}
    try:
        with registry.open("x", encoding="utf-8") as stream:
            json.dump(registration, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        existing = read_json(registry)
        require(existing.get("identity_sha256") == identity_hash and existing.get("identity") == identity,
                "This workspace's holdout is already frozen to a different source/protocol/code identity")
        registration = existing
    artifact = root / "project/artifacts" / registration["run_id"]
    tracked = root / "project/results" / artifact.name
    artifact.mkdir(exist_ok=True)
    tracked.mkdir(parents=True, exist_ok=True)
    manifest_path = artifact / "freeze_manifest.json"
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        require(manifest.get("identity") == identity and manifest.get("identity_sha256") == identity_hash,
                "Frozen manifest identity mismatch")
        require(sha256(artifact / "configuration.json") == manifest["frozen_configuration_sha256"] and
                sha256(artifact / "protocol.json") == identity["protocol_sha256"],
                "Frozen configuration or protocol snapshot hash mismatch")
        return artifact, tracked, manifest, config, result, protocol
    frozen_config = deepcopy(config)
    frozen_config.update({"final_test_enabled": True, "analysis_name": "frozen_head_onnx",
                          "frozen_source_artifact": identity["source_artifact"],
                          "heldout_protocol_sha256": identity["protocol_sha256"]})
    frozen_config["comparison_baselines"] = list(dict.fromkeys(
        [*config.get("comparison_baselines", []), "weighted_rrf"]))
    atomic_json(artifact / "configuration.json", frozen_config)
    shutil.copyfile(protocol_path, artifact / "protocol.json")
    manifest = {"run_id": artifact.name, "identity_sha256": identity_hash, "identity": identity,
                "frozen_at_utc": registration["registered_at_utc"], "status": "prepared", "attempts": [],
                "frozen_configuration_sha256": sha256(artifact / "configuration.json"),
                "primary_contrast": {"baseline": "weighted_rrf", "challenger": "logistic", "metric": "ndcg@10"},
                "selected_model_parameters": deepcopy(result["selected_models"]),
                "selected_weighted_rrf": deepcopy(result["weighted_rrf_selection"]),
                "limitations": deepcopy(protocol["limitations"]),
                "new_training_or_indexing": False, "heldout_scored": False}
    for name in ("configuration.json", "protocol.json"):
        shutil.copyfile(artifact / name, tracked / name)
    save_manifest(artifact, tracked, manifest)
    return artifact, tracked, manifest, config, result, protocol


def save_manifest(artifact, tracked, manifest):
    atomic_json(artifact / "freeze_manifest.json", manifest)
    atomic_json(tracked / "freeze_manifest.json", manifest)


@contextmanager
def evaluation_lock(artifact):
    # OS-held locks release even after a process crash; the file is an enduring
    # journal marker, not an unrecoverable stale PID flag.
    stream = (artifact / "evaluation.lock").open("a+b")
    try:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        stream.close()
        raise FreezeError("Another process is evaluating this frozen holdout") from error
    try:
        yield
    finally:
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def require_current_identity(source, protocol_path, manifest):
    identity = manifest["identity"]
    require({name: sha256(source / name) for name in SOURCE_FILES} == identity["source_files_sha256"],
            "Frozen source/model/candidate SHA-256 mismatch")
    require(sha256(protocol_path) == identity["protocol_sha256"], "Frozen protocol SHA-256 mismatch")
    require(execution_hashes() == identity["execution_code_sha256"], "Frozen training/model/metric code SHA-256 mismatch")


def mirror_results(tracked, result, rows):
    atomic_json(tracked / "results.json", result)
    with (tracked / "per_query_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = ["query_id", "group_id", "split", "method", "ndcg@10", "recall@20", "candidate_recall"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            for method in row["rankings"]:
                writer.writerow({name: row[name] for name in ("query_id", "group_id", "split", "candidate_recall")}
                                | {"method": method} | row[method])


def complete_evaluation(root, source, protocol_path, artifact, tracked, manifest, base_config, main_result, protocol):
    started = time.perf_counter()
    verifier = load_verifier()
    check = verifier.verify_experiment(source, write=False)
    require(check["status"] == "passed" and check["checks"].get("exact_candidate_union") == "saved_retrieval_runs",
            "Primary source failed independent verification: " + check.get("error", str(check["checks"])))
    frozen_config = read_json(artifact / "configuration.json")
    bundle = joblib.load(source / "search_bundle.joblib")
    require(bundle["configuration"] == base_config, "Source model bundle/configuration mismatch")
    require(bundle.get("rrf_constant") == main_result["rrf_constant"] and
            bundle.get("weighted_rrf_selection") == main_result["weighted_rrf_selection"],
            "Saved model bundle and source RRF selections differ")
    require(set(bundle["models"]) == {"logistic", "random_forest"} and
            all(model.get("feature_indices", list(range(len(FEATURE_NAMES)))) == list(range(len(FEATURE_NAMES)))
                for model in bundle["models"].values()), "Source model feature selection differs from the fixed nine features")
    require(all(model.get("parameters") == main_result["selected_models"][kind]["parameters"]
                for kind, model in bundle["models"].items()),
            "Saved fitted-model parameters differ from source selection metadata")
    dataset = {"document_ids": bundle["document_ids"], "documents": bundle["documents"],
               "queries": read_json(source / "split_manifest.json"), "audit": main_result["data_audit"]}
    _, _, records, replay = replay_retrieval(source, dataset, base_config)
    test_records = [record for record in records if record["query"]["split"] == "test"]
    require(bool(test_records), "Frozen retained holdout is empty")
    prediction_path = artifact / "test_predictions.json"
    if prediction_path.is_file():
        predictions = read_json(prediction_path)
        require(predictions["identity_sha256"] == manifest["identity_sha256"] and
                predictions["rows_sha256"] == object_sha256(predictions["rows"]),
                "Saved frozen predictions identity/content hash mismatch")
        if manifest.get("test_predictions_sha256"):
            require(sha256(prediction_path) == manifest["test_predictions_sha256"], "Saved prediction file SHA-256 mismatch")
        test_rows = predictions["rows"]
        manifest["attempts"][-1]["reused_saved_predictions"] = True
    else:
        test_rows = evaluate(test_records, bundle["models"], main_result["rrf_constant"], dataset,
                             weighted=main_result["weighted_rrf_selection"])
        atomic_json(prediction_path, {"identity_sha256": manifest["identity_sha256"],
                                     "predicted_at_utc": utc_now(), "rows_sha256": object_sha256(test_rows),
                                     "rows": test_rows})
        manifest["attempts"][-1]["reused_saved_predictions"] = False
    require({row["query_id"] for row in test_rows} == {r["query"]["query_id"] for r in test_records}
            and len(test_rows) == len(test_records), "Frozen test prediction query cohort differs")
    manifest["test_predictions_sha256"] = sha256(prediction_path)
    manifest["heldout_scored"] = True
    save_manifest(artifact, tracked, manifest)
    old_rows = read_json(source / "per_query_metrics.json")
    require(all(row["split"] in ("train", "dev") for row in old_rows), "Primary source contains pre-existing test metrics")
    rows = old_rows + test_rows
    summary = summarize(rows, bundle["models"], frozen_config)
    primary = next(item for item in summary["test"]["paired_contrasts"]
                   if item["baseline"] == "weighted_rrf" and item["challenger"] == "logistic" and item["metric"] == "ndcg@10")
    result = deepcopy(main_result)
    result.update({"run_id": artifact.name, "scope": "frozen_provided_qrels_holdout", "final_test_evaluated": True,
                   "test_review_status": "Preregistered provided-qrels evaluation; review/source-label limitations retained",
                   "configuration_sha256": sha256(artifact / "configuration.json"), "summary": summary,
                   "primary_holdout_contrast": primary, "offline_index_seconds": 0.,
                   "source_offline_index_seconds": main_result.get("offline_index_seconds"),
                   "source_sha256_at_run_start": manifest["identity"]["execution_code_sha256"],
                   "retrieval_reuse": {**replay, "scope": "Reuse saved indexes/candidates/raw features and already fitted models; no indexing, retrieval, fitting, CV or selection"},
                   "frozen_evaluation": {"identity_sha256": manifest["identity_sha256"],
                                         "frozen_at_utc": manifest["frozen_at_utc"],
                                         "source_artifact": manifest["identity"]["source_artifact"],
                                         "protocol_sha256": manifest["identity"]["protocol_sha256"],
                                         "source_files_sha256": manifest["identity"]["source_files_sha256"],
                                         "primary_contrast": manifest["primary_contrast"],
                                         "original_train_dev_rows_preserved": True,
                                         "no_new_training_or_indexing": True},
                   "limitations": list(dict.fromkeys([*main_result.get("limitations", []), *protocol["limitations"]])),
                   "latency_scope": "Retrieval/feature timings inherited from the primary run; frozen-model rerank times measured only for newly evaluated test rows; no new index or fit",
                   "wall_seconds": time.perf_counter() - started})
    result["protocol_input_hashes"] = {**main_result.get("protocol_input_hashes", {}),
                                      protocol_path.relative_to(root).as_posix(): sha256(protocol_path)}
    for name in ("data_audit.json", "split_manifest.json", "candidate_runs.json"):
        shutil.copyfile(source / name, artifact / name)
    atomic_json(artifact / "per_query_metrics.json", rows)
    atomic_json(artifact / "results.json", result)
    # A failed prior publication may have left SQLite partially constructed.
    # Build a fresh sibling and atomically replace it, never alter the source.
    database = artifact / ("experiments." + uuid.uuid4().hex + ".sqlite")
    store_sqlite(database, dataset, records, rows, artifact.name, frozen_config)
    database.replace(artifact / "experiments.sqlite")
    frozen_bundle = dict(bundle, configuration=frozen_config)
    temporary_bundle = artifact / ("search_bundle." + uuid.uuid4().hex + ".joblib")
    joblib.dump(frozen_bundle, temporary_bundle, compress=0)
    temporary_bundle.replace(artifact / "search_bundle.joblib")
    mirror_results(tracked, result, rows)
    if (source / "development_metrics.png").is_file():
        for destination in (artifact, tracked):
            shutil.copyfile(source / "development_metrics.png", destination / "development_metrics.png")
    verification = verifier.verify_experiment(artifact)
    require(verification["status"] == "passed", "Frozen output failed independent verification: " + verification.get("error", "unknown"))
    require_current_identity(source, protocol_path, manifest)
    manifest.update({"status": "completed", "completed_at_utc": utc_now(),
                     "output_sha256": {name: sha256(artifact / name) for name in OUTPUT_FILES}})
    manifest["attempts"][-1].update({"status": "completed", "finished_at_utc": utc_now()})
    save_manifest(artifact, tracked, manifest)
    return result


def freeze_holdout(source, protocol_path, *, evaluate=False, resume=False, root=ROOT):
    root = Path(root).resolve()
    source = (root / source).resolve()
    protocol_path = (root / protocol_path).resolve()
    require(not resume or evaluate, "--resume requires explicit --evaluate")
    artifact, tracked, manifest, config, result, protocol = prepare_freeze(source, protocol_path, root)
    if not evaluate:
        return artifact, manifest
    with evaluation_lock(artifact), threadpool_limits(limits=2):
        manifest = read_json(artifact / "freeze_manifest.json")
        require_current_identity(source, protocol_path, manifest)
        if manifest["status"] == "completed":
            require(all(sha256(artifact / name) == value for name, value in manifest["output_sha256"].items()),
                    "Completed frozen output SHA-256 mismatch")
            verification = load_verifier().verify_experiment(artifact, write=False)
            require(verification["status"] == "passed", "Existing frozen result failed verification")
            return artifact, read_json(artifact / "results.json")
        require(manifest["status"] == "prepared" or resume, "Prior evaluation attempt exists; use explicit --resume with identical frozen inputs")
        manifest["status"] = "evaluating"
        manifest["attempts"].append({"attempt": len(manifest["attempts"]) + 1, "started_at_utc": utc_now(),
                                     "resume": bool(resume), "wrapper_sha256": sha256(Path(__file__))})
        save_manifest(artifact, tracked, manifest)
        try:
            result = complete_evaluation(root, source, protocol_path, artifact, tracked, manifest, config, result, protocol)
        except Exception as error:
            manifest["status"] = "failed"
            manifest["attempts"][-1].update({"status": "failed", "finished_at_utc": utc_now(),
                                             "error": f"{type(error).__name__}: {error}"})
            save_manifest(artifact, tracked, manifest)
            raise
    return artifact, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("main_artifact", type=Path)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--evaluate", action="store_true", help="Explicitly evaluate the fixed retained holdout")
    parser.add_argument("--resume", action="store_true", help="Continue an identical frozen technical failure; reuse saved predictions")
    args = parser.parse_args()
    try:
        artifact, payload = freeze_holdout(args.main_artifact, args.protocol, evaluate=args.evaluate, resume=args.resume)
    except (FreezeError, OSError) as error:
        parser.exit(2, f"Frozen holdout rejected: {error}\n")
    print(json.dumps({"artifact": str(artifact), "status": "completed" if args.evaluate else payload["status"],
                      "heldout_scored": bool(args.evaluate or payload.get("heldout_scored")),
                      "primary_contrast": payload.get("primary_holdout_contrast", payload.get("primary_contrast"))}, indent=2))


if __name__ == "__main__":
    main()
