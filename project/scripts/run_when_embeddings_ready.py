"""Launch a fresh primary process only after a verified corpus warmup completes."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "project/configs/iteration2/head_onnx.json")
    args = parser.parse_args()
    cache, config_path = args.cache.resolve(), args.config.resolve()
    if not cache.is_relative_to(ROOT / "project/data/cache/embeddings") or not config_path.is_relative_to(ROOT):
        parser.error("Use this workspace's embedding cache and configuration.")
    config_hash = digest(config_path)
    config = read_json(config_path)
    if config["semantic_backend"] != "onnx" or config.get("document_policy") != "head":
        parser.error("This coordinator expects the declared head/ONNX primary configuration.")
    print("Waiting for complete corpus checkpoint: " + cache.name, flush=True)
    complete = cache / "warmup.json"
    while not complete.is_file():
        if digest(config_path) != config_hash:
            raise RuntimeError("Primary configuration changed while waiting; review and restart explicitly.")
        time.sleep(15)
    warm = read_json(complete)
    metadata = warm["metadata"]
    expected = {"max_seq_length": config["encoder_max_seq_length"],
                "batch_size": config["encoder_batch_size"], "intra_op_threads": config["encoder_threads"]}
    if (metadata["cache_key"] != cache.name or warm["head_words"] != config["head_words"]
            or any(metadata[key] != value for key, value in expected.items())
            or metadata["documents"] != metadata["documents_encoded_this_run"] + metadata["cached_documents_reused"]):
        raise RuntimeError("Completed warmup does not match the declared primary encoder settings.")
    corpus = ROOT / config["source"] / "corpus.zip"
    if warm["corpus_zip_sha256"] != digest(corpus) or digest(config_path) != config_hash:
        raise RuntimeError("Corpus or configuration identity changed before launch.")
    print("Full corpus cache ready; launching the primary experiment.", flush=True)
    before = {path.name for path in (ROOT / "project/artifacts").iterdir()}
    subprocess.run([sys.executable, str(ROOT / "project/scripts/run_experiment.py"),
                    "--config", str(config_path)], cwd=ROOT, check=True)
    completed = [path for path in (ROOT / "project/artifacts").iterdir()
                 if path.name not in before and (path / "results.json").is_file()
                 and read_json(path / "configuration.json") == config]
    if len(completed) != 1:
        raise RuntimeError("Could not uniquely locate the completed primary result.")
    tracked = ROOT / "project/results" / completed[0].name
    for target in (completed[0], tracked):
        (target / "embedding_build.json").write_text(json.dumps(warm, indent=2), encoding="utf-8")
    pointer = {"artifact": str(completed[0].relative_to(ROOT)).replace("\\", "/"),
               "configuration_sha256": config_hash, "cache_key": cache.name,
               "full_embedding_build_evidence": str((tracked / "embedding_build.json").relative_to(ROOT)).replace("\\", "/")}
    (ROOT / "project/results/iteration2_main_run.json").write_text(json.dumps(pointer, indent=2), encoding="utf-8")
    print(json.dumps(pointer, indent=2), flush=True)


if __name__ == "__main__":
    main()
