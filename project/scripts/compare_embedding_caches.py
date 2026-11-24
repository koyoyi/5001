"""Compare common FP32 vectors from two content-verified corpus caches."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def read_cache(folder):
    identity = json.loads((folder / "identity.json").read_text(encoding="utf-8"))
    count = identity["identity"]["document_count"]
    dimensions = identity["identity"]["dimensions"]
    vectors = np.empty((count, dimensions), dtype=np.float32)
    valid = np.zeros(count, dtype=bool)
    for sidecar in folder.glob("batch_*.json"):
        batch = json.loads(sidecar.read_text(encoding="utf-8"))
        path = sidecar.with_suffix(".npy")
        if batch["cache_key"] != identity["cache_key"]:
            raise ValueError("Batch identity differs from its cache")
        if hashlib.sha256(path.read_bytes()).hexdigest() != batch["vectors_sha256"]:
            raise ValueError("Batch checksum mismatch")
        block = np.load(path, allow_pickle=False)
        if block.shape != (batch["end"] - batch["start"], dimensions) or not np.isfinite(block).all():
            raise ValueError("Invalid cached vector shape/values")
        vectors[batch["start"]:batch["end"]] = block
        valid[batch["start"]:batch["end"]] = True
    return identity, vectors, valid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "project/results/neural_corpus_backend_comparison.json")
    args = parser.parse_args()
    left, a, mask_a = read_cache(args.reference)
    right, b, mask_b = read_cache(args.candidate)
    for key in ("ordered_document_text_sha256", "document_count", "model_onnx_sha256", "tokenizer_and_config_sha256",
                "query_instruction", "document_instruction", "max_seq_length", "normalization", "precision", "dimensions"):
        if left["identity"][key] != right["identity"][key]:
            raise ValueError("Semantic cache identity differs: " + key)
    selected = mask_a & mask_b
    if not selected.any():
        raise ValueError("No common completed document vectors")
    a, b = a[selected].astype(np.float64), b[selected].astype(np.float64)
    cosine = np.sum(a * b, axis=1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
    report = {"scope": "Corpus-only numerical backend comparison; no queries, judgments or ranking metrics read",
              "reference_cache": left["cache_key"], "candidate_cache": right["cache_key"],
              "reference_provider": left["identity"]["execution_provider"],
              "candidate_provider": right["identity"]["execution_provider"],
              "common_documents": int(selected.sum()), "total_documents": len(selected),
              "minimum_cosine": float(cosine.min()), "mean_cosine": float(cosine.mean()),
              "max_absolute_difference": float(np.max(np.abs(a - b))),
              "predefined_minimum_cosine": .99999, "passed": bool((cosine >= .99999).all())}
    if not args.output.resolve().is_relative_to(ROOT / "project/results"):
        parser.error("Save comparison evidence inside project/results")
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
