"""Probe official pinned FP32 BGE ONNX against PyTorch on corpus text only.

No relevance judgments, questions or held-out labels are read. This small
single-measurement probe is an engineering diagnostic, not a latency benchmark.
"""

from datetime import datetime, timezone
from pathlib import Path
import argparse
import gc
import hashlib
import json
import sys
import time
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parents[2]
for vendor in (".vendor-hf", ".vendor-onnx", ".vendor-onnx-compat"):
    path = ROOT / "project" / vendor
    if path.is_dir():
        sys.path.insert(0, str(path))

REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
MODEL = "BAAI/bge-small-en-v1.5"
INSTRUCTION = "Represent this sentence for searching relevant passages: "


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def download_onnx(model_path):
    manifest_path = model_path / "onnx_download_manifest.json"
    target = model_path / "onnx" / "model.onnx"
    if target.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["revision"] == REVISION and digest(target) == manifest["files"][0]["sha256"]:
            return manifest
    api = f"https://huggingface.co/api/models/{MODEL}/revision/{REVISION}?blobs=true"
    request = urllib.request.Request(api, headers={"User-Agent": "SDSC5001-research"})
    with urllib.request.urlopen(request, timeout=60) as response:
        metadata = json.load(response)
    entry = next(item for item in metadata["siblings"] if item["rfilename"] == "onnx/model.onnx")
    expected = entry["lfs"]["sha256"]
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/onnx/model.onnx"
    request = urllib.request.Request(url, headers={"User-Agent": "SDSC5001-research"})
    temporary = target.with_suffix(".onnx.part")
    checksum, size = hashlib.sha256(), 0
    with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as handle:
        for chunk in iter(lambda: response.read(1024 * 1024), b""):
            handle.write(chunk)
            checksum.update(chunk)
            size += len(chunk)
            if size % (16 * 1024 * 1024) < len(chunk):
                print(f"Downloaded ONNX {size / 1024**2:.1f} MiB", flush=True)
    if checksum.hexdigest() != expected or size != entry["lfs"]["size"]:
        raise ValueError("Pinned official ONNX file failed upstream size/SHA-256 validation")
    temporary.replace(target)
    manifest = {"model": MODEL, "revision": REVISION,
                "files": [{"path": "onnx/model.onnx", "bytes": size,
                           "sha256": expected, "upstream_lfs_sha256_verified": True,
                           "source_url": url}],
                "downloaded_at_utc": datetime.now(timezone.utc).isoformat()}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def create_session(onnx_path, threads):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(onnx_path), sess_options=options,
                                providers=["CPUExecutionProvider"])


def encode_ort(session, tokenizer, texts, max_seq_length=512):
    import numpy as np
    tokens = tokenizer(texts, padding=True, truncation=True, max_length=max_seq_length,
                       return_tensors="np")
    inputs = {item.name: np.asarray(tokens[item.name], dtype=np.int64)
              for item in session.get_inputs()}
    output = session.run(None, inputs)[0]
    if output.ndim != 3:
        raise ValueError(f"Expected token embeddings before CLS pooling, got {output.shape}")
    vectors = np.asarray(output[:, 0, :], dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(vectors).all() or (norms == 0).any():
        raise ValueError("Nonfinite or zero ONNX embedding")
    return vectors / norms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--documents", type=int, choices=[16, 32], default=16)
    parser.add_argument("--model", type=Path, default=ROOT / "project/models/bge-small-en-v1.5")
    parser.add_argument("--corpus", type=Path, default=ROOT / "project/data/candidates/2026-11-12_technical_qa_audit/nvidia__TechQA-RAG-Eval/corpus.zip")
    parser.add_argument("--output", type=Path, default=ROOT / "project/results/neural_backend_probe.json")
    args = parser.parse_args()
    if args.download:
        download_onnx(args.model)
    onnx_path = args.model / "onnx/model.onnx"
    if not onnx_path.is_file():
        parser.error("Pinned ONNX file missing; rerun with --download")
    import numpy as np
    import onnxruntime as ort
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer

    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    with zipfile.ZipFile(args.corpus) as archive:
        members = sorted((item for item in archive.infolist() if not item.is_dir()), key=lambda item: item.filename)[:args.documents]
        documents = [" ".join(archive.read(item).decode("utf-8", errors="replace").split()[:256]) for item in members]
    text_hash = hashlib.sha256(json.dumps(documents, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
    print(f"Probing {len(documents)} real document heads; no qrels or query files", flush=True)
    model = SentenceTransformer(str(args.model), device="cpu", local_files_only=True)
    model.max_seq_length = 512
    model.encode(documents, batch_size=len(documents), normalize_embeddings=True, show_progress_bar=False)
    started = time.perf_counter()
    reference = np.asarray(model.encode(documents, batch_size=len(documents), normalize_embeddings=True,
                                        show_progress_bar=False), dtype=np.float32)
    torch_seconds = time.perf_counter() - started
    static_queries = [INSTRUCTION + "How can a service initialization error be diagnosed?",
                      INSTRUCTION + "Where is the product configuration log stored?"]
    query_reference = np.asarray(model.encode(static_queries, normalize_embeddings=True, show_progress_bar=False))
    print(f"PyTorch8 timed document encoding: {torch_seconds:.3f}s", flush=True)
    del model
    gc.collect()
    results = []
    for threads in (8, 4, 2):
        session = create_session(onnx_path, threads)
        encode_ort(session, tokenizer, documents)
        started = time.perf_counter()
        vectors = encode_ort(session, tokenizer, documents)
        seconds = time.perf_counter() - started
        cosine = np.sum(vectors * reference, axis=1)
        query_vectors = encode_ort(session, tokenizer, static_queries)
        query_cosine = np.sum(query_vectors * query_reference, axis=1)
        row = {"threads": threads, "seconds": seconds, "speedup_vs_torch8": torch_seconds / seconds,
               "minimum_document_cosine_vs_torch": float(cosine.min()),
               "minimum_query_cosine_vs_torch": float(query_cosine.min()),
               "maximum_absolute_embedding_difference": float(np.max(np.abs(vectors - reference))),
               "equivalence_threshold": .99999,
               "equivalence_probe_passed": bool(min(cosine.min(), query_cosine.min()) >= .99999),
               "inputs": [item.name for item in session.get_inputs()],
               "outputs": [item.name for item in session.get_outputs()]}
        print(json.dumps(row), flush=True)
        results.append(row)
        del session
        gc.collect()
    result = {"scope": "Small CPU engineering probe, one warm-up and one timed batch per backend",
              "no_query_or_qrel_files_read": True, "precision": "FP32; no quantization",
              "pooling": "CLS then L2 normalization", "document_count": len(documents),
              "input_policy": "first 256 whitespace words; tokenizer max_length 512",
              "query_instruction": INSTRUCTION, "document_instruction": "",
              "sample_document_ids": [Path(item.filename).name for item in members],
              "ordered_sample_text_sha256": text_hash,
              "model_onnx_sha256": digest(onnx_path),
              "model_torch_sha256": digest(args.model / "model.safetensors"),
              "torch_threads": 8, "torch_seconds": torch_seconds,
              "runtime": {"torch": torch.__version__, "onnxruntime": ort.__version__, "numpy": np.__version__},
              "measurements": results, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
              "references": [f"https://huggingface.co/{MODEL}/tree/{REVISION}/onnx",
                             "https://onnxruntime.ai/docs/performance/tune-performance/threading.html"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Probe saved to " + str(args.output), flush=True)


if __name__ == "__main__":
    main()
