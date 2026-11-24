"""Compare isolated CPU and DirectML FP32 inference on corpus heads only.

Run --backend cpu first, then --backend dml in a fresh process. No query/qrel
files, labels, training, driver installation or system CUDA changes are used.
"""

from datetime import datetime, timezone
from pathlib import Path
import argparse
import ctypes
import hashlib
import json
import shutil
import subprocess
import sys
import time
import uuid


ROOT = Path(__file__).resolve().parents[2]


def adapters():
    """Read DXGI adapter enumeration, matching DirectML's device_id order."""
    class GUID(ctypes.Structure):
        _fields_ = [("a", ctypes.c_uint32), ("b", ctypes.c_uint16),
                    ("c", ctypes.c_uint16), ("d", ctypes.c_ubyte * 8)]

    class LUID(ctypes.Structure):
        _fields_ = [("low", ctypes.c_uint32), ("high", ctypes.c_int32)]

    class DESC(ctypes.Structure):
        _fields_ = [("description", ctypes.c_wchar * 128), ("vendor", ctypes.c_uint32),
                    ("device", ctypes.c_uint32), ("subsystem", ctypes.c_uint32),
                    ("revision", ctypes.c_uint32), ("video_memory", ctypes.c_size_t),
                    ("system_memory", ctypes.c_size_t), ("shared_memory", ctypes.c_size_t),
                    ("luid", LUID), ("flags", ctypes.c_uint32)]

    factory = ctypes.c_void_p()
    iid = GUID.from_buffer_copy(uuid.UUID("770aae78-f26f-4dba-a829-253c83d1b387").bytes_le)
    create = ctypes.WinDLL("dxgi").CreateDXGIFactory1
    create.argtypes = [ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p)]
    create.restype = ctypes.c_long
    if create(ctypes.byref(iid), ctypes.byref(factory)) < 0:
        raise RuntimeError("DXGI adapter enumeration failed")

    def method(obj, index, result, *arguments):
        vtable = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        return ctypes.WINFUNCTYPE(result, ctypes.c_void_p, *arguments)(vtable[index])

    values = []
    try:
        enumerate_adapter = method(factory, 12, ctypes.c_long, ctypes.c_uint32,
                                   ctypes.POINTER(ctypes.c_void_p))
        for index in range(16):
            adapter = ctypes.c_void_p()
            status = enumerate_adapter(factory, index, ctypes.byref(adapter))
            if status < 0:
                break
            try:
                description = DESC()
                status = method(adapter, 10, ctypes.c_long, ctypes.POINTER(DESC))(
                    adapter, ctypes.byref(description))
                if status >= 0:
                    values.append({"device_id": index, "description": description.description,
                                   "vendor_id": int(description.vendor), "pci_device_id": int(description.device),
                                   "dedicated_video_memory_mib": description.video_memory / 1024**2,
                                   "adapter_luid": f"{description.luid.high & 0xffffffff:08x}{description.luid.low:08x}"})
            finally:
                method(adapter, 2, ctypes.c_ulong)(adapter)
    finally:
        method(factory, 2, ctypes.c_ulong)(factory)
    return values


def gpu_memory():
    command = shutil.which("nvidia-smi")
    if not command:
        return {"status": "nvidia-smi not on PATH"}
    result = subprocess.run([command, "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
    return {"status": "read" if result.returncode == 0 else "unavailable", "csv": result.stdout.strip()}


def sha256(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def encode(session, tokenizer, texts):
    import numpy as np
    tokens = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="np")
    inputs = {item.name: np.asarray(tokens[item.name], dtype=np.int64) for item in session.get_inputs()}
    hidden = session.run(["last_hidden_state"], inputs)[0]
    vectors = np.asarray(hidden[:, 0, :], dtype=np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["cpu", "dml"], required=True)
    parser.add_argument("--device-id", type=int)
    parser.add_argument("--batch64-only", action="store_true", help="Probe 64 real heads once; compare the first 32 vectors with the saved CPU reference")
    args = parser.parse_args()
    if args.batch64_only and args.backend != "dml":
        parser.error("--batch64-only is a DirectML-only engineering probe")
    for folder in (".vendor-hf", ".vendor-onnx-compat" if args.backend == "cpu" else ".vendor-onnx-dml"):
        sys.path.insert(0, str(ROOT / "project" / folder))
    sys.path.insert(0, str(ROOT / "project/src"))
    import numpy as np
    import onnxruntime as ort
    from transformers import AutoTokenizer
    from ragstudy.neural import document_heads_from_zip, INSTRUCTION

    model = ROOT / "project/models/bge-small-en-v1.5"
    corpus = ROOT / "project/data/candidates/2026-11-12_technical_qa_audit/nvidia__TechQA-RAG-Eval/corpus.zip"
    ids, documents = document_heads_from_zip(corpus)
    count = 64 if args.batch64_only else 32
    ids, documents = ids[:count], documents[:count]
    sample_hash = hashlib.sha256(json.dumps(documents, ensure_ascii=False).encode("utf-8")).hexdigest()
    onnx_hash = sha256(model / "onnx/model.onnx")
    cache = ROOT / "project/data/cache/gpu_probe"
    cache.mkdir(parents=True, exist_ok=True)
    reference_file = cache / "cpu_reference.npz"
    reference_meta = cache / "cpu_reference.json"
    query_texts = [INSTRUCTION + "How can a service initialization error be diagnosed?",
                   INSTRUCTION + "Where is the product configuration log stored?"]
    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 8 if args.backend == "cpu" else 2
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    metadata = {"backend": args.backend, "onnxruntime_version": ort.__version__,
                "onnxruntime_path": str(Path(ort.__file__).resolve()),
                "model_onnx_sha256": onnx_hash, "sample_text_sha256": sample_hash,
                "document_ids": ids, "head_words": 256, "max_seq_length": 512,
                "precision": "FP32 ONNX output; no model quantization",
                "no_query_or_qrel_files_read": True,
                "scope": "Small engineering probe, one warm and one timed batch; not a steady-state latency benchmark"}
    if args.backend == "cpu":
        providers = ["CPUExecutionProvider"]
    else:
        options.enable_mem_pattern = False
        available = adapters()
        device = args.device_id if args.device_id is not None else next(
            item["device_id"] for item in available if item["vendor_id"] == 0x10DE)
        metadata.update({"adapters": available, "selected_device_id": device,
                         "selected_adapter": next(item for item in available if item["device_id"] == device),
                         "gpu_memory_before": gpu_memory()})
        providers = [("DmlExecutionProvider", {"device_id": str(device)})]
        old = json.loads(reference_meta.read_text(encoding="utf-8"))
        reference_text_hash = hashlib.sha256(json.dumps(documents[:32], ensure_ascii=False).encode("utf-8")).hexdigest()
        if old["model_onnx_sha256"] != onnx_hash or old["sample_text_sha256"] != reference_text_hash:
            raise ValueError("CPU reference model/text identity mismatch")
    print(json.dumps({"backend": args.backend, "providers": providers,
                      "adapter": metadata.get("selected_adapter")}), flush=True)
    session = ort.InferenceSession(str(model / "onnx/model.onnx"), sess_options=options, providers=providers)
    session.disable_fallback()
    metadata["active_providers"] = session.get_providers()
    expected_provider = "CPUExecutionProvider" if args.backend == "cpu" else "DmlExecutionProvider"
    if expected_provider not in session.get_providers():
        raise RuntimeError("Requested provider did not initialize")
    measurements = []
    vectors = None
    for size in ((64,) if args.batch64_only else (16, 32)):
        batch = documents[:size]
        encode(session, tokenizer, batch)
        started = time.perf_counter()
        vectors = encode(session, tokenizer, batch)
        elapsed = time.perf_counter() - started
        measurements.append({"batch_size": size, "seconds": elapsed, "documents_per_second": size / elapsed})
        print(json.dumps(measurements[-1]), flush=True)
    query_vectors = encode(session, tokenizer, query_texts)
    metadata["measurements"] = measurements
    if args.backend == "cpu":
        np.savez(reference_file, documents=vectors, queries=query_vectors)
        reference_meta.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    else:
        with np.load(reference_file, allow_pickle=False) as reference:
            comparison = vectors[:len(reference["documents"])]
            cosine = np.sum(comparison * reference["documents"], axis=1)
            query_cosine = np.sum(query_vectors * reference["queries"], axis=1)
            maximum = float(np.max(np.abs(comparison - reference["documents"])))
        metadata.update({"minimum_document_cosine_vs_cpu": float(cosine.min()),
                         "minimum_query_cosine_vs_cpu": float(query_cosine.min()),
                         "maximum_absolute_embedding_difference": maximum,
                         "equivalence_threshold": .99999,
                         "numerical_comparison_documents": len(comparison),
                         "equivalence_probe_passed": bool(min(cosine.min(), query_cosine.min()) >= .99999),
                         "cpu_reference": old, "gpu_memory_after": gpu_memory(),
                         "reference_vectors_sha256": sha256(reference_file),
                         "references": ["https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html"]})
        path = ROOT / "project/results" / ("neural_gpu_batch64_probe.json" if args.batch64_only else "neural_gpu_probe.json")
        path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
