"""Local FP32 ONNX BGE retrieval with verified, resumable embedding batches.

The caller supplies the same document heads used by lexical retrieval. Cache
hits retain their original computation provenance; cache loading is never
reported as a fresh full-corpus encoding. Runtime objects are reconstructed
lazily after unpickling and are not stored in joblib artifacts.
"""

from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import hashlib
import json
import os
import time
import uuid
import zipfile

import numpy as np


INSTRUCTION = "Represent this sentence for searching relevant passages: "
CACHE_FORMAT = 1


def _sha256(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def _ordered_text_hash(texts):
    checksum = hashlib.sha256()
    for text in texts:
        raw = text.encode("utf-8")
        checksum.update(len(raw).to_bytes(8, "big"))
        checksum.update(raw)
    return checksum.hexdigest()


def _versions():
    values = {}
    for name in ("onnxruntime", "transformers", "tokenizers"):
        try:
            values[name] = version(name)
        except PackageNotFoundError:
            values[name] = "not_installed"
    return values


def dml_adapters():
    """Enumerate Windows DXGI adapters in DirectML device-ID order."""
    import ctypes
    class GUID(ctypes.Structure):
        _fields_ = [("a", ctypes.c_uint32), ("b", ctypes.c_uint16), ("c", ctypes.c_uint16),
                    ("d", ctypes.c_ubyte * 8)]
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
    create.argtypes, create.restype = [ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p)], ctypes.c_long
    if create(ctypes.byref(iid), ctypes.byref(factory)) < 0:
        raise RuntimeError("DXGI adapter enumeration failed")
    def method(obj, index, result, *arguments):
        vtable = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        return ctypes.WINFUNCTYPE(result, ctypes.c_void_p, *arguments)(vtable[index])
    values = []
    try:
        enumerate_adapter = method(factory, 12, ctypes.c_long, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p))
        for index in range(16):
            adapter = ctypes.c_void_p()
            if enumerate_adapter(factory, index, ctypes.byref(adapter)) < 0:
                break
            try:
                desc = DESC()
                if method(adapter, 10, ctypes.c_long, ctypes.POINTER(DESC))(adapter, ctypes.byref(desc)) >= 0:
                    values.append({"device_id": index, "description": desc.description,
                                   "vendor_id": int(desc.vendor), "pci_device_id": int(desc.device),
                                   "dedicated_video_memory_mib": desc.video_memory / 1024**2})
            finally:
                method(adapter, 2, ctypes.c_ulong)(adapter)
    finally:
        method(factory, 2, ctypes.c_ulong)(factory)
    return values


def _atomic_json(path, value):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".part")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _valid_vectors(vectors, rows, dimensions):
    return (vectors.shape == (rows, dimensions) and vectors.dtype == np.float32
            and np.isfinite(vectors).all()
            and np.allclose(np.linalg.norm(vectors, axis=1), 1., atol=1e-4, rtol=0))


class OnnxEncoderIndex:
    """Frozen official ONNX token encoder, CLS pooling, L2 and exact search."""

    def __init__(self, documents, model_path, cache_dir=None, batch_size=32,
                 max_seq_length=512, threads=4, query_instruction=INSTRUCTION,
                 provider="CPUExecutionProvider", device_id=0):
        started = time.perf_counter()
        if any(not isinstance(value, int) or value < 1
               for value in (batch_size, max_seq_length, threads)):
            raise ValueError("batch_size, max_seq_length and threads must be positive integers")
        if not all(isinstance(text, str) for text in documents):
            raise ValueError("All documents must be strings")
        self.model_path = str(Path(model_path).expanduser().resolve())
        model = Path(self.model_path)
        self.onnx_path = str(model / "onnx" / "model.onnx")
        self.batch_size, self.max_seq_length, self.threads = batch_size, max_seq_length, threads
        self.query_instruction = query_instruction
        if provider not in ("CPUExecutionProvider", "DmlExecutionProvider"):
            raise ValueError("provider must be CPUExecutionProvider or DmlExecutionProvider")
        if not isinstance(device_id, int) or device_id < 0:
            raise ValueError("device_id must be a nonnegative integer")
        self.provider, self.device_id, self.device_metadata = provider, device_id, None
        if provider == "DmlExecutionProvider":
            self.device_metadata = next((adapter for adapter in dml_adapters()
                                         if adapter["device_id"] == device_id), None)
            if self.device_metadata is None:
                raise ValueError("Requested DirectML adapter is unavailable")
        self.n_documents = len(documents)
        self._session = self._tokenizer = None
        self._runtime_initialization_seconds = 0.
        config = json.loads((model / "config.json").read_text(encoding="utf-8"))
        self.dimensions = int(config["hidden_size"])
        if max_seq_length > int(config["max_position_embeddings"]):
            raise ValueError("max_seq_length exceeds the local model's positional capacity")
        manifest = json.loads((model / "onnx_download_manifest.json").read_text(encoding="utf-8"))
        entries = [entry for entry in manifest["files"] if entry["path"] == "onnx/model.onnx"]
        if len(entries) != 1:
            raise ValueError("Exactly one pinned ONNX model entry is required")
        onnx_hash = _sha256(self.onnx_path)
        if entries[0]["sha256"] != onnx_hash:
            raise ValueError("Local ONNX model SHA-256 differs from its download manifest")
        if "bytes" in entries[0] and entries[0]["bytes"] != Path(self.onnx_path).stat().st_size:
            raise ValueError("Local ONNX model size differs from its download manifest")
        tokenizer_files = ("config.json", "tokenizer.json", "tokenizer_config.json",
                           "special_tokens_map.json", "vocab.txt", "1_Pooling/config.json")
        file_hashes = {name: _sha256(model / name) for name in tokenizer_files if (model / name).is_file()}
        for required in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"):
            if required not in file_hashes:
                raise FileNotFoundError("Missing local tokenizer file: " + required)
        pooling_path = model / "1_Pooling" / "config.json"
        if pooling_path.is_file():
            pooling = json.loads(pooling_path.read_text(encoding="utf-8"))
            if not pooling.get("pooling_mode_cls_token") or any(
                    pooling.get(key, False) for key in ("pooling_mode_mean_tokens", "pooling_mode_max_tokens",
                                                       "pooling_mode_mean_sqrt_len_tokens")):
                raise ValueError("Only the pinned CLS pooling configuration is supported")
        self.runtime_versions = _versions()
        if provider == "DmlExecutionProvider":
            self.runtime_versions["onnxruntime"] = version("onnxruntime-directml")
            self.runtime_versions["onnxruntime_distribution"] = "onnxruntime-directml"
        identity = {
            "format": CACHE_FORMAT, "ordered_document_text_sha256": _ordered_text_hash(documents),
            "document_count": len(documents), "model_onnx_sha256": onnx_hash,
            "model": manifest["model"], "revision": manifest["revision"],
            "tokenizer_and_config_sha256": file_hashes,
            "query_instruction": query_instruction, "document_instruction": "",
            "max_seq_length": max_seq_length, "normalization": "CLS then L2",
            "precision": "float32", "dimensions": self.dimensions,
            "batch_size": batch_size, "intra_op_threads": threads,
            "execution_provider": provider, "runtime_versions": self.runtime_versions,
        }
        # Preserve the existing CPU identity byte-for-byte. GPU adapters/builds
        # receive separate cache keys and never consume partial CPU vectors.
        if provider == "DmlExecutionProvider":
            identity.update({"device_id": device_id, "adapter": self.device_metadata})
        self.cache_key = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        cache_root = Path(cache_dir).expanduser().resolve() if cache_dir is not None else model / "embedding_cache"
        self.cache_path = str(cache_root / self.cache_key)
        cache = Path(self.cache_path)
        cache.mkdir(parents=True, exist_ok=True)
        _atomic_json(cache / "identity.json", {"cache_key": self.cache_key, "identity": identity})
        self.document_vectors = np.empty((len(documents), self.dimensions), dtype=np.float32)
        encoded_batches = cached_batches = corrupted_batches = encoded_documents = cached_documents = 0
        encoding_seconds = lookup_seconds = stored_encoding_seconds = 0.
        print(f"ONNX FP32 embedding cache {self.cache_key[:12]}: {len(documents)} documents", flush=True)
        for start in range(0, len(documents), batch_size):
            end = min(start + batch_size, len(documents))
            batch_hash = _ordered_text_hash(documents[start:end])
            data_path = cache / f"batch_{start:08d}_{end:08d}.npy"
            sidecar = data_path.with_suffix(".json")
            lookup_started = time.perf_counter()
            vectors, checkpoint = self._read_batch(data_path, sidecar, start, end, batch_hash)
            lookup_seconds += time.perf_counter() - lookup_started
            if vectors is not None:
                cached_batches += 1
                cached_documents += end - start
            else:
                corrupted_batches += int(data_path.exists() or sidecar.exists())
                encode_started = time.perf_counter()
                vectors = np.asarray(self._encode_texts(documents[start:end]), dtype=np.float32)
                seconds = time.perf_counter() - encode_started
                if not _valid_vectors(vectors, end - start, self.dimensions):
                    raise ValueError("Encoder produced invalid, nonnormalized or wrong-shape vectors")
                temporary = data_path.with_name(data_path.name + "." + uuid.uuid4().hex + ".part")
                with temporary.open("wb") as handle:
                    np.save(handle, vectors, allow_pickle=False)
                    handle.flush()
                    os.fsync(handle.fileno())
                temporary.replace(data_path)
                checkpoint = {"cache_key": self.cache_key, "start": start, "end": end,
                              "ordered_batch_text_sha256": batch_hash,
                              "vectors_sha256": _sha256(data_path), "shape": list(vectors.shape),
                              "dtype": "float32", "encoding_wall_seconds": seconds,
                              "completed_at_utc": datetime.now(timezone.utc).isoformat()}
                _atomic_json(sidecar, checkpoint)
                encoding_seconds += seconds
                encoded_batches += 1
                encoded_documents += end - start
            stored_encoding_seconds += checkpoint["encoding_wall_seconds"]
            self.document_vectors[start:end] = vectors
            if encoded_batches and (encoded_batches % 10 == 0 or end == len(documents)):
                print(f"ONNX checkpoint {end}/{len(documents)} documents; reused={cached_documents}, encoded={encoded_documents}", flush=True)
        self.metadata = {
            "method": "frozen_onnx_fp32_cls_cosine", "neural_encoder": True,
            "local_model_path": self.model_path, "local_files_only": True,
            "model_onnx_sha256": onnx_hash, "model_revision": manifest["revision"],
            "pinned_onnx_download_manifest": manifest,
            "device": "cpu" if provider == "CPUExecutionProvider" else "directml_gpu", "precision": "float32",
            "execution_provider": provider, "device_id": device_id, "adapter": self.device_metadata,
            "pooling": "CLS", "normalize_embeddings": True, "max_seq_length": max_seq_length,
            "query_instruction": query_instruction, "document_instruction": "",
            "input_document_policy": "caller supplies identical document heads to all retrievers",
            "search": "exact dot product", "dimensions": self.dimensions, "documents": len(documents),
            "batch_size": batch_size, "intra_op_threads": threads, "inter_op_threads": 1,
            "runtime_versions": self.runtime_versions,
            "cache_key": self.cache_key, "cache_directory": self.cache_path,
            "cache_hit": bool(len(documents) and cached_documents == len(documents)),
            "cached_batches_reused": cached_batches, "cached_documents_reused": cached_documents,
            "batches_encoded_this_run": encoded_batches, "documents_encoded_this_run": encoded_documents,
            "invalid_cached_batches_recomputed": corrupted_batches,
            "encoding_wall_seconds_this_run": encoding_seconds,
            "cache_lookup_seconds_this_run": lookup_seconds,
            "runtime_initialization_seconds_this_run": self._runtime_initialization_seconds,
            "successful_cached_batch_recorded_encoding_seconds": stored_encoding_seconds,
            "construction_wall_seconds_this_run": time.perf_counter() - started,
            "timing_scope": "Encoding includes tokenization and lazy runtime startup; cached historical batch times are separate from this run",
            "cache_integrity": "Per-batch model/text identity, SHA-256, shape, float32, finite values and L2 norm checked",
            "session_lazy_reload_on_unpickle": True,
        }
        _atomic_json(cache / "latest_access.json", self.metadata)

    def _read_batch(self, path, sidecar, start, end, batch_hash):
        try:
            checkpoint = json.loads(sidecar.read_text(encoding="utf-8"))
            if (checkpoint["cache_key"] != self.cache_key or checkpoint["start"] != start
                    or checkpoint["end"] != end or checkpoint["ordered_batch_text_sha256"] != batch_hash
                    or checkpoint["dtype"] != "float32" or checkpoint["shape"] != [end - start, self.dimensions]
                    or checkpoint["vectors_sha256"] != _sha256(path)
                    or not np.isfinite(checkpoint["encoding_wall_seconds"])
                    or checkpoint["encoding_wall_seconds"] < 0):
                return None, None
            with path.open("rb") as handle:
                vectors = np.load(handle, allow_pickle=False)
            return (vectors, checkpoint) if _valid_vectors(vectors, end - start, self.dimensions) else (None, None)
        except (OSError, ValueError, TypeError, KeyError):
            return None, None

    def _ensure_runtime(self):
        if self._session is not None and self._tokenizer is not None:
            return
        started = time.perf_counter()
        import onnxruntime as ort
        from transformers import AutoTokenizer
        if ort.__version__ != self.runtime_versions["onnxruntime"]:
            raise RuntimeError("ONNX Runtime import differs from the version included in the cache identity")
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        options = ort.SessionOptions()
        options.intra_op_num_threads = self.threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3
        providers = ["CPUExecutionProvider"]
        if self.provider == "DmlExecutionProvider":
            options.enable_mem_pattern = False
            providers = [("DmlExecutionProvider", {"device_id": str(self.device_id)})]
        session = ort.InferenceSession(self.onnx_path, sess_options=options,
                                       providers=providers)
        session.disable_fallback()
        if self.provider not in session.get_providers():
            raise RuntimeError("Requested execution provider did not initialize")
        outputs = session.get_outputs()
        if not outputs or outputs[0].name != "last_hidden_state":
            raise ValueError("Expected the pinned ONNX last_hidden_state output")
        self._tokenizer, self._session = tokenizer, session
        self._runtime_initialization_seconds += time.perf_counter() - started

    def _encode_texts(self, texts):
        if not texts:
            return np.empty((0, self.dimensions), dtype=np.float32)
        self._ensure_runtime()
        tokens = self._tokenizer(texts, padding=True, truncation=True,
                                 max_length=self.max_seq_length, return_tensors="np")
        inputs = {item.name: np.asarray(tokens[item.name], dtype=np.int64)
                  for item in self._session.get_inputs()}
        output = self._session.run(["last_hidden_state"], inputs)[0]
        if output.ndim != 3 or output.shape[0] != len(texts) or output.shape[2] != self.dimensions:
            raise ValueError("Unexpected ONNX token embedding shape")
        vectors = np.asarray(output[:, 0, :], dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if not np.isfinite(vectors).all() or (norms == 0).any():
            raise ValueError("Nonfinite or zero ONNX embedding")
        return vectors / norms

    def encode_query(self, query):
        return self._encode_texts([self.query_instruction + query])[0]

    def score(self, query):
        return self.document_vectors @ self.encode_query(query)

    def query_scores_batch(self, queries):
        if not queries:
            return np.empty((0, self.n_documents), dtype=np.float32)
        embeddings = np.vstack([self._encode_texts([self.query_instruction + query for query in queries[start:start + self.batch_size]])
                                for start in range(0, len(queries), self.batch_size)])
        return embeddings @ self.document_vectors.T

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_session"] = state["_tokenizer"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.provider = state.get("provider", "CPUExecutionProvider")
        self.device_id = state.get("device_id", 0)
        self.device_metadata = state.get("device_metadata")
        self._session = self._tokenizer = None


def document_heads_from_zip(corpus, head_words=256):
    """Match the experiment's sorted, exact-deduplicated corpus without qrels."""
    if not isinstance(head_words, int) or head_words < 1:
        raise ValueError("head_words must be a positive integer")
    originals, groups = {}, {}
    with zipfile.ZipFile(corpus) as archive:
        for member in sorted(archive.infolist(), key=lambda item: item.filename):
            if member.is_dir():
                continue
            document_id = Path(member.filename).name
            if document_id in originals:
                raise ValueError("Duplicate archive basename: " + document_id)
            text = archive.read(member).decode("utf-8", errors="replace")
            if not text.strip():
                raise ValueError("Empty corpus document: " + document_id)
            originals[document_id] = text
            checksum = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
            groups.setdefault(checksum, []).append(document_id)
    ids = sorted(min(group) for group in groups.values())
    return ids, [" ".join(originals[document_id].split()[:head_words]) for document_id in ids]


def main():
    """Warm/resume the corpus cache without reading query or relevance files."""
    import argparse
    import sys
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--model", type=Path, default=root / "project/models/bge-small-en-v1.5")
    parser.add_argument("--corpus", type=Path, default=root / "project/data/candidates/2026-11-12_technical_qa_audit/nvidia__TechQA-RAG-Eval/corpus.zip")
    parser.add_argument("--cache-dir", type=Path, default=root / "project/data/cache/embeddings")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--head-words", type=int, default=256)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--provider", choices=["CPUExecutionProvider", "DmlExecutionProvider"], default="CPUExecutionProvider")
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()
    runtime = ".vendor-onnx-dml" if args.provider == "DmlExecutionProvider" else ".vendor-onnx-compat"
    for vendor in (".vendor-hf", runtime):
        path = root / "project" / vendor
        if path.is_dir():
            sys.path.insert(0, str(path))
    ids, documents = document_heads_from_zip(args.corpus, args.head_words)
    index = OnnxEncoderIndex(documents, args.model, cache_dir=args.cache_dir,
                             batch_size=args.batch_size, max_seq_length=args.max_seq_length,
                             threads=args.threads, provider=args.provider, device_id=args.device_id)
    report = {"scope": "Corpus-only cache warm/resume; no query or qrel files read",
              "corpus_zip_sha256": _sha256(args.corpus), "head_words": args.head_words,
              "ordered_document_ids_sha256": _ordered_text_hash(ids), "metadata": index.metadata}
    _atomic_json(Path(index.cache_path) / "warmup.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
