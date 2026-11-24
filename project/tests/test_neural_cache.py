"""Offline fault-injection checks for resumable FP32 embedding caches."""

import hashlib
import json
from pathlib import Path
import pickle
import sys
import zipfile

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ragstudy import neural


@pytest.fixture
def local_model(tmp_path, monkeypatch):
    model = tmp_path / "model"
    (model / "onnx").mkdir(parents=True)
    (model / "onnx/model.onnx").write_bytes(b"mock-fp32-model-bytes")
    (model / "config.json").write_text(json.dumps({"hidden_size": 4, "max_position_embeddings": 512}), encoding="utf-8")
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"):
        (model / name).write_text("{}", encoding="utf-8")
    (model / "onnx_download_manifest.json").write_text(json.dumps({
        "model": "offline-test-encoder", "revision": "pinned-test-revision",
        "files": [{"path": "onnx/model.onnx", "sha256": neural._sha256(model / "onnx/model.onnx")}],
    }), encoding="utf-8")
    monkeypatch.setattr(neural, "_versions", lambda: {"onnxruntime": "fake", "transformers": "fake", "tokenizers": "fake"})
    calls = []

    def encode(self, texts):
        calls.append(list(texts))
        vectors = np.asarray([[int.from_bytes(hashlib.sha256(text.encode()).digest()[i:i+2], "big") + 1
                               for i in range(0, 8, 2)] for text in texts], dtype=np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    monkeypatch.setattr(neural.OnnxEncoderIndex, "_encode_texts", encode)
    return model, calls, encode


def test_complete_cache_reuse_preserves_order_and_reports_no_new_encoding(local_model, tmp_path):
    model, calls, _ = local_model
    documents = ["alpha", "beta", "gamma"]
    first = neural.OnnxEncoderIndex(documents, model, tmp_path / "cache", batch_size=2)
    assert len(calls) == 2
    assert first.metadata["documents_encoded_this_run"] == 3
    calls.clear()
    reused = neural.OnnxEncoderIndex(documents, model, tmp_path / "cache", batch_size=2)
    assert calls == []
    np.testing.assert_array_equal(reused.document_vectors, first.document_vectors)
    assert reused.metadata["cache_hit"]
    assert reused.metadata["cached_documents_reused"] == 3
    assert reused.metadata["encoding_wall_seconds_this_run"] == 0
    assert reused.metadata["successful_cached_batch_recorded_encoding_seconds"] > 0


def test_interrupted_encoding_resumes_only_unfinished_batches(local_model, tmp_path, monkeypatch):
    model, calls, original_encode = local_model
    documents = ["a", "b", "c", "d", "e"]

    def interrupted(self, texts):
        if len(calls) == 1:
            raise RuntimeError("Simulated interrupted batch")
        return original_encode(self, texts)

    monkeypatch.setattr(neural.OnnxEncoderIndex, "_encode_texts", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        neural.OnnxEncoderIndex(documents, model, tmp_path / "cache", batch_size=2)
    monkeypatch.setattr(neural.OnnxEncoderIndex, "_encode_texts", original_encode)
    calls.clear()
    resumed = neural.OnnxEncoderIndex(documents, model, tmp_path / "cache", batch_size=2)
    assert calls == [["c", "d"], ["e"]]
    assert resumed.metadata["cached_documents_reused"] == 2
    assert resumed.metadata["documents_encoded_this_run"] == 3


def test_corrupt_cache_batch_is_detected_and_only_that_batch_is_recomputed(local_model, tmp_path):
    model, calls, _ = local_model
    index = neural.OnnxEncoderIndex(["a", "b", "c"], model, tmp_path / "cache", batch_size=2)
    batch = next(Path(index.cache_path).glob("batch_00000000_*.npy"))
    batch.write_bytes(b"corrupt-not-an-array")
    calls.clear()
    repaired = neural.OnnxEncoderIndex(["a", "b", "c"], model, tmp_path / "cache", batch_size=2)
    assert calls == [["a", "b"]]
    assert repaired.metadata["invalid_cached_batches_recomputed"] == 1
    assert repaired.metadata["cached_documents_reused"] == 1
    np.testing.assert_array_equal(repaired.document_vectors, index.document_vectors)


def test_text_order_tokenizer_and_instruction_change_the_cache_identity(local_model, tmp_path):
    model, calls, _ = local_model
    original = neural.OnnxEncoderIndex(["a", "b"], model, tmp_path / "cache", batch_size=2)
    reversed_index = neural.OnnxEncoderIndex(["b", "a"], model, tmp_path / "cache", batch_size=2)
    instructed = neural.OnnxEncoderIndex(["a", "b"], model, tmp_path / "cache", batch_size=2,
                                         query_instruction="different prefix: ")
    (model / "vocab.txt").write_text("changed vocabulary", encoding="utf-8")
    tokenizer_changed = neural.OnnxEncoderIndex(["a", "b"], model, tmp_path / "cache", batch_size=2)
    assert len({item.cache_key for item in (original, reversed_index, instructed, tokenizer_changed)}) == 4
    assert len(calls) == 4


def test_tampered_model_rejected_before_cache_reuse(local_model, tmp_path):
    model, _, _ = local_model
    neural.OnnxEncoderIndex(["a"], model, tmp_path / "cache")
    (model / "onnx/model.onnx").write_bytes(b"changed-model")
    with pytest.raises(ValueError, match="SHA-256"):
        neural.OnnxEncoderIndex(["a"], model, tmp_path / "cache")


def test_pickle_excludes_runtime_objects_and_query_instruction_is_retained(local_model, tmp_path):
    model, calls, _ = local_model
    index = neural.OnnxEncoderIndex(["a", "b"], model, tmp_path / "cache")
    index._session = lambda: "not pickleable"
    index._tokenizer = lambda: "not pickleable"
    restored = pickle.loads(pickle.dumps(index))
    assert restored._session is restored._tokenizer is None
    calls.clear()
    scores = restored.score("question")
    assert calls == [[neural.INSTRUCTION + "question"]]
    assert scores.shape == (2,)
    assert np.isfinite(scores).all()
    assert restored.query_scores_batch([]).shape == (0, 2)


def test_empty_corpus_and_invalid_parameters(local_model, tmp_path):
    model, _, _ = local_model
    empty = neural.OnnxEncoderIndex([], model, tmp_path / "cache")
    assert empty.document_vectors.shape == (0, 4)
    for parameter in ({"batch_size": 0}, {"threads": 0}, {"max_seq_length": 513}):
        with pytest.raises(ValueError):
            neural.OnnxEncoderIndex(["a"], model, tmp_path / "cache", **parameter)


def test_corpus_only_loader_uses_same_sorted_exact_alias_policy(tmp_path):
    corpus = tmp_path / "corpus.zip"
    with zipfile.ZipFile(corpus, "w") as archive:
        archive.writestr("nested/z.txt", "  same source text  ")
        archive.writestr("a.txt", "same source text")
        archive.writestr("m.txt", "different words for evidence")
    ids, documents = neural.document_heads_from_zip(corpus, head_words=2)
    assert ids == ["a.txt", "m.txt"]
    assert documents == ["same source", "different words"]


def test_gpu_provider_gets_separate_cache_and_retains_lazy_device_identity(local_model, tmp_path, monkeypatch):
    model, _, _ = local_model
    monkeypatch.setattr(neural, "version", lambda name: "fake")
    monkeypatch.setattr(neural, "dml_adapters", lambda: [
        {"device_id": 1, "description": "Synthetic GPU", "vendor_id": 4318,
         "pci_device_id": 9504, "dedicated_video_memory_mib": 6000.}])
    cpu = neural.OnnxEncoderIndex(["a", "b"], model, tmp_path / "cache")
    gpu = neural.OnnxEncoderIndex(["a", "b"], model, tmp_path / "cache",
                                   provider="DmlExecutionProvider", device_id=1)
    assert cpu.cache_key != gpu.cache_key
    identity = json.loads((Path(cpu.cache_path) / "identity.json").read_text())["identity"]
    assert "device_id" not in identity and "adapter" not in identity
    assert "onnxruntime_distribution" not in identity["runtime_versions"]
    assert gpu.metadata["cached_documents_reused"] == 0
    assert gpu.runtime_versions["onnxruntime_distribution"] == "onnxruntime-directml"
    restored = pickle.loads(pickle.dumps(gpu))
    assert restored.provider == "DmlExecutionProvider" and restored.device_id == 1
    assert restored._session is restored._tokenizer is None


def test_old_cpu_pickle_state_remains_compatible(local_model, tmp_path):
    model, _, _ = local_model
    original = neural.OnnxEncoderIndex(["a"], model, tmp_path / "cache")
    old_state = original.__getstate__()
    for key in ("provider", "device_id", "device_metadata"):
        old_state.pop(key)
    restored = neural.OnnxEncoderIndex.__new__(neural.OnnxEncoderIndex)
    restored.__setstate__(old_state)
    assert restored.provider == "CPUExecutionProvider" and restored.device_id == 0
    assert restored.cache_key == original.cache_key
