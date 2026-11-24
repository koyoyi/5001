"""Citation-preserving evidence search, without generating unsupported answers."""
from pathlib import Path
import hashlib
import json
import sys
import time
import joblib
import numpy as np
from .features import build_features
from .experiment import predict
from .retrieval import top_k, rrf_scores


def _relocate_encoder(semantic, root):
    """Bundles store absolute model paths; re-point them after the repository moves, same bytes only."""
    stored = getattr(semantic, "onnx_path", None)
    if stored is None or Path(stored).is_file():
        return
    parts = Path(semantic.model_path).parts
    if "project" not in parts:
        raise FileNotFoundError("Encoder model not found: " + semantic.model_path)
    model = root.joinpath("project", *parts[len(parts) - parts[::-1].index("project"):])
    onnx = model / "onnx" / "model.onnx"
    expected = semantic.metadata.get("model_onnx_sha256")
    if not onnx.is_file() or hashlib.sha256(onnx.read_bytes()).hexdigest() != expected:
        raise FileNotFoundError("Relocated encoder is missing or differs from the recorded hash: " + str(onnx))
    semantic.model_path, semantic.onnx_path = str(model), str(onnx)


def load_bundle(artifact):
    artifact = Path(artifact)
    config = json.loads((artifact / "configuration.json").read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parents[3]
    folders = [".vendor-hf", ".vendor-onnx-dml" if config.get("encoder_provider") == "DmlExecutionProvider" else ".vendor-onnx-compat"]
    for folder in folders:
        path = root / "project" / folder
        if path.is_dir():
            if str(path) in sys.path:
                sys.path.remove(str(path))
            sys.path.insert(0, str(path))
    bundle = joblib.load(artifact / "search_bundle.joblib")
    _relocate_encoder(bundle["semantic"], root)
    return bundle


def search(bundle, question, method="logistic", limit=5):
    if not question.strip():
        raise ValueError("Please enter a nonempty question.")
    if not 1 <= limit <= 20:
        raise ValueError("limit must be between 1 and 20.")
    start = time.perf_counter()
    ids, documents = bundle["document_ids"], bundle["documents"]
    if method in ("bm25", "semantic"):
        scores = bundle[method].score(question)
        retrieval_end = time.perf_counter()
        order = top_k(scores, bundle["configuration"]["top_k"], document_ids=ids,
                      min_score=0. if method == "bm25" else None)
        chosen = order[:limit]
        end = time.perf_counter()
        return {"question": question, "method": method, "candidate_count": len(order),
                "mode": "retrieved_source_excerpts; not a generated answer or calibrated probability",
                "timing_ms": {"retrieval": (retrieval_end - start) * 1000, "candidate_features": 0.,
                              "rerank_and_output": (end - retrieval_end) * 1000, "total": (end - start) * 1000},
                "evidence": [{"citation": f"[{ids[position]}]", "document_id": ids[position],
                              "ranking_score": float(scores[position]), "excerpt": documents[position][:1200]}
                             for position in chosen]}
    lexical = bundle["bm25"].score(question)
    semantic = bundle["semantic"].score(question)
    retrieval_end = time.perf_counter()
    lexical_order = top_k(lexical, bundle["configuration"]["top_k"], document_ids=ids, min_score=0.)
    semantic_order = top_k(semantic, bundle["configuration"]["top_k"], document_ids=ids)
    candidates = np.asarray(sorted(set(lexical_order.tolist()) | set(semantic_order.tolist())), dtype=int)
    features = None if method in ("rrf", "weighted_rrf") else build_features(
        question, candidates, documents, lexical, semantic, lexical_order, semantic_order,
        bm25_index=bundle["bm25"])
    feature_end = time.perf_counter()
    if method == "rrf":
        scores = rrf_scores(candidates, lexical_order, semantic_order, constant=bundle["rrf_constant"])
    elif method == "weighted_rrf" and bundle.get("weighted_rrf_selection") is not None:
        chosen = bundle["weighted_rrf_selection"]
        scores = rrf_scores(candidates, lexical_order, semantic_order, constant=chosen["constant"],
                            semantic_weight=chosen["semantic_weight"])
    elif method in bundle["models"]:
        scores = predict(bundle["models"][method], features)
    else:
        raise ValueError("Choose an available bm25, semantic, rrf, weighted_rrf, logistic or random_forest method.")
    order = top_k(scores, limit, document_ids=[ids[index] for index in candidates])
    end = time.perf_counter()
    return {"question": question, "method": method, "candidate_count": len(candidates),
            "mode": "retrieved_source_excerpts; not a generated answer or calibrated probability",
            "timing_ms": {"retrieval": (retrieval_end - start) * 1000,
                          "candidate_features": (feature_end - retrieval_end) * 1000,
                          "rerank_and_output": (end - feature_end) * 1000,
                          "total": (end - start) * 1000},
            "evidence": [{"citation": f"[{ids[candidates[position]]}]", "document_id": ids[candidates[position]],
                          "ranking_score": float(scores[position]),
                          "excerpt": documents[candidates[position]][:1200]}
                         for position in order]}
