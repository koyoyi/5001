"""Exact lexical, latent-semantic and optional local neural retrieval.

All indices receive the same, caller-prepared document text. In particular, any
head-only truncation policy belongs to the experiment configuration rather than
an undocumented preprocessing difference between retrievers.
"""

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Hashable, Iterable, Sequence

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize


_TOKEN = re.compile(r"[a-z0-9_]+(?:[.+#:/\\-][a-z0-9_]+)*[+#]*")


def tokenize(text: str) -> list[str]:
    """Keep English technical terms, version numbers, digits, C++ and C#."""
    return _TOKEN.findall(text.casefold())


def top_k(
    scores: Sequence[float],
    k: int,
    document_ids: Sequence[Hashable] | None = None,
    *,
    min_score: float | None = None,
) -> np.ndarray:
    """Return document positions, highest score first, with deterministic ties.

    Ties use string document IDs if supplied, then original positions. Without
    IDs, original positions resolve ties. ``min_score=0`` retains strictly
    positive scores only, useful for BM25. By default zero and negative cosine
    scores remain eligible; the caller must choose the retrieval policy.
    """
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("scores must be a finite one-dimensional array")
    if not isinstance(k, (int, np.integer)) or k < 0:
        raise ValueError("k must be a nonnegative integer")
    positions = np.arange(values.size, dtype=np.int64)
    if document_ids is not None and len(document_ids) != values.size:
        raise ValueError("document_ids must align with scores")
    if min_score is not None:
        if not np.isfinite(min_score):
            raise ValueError("min_score must be finite")
        positions = positions[values > min_score]
    if not positions.size or k == 0:
        return np.empty(0, dtype=np.int64)
    # Sort only the score boundary and the selected top-k. Keep deterministic
    # handling of all boundary ties, including an all-zero score vector.
    if positions.size > k:
        boundary = np.partition(values[positions], positions.size - k)[positions.size - k]
        better = positions[values[positions] > boundary]
        tied = positions[values[positions] == boundary]
        needed = k - len(better)
        if document_ids is not None:
            tied = np.asarray(sorted(tied, key=lambda pos: (str(document_ids[pos]), int(pos))), dtype=np.int64)
        positions = np.concatenate([better, tied[:needed]])
    if document_ids is None:
        order = np.lexsort((positions, -values[positions]))
    else:
        ids = np.asarray([str(value) for value in document_ids])
        order = np.lexsort((positions, ids[positions], -values[positions]))
    return positions[order[:k]]


class BM25Index:
    """Robertson BM25 with positive log1p IDF and exact sparse postings."""

    def __init__(self, documents: list[str], k1: float = 1.2, b: float = 0.75):
        if not np.isfinite(k1) or k1 <= 0 or not np.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("BM25 requires k1 > 0 and 0 <= b <= 1")
        self.n_documents = len(documents)
        self.vectorizer = CountVectorizer(
            tokenizer=tokenize, token_pattern=None, lowercase=False,
            dtype=np.float32,
        )
        try:
            counts = self.vectorizer.fit_transform(documents).tocsc()
        except ValueError as error:
            if "empty vocabulary" not in str(error):
                raise
            self.vectorizer = None
            counts = sparse.csc_matrix((self.n_documents, 0), dtype=np.float32)
        lengths = np.asarray(counts.sum(axis=1)).ravel()
        self.document_lengths = lengths
        average_length = float(lengths.mean()) if lengths.size else 0.0
        if counts.shape[1]:
            document_frequencies = np.diff(counts.indptr)
            idf = np.log1p(
                (self.n_documents - document_frequencies + 0.5)
                / (document_frequencies + 0.5)
            )
            norm = k1 * (1 - b + b * lengths / average_length)
            counts.data = (
                counts.data * (k1 + 1)
                / (counts.data + norm[counts.indices])
                * np.repeat(idf, document_frequencies)
            ).astype(np.float32)
        self.weights = counts
        self.metadata = {
            "method": "bm25", "k1": float(k1), "b": float(b),
            "idf": "log(1 + (N - df + 0.5) / (df + 0.5))",
            "query_term_frequency": "linear",
            "documents": self.n_documents,
            "vocabulary_size": counts.shape[1],
            "average_document_tokens": average_length,
            "tokenizer": _TOKEN.pattern,
        }

    def score(self, query: str) -> np.ndarray:
        if self.vectorizer is None:
            return np.zeros(self.n_documents, dtype=np.float32)
        term_counts = Counter(tokenize(query))
        vocabulary = self.vectorizer.vocabulary_
        selected = [(vocabulary[term], count) for term, count in term_counts.items()
                    if term in vocabulary]
        if not selected:
            return np.zeros(self.n_documents, dtype=np.float32)
        indices, frequencies = zip(*selected)
        return np.asarray(
            self.weights[:, list(indices)].dot(
                np.asarray(frequencies, dtype=np.float32)
            )
        ).ravel()

    def term_coverage(self, query: str, candidates: Sequence[int]) -> np.ndarray:
        """Exact unique query-term coverage using the same lexical postings."""
        terms = set(tokenize(query))
        if not terms or self.vectorizer is None:
            return np.zeros(len(candidates), dtype=np.float64)
        indices = [self.vectorizer.vocabulary_[term] for term in terms if term in self.vectorizer.vocabulary_]
        if not indices:
            return np.zeros(len(candidates), dtype=np.float64)
        hits = (self.weights[:, indices][list(candidates)] > 0).sum(axis=1)
        return np.asarray(hits).ravel().astype(np.float64) / len(terms)


class SparseLSAIndex:
    """Unsupervised TF-IDF/SVD cosine retrieval, not a neural dense encoder."""

    def __init__(self, documents: list[str], dimensions: int = 128,
                 seed: int = 20261114):
        if not isinstance(dimensions, int) or dimensions < 1:
            raise ValueError("dimensions must be a positive integer")
        self.n_documents = len(documents)
        self.vectorizer = TfidfVectorizer(
            tokenizer=tokenize, token_pattern=None, lowercase=False,
            max_features=60000, sublinear_tf=True, dtype=np.float32,
        )
        self.svd = None
        try:
            matrix = self.vectorizer.fit_transform(documents)
        except ValueError as error:
            if "empty vocabulary" not in str(error):
                raise
            self.vectorizer = None
            matrix = sparse.csr_matrix((self.n_documents, 0), dtype=np.float32)
        effective_dimensions = min(
            dimensions, max(1, matrix.shape[1] - 1),
            max(1, self.n_documents - 1),
        ) if matrix.shape[1] else 0
        if self.n_documents > 1 and matrix.shape[1] > 1:
            self.svd = TruncatedSVD(
                n_components=effective_dimensions, random_state=seed,
            )
            embeddings = self.svd.fit_transform(matrix)
        else:
            embeddings = matrix.toarray()
            effective_dimensions = matrix.shape[1]
        self.document_vectors = normalize(
            embeddings, copy=False,
        ).astype(np.float32) if embeddings.shape[1] else embeddings
        self.metadata = {
            "method": "tfidf_lsa_cosine",
            "neural_encoder": False,
            "fit_scope": "shared document corpus only; no query labels",
            "dimensions_requested": dimensions,
            "dimensions": effective_dimensions,
            "max_features": 60000,
            "sublinear_tf": True,
            "normalization": "l2",
            "search": "exact dot product",
            "seed": seed,
            "documents": self.n_documents,
            "vocabulary_size": matrix.shape[1],
        }

    def query_scores_batch(self, queries: Sequence[str]) -> np.ndarray:
        if not queries:
            return np.empty((0, self.n_documents), dtype=np.float32)
        if self.vectorizer is None:
            return np.zeros((len(queries), self.n_documents), dtype=np.float32)
        matrix = self.vectorizer.transform(queries)
        embeddings = self.svd.transform(matrix) if self.svd else matrix.toarray()
        embeddings = normalize(embeddings, copy=False).astype(np.float32)
        return embeddings @ self.document_vectors.T

    def score(self, query: str) -> np.ndarray:
        return self.query_scores_batch([query])[0]


class SentenceEncoderIndex:
    """Frozen local SentenceTransformer with exact normalized dot products.

    A local model directory is required, and loading is explicitly offline.
    The default query instruction follows BGE English retrieval. Callers using
    another model must supply its appropriate instruction (possibly empty).
    """

    def __init__(self, documents: list[str], model_path: str | Path,
                 batch_size: int = 32,
                 query_instruction: str =
                 "Represent this sentence for searching relevant passages: ",
                 max_seq_length: int = 512):
        path = Path(model_path).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError("A local model directory is required: " + str(path))
        if batch_size < 1 or max_seq_length < 1:
            raise ValueError("batch_size and max_seq_length must be positive")
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            str(path), device="cpu", local_files_only=True,
        )
        self.model.max_seq_length = max_seq_length
        self.query_instruction = query_instruction
        self.batch_size = batch_size
        self.n_documents = len(documents)
        print(f"Encoding {len(documents)} documents with local encoder {path.name}",
              flush=True)
        if documents:
            self.document_vectors = np.asarray(self.model.encode(
                documents, batch_size=batch_size, show_progress_bar=True,
                normalize_embeddings=True, convert_to_numpy=True,
            ), dtype=np.float32)
        else:
            self.document_vectors = np.empty(
                (0, self.model.get_sentence_embedding_dimension()), dtype=np.float32,
            )
        config_path = path / "config.json"
        config_bytes = config_path.read_bytes() if config_path.exists() else None
        config = json.loads(config_bytes) if config_bytes else {}
        self.metadata = {
            "method": "frozen_sentence_transformer_cosine",
            "neural_encoder": True,
            "local_model_path": str(path),
            "local_files_only": True,
            "device": "cpu",
            "model_config_sha256": hashlib.sha256(config_bytes).hexdigest()
            if config_bytes else None,
            "model_config_revision": config.get("_commit_hash"),
            "query_instruction": query_instruction,
            "document_instruction": "",
            "max_seq_length": max_seq_length,
            "input_document_policy": "caller supplies identical text to all retrievers",
            "normalize_embeddings": True,
            "search": "exact dot product",
            "batch_size": batch_size,
            "dimensions": self.document_vectors.shape[1],
            "documents": self.n_documents,
        }

    def encode_query(self, query: str) -> np.ndarray:
        return np.asarray(self.model.encode(
            [self.query_instruction + query], batch_size=1, show_progress_bar=False,
            normalize_embeddings=True, convert_to_numpy=True,
        )[0], dtype=np.float32)

    def query_scores_batch(self, queries: Sequence[str]) -> np.ndarray:
        if not queries:
            return np.empty((0, self.n_documents), dtype=np.float32)
        embeddings = np.asarray(self.model.encode(
            [self.query_instruction + query for query in queries],
            batch_size=self.batch_size, show_progress_bar=False,
            normalize_embeddings=True, convert_to_numpy=True,
        ), dtype=np.float32)
        return embeddings @ self.document_vectors.T

    def score(self, query: str) -> np.ndarray:
        return self.document_vectors @ self.encode_query(query)


def rrf_scores(candidates: Iterable[Hashable], bm25_order: Iterable[Hashable],
               dense_order: Iterable[Hashable], constant: float = 60,
               semantic_weight: float = 1.) -> np.ndarray:
    """Rank-one RRF scores aligned with the complete supplied candidate union."""
    candidates = list(candidates)
    if len(set(candidates)) != len(candidates):
        raise ValueError("candidate union must contain unique document IDs")
    if not np.isfinite(constant) or constant < 0:
        raise ValueError("RRF constant must be finite and nonnegative")
    if not np.isfinite(semantic_weight) or semantic_weight < 0:
        raise ValueError("semantic_weight must be finite and nonnegative")
    result = np.zeros(len(candidates), dtype=np.float64)
    for weight, order in ((1., bm25_order), (semantic_weight, dense_order)):
        ranks = {}
        for document_id in order:
            if document_id not in ranks:
                ranks[document_id] = len(ranks) + 1
        result += np.asarray([
            weight / (constant + ranks[document_id]) if document_id in ranks else 0.0
            for document_id in candidates
        ])
    return result
