"""Query-candidate features and query-balanced proxy-negative sampling."""
import numpy as np
from .retrieval import tokenize

FEATURE_NAMES = ["bm25_score", "semantic_score", "bm25_reciprocal_rank", "semantic_reciprocal_rank",
                 "missing_bm25_rank", "missing_semantic_rank", "query_term_coverage",
                 "log_query_length", "log_document_length"]


def build_features(question, candidates, documents, bm25_scores, semantic_scores, bm25_order, semantic_order,
                   bm25_index=None):
    lexical_ranks = {int(doc): rank for rank, doc in enumerate(bm25_order, 1)}
    semantic_ranks = {int(doc): rank for rank, doc in enumerate(semantic_order, 1)}
    query_tokens = tokenize(question)
    query_terms = set(query_tokens)
    fast = bm25_index is not None and hasattr(bm25_index, "document_lengths")
    coverage = bm25_index.term_coverage(question, candidates) if fast else None
    features = []
    for position, doc in enumerate(candidates):
        doc = int(doc)
        terms = None if fast else tokenize(documents[doc])
        features.append([float(bm25_scores[doc]), float(semantic_scores[doc]),
                         1 / lexical_ranks[doc] if doc in lexical_ranks else 0.,
                         1 / semantic_ranks[doc] if doc in semantic_ranks else 0.,
                         float(doc not in lexical_ranks), float(doc not in semantic_ranks),
                         float(coverage[position]) if fast else len(query_terms.intersection(terms)) / max(1, len(query_terms)),
                         np.log1p(len(query_tokens)),
                         np.log1p(bm25_index.document_lengths[doc] if fast else len(terms))])
    return np.asarray(features, dtype=np.float64).reshape(-1, len(FEATURE_NAMES))


def sample_training_rows(records, max_negatives=50, seed=20261114,
                         negative_sampling="random", balance_classes=False, query_weight_total=None):
    """Keep retrieved positives only; missed positives never enter candidate sets."""
    if not isinstance(max_negatives, int) or max_negatives < 1:
        raise ValueError("max_negatives must be a positive integer.")
    random = np.random.RandomState(seed)
    arrays, labels, weights, groups = [], [], [], []
    missed = []
    for record in records:
        positives = np.flatnonzero(record["labels"] == 1)
        negatives = np.flatnonzero(record["labels"] == 0)
        if not len(positives):
            missed.append(record["query"]["query_id"])
            continue
        if not len(negatives):
            continue
        if negative_sampling == "random":
            negatives = random.choice(negatives, size=min(max_negatives, len(negatives)), replace=False)
        elif negative_sampling == "hard_bm25":
            negatives = np.asarray(sorted(negatives, key=lambda index: (-record["features"][index, 0], int(index)))[:max_negatives])
        else:
            raise ValueError("negative_sampling must be random or hard_bm25.")
        selected = np.concatenate([positives, negatives])
        arrays.append(record["features"][selected])
        labels.append(record["labels"][selected])
        if balance_classes:
            weights.extend([.5 / len(positives)] * len(positives) + [.5 / len(negatives)] * len(negatives))
        else:
            weights.extend([1 / len(selected)] * len(selected))
        groups.extend([record["query"]["group_id"]] * len(selected))
    if not arrays:
        raise ValueError("No training query has both a retrieved positive and a proxy negative.")
    weights = np.asarray(weights, dtype=np.float64)
    if query_weight_total is None:
        weights /= weights.mean()
    else:
        if not np.isfinite(query_weight_total) or query_weight_total <= 0:
            raise ValueError("query_weight_total must be finite and positive.")
        weights *= query_weight_total
    return np.vstack(arrays), np.concatenate(labels), weights, np.asarray(groups), missed
