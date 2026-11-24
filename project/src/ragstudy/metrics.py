"""Binary relevance metrics and cluster-aware paired uncertainty estimates."""

from collections import defaultdict
from typing import Hashable, Iterable, Sequence

import numpy as np


def ranking_metrics(ranked_document_ids: Iterable[Hashable],
                    positive_ids: Iterable[Hashable], ndcg_k: int = 10,
                    recall_k: int = 20) -> dict[str, float]:
    """Evaluate unique ranked IDs against all known positives, including misses.

    The ideal DCG uses every gold document, not merely the retrieved candidates.
    Queries without positives receive zero for both metrics; callers should
    explicitly describe whether such queries belong to the evaluation cohort.
    """
    if ndcg_k < 1 or recall_k < 1:
        raise ValueError("metric cutoffs must be positive")
    positives = set(positive_ids)
    ranking = list(dict.fromkeys(ranked_document_ids))
    hits = np.asarray([item in positives for item in ranking[:ndcg_k]], dtype=float)
    discounts = 1.0 / np.log2(np.arange(hits.size) + 2.0)
    dcg = float(hits @ discounts)
    ideal_count = min(len(positives), ndcg_k)
    ideal = float((1.0 / np.log2(np.arange(ideal_count) + 2.0)).sum())
    recall = len(set(ranking[:recall_k]) & positives) / len(positives) if positives else 0.0
    return {
        f"ndcg@{ndcg_k}": dcg / ideal if ideal else 0.0,
        f"recall@{recall_k}": float(recall),
    }


def paired_group_bootstrap(rows: Sequence[dict], baseline: str, challenger: str,
                           metric: str, iterations: int = 2000,
                           seed: int = 20261114) -> dict:
    """Resample independent groups and recompute the query-macro mean difference.

    Each input row represents one unique query and has ``query_id``, ``group_id``
    and nested method metrics, e.g. ``{"rrf": {"ndcg@10": 0.5}}``. Groups are
    sampled with replacement; *all* query differences in a sampled group stay
    together, and query counts supply the mean's denominator. This matters when
    multiple cases share the same gold document. The interval is a percentile
    bootstrap confidence interval, not a calibrated significance test.
    """
    if not rows:
        raise ValueError("bootstrap requires at least one evaluated query")
    if not isinstance(iterations, (int, np.integer)) or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    seen = set()
    grouped = defaultdict(list)
    baseline_values, challenger_values = [], []
    for row in rows:
        query_id = row["query_id"]
        if query_id in seen:
            raise ValueError("bootstrap rows must contain unique query IDs")
        seen.add(query_id)
        group_id = row["group_id"]
        baseline_value = float(row[baseline][metric])
        challenger_value = float(row[challenger][metric])
        if not np.isfinite([baseline_value, challenger_value]).all():
            raise ValueError("bootstrap metrics must be finite")
        baseline_values.append(baseline_value)
        challenger_values.append(challenger_value)
        grouped[group_id].append(challenger_value - baseline_value)
    # Insertion order makes the exact RNG draw independent of group-ID types.
    group_sums = np.asarray([sum(values) for values in grouped.values()])
    group_sizes = np.asarray([len(values) for values in grouped.values()])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(grouped), size=(iterations, len(grouped)))
    differences = group_sums[draws].sum(axis=1) / group_sizes[draws].sum(axis=1)
    lower, upper = np.percentile(differences, [2.5, 97.5])
    baseline_mean = float(np.mean(baseline_values))
    challenger_mean = float(np.mean(challenger_values))
    return {
        "baseline": baseline, "challenger": challenger, "metric": metric,
        "baseline_mean": baseline_mean, "challenger_mean": challenger_mean,
        "difference": challenger_mean - baseline_mean,
        "ci_lower": float(lower), "ci_upper": float(upper),
        "confidence_level": 0.95, "resampling_unit": "group",
        "aggregation_unit": "query",
        "n_queries": len(rows), "n_groups": len(grouped),
        "iterations": int(iterations), "seed": int(seed),
    }
