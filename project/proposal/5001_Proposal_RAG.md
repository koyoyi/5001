# DSC5001 Project Proposal

**Project title:** Can Learned Reranking Improve Hybrid Retrieval for RAG?

**Team:** [TEAM NAME]  
**Submitted:** [DD MONTH 2026]  
**Draft date:** 7 November 2026

| Member | UID | Proposed responsibility |
|---|---|---|
| [MEMBER 1] | [UID 1] | Data audit, indexing and candidate retrieval |
| [MEMBER 2] | [UID 2] | Baselines and logistic reranker |
| [MEMBER 3] | [UID 3] | Random forest and grouped validation |
| [MEMBER 4] | [UID 4] | Evaluation, application and presentation |
| [OPTIONAL MEMBER 5] | [UID 5] | Reproducibility and robustness |

## 1. Background and Motivation

Retrieval-augmented generation (RAG) depends on finding useful evidence before generating an answer. Keyword and vector retrieval capture different relevance signals, but fixed fusion rules may not combine them effectively for every query. We will test whether lightweight supervised rerankers improve evidence retrieval sufficiently to justify their latency, informing the design of a searchable, citation-backed application.

## 2. Research Questions

1. **Complementarity:** Does hybrid candidate retrieval recover relevant passages missed by BM25 or dense retrieval alone?
2. **Generalisation:** On unseen queries, can logistic regression or a random forest improve nDCG@10 and Recall@20 over fixed reciprocal rank fusion (RRF), using identical candidates?
3. **Practical value:** Are improvements consistent across query types and worth the additional online latency? Which queries become worse?

Negative or negligible improvements are valid outcomes.

## 3. Data

The primary dataset is the public **BEIR FiQA** snapshot [1, 2], accessed and audited on 7 November 2026. It contains 57,638 passages and 6,648 queries: **5,500 training, 500 development and 648 test queries**. Their relevance files contain respectively **14,166, 1,238 and 1,706 query-passage judgments**, all scored 1; these row counts are not independent query counts. Query identifiers do not overlap across splits.

A training observation is a query-candidate pair; evaluation aggregates by query. The target is observed relevance. Predictors include BM25 score, cosine similarity, truncated retrieval ranks, missing-rank indicators, lexical overlap and text lengths. No test judgments enter features, indexing decisions or training. Passage IDs and original text remain traceable; document-level units are retained to avoid ambiguous chunk-to-label mapping.

All qrel scores are positive; no explicit negatives are supplied. For each query, retain positives present in the candidate union and sample at most 50 unjudged candidates as proxy negatives using a fixed seed. Log queries with no retrieved positives; never inject missed gold passages. Audit empty text and duplicates, and vary the negative-sampling ratio in robustness analysis. **SciFact** is a smaller fallback with 5,183 documents and 300 test queries [3]; switching requires a recorded scope change, rather than selecting whichever dataset produces better results.

## 4. Proposed Methods

- **Retrieval baselines:** BM25 and one frozen pretrained dense encoder, each searching the same corpus. Encoder weights remain fixed; passages are encoded offline.
- **Shared candidates and strong fusion baseline:** Take the deduplicated union of each retriever's top 100, giving at most 200 candidates. RRF, logistic regression and random forest rank this identical, complete union; no preliminary RRF truncation applies. RRF uses reciprocal ranks, avoiding addition of incompatible raw scores; its constant is selected only on development data.
- **Interpretable learning:** Train L2-regularised logistic regression and sort by its relevance score, which is not a calibrated relevance probability. Fold-specific scaling and query weighting prevent preprocessing leakage and dominance by queries with more sampled pairs.
- **Nonlinear comparison:** A depth/leaf-constrained random forest tests whether interactions improve ranking. Use a small, prespecified hyperparameter budget rather than adding many models.

These methods address RQ1-RQ3 through candidate coverage, held-out ranking comparisons and measured cost. The statistical contribution is training and validating relevance models; calling an embedding API alone is insufficient.

## 5. Validation and Evaluation

Preserve official train/dev/test splits. Tune model complexity using query-grouped cross-validation within training data; development data supports final pipeline selection. Every candidate for a query stays in one fold. Inspect cross-split near duplicates before fitting and report any fixed exclusion rule. Freeze preprocessing, sampling, candidates and hyperparameters before opening test results. Shared corpus access is permitted; test relevance supervision is not.

Primary metrics are macro-averaged **nDCG@10 and Recall@20**. Report union candidate recall as a ceiling, separating retrieval failures from reranking failures. Compare methods on the same queries with paired query-bootstrap confidence intervals. Inspect performance by query length and lexical/semantic agreement, plus coefficient or feature-ablation analyses. Scores reflect incomplete qrels, not exhaustive human relevance.

Measure online retrieval and reranking latency separately and jointly under fixed hardware, cache and concurrency conditions, with repeated-query median and p95 summaries. Report offline embedding/indexing time separately. Verify candidate IDs, duplicate removal and metric calculations against BEIR evaluation tools [3].

Generation is a small, fixed-model demonstration showing retrieved passage IDs and citations. Transparent qualitative checks will examine citation support; qrels will not be presented as answer-level ground truth or used to claim improved answer accuracy.

## 6. Expected Contribution

Deliverables include reproducible lexical/vector indexes, SQL experiment records, a lightweight learned reranker, an evaluation notebook and a minimal evidence-search API/interface. The study will establish whether supervised reranking adds value beyond strong fixed fusion in this setting, without presuming that it wins. Limitations include one domain, sparse judgments, proxy negatives and dependence on a frozen encoder. This is an in-domain supervised experiment, not a zero-shot BEIR claim or a new foundation model.

## 7. Timeline

| Dates in November 2026 | Milestone |
|---|---|
| 7-13 | Audit splits; build indexes, BM25/dense/RRF and shared candidates |
| 14-20 | Train logistic/RF models; grouped validation and pipeline freeze |
| 21-26 | Final evaluation, latency/robustness analysis and citation demo |
| 27-28 | At most 10 slides, 8-minute recording, code appendix and early submission |

Template deadline: **29 November 2026, 11:59 PM**; subsequent course notices take precedence. No experimental performance is claimed in this proposal.

### References

1. BEIR. [FiQA corpus and queries](https://huggingface.co/datasets/BeIR/fiqa).
2. BEIR. [FiQA train/dev/test relevance judgments](https://huggingface.co/datasets/BeIR/fiqa-qrels).
3. BEIR. [Official datasets and evaluation tools](https://github.com/beir-cellar/beir).
4. Sentence Transformers. [Retrieve & Re-Rank](https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html).

Sources accessed 7 November 2026.
