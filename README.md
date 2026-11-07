# SDSC5001 - RAG Retrieval and Learned Reranking

**Research question:** Can learned reranking improve hybrid retrieval for RAG?

The project studies evidence retrieval for a citation-backed RAG application. It compares BM25, dense retrieval, reciprocal rank fusion (RRF), and lightweight supervised rerankers on BEIR FiQA. The statistical-learning work is the training and evaluation of logistic regression and random forest ranking models.

**Status: proposal and experimental protocol prepared; experiments have not been run.** No retrieval scores, latency gains or answer-accuracy improvements are claimed.

## Project documents

| Document | Purpose |
|---|---|
| [English proposal](project/proposal/5001_Proposal_RAG.md) | Seven sections matching the course template |
| [Proposal PDF](output/pdf/5001_Proposal_RAG.pdf) | Formatted review/submission draft |
| [中文项目任务书](project/5001_RAG_项目任务书.md) | Methods, validation, implementation plan and deliverables |
| [Experiment configuration](project/configs/rag_experiment.json) | Planned candidates, features, model budgets and metrics |
| [Source audit](project/data/fiqa/source_audit.json) | Verified source information and split counts |

Team name, member names, UIDs and submission date still need the group's details. The proposal contains no invented team information.

## Architecture

~~~mermaid
flowchart LR
    Q[Query] --> B[BM25 top 100]
    Q --> D[Frozen dense encoder: top 100]
    B --> U[Deduplicated union: at most 200]
    D --> U
    U --> F[Fixed RRF]
    U --> L[Learned logistic / RF reranking]
    F --> E[Query-level evaluation]
    L --> E
    L --> C[Evidence and passage IDs]
    C --> G[Fixed-model answer with citations: demo]
~~~

## Dataset and comparisons

The primary benchmark is [BEIR FiQA](https://huggingface.co/datasets/BeIR/fiqa):

- 57,638 passages; 5,500 training, 500 development and 648 test queries.
- [Relevance judgments](https://huggingface.co/datasets/BeIR/fiqa-qrels) are positive-only. Unjudged candidates sampled for training are proxy negatives, not confirmed irrelevant passages.
- RRF and both learned models rank the same candidate union.
- Primary metrics: nDCG@10 and Recall@20, with candidate recall and online latency reported separately.
- Official query splits remain separate; candidates from one query never cross validation folds.

The generation layer is a citation-backed demonstration. Retrieval judgments do not constitute answer-level ground truth.

## Implementation milestones

| Stage | Deliverable | Current state |
|---|---|---|
| 1 | Source and split audit, proposal, protocol | Prepared |
| 2 | BM25/dense indexes and shared candidate runs | Planned |
| 3 | Logistic/RF rerankers and grouped validation | Planned |
| 4 | Frozen test evaluation, latency and failure analysis | Planned |
| 5 | Evidence-search API/interface, citation demo and presentation | Planned |

The implementation is planned around Python, scikit-learn, Sentence Transformers, a lexical index, an exact vector-search baseline, and SQL experiment records. Notebook workflows will accompany the implementation. Model weights, corpora, indexes and caches are not part of this documentation commit.

## Course materials

- [Study notes and lecture navigation](notebook/README.md)
- [Lecture PDFs](output/pdf/)
- [Project navigation](project/README.md)

Assignments and generated homework outputs stay local and are excluded from Git tracking.

Earlier topic drafts are retained under [the project archive](project/archive/2026-11-07_prior_topics/README.md). RAG is the active project direction.
