# SDSC5001 - RAG Retrieval and Learned Reranking

**Research question:** Can learned reranking improve hybrid retrieval for RAG?

The project studies evidence retrieval for a citation-backed RAG application. It compares BM25, dense retrieval, reciprocal rank fusion (RRF), and lightweight supervised rerankers on BEIR FiQA. The statistical-learning work is the training and evaluation of logistic regression and random forest ranking models.

**Status (14 November 2026): the original FiQA proposal is retained; TechQA now has three completed retrieval pipelines, six training sensitivity runs, independently checked SQL/ranking evidence and one frozen held-out evaluation.** The full pinned BGE FP32 corpus was encoded on the RTX 3060 through DirectML. LSA remains a separately identified baseline.

The held-out primary comparison does **not** establish a reliable learned-reranking gain over weighted RRF: Logistic nDCG@10 is 0.5788 versus 0.5847, with a paired component-bootstrap difference interval spanning zero. See [the current experiment report](project/results/iteration2_summary.md) and [the fixed protocol](project/protocol/TechQA_v2.md).

Start with [the implementation and commands](project/README.md) and [the exploratory configuration](project/configs/techqa_exploratory.json). The pinned neural backend is a separate option; results identify the backend actually used.

## Project documents

| Document | Purpose |
|---|---|
| [English proposal](project/proposal/5001_Proposal_RAG.md) | Seven sections matching the course template |
| [Proposal PDF](output/pdf/5001_Proposal_100032.pdf) | Existing formatted review/submission draft |
| [中文项目任务书](5001_RAG_项目任务书.md) | Original methods, validation and deliverables |
| [Experiment configuration](project/configs/rag_experiment.json) | Planned candidates, features, model budgets and metrics |
| [Source audit](project/data/fiqa/source_audit.json) | Verified source information and split counts |

The five member names and UIDs have been entered in the proposal. Team name and submission date remain to be filled; responsibilities are proposed. The proposal describes FiQA; the later [TechQA audit and recommendation](project/data/candidates/2026-11-12_technical_qa_audit/recommended_experiment_plan.json) motivated a separately named exploratory implementation rather than silently replacing that proposal.

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

## Original proposal dataset and comparisons

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
| 1 | Source and split audit, proposal, protocol | FiQA proposal plus pinned TechQA audit |
| 2 | Retrieval indexes and shared candidate runs | Head/full LSA and full-corpus BGE FP32 runs completed |
| 3 | Logistic/RF rerankers and grouped validation | Three-fold component CV and six one-factor sensitivities completed |
| 4 | Frozen test evaluation, latency and failure analysis | One frozen 84-query evaluation completed; sampling failure diagnosed |
| 5 | Evidence search, citation demo and presentation | Evidence CLI plus local generation workflow; generated outputs require qualitative review |

The implementation uses Python, scikit-learn, exact sparse/vector search and SQL experiment records. Sentence Transformers is optional for the local neural backend. Generated models, indexes and caches stay in ignored local folders; compact results and verification evidence live under `project/results/`.

## Course materials

- [Study notes and lecture navigation](notebook/README.md)
- [Lecture PDFs](output/pdf/)
- [Project navigation](project/README.md)

Assignments and generated homework outputs stay local and are excluded from Git tracking.

Earlier topic drafts are retained under [the project archive](project/archive/2026-11-07_prior_topics/README.md). RAG is the active project direction.
