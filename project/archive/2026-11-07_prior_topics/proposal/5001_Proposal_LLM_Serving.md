> Archived on 2026-11-07. The active project is [RAG retrieval and learned reranking](../../../5001_RAG_项目任务书.md).

# DSC5001 Project Proposal

**Project title:** Can Statistical Learning Guide Concurrency Selection for LLM Serving?

**Team name:** [TEAM NAME TO BE COMPLETED]  
**Submitted:** [DD MONTH 2026 TO BE COMPLETED]  
**Draft prepared:** 7 November 2026

| Member | UID | Main responsibility (proposed) |
|---|---|---|
| [MEMBER 1 NAME] | [MEMBER 1 UID] | Inference service, experimental design and data collection |
| [MEMBER 2 NAME] | [MEMBER 2 UID] | Data quality, SQL aggregation and baseline/Ridge models |
| [MEMBER 3 NAME] | [MEMBER 3 UID] | Random forest, grouped validation and configuration recommendation |
| [MEMBER 4 NAME] | [MEMBER 4 UID] | Independent verification, interpretation and presentation |
| [OPTIONAL MEMBER 5 NAME; REMOVE IF INAPPLICABLE] | [MEMBER 5 UID] | Reproducibility, robustness analysis and project coordination |

## 1. Background and Motivation

Operators of language-model services must choose how many requests the server can process concurrently while balancing response latency, successful completions and resource limits. A configuration that performs well under one request pattern may perform poorly when prompt lengths or arrival rates change. We will investigate whether supervised statistical learning can guide this decision more effectively than a fixed configuration: a pretrained LLM is the system under test, while Ridge regression and random forests are the performance models that we train and evaluate.

## 2. Research Questions

- **RQ1 - Predictive value:** Can Ridge regression or a random forest predict run-level tail latency for previously unmeasured workload-configuration combinations more accurately than training-mean baselines?
- **RQ2 - Interactions:** Do interactions between server concurrency, prompt length and offered request rate explain performance differences beyond an additive linear model?
- **RQ3 - Decision value:** Under predeclared latency and failure limits, does model-guided configuration selection achieve higher timely successful throughput than a fixed configuration, and how close is it to the best candidate measured afterwards?

These questions allow negative results: nonlinear models may add little predictive value, or prediction errors may make a simple fixed configuration preferable.

## 3. Data

- **Source and period:** We will generate controlled experiments locally during November 2026 using one pinned release of [llama.cpp server][1] and one fixed file from the official [Qwen3-0.6B-GGUF repository][2]. We will record the backend revision, model checksum, launch arguments and experiment dates. No measurements or trained-model results are available at proposal preparation.
- **Platform:** The available computer has an Intel i7-11800H, approximately 32 GB RAM and an RTX 3060 Laptop GPU with 6 GB VRAM. Backend compatibility and the final context/concurrency limits require a pilot.
- **Unit and planned size:** The primary observation is one complete experimental **run**, rather than one HTTP request. The initial plan has three server-slot settings (1, 2 and 4), three prompt-length profiles and four offered-rate levels, with at least three repeated runs per combination: approximately 108 run-level observations. Rates, run duration and request counts will be frozen after a separate pilot establishes measurement reliability and feasibility.
- **Responses:** The primary continuous response is run-level p95 time to first generated content (TTFT), with a training-time log transformation considered. A second continuous response is goodput: successful requests satisfying the frozen latency criteria divided by the declared observation duration. End-to-end latency, failures, timeouts and rejection rates are mandatory accompanying outcomes.
- **Predictors:** Only information available before a run: server slots, the planned prompt-token distribution, offered request rate and fixed generation budget. Limited interactions will be constructed. Measured throughput, queue length, GPU utilisation and realised output length are outcomes, not predictors for advance recommendation.
- **Quality and measurement:** Request logs will contain planned arrival, actual send, first valid generated text, completion, token counts and status. An asynchronous open-loop load generator will preserve the planned arrival process and expose client scheduling delays. Model precision, generation policy, cache policy and effective per-slot context capacity will be controlled. SQL queries will reconcile run summaries against request records, including failures and unfinished requests. Slots denote parallel sequence capacity, not a fixed batch size.

## 4. Proposed Methods

1. **Baselines for RQ1:** Use the training-set mean and the training-only mean for each slot setting. An additive regularised linear model provides a further reference for the interaction analysis.
2. **Ridge regression for RQ1-RQ2:** Standardise numerical predictors within each training fold, encode slot settings, and add a small prespecified set of interactions such as prompt length with offered rate and slot setting with offered rate. Compare additive and interaction models under identical validation splits.
3. **Random forest for RQ1-RQ2:** Fit a shallow, regularised forest to test whether nonlinearities improve held-out predictions. Limit tree depth and leaf complexity because the number of independent runs is small. A shared feature set and tuning budget will support a fair comparison.
4. **Configuration selection for RQ3:** Fit latency and goodput predictors using development data. For a known workload, score the three candidate slot settings and choose the candidate with the highest predicted goodput among those meeting the predicted latency criterion and lying within the training coverage. Return "no supported feasible configuration" if none qualifies. This screening rule does not guarantee a service-level objective: observed failure and rejection limits must also be checked in independent evaluation.

The learned performance models, generalisation tests and configuration decision form the statistical-learning contribution. Installing a pretrained LLM and comparing benchmark bars alone would not answer the research questions.

## 5. Validation and Evaluation

**Data separation and tuning.** Exclude pilot measurements from the final evaluation. Before fitting, reserve approximately 20% of complete workload-configuration combinations as an untouched test set. All repeated runs for a combination remain in the same partition. Use grouped cross-validation on the remaining combinations for preprocessing and a small hyperparameter search; neither individual requests nor repetitions of the same combination may cross folds. Freeze the test manifest, models and recommendation rule before revealing final outcomes.

**Independent decision test.** In a later measurement session, evaluate a small prespecified set of workloads and all candidate configurations in randomised order. Freeze each model recommendation before inspecting these results. Compare the recommendation with a fixed slot setting chosen using development data and with the best candidate identified retrospectively from the new measurements. The latter is an evaluation reference, not information available to the recommender. Repeat or block comparisons across sessions when feasible to assess sensitivity to laptop/system variation.

**Metrics and uncertainty.** Report MAE and RMSE for predicted p95 TTFT on the original time scale, along with errors by workload intensity. For recommendations, report goodput, latency-limit violations, failure/rejection rates and the gap to the retrospectively best feasible candidate. Report offered, sent, admitted (where applicable), completed, failed, rejected and unfinished request counts. Specify arrival, drain and timeout boundaries in advance and use a common evaluation duration; requests that fail or remain unfinished do not earn goodput. Successful-request p95 must never substitute for the full failure accounting. Use run/session-level paired summaries or block bootstrap intervals where the number of independent blocks permits; do not treat correlated requests as independent replicates.

**Interpretation and robustness.** Examine Ridge coefficients on the defined scale and held-out permutation importance cautiously. Compare additive versus interaction models and inspect light-load versus near-saturation errors. Check sensitivity to cache reset, run order and TTFT measurement from actual send versus planned arrival. Quantile stability will be assessed during the pilot; if p95 cannot be estimated reliably within the experiment budget, revise the primary response and freeze that revision before formal collection. Runs with undefined latency due to widespread failure will be recorded as infeasible/failed outcomes rather than silently deleted or assigned an invented p95.

## 6. Expected Contribution

The study will test whether a small, interpretable performance-learning pipeline adds practical value to concurrency selection on a local LLM service. Deliverables will include an asynchronous load generator, reproducible experiment configurations, SQL data-quality checks, a model-comparison notebook and an independently evaluated recommendation rule. This combines AI systems engineering with the regression, regularisation, model comparison and validation methods taught in DSC5001.

We will not train or fine-tune the language model itself, claim a new inference engine, or assume that random forests must win. Results will concern one model, backend and consumer GPU within the measured workload range. Limited independent runs, uncertain tail estimates, laptop variability and early generation termination may constrain the conclusions. The experiment is informed by public measurement guidance [3], but will not claim an official MLPerf result, production capacity, or transfer to other hardware without additional evidence.

## 7. Timeline

| Stage | Dates in 2026 | Milestone |
|---|---|---|
| Week 1 | 7-13 November | Compatibility and measurement pilot; freeze workload/metric protocol; implement logging and begin collection |
| Week 2 | 14-20 November | Finish collection and SQL checks; freeze grouped splits; fit baselines, Ridge and constrained random forest |
| Week 3 | 21-26 November | Independent recommendation test; interpretation, robustness and failure cases; freeze results |
| Submission preparation | 27-28 November | Prepare at most 10 slides, an 8-minute recording and code appendix; final checks and early submission |

The template deadline is **29 November 2026, 11:59 PM**; any subsequent course announcement takes precedence. The plan reserves at least one day of buffer. A custom admission proxy or dashboard is optional and will be attempted only after the core experiments and statistical evaluation are complete.

### References

1. ggml-org. *llama.cpp: HTTP server documentation*. Accessed 7 November 2026. [Official repository][1].
2. Qwen Team. *Qwen3-0.6B-GGUF model repository*. Accessed 7 November 2026. [Official model repository][2].
3. MLCommons. *MLPerf Inference Rules*. Accessed 7 November 2026. [Official measurement rules][3].

[1]: https://github.com/ggml-org/llama.cpp/tree/master/tools/server
[2]: https://huggingface.co/Qwen/Qwen3-0.6B-GGUF
[3]: https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc
