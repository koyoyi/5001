# Data sources

The original proposal uses BEIR FiQA; see [the source audit](fiqa/source_audit.json). The executable implementation now has a separately configured TechQA exploratory track using the [pinned local audit](candidates/2026-11-12_technical_qa_audit/techqa_audit.json). Its custom component-disjoint splits must not be reported as official FiQA or TechQA benchmark splits.

No full FiQA corpus or results were generated for the proposal. The TechQA JSON and ZIP are already available under the audited candidate folder, with hashes in its download manifest. The experiment checks these hashes, keeps only original corpus text, merges exact document aliases, excludes mixed-answerability duplicate questions and groups shared-gold/automatically near-duplicate cases before splitting. Indexes and models are written to ignored `project/artifacts/` and `project/models/`, not to this source folder. Human semantic duplicate review remains pending before final test.

Earlier Energy Efficiency and ESIF metadata have moved to [the topic archive](../archive/2026-11-07_prior_topics/data/).
