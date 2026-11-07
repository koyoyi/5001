# SDSC5001 Project：RAG 检索与学习型重排序

当前方向（2026-11-07）：**面向 RAG 的混合检索与学习型重排序**。

研究问题：在相同候选文档上，自己训练的轻量排序模型，能否比固定融合更好地检索相关证据，且额外延迟可接受？

## 正式入口

- [英文 Proposal](proposal/5001_Proposal_RAG.md)
- [排版 PDF](../output/pdf/5001_Proposal_RAG.pdf)
- [中文任务书与实验协议](5001_RAG_项目任务书.md)
- [实验配置](configs/rag_experiment.json)
- [FiQA 来源与划分核验](data/fiqa/source_audit.json)
- [老师的 Proposal 模板](<proposal template/Team-XX-Proposal-Template.pdf>)

## 当前状态

已完成选题、数据来源和划分核验、研究设计与 Proposal。尚未运行检索模型、训练重排器或计算成绩。成员、UID、团队名和提交日期保留待填。

预训练 embedding／生成模型作为固定组件；5001 的统计学习部分为 Logistic／RF 重排、按 query 验证及效果与延迟比较。

## 历史记录

建筑、设施用电和推理服务性能方案均已归档，见 [旧主题索引](archive/2026-11-07_prior_topics/README.md)。它们不再是当前项目计划。

## 重建 Proposal PDF

在 5001 仓库根目录，用 Python 安装 [排版依赖](scripts/requirements-pdf.txt)，然后运行：

~~~powershell
python -m pip install -r project/scripts/requirements-pdf.txt
python project/scripts/build_proposal_pdf.py
~~~

PDF 从正式 Markdown 生成，输出到 output/pdf/5001_Proposal_RAG.pdf。脚本只负责文档排版，不执行 RAG 实验。
