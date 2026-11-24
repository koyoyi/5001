# SDSC5001 Project：RAG 检索与学习型重排序

当前方向：**面向 RAG 的混合检索与学习型重排序**。2026-11-14 已完成第二轮 TechQA 实验和一次冻结保留集评价；FiQA Proposal 作为原提交方案保留。

研究问题：在相同候选文档上，自己训练的轻量排序模型，能否比固定融合更好地检索相关证据，且额外延迟可接受？

## 正式入口

- [最新实验结果与配对区间](results/iteration2_summary.md)
- [独立重复在线耗时](results/iteration2_latency.md)
- [生成演示与实际失败案例](results/generation_demo_review.md)
- [第二轮协议和停止规则](protocol/TechQA_v2.md)
- [冻结保留集核验](results/frozen_e2324b7a967c5dbe2766ecc3/verification.json)
- [英文 Proposal](proposal/5001_Proposal_RAG.md)
- [排版 PDF](../output/pdf/5001_Proposal_100032.pdf)
- [中文任务书与实验协议](../5001_RAG_项目任务书.md)
- [实验配置](configs/rag_experiment.json)
- [FiQA 来源与划分核验](data/fiqa/source_audit.json)
- [老师的 Proposal 模板](<proposal template/Team-XX-Proposal-Template.pdf>)

## 当前状态

已完成原研究设计与 Proposal，以及 TechQA 的 head/full LSA、完整 BGE FP32、统一候选、Logistic／RF、三折组件 CV、六项训练对照、配对区间、SQLite 和证据检索。一次冻结评价使用 84 个保留查询／72 个组件，主要对比 Logistic 减加权 RRF 的 nDCG@10 差值为 −0.00598，95% 区间 [−0.02343, 0.01179]；目前不支持学习重排稳定优于强融合。完整数值见上方报告。成员 UID 和拟议分工保留，真实贡献与最终提交信息仍需实际填写。

预训练 embedding／生成模型作为固定组件；5001 的统计学习部分为 Logistic／RF 重排、按 query 验证及效果与延迟比较。

## 安装与运行

以下命令从 **5001 仓库根目录** 执行，不需要修改全局 Git 或 Python 设置。推荐使用独立虚拟环境。

```powershell
python -m pip install -r project/requirements.txt
python -m pytest project/tests -q
python project/scripts/run_experiment.py --config project/configs/iteration2/head_lsa.json
```

采用已下载且有 SHA-256 清单的 NVIDIA TechQA JSON／ZIP，不自动联网。当前使用 [iteration2 配置](configs/iteration2/)，原 [techqa_exploratory.json](configs/techqa_exploratory.json)作为首轮历史。运行会重新核验源文件、建立索引、训练并评价，产物写入 `project/artifacts/<run_id>/`；该目录被 Git 忽略。紧凑指标、图表与核验记录保存在 `project/results/<run_id>/`。

获得运行目录后核验、检索和测量：

```powershell
python project/scripts/verify_experiment.py project/artifacts/<run_id> --recompute-candidates
python project/scripts/search.py project/artifacts/<run_id> --query "How do I resolve an IBM WebSphere connection timeout?" --method logistic --limit 3
python project/scripts/benchmark_search.py project/artifacts/<run_id> --queries 20 --repeats 5
```

把 `<run_id>` 替换为实际目录名。主神经运行是 `20261114T144053296846Z_onnx`；冻结评价是 `frozen_e2324b7a967c5dbe2766ecc3`。检索展示原文摘录与文档 ID，模型分数只用于排序。冻结本地生成是独立定性演示，不构成答案正确率评估。

### 神经向量后端

本机推荐 Windows DirectML 路径，RTX 3060 为 DXGI device 1。选择与硬件相符的独立环境；CPU 和 DirectML 的 ORT 包分别使用各自 requirements：

```powershell
python -m pip install -r project/requirements-directml.txt
python project/scripts/download_encoder.py --onnx
python project/scripts/run_experiment.py --config project/configs/iteration2/head_onnx.json
```

CPU／原 Torch 路径仍可单独运行：

```powershell
python -m pip install -r project/requirements-neural.txt
python project/scripts/download_encoder.py
python project/scripts/run_experiment.py --backend neural
```

下载脚本固定 `BAAI/bge-small-en-v1.5` 的 revision，记录每个文件 SHA-256，并保存到被忽略的 `project/models/`。后端只加载本地模型，不自动下载或执行远程模型代码。Query 使用 BGE 英文检索指令；文档无前缀；向量归一化并做精确 cosine 检索。

本机 Torch 是 CPU 版，但硬件有 RTX 3060 Laptop；通过隔离的 DirectML 后端使用 GPU，无须改驱动或全局 Torch。兼容依赖放在被忽略的 `.vendor-hf/`、`.vendor-onnx-compat/`、`.vendor-onnx-dml/`；新机器推荐独立环境。GPU 首次完整编码约 550 秒；与 19,552 篇已有 CPU 向量的逐文档数值对照通过。缓存加载和首次编码的开销分别记录，见 [GPU 证据](results/neural_corpus_backend_comparison.json)及主运行的 `embedding_build.json`。

## 当前实验口径

- 原 Proposal／`rag_experiment.json` 仍描述 FiQA；`techqa_exploratory.json` 与结果中的数据审计定义 TechQA 探索轨道，两者不混用。
- 输入原始 28,481 篇文档；按完全相同正文合并后为 28,478 篇。581 个规范问题中，第二轮按预设语义规则整体隔离了一个跨旧分区的 MQ 问题组件，当前保留 578 个。
- 共享 gold、字符近重复与已接受语义边构成组件。历史分区锁定，其余样本不重新随机分配；当前为 train/dev/test 410／84／84 个查询、336／72／72 个组件。语义审阅由 Codex 助手完成，实际成员／领域专家尚未独立核验，覆盖边界和源标注冲突见 group_review。
- 各检索器均接收每篇文档前 256 个空白分隔词；原文和原始文档 ID 保留。神经编码仍受其 WordPiece 长度上限限制。此 head-only 方案会漏掉长文后半段证据，需要后续分块与文档级聚合实验。
- BM25 和语义检索各取 top100；普通／加权 RRF、Logistic 和 RF 排序完整并集，不补 gold、不先截断。加权基线包含语义权重 0 的 BM25-only 端点。两种原始分数均对并集补算，缺榜名次另设标记。
- 只从训练候选内抽取至多 50 个未标注文档作代理负例，每个查询总权重相同；未召回正例的查询仍进入总体评价。CV 按组件分组，标准化仅在训练折拟合；超参数按训练 CV 选择，RRF 常数只看开发集。
- 开发集报告 query 宏平均 nDCG@10、Recall@20 与候选召回；置信区间按独立组件配对重采样。每个问题当前只有一篇已知 gold，因此 Recall@20 也是该已知文档的 hit@20，不能解释为穷尽证据召回。
- 开发配置仍是 `final_test_enabled=false`。2026-11-14 已对另行冻结的模型／候选／特征／协议评价保留集一次；该独立结果和哈希可核验，不再据这批测试分数继续选参。本地预先记录协议没有对外注册，开发区间和次要比较均按探索证据解释。
- SQLite 保留原始 ID、分区、qrels、完整候选及其特征、各方法 top 20 和配置；没有把测试标签用于特征或拟合。完整候选检索列表由新的运行另存 `candidate_runs.json`，早期本轮运行可从本地索引重算核验。

源码位于 [src/ragstudy](src/ragstudy/)，入口位于 [scripts](scripts/)，重要的数学与隔离性质由 [tests](tests/) 检查。复核 notebook 见 [notebooks](notebooks/)。

## 本地生成演示

[五例结果与限制](results/generation_demo_review.md)保留完整失败、拒答和有内容回答。生成器使用冻结 Qwen 0.5B、贪心解码和两个独立模型调用，分别生成内容与来源标签；来源窗口只取已检索文档。引用语法、内容可用性与语义支持分开核查，不把引用格式通过当作答案正确。

```powershell
python -m pip install -r project/requirements-generation.txt
python project/scripts/run_generation_demo.py --download-model --generator qwen2.5-0.5b-instruct
python project/scripts/run_generation_demo.py --artifact project/artifacts/20261114T144053296846Z_onnx --generator qwen2.5-0.5b-instruct --limit 5 --threads 2
```

当前案例已实际运行；重跑会产生新的历史记录。小模型质量有限，原始输出可审阅，实际成员的独立评审仍待完成。

## 命令行 RAG 演示

[rag_demo.py](scripts/rag_demo.py) 串起完整流程：问题 → 检索排名 → 原文证据 → 生成答案 → 引用来源 → 分层诊断。默认使用主运行 `20261114T144053296846Z_onnx` 和报告推荐的加权 RRF，取前 3 篇给生成器；`--method` 可换成 logistic、RF 等对比。

```powershell
python project/scripts/rag_demo.py --list-examples          # dev 题按检索结果分为漏检／排序失误／命中
python project/scripts/rag_demo.py --example DEV_Q014       # 已知题目，带 gold 诊断
python project/scripts/rag_demo.py --replay                 # 重新解释已保存的 5 个 Qwen 案例，不加载模型
python project/scripts/rag_demo.py --generator none         # 交互输入问题，只看检索和证据
```

诊断分三层：检索层（gold 是否进候选、排第几、有没有交给生成器）、证据层（问题或证据被 token 预算截断）、生成层（拒答、只输出标签、引用未确认）。gold 诊断只对 train/dev 原题显示；测试集问题即使原文输入也不显示任何 gold 信息，保留集保持冻结。所有答案都标明"正确性未自动核验"。`--save` 把完整结果另存到被忽略的 `project/artifacts/<run_id>/rag_demo_history/`。

生成器默认是冻结的本地 Qwen2.5-0.5B（与上节相同的 120 词证据窗口和两阶段引用）。可选 `--generator claude` 调用 Claude API（默认 `claude-opus-5`，可用 `--claude-model` 改）：

- 每篇检索到的文档以纯文本文档发送（前 3,000 词，超出时诊断里会注明），开启 citations；返回的每段引用都带字符偏移，脚本逐段核对是否与原文完全一致。
- 开启了 server-side fallback（`fallbacks="default"`），被安全分类器拒绝时由服务端改用推荐模型，诊断中会注明实际回答的模型。
- **问题和文档会离开本机。** 脚本启动时打印目标端点；凭据按 SDK 默认顺序解析（`ANTHROPIC_API_KEY`、`ANTHROPIC_AUTH_TOKEN`、`ant auth login` 配置），`ANTHROPIC_BASE_URL` 若已设置也会生效。
- SDK 装在被忽略的 `project/.vendor-claude/`，不改全局 Python：`python -m pip install --target project/.vendor-claude --no-deps -r project/requirements-claude.txt`。

Claude 路径的测试用 SDK 的 mock transport 离线完成，不发送真实请求；这里的生成结果同样是定性演示，不是问答准确率评估。

五本 notebook 已对真实本地运行执行。重建／重跑方式如下；第一条会重建 notebook 源文件，第二条执行并记录哈希与代码单元数：

```powershell
python -m pip install -r project/requirements-notebooks.txt
python project/scripts/build_review_notebooks.py
python project/scripts/execute_review_notebooks.py project/artifacts/<run_id> --kernel python3
```

本机 Jupyter 组件在内核退出时出现原生断言，因此另在被忽略的 `project/.vendor-jupyter/` 隔离使用 nbclient 0.10.2／pyzmq 26.4.0，并调整为先关闭客户端通道、再退出内核。复跑脚本会同时为客户端和内核引入该目录；未改全局环境。

## 历史记录

建筑、设施用电和推理服务性能方案均已归档，见 [旧主题索引](archive/2026-11-07_prior_topics/README.md)。它们不再是当前项目计划。

## 重建 Proposal PDF

在 5001 仓库根目录，用 Python 安装 [排版依赖](scripts/requirements-pdf.txt)，然后运行：

~~~powershell
python -m pip install -r project/scripts/requirements-pdf.txt
python project/scripts/build_proposal_pdf.py
~~~

PDF 从正式 Markdown 生成，输出到 `output/pdf/5001_Proposal_100032.pdf`。脚本只负责原 Proposal 排版，不执行 RAG 实验；本轮没有覆盖现有 PDF。
