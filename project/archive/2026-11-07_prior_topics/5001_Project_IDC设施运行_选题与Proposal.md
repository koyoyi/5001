> Archived on 2026-11-07. The active project is [RAG retrieval and learned reranking](../../5001_RAG_项目任务书.md).

# SDSC5001：面向 IDC 职业技能的选题与 Proposal 草稿

> 状态更新（2026-11-07）：用户明确要求不再采用暖通方向，改为计算机系统与性能工程。新的首选入口：[AI推理性能任务书](5001_求职技能任务书_AI推理性能.md)。本文保留为旧方案。

更新：2026-11-07。状态：推荐方案，尚未开始模型训练，未产生预测成绩，未提交课程系统。此前“未见建筑几何泛化”方案未被采用，本文件是新的讨论入口。

## 1. 推荐结论

**推荐做：IDC 冷却及辅助设备用电的短期预测与工况分析。**

建议英文题目：

**Can IT Load and Weather History Improve One-Hour-Ahead Cooling Power Forecasts in a Data Centre?**

核心问题：在已经知道设备近期功率变化的情况下，额外接入 IT 负载与室外温湿度历史，是否真的能改善下一小时冷却分项功率预测？

这个题与你目前“设施智能化／能源运行分析、数据中心热系统数字化”的求职方向相符。它能同时练到测点理解、数据质量、SQL、统计学习、验证和英文工程表达。这里的职业匹配判断依据你现有职业规划，不代表某家企业的录用承诺。

5001 保持统计学习主线；数据处理与展示构成作品交付。暂不把控制器、复杂云平台或深度学习加入课程必做范围。

## 2. 三个方向怎么选

| 顺位 | 题目 | 职业能力增量 | 取舍 |
|---|---|---|---|
| **推荐** | ESIF 数据中心冷却分项功率预测 | 设施测点、天气与负载关联、SQL 时窗、数据质量、模型验证 | 更直接连接暖通与设施数字化；须先处理长时间跨度内的质量和工况变化 |
| IDC 备选 | Marconi A2 节点入口温度预测 | 节点遥测、滞后特征、跨设备误差、温度趋势解释 | 与已有 Eagle 工作更相似；入口空气温度不同于 GPU 芯片温度，仍有学习增量 |
| 工程技能备选 | 自建 AI 推理性能评测与配置推荐 | Python、SQL、性能测量、日志和交付 | 更贴近软件性能与平台方向；与冷却设施岗位的连接较弱，需要自己采集公平、可重复的基准数据 |

如果 ESIF 的时间覆盖和字段能够支持建模，就选第一项。第一阶段审计若发现连续有效窗口不足，再切换温度预测；不必为了标题包含 IDC 而虚构冷却优化任务。

### 与现有课程项目的分工

| 项目 | 主要对象与问题 | 主要证据 |
|---|---|---|
| 5001 新建议 | 设施冷却分项功率；历史外生信息是否增加预测价值 | 持续性／自回归／Ridge／RF 比较，时间外测、消融、工况误差 |
| 5002 现有 Eagle | 作业申请、等待、资源使用及 GPU 遥测 | 大文件处理、覆盖率、分层统计与可视化 |
| 5003 现有数据库项目 | 索引机制与查询性能 | SQL、执行计划、正确性和公平基准 |

同属数据中心领域，但数据、研究问题及交付证据不同。课程对跨课程材料复用若另有规定，应以课程要求为准。

## 3. 已核实的数据与尚未核实的部分

来源：Clark 与 Strelka 发布的 **NLR HPC Facility Power Usage Effectiveness (PUE) Data**，研究对象是 ESIF HPC 数据中心。

- [官方数据目录与下载](https://data.nlr.gov/submissions/300)
- [数据 DOI](https://doi.org/10.7799/3015212)
- [OEDI 镜像入口](https://data.openei.org/submissions/8616)
- [官方功率计量边界说明](https://www.nlr.gov/computational-science/measuring-efficiency-pue)
- [官方冷却系统说明](https://www.nlr.gov/computational-science/data-center-cooling-system)
- [原站数据许可](https://data.nlr.gov/node/300/license)

本次只读取网页、说明及 Parquet 元数据，没有下载全部运行数据。核验记录保存在 [source_metadata_audit.json](data/esif_pue/source_metadata_audit.json)。

### 3.1 文件与实际字段

官方列出的功率、天气 Parquet 文件分别约 104.6 MB 和 56.1 MB；压缩 CSV 是同类数据的另一种格式，不必两套都下载。

已读取的 Parquet 元数据：

| 文件 | 原始记录数 | 时间字段的元数据范围 | 含义 |
|---|---:|---|---|
| 功率 | 4,569,543 | 2015-11-10 至 2025-08-29 | 不是已清洗的独立小时样本数 |
| 天气 | 4,034,917 | 最小值落在 1970-01-01，最大值到 2025-08-29 | 最早时间明显需要核查，不能当作正常观测起点 |

这些是文件元数据证据，不证明中间连续、等间隔或全部有效。网页的提交／更新日期也不是观测时段。

| 字段 | 本项目中的角色 | 工程口径 |
|---|---|---|
| cooling_kw | **主预测目标** | 原站定义的室外冷却设备风机、伴热等用电类别，含专用塔过滤泵的计入功率；不等于整套冷却系统用电 |
| pump_kw | 补充目标或历史输入 | 水循环及相关增压泵类别，不能仅凭字段名推断具体运行设备 |
| hvac_kw | 补充目标或历史输入 | 风墙、电气房风机盘管、补风等类别 |
| it_power_kw | 历史输入 | IT 设备功率 |
| outdoor_air_temp | 历史输入 | 文件实际名称；网页写作 outside_air_temp，温度单位为 °F |
| outdoor_air_humidity | 历史输入 | 文件实际名称；网页写作 outside_air_humidity |
| pue | 描述性参考 | 先核分子、分母、时间聚合及坏值，不作主目标 |

代码以实读 schema 为准，统一别名并记录映射。tags、unit 等元数据也要检查，避免混合不同来源或重复序列。

**边界处理：** 主任务先预测 cooling_kw，pump_kw、hvac_kw 分别作扩展，不直接相加称为“总冷却功率”或“设施总功率”。按原站分类，2.67 kW 的专用塔过滤泵已归入 cooling 类，不应再次补加。

### 3.2 第一阶段必须解决的数据问题

1. **采样与覆盖**：实际采样间隔尚未实测；先检查间隔分布、重复时间、长缺口及各字段覆盖，再决定小时聚合规则。
2. **时间与时区**：Parquet 时间戳没有声明 UTC 调整，不能直接标为 UTC。先保留源时间基准，查明时区后再转换与解释日历特征。
3. **异常值**：元数据显示负功率、负 PUE 和极大比率。不能把负值一律填零，也不能靠随意截断让模型变好；应核单位、状态、计量含义并保留质量标记。
4. **长期工况变化**：十年跨度可能混合设备或系统变化。先选覆盖充分的连续窗口，不默认十年属于同一数据生成机制。
5. **可用样本量**：冻结窗口和清洗规则后，统计有效小时数、训练／验证／测试数量以及每项排除原因，再填进正式 Proposal。

初步计划优先检查一段完整的 12 个月窗口。窗口选择依据时间覆盖和可解释的数据质量规则，须在模型评分前冻结；不能根据哪个测试年份分数更好来挑窗口。

## 4. 研究问题与预测时点

### RQ1：额外测点信息有没有用？

在相同自回归特征和日历特征基础上，加入历史 IT 负载、历史天气，能否降低时间留出集上的预测误差？若没有改善，这同样是有效结果。

### RQ2：非线性是否值得？

Ridge 加入有限的温度—负载交互后，随机森林是否仍有稳定增益？额外复杂度能否被误差改善和工况解释支持？

### RQ3：模型在哪些工况下失效？

在高负载、较高室外温度和负载快速变化时，误差是否更大？这种现象是否同时伴随数据覆盖下降或超出训练范围？

### 统一预测定义

在整点 t，用截至 t 已结束的小时及更早观测，预测下一个完整小时：

$$
y_t=\frac{1}{1\ \mathrm{hour}}\int_t^{t+1\mathrm{hour}}P_{\mathrm{cooling}}(s)\,ds.
$$

输入窗口例如 [t−1h, t)、[t−2h, t−1h)，标签窗口为 [t, t+1h)。这是下一小时平均功率，单位 kW。

- 不规则采样优先采用有最大可延续间隔限制的时间加权均值；不能跨长缺口延续读数。
- 覆盖不足的小时保留标记并按预先冻结规则排除或处理；不把缺测填零。
- 对齐天气时只能使用预测时点已出现的观测，不能用未来最近邻值。
- 未来一小时真实 IT 功率、真实天气与真实 PUE 均不得作为该小时的预测输入。
- 数据只有实测天气，因此本题研究历史天气信息，不能写成使用天气预报。

## 5. 方法与公平验证

### 5.1 模型阶梯

1. **持续性基线**：下一小时与刚结束小时相同。
2. **昨日同小时基线**：使用 24 小时前的相应小时；缺测时按事先规定的回退规则处理。
3. **自回归 Ridge**：功率滞后、过去滚动统计和统一的日历特征。
4. **带 IT／天气的 Ridge**：加入历史 IT、温湿度及少量预先说明的交互；检查标准化和共线性。
5. **随机森林**：检验非线性和交互是否增加泛化能力。若时间充裕，再用样条／GAM 补解释，不必堆模型。

滚动统计必须只看过去。标准化、插补、特征筛选和调参均在相应训练折内完成。

### 5.2 特征消融

| 组别 | 共同部分 | 增加部分 | 回答的问题 |
|---|---|---|---|
| A | 功率历史＋日历 | 无 | 自回归已能解释多少变化？ |
| B | 与 A 一致 | IT 历史 | 负载信息有无额外价值？ |
| C | 与 A 一致 | 天气历史 | 环境信息有无额外价值？ |
| D | 与 A 一致 | IT＋天气历史 | 两者是否互补？ |

各组使用相同时间窗、同一测试时刻、相同调参预算。先完成 Ridge 的四组，再进行 RF 比较。

### 5.3 时间验证

- 冻结有效窗口后，将最后约 20% 时间作为最终测试，具体日期写入配置。
- 前约 80% 用扩展窗口或滚动起点验证，至少三折；训练标签必须在该折预测起点之前已完整观测。
- 最终测试不参与调参、清洗阈值选择或窗口选择，结果只在冻结模型后计算。
- 测试中允许使用当时已观测到的前一小时功率进行滚动一步预测；这不等于在测试起点一次预测整个未来月份。模型参数不随测试标签反复重训。
- 各模型在相同有效测试小时上比较，并报告共同覆盖率和被排除小时。

### 5.4 指标和解释

主指标 MAE，补充 RMSE；报告相对持续性基线的技能分数：

$$
\mathrm{Skill}_{MAE}=1-\frac{MAE_{\mathrm{model}}}{MAE_{\mathrm{persistence}}}.
$$

正值代表改善，负值代表退步；基线误差为零时不计算这个比率。与昨日同小时的比较另列，不能混换分母。

比较同一时刻的配对误差，按日或周进行区块重采样估计不确定性，并说明区块长度敏感性。分层分析优先按预测时点已知的负载／天气条件；若用未来真实工况分层，应明确属于事后诊断。

解释包括 Ridge 系数、时间留出上的特征组增量、受相关特征影响的置换重要性，以及两段具体运行案例。统计关联不能直接解释为调高某个设定值后的因果收益。

## 6. 职业技能与具体交付

| 能力 | 要完成的实际工作 | 面试可展示的证据 |
|---|---|---|
| 设施工程理解 | 画出 IT 发热、液冷／空气侧、热回收和散热设备关系；标清实际可用测点 | 一张系统图及字段边界表 |
| 数据处理 | Parquet 按列读取、时间／单位统一、重复和覆盖检查 | 可重跑脚本、质量报告、纳入排除表 |
| SQL | 建小时事实表、质量表、预测表；使用窗口函数生成或复核滞后 | SQL 文件及与 Python 对照的样本结果 |
| 统计学习 | 基线、Ridge、RF、消融、时间外测 | 模型比较表、逐工况误差、失败案例 |
| 运行展示 | 时间线同时显示实际／预测功率、天气、IT 负载和数据覆盖 | 一个简洁面板或可复现报告 |
| 复现与表达 | 依赖、配置、日志、英文 README、个人分工 | 他人能复跑的入口和 5 分钟英文讲解 |

建议 SQL 使用本地 DuckDB 直接查询 Parquet；若岗位练习明确需要 MySQL，再导入小时表。课程阶段选择一个工具完成交付即可。

代码流可以整理成：

~~~text
01_source_and_quality.ipynb      来源、schema、覆盖、单位与时间核查
02_hourly_features.ipynb        小时聚合、只看过去的特征、SQL核对
03_baselines_and_ridge.ipynb    持续性、自回归、四组消融
04_random_forest.ipynb         限定调参预算的非线性比较
05_evaluation_and_cases.ipynb  最终外测、工况误差、两段案例
~~~

以上是拟交付结构，尚未创建这些 Notebook，也不表示 SQL、面板或实验已经完成。FastAPI／Docker 可在课程核心结论固定后加一项，不作为统计结果成立的前提。

## 7. 工程适用范围与切换条件

ESIF 采用温水液冷、热回收及室外散热设备；官方说明其利用当地半干旱气候，不采用机械制冷。它是学习真实数据中心设施数据的合适案例，但不代表香港／新加坡常见机房的气候和冷却架构。[官方液冷说明](https://www.nlr.gov/computational-science/warm-water-liquid-cooling)

本项目可支持“在该设施、该时段的离线功率预测与误差分析”。没有供回水温度、流量、阀位、控制设定或干预记录，就不能据此计算现场 COP、设计闭环控制或声称已经节能。

PUE 是补充描述指标：低 IT 负载能让比率显著变化；用构成它的同小时功率分项预测同小时 PUE，容易只是重建公式。期间能效应先核计量边界，再按能量比计算，不能直接把所有瞬时 PUE 做算术平均当年度 PUE。

**第一阶段的切换条件：** 若在固定时间预算内无法获得足够连续、可靠且可解释的功率／天气窗口，则改做以下备选，不反复扩张清洗范围。

### IDC 备选题

**Can Temperature and Power History Improve 30-Minute-Ahead Server Inlet Temperature Forecasts?**

[HazardNet 官方数据](https://zenodo.org/records/10050368) 包含 Marconi A2 的节点入口温度、出口温度与功率，覆盖 2019-01-14 至 2019-12-31、3,312 个节点，压缩文件合计约 1.02 GB。先按训练期覆盖选定 32–64 个节点，对比持续性、Ridge 和 RF；采样频率仍须实读文件确认。

主任务选连续温度预测，不把由温度规则生成的“高温事件”当作大量现场故障标签。该数据不能直接支持 PUE 或冷却能耗优化；作者完整深度学习框架不列为本课程必复现内容。[作者代码](https://github.com/MSKazemi/HazardNet)

## 8. 英文 Proposal 草稿

Team / members / UIDs / responsibilities: **to be supplied by the group**.

### Project title

Can IT Load and Weather History Improve One-Hour-Ahead Cooling Power Forecasts in a Data Centre?

### 1. Background and Motivation

Data-centre facilities combine computing loads with thermal-support equipment. A useful operational model should demonstrate value beyond simply carrying the most recent measurement forward. We will investigate whether historical IT load and outdoor conditions improve short-term forecasts of the documented cooling-power category at the ESIF HPC data centre. This is an offline forecasting study intended to inform monitoring-data selection and engineering review.

### 2. Research Questions

1. Do historical IT load and weather improve forecasts beyond a common autoregressive and calendar-feature baseline?
2. Does a random forest provide reliable improvements over Ridge regression with limited interaction terms?
3. How do errors change across operating conditions and data-coverage levels?

### 3. Data

We will use the NLR ESIF facility PUE dataset, DOI: 10.7799/3015212. Parquet metadata indicate approximately 4.57 million power records and 4.03 million weather records; these are raw records rather than independent hourly observations. We will audit sampling intervals, timestamps, source identifiers, units, missingness and invalid readings before freezing a continuous analysis window. Exact dates and usable hourly sample counts will be reported after this audit.

The primary response will be the next complete hour's average cooling_kw. Historical IT power, weather and permitted lagged power measurements will provide predictors. Pump and HVAC power may be examined separately as extensions. The cooling field denotes the provider's measurement category rather than total cooling-system electricity. Future realised weather and IT load will not be used as forecast-time inputs.

### 4. Proposed Methods

We will compare persistence and previous-day benchmarks, autoregressive Ridge regression, Ridge with additional IT/weather features, and a random forest. Four matched feature groups will isolate the incremental value of IT history, weather history and their combination. Preprocessing and hyperparameter selection will be fitted within each training fold.

### 5. Validation and Evaluation

After the analysis window is frozen, approximately the last 20% of its timeline will be reserved for final testing. Earlier observations will support rolling-origin validation. At each forecast origin, all features and training labels must already be observable. Testing will evaluate rolling one-hour-ahead predictions with frozen model parameters.

We will report MAE, RMSE, improvement over each naive benchmark, and paired error differences with day- or week-block resampling. Models will be compared on common valid hours. Operating-condition analyses and feature ablations will assess robustness. The final test set will not be used to choose models or data-cleaning thresholds.

### 6. Expected Contribution and Limitations

The contribution is evidence about whether additional operational measurements justify their integration beyond a strong autoregressive baseline. No improvement is also an informative result. Deliverables will include an auditable hourly dataset, reproducible model comparisons, SQL transformations and an engineering interpretation of failure cases.

The study concerns one historical facility with a particular cooling architecture and climate. It will not establish causal energy savings, fault-detection accuracy or the performance of an implemented controller.

### 7. Timeline

- 7–10 November: obtain the selected source files, audit quality and freeze the task/window.
- 11–17 November: build hourly data, SQL checks, baselines and Ridge ablations.
- 18–23 November: complete random-forest comparisons, robustness analyses and cases.
- 24–27 November: freeze results, prepare slides, recording and code appendix.
- 28 November: reproduction check and submission buffer.

The locally stored course template specifies a final deadline of **29 November 2026, 23:59**, a maximum of ten slides and an eight-minute recording. Later course announcements take precedence.

## 9. 当前状态与正式提交前待补

- 已完成：课程模板核对、职业方向匹配、官方来源及文件元数据核查、研究设计草稿。
- 待完成：下载所选运行数据、确认采样／时区／计量边界、冻结窗口及实际 N、训练和测试。
- 待小组填写：成员、UID、角色、团队名与课程最新要求。
- 数据使用：保留原站完整许可 notice 并署名；不能因镜像页面的标识就自行替换原站条款。
- 本文没有实验数值、已部署系统或节能成绩；这些内容须以实际完成记录补充。
