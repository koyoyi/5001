> Archived on 2026-11-07. The active project is [RAG retrieval and learned reranking](../../5001_RAG_项目任务书.md).

# SDSC5001：选题建议与Proposal雏形

> 状态更新（2026-11-07）：用户未采用本文的建筑几何方向，改为优先考虑职业技能及 IDC 场景。新的讨论入口：[IDC设施运行选题与Proposal](5001_Project_IDC设施运行_选题与Proposal.md)。本文保留作旧方案参考。

整理日期：2026-11-07。依据本地 [老师的Proposal模板](<../../proposal template/Team-XX-Proposal-Template.pdf>)、5001课程Topic 1–9及公开数据核验。本文供小组讨论，尚未训练模型、没有实验成绩，也未提交。

## 1. 建议选这个题

**中文：统计学习模型能否泛化到未见建筑形状？——供暖与制冷负荷预测中的模型与验证方法比较**

**English: Can Statistical Learning Models Generalise to Unseen Building Geometries? A Study of Heating and Cooling Load Prediction**

题眼是：**模型能预测“已见形状的新配置”，是否也能预测“没见过的建筑形状”？** 通过建筑几何、朝向与玻璃配置，研究泛化条件，而非只比较几个模型的最高分。

它与你的HVAC/建筑背景衔接，也与6004道路碰撞、5002计算负载项目区分。无需新建物理仿真、使用GPU或训练深度网络，可把工作放在研究设计、模型比较与解释。

## 2. 三个候选的取舍

| 顺位 | 候选 | 优点 | 主要限制 |
|---|---|---|---|
| 首选 | 未见建筑几何上的供暖/制冷负荷泛化 | 数据小、结构明确，适合回归、正则化、非线性和分组验证 | 模拟数据，只有12种基础形状 |
| 备选一 | 室内温湿度能否在天气之外增加家庭电器能耗估计价值？ | 真实传感数据，可做特征增量与时间留出 | 单户约4.5个月；天气插值，未来预测的时点需额外核查 |
| 备选二 | 天气与日历怎样共同解释共享单车需求？ | 需求直观，模型容易展示 | 与6004同属交通域；组成目标的计数列不能作为输入 |

备选资料：[UCI Appliances Energy Prediction](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction)、[UCI Bike Sharing](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)。Appliances官方列19,735条10分钟记录；Bike的官网实例数与常用hour.csv计数有差异，真做时以下载文件核验。预测cnt不能使用casual、registered，两者之和就是目标。

首选是基于课程契合、研究问题清晰和可完成性作出的判断，不是认为模拟数据更接近真实建筑运行。

## 3. 数据已经实际核验

来源：[UCI Energy Efficiency](https://archive.ics.uci.edu/dataset/242/energy+efficiency)，Tsanas与Xifara，2012；已保存 [原始ENB2012_data.xlsx](data/energy_efficiency/ENB2012_data.xlsx)。

- 768行，8个输入X1–X8、2个响应Y1/Y2，无缺失。
- 一行是一个**模拟设计配置**，不是一栋独立监测建筑，也不是时间记录。
- X1–X5组合恰好12种基础几何，每种64条。
- Y1为Heating Load、Y2为Cooling Load。官方变量表未清楚给响应单位，未核原论文前不自行标成kW、kWh或年耗电量。
- 数据由Ecotect模拟生成，不含现场节能干预或控制收益。

| 输入组 | 列 | 处理 |
|---|---|---|
| 基础几何 | X1紧凑度、X2表面积、X3墙面积、X4屋顶面积、X5总高度 | 定义几何组并检查冗余 |
| 朝向 | X6 | 按类别编码，不机械视为等距连续量 |
| 玻璃配置 | X7玻璃面积、X8分布 | 连续比例/配置类别分别处理 |

**数据特有的方法问题：** 原始数值精确满足 **X2 = X3 + 2×X4**。带截距的五个几何输入共六列，数值秩只有五。OLS应预先移除冗余X2，不能声称全部系数均为唯一独立效应；Ridge可比较正则化，但不能将相关变量解释成独立物理贡献。

只以设计变量预测响应：预测Y1不加入Y2，反向亦然。另一个模拟响应在设计应用时通常并非已知输入。

核验记录：[数据审计](../../../tmp/project_topic_20261107/data_audit.json)、[候选分组协议](../../../tmp/project_topic_20261107/candidate_split_plan.json)。这些只检查结构和拟定协议，没有模型成绩。

## 4. 三个研究问题

**RQ1：改变验证目标，模型性能与排名如何变化？**

在同一开发数据上比较随机行CV和按几何分组CV。前者针对已见形状的新配置，后者针对未见几何。不能预先宣布随机划分“作弊”或一定分数更高；它们估计不同情境。

**RQ2：非线性模型相对线性模型的改善，是否保留到新几何？**

比较OLS/Ridge与Random Forest；有余力再加低复杂度样条/加性模型。不预设越复杂越好。

**RQ3：几何以外，朝向与玻璃配置增加多少预测信息？**

相同模型、相同折上比较“仅几何”与“几何+朝向+玻璃”。结果是预测信息增量，不是真实建筑中的因果节能效应。

## 5. 最小可完成的方法

| 层次 | 方法 | 用途 |
|---|---|---|
| 无特征基准 | 训练均值 | 判断模型是否优于简单参照 |
| 线性 | OLS、Ridge | 可解释关系、共线性与正则化 |
| 灵活非线性 | Random Forest | 检验非线性/交互是否有用 |
| 可选 | 少量结点的样条或加性回归 | 比较较低复杂度的非线性 |

主任务Y1，Y2作为预先声明的次要响应，分别报告；不能把相关Y2当成完全独立外部复现。最低版本做到均值、OLS、Ridge、RF即可，不堆LSTM、Transformer或不必要的聚类。

缩放、编码与调参放入训练流程，每折只在训练部分拟合。Ridge与RF使用小规模参数网格，不以大规模搜参代替研究问题。

## 6. 验证协议与边界

### 6.1 留出三种几何作最终测试

候选协议用固定种子5001，从12组留3组，共192条；剩9组576条作开发。组ID只从输入几何生成，不按响应或模型成绩挑容易测试组，当前候选ID为0、10、11，见JSON。

开发阶段检查该划分的输入范围，说明测试几何属于插值、边缘还是外推情境。部分未见形状可能在训练范围外，不能将所有困难都归因于分组一个因素；不能看测试成绩后换种子。

开发数据做3折GroupKFold，几何不跨折；随机行3折使用相同开发行和候选模型，作为另一种泛化目标的对照。它们共享数据，不是独立重复实验。

模型、输入和参数方案按开发结果固定后，才评价最终三组测试。当前只做了结构审计，尚未运行成绩。

### 6.2 评价与解释

- 主指标MAE，辅助RMSE、R²；两响应分开，用原数值尺度。
- 展示每个测试几何的误差，192条配置不等于192栋独立建筑。
- 稳健性看开发组/折、模型复杂度与输入组消融。
- 最终仅3个几何，不用逐行bootstrap的窄区间冒充可靠的总体泛化区间；优先展示分组误差。
- 几何变量确定性/相关关系强，单列置换可能生成不合理组合；成组消融比“一个重要性数字证明物理作用”更合适。

### 6.3 可以与不可以得出的结论

预期贡献是分清验证目标、复杂度和设计特征的增量价值。没有显著改善或某类特征几乎无增量，同样可以是合理结果。

只有12种基础形状，且为模拟设计空间。不能由此证明真实楼宇节能百分比、HVAC控制收益、跨气候泛化或因果效应。无需为了显得题目大而承诺这些。

## 7. 按老师模板组织的英文Proposal初稿

团队名称、姓名、UID与实际分工尚未提供，正式提交时按原模板补齐。此稿未编造团队信息，也不是已提交版本。

### Project title

**Can Statistical Learning Models Generalise to Unseen Building Geometries? A Study of Heating and Cooling Load Prediction**

### 1. Background and Motivation

Fast estimates of building heating and cooling requirements could support early comparisons of design alternatives. However, performance on new configurations of familiar shapes may not represent performance on unseen geometries. This project will examine that distinction using a building simulation benchmark and statistical learning methods. We will identify when model complexity and additional design variables provide useful predictive information within the available simulation setting.

### 2. Research Questions

1. How do prediction errors and model rankings differ between random-row and geometry-grouped validation?
2. Do nonlinear models retain their advantage over linear and regularised models when the geometry is unseen?
3. How much predictive information do orientation and glazing variables add beyond basic geometry?

### 3. Data

We will use the UCI Energy Efficiency dataset by Tsanas and Xifara (2012), accessed on 7 November 2026. The file contains 768 simulated configurations, eight predictors and two continuous responses: Heating Load and Cooling Load. Our structural audit identifies 12 geometry groups with 64 configurations each, using the first five geometry variables. No missing entries were found.

Each row is a simulated configuration rather than an independent measured building or a time observation. We will remove the exactly redundant surface-area variable from the OLS specification because X2 equals X3 plus twice X4. Orientation and glazing distribution will be treated as categorical where appropriate. Heating Load is the primary outcome and Cooling Load a secondary outcome; neither response will predict the other.

Source: https://archive.ics.uci.edu/dataset/242/energy+efficiency

### 4. Proposed Methods

We will compare a training-mean benchmark, OLS, Ridge regression and Random Forest. OLS and Ridge provide linear baselines and a regularisation comparison; Random Forest tests whether nonlinearities and interactions improve prediction. A low-complexity spline or additive model will be an optional extension.

Geometry-only and full-design inputs will be compared using the same folds. Preprocessing and tuning will be fitted within training folds. The focus is a small, justified model ladder rather than a large algorithm search.

### 5. Validation and Evaluation

Three geometry groups will form a final test set, leaving nine groups for development: 192 test and 576 development configurations. The candidate split uses predictor-only group identities and a fixed seed.

We will use three-fold grouped cross-validation for tuning and compare it with random-row three-fold validation on the same development rows. These protocols represent different generalisation settings, so neither is inherently invalid. Model and feature decisions will be frozen before final evaluation.

MAE is the primary metric; RMSE and R-squared are complementary. Errors will be reported by held-out geometry, including whether these geometries require interpolation or extrapolation. Robustness checks will use development folds and feature-group ablations. With only three final test groups, we will avoid precision claims that treat all configurations as independent buildings.

### 6. Expected Contribution

The project will compare validation targets, model complexity and the incremental information in design features. It could help users of simulation surrogates distinguish good performance within familiar geometries from evidence of transfer to new ones. We do not assume that flexible models or extra predictors necessarily improve results.

Limitations include few geometry groups and simulated outcomes. We will not claim causal design effects, actual energy savings or transfer to other climates.

### 7. Timeline

| Stage | Dates | Milestone |
|---|---|---|
| 1 | 7–11 November 2026 | Data audit, protocol and development-set EDA |
| 2 | 12–18 November 2026 | Baselines, main models and cross-validation |
| 3 | 19–24 November 2026 | Ablations, interpretation, robustness; freeze analysis |
| 4 | 25–28 November 2026 | Up to 10 slides, 8-minute recording, code appendix and submission |

The stored template lists **29 November 2026, 11:59 PM** as the final deadline. We plan to finish by 28 November to keep a one-day buffer. Later course announcements take precedence over the stored template.

### References

1. Tsanas, A., & Xifara, A. (2012). Energy Efficiency [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C51307. Accessed 7 November 2026.
2. SDSC5001 materials on linear regression, model selection and regularisation, moving beyond linearity and tree-based methods.

## 8. 八分钟展示的预留结构

可按八页组织：①问题与价值；②数据/12种几何；③验证目标及分组图；④模型梯度；⑤主要性能；⑥特征组消融；⑦分组误差与解释；⑧结论和边界。

第⑤–⑦页等实验完成再填，不预写“99%准确率”“节能20%”。当前阶段先确定问题、数据与评价协议。
