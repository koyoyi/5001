# SDSC5001 Topic 5：课件逐页对应表

> 来源：`SDSC5001_Topic 5 Model selection and regularization.pptx`，共 36 页。页码使用课件右下角页码，与导出 PDF 页序一致。

[详细笔记](5001_Topic5_详细笔记.md) · [公式速查](Topic5_公式与易错点速查.md) · [完整 PDF](../output/pdf/SDSC5001_Topic5_slides.pdf)

课件按页保留标题，详细笔记将连续页合并为主题节。点击页码打开对应 PDF 页，点击笔记节号跳到讲解。原课件的第 6、30 页只有标题，笔记已将相关扩展明确标成补充。

| 课件页 | 原页标题 | 对应学习内容 | 详细笔记 |
|---|---|---|---|
| [1](../output/pdf/SDSC5001_Topic5_slides.pdf#page=1) | Topic 5. Model Selection and Regularization | Topic 5 标题与课程名称；确定本讲范围 | [第 1 节](5001_Topic5_详细笔记.md#section-1) |
| [2](../output/pdf/SDSC5001_Topic5_slides.pdf#page=2) | Sparse Regression | 稀疏模型、真实支持集与候选冗余变量 | [第 1 节](5001_Topic5_详细笔记.md#section-1) |
| [3](../output/pdf/SDSC5001_Topic5_slides.pdf#page=3) | Why Do We Care? | 共线性、过拟合、解释成本；区分预测与变量识别 | [第 1 节](5001_Topic5_详细笔记.md#section-1) |
| [4](../output/pdf/SDSC5001_Topic5_slides.pdf#page=4) | Popular Techniques | 方法路线：子集、逐步、收缩、降维 | [第 1 节](5001_Topic5_详细笔记.md#section-1) |
| [5](../output/pdf/SDSC5001_Topic5_slides.pdf#page=5) | Best Subset Selection | 固定规模内找最小 RSS，再跨规模用验证/准则选择 | [第 2 节](5001_Topic5_详细笔记.md#section-2) |
| [6](../output/pdf/SDSC5001_Topic5_slides.pdf#page=6) | Adjustment to Training Error | 原页只有标题；训练误差乐观性及修正推导为补充 | [第 2 节](5001_Topic5_详细笔记.md#section-2) |
| [7](../output/pdf/SDSC5001_Topic5_slides.pdf#page=7) | Model Selection Criteria | 按原页保留 Cp/AIC/BIC 的缩放公式，区分调整 R² | [第 2 节](5001_Topic5_详细笔记.md#section-2) |
| [8](../output/pdf/SDSC5001_Topic5_slides.pdf#page=8) | An Illustrative Example | 读 Cp、BIC 的最低点与调整 R² 的最高点；保留原图 | [第 2 节](5001_Topic5_详细笔记.md#section-2) |
| [9](../output/pdf/SDSC5001_Topic5_slides.pdf#page=9) | Forward/Backward Selection | 前向、后向与混合加入/删除；核对 Stagewise 用词 | [第 3 节](5001_Topic5_详细笔记.md#section-3) |
| [10](../output/pdf/SDSC5001_Topic5_slides.pdf#page=10) | Some Remarks | 计算更省但无全局最优保证；前向/后向起点差异 | [第 3 节](5001_Topic5_详细笔记.md#section-3) |
| [11](../output/pdf/SDSC5001_Topic5_slides.pdf#page=11) | Shrinkage Methods | 惩罚目标、截距、中心化与标准化 | [第 4 节](5001_Topic5_详细笔记.md#section-4) |
| [12](../output/pdf/SDSC5001_Topic5_slides.pdf#page=12) | Ridge Regression | Ridge 的平方 L2 惩罚，λ 与偏差/方差权衡 | [第 5 节](5001_Topic5_详细笔记.md#section-5) |
| [13](../output/pdf/SDSC5001_Topic5_slides.pdf#page=13) | Ridge Regression (Cont.) | Ridge 闭式解及约束形式；补充 SVD 与有效自由度 | [第 5 节](5001_Topic5_详细笔记.md#section-5) |
| [14](../output/pdf/SDSC5001_Topic5_slides.pdf#page=14) | Example | Ridge 系数路径；相关设计下单个系数不必单调缩小 | [第 5 节](5001_Topic5_详细笔记.md#section-5) |
| [15](../output/pdf/SDSC5001_Topic5_slides.pdf#page=15) | An Example (Cont.) | 平方偏差、方差与测试 MSE；核对图中的 Bias 标签 | [第 5 节](5001_Topic5_详细笔记.md#section-5) |
| [16](../output/pdf/SDSC5001_Topic5_slides.pdf#page=16) | LASSO | Lasso 的 L1 惩罚、约束与一般优化方法 | [第 6 节](5001_Topic5_详细笔记.md#section-6) |
| [17](../output/pdf/SDSC5001_Topic5_slides.pdf#page=17) | Sparse Solution | 菱形与圆形约束；用几何及补充 KKT 理解稀疏 | [第 6 节](5001_Topic5_详细笔记.md#section-6) |
| [18](../output/pdf/SDSC5001_Topic5_slides.pdf#page=18) | Example: Prostate Cancer | 前列腺癌示例：8 个输入列名与响应 lpsa | [第 6 节](5001_Topic5_详细笔记.md#section-6) |
| [19](../output/pdf/SDSC5001_Topic5_slides.pdf#page=19) | Example: Prostate Cancer (Cont.) | Ridge/Lasso 系数路径；横轴分别为 df 与收缩预算比例 | [第 6 节](5001_Topic5_详细笔记.md#section-6) |
| [20](../output/pdf/SDSC5001_Topic5_slides.pdf#page=20) | Ridge vs. Lasso | Ridge 与 Lasso 的稀疏性、稳定性和预测比较 | [第 6 节](5001_Topic5_详细笔记.md#section-6) |
| [21](../output/pdf/SDSC5001_Topic5_slides.pdf#page=21) | A Special Simple Case | 无截距 X=I 特例：比例收缩与软阈值 λ/2 | [第 7 节](5001_Topic5_详细笔记.md#section-7) |
| [22](../output/pdf/SDSC5001_Topic5_slides.pdf#page=22) | Bridge Estimators | Bridge 的 r、L0 计数与无穷范数；说明原式漏绝对值 | [第 8 节](5001_Topic5_详细笔记.md#section-8) |
| [23](../output/pdf/SDSC5001_Topic5_slides.pdf#page=23) | Constrained Areas of Bridge Regressions | 不同 r 的约束域；0<r<1 非凸 | [第 8 节](5001_Topic5_详细笔记.md#section-8) |
| [24](../output/pdf/SDSC5001_Topic5_slides.pdf#page=24) | Nonnegative Garrote | Garrote 非负乘子；核对目标与解的常数及正交条件 | [第 8 节](5001_Topic5_详细笔记.md#section-8) |
| [25](../output/pdf/SDSC5001_Topic5_slides.pdf#page=25) | Other Extensions | Group Lasso、Elastic Net、仅差分的 Fused 版本 | [第 9 节](5001_Topic5_详细笔记.md#section-9) |
| [26](../output/pdf/SDSC5001_Topic5_slides.pdf#page=26) | Principal Component Analysis (PCA) | PCA 最大方差、单位方向与不相关约束 | [第 10 节](5001_Topic5_详细笔记.md#section-10) |
| [27](../output/pdf/SDSC5001_Topic5_slides.pdf#page=27) | Example | 散点点云方向与投影方差的几何解释 | [第 10 节](5001_Topic5_详细笔记.md#section-10) |
| [28](../output/pdf/SDSC5001_Topic5_slides.pdf#page=28) | PCA and Eigen-decomposition | 协方差特征分解、解释方差比例与选维数 | [第 10 节](5001_Topic5_详细笔记.md#section-10) |
| [29](../output/pdf/SDSC5001_Topic5_slides.pdf#page=29) | Projection Pursuit (PP) | PP 的非正态投影、双峰反例、累积量及尺度条件 | [第 11 节](5001_Topic5_详细笔记.md#section-11) |
| [30](../output/pdf/SDSC5001_Topic5_slides.pdf#page=30) | Projection Pursuit (PP) | 原页只有标题；不假定有额外算法或图示 | [第 11 节](5001_Topic5_详细笔记.md#section-11) |
| [31](../output/pdf/SDSC5001_Topic5_slides.pdf#page=31) | Principal Component Regression (PCR) | PCR 得分回归；回归斜率与相关系数的区别 | [第 12 节](5001_Topic5_详细笔记.md#section-12) |
| [32](../output/pdf/SDSC5001_Topic5_slides.pdf#page=32) | Some Simulated Examples with PCR | 成分数改变偏差、方差和测试误差；保留模拟原图 | [第 12 节](5001_Topic5_详细笔记.md#section-12) |
| [33](../output/pdf/SDSC5001_Topic5_slides.pdf#page=33) | Some Remarks on PCR | PCR 的适用前提、维数选择及与原始变量筛选的区别 | [第 12 节](5001_Topic5_详细笔记.md#section-12) |
| [34](../output/pdf/SDSC5001_Topic5_slides.pdf#page=34) | Partial Least Squares (PLS) | PLS 单变量权重、第一得分及响应/输入残差化 | [第 13 节](5001_Topic5_详细笔记.md#section-13) |
| [35](../output/pdf/SDSC5001_Topic5_slides.pdf#page=35) | Partial Least Squares (Cont.) | 迭代提取、成分数和恢复 OLS 的秩条件 | [第 13 节](5001_Topic5_详细笔记.md#section-13) |
| [36](../output/pdf/SDSC5001_Topic5_slides.pdf#page=36) | PCR vs PLS | PCR/PLS 第一方向原图；监督与无监督方向差别 | [第 13 节](5001_Topic5_详细笔记.md#section-13) |

## 综合复习与补充材料

| 详细笔记位置 | 内容 | 来源关系 |
|---|---|---|
| [第 0 节](5001_Topic5_详细笔记.md#section-0) | 主线、符号与损失缩放 | 为统一课件记号而整理 |
| [第 14 节](5001_Topic5_详细笔记.md#section-14) | 全讲方法对照、建模流程 | 综合各页并补充实践解释 |
| [第 15 节](5001_Topic5_详细笔记.md#section-15) | sklearn Pipeline 与 CV 代码 | 补充示例，非课件数据复现 |
| [第 16 节](5001_Topic5_详细笔记.md#section-16) | 14 道自测与答案要点 | 补充自测，非课件原题 |
| [第 17 节](5001_Topic5_详细笔记.md#section-17) | 课件表述核对清单 | 标出本地原文与一致公式的差异 |

方程推导、数值算例和条件说明用于辅助理解；不将其当作教师额外讲义或正式勘误。
