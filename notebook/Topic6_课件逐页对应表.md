# SDSC5001 Topic 6：课件逐页对应表

> 来源：`SDSC5001_Topic 6 Classification.pptx`，共 47 页。页码与导出 PDF 页序一致。

[详细笔记](5001_Topic6_详细笔记.md) · [图片内嵌版](5001_Topic6_详细笔记.html) · [公式速查](Topic6_公式与易错点速查.md) · [完整 PDF](../output/pdf/SDSC5001_Topic6_slides.pdf)

标题按原页保留。点击课件页码查看 PDF，点击节号查看对应讲解；一节可覆盖相邻多页。

| 课件页 | 原页标题 | 学习内容 | 笔记位置 |
|---|---|---|---|
| [1](../output/pdf/SDSC5001_Topic6_slides.pdf#page=1) | Topic 6. Classification | 课程标题；本讲从回归转向离散类别预测 | [第 1 节](5001_Topic6_详细笔记.md#section-1) |
| [2](../output/pdf/SDSC5001_Topic6_slides.pdf#page=2) | Examples of Classification | 交易欺诈、DNA/疾病、糖尿病类型等分类应用 | [第 1 节](5001_Topic6_详细笔记.md#section-1) |
| [3](../output/pdf/SDSC5001_Topic6_slides.pdf#page=3) | General Setup | 分类器映射与总体错分风险，区别训练/测试误差 | [第 1 节](5001_Topic6_详细笔记.md#section-1) |
| [4](../output/pdf/SDSC5001_Topic6_slides.pdf#page=4) | Some Concepts | 每类分数、argmax 及实际多类边界的最高分条件 | [第 1 节](5001_Topic6_详细笔记.md#section-1) |
| [5](../output/pdf/SDSC5001_Topic6_slides.pdf#page=5) | A Special Case: Binary Classification | 二分类 0/1 与 ±1 编码、阈值与边界 | [第 1 节](5001_Topic6_详细笔记.md#section-1) |
| [6](../output/pdf/SDSC5001_Topic6_slides.pdf#page=6) | Linear Regression for Classification | 三类响应直接编码成 1、2、3 做回归的问题 | [第 2 节](5001_Topic6_详细笔记.md#section-2) |
| [7](../output/pdf/SDSC5001_Topic6_slides.pdf#page=7) | Issues about Coding | 类别重编号改变拟合；二类哑变量的出发点 | [第 2 节](5001_Topic6_详细笔记.md#section-2) |
| [8](../output/pdf/SDSC5001_Topic6_slides.pdf#page=8) | Coding with Dummy Variables | one-hot 响应矩阵与逐列最小二乘 | [第 2 节](5001_Topic6_详细笔记.md#section-2) |
| [9](../output/pdf/SDSC5001_Topic6_slides.pdf#page=9) | Further Issues | 概率越界、Masking 遮蔽与原图 | [第 2 节](5001_Topic6_详细笔记.md#section-2) |
| [10](../output/pdf/SDSC5001_Topic6_slides.pdf#page=10) | Bayes Rule | 最大后验概率 Bayes 规则；补充条件风险证明 | [第 3 节](5001_Topic6_详细笔记.md#section-3) |
| [11](../output/pdf/SDSC5001_Topic6_slides.pdf#page=11) | Discriminant Analysis | 先验、类内密度与 Bayes 公式，比较 πk fk | [第 3 节](5001_Topic6_详细笔记.md#section-3) |
| [12](../output/pdf/SDSC5001_Topic6_slides.pdf#page=12) | Linear Discriminant Analysis (LDA) | 共享协方差高斯；密度维数常数与 log-odds 展开 | [第 4 节](5001_Topic6_详细笔记.md#section-4) · [作业 2(a) 对照](5001_Topic6_详细笔记.md#lda-qda-question-2a) |
| [13](../output/pdf/SDSC5001_Topic6_slides.pdf#page=13) | LDA (Cont.) | 样本先验、类均值与合并类内协方差 | [第 4 节](5001_Topic6_详细笔记.md#section-4) · [作业 2(a) 对照](5001_Topic6_详细笔记.md#lda-qda-question-2a) |
| [14](../output/pdf/SDSC5001_Topic6_slides.pdf#page=14) | LDA (Cont.) | LDA 判别函数与线性决策面 | [第 4 节](5001_Topic6_详细笔记.md#section-4) |
| [15](../output/pdf/SDSC5001_Topic6_slides.pdf#page=15) | An Illustrative Example | 三类等协方差正态图；总体 Bayes 与样本 LDA | [第 4 节](5001_Topic6_详细笔记.md#section-4) |
| [16](../output/pdf/SDSC5001_Topic6_slides.pdf#page=16) | Quadratic Discriminant Analysis (QDA) | QDA 判别函数、每类协方差与原式连续负号核对 | [第 5 节](5001_Topic6_详细笔记.md#section-5) · [作业 2(a) 对照](5001_Topic6_详细笔记.md#lda-qda-question-2a) |
| [17](../output/pdf/SDSC5001_Topic6_slides.pdf#page=17) | LDA and QDA | LDA/QDA/Bayes 比较图与偏差–方差权衡 | [第 5 节](5001_Topic6_详细笔记.md#section-5) · [作业 2(a) 对照](5001_Topic6_详细笔记.md#lda-qda-question-2a) |
| [18](../output/pdf/SDSC5001_Topic6_slides.pdf#page=18) | Naïve Bayes | 高维联合密度估计的困难与 NB 动机 | [第 6 节](5001_Topic6_详细笔记.md#section-6) |
| [19](../output/pdf/SDSC5001_Topic6_slides.pdf#page=19) | Naïve Bayes | 给定类别后的独立、单变量密度乘积与分布选择 | [第 6 节](5001_Topic6_详细笔记.md#section-6) |
| [20](../output/pdf/SDSC5001_Topic6_slides.pdf#page=20) | Performance of Naïve Bayes | 降低方差的代价、分类效果与概率校准区别 | [第 6 节](5001_Topic6_详细笔记.md#section-6) |
| [21](../output/pdf/SDSC5001_Topic6_slides.pdf#page=21) | Binary Logistic Regression | 二分类 Logistic、合法概率与严格开区间值域 | [第 7 节](5001_Topic6_详细笔记.md#section-7) |
| [22](../output/pdf/SDSC5001_Topic6_slides.pdf#page=22) | Logit and Odds Ratio | Logit、odds、OR 的系数解释及概率比区别 | [第 7 节](5001_Topic6_详细笔记.md#section-7) |
| [23](../output/pdf/SDSC5001_Topic6_slides.pdf#page=23) | Logit Function | Logit 与 Logistic 逆变换原图 | [第 7 节](5001_Topic6_详细笔记.md#section-7) |
| [24](../output/pdf/SDSC5001_Topic6_slides.pdf#page=24) | Comparison b/w Linear and Logistic Regression | 线性概率与 Logistic 拟合 Default 的图形比较 | [第 7 节](5001_Topic6_详细笔记.md#section-7) |
| [25](../output/pdf/SDSC5001_Topic6_slides.pdf#page=25) | Fitting Binary Logistic Regression | Bernoulli 似然、对数似然与交叉熵 | [第 8 节](5001_Topic6_详细笔记.md#section-8) |
| [26](../output/pdf/SDSC5001_Topic6_slides.pdf#page=26) | MLE for Logistic Regression | 梯度、Hessian、凹性及有限 MLE 存在条件 | [第 8 节](5001_Topic6_详细笔记.md#section-8) |
| [27](../output/pdf/SDSC5001_Topic6_slides.pdf#page=27) | Iteratively Reweighted Least Square (IRLS) | Newton 更新、工作响应、权重与 IRLS | [第 8 节](5001_Topic6_详细笔记.md#section-8) |
| [28](../output/pdf/SDSC5001_Topic6_slides.pdf#page=28) | Multiclass Logistic Regression | 基准类别 log-odds；补充 softmax 与可识别性 | [第 9 节](5001_Topic6_详细笔记.md#section-9) |
| [29](../output/pdf/SDSC5001_Topic6_slides.pdf#page=29) | Probit Model | Probit 的标准正态 CDF 链接及与 Logistic 比较 | [第 9 节](5001_Topic6_详细笔记.md#section-9) |
| [30](../output/pdf/SDSC5001_Topic6_slides.pdf#page=30) | Logistic Regression and LDA | LDA 线性 log-odds、截距公式核对、生成/判别区别 | [第 10 节](5001_Topic6_详细笔记.md#section-10) |
| [31](../output/pdf/SDSC5001_Topic6_slides.pdf#page=31) | Separating Hyperplane | 线性分离方法的整体几何出发点 | [第 11 节](5001_Topic6_详细笔记.md#section-11) |
| [32](../output/pdf/SDSC5001_Topic6_slides.pdf#page=32) | Geometric View of Classification | p 维特征空间中的 n 个训练点与标签 | [第 11 节](5001_Topic6_详细笔记.md#section-11) |
| [33](../output/pdf/SDSC5001_Topic6_slides.pdf#page=33) | Hyperplane | p−1 维仿射超平面及二维直线示例 | [第 11 节](5001_Topic6_详细笔记.md#section-11) |
| [34](../output/pdf/SDSC5001_Topic6_slides.pdf#page=34) | Review of Vector Algebra | 单位法向量、点面带符号距离推导 | [第 11 节](5001_Topic6_详细笔记.md#section-11) |
| [35](../output/pdf/SDSC5001_Topic6_slides.pdf#page=35) | Geometry of A Hyperplane | 法向量与垂直投影几何原图 | [第 11 节](5001_Topic6_详细笔记.md#section-11) |
| [36](../output/pdf/SDSC5001_Topic6_slides.pdf#page=36) | Separating Hyperplane | ±1 标签、完全分离与 yi f(xi)>0 | [第 11 节](5001_Topic6_详细笔记.md#section-11) |
| [37](../output/pdf/SDSC5001_Topic6_slides.pdf#page=37) | Prediction | 新点分类；区分分数、距离、真实标签和信心 | [第 11 节](5001_Topic6_详细笔记.md#section-11) |
| [38](../output/pdf/SDSC5001_Topic6_slides.pdf#page=38) | How to Do It Right? | 可分超平面不唯一，需要选择准则 | [第 11 节](5001_Topic6_详细笔记.md#section-11) |
| [39](../output/pdf/SDSC5001_Topic6_slides.pdf#page=39) | Margin | 最小点面距离定义单侧间隔 | [第 12 节](5001_Topic6_详细笔记.md#section-12) |
| [40](../output/pdf/SDSC5001_Topic6_slides.pdf#page=40) | Maximal Margin Classifier | 最大间隔与最宽分隔带中线原图 | [第 12 节](5001_Topic6_详细笔记.md#section-12) |
| [41](../output/pdf/SDSC5001_Topic6_slides.pdf#page=41) | Optimal Separating Hyperplane | 几何间隔优化；C 表示距离，本文统一写 M | [第 12 节](5001_Topic6_详细笔记.md#section-12) |
| [42](../output/pdf/SDSC5001_Topic6_slides.pdf#page=42) | Estimation | 尺度规范化与硬间隔原问题；拉格朗日转第13节 | [第 12 节](5001_Topic6_详细笔记.md#section-12) |
| [43](../output/pdf/SDSC5001_Topic6_slides.pdf#page=43) | Estimation (Cont.) | 驻点条件、对偶 QP、补充 KKT/支持向量 | [第 13 节](5001_Topic6_详细笔记.md#section-13) |
| [44](../output/pdf/SDSC5001_Topic6_slides.pdf#page=44) | Linear and Quadratic Programming | LP/QP 一般形式，凸二次目标与线性约束 | [第 13 节](5001_Topic6_详细笔记.md#section-13) |
| [45](../output/pdf/SDSC5001_Topic6_slides.pdf#page=45) | Evaluating Classification Model | 测试错分率、ROC 坐标、连续分数与阈值 | [第 14 节](5001_Topic6_详细笔记.md#section-14) |
| [46](../output/pdf/SDSC5001_Topic6_slides.pdf#page=46) | Other Performance Measures | Sensitivity、Specificity、Precision、F1、AUC、Youden | [第 14 节](5001_Topic6_详细笔记.md#section-14) |
| [47](../output/pdf/SDSC5001_Topic6_slides.pdf#page=47) | Example | 原表计数完整重建，复算指标并分析不平衡 | [第 15 节](5001_Topic6_详细笔记.md#section-15) |

## 综合与补充材料

- [第 0 节](5001_Topic6_详细笔记.md#section-0)：符号与章节主线。
- [第 16 节](5001_Topic6_详细笔记.md#section-16)：方法对比、验证与阈值选择流程。
- [第 17 节](5001_Topic6_详细笔记.md#section-17)：16 道补充自测与答案。
- [第 18 节](5001_Topic6_详细笔记.md#section-18)：原文公式与措辞核对。
- [第 19 节](5001_Topic6_详细笔记.md#section-19)：概念的意义、应用选择与违约/筛查/邮件三个场景；各主题末尾也已补充用途说明，均属学习补充。

Bayes 风险证明、成本阈值、手算 Newton 更新、KKT、一维对偶算例及原表新增指标均作为补充说明，不冒充课件原题或原始实验。
