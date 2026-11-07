# SDSC5001 Topic 6：公式与易错点速查

> 配套[详细笔记](5001_Topic6_详细笔记.md)、[逐页对应表](Topic6_课件逐页对应表.md)及[课件 PDF](../output/pdf/SDSC5001_Topic6_slides.pdf)。以下按本地 47 页课件整理；标出的条件与推导包含学习补充。

## 1. Bayes 规则〔第 3～11 页〕

$$p_k(x)=P(Y=k\mid X=x)=\frac{\pi_kf_k(x)}{\sum_l\pi_lf_l(x)}.$$

$$\boxed{G^*(x)=\arg\max_kp_k(x)=\arg\max_k\pi_kf_k(x).}$$

理由：预测为 $k$ 的条件 0–1 损失为 $1-p_k(x)$。最小总体错分率为 $E[1-\max_k p_k(X)]$，不必等于 0。

补充成本阈值：正确分类成本为 0，误报和漏报成本分别为 $c_{FP},c_{FN}$ 时，二分类预测正类的阈值为

$$p(x)>\frac{c_{FP}}{c_{FP}+c_{FN}}.$$

分类分数可以是任意实数；概率必须合法。多分类两类分数相等还需它们同时最高，才决定实际边界。

## 2. LDA / QDA / Naïve Bayes〔第 12～20 页〕

| 方法 | 假设 | 取最大值的判别分数 |
|---|---|---|
| LDA | $X\mid Y=k\sim N_p(\mu_k,\Sigma)$ | $\log\pi_k+x^T\Sigma^{-1}\mu_k-\tfrac12\mu_k^T\Sigma^{-1}\mu_k$ |
| QDA | $X\mid Y=k\sim N_p(\mu_k,\Sigma_k)$ | $\log\pi_k-\tfrac12\log\lvert\Sigma_k\rvert-\tfrac12(x-\mu_k)^T\Sigma_k^{-1}(x-\mu_k)$ |
| Naïve Bayes | 给定类别后，各特征独立 | $\log\pi_k+\sum_j\log f_{kj}(x_j)$ |

LDA 的共享二次项相消，边界线性；QDA 的类相关二次项一般不消失，边界通常二次。Naïve Bayes 的边界形状依赖单变量分布，不必线性。

课件参数估计：

$$\hat\pi_k=n_k/n,\qquad\hat\mu_k=\frac1{n_k}\sum_{i:y_i=k}x_i,$$

$$\hat\Sigma_{LDA}=\frac1{n-K}\sum_i(x_i-\hat\mu_{y_i})(x_i-\hat\mu_{y_i})^T,$$

$$\hat\Sigma_k^{QDA}=\frac1{n_k-1}\sum_{i:y_i=k}(x_i-\hat\mu_k)(x_i-\hat\mu_k)^T.$$

分母按课件保留无偏估计约定，不直接换成 MLE 的分母。矩阵求逆需有效秩；QDA 对每类样本量的要求通常更高。

Naïve Bayes 要求的是**类内条件独立**；不是总体独立，也不是简单的两两不相关。对数计算避免概率连乘下溢，离散零频数可用平滑处理。

## 3. Logistic 与 OR〔第 21～24 页〕

$$\eta=\beta_0+x^T\beta,\qquad p(x)=\sigma(\eta)=\frac1{1+e^{-\eta}},$$

$$\operatorname{logit}(p)=\log\frac{p}{1-p}=\eta,\qquad odds=\frac p{1-p}=e^\eta.$$

保持其他项不变，$x_j$ 增加 $\Delta$，odds ratio 为 $e^{\Delta\beta_j}$，一般不等于概率比。

$$p(x)>t\iff\eta>\log\frac t{1-t}.$$

阈值 0.5 对应 $\eta>0$；阈值 0.2 对应 $\eta>\log0.25\approx-1.3863$。有限参数的概率严格处于 $(0,1)$。

## 4. MLE / Newton / IRLS〔第 25～27 页〕

令 $\tilde x_i=(1,x_i^T)^T$，$\theta$ 含截距，$p_i=\sigma(\tilde x_i^T\theta)$：

$$\ell(\theta)=\sum_i[y_i\log p_i+(1-y_i)\log(1-p_i)]
=\sum_i[y_i\tilde x_i^T\theta-\log(1+e^{\tilde x_i^T\theta})].$$

$$g=\tilde X^T(y-p),\qquad H=-\tilde X^TW\tilde X,\quad W=diag\{p_i(1-p_i)\}.$$

$$\boxed{\theta^{new}=\theta^{old}+(\tilde X^TW\tilde X)^{-1}\tilde X^T(y-p).}$$

$$z=\tilde X\theta^{old}+W^{-1}(y-p),\qquad
\theta^{new}=(\tilde X^TW\tilde X)^{-1}\tilde X^TWz.$$

$p,W,z$ 每轮更新；实现上解线性方程更合适。对数似然凹不保证有限 MLE 存在：完全/准完全分离需要另行处理。

## 5. 多分类、Probit 与 LDA 联系〔第 28～30 页〕

以第 $K$ 类为基准，$\eta_k=\tilde x^T\theta_k$、$\eta_K=0$：

$$\log\frac{p_k}{p_K}=\eta_k,\qquad
p_k=\frac{e^{\eta_k}}{\sum_{l=1}^K e^{\eta_l}}.$$

Probit：$p(x)=\Phi(\beta_0+x^T\beta)$，其中 $\Phi$ 为标准正态 CDF。

二分类 LDA 的线性 log-odds 系数：

$$c_1=\Sigma^{-1}(\mu_1-\mu_0),$$

$$c_0=\log\frac\pi{1-\pi}-\frac12(\mu_1+\mu_0)^T\Sigma^{-1}(\mu_1-\mu_0).$$

LDA 与 Logistic 可有相同的 log-odds 形式，但前者建模类内特征分布，后者直接拟合条件类别概率。

## 6. 超平面与最大间隔〔第 31～44 页〕

$$f(x)=\beta_0+x^T\beta,\qquad L=\{x:f(x)=0\},\quad\beta\ne0.$$

$$d_{signed}=\frac{f(x)}{\lVert\beta\rVert_2},\qquad
d=\frac{\lvert f(x)\rvert}{\lVert\beta\rVert_2}.$$

二类标签取 $\pm1$，分离条件为 $y_i f(x_i)>0$。规范化到 $y_if(x_i)\ge1$ 后，硬间隔问题为

$$\boxed{\min_{\beta,\beta_0}\frac12\lVert\beta\rVert_2^2
\quad\text{s.t.}\quad y_i(\beta_0+x_i^T\beta)\ge1.}$$

单侧间隔为 $1/\lVert\beta\rVert$，分隔带宽度为 $2/\lVert\beta\rVert$。课件第 41 页的 $C$ 是几何间隔，不是软间隔软件参数。不可分数据无法直接满足硬间隔约束。

$$\beta=\sum_i\alpha_i y_i x_i,\qquad \sum_i\alpha_i y_i=0.$$

$$\boxed{\max_{\alpha_i\ge0}\sum_i\alpha_i-
\frac12\sum_{i,j}\alpha_i\alpha_jy_iy_jx_i^Tx_j
\quad\text{s.t.}\quad\sum_i\alpha_i y_i=0.}$$

补充互补松弛：$\alpha_i[y_if(x_i)-1]=0$。$\alpha_i>0$ 的点在支持平面上；可由它恢复 $\beta_0=y_i-x_i^T\beta$。

## 7. 评价指标〔第 45～47 页〕

先确定正类，再按真实/预测标签认 TP、FP、FN、TN。分母不同，意义不同。

| 指标 | 公式 | 在问什么 |
|---|---|---|
| Error | $(FP+FN)/n$ | 所有样本中分错多少 |
| Accuracy | $(TP+TN)/n$ | 所有样本中分对多少 |
| Sensitivity / Recall / TPR | $TP/(TP+FN)$ | 真实正类抓住多少 |
| Specificity / TNR | $TN/(TN+FP)$ | 真实负类排除多少 |
| FPR | $FP/(TN+FP)$ | 真实负类误报多少 |
| Precision / PPV | $TP/(TP+FP)$ | 预测正类有多少是真的 |
| F1 | $2TP/(2TP+FP+FN)$ | Precision 与 Recall 的调和平均 |
| Youden $J$ | $TPR-FPR$ | ROC 点与对角线的纵向差 |

ROC：横轴 FPR，纵轴 TPR，使用连续分数遍历阈值。AUC 衡量排序，不是固定阈值准确率，也不是概率校准。补充解释：随机一正一负，正类分数较高的概率，加上并列概率的一半。

第 47 页计数：$TN=9644,FP=23,FN=252,TP=81$。结果为 Error 2.75%、Recall 24.32%、Specificity 99.7621%、FPR 0.2379%；补充 Precision 77.88%、F1 37.07%。不能仅凭这一张表恢复完整 AUC。

## 8. 原文勘核与常见误用

- 第 12 页正态密度常数应为 $(2\pi)^{-p/2}$，原页维数指数漏了 $p$。
- 第 16 页 QDA 距离项前多出一个负号；该项应为负的平方距离。
- 第 30 页 $c_0$ 最后一个括号应为均值差，不能两个都写均值和。
- 多类数字编码不代表自然距离；one-hot 也不自动解决概率越界与遮蔽。
- 概率曲线非线性不妨碍 log-odds/边界线性；边界线性也不意味着用 OLS 训练。
- 原始分数、几何距离与概率置信度含义不同；系数缩放必须考虑。
- Recall 不等于 Precision，FPR 的分母是真实负类，不是预测正类。
- 阈值和超参数在验证数据上选，不能看完测试结果再调到最优后报告。

建议能独立推导 Bayes 规则、LDA 二次项相消、Logistic 梯度/Hessian、点面距离和硬间隔对偶。这是学习建议，不代表教师公布的考试重点。

## 9. 应用选型速记

以下为应用补充，完整场景见[详细笔记第 19 节](5001_Topic6_详细笔记.md#section-19)；各主题末尾也有用途说明。

| 概念/方法 | 实际意义与考虑场景 | 选择时的条件 |
|---|---|---|
| 先验与后验 | 将人群基础发生率和个体特征合并，形成风险判断 | 后验概率应适配使用人群 |
| Bayes 决策与成本阈值 | 把风险转为复核、初筛等行动 | 先明确误报/漏报成本与行动 |
| LDA | 共享协方差，以较少参数利用类间均值差 | 类内高斯与共享协方差大致合理，矩阵可处理 |
| QDA | 利用类别间方差和相关结构差异 | 各类数据足够，额外灵活性经验证有用 |
| Naïve Bayes | 将难估的高维联合分布拆成单变量分布 | 单变量模型匹配特征，关注重复证据与校准 |
| Logistic / OR | 预测二元风险，解释固定其他项时的 odds 变化 | 检查 log-odds 形式、尺度、交互与正则化需要 |
| 多分类 Logistic | 对互斥多类输出总和为 1 的概率 | 一条观测属于一个类别，多标签需另设模型 |
| Probit | 用潜在正态倾向跨过阈值解释二元选择 | 与 Logistic 的链接和系数尺度不同 |
| MLE / IRLS | 分别规定学习标准与数值求解步骤 | 处理分离、秩和收敛问题 |
| 几何距离 / 最大间隔 | 衡量跨边界所需移动，在可分面中选较宽余量 | 距离依赖尺度，硬间隔要求可分 |
| 支持向量 / 对偶 | 找到约束边界的样本，理解内积及核扩展 | 边界样本不等于类内典型样本，对偶不总更快 |
| Recall / Precision | 分别检查目标覆盖率、告警命中率 | 结合后续处理能力，避免只顾一端 |
| ROC / AUC | 在阈值未定时比较排序区分能力 | 再检查实际阈值、工作区间与概率校准 |
| Youden / F1 | 提供特定统计折中准则 | 不自动代表实际业务成本最小 |

例如疾病初筛常重视漏诊，同时检查复查数量；人工反欺诈复核需同时看处理量、Precision 和 Recall。方法名称或单个高分指标，都不足以替代任务定义与独立验证。
