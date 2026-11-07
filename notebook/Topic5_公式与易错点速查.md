# SDSC5001 Topic 5：公式与易错点速查

> 范围：本地 Topic 5 课件第 1～36 页。配套[详细笔记](5001_Topic5_详细笔记.md)、[逐页对应表](Topic5_课件逐页对应表.md)和[课件 PDF](../output/pdf/SDSC5001_Topic5_slides.pdf)。本页是学习整理，补充推导与条件不冒充课件原句。

## 1. 先固定符号与损失缩放

$n$ 为样本数，$p$ 为候选斜率数，$d$ 为选中的斜率数；$X$ 为设计矩阵，$y$ 为响应。以下无截距公式均假定数据已中心化，或题目明确设为无截距问题。截距一般不惩罚。

$$RSS=\lVert y-X\beta\rVert_2^2,\qquad (u)_+=\max(u,0).$$

**课件第 12～21 页主要使用 $RSS+\lambda P(\beta)$。** 第 24～25 页有 $\tfrac12RSS$，不要机械沿用前面的阈值。

下表的正交特例设 $X=I_n$、$p=n$；它是独立的无截距代数例子。

| 目标函数 | Ridge 每维解 | Lasso 每维解 |
|---|---|---|
| $RSS+\lambda P(\beta)$ | $y_j/(1+\lambda)$ | $\operatorname{sign}(y_j)(\lvert y_j\rvert-\lambda/2)_+$ |
| $\tfrac12RSS+\lambda P(\beta)$ | $y_j/(1+2\lambda)$ | $\operatorname{sign}(y_j)(\lvert y_j\rvert-\lambda)_+$ |
| $\tfrac1{2n}RSS+\alpha P(\beta)$ | $y_j/(1+2n\alpha)$ | $\operatorname{sign}(y_j)(\lvert y_j\rvert-n\alpha)_+$ |

Ridge 取 $P=\lVert\beta\rVert_2^2$，Lasso 取 $P=\lVert\beta\rVert_1$。上表不把 Ridge 的平方惩罚额外写成 $\tfrac12\lVert\beta\rVert_2^2$；若题目又改了惩罚系数，须重新换算。

软件中，scikit-learn 的 `Ridge(alpha)` 对应 $RSS+\alpha\lVert\beta\rVert_2^2$；`Lasso(alpha)` 对应 $RSS/(2n)+\alpha\lVert\beta\rVert_1$。与课件 Lasso 比较时，在同一训练数据尺度上有 $\lambda_{slide}=2n\alpha$。

## 2. 模型选择准则〔第 5～10 页〕

课件定义如下，$\hat\sigma^2$ 为模型间固定使用的噪声方差估计：

$$C_p=\frac{RSS_d+2d\hat\sigma^2}{n},$$

$$AIC_{slide}=\frac{RSS_d+2d\hat\sigma^2}{n\hat\sigma^2},\qquad BIC_{slide}=\frac{RSS_d+d\log n\,\hat\sigma^2}{n\hat\sigma^2}.$$

三者都取较小值。同一 $\hat\sigma^2>0$ 下，$C_p$ 和课件 AIC 排序一致。$n>e^2$ 时 BIC 的每参数惩罚更强，通常偏向简单模型。

若每个模型都包含同一截距，把 $d$ 改为 $d+1$ 只改变共同项，不影响上述排序；在参数计数、方差估计与推导中仍应正确计算截距自由度。

$$R^2_{adj}=1-\frac{RSS/(n-d-1)}{TSS/(n-1)},\qquad n>d+1.$$

调整后的 $R^2$ 取较大值。训练 RSS 只会随嵌套 OLS 模型规模增加而下降，普通 $R^2$ 只会增加，二者不能直接决定最佳规模。

| 搜索方法 | 候选范围/计算量 | 需要记住的限制 |
|---|---|---|
| 最优子集 | 全部 $2^p$ 个子集；每规模先选最小 RSS | 计算昂贵，搜索最优不等于测试误差最优 |
| 前向选择 | 从空模型逐个加入；约 $1+p(p+1)/2$ 次候选拟合 | 贪心路径，不能保证全局最佳子集 |
| 后向选择 | 从完整模型逐个删除 | 起始完整 OLS 需要可拟合的秩与样本条件 |
| 混合逐步选择 | 加入与删除交替 | 第 9 页称 Stagewise；其他书可能用该词指小步长更新 |

**与其他资料对照时：** 一般似然定义为 $AIC=-2\log L+2k$、$BIC=-2\log L+k\log n$；重新估计高斯误差方差时会出现 $n\log(RSS/n)$。不要把不同缩放的公式直接混算数值。

补充原理：事先固定的正确线性模型，有效维数为 $k$，固定 $X$ 下训练 RSS 的期望为 $(n-k)\sigma^2$，同一设计点的独立新响应误差期望为 $(n+k)\sigma^2$，差值为 $2k\sigma^2$。先做数据驱动的模型搜索还会有选择效应。

## 3. Ridge〔第 11～15 页〕

$$\hat\beta^R=\arg\min_\beta\{RSS+\lambda\lVert\beta\rVert_2^2\}=(X^TX+\lambda I)^{-1}X^Ty.$$

- $\lambda>0$ 时矩阵正定，斜率解唯一，即使 $p>n$ 也成立。
- $\lambda\to\infty$ 时斜率趋向 0；若有不惩罚的截距，预测趋向 $\bar y$。
- 约束形式为 $\min RSS$，使 $\lVert\beta\rVert_2^2\le s$。$s$ 越紧通常对应 $\lambda$ 越大，两者数值不相等。
- 通常不产生一批恰为 0 的系数；相关设计下，单个原始系数的绝对值不必随 $\lambda$ 单调下降。

补充 SVD：若紧 SVD 为 $X=UDV^T$，

$$\hat y^R=U\operatorname{diag}\left(\frac{d_j^2}{d_j^2+\lambda}\right)U^Ty,\qquad df(\lambda)=\sum_j\frac{d_j^2}{d_j^2+\lambda}.$$

小奇异值方向收缩更强。此处 $df$ 为斜率部分的有效自由度；若计入独立估计的截距，总有效自由度加 1。

$$MSE(x_0)=\sigma^2+Bias^2[\hat f(x_0)]+Var[\hat f(x_0)].$$

第 15 页黑线标 Bias，但平方误差分解中应理解为平方偏差贡献。正则化通常降低方差并引入偏差；需要调参，不能断言任何正 $\lambda$ 都优于 OLS。

## 4. Lasso〔第 16～21 页〕

$$\hat\beta^L=\arg\min_\beta\{RSS+\lambda\lVert\beta\rVert_1\},\qquad \lVert\beta\rVert_1=\sum_j\lvert\beta_j\rvert.$$

约束形式为 $\lVert\beta\rVert_1\le s$。二维菱形的角落对应零坐标，解释了稀疏倾向；一般设计没有 Ridge 式统一矩阵闭式解。

补充 KKT，设 $r=y-X\hat\beta$：

$$2x_j^Tr=\lambda\operatorname{sign}(\hat\beta_j)\quad(\hat\beta_j\ne0),$$

$$\lvert2x_j^Tr\rvert\le\lambda\quad(\hat\beta_j=0).$$

在 $X=I$ 下得到软阈值 $\operatorname{sign}(y_j)(\lvert y_j\rvert-\lambda/2)_+$。保留下来的大系数也受收缩，不能将软阈值理解为“只删小系数”。

第 19 页左图是 Ridge，横轴为 $df(\lambda)$；右图是 Lasso，横轴为收缩预算比例 $s$。两图向右都表示放宽正则化、增加复杂度，不能读成向右 $\lambda$ 增大。

## 5. 扩展惩罚〔第 22～25 页〕

| 方法 | 惩罚/构造 | 结果与条件 |
|---|---|---|
| Bridge | $\sum_j\lvert\beta_j\rvert^r$ | $r=1$ 是 Lasso，$r=2$ 是 Ridge；$0<r<1$ 非凸 |
| $L_0$ | 非零系数个数 | 组合选择；$X=I$、$RSS+\lambda\lVert\beta\rVert_0$ 给硬阈值 $\sqrt\lambda$ |
| Nonnegative Garrote | $\beta_j=c_j\tilde\beta_j$，$c_j\ge0$，惩罚 $\sum c_j$ | 保留初始符号；初始为 0 的坐标无法重新激活 |
| Group Lasso | $\sum_{g=1}^G\lVert\beta_g\rVert_2$ | 整组选择，组内不自动继续稀疏 |
| Elastic Net | $\lambda_1\lVert\beta\rVert_1+\lambda_2\lVert\beta\rVert_2^2$ | 同时稀疏与稳定相关变量权重 |
| 课件 Fused 版本 | $\sum_{j=2}^p\lvert\beta_j-\beta_{j-1}\rvert$ | 相邻系数融合，差分为零不等于系数为零 |

第 24 页 Garrote 的目标为

$$\min_{c\ge0}\frac12\left\lVert y-\sum_jc_j\tilde\beta_jx_j\right\rVert_2^2+\lambda\sum_jc_j.$$

在 $X=I$、$\tilde\beta=y$ 下，一致的解为

$$\hat\beta_j^{NG}=\left(1-\frac{\lambda}{y_j^2}\right)_+y_j\quad(y_j\ne0).$$

若 $y_j=0$，系数直接为 0，不代入分母。课件解中分母有 $2\tilde\beta_j^2$，对应的是没有 $1/2$ 的 RSS 目标；一般相关设计没有该逐坐标公式。

标准无穷范数为 $\lVert\beta\rVert_\infty=\max_j\lvert\beta_j\rvert$。第 22 页漏了绝对值。有限 $r$ 的 Bridge 和式不是通常 $L_r$ 范数本身，要取 $1/r$ 次方才是范数形式（$r\ge1$）。

## 6. PCA 与 PP〔第 26～30 页〕

PCA 寻找单位长度、投影方差最大的方向，后续成分与此前成分不相关：

$$\max_{a^Ta=1}a^T\Sigma a.$$

令 $\ell_1\ge\cdots\ge\ell_p\ge0$ 为协方差特征值，$e_j$ 为对应正交单位特征向量，则

$$U_j=e_j^TX,\quad Var(U_j)=\ell_j,\quad Cov(U_j,U_l)=0\ (j\ne l).$$

$$PVE_j=\frac{\ell_j}{\sum_l\ell_l},\qquad CPVE_K=\frac{\sum_{j=1}^K\ell_j}{\sum_{j=1}^p\ell_j}.$$

主成分不相关通常不能推出独立；保留 $X$ 的 95% 方差也不能推出保留 95% 的预测能力。是否先标准化取决于变量量纲与问题含义。

PP 在本讲寻找投影的非正态结构：$\max_{a\in\mathcal A}I(a^TX)$。方向尺度要固定，或指标本身应尺度不变。课件给出

$$I_1(z)=\frac{\lvert\kappa_m(z)\rvert}{\kappa_2(z)^{m/2}},\qquad I_2(z)=\kappa_3(z)^2+\frac14\kappa_4(z)^2.$$

第二式需结合标准化理解。$\kappa_4=E[(z-Ez)^4]-3Var(z)^2$，不是原始四阶矩。高斯的三阶及更高阶累积量为零；对称双峰的三阶量也可能为零。

## 7. PCR 与 PLS〔第 31～36 页〕

PCR 先从 $X$ 做 PCA 得 $Q_K$，再令 $Z=XQ_K$ 并回归。对于中心化、非零正交得分：

$$\hat\gamma_j=\frac{z_j^Ty_c}{z_j^Tz_j},\qquad \hat\beta^{PCR}=Q_K\hat\gamma.$$

第 31 页称 $\hat\gamma_j$ 为相关系数；这里应为回归斜率。两者关系是 $\hat\gamma_j=r(y,z_j)s_y/s_{z_j}$。

PLS 构造方向时已使用 $y$。按课件的第一步：

$$\gamma_j=\frac{x_j^Ty}{x_j^Tx_j},\qquad z_1=X\gamma,\qquad b_1=\frac{z_1^Ty}{z_1^Tz_1}.$$

然后残差化响应与预测列，继续构造方向。各列同尺度时，第一权重方向与 $X^Ty$ 成比例；不同 PLS 算法的归一化可能不同。

| 方法 | 构造方向时使用什么 | 对奇异方向的处理 | 主要风险 |
|---|---|---|---|
| PCR | 只用 $X$ | 保留前 $K$ 个，其余截断 | 响应信号可能在小方差方向 |
| Ridge | 用 $X,y$ 估计系数 | 乘 $d_j^2/(d_j^2+\lambda)$，连续收缩 | 惩罚过大产生较大偏差 |
| PLS | 同时用 $X,y$ 构造得分 | 按响应关联构造较少方向 | 成分过多也会拟合噪声 |

PCR/PLS 保留足够方向并张成 $X$ 列空间时，可恢复 OLS 投影拟合；不能在秩不足时声称一定恢复唯一 OLS 系数。二者通常也不筛掉原始变量，原尺度系数往往仍稠密。

## 8. 做题与实践时的核对顺序

1. 写出目标函数，圈出 RSS 前的系数，确认是否惩罚截距。
2. 判断题目是一般设计、正交设计，还是 $X=I$；别把特殊公式套到相关设计。
3. 判断是在比较子集规模、收缩强度，还是成分数。
4. 看清横轴：$\lambda$、约束预算 $s$、$df$、成分数的方向含义不同。
5. 分清“原始变量少”“得分维数少”“有效自由度低”“系数差分少”。
6. 所有标准化、筛选、PCA/PLS 都在 CV 的训练折中学习；保留独立测试集做最终评价。

建议能独立推导 Ridge 正规方程、Lasso 软阈值和 PCA 特征向量解，并解释 PCR/PLS 差异。这是基于本讲内容的复习建议，不代表教师公布的考试重点。
