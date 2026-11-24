# 第二轮在线耗时记录

每条管线固定前20个开发查询，各方法预热一轮后重复5轮；6种方法共600个正式样本。串行执行，不包含下载、模型加载、离线文档编码或生成模型。BGE使用RTX3060的DirectML，LSA使用CPU；以下是各完整管线的工程测量，不能把跨后端差异单独归因于算法。

| 管线 | 方法 | median ms | p95 ms |
| --- | --- | ---: | ---: |
| Head LSA | bm25 | 7.63 | 10.71 |
| Head LSA | semantic | 27.97 | 46.58 |
| Head LSA | rrf | 37.38 | 58.13 |
| Head LSA | weighted_rrf | 31.66 | 37.47 |
| Head LSA | logistic | 33.66 | 45.18 |
| Head LSA | random_forest | 53.89 | 71.71 |
| Full LSA | bm25 | 14.56 | 18.88 |
| Full LSA | semantic | 35.29 | 44.48 |
| Full LSA | rrf | 47.46 | 68.31 |
| Full LSA | weighted_rrf | 34.26 | 43.92 |
| Full LSA | logistic | 32.39 | 44.07 |
| Full LSA | random_forest | 48.84 | 58.99 |
| BGE FP32 DirectML | bm25 | 5.74 | 6.80 |
| BGE FP32 DirectML | semantic | 66.48 | 97.88 |
| BGE FP32 DirectML | rrf | 71.41 | 104.03 |
| BGE FP32 DirectML | weighted_rrf | 71.13 | 92.09 |
| BGE FP32 DirectML | logistic | 69.78 | 90.86 |
| BGE FP32 DirectML | random_forest | 91.21 | 126.21 |

主神经管线中，Logistic与加权RRF的总耗时接近，不能因这一次中位数略低就宣称学习更快；RF包含更多预测计算。排序质量的主要对比区间跨0，运行耗时也不足以支持更复杂模型必然值得部署。

测量设置为BLAS/OpenMP两线程、并发1；BGE的ORT配置保留8线程与GPU device1。结果不代表冷启动、并发服务、其他设备或所有查询分布的SLA。

原始证据：

- [Head LSA 原始样本与协议](20261114T135003226733Z_lsa/latency_benchmark.json)
- [Full LSA 原始样本与协议](20261114T135456567288Z_lsa/latency_benchmark.json)
- [BGE FP32 DirectML 原始样本与协议](20261114T144053296846Z_onnx/latency_benchmark.json)
