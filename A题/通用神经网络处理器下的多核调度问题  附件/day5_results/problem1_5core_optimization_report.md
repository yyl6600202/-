# 问题一：5 核定向优化简报

## 目标与方法

本轮针对 5 核执行配置完成 `case_001`–`case_100` 共 100 个输入。每个 case 都先用单子图方案得到单核基准，再按图结构选择候选方案。`wide_shallow` 图比较 `single / continuous / wave / aligned / components` 五种切分；其他图先沿用 `mixed/high_spill_risk → components`、`deep_narrow → single`，并以单核方案作安全回退。官方问题一评估器以 Makespan 最小为第一目标，额外搬运量和子图数只用于平局。

初始批处理共接受 254 个候选评估，2 个结构上可生成但被官方评估器拒绝的候选被隔离；随后对 13 个加速比不超过 1.2 的非宽浅 case 追加 `single / continuous / wave / components` 扫描，共 52 个候选，最后按官方 Makespan 合并。

## 100-case 结果

| 指标 | 结果 |
|---|---:|
| 核数 | 5 |
| case 数 | 100 |
| 算术平均加速比 | **2.9145×** |
| 几何平均加速比 | 2.5299× |
| 中位数加速比 | 3.0918× |
| 总周期比（单核总周期 / 5 核总周期） | **3.3007×** |
| 比单核更快 | 87 |
| 与单核持平 | 13 |
| 比单核更慢 | 0 |
| 最小 / 最大加速比 | 1.0000× / 5.4115× |

按图类别统计如下：

| 图类别 | case 数 | 平均加速比 | 更快 / 持平 |
|---|---:|---:|---:|
| `wide_shallow` | 21 | 3.2633× | 19 / 2 |
| `high_spill_risk` | 55 | 3.3471× | 54 / 1 |
| `mixed` | 17 | 1.8723× | 14 / 3 |
| `deep_narrow` | 7 | 1.0000× | 0 / 7 |

## 定向改进

补扫后有 5 个 case 更换方案：

| case | 原选择 | 新选择 | 新加速比 |
|---|---|---|---:|
| `case_002` | single | wave | 1.0112× |
| `case_062` | single | wave | 1.0269× |
| `case_063` | single | wave | 1.0242× |
| `case_082` | single | continuous | 1.0077× |
| `case_085` | single | continuous | 1.0026× |

这 5 个 case 将总体算术平均从初始批处理的 2.9138× 提升到 **2.9145×**；所有变更都经过官方评估器复核，没有引入回退。`deep_narrow` 的 7 个 case 仍采用单核方案，因为当前候选切分没有降低官方 Makespan。

## 文件与复核

- 100-case 最终汇总：`problem1_5core_optimized_summary.csv`
- 最终统计：`problem1_5core_optimized_stats.json`
- 100 个初始胜出方案：`problem1_5core_plans/`
- 5 个补扫后改进方案：`problem1_5core_optimized_plans/`
- 初始批处理原始结果：`problem1_5core_summary.csv`
- 非宽浅低分补扫：`low_nondeep_sweep_5core.csv`

官方 CLI 已复核 `case_001`、`case_002`、`case_062`、`case_082`、`case_085`，输出 Makespan 分别为 47,502、259,045、2,780,149、715,556、3,179,494，与最终汇总一致。

本结果是当前五类结构化候选和两轮定向扫描下的可复现实验结果；它不等同于对所有可能的任意子图划分做穷举最优搜索。
