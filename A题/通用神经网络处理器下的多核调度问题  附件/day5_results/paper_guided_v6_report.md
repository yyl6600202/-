# 问题一：论文引导的 5 核调度改进（v6）

本轮固定 5 核，以 v5 的 100 个官方合法方案为基线。候选生成吸收 HEFT/CPOP 的上行秩、下行秩关键路径优先级，以及 CNGA 的通信感知聚簇思想；对每个候选仍调用问题一官方事件模拟器，只有 Makespan 严格下降才替换基线。

- 算术平均加速比：**3.357806×**
- 几何平均加速比：2.989956×
- 中位数加速比：3.716115×
- 总周期比：3.654100×
- 100 个 case，官方方案校验错误：0，结构校验错误：0
- 论文引导候选评估：52 个 case；接受改进：5 个

## 接受的改进

| case | v5 Makespan | v6 Makespan | 加速比 | 规则 |
|---|---:|---:|---:|---|
| case_016 | 7701686 | 7701677 | 1.001798 | `paper_cpop_root_contiguous_weighted` |
| case_024 | 2541506 | 2541497 | 1.005448 | `paper_cpop_root_contiguous_weighted` |
| case_044 | 131105 | 127941 | 1.206861 | `paper_cpop_root_contiguous_weighted` |
| case_073 | 3517515 | 3063665 | 3.290708 | `paper_cpop_root_contiguous_weighted` |
| case_083 | 442480 | 405880 | 2.606665 | `paper_cpop_root_contiguous_weighted` |

其余 case 保留 v5 方案。完整汇总见 `paper_guided_v6_summary.csv`，最终方案见 `paper_guided_v6_plans/`。
